"""Snowflake client with resilient typed inserts, cache helpers, and dry-run fallback."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv

try:
    import snowflake.connector as sf
except Exception:  # pragma: no cover
    sf = None


load_dotenv()


class SnowflakeClient:
    """Data-access helper; uses Snowflake when configured and memory fallback otherwise."""

    def __init__(self) -> None:
        self.account = os.getenv("SNOWFLAKE_ACCOUNT")
        self.user = os.getenv("SNOWFLAKE_USER")
        self.password = os.getenv("SNOWFLAKE_PASSWORD")
        self.private_key = os.getenv("SNOWFLAKE_PRIVATE_KEY")
        self.warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
        self.database = os.getenv("SNOWFLAKE_DATABASE", "OPENSOURCEOPS")
        self.schema = os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")
        self.role = os.getenv("SNOWFLAKE_ROLE")
        self._conn = None
        self._last_insert_error: str | None = None
        self._table_columns_cache: Dict[str, set[str]] = {}
        self._mem: Dict[str, List[Dict[str, Any]]] = {
            "RUN_CONTEXT": [],
            "RUN_FEEDBACK": [],
            "REPOS": [],
            "REPO_SCORES": [],
            "REPO_AI": [],
            "DOMAIN_AI": [],
            "ISSUES": [],
            "CONTRIBUTION_OPPORTUNITIES": [],
            "RUN_ACTIONS": [],
            "RUN_LOG": [],
            "PLAYBOOK_STORAGE": [],
            "REPO_EVIDENCE_INDEX": [],
            "CHAT_TRACES": [],
            "CHAT_EVAL_LOGS": [],
        }

    def is_configured(self) -> bool:
        return bool(sf and self.account and self.user and self.warehouse and (self.password or self.private_key))

    def connect(self):
        if self._conn:
            return self._conn
        if not self.is_configured():
            return None
        kwargs = {
            "account": self.account,
            "user": self.user,
            "warehouse": self.warehouse,
            "database": self.database,
            "schema": self.schema,
        }
        if self.role:
            kwargs["role"] = self.role
        if self.private_key:
            kwargs["private_key"] = self.private_key
        else:
            kwargs["password"] = self.password
        self._conn = sf.connect(**kwargs)
        return self._conn

    def execute(self, sql: str, params: Optional[Iterable] = None) -> List[tuple]:
        conn = self.connect()
        if not conn:
            return []
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            try:
                return cur.fetchall()
            except Exception:
                return []

    def execute_safe(self, sql: str, params: Optional[Iterable] = None) -> List[tuple]:
        """Best-effort query execution for compatibility/fallback reads."""
        try:
            return self.execute(sql, params)
        except Exception:
            return []

    def run_setup_sql(self, sql_path: str = "sql/setup.sql") -> None:
        conn = self.connect()
        if not conn:
            raise RuntimeError("Snowflake is not configured.")
        with open(sql_path, "r", encoding="utf-8") as f:
            script = f.read()
        statements = [s.strip() for s in script.split(";") if s.strip()]
        with conn.cursor() as cur:
            for stmt in statements:
                try:
                    cur.execute(stmt)
                except Exception as exc:
                    # Normalize SQL by removing comment lines so prefix checks are reliable.
                    non_comment_lines = [
                        ln for ln in stmt.splitlines() if not ln.strip().startswith("--")
                    ]
                    normalized_stmt = " ".join(non_comment_lines).strip()
                    upper_stmt = normalized_stmt.upper()
                    msg = str(exc).lower()
                    privilege_limited_stmt = any(
                        upper_stmt.startswith(prefix) or f" {prefix}" in f" {upper_stmt}"
                        for prefix in [
                            "CREATE DATABASE",
                            "CREATE SCHEMA",
                            "USE DATABASE",
                            "USE SCHEMA",
                        ]
                    )
                    if privilege_limited_stmt and "insufficient privileges" in msg:
                        head = upper_stmt.splitlines()[0] if upper_stmt else "UNKNOWN"
                        print(f"[init_snowflake] Skipping privileged statement: {head}")
                        continue
                    if "insufficient privileges" in msg:
                        head = upper_stmt.splitlines()[0] if upper_stmt else "UNKNOWN"
                        print(f"[init_snowflake] Skipping statement due to role permissions: {head}")
                        continue
                    raise

    def _get_table_columns(self, table: str) -> set[str]:
        key = table.upper()
        if key in self._table_columns_cache:
            return self._table_columns_cache[key]
        rows = self.execute_safe(
            f"""
            SELECT COLUMN_NAME
            FROM {self.database}.INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s
            """,
            [self.schema.upper(), key],
        )
        cols = {str(r[0]).lower() for r in rows} if rows else set()
        self._table_columns_cache[key] = cols
        return cols

    def _insert_mem(self, table: str, row: Dict[str, Any]) -> None:
        self._mem.setdefault(table, []).append(row)

    def _insert_sql(self, table: str, row: Dict[str, Any]) -> None:
        table_cols = self._get_table_columns(table)
        filtered = {k: v for k, v in row.items() if not table_cols or k.lower() in table_cols}
        if not filtered:
            return

        cols = list(filtered.keys())
        values: List[Any] = []
        select_exprs: List[str] = []

        typed_json_cols = {
            "RUN_CONTEXT": {"constraints": "VARIANT", "optimizer": "VARIANT"},
            "REPOS": {"topics": "ARRAY"},
            "ISSUES": {"labels": "ARRAY"},
            "REPO_AI": {"risks": "ARRAY", "raw_json": "VARIANT"},
            "DOMAIN_AI": {"playbook": "VARIANT"},
            "CONTRIBUTION_OPPORTUNITIES": {"labels": "ARRAY"},
            "PLAYBOOK_STORAGE": {"payload": "VARIANT"},
            "REPO_EVIDENCE_INDEX": {"chunk_meta": "VARIANT"},
            "CHAT_TRACES": {"citations": "ARRAY", "next_checks": "ARRAY"},
        }
        table_col_types = typed_json_cols.get(table, {})

        for col in cols:
            val = filtered[col]
            target_type = table_col_types.get(col)
            if target_type == "ARRAY":
                values.append(json.dumps(val if isinstance(val, list) else []))
                select_exprs.append("TO_ARRAY(PARSE_JSON(%s))")
            elif target_type == "VARIANT":
                values.append(json.dumps(val if isinstance(val, (dict, list)) else val))
                select_exprs.append("PARSE_JSON(%s)")
            else:
                values.append(val)
                select_exprs.append("%s")

        sql = (
            f"INSERT INTO {self.database}.{self.schema}.{table} ({', '.join(cols)}) "
            f"SELECT {', '.join(select_exprs)}"
        )
        self.execute(sql, values)

    def insert_rows(self, table: str, rows: List[Dict[str, Any]]) -> None:
        self._last_insert_error = None
        for row in rows:
            try:
                if self.is_configured():
                    self._insert_sql(table, row)
                else:
                    self._insert_mem(table, row)
            except Exception as exc:
                self._last_insert_error = f"{table}: {exc}"
                self._insert_mem(table, row)

    def consume_last_insert_error(self) -> str | None:
        err = self._last_insert_error
        self._last_insert_error = None
        return err

    def save_run_context(
        self,
        run_id: str,
        persona: str,
        domain: str,
        intent: str,
        constraints: Dict[str, Any],
        cache_key: str,
        optimizer: Dict[str, Any],
        previous_run_id: str | None = None,
    ) -> None:
        self.insert_rows(
            "RUN_CONTEXT",
            [
                {
                    "run_id": run_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "persona": persona,
                    "domain": domain,
                    "intent": intent,
                    "constraints": constraints,
                    "cache_key": cache_key,
                    "optimizer": optimizer,
                    "previous_run_id": previous_run_id,
                }
            ],
        )

    def save_feedback(self, run_id: str, section: str, feedback_text: str) -> None:
        self.insert_rows(
            "RUN_FEEDBACK",
            [
                {
                    "run_id": run_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "section": section,
                    "feedback_text": feedback_text,
                }
            ],
        )

    def save_action(
        self,
        run_id: str,
        action_type: str,
        status: str,
        target_url: str | None,
        detail: str,
    ) -> None:
        self.insert_rows(
            "RUN_ACTIONS",
            [
                {
                    "run_id": run_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "action_type": action_type,
                    "status": status,
                    "target_url": target_url,
                    "detail": detail,
                }
            ],
        )

    def log_run(self, run_id: str, agent: str, step: str, status: str, detail: str) -> None:
        row = {
            "run_id": run_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "agent": agent,
            "step": step,
            "status": status,
            "detail": detail[:2000],
        }
        if self.is_configured():
            self._insert_sql("RUN_LOG", row)
        else:
            self._insert_mem("RUN_LOG", row)

    def fetch_run_logs(self, run_id: str) -> List[Dict[str, Any]]:
        if self.is_configured():
            rows = self.execute(
                f"SELECT run_id, ts, agent, step, status, detail FROM {self.database}.{self.schema}.RUN_LOG WHERE run_id=%s ORDER BY ts ASC",
                [run_id],
            )
            return [
                {
                    "run_id": r[0],
                    "ts": str(r[1]),
                    "agent": r[2],
                    "step": r[3],
                    "status": r[4],
                    "detail": r[5],
                }
                for r in rows
            ]
        return [r for r in self._mem["RUN_LOG"] if r.get("run_id") == run_id]

    def fetch_recent_run_id_by_cache(
        self, cache_key: str, persona: str, max_age_hours: int = 6
    ) -> str | None:
        if self.is_configured():
            rows = self.execute_safe(
                f"""
                SELECT run_id
                FROM {self.database}.{self.schema}.RUN_CONTEXT
                WHERE cache_key=%s AND persona=%s
                  AND created_at >= DATEADD(hour, -%s, CURRENT_TIMESTAMP())
                ORDER BY created_at DESC
                LIMIT 1
                """,
                [cache_key, persona, max_age_hours],
            )
            return rows[0][0] if rows else None

        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        items = [
            r
            for r in self._mem["RUN_CONTEXT"]
            if r.get("cache_key") == cache_key and r.get("persona") == persona
        ]
        items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        if not items:
            return None
        created = datetime.fromisoformat(items[0]["created_at"])
        if created < cutoff:
            return None
        return items[0]["run_id"]

    def load_cached_bundle(self, run_id: str) -> Dict[str, Any]:
        """Load cached artifacts for a run_id from Snowflake or fallback memory."""
        if self.is_configured():
            ctx = self.execute_safe(
                f"SELECT optimizer FROM {self.database}.{self.schema}.RUN_CONTEXT WHERE run_id=%s ORDER BY created_at DESC LIMIT 1",
                [run_id],
            )
            repos = self.execute_safe(
                f"""
                SELECT full_name, html_url, description, topics, repo_language, stargazers_count,
                       forks_count, open_issues_count, watchers_count, license, pushed_at, created_at,
                       contributors_count
                FROM {self.database}.{self.schema}.REPOS
                WHERE run_id=%s
                """,
                [run_id],
            )
            scored = self.execute_safe(
                f"""
                SELECT s.full_name, s.intent_match_score, s.activity_score, s.adoption_score,
                       s.maintenance_score, s.community_score, s.health_score, s.risk_score, s.final_score,
                       r.html_url, r.description, r.topics, r.repo_language, r.stargazers_count, r.forks_count,
                       r.open_issues_count, r.watchers_count, r.license, r.pushed_at, r.created_at
                FROM {self.database}.{self.schema}.REPO_SCORES s
                LEFT JOIN {self.database}.{self.schema}.REPOS r
                    ON s.run_id=r.run_id AND s.full_name=r.full_name
                WHERE s.run_id=%s
                ORDER BY s.final_score DESC
                """,
                [run_id],
            )
            if not scored:
                # Backward compatibility with legacy REPO_SCORES schema.
                scored = self.execute_safe(
                    f"""
                    SELECT s.full_name, NULL AS intent_match_score, s.activity_score, s.adoption_score,
                           s.maintenance_score, s.community_score, s.health_score, s.risk_score, s.health_score AS final_score,
                           r.html_url, r.description, r.topics, r.repo_language, r.stargazers_count, r.forks_count,
                           r.open_issues_count, r.watchers_count, r.license, r.pushed_at, r.created_at
                    FROM {self.database}.{self.schema}.REPO_SCORES s
                    LEFT JOIN {self.database}.{self.schema}.REPOS r
                        ON s.run_id=r.run_id AND s.full_name=r.full_name
                    WHERE s.run_id=%s
                    ORDER BY s.health_score DESC
                    """,
                    [run_id],
                )
            repo_ai = self.execute_safe(
                f"SELECT full_name, raw_json FROM {self.database}.{self.schema}.REPO_AI WHERE run_id=%s",
                [run_id],
            )
            domain_ai = self.execute_safe(
                f"SELECT playbook, playbook_md FROM {self.database}.{self.schema}.DOMAIN_AI WHERE run_id=%s ORDER BY generated_at DESC LIMIT 1",
                [run_id],
            )
            if not domain_ai:
                domain_ai = self.execute_safe(
                    f"SELECT playbook_json, playbook_md FROM {self.database}.{self.schema}.DOMAIN_AI WHERE run_id=%s ORDER BY generated_at DESC LIMIT 1",
                    [run_id],
                )
            opps = self.execute_safe(
                f"""
                SELECT repo_full_name, item_type, item_id, title, body, labels, comments, updated_at, url,
                       engagement_score, scope_score, relevance_score, ease_score, final_score, why,
                       suggested_next_action, difficulty
                FROM {self.database}.{self.schema}.CONTRIBUTION_OPPORTUNITIES
                WHERE run_id=%s
                ORDER BY final_score DESC
                """,
                [run_id],
            )
            return {
                "optimizer": (ctx[0][0] if ctx and isinstance(ctx[0][0], dict) else json.loads(ctx[0][0] or "{}") if ctx else {}),
                "repos": [
                    {
                        "full_name": r[0],
                        "html_url": r[1],
                        "description": r[2],
                        "topics": json.loads(r[3]) if isinstance(r[3], str) else (r[3] or []),
                        "language": r[4],
                        "stargazers_count": r[5],
                        "forks_count": r[6],
                        "open_issues_count": r[7],
                        "watchers_count": r[8],
                        "license": r[9],
                        "pushed_at": str(r[10]) if r[10] else None,
                        "created_at": str(r[11]) if r[11] else None,
                        "contributors_count": r[12],
                    }
                    for r in repos
                ],
                "scored_repos": [
                    {
                        "full_name": r[0],
                        "intent_match_score": r[1],
                        "activity_score": r[2],
                        "adoption_score": r[3],
                        "maintenance_score": r[4],
                        "community_score": r[5],
                        "health_score": r[6],
                        "risk_score": r[7],
                        "final_score": r[8],
                        "html_url": r[9],
                        "description": r[10],
                        "topics": json.loads(r[11]) if isinstance(r[11], str) else (r[11] or []),
                        "language": r[12],
                        "stargazers_count": r[13],
                        "forks_count": r[14],
                        "open_issues_count": r[15],
                        "watchers_count": r[16],
                        "license": r[17],
                        "pushed_at": str(r[18]) if r[18] else None,
                        "created_at": str(r[19]) if r[19] else None,
                    }
                    for r in scored
                ],
                "repo_ai": {
                    r[0]: (r[1] if isinstance(r[1], dict) else json.loads(r[1] or "{}"))
                    for r in repo_ai
                },
                "playbook_json": (domain_ai[0][0] if domain_ai and isinstance(domain_ai[0][0], dict) else json.loads(domain_ai[0][0] or "{}") if domain_ai else {}),
                "playbook_md": domain_ai[0][1] if domain_ai else "",
                "opportunities": [
                    {
                        "repo_full_name": o[0],
                        "item_type": o[1],
                        "item_id": o[2],
                        "title": o[3],
                        "body": o[4],
                        "labels": json.loads(o[5]) if isinstance(o[5], str) else (o[5] or []),
                        "comments": o[6],
                        "updated_at": str(o[7]) if o[7] else None,
                        "url": o[8],
                        "engagement_score": o[9],
                        "scope_score": o[10],
                        "relevance_score": o[11],
                        "ease_score": o[12],
                        "final_score": o[13],
                        "why": o[14],
                        "suggested_next_action": o[15],
                        "difficulty": o[16],
                    }
                    for o in opps
                ],
            }

        repos = [r for r in self._mem["REPOS"] if r.get("run_id") == run_id]
        scored = [r for r in self._mem["REPO_SCORES"] if r.get("run_id") == run_id]
        repo_ai_rows = [r for r in self._mem["REPO_AI"] if r.get("run_id") == run_id]
        domain_rows = [r for r in self._mem["DOMAIN_AI"] if r.get("run_id") == run_id]
        opps = [r for r in self._mem["CONTRIBUTION_OPPORTUNITIES"] if r.get("run_id") == run_id]
        ctx_rows = [r for r in self._mem["RUN_CONTEXT"] if r.get("run_id") == run_id]
        domain_rows.sort(key=lambda x: x.get("generated_at", ""), reverse=True)
        ctx_rows.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        repo_map = {r.get("full_name"): r for r in repos}
        merged_scores = []
        for score in scored:
            merged_scores.append({**repo_map.get(score.get("full_name"), {}), **score})
        merged_scores.sort(key=lambda x: x.get("final_score", 0), reverse=True)

        return {
            "optimizer": ctx_rows[0].get("optimizer", {}) if ctx_rows else {},
            "repos": repos,
            "scored_repos": merged_scores,
            "repo_ai": {r.get("full_name"): r.get("raw_json", {}) for r in repo_ai_rows},
            "playbook_json": domain_rows[0].get("playbook", {}) if domain_rows else {},
            "playbook_md": domain_rows[0].get("playbook_md", "") if domain_rows else "",
            "opportunities": sorted(opps, key=lambda x: x.get("final_score", 0), reverse=True),
        }


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="Run sql/setup.sql")
    args = parser.parse_args()

    client = SnowflakeClient()
    if args.init:
        client.run_setup_sql("sql/setup.sql")
        print("Snowflake setup complete.")


if __name__ == "__main__":
    _main()
