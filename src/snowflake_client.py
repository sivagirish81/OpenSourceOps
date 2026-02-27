"""Snowflake client with local dry-run fallback for hackathon resiliency."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv

try:
    import snowflake.connector as sf
except Exception:  # pragma: no cover - optional dependency at runtime
    sf = None


load_dotenv()


class SnowflakeClient:
    """Data-access helper; falls back to in-memory stores when not configured."""

    def __init__(self) -> None:
        self.account = os.getenv("SNOWFLAKE_ACCOUNT")
        self.user = os.getenv("SNOWFLAKE_USER")
        self.password = os.getenv("SNOWFLAKE_PASSWORD")
        self.private_key = os.getenv("SNOWFLAKE_PRIVATE_KEY")
        self.warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
        self.database = os.getenv("SNOWFLAKE_DATABASE", "OPENSOURCEOPS")
        self.schema = os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")
        self._conn = None
        self._last_insert_error: str | None = None
        self._mem: Dict[str, List[Dict]] = {
            "REPOS": [],
            "REPO_SCORES": [],
            "REPO_AI": [],
            "DOMAIN_AI": [],
            "ISSUES": [],
            "ISSUE_SCORES": [],
            "RUN_LOG": [],
        }

    def is_configured(self) -> bool:
        return bool(sf and self.account and self.user and (self.password or self.private_key))

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

    def run_setup_sql(self, sql_path: str = "sql/setup.sql") -> None:
        conn = self.connect()
        if not conn:
            raise RuntimeError("Snowflake is not configured.")
        with open(sql_path, "r", encoding="utf-8") as f:
            script = f.read()
        statements = [s.strip() for s in script.split(";") if s.strip()]
        with conn.cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)

    def _insert_mem(self, table: str, row: Dict) -> None:
        self._mem[table].append(row)

    def _insert_sql(self, table: str, row: Dict) -> None:
        cols = list(row.keys())
        values: List[Any] = []
        select_exprs: List[str] = []
        typed_json_cols = {
            "REPOS": {"topics": "ARRAY"},
            "ISSUES": {"labels": "ARRAY"},
            "REPO_AI": {"risks": "ARRAY", "raw_json": "VARIANT"},
            "DOMAIN_AI": {"playbook_json": "VARIANT"},
        }
        table_col_types = typed_json_cols.get(table, {})

        for col in cols:
            val = row[col]
            target_type = table_col_types.get(col)
            if target_type == "ARRAY":
                values.append(json.dumps(val if isinstance(val, list) else []))
                select_exprs.append("TO_ARRAY(PARSE_JSON(%s))")
            elif target_type == "VARIANT":
                values.append(json.dumps(val if isinstance(val, (dict, list)) else val))
                select_exprs.append("PARSE_JSON(%s)")
            elif isinstance(val, (dict, list)):
                values.append(json.dumps(val))
                select_exprs.append("PARSE_JSON(%s)")
            else:
                values.append(val)
                select_exprs.append("%s")

        sql = (
            f"INSERT INTO {self.database}.{self.schema}.{table} ({', '.join(cols)}) "
            f"SELECT {', '.join(select_exprs)}"
        )
        self.execute(sql, values)

    def insert_rows(self, table: str, rows: List[Dict]) -> None:
        self._last_insert_error = None
        for row in rows:
            try:
                if self.is_configured():
                    self._insert_sql(table, row)
                else:
                    self._insert_mem(table, row)
            except Exception as exc:
                self._last_insert_error = f"{table}: {exc}"
                # Fall back to in-memory logging path to avoid dropping pipeline output.
                self._insert_mem(table, row)

    def consume_last_insert_error(self) -> str | None:
        err = self._last_insert_error
        self._last_insert_error = None
        return err

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

    def fetch_run_logs(self, run_id: str) -> List[Dict]:
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

    def fetch_cached_domain_run(
        self, domain: str, language: str | None, max_age_hours: int = 6
    ) -> Optional[Dict[str, Any]]:
        if self.is_configured():
            rows = self.execute(
                f"""
                SELECT run_id, playbook_json, playbook_md, generated_at
                FROM {self.database}.{self.schema}.DOMAIN_AI
                WHERE domain=%s AND COALESCE(language,'')=COALESCE(%s,'')
                AND generated_at >= DATEADD(hour, -%s, CURRENT_TIMESTAMP())
                ORDER BY generated_at DESC
                LIMIT 1
                """,
                [domain, language, max_age_hours],
            )
            if not rows:
                return None
            payload = rows[0][1]
            playbook_json = payload if isinstance(payload, dict) else json.loads(payload or "{}")
            return {
                "run_id": rows[0][0],
                "playbook_json": playbook_json,
                "playbook_md": rows[0][2],
                "generated_at": str(rows[0][3]),
            }

        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        items = [
            r
            for r in self._mem["DOMAIN_AI"]
            if r.get("domain") == domain and (r.get("language") or "") == (language or "")
        ]
        items.sort(key=lambda x: x.get("generated_at", ""), reverse=True)
        if not items:
            return None
        newest = items[0]
        ts = datetime.fromisoformat(newest["generated_at"])
        if ts < cutoff:
            return None
        return newest


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
