"""Snowflake-backed storage layer for OSS Due Diligence AI."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

try:
    import snowflake.connector
except Exception:  # pragma: no cover
    snowflake = None  # type: ignore[assignment]
else:  # pragma: no cover
    snowflake = snowflake.connector


load_dotenv()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, default=str)


@dataclass
class SnowflakeConfig:
    account: str
    user: str
    warehouse: str
    database: str
    schema: str
    role: str | None
    password: str | None
    private_key: str | None

    @classmethod
    def from_env(cls) -> "SnowflakeConfig":
        return cls(
            account=os.getenv("SNOWFLAKE_ACCOUNT", ""),
            user=os.getenv("SNOWFLAKE_USER", ""),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", ""),
            database=os.getenv("SNOWFLAKE_DATABASE", "OPENSOURCEOPS"),
            schema=os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
            role=os.getenv("SNOWFLAKE_ROLE"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            private_key=os.getenv("SNOWFLAKE_PRIVATE_KEY"),
        )

    def missing(self) -> List[str]:
        missing = []
        if not self.account:
            missing.append("SNOWFLAKE_ACCOUNT")
        if not self.user:
            missing.append("SNOWFLAKE_USER")
        if not self.warehouse:
            missing.append("SNOWFLAKE_WAREHOUSE")
        if not (self.password or self.private_key):
            missing.append("SNOWFLAKE_PASSWORD or SNOWFLAKE_PRIVATE_KEY")
        return missing


class SnowflakeStore:
    """Typed Snowflake persistence API used by the Streamlit app."""

    def __init__(self, cfg: Optional[SnowflakeConfig] = None) -> None:
        self.cfg = cfg or SnowflakeConfig.from_env()
        self._conn = None

    def is_configured(self) -> bool:
        return snowflake is not None and not self.cfg.missing()

    def connect(self):
        if self._conn:
            return self._conn
        if not self.is_configured():
            raise RuntimeError(f"Snowflake not configured: missing {', '.join(self.cfg.missing())}")
        kwargs: Dict[str, Any] = {
            "account": self.cfg.account,
            "user": self.cfg.user,
            "warehouse": self.cfg.warehouse,
            "database": self.cfg.database,
            "schema": self.cfg.schema,
        }
        if self.cfg.role:
            kwargs["role"] = self.cfg.role
        if self.cfg.private_key:
            kwargs["private_key"] = self.cfg.private_key
        else:
            kwargs["password"] = self.cfg.password
        self._conn = snowflake.connect(**kwargs)  # type: ignore[union-attr]
        return self._conn

    def test_connection(self) -> tuple[bool, str]:
        try:
            rows = self.query("SELECT CURRENT_ACCOUNT(), CURRENT_USER(), CURRENT_ROLE()")
            row = rows[0] if rows else ("unknown", "unknown", "unknown")
            return True, f"Connected account={row[0]} user={row[1]} role={row[2]}"
        except Exception as exc:
            return False, str(exc)

    def ensure_schema(self, schema_sql_path: str = "storage/schema.sql") -> None:
        conn = self.connect()
        script = Path(schema_sql_path).read_text(encoding="utf-8")
        statements = [s.strip() for s in script.split(";") if s.strip()]
        with conn.cursor() as cur:
            for stmt in statements:
                try:
                    cur.execute(stmt)
                except Exception as exc:
                    # Allow app-role execution even without account-level privileges.
                    non_comment_lines = [ln for ln in stmt.splitlines() if not ln.strip().startswith("--")]
                    normalized = " ".join(non_comment_lines).strip().upper()
                    msg = str(exc).lower()
                    privileged_prefixes = (
                        "CREATE DATABASE",
                        "CREATE SCHEMA",
                        "USE DATABASE",
                        "USE SCHEMA",
                    )
                    if "insufficient privileges" in msg and normalized.startswith(privileged_prefixes):
                        continue
                    raise

    def query(self, sql: str, params: Optional[List[Any]] = None) -> List[tuple]:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            try:
                return cur.fetchall()
            except Exception:
                return []

    def execute(self, sql: str, params: Optional[List[Any]] = None) -> None:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(sql, params or ())

    def save_company_profile(self, profile: Dict[str, Any], company_id: str | None = None) -> str:
        company_id = company_id or f"co_{uuid.uuid4().hex[:12]}"
        self.execute(
            f"""
            INSERT INTO {self.cfg.database}.{self.cfg.schema}.COMPANY_PROFILES
            (company_id, profile, updated_at)
            SELECT %s, PARSE_JSON(%s), CURRENT_TIMESTAMP()
            """,
            [company_id, _json(profile)],
        )
        return company_id

    # Required method aliases
    def upsert_company_profile(self, profile: Dict[str, Any], company_id: str | None = None) -> str:
        return self.save_company_profile(profile, company_id=company_id)

    def list_company_profiles(self) -> List[Dict[str, Any]]:
        rows = self.query(
            f"""
            SELECT company_id, created_at, updated_at, profile
            FROM {self.cfg.database}.{self.cfg.schema}.COMPANY_PROFILES
            ORDER BY updated_at DESC
            """
        )
        out = []
        for r in rows:
            payload = r[3] if isinstance(r[3], dict) else json.loads(r[3] or "{}")
            out.append(
                {
                    "company_id": r[0],
                    "created_at": str(r[1]),
                    "updated_at": str(r[2]),
                    "profile": payload,
                    "name": payload.get("organization", {}).get("name", r[0]),
                }
            )
        return out

    def get_company_profile(self, company_id: str) -> Dict[str, Any] | None:
        rows = self.query(
            f"""
            SELECT profile
            FROM {self.cfg.database}.{self.cfg.schema}.COMPANY_PROFILES
            WHERE company_id=%s
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            [company_id],
        )
        if not rows:
            return None
        payload = rows[0][0]
        return payload if isinstance(payload, dict) else json.loads(payload or "{}")

    def save_scouting_run(
        self,
        company_id: str,
        requirements: str,
        constraints: Dict[str, Any],
        query_pack: Dict[str, Any],
        top3: List[Dict[str, Any]],
    ) -> str:
        scout_run_id = f"scout_{uuid.uuid4().hex[:12]}"
        self.execute(
            f"""
            INSERT INTO {self.cfg.database}.{self.cfg.schema}.SCOUTING_RUNS
            (scout_run_id, company_id, requirements, constraints, query_pack, top3)
            SELECT %s, %s, %s, PARSE_JSON(%s), PARSE_JSON(%s), PARSE_JSON(%s)
            """,
            [scout_run_id, company_id, requirements, _json(constraints), _json(query_pack), _json(top3)],
        )
        return scout_run_id

    def create_scouting_run(
        self,
        company_id: str,
        requirements: str,
        constraints: Dict[str, Any],
        query_pack: Dict[str, Any],
        top3: List[Dict[str, Any]],
    ) -> str:
        return self.save_scouting_run(company_id, requirements, constraints, query_pack, top3)

    def save_repo_candidate(self, scout_run_id: str, company_id: str, candidate: Dict[str, Any]) -> None:
        self.execute(
            f"""
            INSERT INTO {self.cfg.database}.{self.cfg.schema}.REPO_CANDIDATES
            (candidate_id, scout_run_id, company_id, repo_full_name, repo_url, score_breakdown, pros, cons, warnings, final_score)
            SELECT %s, %s, %s, %s, %s, PARSE_JSON(%s), PARSE_JSON(%s), PARSE_JSON(%s), PARSE_JSON(%s), %s
            """,
            [
                f"cand_{uuid.uuid4().hex[:12]}",
                scout_run_id,
                company_id,
                candidate.get("full_name"),
                candidate.get("html_url"),
                _json(candidate.get("score_breakdown", {})),
                _json(candidate.get("pros", [])),
                _json(candidate.get("cons", [])),
                _json(candidate.get("constraint_warnings", [])),
                float(candidate.get("final_score", 0.0)),
            ],
        )

    def get_scouting_run(self, scout_run_id: str) -> Dict[str, Any] | None:
        rows = self.query(
            f"""
            SELECT scout_run_id, company_id, created_at, requirements, constraints, query_pack, top3
            FROM {self.cfg.database}.{self.cfg.schema}.SCOUTING_RUNS
            WHERE scout_run_id=%s
            LIMIT 1
            """,
            [scout_run_id],
        )
        if not rows:
            return None
        r = rows[0]
        parse = lambda v: v if isinstance(v, dict) or isinstance(v, list) else json.loads(v or "{}")
        return {
            "scout_run_id": r[0],
            "company_id": r[1],
            "created_at": str(r[2]),
            "requirements": r[3],
            "constraints": parse(r[4]),
            "query_pack": parse(r[5]),
            "top3": parse(r[6]),
        }

    def save_repo_ingestion(
        self,
        scout_run_id: str,
        company_id: str,
        repo_full_name: str,
        repo_url: str,
        commit_sha: str,
        files_indexed: int,
        chunks_indexed: int,
        budget_usage: Dict[str, Any],
        file_manifest: List[str],
        ingest_run_id: str | None = None,
    ) -> str:
        ingest_run_id = ingest_run_id or f"ing_{uuid.uuid4().hex[:12]}"
        self.execute(
            f"""
            INSERT INTO {self.cfg.database}.{self.cfg.schema}.REPO_INGESTIONS
            (ingest_run_id, scout_run_id, company_id, repo_full_name, repo_url, commit_sha, files_indexed,
             chunks_indexed, budget_usage, file_manifest)
            SELECT %s, %s, %s, %s, %s, %s, %s, %s, PARSE_JSON(%s), PARSE_JSON(%s)
            """,
            [
                ingest_run_id,
                scout_run_id,
                company_id,
                repo_full_name,
                repo_url,
                commit_sha,
                files_indexed,
                chunks_indexed,
                _json(budget_usage),
                _json(file_manifest),
            ],
        )
        return ingest_run_id

    def create_ingestion_job(
        self,
        company_id: str,
        repo_full_name: str,
        repo_url: str,
        ingest_run_id: str | None = None,
    ) -> Dict[str, str]:
        ingest_run_id = ingest_run_id or f"ing_{uuid.uuid4().hex[:12]}"
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        self.execute(
            f"""
            INSERT INTO {self.cfg.database}.{self.cfg.schema}.INGESTION_JOBS
            (job_id, ingest_run_id, company_id, repo_full_name, repo_url, status, progress, message)
            SELECT %s, %s, %s, %s, %s, 'QUEUED', 0.0, 'Queued'
            """,
            [job_id, ingest_run_id, company_id, repo_full_name, repo_url],
        )
        return {"job_id": job_id, "ingest_run_id": ingest_run_id}

    def update_ingestion_job(
        self,
        job_id: str,
        status: str,
        progress: float,
        message: str,
        error: str | None = None,
    ) -> None:
        self.execute(
            f"""
            UPDATE {self.cfg.database}.{self.cfg.schema}.INGESTION_JOBS
            SET status=%s, progress=%s, message=%s, updated_at=CURRENT_TIMESTAMP(), error=%s
            WHERE job_id=%s
            """,
            [status, float(progress), message[:1000], error, job_id],
        )

    def get_ingestion_job(self, job_id: str) -> Dict[str, Any] | None:
        rows = self.query(
            f"""
            SELECT job_id, ingest_run_id, company_id, repo_full_name, repo_url, status, progress, message, created_at, updated_at, error
            FROM {self.cfg.database}.{self.cfg.schema}.INGESTION_JOBS
            WHERE job_id=%s
            LIMIT 1
            """,
            [job_id],
        )
        if not rows:
            return None
        r = rows[0]
        return {
            "job_id": r[0],
            "ingest_run_id": r[1],
            "company_id": r[2],
            "repo_full_name": r[3],
            "repo_url": r[4],
            "status": r[5],
            "progress": float(r[6] or 0.0),
            "message": r[7],
            "created_at": str(r[8]),
            "updated_at": str(r[9]),
            "error": r[10],
        }

    def count_evidence_chunks(self, ingest_run_id: str) -> int:
        rows = self.query(
            f"""
            SELECT COUNT(*) FROM {self.cfg.database}.{self.cfg.schema}.EVIDENCE_CHUNKS
            WHERE ingest_run_id=%s
            """,
            [ingest_run_id],
        )
        return int(rows[0][0]) if rows else 0

    def get_evidence_source_breakdown(self, ingest_run_id: str) -> Dict[str, int]:
        rows = self.query(
            f"""
            SELECT source_type, COUNT(*)
            FROM {self.cfg.database}.{self.cfg.schema}.REPO_EVIDENCE
            WHERE ingest_run_id=%s
            GROUP BY source_type
            ORDER BY COUNT(*) DESC
            """,
            [ingest_run_id],
        )
        out: Dict[str, int] = {}
        for r in rows:
            out[str(r[0] or "unknown")] = int(r[1] or 0)
        return out

    def get_latest_ingestion(self, company_id: str, repo_full_name: str) -> Dict[str, Any] | None:
        rows = self.query(
            f"""
            SELECT ingest_run_id, scout_run_id, company_id, repo_full_name, repo_url, commit_sha,
                   files_indexed, chunks_indexed, budget_usage, file_manifest, created_at
            FROM {self.cfg.database}.{self.cfg.schema}.REPO_INGESTIONS
            WHERE company_id=%s AND repo_full_name=%s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [company_id, repo_full_name],
        )
        if not rows:
            return None
        r = rows[0]
        parse = lambda v: v if isinstance(v, (dict, list)) else json.loads(v or "{}")
        return {
            "ingest_run_id": r[0],
            "scout_run_id": r[1],
            "company_id": r[2],
            "repo_full_name": r[3],
            "repo_url": r[4],
            "commit_sha": r[5],
            "files_indexed": int(r[6] or 0),
            "chunks_indexed": int(r[7] or 0),
            "budget_usage": parse(r[8]),
            "file_manifest": parse(r[9]),
            "created_at": str(r[10]),
        }

    def save_evidence_chunks(self, ingest_run_id: str, chunks: List[Dict[str, Any]]) -> None:
        for idx, c in enumerate(chunks):
            chunk_id = f"chunk_{uuid.uuid4().hex[:12]}_{idx}"
            self.execute(
                f"""
                INSERT INTO {self.cfg.database}.{self.cfg.schema}.EVIDENCE_CHUNKS
                (chunk_id, ingest_run_id, repo_url, commit_sha, file_path, start_line, end_line, heading,
                 symbol_name, content_type, chunk_text, chunk_pointer, metadata)
                SELECT %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, PARSE_JSON(%s)
                """,
                [
                    chunk_id,
                    ingest_run_id,
                    c.get("repo_url"),
                    c.get("commit_sha"),
                    c.get("file_path"),
                    int(c.get("start_line") or 1),
                    int(c.get("end_line") or 1),
                    c.get("heading"),
                    c.get("symbol_name"),
                    c.get("content_type"),
                    c.get("chunk_text", "")[:12000],
                    c.get("url"),
                    _json({"ingested_from": "repo_indexer"}),
                ],
            )

    def save_repo_evidence_batch(
        self,
        ingest_run_id: str,
        repo_full_name: str,
        evidence_batch: List[Dict[str, Any]],
    ) -> None:
        for e in evidence_batch:
            self.execute(
                f"""
                INSERT INTO {self.cfg.database}.{self.cfg.schema}.REPO_EVIDENCE
                (evidence_id, ingest_run_id, repo_full_name, repo_url, commit_or_ref, source_type, file_path, source_url,
                 api_ref, retrieved_at, content, line_map, sha, metadata)
                SELECT %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, PARSE_JSON(%s), %s, PARSE_JSON(%s)
                """,
                [
                    e.get("evidence_id") or f"ev_{uuid.uuid4().hex[:12]}",
                    ingest_run_id,
                    repo_full_name,
                    e.get("repo_url"),
                    e.get("commit_or_ref"),
                    e.get("source_type"),
                    e.get("file_path"),
                    e.get("url"),
                    e.get("api_ref"),
                    e.get("retrieved_at"),
                    (e.get("content") or "")[:12000],
                    _json(e.get("line_map", [])),
                    e.get("sha"),
                    _json(e.get("structured", {})),
                ],
            )

    def save_repo_signals(self, ingest_run_id: str, repo_full_name: str, signals: List[Dict[str, Any]]) -> None:
        for s in signals:
            self.execute(
                f"""
                INSERT INTO {self.cfg.database}.{self.cfg.schema}.REPO_SIGNALS
                (signal_id, ingest_run_id, repo_full_name, category, signal_name, signal_value, evidence_id, citations)
                SELECT %s, %s, %s, %s, %s, PARSE_JSON(%s), %s, PARSE_JSON(%s)
                """,
                [
                    f"sig_{uuid.uuid4().hex[:12]}",
                    ingest_run_id,
                    repo_full_name,
                    s.get("category"),
                    s.get("signal_name"),
                    _json(s.get("value")),
                    s.get("evidence_id"),
                    _json(s.get("citations", [])),
                ],
            )

    def load_repo_evidence(self, ingest_run_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        rows = self.query(
            f"""
            SELECT evidence_id, repo_url, commit_or_ref, source_type, file_path, source_url, api_ref, retrieved_at,
                   content, line_map, sha, metadata
            FROM {self.cfg.database}.{self.cfg.schema}.REPO_EVIDENCE
            WHERE ingest_run_id=%s
            ORDER BY created_at ASC
            LIMIT %s
            """,
            [ingest_run_id, limit],
        )
        parse = lambda v: v if isinstance(v, (dict, list)) else json.loads(v or "{}")
        return [
            {
                "evidence_id": r[0],
                "repo_url": r[1],
                "commit_or_ref": r[2],
                "source_type": r[3],
                "file_path": r[4],
                "url": r[5],
                "api_ref": r[6],
                "retrieved_at": str(r[7]),
                "content": r[8],
                "line_map": parse(r[9]),
                "sha": r[10],
                "structured": parse(r[11]),
            }
            for r in rows
        ]

    def load_repo_signals(self, ingest_run_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        rows = self.query(
            f"""
            SELECT category, signal_name, signal_value, evidence_id, citations
            FROM {self.cfg.database}.{self.cfg.schema}.REPO_SIGNALS
            WHERE ingest_run_id=%s
            ORDER BY created_at ASC
            LIMIT %s
            """,
            [ingest_run_id, limit],
        )
        parse = lambda v: v if isinstance(v, (dict, list)) else json.loads(v or "{}")
        return [
            {
                "category": r[0],
                "signal_name": r[1],
                "value": parse(r[2]),
                "evidence_id": r[3],
                "citations": parse(r[4]),
            }
            for r in rows
        ]

    def load_evidence_chunks(self, ingest_run_id: str, limit: int = 600) -> List[Dict[str, Any]]:
        rows = self.query(
            f"""
            SELECT repo_url, commit_sha, file_path, start_line, end_line, heading, symbol_name,
                   content_type, chunk_text, chunk_pointer
            FROM {self.cfg.database}.{self.cfg.schema}.EVIDENCE_CHUNKS
            WHERE ingest_run_id=%s
            ORDER BY created_at ASC
            LIMIT %s
            """,
            [ingest_run_id, limit],
        )
        return [
            {
                "repo_url": r[0],
                "commit_sha": r[1],
                "file_path": r[2],
                "start_line": int(r[3] or 1),
                "end_line": int(r[4] or 1),
                "heading": r[5],
                "symbol_name": r[6],
                "content_type": r[7],
                "chunk_text": r[8],
                "url": r[9],
            }
            for r in rows
        ]

    def start_due_diligence_run(
        self,
        company_id: str,
        scout_run_id: str,
        ingest_run_id: str,
        repo_full_name: str,
        repo_url: str,
    ) -> str:
        dd_run_id = f"dd_{uuid.uuid4().hex[:12]}"
        self.execute(
            f"""
            INSERT INTO {self.cfg.database}.{self.cfg.schema}.DUE_DILIGENCE_RUNS
            (dd_run_id, company_id, scout_run_id, ingest_run_id, repo_full_name, repo_url, status)
            SELECT %s, %s, %s, %s, %s, %s, 'RUNNING'
            """,
            [dd_run_id, company_id, scout_run_id, ingest_run_id, repo_full_name, repo_url],
        )
        return dd_run_id

    def create_due_diligence_run(
        self,
        company_id: str,
        scout_run_id: str,
        ingest_run_id: str,
        repo_full_name: str,
        repo_url: str,
    ) -> str:
        return self.start_due_diligence_run(company_id, scout_run_id, ingest_run_id, repo_full_name, repo_url)

    def save_finding(self, dd_run_id: str, finding: Dict[str, Any]) -> None:
        self.execute(
            f"""
            INSERT INTO {self.cfg.database}.{self.cfg.schema}.FINDINGS
            (finding_id, dd_run_id, agent_name, category, claim, confidence, citations, base_severity,
             context_multiplier, adjusted_severity, multiplier_rationale, next_checks, owner, effort)
            SELECT %s, %s, %s, %s, %s, %s, PARSE_JSON(%s), %s, %s, %s, %s, PARSE_JSON(%s), %s, %s
            """,
            [
                finding.get("finding_id") or f"f_{uuid.uuid4().hex[:12]}",
                dd_run_id,
                finding.get("agent_name"),
                finding.get("category"),
                finding.get("claim"),
                finding.get("confidence"),
                _json(finding.get("citations", [])),
                float(finding.get("base_severity", 0.0)),
                float(finding.get("context_multiplier", 1.0)),
                float(finding.get("adjusted_severity", 0.0)),
                finding.get("multiplier_rationale", ""),
                _json(finding.get("next_checks", [])),
                finding.get("owner", "Eng"),
                finding.get("effort", "M"),
            ],
        )

    def save_findings_batch(self, dd_run_id: str, findings: List[Dict[str, Any]]) -> None:
        for f in findings:
            self.save_finding(dd_run_id, f)

    def save_transcript(self, dd_run_id: str, agent_name: str, step: str, status: str, payload: Dict[str, Any]) -> None:
        self.execute(
            f"""
            INSERT INTO {self.cfg.database}.{self.cfg.schema}.AGENT_TRANSCRIPTS
            (transcript_id, dd_run_id, agent_name, step, status, payload)
            SELECT %s, %s, %s, %s, %s, PARSE_JSON(%s)
            """,
            [f"t_{uuid.uuid4().hex[:12]}", dd_run_id, agent_name, step, status, _json(payload)],
        )

    def complete_due_diligence_run(self, dd_run_id: str, decision: str, report_json: Dict[str, Any], report_md: str) -> None:
        self.execute(
            f"""
            UPDATE {self.cfg.database}.{self.cfg.schema}.DUE_DILIGENCE_RUNS
            SET status='COMPLETE',
                decision=%s,
                report_json=PARSE_JSON(%s),
                report_md=%s
            WHERE dd_run_id=%s
            """,
            [decision, _json(report_json), report_md, dd_run_id],
        )
        self.execute(
            f"""
            INSERT INTO {self.cfg.database}.{self.cfg.schema}.REPORTS
            (report_id, dd_run_id, decision, report_json, report_md)
            SELECT %s, %s, %s, PARSE_JSON(%s), %s
            """,
            [f"rep_{uuid.uuid4().hex[:12]}", dd_run_id, decision, _json(report_json), report_md],
        )

    def save_final_report(self, dd_run_id: str, decision: str, report_json: Dict[str, Any], report_md: str) -> None:
        self.complete_due_diligence_run(dd_run_id, decision, report_json, report_md)

    def load_due_diligence(self, dd_run_id: str) -> Dict[str, Any]:
        run_rows = self.query(
            f"""
            SELECT dd_run_id, company_id, scout_run_id, ingest_run_id, repo_full_name, repo_url, created_at,
                   status, decision, report_json, report_md
            FROM {self.cfg.database}.{self.cfg.schema}.DUE_DILIGENCE_RUNS
            WHERE dd_run_id=%s
            LIMIT 1
            """,
            [dd_run_id],
        )
        finding_rows = self.query(
            f"""
            SELECT finding_id, agent_name, category, claim, confidence, citations, base_severity,
                   context_multiplier, adjusted_severity, multiplier_rationale, next_checks, owner, effort
            FROM {self.cfg.database}.{self.cfg.schema}.FINDINGS
            WHERE dd_run_id=%s
            ORDER BY adjusted_severity DESC
            """,
            [dd_run_id],
        )
        transcript_rows = self.query(
            f"""
            SELECT created_at, agent_name, step, status, payload
            FROM {self.cfg.database}.{self.cfg.schema}.AGENT_TRANSCRIPTS
            WHERE dd_run_id=%s
            ORDER BY created_at ASC
            """,
            [dd_run_id],
        )
        if not run_rows:
            return {}
        rr = run_rows[0]
        parse = lambda v: v if isinstance(v, (dict, list)) else json.loads(v or "{}")
        return {
            "dd_run_id": rr[0],
            "company_id": rr[1],
            "scout_run_id": rr[2],
            "ingest_run_id": rr[3],
            "repo_full_name": rr[4],
            "repo_url": rr[5],
            "created_at": str(rr[6]),
            "status": rr[7],
            "decision": rr[8],
            "report_json": parse(rr[9]),
            "report_md": rr[10],
            "findings": [
                {
                    "finding_id": r[0],
                    "agent_name": r[1],
                    "category": r[2],
                    "claim": r[3],
                    "confidence": r[4],
                    "citations": parse(r[5]),
                    "base_severity": float(r[6] or 0.0),
                    "context_multiplier": float(r[7] or 1.0),
                    "adjusted_severity": float(r[8] or 0.0),
                    "multiplier_rationale": r[9],
                    "next_checks": parse(r[10]),
                    "owner": r[11],
                    "effort": r[12],
                }
                for r in finding_rows
            ],
            "transcripts": [
                {
                    "created_at": str(r[0]),
                    "agent_name": r[1],
                    "step": r[2],
                    "status": r[3],
                    "payload": parse(r[4]),
                }
                for r in transcript_rows
            ],
        }
