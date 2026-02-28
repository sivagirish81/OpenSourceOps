"""Storage interfaces for playbooks, evidence index, chat traces, and eval logs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Protocol

from src.snowflake_client import SnowflakeClient


class Store(Protocol):
    def save_playbook(self, run_id: str, payload: Dict[str, Any]) -> None: ...
    def save_evidence_chunks(self, repo_id: str, chunks: List[Dict[str, Any]]) -> None: ...
    def load_evidence_chunks(self, repo_id: str) -> List[Dict[str, Any]]: ...
    def save_chat_trace(self, trace: Dict[str, Any]) -> None: ...
    def save_eval_log(self, log: Dict[str, Any]) -> None: ...


@dataclass
class LocalStore:
    playbooks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    evidence: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    traces: List[Dict[str, Any]] = field(default_factory=list)
    evals: List[Dict[str, Any]] = field(default_factory=list)

    def save_playbook(self, run_id: str, payload: Dict[str, Any]) -> None:
        self.playbooks[run_id] = payload

    def save_evidence_chunks(self, repo_id: str, chunks: List[Dict[str, Any]]) -> None:
        self.evidence[repo_id] = chunks

    def load_evidence_chunks(self, repo_id: str) -> List[Dict[str, Any]]:
        return self.evidence.get(repo_id, [])

    def save_chat_trace(self, trace: Dict[str, Any]) -> None:
        self.traces.append(trace)

    def save_eval_log(self, log: Dict[str, Any]) -> None:
        self.evals.append(log)


class SnowflakeStore:
    """Snowflake-backed store with LocalStore fallback when writes fail."""

    def __init__(self, sf: SnowflakeClient, fallback: LocalStore | None = None) -> None:
        self.sf = sf
        self.fallback = fallback or LocalStore()

    def _ts(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def save_playbook(self, run_id: str, payload: Dict[str, Any]) -> None:
        row = {
            "run_id": run_id,
            "created_at": self._ts(),
            "payload": payload,
        }
        self.sf.insert_rows("PLAYBOOK_STORAGE", [row])
        if self.sf.consume_last_insert_error():
            self.fallback.save_playbook(run_id, payload)

    def save_evidence_chunks(self, repo_id: str, chunks: List[Dict[str, Any]]) -> None:
        rows = []
        for c in chunks:
            rows.append(
                {
                    "repo_id": repo_id,
                    "repo_url": c.get("repo_url"),
                    "commit_sha": c.get("commit_sha"),
                    "file_path": c.get("file_path"),
                    "start_line": c.get("start_line"),
                    "end_line": c.get("end_line"),
                    "heading": c.get("heading"),
                    "symbol_name": c.get("symbol_name"),
                    "content_type": c.get("content_type"),
                    "chunk_text": c.get("chunk_text"),
                    "chunk_meta": {
                        "url": c.get("url"),
                        "embedding": c.get("embedding"),
                    },
                    "created_at": self._ts(),
                }
            )
        self.sf.insert_rows("REPO_EVIDENCE_INDEX", rows)
        if self.sf.consume_last_insert_error():
            self.fallback.save_evidence_chunks(repo_id, chunks)

    def load_evidence_chunks(self, repo_id: str) -> List[Dict[str, Any]]:
        if self.sf.is_configured():
            rows = self.sf.execute_safe(
                f"""
                SELECT repo_url, commit_sha, file_path, start_line, end_line, heading, symbol_name,
                       content_type, chunk_text, chunk_meta
                FROM {self.sf.database}.{self.sf.schema}.REPO_EVIDENCE_INDEX
                WHERE repo_id=%s
                ORDER BY created_at ASC
                """,
                [repo_id],
            )
            if rows:
                out = []
                for r in rows:
                    meta = r[9] if isinstance(r[9], dict) else json.loads(r[9] or "{}")
                    out.append(
                        {
                            "repo_url": r[0],
                            "commit_sha": r[1],
                            "file_path": r[2],
                            "start_line": r[3],
                            "end_line": r[4],
                            "heading": r[5],
                            "symbol_name": r[6],
                            "content_type": r[7],
                            "chunk_text": r[8],
                            "url": meta.get("url"),
                            "embedding": meta.get("embedding"),
                        }
                    )
                return out
        return self.fallback.load_evidence_chunks(repo_id)

    def save_chat_trace(self, trace: Dict[str, Any]) -> None:
        row = {
            "trace_id": trace.get("trace_id"),
            "repo_id": trace.get("repo_id"),
            "created_at": self._ts(),
            "mode": trace.get("mode"),
            "question": trace.get("question"),
            "answer": trace.get("answer"),
            "confidence": trace.get("confidence"),
            "citations": trace.get("citations", []),
            "next_checks": trace.get("next_checks", []),
            "question_type": trace.get("question_type"),
        }
        self.sf.insert_rows("CHAT_TRACES", [row])
        if self.sf.consume_last_insert_error():
            self.fallback.save_chat_trace(trace)

    def save_eval_log(self, log: Dict[str, Any]) -> None:
        row = {
            "eval_id": log.get("eval_id"),
            "repo_id": log.get("repo_id"),
            "created_at": self._ts(),
            "metric": log.get("metric"),
            "value": log.get("value"),
            "detail": log.get("detail"),
        }
        self.sf.insert_rows("CHAT_EVAL_LOGS", [row])
        if self.sf.consume_last_insert_error():
            self.fallback.save_eval_log(log)
