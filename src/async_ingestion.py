"""Background ingestion worker for Streamlit-triggered async indexing."""

from __future__ import annotations

import threading
from typing import Any, Dict, List

from storage.snowflake_store import SnowflakeStore
from src.due_diligence_agents import RepoLibrarianAgent


ACTIVE_JOBS: dict[str, threading.Thread] = {}
_LOCK = threading.Lock()


def start_async_ingestion_job(
    company_id: str,
    scout_run_id: str,
    repo_full_name: str,
    repo_url: str,
    company_profile: Dict[str, Any],
    ref: str | None,
    index_profile: str = "standard",
) -> Dict[str, str]:
    store = SnowflakeStore()
    seed = store.create_ingestion_job(
        company_id=company_id,
        repo_full_name=repo_full_name,
        repo_url=repo_url,
    )
    job_id = seed["job_id"]
    ingest_run_id = seed["ingest_run_id"]

    def worker():
        s = SnowflakeStore()
        librarian = RepoLibrarianAgent()
        manifest: set[str] = set()
        total_chunks = 0

        def progress_cb(frac: float, message: str):
            s.update_ingestion_job(job_id, "RUNNING", frac, message)

        def chunk_sink(chunks: List[Dict[str, Any]], rel_path: str):
            nonlocal total_chunks
            total_chunks += len(chunks)
            if rel_path:
                manifest.add(rel_path)
            s.save_evidence_chunks(ingest_run_id=ingest_run_id, chunks=chunks)

        try:
            s.update_ingestion_job(job_id, "RUNNING", 0.01, "Starting background ingestion")
            idx = librarian.run(
                company_profile=company_profile,
                repo_url=repo_url,
                full_name=repo_full_name,
                ref=ref,
                index_profile=index_profile,
                progress_cb=progress_cb,
                chunk_sink=chunk_sink,
            )
            if idx.get("raw_evidence"):
                s.save_repo_evidence_batch(
                    ingest_run_id=ingest_run_id,
                    repo_full_name=repo_full_name,
                    evidence_batch=idx["raw_evidence"],
                )
            if idx.get("signals"):
                s.save_repo_signals(
                    ingest_run_id=ingest_run_id,
                    repo_full_name=repo_full_name,
                    signals=idx["signals"],
                )
            s.save_repo_ingestion(
                scout_run_id=scout_run_id,
                company_id=company_id,
                repo_full_name=repo_full_name,
                repo_url=repo_url,
                commit_sha=idx["commit_sha"],
                files_indexed=len(manifest) or len(idx.get("file_manifest", [])),
                chunks_indexed=total_chunks or len(idx.get("chunks", [])),
                budget_usage=idx.get("budget_usage", {}),
                file_manifest=sorted(manifest) or idx.get("file_manifest", []),
                ingest_run_id=ingest_run_id,
            )
            s.update_ingestion_job(job_id, "COMPLETE", 1.0, f"Completed ({total_chunks} chunks)")
        except Exception as exc:
            s.update_ingestion_job(job_id, "FAILED", 1.0, "Ingestion failed", error=str(exc))

    t = threading.Thread(target=worker, daemon=True, name=f"ingestion-{job_id}")
    with _LOCK:
        ACTIVE_JOBS[job_id] = t
    t.start()
    return {"job_id": job_id, "ingest_run_id": ingest_run_id}
