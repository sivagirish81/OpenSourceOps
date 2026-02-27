"""RUN_LOG helpers to keep consistent, readable audit entries."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

from src.snowflake_client import SnowflakeClient


class RunLogger:
    """Wrapper around SnowflakeClient log/read operations."""

    def __init__(self, snowflake_client: SnowflakeClient) -> None:
        self.sf = snowflake_client

    def log(self, run_id: str, agent: str, step: str, status: str, detail: str) -> None:
        detail = detail.strip() if detail else ""
        if len(detail) > 1800:
            detail = detail[:1800] + "..."
        self.sf.log_run(run_id=run_id, agent=agent, step=step, status=status, detail=detail)

    def trace(self, run_id: str) -> List[Dict]:
        return self.sf.fetch_run_logs(run_id)

    @staticmethod
    def new_run_id(prefix: str = "run") -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"{prefix}_{ts}"
