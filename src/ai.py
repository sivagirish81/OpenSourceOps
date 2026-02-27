"""Snowflake AI_COMPLETE wrappers with strict JSON parsing and fallback behavior."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from src.run_logger import RunLogger
from src.snowflake_client import SnowflakeClient


REPO_PROMPT = """
You are an enterprise OSS analyst. Return strict JSON only with keys:
{ "maturity": string, "summary": string, "risks": [string], "recommended_use": string, "engagement_plan": string }
Analyze this repository payload:
{payload}
""".strip()

PLAYBOOK_PROMPT = """
You are an OSS strategy consultant. Return strict JSON only with keys:
{ "executive_summary": string, "top_repos": [string], "maturity_and_risks": [string], "plan_30": [string], "plan_60": [string], "plan_90": [string], "technical_notes": [string], "engagement_strategy": [string], "artifacts": [string] }
Generate a domain adoption playbook from this input:
{payload}
""".strip()


class AIClient:
    """Cortex AI_COMPLETE integration with robust fallback for hackathon demos."""

    def __init__(self, snowflake_client: SnowflakeClient, logger: RunLogger) -> None:
        self.sf = snowflake_client
        self.logger = logger

    def _ai_complete(self, model: str, prompt: str) -> str:
        if not self.sf.is_configured():
            raise RuntimeError("Snowflake unavailable; AI_COMPLETE fallback mode")

        # AI_COMPLETE signatures can vary by account version; try two patterns.
        queries = [
            "SELECT AI_COMPLETE(%s, %s)",
            "SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s)",
        ]
        for query in queries:
            try:
                rows = self.sf.execute(query, [model, prompt])
                if rows and rows[0]:
                    return str(rows[0][0])
            except Exception:
                continue
        raise RuntimeError("Unable to call Snowflake AI_COMPLETE")

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.replace("json", "", 1).strip()
        return json.loads(raw)

    def repo_summary(self, run_id: str, repo_payload: Dict[str, Any], model: str = "snowflake-arctic") -> Dict[str, Any]:
        prompt = REPO_PROMPT.replace("{payload}", json.dumps(repo_payload))
        try:
            raw = self._ai_complete(model=model, prompt=prompt)
            parsed = self._parse_json(raw)
            required = {"maturity", "summary", "risks", "recommended_use", "engagement_plan"}
            if not required.issubset(set(parsed.keys())):
                raise ValueError("Missing required keys")
            return parsed
        except Exception as exc:
            self.logger.log(run_id, "StrategistAgent", "repo_ai_fallback", "INFO", f"AI JSON parse failed: {exc}")
            return {
                "maturity": "emerging",
                "summary": f"{repo_payload.get('full_name')} is active with moderate OSS traction.",
                "risks": ["AI fallback used; validate manually", "Potential maintainer concentration"],
                "recommended_use": "Pilot in non-critical workloads first.",
                "engagement_plan": "Engage maintainers via issues and discussions; contribute docs/tests.",
            }

    def domain_playbook(self, run_id: str, domain_payload: Dict[str, Any], model: str = "snowflake-arctic") -> Dict[str, Any]:
        prompt = PLAYBOOK_PROMPT.replace("{payload}", json.dumps(domain_payload))
        try:
            raw = self._ai_complete(model=model, prompt=prompt)
            parsed = self._parse_json(raw)
            required = {
                "executive_summary",
                "top_repos",
                "maturity_and_risks",
                "plan_30",
                "plan_60",
                "plan_90",
                "technical_notes",
                "engagement_strategy",
                "artifacts",
            }
            if not required.issubset(set(parsed.keys())):
                raise ValueError("Missing required keys")
            return parsed
        except Exception as exc:
            self.logger.log(run_id, "StrategistAgent", "domain_ai_fallback", "INFO", f"AI JSON parse failed: {exc}")
            top = [r.get("full_name") for r in domain_payload.get("repos", [])[:5]]
            return {
                "executive_summary": f"Fallback playbook for {domain_payload.get('domain')} using deterministic scores.",
                "top_repos": top,
                "maturity_and_risks": ["Risk based on activity, maintenance, and community metrics."],
                "plan_30": ["Shortlist top 3 repos and run PoC.", "Define adoption success metrics."],
                "plan_60": ["Integrate selected repo into one pilot workflow.", "Begin internal docs."],
                "plan_90": ["Promote successful pilot to production candidate.", "Set governance cadence."],
                "technical_notes": ["Validate license compatibility", "Review API stability and release cadence"],
                "engagement_strategy": ["Join community channels", "Contribute bug reports and docs"],
                "artifacts": ["Architecture decision record", "Risk register", "Migration checklist"],
            }


def playbook_to_markdown(domain: str, playbook: Dict[str, Any]) -> str:
    """Render playbook JSON to markdown for Streamlit and Notion publishing."""
    parts: List[str] = [f"# OpenSourceOps Adoption Playbook: {domain}"]
    parts.append(f"\n## Executive Summary\n{playbook.get('executive_summary', '')}")

    for key, title in [
        ("top_repos", "Top Repositories"),
        ("maturity_and_risks", "Maturity and Risks"),
        ("plan_30", "30-Day Plan"),
        ("plan_60", "60-Day Plan"),
        ("plan_90", "90-Day Plan"),
        ("technical_notes", "Technical Notes"),
        ("engagement_strategy", "Engagement Strategy"),
        ("artifacts", "Artifacts"),
    ]:
        parts.append(f"\n## {title}")
        values = playbook.get(key, []) or []
        for item in values:
            parts.append(f"- {item}")

    return "\n".join(parts)
