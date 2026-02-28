"""Snowflake Cortex AI_COMPLETE wrappers for optimization, reasoning, and refinement."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from src.query_pack import generate_query_pack
from src.run_logger import RunLogger
from src.snowflake_client import SnowflakeClient


class AIClient:
    """Cortex AI_COMPLETE integration with robust JSON parsing and graceful fallback."""

    def __init__(self, snowflake_client: SnowflakeClient, logger: RunLogger) -> None:
        self.sf = snowflake_client
        self.logger = logger
        self._logged_fallback_keys: set[str] = set()

    def _models(self, requested: str) -> List[str]:
        primary = os.getenv("CORTEX_MODEL_PRIMARY", requested)
        fallback = os.getenv("CORTEX_MODEL_FALLBACK", "mistral-7b")
        models = [primary]
        if fallback and fallback not in models:
            models.append(fallback)
        return models

    def _ai_complete(self, model: str, prompt: str) -> str:
        if not self.sf.is_configured():
            raise RuntimeError("Snowflake unavailable; AI_COMPLETE fallback mode")

        queries = [
            "SELECT AI_COMPLETE(%s, %s)",
            "SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s)",
            "SELECT SNOWFLAKE.CORTEX.TRY_COMPLETE(%s, %s)",
        ]
        for candidate_model in self._models(model):
            for query in queries:
                try:
                    rows = self.sf.execute(query, [candidate_model, prompt])
                    if rows and rows[0] and rows[0][0]:
                        return str(rows[0][0])
                except Exception as exc:
                    last_error = str(exc)
                    continue
        raise RuntimeError(
            f"Unable to call Snowflake AI_COMPLETE ({last_error if 'last_error' in locals() else 'unknown error'})"
        )

    @staticmethod
    def _parse_json(raw: str) -> Any:
        text = (raw or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        parsed: Any = json.loads(text)
        # Some models return a JSON string containing the JSON object.
        for _ in range(2):
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            else:
                break
        return parsed

    @staticmethod
    def _as_object(raw: str) -> Dict[str, Any]:
        parsed = AIClient._parse_json(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
        return parsed

    @staticmethod
    def _json_limited(payload: Dict[str, Any], max_chars: int = 6000) -> str:
        text = json.dumps(payload, default=str)
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "...[truncated]"

    def _contributor_master_prompt(self, payload: Dict[str, Any], feedback_text: str | None, max_chars: int) -> str:
        payload_text = self._json_limited(payload, max_chars=max_chars)
        feedback = feedback_text or ""
        return f"""
You are OpenSourceOps.
Return EXACTLY one JSON object. No markdown, no prose, no code fences.
Do NOT return a JSON string; return a JSON object.

TASK
Given persona, intent, constraints, repos, and opportunities:
1) Select ONE best repo or null if constraints cannot be satisfied.
2) Rank top repos with evidence.
3) Generate a practical weekly contribution plan from provided evidence only.

HARD RULES
- Ignore any preselected recommendation in input; choose independently.
- If constraints.maturity_preference == "Stable only" and no candidate is stable, set recommended_repo = null.
- Exclude curated lists/tutorial-only repos (e.g., awesome-*) unless intent explicitly requests them.
- Prefer constraints.language when provided.
- Use only payload evidence (description/topics/metrics/opportunities). No external facts.
- If opportunities are sparse, return fewer items; do not invent.
- Keep output concise and actionable.

OUTPUT SCHEMA (MUST MATCH)
{{
  "executive_summary": "string",
  "recommended_repo": {{
    "full_name": "string",
    "why_best": ["string"],
    "risks": ["string"],
    "fit_to_intent": "string"
  }} | null,
  "top_repos": [
    {{
      "full_name": "string",
      "rank": 1,
      "fit_score_reason": "string",
      "evidence": {{
        "description": "string",
        "topics": ["string"],
        "key_signals": {{
          "health_score": 0.0,
          "risk_score": 0.0,
          "intent_match_score": 0.0
        }}
      }}
    }}
  ],
  "weekly_plan": ["string"],
  "opportunity_explanations": [
    {{
      "repo_full_name": "string",
      "item_type": "ISSUE|DISCUSSION",
      "title": "string",
      "why_high_value": "string",
      "suggested_next_action": "string",
      "difficulty": "Easy|Medium|Hard",
      "contact_handles": ["string"]
    }}
  ],
  "possible_contacts": ["string"],
  "changes_and_justification": {{
    "kept_or_changed": "kept|changed|none",
    "explanation": "string"
  }}
}}

SIZE LIMITS
- top_repos: max 5
- weekly_plan: max 6 steps
- opportunity_explanations: max 8
- why_best/risks: max 4 bullets each

PAYLOAD:
{payload_text}

FEEDBACK:
{feedback}
""".strip()

    def optimize_query_pack(
        self,
        run_id: str,
        persona: str,
        domain: str,
        intent: str,
        constraints: Dict[str, Any],
        feedback_text: str | None = None,
        model: str = "snowflake-arctic",
    ) -> Dict[str, Any]:
        """Use Cortex to optimize repo queries and ranking weights; fallback to deterministic pack."""
        fallback = generate_query_pack(
            domain=domain,
            intent=intent,
            constraints=constraints,
            persona=persona,
            feedback_text=feedback_text,
        )

        prompt = (
            "Return strict JSON with keys: keywords (array), topics (array), repo_queries (array max 6), "
            "ranking_weights (object with health,intent,risk/community).\n"
            f"persona={persona}\n"
            f"domain={domain}\n"
            f"intent={intent}\n"
            f"constraints={json.dumps(constraints)}\n"
            f"feedback={feedback_text or ''}\n"
            "Only output JSON."
        )

        try:
            raw = self._ai_complete(model, prompt)
            parsed = self._parse_json(raw)
            queries = (parsed.get("repo_queries") or [])[:6]
            if not queries:
                raise ValueError("No queries returned")
            return {
                "keywords": (parsed.get("keywords") or fallback.keywords)[:8],
                "topics": (parsed.get("topics") or fallback.topics)[:8],
                "repo_queries": queries,
                "ranking_weights": parsed.get("ranking_weights") or fallback.ranking_weights,
            }
        except Exception as exc:
            self.logger.log(run_id, "ScoutAgent", "optimizer_fallback", "INFO", f"Using deterministic query pack: {exc}")
            return {
                "keywords": fallback.keywords,
                "topics": fallback.topics,
                "repo_queries": fallback.queries,
                "ranking_weights": fallback.ranking_weights,
            }

    def repo_summary(self, run_id: str, repo_payload: Dict[str, Any], model: str = "snowflake-arctic") -> Dict[str, Any]:
        prompt = (
            "Return strict JSON only with keys: maturity, summary, risks, recommended_use, engagement_plan.\n"
            f"payload={self._json_limited(repo_payload, max_chars=2500)}"
        )
        try:
            raw = self._ai_complete(model, prompt)
            parsed = self._as_object(raw)
            required = {"maturity", "summary", "risks", "recommended_use", "engagement_plan"}
            if not required.issubset(set(parsed.keys())):
                raise ValueError("Missing required keys")
            return parsed
        except Exception as exc:
            key = f"{run_id}:repo_ai_fallback"
            if key not in self._logged_fallback_keys:
                self.logger.log(run_id, "StrategistAgent", "repo_ai_fallback", "INFO", f"repo_summary fallback: {exc}")
                self._logged_fallback_keys.add(key)
            repo = repo_payload.get("repo", {})
            full_name = repo.get("full_name") or repo_payload.get("full_name") or "selected repository"
            desc = repo.get("description") or ""
            final_score = ((repo.get("scores") or {}).get("final")) or repo_payload.get("final_score")
            return {
                "maturity": "emerging",
                "summary": f"{full_name} is a strong intent match (final_score={final_score}) with active maintenance. {desc}",
                "risks": ["Validate roadmap fit", "Review maintainer responsiveness"],
                "recommended_use": "Pilot first in a non-critical workflow.",
                "engagement_plan": "Open a scoped issue/discussion and align with maintainers.",
            }

    def adopter_playbook(
        self,
        run_id: str,
        payload: Dict[str, Any],
        feedback_text: str | None = None,
        model: str = "snowflake-arctic",
    ) -> Dict[str, Any]:
        def _prompt(max_chars: int) -> str:
            return (
                "Return strict JSON only with keys: executive_summary, recommended_repo, top_repos, why_it_fits, "
                "integration_plan, migration_strategy, operational_considerations, risk_register, security_checklist, "
                "license_summary, poc_plan, alternatives, why_best, why_not_alternatives.\n"
                f"payload={self._json_limited(payload, max_chars=max_chars)}\nfeedback={feedback_text or ''}"
            )
        try:
            raw = self._ai_complete(model, _prompt(5000))
            parsed = self._as_object(raw)
            required = {
                "executive_summary",
                "recommended_repo",
                "top_repos",
                "why_it_fits",
                "integration_plan",
                "migration_strategy",
                "operational_considerations",
                "risk_register",
                "security_checklist",
                "license_summary",
                "poc_plan",
                "alternatives",
                "why_best",
                "why_not_alternatives",
            }
            if not required.issubset(set(parsed.keys())):
                raise ValueError("Missing required keys")
            return parsed
        except Exception as exc:
            if "max tokens" in str(exc).lower():
                try:
                    raw = self._ai_complete(model, _prompt(2200))
                    parsed = self._as_object(raw)
                    return parsed
                except Exception:
                    pass
            key = f"{run_id}:adopter_playbook_fallback"
            if key not in self._logged_fallback_keys:
                self.logger.log(run_id, "StrategistAgent", "adopter_playbook_fallback", "INFO", f"fallback: {exc}")
                self._logged_fallback_keys.add(key)
            top = [r.get("full_name") for r in payload.get("repos", [])[:5]]
            rec = top[0] if top else None
            return {
                "executive_summary": f"Adoption plan for {payload.get('domain')} generated via deterministic fallback.",
                "recommended_repo": rec,
                "top_repos": top,
                "why_it_fits": [
                    f"{rec} has the strongest intent/health/risk blend among candidates."
                    if rec else "No candidate met quality filters.",
                ],
                "integration_plan": [
                    "Define integration boundary and required APIs/events.",
                    "Implement adapter layer and sandbox deployment.",
                    "Validate reliability under failure/retry scenarios.",
                ],
                "migration_strategy": [
                    "Run side-by-side with current solution for one critical workflow.",
                    "Backfill data/state and validate parity.",
                    "Cut over incrementally behind a feature flag.",
                ],
                "operational_considerations": [
                    "Monitoring: success rate, latency, queue depth, retries.",
                    "Scaling: worker autoscaling + concurrency limits.",
                    "Upgrades: version pinning and rollout checklist.",
                ],
                "risk_register": [
                    "Vendor/community risk | Medium | Medium | Establish maintainer engagement cadence | Platform lead",
                    "Operational complexity | High | Medium | Build runbooks and SLO alerts early | SRE lead",
                ],
                "security_checklist": [
                    "Review authn/authz model and secret handling.",
                    "Run dependency and license scans in CI.",
                    "Validate audit logging and data retention policies.",
                ],
                "license_summary": "Validate repository license compatibility with internal policy before production adoption.",
                "poc_plan": [
                    "Week 1: integration spike + baseline reliability tests.",
                    "Week 2: pilot workflow in staging + go/no-go review.",
                ],
                "alternatives": top[1:4],
                "why_best": "Top-ranked repo has strongest combined health, intent match, and risk profile.",
                "why_not_alternatives": ["Lower intent fit", "Higher operational risk"],
            }

    def contributor_plan(
        self,
        run_id: str,
        payload: Dict[str, Any],
        feedback_text: str | None = None,
        model: str = "snowflake-arctic",
    ) -> Dict[str, Any]:
        def _prompt(max_chars: int) -> str:
            return self._contributor_master_prompt(payload, feedback_text, max_chars=max_chars)
        try:
            raw = self._ai_complete(model, _prompt(5000))
            parsed = self._as_object(raw)
            required = {
                "executive_summary",
                "recommended_repo",
                "top_repos",
                "weekly_plan",
                "opportunity_explanations",
                "possible_contacts",
                "changes_and_justification",
            }
            if not required.issubset(set(parsed.keys())):
                raise ValueError("Missing required keys")
            if isinstance(parsed.get("changes_and_justification"), str):
                parsed["changes_and_justification"] = {
                    "kept_or_changed": "none",
                    "explanation": parsed["changes_and_justification"],
                }
            return parsed
        except Exception as exc:
            if "max tokens" in str(exc).lower():
                try:
                    raw = self._ai_complete(model, _prompt(2200))
                    parsed = self._as_object(raw)
                    if isinstance(parsed.get("changes_and_justification"), str):
                        parsed["changes_and_justification"] = {
                            "kept_or_changed": "none",
                            "explanation": parsed["changes_and_justification"],
                        }
                    return parsed
                except Exception:
                    pass
            key = f"{run_id}:contributor_plan_fallback"
            if key not in self._logged_fallback_keys:
                self.logger.log(run_id, "StrategistAgent", "contributor_plan_fallback", "INFO", f"fallback: {exc}")
                self._logged_fallback_keys.add(key)
            top = [r.get("full_name") for r in payload.get("repos", [])[:5]]
            recommended_repo = payload.get("recommended_repo") or (top[0] if top else None)
            opps = [
                o for o in payload.get("opportunities", [])
                if not recommended_repo or o.get("repo_full_name") == recommended_repo
            ][:6]
            contacts = sorted(
                {
                    h
                    for o in opps
                    for h in (o.get("contact_handles") or [])
                    if h
                }
            )
            return {
                "executive_summary": f"Top recommendation is {recommended_repo}, chosen for the best blend of intent fit, issue quality, and maintainer activity.",
                "recommended_repo": {
                    "full_name": recommended_repo,
                    "why_best": ["Strong intent fit and active code-oriented opportunity set."],
                    "risks": ["Confirm roadmap alignment with maintainers before implementation."],
                    "fit_to_intent": f"Best alignment with intent for domain work on {payload.get('domain')}.",
                } if recommended_repo else None,
                "top_repos": [recommended_repo] if recommended_repo else top[:1],
                "weekly_plan": [
                    f"Week 1: pick 2 feature/enhancement issues in {recommended_repo}, confirm acceptance criteria, and post a scoped implementation plan.",
                    f"Week 2: ship one code contribution in {recommended_repo} and iterate on maintainer review quickly.",
                    f"Week 3: complete a second contribution (tests/docs/refactor) and close feedback loops with maintainers.",
                ],
                "opportunity_explanations": [f"{o.get('title')}: {o.get('why')}" for o in opps],
                "possible_contacts": contacts[:12],
                "changes_and_justification": {
                    "kept_or_changed": "none",
                    "explanation": (
                        f"{recommended_repo} remains top because it has highest final score and strongest contributor-ready issue signals."
                        if recommended_repo else
                        "Recommendation derived from deterministic scoring because AI output was unavailable."
                    ),
                },
            }


def playbook_to_markdown(persona: str, domain: str, payload: Dict[str, Any]) -> str:
    """Render adopter/contributor output JSON to markdown."""
    title = "Adoption Playbook" if persona == "Adopter" else "Contribution Plan"
    out: List[str] = [f"# OpenSourceOps {title}: {domain}"]
    out.append(f"\n## Executive Summary\n{payload.get('executive_summary', '')}")

    if persona == "Adopter":
        sections = [
            ("recommended_repo", "Top Recommended Repository"),
            ("top_repos", "Top Repositories"),
            ("why_it_fits", "Why It Fits"),
            ("integration_plan", "Integration Plan"),
            ("migration_strategy", "Migration Strategy"),
            ("operational_considerations", "Operational Considerations"),
            ("risk_register", "Risk Register"),
            ("security_checklist", "Security Checklist"),
            ("license_summary", "License Summary"),
            ("poc_plan", "Proof Of Concept Plan"),
            ("alternatives", "Alternatives"),
            ("why_best", "Why Best"),
            ("why_not_alternatives", "Why Not Alternatives"),
        ]
    else:
        sections = [
            ("recommended_repo", "Top Recommended Repository"),
            ("top_repos", "Top Repositories"),
            ("weekly_plan", "Weekly Plan"),
            ("opportunity_explanations", "Opportunity Explanations"),
            ("possible_contacts", "Possible GitHub Handles To Contact"),
            ("changes_and_justification", "Changes and Justification"),
        ]

    for key, title in sections:
        out.append(f"\n## {title}")
        value = payload.get(key, [])
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    label = item.get("full_name") or item.get("title") or json.dumps(item)
                    out.append(f"- {label}")
                else:
                    out.append(f"- {item}")
        elif isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, list):
                    out.append(f"- {k}:")
                    for item in v:
                        out.append(f"  - {item}")
                else:
                    out.append(f"- {k}: {v}")
        else:
            out.append(str(value))

    return "\n".join(out)
