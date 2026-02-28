"""CrewAI-style four-agent orchestration for OpenSourceOps persona workflows."""

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.ai import AIClient, playbook_to_markdown
from src.composio_integration import create_notion_playbook
from src.github_client import GitHubClient
from src.query_pack import generate_query_pack
from src.run_logger import RunLogger
from src.scoring import score_opportunities, score_repo
from src.snowflake_client import SnowflakeClient

try:
    from crewai import Agent
except Exception:  # pragma: no cover
    Agent = None


@dataclass
class AgentContext:
    run_id: str
    persona: str
    domain: str
    intent: str
    constraints: Dict[str, Any]
    cache_key: str
    force_refresh: bool = False
    feedback_section: str | None = None
    feedback_text: str | None = None
    previous_run_id: str | None = None


class BaseOpsAgent:
    name = "BaseAgent"
    objective = ""

    def __init__(self, logger: RunLogger):
        self.logger = logger
        self.crewai_agent = self._build_crewai_agent()

    def _build_crewai_agent(self):
        if Agent is None:
            return None
        # Keep CrewAI object creation optional to avoid implicit OpenAI dependency.
        if not os.getenv("OPENAI_API_KEY"):
            return None
        try:
            return Agent(role=self.name, goal=self.objective, backstory="OpenSourceOps specialized agent")
        except Exception:
            return None

    def _log_start(self, ctx: AgentContext, step: str):
        self.logger.log(ctx.run_id, self.name, step, "STARTED", f"{step} started")

    def _log_success(self, ctx: AgentContext, step: str, detail: str):
        self.logger.log(ctx.run_id, self.name, step, "SUCCESS", detail)

    def _log_failure(self, ctx: AgentContext, step: str, exc: Exception):
        detail = f"{type(exc).__name__}: {exc} | {traceback.format_exc(limit=1)}"
        self.logger.log(ctx.run_id, self.name, step, "FAILED", detail)


class ScoutAgent(BaseOpsAgent):
    name = "ScoutAgent"
    objective = "Generate optimized query pack and discover candidate repositories and raw opportunities."

    def __init__(self, logger: RunLogger, gh: GitHubClient, sf: SnowflakeClient, ai: AIClient):
        super().__init__(logger)
        self.gh = gh
        self.sf = sf
        self.ai = ai

    def run(self, ctx: AgentContext) -> Dict[str, Any]:
        step = "discover"
        self._log_start(ctx, step)
        try:
            optimizer = self.ai.optimize_query_pack(
                run_id=ctx.run_id,
                persona=ctx.persona,
                domain=ctx.domain,
                intent=ctx.intent,
                constraints=ctx.constraints,
                feedback_text=ctx.feedback_text,
            )
            base_pack = generate_query_pack(
                domain=ctx.domain,
                intent=ctx.intent,
                constraints=ctx.constraints,
                persona=ctx.persona,
                feedback_text=ctx.feedback_text,
            )
            combined_queries = list(dict.fromkeys([*(optimizer.get("repo_queries") or []), *base_pack.queries]))
            optimizer["keywords"] = list(dict.fromkeys([*(optimizer.get("keywords") or []), *base_pack.keywords]))[:8]
            optimizer["topics"] = list(dict.fromkeys([*(optimizer.get("topics") or []), *base_pack.topics]))[:8]

            enriched_topics = self._topic_enrichment(ctx, optimizer.get("keywords", []))
            optimizer["topics"] = list(dict.fromkeys([*optimizer.get("topics", []), *enriched_topics]))[:8]

            self._log_success(
                ctx,
                "query_optimizer",
                f"Generated {len(combined_queries)} candidate queries with {len(optimizer.get('topics', []))} topics",
            )
            optimizer["repo_queries"] = self._sanitize_queries(combined_queries, ctx)

            max_queries = min(int(os.getenv("GITHUB_MAX_SEARCH_QUERIES", "3")), 6)
            top_k = int(os.getenv("GITHUB_TOP_K_REPOS", "25"))
            per_page = int(os.getenv("GITHUB_PER_QUERY_PAGE_SIZE", "15"))
            include_contributors = os.getenv("GITHUB_FETCH_CONTRIBUTORS", "false").lower() == "true"

            repos = (
                self.gh.search_repositories(
                    optimizer.get("repo_queries", []),
                    top_k=top_k,
                    max_queries=max_queries,
                    per_page=per_page,
                    include_contributors=include_contributors,
                )
                if self.gh.is_configured()
                else []
            )
            if not repos and not self.gh.is_configured():
                repos = self._mock_repos(ctx)
                self.logger.log(ctx.run_id, self.name, "dry_run_repos", "INFO", "GITHUB_TOKEN missing; using mock repos")
            else:
                hinted_repos = self._feedback_repo_hints(ctx.feedback_text)
                repos = self._inject_hinted_repos(repos, hinted_repos)
                repos = self._filter_repos(repos, ctx, optimizer, hinted_repos)

            repo_rows = [
                {
                    "run_id": ctx.run_id,
                    "cache_key": ctx.cache_key,
                    "persona": ctx.persona,
                    "domain": ctx.domain,
                    "language": ctx.constraints.get("language"),
                    "full_name": r.get("full_name"),
                    "html_url": r.get("html_url"),
                    "description": r.get("description"),
                    "topics": r.get("topics", []),
                    "repo_language": r.get("language"),
                    "stargazers_count": r.get("stargazers_count", 0),
                    "forks_count": r.get("forks_count", 0),
                    "open_issues_count": r.get("open_issues_count", 0),
                    "watchers_count": r.get("watchers_count", 0),
                    "license": r.get("license"),
                    "pushed_at": r.get("pushed_at"),
                    "created_at": r.get("created_at"),
                    "contributors_count": r.get("contributors_count", 0),
                }
                for r in repos
            ]
            self.sf.insert_rows("REPOS", repo_rows)

            issues: List[Dict[str, Any]] = []
            discussions: List[Dict[str, Any]] = []
            discussions_available = False

            if ctx.persona == "Contributor" and repos:
                issues = self.gh.fetch_issues_for_repos(
                    repos,
                    per_repo=int(os.getenv("GITHUB_ISSUES_PER_REPO", "10")),
                    max_repos=int(os.getenv("GITHUB_ISSUE_REPO_LIMIT", "5")),
                ) if self.gh.is_configured() else []
                issue_rows = [
                    {
                        "run_id": ctx.run_id,
                        "cache_key": ctx.cache_key,
                        "persona": ctx.persona,
                        "domain": ctx.domain,
                        "full_name": i.get("repo_full_name"),
                        "issue_id": i.get("item_id"),
                        "title": i.get("title"),
                        "body": i.get("body"),
                        "labels": i.get("labels", []),
                        "comments": i.get("comments", 0),
                        "created_at": None,
                        "updated_at": i.get("updated_at"),
                        "html_url": i.get("url"),
                    }
                    for i in issues
                ]
                self.sf.insert_rows("ISSUES", issue_rows)

                discussions, discussions_available = self.gh.fetch_discussions_for_repos(
                    repos,
                    per_repo=int(os.getenv("GITHUB_DISCUSSIONS_PER_REPO", "8")),
                    max_repos=int(os.getenv("GITHUB_DISCUSSION_REPO_LIMIT", "4")),
                ) if self.gh.is_configured() else ([], False)

                if not discussions_available:
                    self.logger.log(
                        ctx.run_id,
                        self.name,
                        "discussions_unavailable",
                        "INFO",
                        "GitHub discussions unavailable. Contributor opportunities use issues only.",
                    )

            self._log_success(
                ctx,
                step,
                f"Discovered repos={len(repos)}, issues={len(issues)}, discussions={len(discussions)}",
            )
            return {
                "optimizer": optimizer,
                "repos": repos,
                "issues": issues,
                "discussions": discussions,
                "discussions_available": discussions_available,
            }
        except Exception as exc:
            self._log_failure(ctx, step, exc)
            fallback = generate_query_pack(ctx.domain, ctx.intent, ctx.constraints, ctx.persona)
            return {
                "optimizer": {
                    "keywords": fallback.keywords,
                    "topics": fallback.topics,
                    "repo_queries": fallback.queries,
                    "ranking_weights": fallback.ranking_weights,
                },
                "repos": [],
                "issues": [],
                "discussions": [],
                "discussions_available": False,
            }

    @staticmethod
    def _sanitize_queries(queries: List[str], ctx: AgentContext) -> List[str]:
        """Ensure query pack keeps strong GitHub search constraints."""
        sanitized: List[str] = []
        lang = ctx.constraints.get("language")
        for q in queries[:6]:
            candidate = q
            if "stars:" not in candidate:
                candidate += " stars:>50"
            if "pushed:>" not in candidate:
                candidate += " pushed:>2024-01-01"
            if "archived:" not in candidate:
                candidate += " archived:false"
            if "fork:" not in candidate:
                candidate += " fork:false"
            if lang and f"language:{lang}" not in candidate:
                candidate += f" language:{lang}"
            sanitized.append(candidate)
        return sanitized

    def _topic_enrichment(self, ctx: AgentContext, keywords: List[str]) -> List[str]:
        if not self.gh.is_configured():
            return []
        query = " ".join([ctx.domain, *keywords[:3]]).strip()
        return self.gh.search_topics(query=query, per_page=6)

    @staticmethod
    def _filter_repos(
        repos: List[Dict[str, Any]],
        ctx: AgentContext,
        optimizer: Dict[str, Any],
        hinted_repos: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """Drop low-signal/off-intent repos after search."""
        min_stars = int(os.getenv("GITHUB_MIN_STARS", "50"))
        keywords = [k.lower() for k in (optimizer.get("keywords") or [])]
        intent_text = f"{ctx.domain} {ctx.intent}".lower()
        allow_tutorial = any(k in intent_text for k in ["tutorial", "awesome", "examples"])
        deny_terms = ["awesome-", "minecraft", "boilerplate", "starter", "template", "tutorial", "course"]

        hinted_set = set(hinted_repos or [])
        filtered: List[Dict[str, Any]] = []
        for r in repos:
            name = (r.get("full_name") or "").lower()
            desc = (r.get("description") or "").lower()
            text = f"{name} {desc} {' '.join(r.get('topics', []) or [])}".lower()
            stars = int(r.get("stargazers_count") or 0)

            if stars < min_stars and (r.get("full_name") not in hinted_set):
                continue
            if not allow_tutorial and any(t in name for t in deny_terms):
                continue

            lang = (ctx.constraints.get("language") or "").lower().strip()
            repo_lang = (r.get("language") or "").lower().strip()
            if lang and repo_lang and lang != repo_lang:
                continue

            matches = sum(1 for k in keywords if len(k) > 2 and k in text)
            hinted = bool(r.get("feedback_hint_match"))
            if keywords and matches == 0 and not hinted:
                # Keep only very high-signal repos when keyword overlap is absent.
                if stars < 5000:
                    continue
            filtered.append(r)

        return filtered[: int(os.getenv("GITHUB_TOP_K_REPOS", "25"))]

    @staticmethod
    def _feedback_repo_hints(feedback_text: str | None) -> List[str]:
        text = (feedback_text or "").lower()
        hints: List[str] = []
        mapping = {
            "temporal": "temporalio/temporal",
            "cadence": "cadence-workflow/cadence",
            "durable task": "temporalio/temporal",
            "durable execution": "temporalio/temporal",
        }
        for token, repo in mapping.items():
            if token in text and repo not in hints:
                hints.append(repo)
        return hints

    def _inject_hinted_repos(self, repos: List[Dict[str, Any]], hinted_repos: List[str]) -> List[Dict[str, Any]]:
        if not hinted_repos or not self.gh.is_configured():
            return repos
        merged = {r.get("full_name"): r for r in repos if r.get("full_name")}
        for full_name in hinted_repos:
            fetched = self.gh.fetch_repository(full_name)
            if fetched:
                fetched["feedback_hint_match"] = True
                merged[full_name] = {**merged.get(full_name, {}), **fetched, "feedback_hint_match": True}
        return list(merged.values())

    @staticmethod
    def _mock_repos(ctx: AgentContext) -> List[Dict[str, Any]]:
        base = [
            ("langchain-ai/langchain", "Framework for building LLM apps"),
            ("run-llama/llama_index", "Context and retrieval orchestration"),
            ("deepset-ai/haystack", "RAG and search pipelines"),
            ("qdrant/qdrant", "Vector search engine"),
            ("chroma-core/chroma", "Embedding database"),
        ]
        rows = []
        for idx, (name, desc) in enumerate(base, start=1):
            rows.append(
                {
                    "full_name": name,
                    "html_url": f"https://github.com/{name}",
                    "description": f"{desc} ({ctx.domain})",
                    "topics": ["rag", "evaluation"],
                    "language": ctx.constraints.get("language") or "Python",
                    "stargazers_count": 2200 - (idx * 120),
                    "forks_count": 300 - (idx * 10),
                    "open_issues_count": 45 + idx,
                    "watchers_count": 150 + (idx * 5),
                    "license": "MIT",
                    "pushed_at": "2026-02-20T00:00:00Z",
                    "created_at": "2020-01-01T00:00:00Z",
                    "contributors_count": 60 + idx,
                }
            )
        return rows


class AnalystAgent(BaseOpsAgent):
    name = "AnalystAgent"
    objective = "Compute deterministic repository and contribution opportunity scores."

    def __init__(self, logger: RunLogger, sf: SnowflakeClient):
        super().__init__(logger)
        self.sf = sf

    def run(
        self,
        ctx: AgentContext,
        repos: List[Dict[str, Any]],
        issues: List[Dict[str, Any]],
        discussions: List[Dict[str, Any]],
        optimizer: Dict[str, Any],
    ) -> Dict[str, Any]:
        step = "score"
        self._log_start(ctx, step)
        try:
            intent_keywords = optimizer.get("keywords") or []
            weights = optimizer.get("ranking_weights") or {}

            scored_repos: List[Dict[str, Any]] = []
            for repo in repos:
                score = score_repo(repo, intent_keywords=intent_keywords)
                merged = {**repo, **score}
                # Persona-specific final score blending with optimizer weights.
                if ctx.persona == "Contributor":
                    merged["final_score"] = round(
                        max(0.0, min(1.0,
                            weights.get("health", 0.3) * merged["health_score"]
                            + weights.get("intent", 0.4) * merged["intent_match_score"]
                            + weights.get("community", 0.3) * merged["community_score"]
                        )),
                        4,
                    )
                else:
                    merged["final_score"] = round(
                        max(0.0, min(1.0,
                            weights.get("health", 0.45) * merged["health_score"]
                            + weights.get("intent", 0.35) * merged["intent_match_score"]
                            + weights.get("risk", 0.2) * (1 - merged["risk_score"])
                        )),
                        4,
                    )
                scored_repos.append(merged)

            scored_repos.sort(key=lambda x: x.get("final_score", 0), reverse=True)
            score_rows = [
                {
                    "run_id": ctx.run_id,
                    "cache_key": ctx.cache_key,
                    "persona": ctx.persona,
                    "full_name": r.get("full_name"),
                    "intent_match_score": r.get("intent_match_score"),
                    "activity_score": r.get("activity_score"),
                    "adoption_score": r.get("adoption_score"),
                    "maintenance_score": r.get("maintenance_score"),
                    "community_score": r.get("community_score"),
                    "health_score": r.get("health_score"),
                    "risk_score": r.get("risk_score"),
                    "final_score": r.get("final_score"),
                }
                for r in scored_repos
            ]
            self.sf.insert_rows("REPO_SCORES", score_rows)

            opportunities: List[Dict[str, Any]] = []
            if ctx.persona == "Contributor":
                issue_first = score_opportunities(issues, intent_keywords)
                discussion_scored = score_opportunities(discussions, intent_keywords)
                # Prefer feature/code contribution issues; only add discussions as fallback context.
                opportunities = issue_first[:40]
                if len(opportunities) < 12:
                    opportunities.extend(discussion_scored[: max(0, 12 - len(opportunities))])
                opp_rows = [
                    {
                        "run_id": ctx.run_id,
                        "domain": ctx.domain,
                        "persona": ctx.persona,
                        "cache_key": ctx.cache_key,
                        "repo_full_name": o.get("repo_full_name"),
                        "item_type": o.get("item_type"),
                        "item_id": o.get("item_id"),
                        "title": o.get("title"),
                        "body": o.get("body"),
                        "labels": o.get("labels", []),
                        "comments": o.get("comments", 0),
                        "updated_at": o.get("updated_at"),
                        "url": o.get("url"),
                        "engagement_score": o.get("engagement_score"),
                        "scope_score": o.get("scope_score"),
                        "relevance_score": o.get("relevance_score"),
                        "ease_score": o.get("ease_score"),
                        "final_score": o.get("final_score"),
                        "why": o.get("why"),
                        "suggested_next_action": o.get("suggested_next_action"),
                        "difficulty": o.get("difficulty"),
                    }
                    for o in opportunities
                ]
                self.sf.insert_rows("CONTRIBUTION_OPPORTUNITIES", opp_rows)

            self._log_success(
                ctx,
                step,
                f"Scored repos={len(scored_repos)} and opportunities={len(opportunities)}",
            )
            return {"scored_repos": scored_repos, "opportunities": opportunities}
        except Exception as exc:
            self._log_failure(ctx, step, exc)
            return {"scored_repos": [], "opportunities": []}


class StrategistAgent(BaseOpsAgent):
    name = "StrategistAgent"
    objective = "Generate structured AI outputs for adopter and contributor personas."

    def __init__(self, logger: RunLogger, sf: SnowflakeClient, ai_client: AIClient):
        super().__init__(logger)
        self.sf = sf
        self.ai = ai_client

    def run(
        self,
        ctx: AgentContext,
        scored_repos: List[Dict[str, Any]],
        opportunities: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        step = "reason_and_plan"
        self._log_start(ctx, step)
        try:
            top_repos = scored_repos[:10]
            repo_ai: Dict[str, Dict[str, Any]] = {}
            repo_ai_rows: List[Dict[str, Any]] = []

            for repo in top_repos:
                repo_payload = {
                    "persona": ctx.persona,
                    "domain": ctx.domain,
                    "intent": ctx.intent,
                    "constraints": ctx.constraints,
                    "repo": {
                        "full_name": repo.get("full_name"),
                        "description": repo.get("description"),
                        "topics": repo.get("topics", []),
                        "scores": {
                            "final": repo.get("final_score"),
                            "health": repo.get("health_score"),
                            "risk": repo.get("risk_score"),
                            "intent": repo.get("intent_match_score"),
                        },
                        "stats": {
                            "stars": repo.get("stargazers_count"),
                            "forks": repo.get("forks_count"),
                            "issues": repo.get("open_issues_count"),
                        },
                    },
                }
                summary = self.ai.repo_summary(ctx.run_id, repo_payload)
                repo_ai[repo.get("full_name")] = summary
                repo_ai_rows.append(
                    {
                        "run_id": ctx.run_id,
                        "cache_key": ctx.cache_key,
                        "persona": ctx.persona,
                        "domain": ctx.domain,
                        "full_name": repo.get("full_name"),
                        "maturity": summary.get("maturity"),
                        "summary": summary.get("summary"),
                        "risks": summary.get("risks", []),
                        "recommended_use": summary.get("recommended_use"),
                        "engagement_plan": summary.get("engagement_plan"),
                        "raw_json": summary,
                    }
                )
            self.sf.insert_rows("REPO_AI", repo_ai_rows)

            payload = {
                "persona": ctx.persona,
                "domain": ctx.domain,
                "intent": ctx.intent,
                "constraints": ctx.constraints,
                "repos": [
                    {
                        "full_name": r.get("full_name"),
                        "final_score": r.get("final_score"),
                        "health_score": r.get("health_score"),
                        "risk_score": r.get("risk_score"),
                        "intent_match_score": r.get("intent_match_score"),
                        "maturity": (repo_ai.get(r.get("full_name"), {}) or {}).get("maturity"),
                    }
                    for r in top_repos
                ],
                "recommended_repo": top_repos[0].get("full_name") if top_repos else None,
                "opportunities": [
                    {
                        "repo_full_name": o.get("repo_full_name"),
                        "item_type": o.get("item_type"),
                        "title": o.get("title"),
                        "labels": (o.get("labels") or [])[:3],
                        "comments": o.get("comments"),
                        "final_score": o.get("final_score"),
                        "why": o.get("why"),
                        "difficulty": o.get("difficulty"),
                        "contact_handles": o.get("contact_handles", []),
                    }
                    for o in opportunities[:10]
                ],
            }

            if ctx.persona == "Adopter":
                structured = self.ai.adopter_playbook(ctx.run_id, payload, ctx.feedback_text)
            else:
                structured = self.ai.contributor_plan(ctx.run_id, payload, ctx.feedback_text)

            playbook_md = playbook_to_markdown(ctx.persona, ctx.domain, structured)
            self.sf.insert_rows(
                "DOMAIN_AI",
                [
                    {
                        "run_id": ctx.run_id,
                        "persona": ctx.persona,
                        "domain": ctx.domain,
                        "cache_key": ctx.cache_key,
                        "playbook": structured,
                        "playbook_md": playbook_md,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    }
                ],
            )

            self._log_success(ctx, step, "Generated structured AI outputs")
            return {
                "repo_ai": repo_ai,
                "playbook_json": structured,
                "playbook_md": playbook_md,
            }
        except Exception as exc:
            self._log_failure(ctx, step, exc)
            return {"repo_ai": {}, "playbook_json": {}, "playbook_md": ""}


class CoordinatorAgent(BaseOpsAgent):
    name = "CoordinatorAgent"
    objective = "Coordinate refinement strategy, cache behavior, and external actions."

    def __init__(self, logger: RunLogger, sf: SnowflakeClient):
        super().__init__(logger)
        self.sf = sf

    @staticmethod
    def should_redo_search(feedback_section: str | None, feedback_text: str | None) -> bool:
        if not feedback_section and not feedback_text:
            return False
        section = (feedback_section or "").lower()
        text = (feedback_text or "").lower()
        hard_refresh_markers = ["repo", "opportun", "constraint", "language", "license", "maturity"]
        return any(m in section for m in hard_refresh_markers) or any(m in text for m in hard_refresh_markers)

    @staticmethod
    def build_changes_justification(previous: Optional[Dict[str, Any]], current: Dict[str, Any]) -> str:
        if not previous:
            return "Initial run generated from current intent and constraints."

        old_top = (previous.get("scored_repos") or [{}])[0].get("full_name") if previous.get("scored_repos") else None
        new_top = (current.get("scored_repos") or [{}])[0].get("full_name") if current.get("scored_repos") else None
        if old_top and new_top and old_top == new_top:
            return f"Top repository remains {new_top} because feedback did not improve alternatives on intent/risk balance."
        if old_top and new_top and old_top != new_top:
            return f"Top repository changed from {old_top} to {new_top} due to improved fit with updated feedback/constraints."
        return "Result set changed based on refreshed optimization and available GitHub signals."

    def finalize(self, ctx: AgentContext, detail: str) -> None:
        self.logger.log(ctx.run_id, self.name, "finalize", "SUCCESS", detail)

    def create_notion(self, ctx: AgentContext, composio_client: Any, playbook_md: str, dry_run: bool = True) -> Dict[str, Any]:
        step = "create_notion_playbook"
        self._log_start(ctx, step)
        try:
            result = create_notion_playbook(
                composio_client=composio_client,
                run_id=ctx.run_id,
                domain=ctx.domain,
                persona=ctx.persona,
                playbook_md=playbook_md,
                dry_run=dry_run,
            )
            status = "SUCCESS" if result.get("ok") else "FAILED"
            self.logger.log(ctx.run_id, self.name, step, status, result.get("message", ""))
            return result
        except Exception as exc:
            self._log_failure(ctx, step, exc)
            return {"ok": False, "message": str(exc), "url": None, "payload": {}}
