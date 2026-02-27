"""CrewAI-inspired multi-agent orchestration for OpenSourceOps."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from typing import Any, Dict, List

from src.ai import AIClient, playbook_to_markdown
from src.composio_integration import create_notion_playbook
from src.github_client import GitHubClient
from src.query_pack import QueryPack, generate_query_pack
from src.run_logger import RunLogger
from src.scoring import score_issues, score_repo
from src.snowflake_client import SnowflakeClient

try:
    from crewai import Agent
except Exception:  # pragma: no cover
    Agent = None


@dataclass
class AgentContext:
    run_id: str
    domain: str
    language: str | None
    mode: str


class BaseOpsAgent:
    """Base class for uniform run logging and robust exception handling."""

    name = "BaseAgent"
    objective = ""

    def __init__(self, logger: RunLogger):
        self.logger = logger
        self.crewai_agent = self._build_crewai_agent()

    def _build_crewai_agent(self):
        if Agent is None:
            return None
        # CrewAI defaults to ChatOpenAI if no LLM is provided, which requires OPENAI_API_KEY.
        # OpenSourceOps runs a deterministic pipeline and only uses CrewAI objects when explicitly enabled.
        if not os.getenv("OPENAI_API_KEY"):
            return None
        try:
            return Agent(
                role=self.name,
                goal=self.objective,
                backstory="OpenSourceOps specialized agent",
            )
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
    objective = "Discover relevant repositories and contribution issues from GitHub."

    def __init__(self, logger: RunLogger, gh: GitHubClient, sf: SnowflakeClient):
        super().__init__(logger)
        self.gh = gh
        self.sf = sf

    def run(self, ctx: AgentContext) -> Dict[str, Any]:
        step = "discover_repositories"
        self._log_start(ctx, step)
        try:
            qp: QueryPack = generate_query_pack(ctx.domain, ctx.language)
            max_queries = int(os.getenv("GITHUB_MAX_SEARCH_QUERIES", "2"))
            top_k = int(os.getenv("GITHUB_TOP_K_REPOS", "25"))
            per_page = int(os.getenv("GITHUB_PER_QUERY_PAGE_SIZE", "15"))
            include_contributors = os.getenv("GITHUB_FETCH_CONTRIBUTORS", "false").lower() == "true"
            repos = (
                self.gh.search_repositories(
                    qp.queries,
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
                self.logger.log(
                    ctx.run_id,
                    self.name,
                    "dry_run_repos",
                    "INFO",
                    "GITHUB_TOKEN missing; generated mock repositories for dry-run.",
                )

            repo_rows = []
            for repo in repos:
                repo_rows.append(
                    {
                        "run_id": ctx.run_id,
                        "domain": ctx.domain,
                        "language": ctx.language,
                        "full_name": repo.get("full_name"),
                        "html_url": repo.get("html_url"),
                        "description": repo.get("description"),
                        "topics": repo.get("topics", []),
                        "repo_language": repo.get("language"),
                        "stargazers_count": repo.get("stargazers_count", 0),
                        "forks_count": repo.get("forks_count", 0),
                        "open_issues_count": repo.get("open_issues_count", 0),
                        "watchers_count": repo.get("watchers_count", 0),
                        "license": repo.get("license"),
                        "pushed_at": repo.get("pushed_at"),
                        "created_at": repo.get("created_at"),
                        "contributors_count": repo.get("contributors_count", 0),
                    }
                )

            self.sf.insert_rows("REPOS", repo_rows)
            insert_err = self.sf.consume_last_insert_error()
            if insert_err:
                self.logger.log(
                    ctx.run_id,
                    self.name,
                    "persist_repos_fallback",
                    "FAILED",
                    f"Snowflake write failed; using in-memory fallback. {insert_err}",
                )
            self._log_success(
                ctx,
                step,
                (
                    f"Fetched {len(repos)} repositories using "
                    f"{min(len(qp.queries), max_queries)} queries (per_page={per_page}, top_k={top_k})"
                ),
            )

            issues = []
            if ctx.mode == "Contributor" and repos:
                issue_step = "discover_issues"
                self._log_start(ctx, issue_step)
                issue_repos = int(os.getenv("GITHUB_ISSUE_REPO_LIMIT", "5"))
                per_repo_issues = int(os.getenv("GITHUB_ISSUES_PER_REPO", "10"))
                issues = (
                    self.gh.fetch_issues_for_repos(
                        repos[:issue_repos],
                        per_repo=per_repo_issues,
                        max_repos=issue_repos,
                    )
                    if self.gh.is_configured()
                    else []
                )
                issue_rows = [
                    {
                        "run_id": ctx.run_id,
                        "domain": ctx.domain,
                        "full_name": i.get("full_name"),
                        "issue_id": i.get("id"),
                        "title": i.get("title"),
                        "body": i.get("body"),
                        "labels": i.get("labels", []),
                        "comments": i.get("comments", 0),
                        "created_at": i.get("created_at"),
                        "updated_at": i.get("updated_at"),
                        "html_url": i.get("html_url"),
                    }
                    for i in issues
                ]
                self.sf.insert_rows("ISSUES", issue_rows)
                issue_insert_err = self.sf.consume_last_insert_error()
                if issue_insert_err:
                    self.logger.log(
                        ctx.run_id,
                        self.name,
                        "persist_issues_fallback",
                        "FAILED",
                        f"Snowflake write failed; using in-memory fallback. {issue_insert_err}",
                    )
                self._log_success(
                    ctx,
                    issue_step,
                    f"Fetched {len(issues)} labeled issues (repos={issue_repos}, per_repo={per_repo_issues})",
                )

            return {"query_pack": qp, "repos": repos, "issues": issues}
        except Exception as exc:
            self._log_failure(ctx, step, exc)
            return {"query_pack": generate_query_pack(ctx.domain, ctx.language), "repos": [], "issues": []}

    @staticmethod
    def _mock_repos(ctx: AgentContext) -> List[Dict[str, Any]]:
        """Generate deterministic mock repositories so dry-run demo still produces output."""
        base = [
            ("langchain-ai/langchain", "Framework for LLM applications"),
            ("run-llama/llama_index", "Data framework for LLM context"),
            ("deepset-ai/haystack", "LLM orchestration and retrieval stack"),
            ("qdrant/qdrant", "Vector search engine"),
            ("chroma-core/chroma", "Embedding database for AI apps"),
            ("openai/evals", "Evaluation framework for LLM systems"),
            ("promptfoo/promptfoo", "Prompt and eval testing"),
            ("mlflow/mlflow", "ML lifecycle platform"),
            ("langfuse/langfuse", "LLM observability and analytics"),
            ("arangodb/arangodb", "Graph + document database"),
        ]
        rows: List[Dict[str, Any]] = []
        for idx, (name, desc) in enumerate(base, start=1):
            rows.append(
                {
                    "full_name": name,
                    "html_url": f"https://github.com/{name}",
                    "description": f"{desc} for domain: {ctx.domain}",
                    "topics": ["opensourceops", "dry-run", "demo"],
                    "language": ctx.language or ("Python" if idx % 2 else "TypeScript"),
                    "stargazers_count": 2500 - idx * 130,
                    "forks_count": 300 - idx * 10,
                    "open_issues_count": 40 + idx,
                    "watchers_count": 120 + idx * 3,
                    "license": "MIT",
                    "pushed_at": "2026-02-20T00:00:00Z",
                    "created_at": "2020-01-01T00:00:00Z",
                    "contributors_count": 50 + idx,
                }
            )
        return rows


class AnalystAgent(BaseOpsAgent):
    name = "AnalystAgent"
    objective = "Compute deterministic health and risk scores for repositories and issues."

    def __init__(self, logger: RunLogger, sf: SnowflakeClient):
        super().__init__(logger)
        self.sf = sf

    def run(self, ctx: AgentContext, repos: List[Dict], issues: List[Dict]) -> Dict[str, Any]:
        step = "score_assets"
        self._log_start(ctx, step)
        try:
            scored_repos = []
            for repo in repos:
                score = score_repo(repo)
                merged = {**repo, **score}
                scored_repos.append(merged)

            scored_repos.sort(key=lambda x: x.get("health_score", 0), reverse=True)

            score_rows = [
                {
                    "run_id": ctx.run_id,
                    "full_name": r["full_name"],
                    "activity_score": r["activity_score"],
                    "adoption_score": r["adoption_score"],
                    "maintenance_score": r["maintenance_score"],
                    "community_score": r["community_score"],
                    "health_score": r["health_score"],
                    "risk_score": r["risk_score"],
                }
                for r in scored_repos
            ]
            self.sf.insert_rows("REPO_SCORES", score_rows)
            score_insert_err = self.sf.consume_last_insert_error()
            if score_insert_err:
                self.logger.log(
                    ctx.run_id,
                    self.name,
                    "persist_repo_scores_fallback",
                    "FAILED",
                    f"Snowflake write failed; using in-memory fallback. {score_insert_err}",
                )

            issue_scores = score_issues(issues) if issues else []
            issue_rows = [
                {
                    "run_id": ctx.run_id,
                    "full_name": next((i.get("full_name") for i in issues if i.get("id") == s.get("issue_id")), None),
                    "issue_id": s["issue_id"],
                    "impact_score": s["impact_score"],
                    "difficulty_score": s["difficulty_score"],
                    "reputation_score": s["reputation_score"],
                }
                for s in issue_scores
            ]
            if issue_rows:
                self.sf.insert_rows("ISSUE_SCORES", issue_rows)
                issue_score_err = self.sf.consume_last_insert_error()
                if issue_score_err:
                    self.logger.log(
                        ctx.run_id,
                        self.name,
                        "persist_issue_scores_fallback",
                        "FAILED",
                        f"Snowflake write failed; using in-memory fallback. {issue_score_err}",
                    )

            self._log_success(
                ctx,
                step,
                f"Scored {len(scored_repos)} repos and {len(issue_scores)} issues",
            )
            return {"scored_repos": scored_repos, "issue_scores": issue_scores}
        except Exception as exc:
            self._log_failure(ctx, step, exc)
            return {"scored_repos": [], "issue_scores": []}


class StrategistAgent(BaseOpsAgent):
    name = "StrategistAgent"
    objective = "Generate AI summaries and domain adoption playbook from top repositories."""

    def __init__(self, logger: RunLogger, sf: SnowflakeClient, ai_client: AIClient):
        super().__init__(logger)
        self.sf = sf
        self.ai = ai_client

    def run(self, ctx: AgentContext, scored_repos: List[Dict]) -> Dict[str, Any]:
        step = "generate_playbook"
        self._log_start(ctx, step)
        try:
            top_repos = scored_repos[:10]
            repo_ai_rows = []
            repo_ai = {}
            for repo in top_repos:
                payload = {
                    "domain": ctx.domain,
                    "full_name": repo.get("full_name"),
                    "description": repo.get("description"),
                    "topics": repo.get("topics", []),
                    "scores": {
                        "health": repo.get("health_score"),
                        "risk": repo.get("risk_score"),
                        "activity": repo.get("activity_score"),
                    },
                    "stats": {
                        "stars": repo.get("stargazers_count"),
                        "forks": repo.get("forks_count"),
                        "issues": repo.get("open_issues_count"),
                    },
                }
                result = self.ai.repo_summary(ctx.run_id, payload)
                repo_ai[repo["full_name"]] = result
                repo_ai_rows.append(
                    {
                        "run_id": ctx.run_id,
                        "domain": ctx.domain,
                        "full_name": repo["full_name"],
                        "maturity": result.get("maturity"),
                        "summary": result.get("summary"),
                        "risks": result.get("risks", []),
                        "recommended_use": result.get("recommended_use"),
                        "engagement_plan": result.get("engagement_plan"),
                        "raw_json": result,
                    }
                )

            self.sf.insert_rows("REPO_AI", repo_ai_rows)
            repo_ai_err = self.sf.consume_last_insert_error()
            if repo_ai_err:
                self.logger.log(
                    ctx.run_id,
                    self.name,
                    "persist_repo_ai_fallback",
                    "FAILED",
                    f"Snowflake write failed; using in-memory fallback. {repo_ai_err}",
                )
            domain_payload = {
                "domain": ctx.domain,
                "language": ctx.language,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "repos": [
                    {
                        "full_name": r.get("full_name"),
                        "health_score": r.get("health_score"),
                        "risk_score": r.get("risk_score"),
                        "ai": repo_ai.get(r.get("full_name"), {}),
                    }
                    for r in top_repos
                ],
            }
            playbook_json = self.ai.domain_playbook(ctx.run_id, domain_payload)
            playbook_md = playbook_to_markdown(ctx.domain, playbook_json)

            self.sf.insert_rows(
                "DOMAIN_AI",
                [
                    {
                        "run_id": ctx.run_id,
                        "domain": ctx.domain,
                        "language": ctx.language,
                        "playbook_json": playbook_json,
                        "playbook_md": playbook_md,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    }
                ],
            )
            domain_ai_err = self.sf.consume_last_insert_error()
            if domain_ai_err:
                self.logger.log(
                    ctx.run_id,
                    self.name,
                    "persist_domain_ai_fallback",
                    "FAILED",
                    f"Snowflake write failed; using in-memory fallback. {domain_ai_err}",
                )
            self._log_success(ctx, step, f"Generated AI summaries for {len(top_repos)} repos")
            return {
                "repo_ai": repo_ai,
                "playbook_json": playbook_json,
                "playbook_md": playbook_md,
            }
        except Exception as exc:
            self._log_failure(ctx, step, exc)
            return {"repo_ai": {}, "playbook_json": {}, "playbook_md": ""}


class CoordinatorAgent(BaseOpsAgent):
    name = "CoordinatorAgent"
    objective = "Coordinate run lifecycle, traceability, and optional Composio actions."

    def __init__(self, logger: RunLogger):
        super().__init__(logger)

    def finalize(self, ctx: AgentContext, result_summary: str) -> None:
        self.logger.log(ctx.run_id, self.name, "finalize", "SUCCESS", result_summary)

    def create_notion(self, ctx: AgentContext, composio_client: Any, playbook_md: str, dry_run: bool = True) -> Dict[str, Any]:
        step = "create_notion_playbook"
        self._log_start(ctx, step)
        try:
            result = create_notion_playbook(
                composio_client=composio_client,
                run_id=ctx.run_id,
                domain=ctx.domain,
                playbook_md=playbook_md,
                dry_run=dry_run,
            )
            status = "SUCCESS" if result.get("ok") else "FAILED"
            self.logger.log(ctx.run_id, self.name, step, status, result.get("message", ""))
            return result
        except Exception as exc:
            self._log_failure(ctx, step, exc)
            return {"ok": False, "message": str(exc), "url": None, "payload": {}}
