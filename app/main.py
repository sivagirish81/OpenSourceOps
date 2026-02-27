"""OpenSourceOps Streamlit app: contributor and enterprise OSS intelligence workflows."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Ensure src package is importable when running `streamlit run app/main.py`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.agents import AgentContext, AnalystAgent, CoordinatorAgent, ScoutAgent, StrategistAgent
from src.ai import AIClient
from src.composio_integration import get_composio_client
from src.github_client import GitHubClient
from src.run_logger import RunLogger
from src.snowflake_client import SnowflakeClient


load_dotenv()
st.set_page_config(page_title="OpenSourceOps", page_icon="OPS", layout="wide")


@st.cache_resource
def init_clients():
    sf = SnowflakeClient()
    logger = RunLogger(sf)
    gh = GitHubClient()
    ai = AIClient(sf, logger)
    return sf, logger, gh, ai


def to_repo_df(scored_repos, repo_ai):
    rows = []
    for repo in scored_repos:
        ai_item = repo_ai.get(repo.get("full_name"), {})
        rows.append(
            {
                "repo": repo.get("full_name"),
                "language": repo.get("language") or repo.get("repo_language"),
                "stars": repo.get("stargazers_count", 0),
                "forks": repo.get("forks_count", 0),
                "health": repo.get("health_score", 0),
                "risk": repo.get("risk_score", 0),
                "maturity": ai_item.get("maturity", "n/a"),
                "summary": ai_item.get("summary", ""),
                "url": repo.get("html_url"),
            }
        )
    return pd.DataFrame(rows)


def to_issue_plan_df(issues, issue_scores):
    score_map = {item["issue_id"]: item for item in issue_scores}
    rows = []
    for issue in issues:
        metrics = score_map.get(issue.get("id"), {})
        rows.append(
            {
                "repo": issue.get("full_name"),
                "title": issue.get("title"),
                "labels": ", ".join(issue.get("labels", [])),
                "impact": metrics.get("impact_score", 0),
                "difficulty": metrics.get("difficulty_score", 0),
                "reputation": metrics.get("reputation_score", 0),
                "url": issue.get("html_url"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["reputation", "impact"], ascending=False)
    return df


def run_pipeline(domain: str, language: str | None, mode: str, force_refresh: bool):
    sf, logger, gh, ai = init_clients()
    run_id = RunLogger.new_run_id(prefix="ossops")
    ctx = AgentContext(run_id=run_id, domain=domain, language=language, mode=mode)

    logger.log(run_id, "CoordinatorAgent", "pipeline_start", "STARTED", f"Mode={mode}; domain={domain}; language={language or 'any'}")

    scout = ScoutAgent(logger, gh, sf)
    analyst = AnalystAgent(logger, sf)
    strategist = StrategistAgent(logger, sf, ai)
    coordinator = CoordinatorAgent(logger)

    try:
        scout_out = scout.run(ctx)
        analyst_out = analyst.run(ctx, scout_out.get("repos", []), scout_out.get("issues", []))
        strategist_out = strategist.run(ctx, analyst_out.get("scored_repos", []))

        coordinator.finalize(
            ctx,
            result_summary=(
                f"Run complete with {len(analyst_out.get('scored_repos', []))} repos, "
                f"{len(scout_out.get('issues', []))} issues"
            ),
        )
    except Exception as exc:
        logger.log(run_id, "CoordinatorAgent", "pipeline_error", "FAILED", str(exc))
        scout_out = {"query_pack": None, "repos": [], "issues": []}
        analyst_out = {"scored_repos": [], "issue_scores": []}
        strategist_out = {"repo_ai": {}, "playbook_json": {}, "playbook_md": ""}

    return {
        "run_id": run_id,
        "domain": domain,
        "language": language,
        "mode": mode,
        "query_pack": scout_out.get("query_pack"),
        "repos": scout_out.get("repos", []),
        "issues": scout_out.get("issues", []),
        "scored_repos": analyst_out.get("scored_repos", []),
        "issue_scores": analyst_out.get("issue_scores", []),
        "repo_ai": strategist_out.get("repo_ai", {}),
        "playbook_json": strategist_out.get("playbook_json", {}),
        "playbook_md": strategist_out.get("playbook_md", ""),
        "run_log": logger.trace(run_id),
        "coordinator": coordinator,
    }


def render_trace(logs):
    st.subheader("Run Trace")
    if not logs:
        st.info("No run trace yet.")
        return
    for line in logs:
        st.markdown(
            f"`{line.get('ts')}` **{line.get('agent')}** -> `{line.get('step')}` [{line.get('status')}]  \\\n{line.get('detail')}"
        )


def notion_publish_ui(result):
    if result.get("mode") != "Enterprise":
        st.info("Notion playbook creation is only active in Enterprise mode.")
        return

    st.subheader("Create Notion Playbook")
    st.caption("This action is opt-in. Preview payload first, then confirm to execute.")

    composio_client = get_composio_client()
    is_dry = composio_client is None
    confirm_key = f"confirm_notion_{result['run_id']}"

    if st.button("Create Notion Playbook", type="primary"):
        st.session_state["show_notion_preview"] = True

    def _preview_body():
        st.markdown("### Markdown Preview")
        st.code(result.get("playbook_md", ""), language="markdown")
        st.checkbox("I confirm I want to send this to Notion.", key=confirm_key)
        if is_dry:
            st.warning("COMPOSIO_API_KEY missing or SDK unavailable. This will run in dry-run mode.")
        if st.button("Confirm and Execute"):
            coordinator = result["coordinator"]
            ctx = AgentContext(
                run_id=result["run_id"],
                domain=result["domain"],
                language=result["language"],
                mode=result["mode"],
            )
            if not st.session_state.get(confirm_key):
                st.error("Please check the confirmation box before executing.")
                return
            response = coordinator.create_notion(
                ctx=ctx,
                composio_client=composio_client,
                playbook_md=result.get("playbook_md", ""),
                dry_run=is_dry,
            )
            st.session_state["notion_response"] = response
            st.session_state["show_notion_preview"] = False
            st.rerun()

    if st.session_state.get("show_notion_preview"):
        if hasattr(st, "dialog"):
            @st.dialog("Notion Playbook Preview")
            def show_preview_dialog():
                _preview_body()

            show_preview_dialog()
        else:
            with st.expander("Notion Playbook Preview", expanded=True):
                _preview_body()

    if st.session_state.get("notion_response"):
        resp = st.session_state["notion_response"]
        if resp.get("ok"):
            st.success(resp.get("message"))
            if resp.get("url"):
                st.markdown(f"Notion page: {resp['url']}")
        else:
            st.error(resp.get("message"))
        st.json(resp.get("payload", {}))


def main():
    st.title("OpenSourceOps")
    st.caption("Autonomous OSINT pipeline for contributors and enterprise adopters")

    with st.sidebar:
        st.header("Inputs")
        domain = st.text_input("Domain", value="RAG evaluation")
        language = st.text_input("Language (optional)", value="")
        mode = st.selectbox("Mode", options=["Contributor", "Enterprise"], index=0)
        force_refresh = st.toggle("Force refresh", value=False)
        run_clicked = st.button("Run Analysis", type="primary")

        st.markdown("---")
        st.caption("Integrations")
        st.write(f"GitHub token: {'configured' if bool(os.getenv('GITHUB_TOKEN')) else 'missing'}")
        st.write(f"Snowflake: {'configured' if bool(os.getenv('SNOWFLAKE_ACCOUNT')) else 'dry-run'}")
        st.write(f"Composio: {'configured' if bool(os.getenv('COMPOSIO_API_KEY')) else 'dry-run'}")

    cache_key = f"{domain.lower()}::{(language.strip() or '').lower()}::{mode}"
    now = datetime.now(timezone.utc)
    app_cache = st.session_state.setdefault("analysis_cache", {})

    if run_clicked:
        used_cache = False
        if not force_refresh and cache_key in app_cache:
            cached_item = app_cache[cache_key]
            cached_ts = datetime.fromisoformat(cached_item["ts"])
            if now - cached_ts <= timedelta(hours=6):
                st.session_state["result"] = cached_item["result"]
                st.session_state["cache_notice"] = "Using cached run result from the last 6 hours."
                used_cache = True
            else:
                app_cache.pop(cache_key, None)

        if force_refresh or not used_cache:
            with st.status("Running OpenSourceOps pipeline...", expanded=True) as status:
                status.write("ScoutAgent: discovering repositories and issues")
                result = run_pipeline(
                    domain=domain,
                    language=language.strip() or None,
                    mode=mode,
                    force_refresh=force_refresh,
                )
                status.write("AnalystAgent: scoring repository and issue candidates")
                status.write("StrategistAgent: generating AI summaries and playbook")
                status.write("CoordinatorAgent: finalizing run log and artifacts")
                status.update(label="Pipeline complete", state="complete")
            st.session_state["result"] = result
            app_cache[cache_key] = {"ts": now.isoformat(), "result": result}

    result = st.session_state.get("result")
    if not result:
        st.info("Provide a domain and click Run Analysis to start.")
        return
    if st.session_state.get("cache_notice"):
        st.info(st.session_state.pop("cache_notice"))

    tabs = st.tabs(["Repositories", "Adoption Playbook", "Contribution Plan", "Run Log"])

    with tabs[0]:
        st.subheader("Ranked Repositories")
        repo_df = to_repo_df(result.get("scored_repos", []), result.get("repo_ai", {}))
        if repo_df.empty:
            st.warning("No repositories discovered. Check GITHUB_TOKEN or refine domain.")
        else:
            st.dataframe(repo_df, use_container_width=True)

    with tabs[1]:
        st.subheader("Adoption Playbook")
        playbook_md = result.get("playbook_md", "")
        if playbook_md:
            st.markdown(playbook_md)
        else:
            st.info("Playbook unavailable for this run.")
        notion_publish_ui(result)

    with tabs[2]:
        st.subheader("Contribution Plan")
        if result.get("mode") != "Contributor":
            st.info("Contribution Plan is only generated for Contributor mode.")
        else:
            issues_df = to_issue_plan_df(result.get("issues", []), result.get("issue_scores", []))
            if issues_df.empty:
                st.warning("No labeled issues found in top repositories.")
            else:
                st.dataframe(issues_df, use_container_width=True)
                st.markdown("### Weekly Plan")
                top_rows = issues_df.head(5).to_dict(orient="records")
                for idx, row in enumerate(top_rows, start=1):
                    st.markdown(f"{idx}. Contribute to **{row['repo']}**: [{row['title']}]({row['url']})")

    with tabs[3]:
        render_trace(result.get("run_log", []))


if __name__ == "__main__":
    main()
