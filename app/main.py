"""OpenSourceOps Streamlit app with persona-adaptive UX, caching, and feedback refinement."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.agents import AgentContext, AnalystAgent, CoordinatorAgent, ScoutAgent, StrategistAgent
from src.ai import AIClient
from src.composio_integration import get_composio_client
from src.github_client import GitHubClient
from src.repo_chat import RepoChatService
from src.run_logger import RunLogger
from src.snowflake_client import SnowflakeClient
from src.store import LocalStore, SnowflakeStore


load_dotenv()
st.set_page_config(page_title="OpenSourceOps", page_icon="OPS", layout="wide")


@st.cache_resource
def init_clients():
    sf = SnowflakeClient()
    logger = RunLogger(sf)
    gh = GitHubClient()
    ai = AIClient(sf, logger)
    chat = RepoChatService(SnowflakeStore(sf, fallback=LocalStore()))
    return sf, logger, gh, ai, chat


def stable_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_cache_key(persona: str, domain: str, intent: str, constraints: Dict[str, Any]) -> str:
    return f"{persona.lower()}::{domain.lower().strip()}::{stable_hash(intent)}::{stable_hash(constraints)}"


def _tokenize_query(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9_\\-]+", (text or "").lower()) if len(t) > 2]


def retrieve_evidence(chunks: List[Dict[str, Any]], query: str, mode: str, k: int = 6) -> List[Dict[str, Any]]:
    terms = _tokenize_query(query)
    scored = []
    for ch in chunks:
        text = (ch.get("text") or "").lower()
        path = (ch.get("path") or "").lower()
        ctype = (ch.get("type") or "").upper()
        score = sum(1 for t in terms if t in text or t in path)
        if mode == "Q&A (grounded)" and ("implemented" in query.lower() or "where" in query.lower()):
            if ctype == "FILE":
                score += 2
        if "breaking" in query.lower() and ctype == "RELEASE":
            score += 3
        if any(s in query.lower() for s in ["oauth", "saml", "kafka", "postgres"]):
            score += 2 if any(s in text for s in ["oauth", "saml", "kafka", "postgres"]) else 0
        if score > 0:
            scored.append((score, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ch for _, ch in scored[:k]]


def grounded_answer(repo: str, query: str, mode: str, evidence: List[Dict[str, Any]]) -> str:
    if not evidence:
        return (
            f"No grounded evidence found in indexed sources for `{query}` in `{repo}`. "
            "Try a narrower question or rebuild index."
        )

    cites = [f"- [{e.get('type')}::{e.get('path')}]({e.get('url')})" for e in evidence if e.get("url")]
    snippets = []
    for e in evidence[:3]:
        raw = (e.get("text") or "").strip().replace("\n", " ")
        snippets.append(f"- {raw[:220]}...")

    if mode == "Scenario brainstorm":
        assumptions = [
            "- Assumption: current infrastructure can run stateful workers and persistent storage.",
            "- Assumption: you need retry/compensation semantics and long-running workflow coordination.",
        ]
        return "\n".join(
            [
                "### Scenario Brainstorm (Grounded)",
                "Potential adoption scenarios:",
                "- Event-driven orchestration across services with durable retries.",
                "- Human-in-the-loop workflow steps with timeout/recovery policies.",
                "- Cross-system process automation with idempotent task handlers.",
                "",
                "Assumptions:",
                *assumptions,
                "",
                "Evidence snippets:",
                *snippets,
                "",
                "Citations:",
                *cites,
            ]
        )
    if mode == "Risk & integration review":
        return "\n".join(
            [
                "### Risk & Integration Review (Grounded)",
                "| Risk | Mitigation |",
                "|---|---|",
                "| Operational complexity | Start with one bounded workflow and add observability first |",
                "| Breaking changes in upgrades | Pin versions and run release-note based upgrade tests |",
                "| Integration mismatch | Validate SDK/transport compatibility in PoC |",
                "",
                "Integration steps:",
                "1. Identify workflow boundaries and failure/retry points.",
                "2. Implement one end-to-end PoC with integration tests.",
                "3. Add monitoring/alerts around workflow latency and failures.",
                "",
                "Grounded citations:",
                *cites,
            ]
        )
    return "\n".join(
        [
            "### Grounded Answer",
            "Most relevant evidence found:",
            *snippets,
            "",
            "Citations:",
            *cites,
        ]
    )


def repo_df(scored_repos: List[Dict[str, Any]], repo_ai: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in scored_repos:
        ai_item = repo_ai.get(r.get("full_name"), {})
        rows.append(
            {
                "repo": r.get("full_name"),
                "stars": r.get("stargazers_count", 0),
                "forks": r.get("forks_count", 0),
                "health": r.get("health_score", 0),
                "risk": r.get("risk_score", 0),
                "intent_match": r.get("intent_match_score", 0),
                "final": r.get("final_score", 0),
                "maturity": ai_item.get("maturity", "n/a"),
                "summary": ai_item.get("summary", ""),
                "url": r.get("html_url"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["final"], ascending=False)
    return df


def opportunity_df(opportunities: List[Dict[str, Any]]) -> pd.DataFrame:
    issue_first = [o for o in opportunities if (o.get("item_type") or "").upper() == "ISSUE"]
    source = issue_first if issue_first else opportunities
    rows = []
    for o in source:
        rows.append(
            {
                "repo": o.get("repo_full_name"),
                "type": o.get("item_type"),
                "title": o.get("title"),
                "difficulty": o.get("difficulty"),
                "engagement": o.get("engagement_score", 0),
                "relevance": o.get("relevance_score", 0),
                "scope": o.get("scope_score", 0),
                "final": o.get("final_score", 0),
                "why": o.get("why", ""),
                "next_action": o.get("suggested_next_action", ""),
                "url": o.get("url"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["final", "engagement"], ascending=False)
    return df


def maybe_log_insert_fallback(sf: SnowflakeClient, logger: RunLogger, run_id: str, agent: str):
    err = sf.consume_last_insert_error()
    if err:
        logger.log(run_id, agent, "snowflake_fallback", "INFO", f"Snowflake write fallback used: {err}")


def run_pipeline(
    persona: str,
    domain: str,
    intent: str,
    constraints: Dict[str, Any],
    force_refresh: bool,
    feedback_section: str | None = None,
    feedback_text: str | None = None,
    previous_result: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    sf, logger, gh, ai, _ = init_clients()
    run_id = RunLogger.new_run_id(prefix="ossops")
    cache_key = build_cache_key(persona, domain, intent, constraints)

    ctx = AgentContext(
        run_id=run_id,
        persona=persona,
        domain=domain,
        intent=intent,
        constraints=constraints,
        cache_key=cache_key,
        force_refresh=force_refresh,
        feedback_section=feedback_section,
        feedback_text=feedback_text,
        previous_run_id=(previous_result or {}).get("run_id"),
    )

    coordinator = CoordinatorAgent(logger, sf)
    scout = ScoutAgent(logger, gh, sf, ai)
    analyst = AnalystAgent(logger, sf)
    strategist = StrategistAgent(logger, sf, ai)

    logger.log(run_id, "CoordinatorAgent", "pipeline_start", "STARTED", f"persona={persona}, domain={domain}")
    sf.save_run_context(
        run_id=run_id,
        persona=persona,
        domain=domain,
        intent=intent,
        constraints=constraints,
        cache_key=cache_key,
        optimizer={},
        previous_run_id=(previous_result or {}).get("run_id"),
    )
    maybe_log_insert_fallback(sf, logger, run_id, "CoordinatorAgent")

    if feedback_text:
        sf.save_feedback(run_id, feedback_section or "general", feedback_text)
        maybe_log_insert_fallback(sf, logger, run_id, "CoordinatorAgent")

    redo_search = force_refresh or coordinator.should_redo_search(feedback_section, feedback_text)

    if not force_refresh and not feedback_text:
        cached_run_id = sf.fetch_recent_run_id_by_cache(cache_key, persona, max_age_hours=int(os.getenv("CACHE_TTL_HOURS", "6")))
        if cached_run_id:
            bundle = sf.load_cached_bundle(cached_run_id)
            logger.log(run_id, "CoordinatorAgent", "cache_hit", "SUCCESS", f"Loaded cached run {cached_run_id}")
            return {
                "run_id": run_id,
                "cache_key": cache_key,
                "persona": persona,
                "domain": domain,
                "intent": intent,
                "constraints": constraints,
                "optimizer": bundle.get("optimizer", {}),
                "repos": bundle.get("repos", []),
                "scored_repos": bundle.get("scored_repos", []),
                "repo_ai": bundle.get("repo_ai", {}),
                "playbook_json": bundle.get("playbook_json", {}),
                "playbook_md": bundle.get("playbook_md", ""),
                "opportunities": bundle.get("opportunities", []),
                "discussions_available": True,
                "changes_justification": "Loaded from 6-hour cache; no recomputation required.",
                "run_log": logger.trace(run_id),
                "coordinator": coordinator,
                "from_cache": True,
            }

    try:
        if not redo_search and previous_result:
            logger.log(run_id, "CoordinatorAgent", "ai_only_regeneration", "INFO", "Reusing cached candidates for AI-only refinement")
            optimizer = previous_result.get("optimizer", {})
            repos = previous_result.get("repos", [])
            scored_repos = previous_result.get("scored_repos", [])
            opportunities = previous_result.get("opportunities", [])
            discussions_available = previous_result.get("discussions_available", True)
        else:
            scout_out = scout.run(ctx)
            maybe_log_insert_fallback(sf, logger, run_id, "ScoutAgent")
            optimizer = scout_out.get("optimizer", {})
            repos = scout_out.get("repos", [])
            discussions_available = scout_out.get("discussions_available", True)

            analyst_out = analyst.run(
                ctx,
                repos=repos,
                issues=scout_out.get("issues", []),
                discussions=scout_out.get("discussions", []),
                optimizer=optimizer,
            )
            maybe_log_insert_fallback(sf, logger, run_id, "AnalystAgent")
            scored_repos = analyst_out.get("scored_repos", [])
            opportunities = analyst_out.get("opportunities", [])

        # Update context with optimizer output after discovery.
        sf.save_run_context(
            run_id=run_id,
            persona=persona,
            domain=domain,
            intent=intent,
            constraints=constraints,
            cache_key=cache_key,
            optimizer=optimizer,
            previous_run_id=(previous_result or {}).get("run_id"),
        )
        maybe_log_insert_fallback(sf, logger, run_id, "CoordinatorAgent")

        strategist_out = strategist.run(ctx, scored_repos=scored_repos, opportunities=opportunities)
        maybe_log_insert_fallback(sf, logger, run_id, "StrategistAgent")

        current = {
            "scored_repos": scored_repos,
            "playbook_json": strategist_out.get("playbook_json", {}),
        }
        changes = coordinator.build_changes_justification(previous_result, current)
        if feedback_text:
            feedback_l = feedback_text.lower()
            mentioned = []
            if "temporal" in feedback_l:
                mentioned.append("temporalio/temporal")
            if "cadence" in feedback_l:
                mentioned.append("cadence-workflow/cadence")
            if mentioned:
                suggested = [r.get("full_name") for r in scored_repos[:15]]
                present = [m for m in mentioned if m in suggested]
                missing = [m for m in mentioned if m not in suggested]
                if present and not missing:
                    changes += f" Feedback acknowledged: {', '.join(present)} matched intent and was included/reranked."
                elif present and missing:
                    changes += (
                        f" Feedback acknowledged: included {', '.join(present)}. "
                        f"Not promoted/added for {', '.join(missing)} due to lower relevance to current constraints."
                    )
                else:
                    changes += (
                        f" Feedback reviewed: {', '.join(mentioned)} did not meet current ranking filters "
                        f"(language/quality/intent signals), so recommendations remained unchanged."
                    )
        coordinator.finalize(ctx, f"Run complete. repos={len(scored_repos)} opportunities={len(opportunities)}")

        return {
            "run_id": run_id,
            "cache_key": cache_key,
            "persona": persona,
            "domain": domain,
            "intent": intent,
            "constraints": constraints,
            "optimizer": optimizer,
            "repos": repos,
            "scored_repos": scored_repos,
            "repo_ai": strategist_out.get("repo_ai", {}),
            "playbook_json": strategist_out.get("playbook_json", {}),
            "playbook_md": strategist_out.get("playbook_md", ""),
            "opportunities": opportunities,
            "discussions_available": discussions_available,
            "changes_justification": changes,
            "run_log": logger.trace(run_id),
            "coordinator": coordinator,
            "from_cache": False,
        }
    except Exception as exc:
        logger.log(run_id, "CoordinatorAgent", "pipeline_error", "FAILED", str(exc))
        return {
            "run_id": run_id,
            "cache_key": cache_key,
            "persona": persona,
            "domain": domain,
            "intent": intent,
            "constraints": constraints,
            "optimizer": {},
            "repos": [],
            "scored_repos": [],
            "repo_ai": {},
            "playbook_json": {},
            "playbook_md": "",
            "opportunities": [],
            "discussions_available": False,
            "changes_justification": f"Run failed: {exc}",
            "run_log": logger.trace(run_id),
            "coordinator": coordinator,
            "from_cache": False,
        }


def render_trace(logs: List[Dict[str, Any]]):
    st.subheader("Run Log")
    if not logs:
        st.info("No run trace available.")
        return
    for line in logs:
        st.markdown(
            f"`{line.get('ts')}` **{line.get('agent')}** -> `{line.get('step')}` [{line.get('status')}]  \\\n{line.get('detail')}"
        )


def render_refine_control(section_key: str, label: str):
    text_key = f"feedback_text_{section_key}"
    st.caption(f"ℹ️ Suggest improvements for: {label}")
    st.text_area("What should change?", key=text_key, height=80, placeholder="Type your refinement request...")
    if st.button("Save feedback", key=f"save_{section_key}"):
        text = st.session_state.get(text_key, "").strip()
        if not text:
            st.warning("Please enter feedback text before saving.")
            return
        st.session_state["pending_feedback"] = {
            "section": section_key,
            "text": text,
        }
        st.session_state.setdefault("feedback_history", []).append(st.session_state["pending_feedback"])
        st.success("Feedback captured.")
        st.rerun()


def notion_publish_ui(result: Dict[str, Any]):
    st.subheader("Create Notion Playbook")
    st.caption("Preview content, confirm, then execute.")

    composio_client = get_composio_client()
    dry_run = composio_client is None
    confirm_key = f"confirm_notion_{result['run_id']}"

    if st.button("Create Notion Playbook", type="primary"):
        st.session_state["show_notion_preview"] = True

    def _preview_body():
        st.markdown("### Markdown Preview")
        st.code(result.get("playbook_md", ""), language="markdown")
        st.checkbox("I confirm I want to send this to Notion.", key=confirm_key)
        if dry_run:
            st.warning("Composio/Notion is not configured. Execution will run in dry-run mode.")

        if st.button("Confirm and Execute"):
            if not st.session_state.get(confirm_key):
                st.error("Please confirm before executing.")
                return
            ctx = AgentContext(
                run_id=result["run_id"],
                persona=result["persona"],
                domain=result["domain"],
                intent=result["intent"],
                constraints=result.get("constraints", {}),
                cache_key=result["cache_key"],
            )
            response = result["coordinator"].create_notion(
                ctx=ctx,
                composio_client=composio_client,
                playbook_md=result.get("playbook_md", ""),
                dry_run=dry_run,
            )
            st.session_state["notion_response"] = response
            st.session_state["show_notion_preview"] = False
            st.rerun()

    if st.session_state.get("show_notion_preview"):
        if hasattr(st, "dialog"):
            @st.dialog("Notion Export Preview")
            def show_dialog():
                _preview_body()

            show_dialog()
        else:
            with st.expander("Notion Export Preview", expanded=True):
                _preview_body()

    if st.session_state.get("notion_response"):
        resp = st.session_state["notion_response"]
        if resp.get("ok"):
            st.success(resp.get("message"))
            if resp.get("url"):
                st.markdown(f"Notion page URL: {resp['url']}")
        else:
            st.error(resp.get("message"))
        st.json(resp.get("payload", {}))


def main():
    st.title("OpenSourceOps")
    persona = st.radio("Persona", ["Contributor", "Adopter"], horizontal=True)

    with st.sidebar:
        st.header("Inputs")
        domain = st.text_input("Domain", value="RAG evaluation")
        intent = st.text_area(
            "Intent message",
            value=(
                "I want to build a reliable RAG evaluation stack for internal benchmarking and governance."
                if persona == "Adopter"
                else "I want to contribute to RAG evaluation tooling and build credibility in evaluation frameworks."
            ),
            height=120,
        )

        st.subheader("Constraints")
        language = st.text_input("Primary language (optional)", value="")
        maturity_pref = st.selectbox("Maturity preference", ["Stable only", "OK emerging"], index=0)

        force_refresh = st.toggle("Force refresh", value=False)
        run_clicked = st.button("Run Analysis", type="primary")

        st.markdown("---")
        pending = st.session_state.get("pending_feedback")
        if pending and pending.get("text"):
            st.info(f"Pending feedback ({pending.get('section')}): {pending.get('text')[:80]}")
            if st.button("Regenerate with feedback"):
                constraints = {
                    "language": language.strip() or None,
                    "maturity_preference": maturity_pref,
                }
                result = run_pipeline(
                    persona=persona,
                    domain=domain,
                    intent=intent,
                    constraints=constraints,
                    force_refresh=force_refresh,
                    feedback_section=pending.get("section"),
                    feedback_text=pending.get("text"),
                    previous_result=st.session_state.get("result"),
                )
                st.session_state["result"] = result
                st.session_state["pending_feedback"] = None
                st.rerun()

        st.markdown("---")
        st.caption("Integrations")
        st.write(f"GitHub: {'configured' if bool(os.getenv('GITHUB_TOKEN')) else 'dry-run'}")
        st.write(f"Snowflake: {'configured' if bool(os.getenv('SNOWFLAKE_ACCOUNT') and os.getenv('SNOWFLAKE_WAREHOUSE')) else 'dry-run'}")
        st.write(f"Composio: {'configured' if bool(os.getenv('COMPOSIO_API_KEY')) else 'dry-run'}")

    constraints = {
        "language": language.strip() or None,
        "maturity_preference": maturity_pref,
    }

    if run_clicked:
        with st.status("Running OpenSourceOps pipeline...", expanded=True) as status:
            status.write("ScoutAgent: query optimization and GitHub discovery")
            status.write("AnalystAgent: scoring repositories and opportunities")
            status.write("StrategistAgent: generating persona-specific outputs")
            status.write("CoordinatorAgent: caching, refinement context, and trace logging")
            result = run_pipeline(
                persona=persona,
                domain=domain,
                intent=intent,
                constraints=constraints,
                force_refresh=force_refresh,
                previous_result=st.session_state.get("result"),
            )
            status.update(label="Run complete", state="complete")
        st.session_state["result"] = result

    result = st.session_state.get("result")
    if not result:
        st.info("Provide persona, domain, and intent, then click Run Analysis.")
        return

    st.markdown("### Changes & Justification")
    st.info(result.get("changes_justification", "No changes summary available."))

    repos_tab, *other_tabs = (
        st.tabs(["Repositories", "Opportunities", "Contribution Plan", "Run Log"])
        if persona == "Contributor"
        else st.tabs(["Repositories", "Adoption Playbook", "Run Log"])
    )

    with repos_tab:
        st.subheader("Ranked Repositories")
        rdf = repo_df(result.get("scored_repos", []), result.get("repo_ai", {}))
        if rdf.empty:
            st.warning("No repositories discovered. Check GitHub token or refine intent/constraints.")
        else:
            top_repo_name = rdf.iloc[0]["repo"]
            top_repo_ai = (result.get("repo_ai", {}) or {}).get(top_repo_name, {})
            st.success(f"Top Recommended Repository: {top_repo_name}")
            if top_repo_ai.get("recommended_use"):
                st.markdown(f"**How to use it:** {top_repo_ai.get('recommended_use')}")
            if top_repo_ai.get("engagement_plan"):
                st.markdown(f"**How to engage:** {top_repo_ai.get('engagement_plan')}")
            st.dataframe(rdf, use_container_width=True)
            for _, row in rdf.head(10).iterrows():
                with st.expander(f"{row['repo']} - AI Summary"):
                    st.write(row.get("summary", ""))
                    st.markdown(f"[Repository]({row.get('url')})")
        render_refine_control("repo_recommendation", "Repository Recommendation")

    if persona == "Contributor":
        opportunities_tab, plan_tab, log_tab = other_tabs

        with opportunities_tab:
            st.subheader("Top Contribution Opportunities")
            if not result.get("discussions_available", True):
                st.caption("Discussions are unavailable for current token/repositories; showing issues-based opportunities.")
            odf = opportunity_df(result.get("opportunities", []))
            if odf.empty:
                st.warning("No opportunities discovered. Try broader intent or force refresh.")
            else:
                st.dataframe(odf, use_container_width=True)
            render_refine_control("opportunities_table", "Opportunities")

        with plan_tab:
            st.subheader("Contribution Plan")
            st.markdown(result.get("playbook_md", "No contribution plan generated."))
            render_refine_control("weekly_plan", "Weekly Plan")

        with log_tab:
            render_trace(result.get("run_log", []))
    else:
        playbook_tab, log_tab = other_tabs

        with playbook_tab:
            st.subheader("Adoption Playbook")
            st.markdown(result.get("playbook_md", "No playbook generated."))
            render_refine_control("risk_analysis", "Risk Analysis")
            render_refine_control("plan_30", "30-Day Plan")
            render_refine_control("plan_60", "60-Day Plan")
            render_refine_control("plan_90", "90-Day Plan")
            notion_publish_ui(result)
            st.markdown("---")
            st.subheader("Repo Deep-Dive Chat")
            st.caption(
                "Grounded Answer Contract: every response includes direct answer, citations, confidence, and next checks if evidence is weak."
            )

            scored = result.get("scored_repos", [])
            if not scored:
                st.info("Run analysis first to enable deep-dive chat.")
            else:
                sf, logger, _, _, chat_service = init_clients()
                repo_candidates = [
                    {"full_name": r.get("full_name"), "url": r.get("html_url")}
                    for r in scored[:10]
                    if r.get("full_name") and r.get("html_url")
                ]
                if not repo_candidates:
                    st.info("No repository URLs available for deep-dive indexing.")
                else:
                    labels = [f"{r['full_name']} ({r['url']})" for r in repo_candidates]
                    selected_label = st.selectbox("Repository to deep-dive", options=labels, index=0)
                    selected = repo_candidates[labels.index(selected_label)]
                    selected_repo = selected["full_name"]
                    selected_repo_url = selected["url"]
                    ref = st.text_input("Optional commit/tag/branch (default latest)", value="", key=f"repo_ref_{selected_repo}")

                    mode_label = st.radio(
                        "Chat mode",
                        ["Q&A (grounded)", "Scenario brainstorm", "Risk & integration review"],
                        horizontal=True,
                    )
                    mode_map = {
                        "Q&A (grounded)": "grounded_qa",
                        "Scenario brainstorm": "scenario_brainstorm",
                        "Risk & integration review": "risk_review",
                    }
                    mode = mode_map[mode_label]

                    if st.button("Build/Refresh Repo Index", key=f"build_index_{selected_repo}"):
                        with st.spinner(f"Indexing {selected_repo}..."):
                            try:
                                idx = chat_service.build_index(selected_repo_url, ref.strip() or None)
                                st.session_state.setdefault("repo_index_meta", {})[selected_repo] = idx
                                logger.log(
                                    result["run_id"],
                                    "CoordinatorAgent",
                                    "repo_index",
                                    "SUCCESS",
                                    f"repo={selected_repo} commit={idx.get('commit_sha')} chunks={idx.get('chunk_count')}",
                                )
                                st.success(
                                    f"Indexed {idx.get('chunk_count')} chunks at commit `{idx.get('commit_sha')}`."
                                )
                            except Exception as exc:
                                logger.log(result["run_id"], "CoordinatorAgent", "repo_index", "FAILED", str(exc))
                                st.error(f"Failed to build index: {exc}")

                    idx_meta = st.session_state.get("repo_index_meta", {}).get(selected_repo)
                    if not idx_meta:
                        st.info("Index not built yet. Click 'Build/Refresh Repo Index'.")
                    else:
                        st.caption(
                            f"Indexed repo_id=`{idx_meta.get('repo_id')}` commit_sha=`{idx_meta.get('commit_sha')}` chunks={idx_meta.get('chunk_count')}"
                        )
                        playbook = result.get("playbook_json", {}) or {}
                        suggested_questions = [
                            f"Where in the code is retry logic implemented in {selected_repo}?",
                            f"How do I integrate/configure {selected_repo} for production?",
                            f"Does {selected_repo} support OAuth/SAML/Kafka/Postgres?",
                            f"What are the top integration risks for adopting {selected_repo}?",
                            f"What are breaking changes between recent versions of {selected_repo}?",
                        ]
                        if playbook.get("integration_plan"):
                            suggested_questions.append(
                                f"Can you turn the integration plan into concrete steps for {selected_repo}?"
                            )
                        if playbook.get("risk_register"):
                            suggested_questions.append(
                                f"Map these risks to owners and mitigations for {selected_repo} adoption."
                            )
                        suggested_questions = suggested_questions[:6]

                        st.markdown("**Suggested questions**")
                        cols = st.columns(2)
                        for i, sq in enumerate(suggested_questions):
                            if cols[i % 2].button(sq, key=f"sq_{selected_repo}_{i}"):
                                st.session_state["chat_prefill_question"] = sq

                        repo_id = idx_meta.get("repo_id")
                        chat_key = f"chat_{repo_id}_{mode}"
                        messages = st.session_state.setdefault(chat_key, [])
                        for m in messages:
                            with st.chat_message(m["role"]):
                                st.markdown(m["content"])

                        queued = st.session_state.pop("chat_prefill_question", None)
                        user_q = queued or st.chat_input(
                            "Ask: Where is X implemented? How to integrate Y? Does it support Z? Risks? Breaking changes?"
                        )
                        if user_q:
                            messages.append({"role": "user", "content": user_q})
                            response = chat_service.chat(repo_id=repo_id, question=user_q, mode=mode)
                            answer = response.get("answer", "")
                            confidence = response.get("confidence", "Low")
                            qtype = response.get("question_type", "B")
                            cites = response.get("citations", [])
                            next_checks = response.get("next_checks", [])

                            citations_md = []
                            for c in cites:
                                line_span = f"L{c.get('start_line')}-L{c.get('end_line')}"
                                citations_md.append(
                                    f"- [{c.get('file_path')}:{line_span}]({c.get('url')})"
                                )
                            cites_block = "\n".join(citations_md) if citations_md else "- No citations"
                            checks_block = "\n".join([f"- {x}" for x in next_checks]) if next_checks else "- None"
                            assistant_text = (
                                f"{answer}\n\n"
                                f"**Question Type:** {qtype}\n"
                                f"**Confidence:** {confidence}\n\n"
                                f"**Citations**\n{cites_block}\n\n"
                                f"**Next checks**\n{checks_block}"
                            )
                            messages.append({"role": "assistant", "content": assistant_text})
                            st.session_state[chat_key] = messages
                            st.rerun()

        with log_tab:
            render_trace(result.get("run_log", []))


if __name__ == "__main__":
    main()
