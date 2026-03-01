"""OpenSourceOps: OSS Due Diligence AI for companies (Snowflake-backed)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from storage.snowflake_store import SnowflakeStore
from src.async_ingestion import start_async_ingestion_job
from src.due_diligence_pipeline import (
    create_tasks_from_next_steps,
    run_due_diligence,
    run_ingestion,
    run_scouting,
)
from src.github_client import GitHubClient


load_dotenv()
st.set_page_config(page_title="OpenSourceOps", page_icon="OPS", layout="wide")


@st.cache_resource
def init_services():
    store = SnowflakeStore()
    github = GitHubClient()
    if not store.is_configured():
        raise RuntimeError(
            "Snowflake is mandatory for this product. Missing env vars: "
            + ", ".join(store.cfg.missing())
        )
    store.ensure_schema()
    return store, github


def _safe_state(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def _reset_downstream_state():
    """Clear dependent state when company/repo/scout context changes."""
    st.session_state["last_scout"] = None
    st.session_state["selected_repo"] = None
    st.session_state["last_ingestion"] = None
    st.session_state["last_dd"] = None
    st.session_state["ingestion_job_id"] = None


def _profile_label(item: Dict[str, Any]) -> str:
    org = item.get("profile", {}).get("organization", {}).get("name", "Unnamed")
    return f"{org} ({item.get('company_id')})"


def page_onboarding(store: SnowflakeStore):
    st.header("1) Company Onboarding")
    st.caption("Capture industry-standard due diligence context. Saved to Snowflake.")

    with st.form("company_onboarding_form"):
        st.subheader("Organization")
        org_name = st.text_input("Organization name")
        industry = st.text_input("Industry")
        company_size = st.selectbox("Company size", ["1-50", "51-200", "201-1000", "1000+"])
        geography = st.text_input("Geography / regions")
        deployment_model = st.selectbox("Deployment model", ["Cloud", "Hybrid", "On-prem"])
        customer_type = st.selectbox("Customer type", ["B2B", "B2C", "B2B2C", "Public sector"])

        st.subheader("Technical Environment")
        cloud_provider = st.multiselect("Cloud provider", ["AWS", "GCP", "Azure", "On-prem"])
        orchestration = st.text_input("Orchestration", value="Kubernetes")
        databases = st.text_input("Databases (comma-separated)", value="Postgres")
        messaging = st.text_input("Messaging (comma-separated)", value="Kafka")
        identity_provider = st.text_input("Identity provider", value="Okta")
        observability_stack = st.text_input("Observability stack (comma-separated)", value="Prometheus, OpenTelemetry")
        cicd_type = st.text_input("CI/CD type", value="GitHub Actions")
        primary_languages = st.text_input("Primary languages (comma-separated)", value="Python, Go")

        st.subheader("Security & Compliance")
        required_frameworks = st.multiselect(
            "Required frameworks",
            ["SOC2", "HIPAA", "GDPR", "ISO27001", "PCI-DSS"],
            default=["SOC2"],
        )
        audit_logs_required = st.checkbox("Audit logs required", value=True)
        encryption_requirements = st.text_input("Encryption requirements", value="Encryption at rest and in transit")
        sbom_required = st.checkbox("SBOM required", value=True)
        data_residency = st.text_input("Data residency restrictions", value="US only")

        st.subheader("Risk Appetite")
        risk_tolerance = st.selectbox("Risk tolerance", ["Low", "Medium", "High"], index=0)
        willing_to_fork = st.checkbox("Willing to fork OSS", value=False)
        require_commercial_support = st.checkbox("Require commercial support", value=True)
        maturity_preference = st.selectbox("Maturity preference", ["Stable only", "Active", "Emerging ok"], index=0)
        min_maintainer_activity = st.text_input("Minimum maintainer activity expectations", value="At least weekly commits or monthly releases")

        submitted = st.form_submit_button("Save Company Profile")

    if submitted:
        profile = {
            "organization": {
                "name": org_name,
                "industry": industry,
                "company_size": company_size,
                "geography_regions": geography,
                "deployment_model": deployment_model,
                "customer_type": customer_type,
            },
            "technical_environment": {
                "cloud_provider": cloud_provider,
                "orchestration": orchestration,
                "databases": [x.strip() for x in databases.split(",") if x.strip()],
                "messaging": [x.strip() for x in messaging.split(",") if x.strip()],
                "identity_provider": identity_provider,
                "observability_stack": [x.strip() for x in observability_stack.split(",") if x.strip()],
                "cicd_type": cicd_type,
                "primary_languages": [x.strip() for x in primary_languages.split(",") if x.strip()],
            },
            "security_compliance": {
                "required_frameworks": required_frameworks,
                "audit_logs_required": audit_logs_required,
                "encryption_requirements": encryption_requirements,
                "sbom_required": sbom_required,
                "data_residency_restrictions": data_residency,
            },
            "risk_appetite": {
                "risk_tolerance": risk_tolerance,
                "willing_to_fork_oss": willing_to_fork,
                "require_commercial_support": require_commercial_support,
                "maturity_preference": maturity_preference,
                "min_maintainer_activity_expectations": min_maintainer_activity,
            },
        }
        company_id = store.save_company_profile(profile)
        st.success(f"Saved profile to Snowflake. company_id={company_id}")
        st.session_state["selected_company_id"] = company_id

    profiles = store.list_company_profiles()
    if profiles:
        st.subheader("Saved Company Profiles")
        df = pd.DataFrame(
            [
                {
                    "company_id": p["company_id"],
                    "name": p.get("name"),
                    "updated_at": p.get("updated_at"),
                    "industry": p.get("profile", {}).get("organization", {}).get("industry"),
                    "risk_tolerance": p.get("profile", {}).get("risk_appetite", {}).get("risk_tolerance"),
                }
                for p in profiles
            ]
        )
        st.dataframe(df, use_container_width=True)


def _current_company(store: SnowflakeStore) -> tuple[str | None, Dict[str, Any] | None, List[Dict[str, Any]]]:
    profiles = store.list_company_profiles()
    if not profiles:
        return None, None, profiles
    options = {_profile_label(p): p["company_id"] for p in profiles}
    default_id = st.session_state.get("selected_company_id") or profiles[0]["company_id"]
    labels = list(options.keys())
    default_idx = next((i for i, lbl in enumerate(labels) if options[lbl] == default_id), 0)
    selected_label = st.selectbox("Company profile", labels, index=default_idx)
    company_id = options[selected_label]
    prev_company = st.session_state.get("_active_company_id")
    if prev_company and prev_company != company_id:
        _reset_downstream_state()
        st.info("Company profile changed. Cleared scouting/ingestion/due-diligence state.")
    st.session_state["selected_company_id"] = company_id
    st.session_state["_active_company_id"] = company_id
    return company_id, store.get_company_profile(company_id), profiles


def page_scouting(store: SnowflakeStore, github: GitHubClient):
    st.header("2) Scouting Results")
    st.caption("Context-aware Top 3 OSS candidates with explainable scoring.")

    company_id, profile, _ = _current_company(store)
    if not company_id or not profile:
        st.warning("Create a company profile first.")
        return
    if not github.is_configured():
        st.warning("GITHUB_TOKEN is missing. Scouting will return empty results until GitHub is configured.")

    requirements = st.text_area(
        "Requirements",
        value="Need a durable workflow orchestration platform with strong reliability, security controls, and enterprise readiness.",
        height=120,
    )
    if st.button("Run Scouting"):
        with st.spinner("Running scouting..."):
            result = run_scouting(store, github, company_id, profile, requirements)
        # New scouting context invalidates previous selection/ingestion/report.
        st.session_state["selected_repo"] = None
        st.session_state["last_ingestion"] = None
        st.session_state["last_dd"] = None
        st.session_state["ingestion_job_id"] = None
        st.session_state["last_scout"] = result
        st.success(f"Scouting complete. scout_run_id={result['scout_run_id']}")

    scout = st.session_state.get("last_scout")
    if not scout:
        st.info("Run scouting to view Top 3.")
        return
    if scout.get("cortex_used"):
        st.success("Cortex reranking applied.")
    else:
        st.info("Cortex reranking unavailable; deterministic ranking used.")

    top3_raw = scout.get("top3", [])
    top3 = [r for r in top3_raw if isinstance(r, dict)]
    st.subheader("Top 3 Candidates")
    if not top3:
        st.error("No repositories matched from GitHub for this query/profile.")
        st.caption("Try broadening requirements, removing strict language constraints, or verify GITHUB_TOKEN access/rate limits.")
        st.json(
            {
                "candidate_count": scout.get("candidate_count", 0),
                "queries": (scout.get("query_pack") or {}).get("repo_queries", []),
                "cortex_used": scout.get("cortex_used", False),
                "raw_top3_type": str(type(top3_raw)),
                "raw_top3_preview": str(top3_raw)[:1000],
            }
        )
        return
    for i, repo in enumerate(top3, start=1):
        with st.container(border=True):
            st.markdown(f"### {i}. {repo['full_name']}")
            st.markdown(f"[Repository]({repo['html_url']})")
            st.write(repo.get("description") or "")
            sb = repo.get("score_breakdown", {})
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Fit", sb.get("fit"))
            c2.metric("Maturity", sb.get("maturity"))
            c3.metric("Health", sb.get("health"))
            c4.metric("Risk", sb.get("risk"))
            c5.metric("Context", sb.get("context_alignment"))
            st.write("**Pros:**")
            for p in repo.get("pros", []):
                if isinstance(p, dict):
                    st.markdown(f"- {p.get('text')}")
                else:
                    st.markdown(f"- {p}")
            st.write("**Cons:**")
            for c in repo.get("cons", []):
                if isinstance(c, dict):
                    st.markdown(f"- {c.get('text')}")
                else:
                    st.markdown(f"- {c}")
            with st.expander("Pros/Cons citations"):
                st.json({"pros": repo.get("pros", []), "cons": repo.get("cons", [])})
            if repo.get("constraint_warnings"):
                st.warning(" | ".join(repo["constraint_warnings"]))
            if st.button(f"Proceed to Due Diligence: {repo['full_name']}", key=f"proceed_{i}"):
                prev_repo = (st.session_state.get("selected_repo") or {}).get("full_name")
                st.session_state["selected_repo"] = repo
                if prev_repo != repo.get("full_name"):
                    st.session_state["last_ingestion"] = None
                    st.session_state["last_dd"] = None
                    st.session_state["ingestion_job_id"] = None
                st.success("Repository selected. Move to Repo Selection + Ingestion page.")


def page_ingestion(store: SnowflakeStore):
    st.header("3) Repo Selection + Ingestion Progress")
    company_id, profile, _ = _current_company(store)
    scout = st.session_state.get("last_scout")
    repo = st.session_state.get("selected_repo")
    if not (company_id and profile and scout and repo):
        st.info("Run scouting and select one repository first.")
        return

    st.write(f"Selected: **{repo['full_name']}**")
    st.write(f"URL: {repo['html_url']}")
    ref = st.text_input("Optional commit/tag/branch", value="")
    index_profile_label = st.radio("Indexing depth", ["Minimal (fast)", "Standard"], horizontal=True, index=1)
    index_profile = "minimal" if index_profile_label.startswith("Minimal") else "standard"
    if index_profile == "minimal":
        st.caption("Expected indexing time: ~10 to 30 seconds. Indexes only the most critical ~10-20 files plus recent releases/issues/discussions.")
    else:
        st.caption("Expected indexing time: ~30 to 90 seconds. Caps indexing to ~50 key files plus small recent release/issues/discussions context.")
    force_reindex = st.checkbox("Force re-index (ignore existing ingestion)", value=False)
    st.caption("Background ingestion continues while you navigate between tabs.")

    # Prevent stale ingestion from a different repo being reused in UI/report.
    current_ing = st.session_state.get("last_ingestion") or {}
    if current_ing and current_ing.get("repo_full_name") != repo.get("full_name"):
        st.session_state["last_ingestion"] = None
        st.session_state["last_dd"] = None

    existing = store.get_latest_ingestion(company_id=company_id, repo_full_name=repo["full_name"])
    if existing and not force_reindex:
        st.info(
            f"Existing ingestion found: {existing['ingest_run_id']} "
            f"(commit={existing.get('commit_sha')[:12] if existing.get('commit_sha') else 'unknown'}, "
            f"chunks={existing.get('chunks_indexed', 0)}, created_at={existing.get('created_at')}). "
            "Click 'Run Repo Ingestion' to reuse it instantly."
        )

    if st.button("Start Background Ingestion"):
        job = start_async_ingestion_job(
            company_id=company_id,
            scout_run_id=scout["scout_run_id"],
            repo_full_name=repo["full_name"],
            repo_url=repo["html_url"],
            company_profile=profile,
            ref=ref.strip() or None,
            index_profile=index_profile,
        )
        st.session_state["ingestion_job_id"] = job["job_id"]
        st.session_state["last_ingestion"] = {
            "ingest_run_id": job["ingest_run_id"],
            "repo_full_name": repo["full_name"],
            "repo_url": repo["html_url"],
            "reused": False,
            "status": "RUNNING",
        }
        st.session_state["last_dd"] = None
        st.success(f"Background ingestion started. job_id={job['job_id']}")

    job_id = st.session_state.get("ingestion_job_id")
    if job_id:
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        message_placeholder = st.empty()
        chunks_placeholder = st.empty()
        poll_hint = st.empty()
        # Keep this render responsive while still giving live updates.
        max_polls = int(os.getenv("INGEST_UI_MAX_POLLS", "25"))
        poll_interval = float(os.getenv("INGEST_UI_POLL_INTERVAL_SEC", "1.2"))
        for _ in range(max_polls):
            job = store.get_ingestion_job(job_id)
            if not job:
                break
            status_placeholder.write(f"Job status: **{job['status']}**")
            progress_placeholder.progress(max(0, min(100, int(job.get("progress", 0.0) * 100))))
            message_placeholder.caption(job.get("message") or "")
            chunks_so_far = store.count_evidence_chunks(job["ingest_run_id"])
            chunks_placeholder.caption(f"Chunks available so far: {chunks_so_far}")

            if job["status"] == "COMPLETE":
                latest = store.get_latest_ingestion(company_id=company_id, repo_full_name=repo["full_name"])
                if latest:
                    st.session_state["last_ingestion"] = {
                        **latest,
                        "ingest_run_id": latest["ingest_run_id"],
                        "commit_sha": latest.get("commit_sha"),
                        "file_manifest": latest.get("file_manifest", []),
                        "budget_usage": latest.get("budget_usage", {}),
                        "files_indexed": latest.get("files_indexed", 0),
                        "chunks_indexed": latest.get("chunks_indexed", 0),
                    }
                poll_hint.success("Background indexing completed.")
                break
            if job["status"] == "FAILED":
                poll_hint.error(job.get("error") or "Ingestion failed.")
                break

            poll_hint.caption("Live polling indexing progress...")
            time.sleep(poll_interval)

    if st.button("Run Repo Ingestion (Foreground)"):
        progress = st.progress(0)
        status = st.empty()

        def _progress_cb(frac: float, message: str):
            progress.progress(max(0, min(100, int(frac * 100))))
            status.write(message)

        with st.spinner("Indexing repository..."):
            ingest = run_ingestion(
                store=store,
                company_id=company_id,
                scout_run_id=scout["scout_run_id"],
                repo_full_name=repo["full_name"],
                repo_url=repo["html_url"],
                company_profile=profile,
                ref=ref.strip() or None,
                index_profile=index_profile,
                force_reindex=force_reindex,
                progress_cb=_progress_cb,
            )
        progress.progress(100)
        st.session_state["last_ingestion"] = ingest
        st.session_state["last_dd"] = None
        if ingest.get("reused"):
            st.success(f"Reused ingestion. ingest_run_id={ingest['ingest_run_id']}")
        else:
            st.success(f"Ingestion complete. ingest_run_id={ingest['ingest_run_id']}")

    ingestion = st.session_state.get("last_ingestion")
    if ingestion:
        st.metric("Commit / Ref", ingestion.get("commit_sha"))
        files_count = ingestion.get("files_indexed") or len(ingestion.get("file_manifest", []))
        chunks_count = ingestion.get("chunks_indexed") or len(ingestion.get("chunks", []))
        budget = ingestion.get("budget_usage", {}) or {}
        files_fetched = budget.get("files_fetched")
        c1, c2, c3 = st.columns(3)
        c1.metric("GitHub Files Fetched", files_fetched if files_fetched is not None else files_count)
        c2.metric("Evidence Items Indexed", chunks_count)
        c3.metric("File Paths Indexed", files_count)
        st.json(budget)
        st.caption(
            "Evidence Items include file-derived evidence plus metadata/release/issue/discussion evidence. "
            "So Evidence Items can be higher than Files Fetched."
        )
        st.caption(
            "Analysis path: deterministic parsers + rule-based agents on stored evidence/signals. "
            "Cortex is currently used only for optional Top-3 scouting rerank."
        )
        ingest_run_id = ingestion.get("ingest_run_id")
        if ingest_run_id:
            breakdown = store.get_evidence_source_breakdown(ingest_run_id)
            if breakdown:
                st.caption("Evidence source breakdown (why items can exceed file count):")
                st.json(breakdown)
        with st.expander("Indexed Files"):
            files = ingestion.get("file_manifest", []) or []
            if not files:
                st.info("No indexed files recorded.")
            else:
                st.dataframe(pd.DataFrame({"file_path": sorted(files)}), use_container_width=True)


def page_due_diligence(store: SnowflakeStore):
    st.header("4) Due Diligence Dashboard")
    llm_provider = os.getenv("LLM_PROVIDER", "auto")
    if os.getenv("OPENAI_API_KEY") or os.getenv("XAI_API_KEY"):
        st.caption(f"Agent reasoning mode: LLM-enabled ({llm_provider}).")
    else:
        st.caption("Agent reasoning mode: deterministic fallback (set OPENAI_API_KEY or XAI_API_KEY for intelligent reasoning).")
    company_id, profile, _ = _current_company(store)
    scout = st.session_state.get("last_scout")
    repo = st.session_state.get("selected_repo")
    ingestion = st.session_state.get("last_ingestion")
    if not (company_id and profile and scout and repo and ingestion):
        st.info("Complete scouting + selection + ingestion first.")
        return
    if ingestion.get("repo_full_name") != repo.get("full_name"):
        st.warning("Selected repo and ingestion context do not match. Re-run ingestion for this repo.")
        st.session_state["last_dd"] = None
        return

    ingest_run_id = ingestion.get("ingest_run_id")
    if ingest_run_id:
        partial_chunks = store.count_evidence_chunks(ingest_run_id)
        if partial_chunks > 0:
            st.info(f"Partial evidence available: {partial_chunks} chunks. You can run due diligence now and rerun later when indexing completes.")
        else:
            st.warning("No evidence chunks available yet. Wait for ingestion to produce initial chunks.")

    if st.button("Run Multi-Agent Due Diligence"):
        with st.spinner("Running due diligence agents..."):
            dd = run_due_diligence(
                store=store,
                company_id=company_id,
                scout_run_id=scout["scout_run_id"],
                ingest_run_id=ingestion["ingest_run_id"],
                company_profile=profile,
                repo_meta=repo,
                requirements=scout.get("requirements", ""),
            )
        st.session_state["last_dd"] = dd
        st.success(f"Due diligence complete. dd_run_id={dd['dd_run_id']}")

    dd = st.session_state.get("last_dd")
    if not dd:
        st.info("Run due diligence to view report.")
        return

    record = store.load_due_diligence(dd["dd_run_id"])
    if (
        record.get("repo_full_name") != repo.get("full_name")
        or record.get("ingest_run_id") != ingest_run_id
        or record.get("company_id") != company_id
    ):
        st.warning("Previous report belongs to a different company/repo/ingestion context. Run due diligence again.")
        st.session_state["last_dd"] = None
        return
    report = record.get("report_json", {})
    findings = record.get("findings", [])

    st.subheader(f"Decision: {report.get('decision')}")
    if findings:
        def _format_citation(c: Dict[str, Any]) -> str:
            path = c.get("file_path") or c.get("doc_section") or c.get("api_ref") or "source"
            sl = c.get("start_line")
            el = c.get("end_line")
            if sl and el:
                path = f"{path}:{sl}-{el}"
            return f"{path} | {c.get('url') or c.get('repo_url') or ''}"

        rows = []
        for f in findings:
            cites = f.get("citations", []) or []
            rows.append(
                {
                    "finding_id": f.get("finding_id"),
                    "agent_name": f.get("agent_name"),
                    "category": f.get("category"),
                    "claim": f.get("claim"),
                    "confidence": f.get("confidence"),
                    "base_severity": f.get("base_severity"),
                    "context_multiplier": f.get("context_multiplier"),
                    "adjusted_severity": f.get("adjusted_severity"),
                    "owner": f.get("owner"),
                    "effort": f.get("effort"),
                    "citation_count": len(cites),
                    "citations_text": " || ".join([_format_citation(c) for c in cites[:3]]),
                    "next_checks_text": " | ".join(f.get("next_checks", [])[:3]),
                }
            )
        df = pd.DataFrame(rows)
        c1, c2 = st.columns(2)
        with c1:
            cat_df = df.groupby("category", as_index=False)["adjusted_severity"].mean()
            st.bar_chart(cat_df.set_index("category"))
        with c2:
            owner_df = df.groupby("owner", as_index=False)["adjusted_severity"].max()
            st.bar_chart(owner_df.set_index("owner"))
        selected_owner = st.selectbox("Filter findings by owner", ["All"] + sorted(df["owner"].dropna().unique().tolist()))
        filtered = df if selected_owner == "All" else df[df["owner"] == selected_owner]
        st.dataframe(
            filtered[
                [
                    "agent_name",
                    "category",
                    "claim",
                    "confidence",
                    "base_severity",
                    "context_multiplier",
                    "adjusted_severity",
                    "owner",
                    "effort",
                    "citation_count",
                ]
            ],
            use_container_width=True,
        )
        with st.expander("Finding evidence details"):
            st.dataframe(
                filtered[["finding_id", "claim", "citations_text", "next_checks_text"]],
                use_container_width=True,
            )

    st.subheader("Prioritized Next Steps")
    st.dataframe(pd.DataFrame(report.get("prioritized_next_steps", [])), use_container_width=True)

    st.subheader("Maintainer Questions")
    for q in report.get("maintainer_questions", []):
        st.markdown(f"- {q}")

    st.subheader("Export")
    report_json_text = json.dumps(report, indent=2)
    report_md_text = record.get("report_md", "")
    st.download_button("Download JSON", report_json_text, file_name=f"{dd['dd_run_id']}.json", mime="application/json")
    st.download_button("Download Markdown", report_md_text, file_name=f"{dd['dd_run_id']}.md", mime="text/markdown")

    st.subheader("Composio: Create Tasks From Next Steps")
    if st.button("Create tasks"):
        result = create_tasks_from_next_steps(dd_result=dd, dry_run=not bool(os.getenv("COMPOSIO_API_KEY")))
        if result.get("ok"):
            st.success(result.get("message"))
        else:
            st.error(result.get("message"))
        st.json(result)


def page_war_room(store: SnowflakeStore):
    st.header("5) War Room Transcript")
    dd = st.session_state.get("last_dd")
    if not dd:
        st.info("Run due diligence first.")
        return
    record = store.load_due_diligence(dd["dd_run_id"])
    st.write(f"Run: {dd['dd_run_id']}")
    transcript = record.get("transcripts", [])
    if not transcript:
        st.warning("No transcripts found.")
        return

    dump_lines = [f"Run: {dd['dd_run_id']}"]
    for t in transcript:
        header = f"{t.get('created_at')} | {t.get('agent_name')} | {t.get('step')} | {t.get('status')}"
        dump_lines.append(header)
        payload = t.get("payload", {}) or {}
        if payload.get("type") == "conversation":
            with st.container(border=True):
                st.markdown(f"**{t.get('agent_name')} Conversation**")
                for msg in payload.get("messages", []):
                    speaker = msg.get("speaker", "Agent")
                    text = msg.get("message", "")
                    st.markdown(f"- **{speaker}:** {text}")
                    dump_lines.append(f"{speaker}: {text}")
        else:
            st.markdown(f"`{header}`")
            st.json(payload)
            dump_lines.append(json.dumps(payload, default=str))
        dump_lines.append("")

    dump_text = "\n".join(dump_lines)
    st.download_button(
        "Download Transcript Dump (.txt)",
        data=dump_text,
        file_name=f"{dd['dd_run_id']}_war_room.txt",
        mime="text/plain",
    )


def main():
    st.title("OpenSourceOps")
    st.caption("OSS Due Diligence AI for Companies (Snowflake-Backed)")
    try:
        store, github = init_services()
    except Exception as exc:
        st.error(str(exc))
        st.info("Set Snowflake env vars in `.env` and restart.")
        return

    _safe_state("selected_company_id", None)
    _safe_state("last_scout", None)
    _safe_state("selected_repo", None)
    _safe_state("last_ingestion", None)
    _safe_state("last_dd", None)

    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Page",
            [
                "Company Onboarding",
                "Scouting Results",
                "Repo Selection + Ingestion",
                "Due Diligence Dashboard",
                "War Room Transcript",
            ],
        )

    if page == "Company Onboarding":
        page_onboarding(store)
    elif page == "Scouting Results":
        page_scouting(store, github)
    elif page == "Repo Selection + Ingestion":
        page_ingestion(store)
    elif page == "Due Diligence Dashboard":
        page_due_diligence(store)
    else:
        page_war_room(store)


if __name__ == "__main__":
    main()
