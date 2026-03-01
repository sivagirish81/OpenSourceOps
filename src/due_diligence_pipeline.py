"""Streamlit-backend pipeline for scouting, ingestion, and due diligence."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from storage.snowflake_store import SnowflakeStore
from src.composio_integration import create_internal_github_artifact, get_composio_client
from src.due_diligence_agents import (
    ArchitectureIntegrationAgent,
    CommunityMaintenanceAgent,
    CrewRuntime,
    JudgeVerifierAgent,
    LicenseComplianceAgent,
    ReliabilityOpsAgent,
    RepoLibrarianAgent,
    ScoutingAgent,
    SecuritySupplyChainAgent,
    report_to_markdown,
)
from src.github_client import GitHubClient
from src.llm_router import LLMRouter

try:
    from crewai import Agent as CrewAgent
    from crewai import Crew, Task
    from langchain.tools import StructuredTool
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    CrewAgent = None
    Crew = None
    Task = None
    StructuredTool = None
    ChatOpenAI = None


def _dialogue_for_agent_fallback(
    agent_name: str,
    repo_name: str,
    notes: Dict[str, Any],
    findings: List[Dict[str, Any]],
    next_agent: str | None,
    peer_agent: str | None,
) -> Dict[str, Any]:
    top = findings[0] if findings else {}
    claim = top.get("claim", "No concrete claim provided.")
    sev = top.get("adjusted_severity", 0.0)
    conf = top.get("confidence", "Low")
    cites = top.get("citations", [])
    cite_hint = "with citations" if cites else "without citations"
    handover = (
        f"Handover to {next_agent}: use my finding and citations to validate impact in your domain."
        if next_agent
        else "No further handover. Awaiting judge consolidation."
    )
    peer_note = (
        f"{peer_agent}, please challenge assumptions and call out missing evidence."
        if peer_agent
        else "Team, verify this signal against your own evidence."
    )
    messages = [
        {
            "speaker": "Coordinator",
            "message": f"{agent_name}, provide your strongest due-diligence signal for {repo_name}.",
        },
        {
            "speaker": agent_name,
            "message": (
                f"Primary finding: {claim} "
                f"(adjusted_severity={sev}, confidence={conf}, {cite_hint})."
            ),
        },
        {
            "speaker": agent_name,
            "message": peer_note,
        },
        {
            "speaker": (peer_agent or "ArchitectureIntegrationAgent"),
            "message": "I reviewed your claim. It is directionally useful, but I need clearer deployment/integration evidence.",
        },
        {
            "speaker": "JudgeVerifierAgent",
            "message": "State any uncertainty and next checks explicitly.",
        },
        {
            "speaker": agent_name,
            "message": f"Notes: {notes}. Next checks are attached to the finding.",
        },
        {
            "speaker": "Coordinator",
            "message": handover,
        },
    ]
    return {"type": "conversation", "messages": messages}


def _judge_resolution_dialogue_fallback(report: Dict[str, Any]) -> Dict[str, Any]:
    decision = report.get("decision", "CONDITIONAL_GO")
    risks = report.get("risk_register", [])
    top_risk = risks[0] if risks else {}
    messages = [
        {"speaker": "Coordinator", "message": "Roundtable check: do we agree on the top risks and owners?"},
        {"speaker": "SecuritySupplyChainAgent", "message": "Security risks remain material unless auditability and SBOM posture are confirmed."},
        {"speaker": "ReliabilityOpsAgent", "message": "Operational readiness is feasible with staged rollout and observability baseline."},
        {"speaker": "LicenseComplianceAgent", "message": "Legal clearance is required before production rollout."},
        {"speaker": "JudgeVerifierAgent", "message": f"Final decision: {decision}."},
        {
            "speaker": "JudgeVerifierAgent",
            "message": (
                f"Top risk: {top_risk.get('claim', 'n/a')} "
                f"(adjusted_severity={top_risk.get('adjusted_severity', 'n/a')})."
            ),
        },
        {
            "speaker": "Coordinator",
            "message": "Action owners assigned in prioritized next steps. Exporting report.",
        },
    ]
    return {"type": "conversation", "messages": messages}


def _evidence_summary(chunks: List[Dict[str, Any]], max_items: int = 8) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ch in chunks:
        text = (ch.get("chunk_text") or "")[:700]
        if not text.strip():
            continue
        out.append(
            {
                "file_path": ch.get("file_path"),
                "start_line": int(ch.get("start_line") or 1),
                "end_line": int(ch.get("end_line") or 1),
                "url": ch.get("url"),
                "snippet": text,
            }
        )
        if len(out) >= max_items:
            break
    return out


def _llm_conversation(
    llm: LLMRouter,
    *,
    agent_name: str,
    repo_name: str,
    requirements: str,
    notes: Dict[str, Any],
    findings: List[Dict[str, Any]],
    company_profile: Dict[str, Any],
    next_agent: str | None,
    peer_agent: str | None,
) -> Dict[str, Any] | None:
    system = (
        "You are generating a realistic multi-agent due-diligence transcript. "
        "Return strict JSON only with shape: "
        "{ \"type\":\"conversation\", \"messages\":[{\"speaker\":string,\"message\":string}] }."
    )
    user = json.dumps(
        {
            "agent_name": agent_name,
            "repo_name": repo_name,
            "requirements": requirements,
            "company_profile": company_profile,
            "notes": notes,
            "findings": findings[:3],
            "next_agent": next_agent,
            "peer_agent": peer_agent,
            "rules": [
                "Use 6-10 messages",
                "Agents can challenge each other",
                "Mention uncertainty when evidence is weak",
                "No hallucinated facts; reference available findings only",
            ],
        }
    )
    try:
        payload = llm.complete_json(system=system, user=user)
        msgs = payload.get("messages", [])
        if payload.get("type") != "conversation" or not isinstance(msgs, list) or not msgs:
            return None
        cleaned = []
        for m in msgs[:12]:
            if not isinstance(m, dict):
                continue
            speaker = str(m.get("speaker") or "Agent")[:80]
            message = str(m.get("message") or "").strip()
            if not message:
                continue
            cleaned.append({"speaker": speaker, "message": message})
        if not cleaned:
            return None
        return {"type": "conversation", "messages": cleaned}
    except Exception:
        return None


def _llm_enrich_findings(
    llm: LLMRouter,
    *,
    agent_name: str,
    requirements: str,
    company_profile: Dict[str, Any],
    repo_meta: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    deterministic_findings: List[Dict[str, Any]],
) -> List[Dict[str, Any]] | None:
    if not deterministic_findings:
        return None
    system = (
        "You are an enterprise OSS due-diligence agent. Return strict JSON only with shape "
        "{ \"findings\": [ {\"claim\":string,\"confidence\":\"High|Medium|Low\","
        "\"next_checks\":[string],\"multiplier_rationale\":string,"
        "\"citations\":[{\"file_path\":string,\"start_line\":number,\"end_line\":number,\"url\":string}] } ] }."
    )
    user = json.dumps(
        {
            "agent_name": agent_name,
            "repo": {"full_name": repo_meta.get("full_name"), "url": repo_meta.get("html_url")},
            "requirements": requirements,
            "company_profile": company_profile,
            "evidence": evidence,
            "starting_findings": deterministic_findings[:3],
            "rules": [
                "Ground every claim in evidence snippets",
                "If not confirmed, set claim exactly to 'Not confirmed from repo evidence'",
                "Keep at most 2 findings",
            ],
        }
    )
    try:
        payload = llm.complete_json(system=system, user=user)
        generated = payload.get("findings", [])
        if not isinstance(generated, list) or not generated:
            return None
        enriched: List[Dict[str, Any]] = []
        for idx, g in enumerate(generated[:2]):
            base = dict(deterministic_findings[min(idx, len(deterministic_findings) - 1)])
            base["claim"] = str(g.get("claim") or base.get("claim"))
            conf = str(g.get("confidence") or base.get("confidence") or "Low")
            base["confidence"] = conf if conf in {"High", "Medium", "Low"} else "Low"
            checks = g.get("next_checks") if isinstance(g.get("next_checks"), list) else base.get("next_checks")
            base["next_checks"] = checks or []
            mr = str(g.get("multiplier_rationale") or base.get("multiplier_rationale") or "")
            base["multiplier_rationale"] = mr
            cits = g.get("citations") if isinstance(g.get("citations"), list) else []
            normalized_cits = []
            for c in cits[:4]:
                if not isinstance(c, dict):
                    continue
                normalized_cits.append(
                    {
                        "repo_url": repo_meta.get("html_url"),
                        "file_path": c.get("file_path"),
                        "start_line": int(c.get("start_line") or 1),
                        "end_line": int(c.get("end_line") or int(c.get("start_line") or 1)),
                        "doc_section": None,
                        "url": c.get("url"),
                    }
                )
            if normalized_cits:
                base["citations"] = normalized_cits
            enriched.append(base)
        return enriched
    except Exception:
        return None


def _crewai_enabled() -> bool:
    return bool(
        os.getenv("ENABLE_CREWAI_RUNTIME", "0") == "1"
        and CrewAgent
        and Crew
        and Task
        and StructuredTool
        and (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("XAI_API_KEY")
        )
    )


def _crewai_llm_from_env():
    if ChatOpenAI is None:
        return None
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1,
        )
    if os.getenv("XAI_API_KEY"):
        return ChatOpenAI(
            model=os.getenv("XAI_MODEL", "grok-2-latest"),
            openai_api_base=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
            openai_api_key=os.getenv("XAI_API_KEY"),
            temperature=0.1,
        )
    return None


def _run_due_diligence_with_crewai(
    *,
    company_profile: Dict[str, Any],
    repo_meta: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    requirements: str,
) -> Dict[str, Any] | None:
    """True CrewAI orchestration with task dependencies and tool-backed executors."""
    if not _crewai_enabled():
        return None
    llm = _crewai_llm_from_env()
    if llm is None:
        return None

    security_agent = SecuritySupplyChainAgent()
    license_agent = LicenseComplianceAgent()
    reliability_agent = ReliabilityOpsAgent()
    architecture_agent = ArchitectureIntegrationAgent()
    community_agent = CommunityMaintenanceAgent()
    judge_agent = JudgeVerifierAgent()

    state: Dict[str, Any] = {"findings": [], "task_outputs": []}

    def _append_output(label: str, payload: Dict[str, Any]) -> str:
        state["task_outputs"].append({"task": label, "output": payload})
        return json.dumps(payload)

    def run_security(request: str = "") -> str:
        out = security_agent.run(company_profile, chunks, repo_meta["html_url"])
        state["findings"].extend(out.findings)
        return _append_output("security", {"agent_name": out.agent_name, "findings": out.findings, "notes": out.notes})

    def run_license(request: str = "") -> str:
        out = license_agent.run(company_profile, chunks, repo_meta["html_url"])
        state["findings"].extend(out.findings)
        return _append_output("license", {"agent_name": out.agent_name, "findings": out.findings, "notes": out.notes})

    def run_reliability(request: str = "") -> str:
        out = reliability_agent.run(company_profile, chunks, repo_meta["html_url"])
        state["findings"].extend(out.findings)
        return _append_output("reliability", {"agent_name": out.agent_name, "findings": out.findings, "notes": out.notes})

    def run_architecture(request: str = "") -> str:
        out = architecture_agent.run(company_profile, chunks, repo_meta["html_url"])
        state["findings"].extend(out.findings)
        return _append_output("architecture", {"agent_name": out.agent_name, "findings": out.findings, "notes": out.notes})

    def run_community(request: str = "") -> str:
        out = community_agent.run(company_profile, repo_meta, repo_meta["html_url"])
        state["findings"].extend(out.findings)
        return _append_output("community", {"agent_name": out.agent_name, "findings": out.findings, "notes": out.notes})

    def run_judge(request: str = "") -> str:
        report = judge_agent.run(
            company_profile=company_profile,
            findings=state["findings"],
            repo_full_name=repo_meta["full_name"],
            requirements=requirements,
        )
        return _append_output("judge", {"report": report})

    sec_tool = StructuredTool.from_function(
        func=run_security,
        name="run_security_supply_chain_agent",
        description="Run security and supply chain analysis using company profile, requirements, and repo evidence.",
    )
    lic_tool = StructuredTool.from_function(
        func=run_license,
        name="run_license_compliance_agent",
        description="Run license and legal compliance analysis using company profile, requirements, and repo evidence.",
    )
    rel_tool = StructuredTool.from_function(
        func=run_reliability,
        name="run_reliability_ops_agent",
        description="Run reliability and operations analysis using company profile, requirements, and repo evidence.",
    )
    arch_tool = StructuredTool.from_function(
        func=run_architecture,
        name="run_architecture_integration_agent",
        description="Run architecture/integration analysis using company profile, requirements, and repo evidence.",
    )
    comm_tool = StructuredTool.from_function(
        func=run_community,
        name="run_community_maintenance_agent",
        description="Run community maintenance and support analysis using company profile and requirements.",
    )
    judge_tool = StructuredTool.from_function(
        func=run_judge,
        name="run_judge_verifier_agent",
        description="Aggregate and judge all findings into final due diligence report.",
    )

    company_context = (
        f"requirements={requirements}\n"
        f"company_profile={json.dumps(company_profile)}\n"
        f"repo={repo_meta.get('full_name')}\n"
    )

    sec_a = CrewAgent(role="SecuritySupplyChainAgent", goal="Produce grounded security finding JSON.", backstory="Security due diligence specialist.", llm=llm, tools=[sec_tool], allow_delegation=False)
    lic_a = CrewAgent(role="LicenseComplianceAgent", goal="Produce grounded license finding JSON.", backstory="Legal/OSS compliance specialist.", llm=llm, tools=[lic_tool], allow_delegation=False)
    rel_a = CrewAgent(role="ReliabilityOpsAgent", goal="Produce grounded reliability finding JSON.", backstory="SRE and production ops specialist.", llm=llm, tools=[rel_tool], allow_delegation=False)
    arch_a = CrewAgent(role="ArchitectureIntegrationAgent", goal="Produce grounded integration finding JSON.", backstory="Architecture integration specialist.", llm=llm, tools=[arch_tool], allow_delegation=False)
    comm_a = CrewAgent(role="CommunityMaintenanceAgent", goal="Produce grounded community/support finding JSON.", backstory="Open-source maintenance specialist.", llm=llm, tools=[comm_tool], allow_delegation=False)
    judge_a = CrewAgent(role="JudgeVerifierAgent", goal="Produce final due diligence report JSON.", backstory="Evidence verifier and decision maker.", llm=llm, tools=[judge_tool], allow_delegation=False)

    t1 = Task(
        description=company_context + "Call tool run_security_supply_chain_agent and return the JSON output only.",
        expected_output="JSON output from run_security_supply_chain_agent",
        agent=sec_a,
        tools=[sec_tool],
    )
    t2 = Task(
        description=company_context + "Use previous context and call tool run_license_compliance_agent. Return JSON only.",
        expected_output="JSON output from run_license_compliance_agent",
        agent=lic_a,
        tools=[lic_tool],
        context=[t1],
    )
    t3 = Task(
        description=company_context + "Use previous context and call tool run_reliability_ops_agent. Return JSON only.",
        expected_output="JSON output from run_reliability_ops_agent",
        agent=rel_a,
        tools=[rel_tool],
        context=[t1, t2],
    )
    t4 = Task(
        description=company_context + "Use previous context and call tool run_architecture_integration_agent. Return JSON only.",
        expected_output="JSON output from run_architecture_integration_agent",
        agent=arch_a,
        tools=[arch_tool],
        context=[t1, t2, t3],
    )
    t5 = Task(
        description=company_context + "Use previous context and call tool run_community_maintenance_agent. Return JSON only.",
        expected_output="JSON output from run_community_maintenance_agent",
        agent=comm_a,
        tools=[comm_tool],
        context=[t1, t2, t3, t4],
    )
    t6 = Task(
        description=company_context + "Use all previous context and call tool run_judge_verifier_agent. Return JSON only.",
        expected_output="JSON output from run_judge_verifier_agent",
        agent=judge_a,
        tools=[judge_tool],
        context=[t1, t2, t3, t4, t5],
    )

    crew = Crew(
        agents=[sec_a, lic_a, rel_a, arch_a, comm_a, judge_a],
        tasks=[t1, t2, t3, t4, t5, t6],
        verbose=False,
    )
    try:
        _ = crew.kickoff()
    except Exception:
        return None

    if not state["findings"]:
        return None
    # Judge tool may fail to execute; fall back locally if report missing.
    report = None
    for item in reversed(state["task_outputs"]):
        if item["task"] == "judge":
            report = (item.get("output") or {}).get("report")
            break
    if not report:
        report = judge_agent.run(
            company_profile=company_profile,
            findings=state["findings"],
            repo_full_name=repo_meta["full_name"],
            requirements=requirements,
        )
    return {
        "findings": state["findings"],
        "report": report,
        "task_outputs": state["task_outputs"],
    }


def _rerank_with_cortex_if_available(
    store: SnowflakeStore,
    requirements: str,
    constraints: Dict[str, Any],
    top3: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], bool]:
    """Optional Snowflake Cortex rerank pass; deterministic order used on failure."""
    if not top3:
        return top3, False
    model = os.getenv("CORTEX_MODEL_PRIMARY", "snowflake-arctic")
    compact = [
        {
            "full_name": r.get("full_name"),
            "description": r.get("description"),
            "score_breakdown": r.get("score_breakdown"),
            "final_score": r.get("final_score"),
            "pros": r.get("pros"),
            "cons": r.get("cons"),
        }
        for r in top3
    ]
    prompt = (
        "Return strict JSON with key 'reranked' as an array of full_name in best-first order.\n"
        "Prioritize closest requirement match; allow closest feasible fit even with minor constraint gaps.\n"
        f"requirements={requirements}\n"
        f"constraints={json.dumps(constraints)}\n"
        f"candidates={json.dumps(compact)}"
    )
    try:
        rows = store.query("SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s)", [model, prompt])
        raw = str(rows[0][0]) if rows else ""
        parsed = json.loads(raw.strip("`").replace("json\n", "", 1))
        order = parsed.get("reranked", [])
        by_name = {r["full_name"]: r for r in top3}
        reranked = [by_name[n] for n in order if n in by_name]
        remainder = [r for r in top3 if r["full_name"] not in set(order)]
        final = (reranked + remainder)[: max(len(top3), 3)]
        if not final:
            return top3, False
        return final, True
    except Exception:
        return top3, False


def run_scouting(
    store: SnowflakeStore,
    github: GitHubClient,
    company_id: str,
    company_profile: Dict[str, Any],
    requirements: str,
) -> Dict[str, Any]:
    scout = ScoutingAgent(github)
    crew_meta = CrewRuntime().kickoff("Context-aware scouting")
    result = scout.run(company_profile, requirements)
    reranked, cortex_used = _rerank_with_cortex_if_available(
        store=store,
        requirements=requirements,
        constraints=result["constraints"],
        top3=result["top3"],
    )
    result["top3"] = reranked[:3]
    result["cortex_used"] = cortex_used
    scout_run_id = store.create_scouting_run(
        company_id=company_id,
        requirements=requirements,
        constraints=result["constraints"],
        query_pack=result["query_pack"],
        top3=result["top3"],
    )
    for cand in result["top3"]:
        store.save_repo_candidate(scout_run_id=scout_run_id, company_id=company_id, candidate=cand)
    return {"scout_run_id": scout_run_id, "requirements": requirements, **result, "crew_meta": crew_meta}


def run_ingestion(
    store: SnowflakeStore,
    company_id: str,
    scout_run_id: str,
    repo_full_name: str,
    repo_url: str,
    company_profile: Dict[str, Any],
    ref: str | None = None,
    index_profile: str = "standard",
    force_reindex: bool = False,
    progress_cb=None,
) -> Dict[str, Any]:
    if not force_reindex:
        existing = store.get_latest_ingestion(company_id=company_id, repo_full_name=repo_full_name)
        if existing:
            if progress_cb:
                progress_cb(1.0, f"Reused existing ingestion {existing['ingest_run_id']}")
            return {
                "ingest_run_id": existing["ingest_run_id"],
                "repo_id": "",
                "commit_sha": existing["commit_sha"],
                "chunks": [],
                "file_manifest": existing.get("file_manifest", []),
                "repo_map": {
                    "docs_files": [],
                    "config_files": [],
                    "code_files": [],
                },
                "budget_usage": existing.get("budget_usage", {}),
                "repo_full_name": repo_full_name,
                "repo_url": repo_url,
                "reused": True,
                "created_at": existing.get("created_at"),
            }

    librarian = RepoLibrarianAgent()
    idx = librarian.run(
        company_profile,
        repo_url=repo_url,
        full_name=repo_full_name,
        ref=ref,
        index_profile=index_profile,
        progress_cb=progress_cb,
    )
    ingest_run_id = store.save_repo_ingestion(
        scout_run_id=scout_run_id,
        company_id=company_id,
        repo_full_name=repo_full_name,
        repo_url=repo_url,
        commit_sha=idx["commit_sha"],
        files_indexed=len(idx["file_manifest"]),
        chunks_indexed=len(idx["chunks"]),
        budget_usage=idx["budget_usage"],
        file_manifest=idx["file_manifest"],
    )
    # Persist both compatibility chunks and normalized evidence/signals.
    store.save_evidence_chunks(ingest_run_id=ingest_run_id, chunks=idx["chunks"])
    if idx.get("raw_evidence"):
        store.save_repo_evidence_batch(
            ingest_run_id=ingest_run_id,
            repo_full_name=repo_full_name,
            evidence_batch=idx["raw_evidence"],
        )
    if idx.get("signals"):
        store.save_repo_signals(
            ingest_run_id=ingest_run_id,
            repo_full_name=repo_full_name,
            signals=idx["signals"],
        )
    return {"ingest_run_id": ingest_run_id, **idx, "reused": False}


def run_due_diligence(
    store: SnowflakeStore,
    company_id: str,
    scout_run_id: str,
    ingest_run_id: str,
    company_profile: Dict[str, Any],
    repo_meta: Dict[str, Any],
    requirements: str = "",
) -> Dict[str, Any]:
    llm = LLMRouter(store)
    dd_run_id = store.create_due_diligence_run(
        company_id=company_id,
        scout_run_id=scout_run_id,
        ingest_run_id=ingest_run_id,
        repo_full_name=repo_meta["full_name"],
        repo_url=repo_meta["html_url"],
    )
    chunks = store.load_evidence_chunks(ingest_run_id)
    normalized = store.load_repo_evidence(ingest_run_id)
    repo_signals = store.load_repo_signals(ingest_run_id)
    if normalized:
        # Prefer signal-based normalized evidence when present.
        chunks = [
            {
                "repo_url": e.get("repo_url"),
                "commit_sha": e.get("commit_or_ref"),
                "file_path": e.get("file_path") or e.get("source_type"),
                "start_line": 1,
                "end_line": len((e.get("content") or "").splitlines()) or 1,
                "heading": e.get("source_type"),
                "symbol_name": None,
                "content_type": "doc",
                "chunk_text": (e.get("content") or json.dumps(e.get("structured") or {}))[:12000],
                "url": e.get("url"),
                "api_ref": e.get("api_ref"),
            }
            for e in normalized
        ]
    if repo_signals:
        for s in repo_signals:
            chunks.append(
                {
                    "repo_url": repo_meta["html_url"],
                    "commit_sha": "signal",
                    "file_path": f"signal:{s.get('category')}:{s.get('signal_name')}",
                    "start_line": 1,
                    "end_line": 1,
                    "heading": "repo_signal",
                    "symbol_name": None,
                    "content_type": "doc",
                    "chunk_text": f"{s.get('signal_name')}={s.get('value')}",
                    "url": ((s.get("citations") or [{}])[0]).get("url"),
                }
            )
    agents = [
        SecuritySupplyChainAgent(),
        LicenseComplianceAgent(),
        ReliabilityOpsAgent(),
        ArchitectureIntegrationAgent(),
        CommunityMaintenanceAgent(),
    ]
    findings: List[Dict[str, Any]] = []
    report = None

    crew_result = _run_due_diligence_with_crewai(
        company_profile=company_profile,
        repo_meta=repo_meta,
        chunks=chunks,
        requirements=requirements,
    )
    if crew_result:
        findings = crew_result["findings"]
        report = crew_result["report"]
        store.save_transcript(
            dd_run_id=dd_run_id,
            agent_name="CrewAI",
            step="orchestration",
            status="SUCCESS",
            payload={
                "mode": "crewai",
                "task_outputs": crew_result.get("task_outputs", []),
                "finding_count": len(findings),
            },
        )
    else:
        for idx, agent in enumerate(agents):
            if isinstance(agent, CommunityMaintenanceAgent):
                out = agent.run(company_profile, repo_meta, repo_meta["html_url"])
            else:
                out = agent.run(company_profile, chunks, repo_meta["html_url"])
            evidence_cards = _evidence_summary(chunks)
            enriched = None
            if llm.enabled():
                enriched = _llm_enrich_findings(
                    llm,
                    agent_name=out.agent_name,
                    requirements=requirements,
                    company_profile=company_profile,
                    repo_meta=repo_meta,
                    evidence=evidence_cards,
                    deterministic_findings=out.findings,
                )
            if enriched:
                out.findings = enriched
                out.notes = {**(out.notes or {}), "reasoning_mode": "llm_enriched"}
            else:
                out.notes = {**(out.notes or {}), "reasoning_mode": "deterministic"}
            findings.extend(out.findings)
            store.save_transcript(
                dd_run_id=dd_run_id,
                agent_name=out.agent_name,
                step="analysis",
                status="SUCCESS",
                payload={"notes": out.notes, "finding_count": len(out.findings)},
            )
            store.save_transcript(
                dd_run_id=dd_run_id,
                agent_name=out.agent_name,
                step="war_room_dialogue",
                status="INFO",
                payload=(
                    _llm_conversation(
                        llm,
                        agent_name=out.agent_name,
                        repo_name=repo_meta["full_name"],
                        requirements=requirements,
                        notes=out.notes,
                        findings=out.findings,
                        company_profile=company_profile,
                        next_agent=(agents[idx + 1].name if idx + 1 < len(agents) else None),
                        peer_agent=(agents[idx - 1].name if idx - 1 >= 0 else None),
                    )
                    or _dialogue_for_agent_fallback(
                        agent_name=out.agent_name,
                        repo_name=repo_meta["full_name"],
                        notes=out.notes,
                        findings=out.findings,
                        next_agent=(agents[idx + 1].name if idx + 1 < len(agents) else None),
                        peer_agent=(agents[idx - 1].name if idx - 1 >= 0 else None),
                    )
                ),
            )
            store.save_findings_batch(dd_run_id, out.findings)

    judge = JudgeVerifierAgent()
    if not report:
        report = judge.run(
            company_profile=company_profile,
            findings=findings,
            repo_full_name=repo_meta["full_name"],
            requirements=requirements,
        )
    # Ensure findings persisted when produced via crew-mode.
    if crew_result:
        store.save_findings_batch(dd_run_id, findings)
    report_md = report_to_markdown(report)
    store.save_transcript(
        dd_run_id=dd_run_id,
        agent_name=judge.name,
        step="finalize_report",
        status="SUCCESS",
        payload={"decision": report["decision"], "findings": len(report["risk_register"])},
    )
    store.save_transcript(
        dd_run_id=dd_run_id,
        agent_name=judge.name,
        step="war_room_dialogue",
        status="INFO",
        payload=(
            _llm_conversation(
                llm,
                agent_name=judge.name,
                repo_name=repo_meta["full_name"],
                requirements=requirements,
                notes={"decision": report.get("decision")},
                findings=report.get("risk_register", [])[:3],
                company_profile=company_profile,
                next_agent=None,
                peer_agent="Coordinator",
            )
            or _judge_resolution_dialogue_fallback(report)
        ),
    )
    store.save_final_report(dd_run_id, report["decision"], report, report_md)
    return {"dd_run_id": dd_run_id, "report_json": report, "report_md": report_md}


def create_tasks_from_next_steps(dd_result: Dict[str, Any], dry_run: bool = True) -> Dict[str, Any]:
    client = get_composio_client()
    steps = dd_result.get("report_json", {}).get("prioritized_next_steps", [])
    if not steps:
        return {"ok": False, "message": "No next steps available.", "payload": {}}
    payload = {
        "title": f"Due diligence follow-up: {dd_result.get('report_json', {}).get('repo', 'repository')}",
        "steps": steps[:8],
    }
    if dry_run or client is None:
        return {
            "ok": True,
            "dry_run": True,
            "message": "Composio not configured. Returning task payload.",
            "payload": payload,
            "how_to_enable": [
                "Set COMPOSIO_API_KEY",
                "Connect GitHub/Jira app in Composio dashboard",
                "Re-run and click Create tasks",
            ],
        }
    first = steps[0]
    body = "\n".join([f"- {s.get('owner')}: {s.get('task')} ({s.get('priority')})" for s in steps[:8]])
    result = create_internal_github_artifact(
        composio_client=client,
        repo=dd_result.get("report_json", {}).get("repo", ""),
        title=payload["title"],
        body=body,
        dry_run=False,
    )
    result["payload"] = payload
    return result
