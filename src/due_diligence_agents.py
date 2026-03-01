"""Context-aware due diligence agents and report assembly."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from jsonschema import validate

from src.github_client import GitHubClient
from src.scoring import score_repo
from src.evidence_collector import RepoEvidenceCollector

try:
    from crewai import Agent, Crew, Task
except Exception:  # pragma: no cover
    Agent = None
    Crew = None
    Task = None


REPORT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["decision", "risk_register", "prioritized_next_steps", "maintainer_questions"],
    "properties": {
        "decision": {"type": "string", "enum": ["GO", "CONDITIONAL_GO", "NO_GO"]},
        "risk_register": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "category",
                    "claim",
                    "base_severity",
                    "context_multiplier",
                    "adjusted_severity",
                    "multiplier_rationale",
                    "confidence",
                    "citations",
                ],
            },
        },
        "prioritized_next_steps": {"type": "array"},
        "maintainer_questions": {"type": "array"},
    },
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9_\-]+", (text or "").lower()) if len(t) > 2]


def build_company_constraints(company_profile: Dict[str, Any]) -> Dict[str, Any]:
    tech = company_profile.get("technical_environment", {})
    risk = company_profile.get("risk_appetite", {})
    sec = company_profile.get("security_compliance", {})
    constraints = {
        "primary_languages": tech.get("primary_languages", []),
        "deployment_model": company_profile.get("organization", {}).get("deployment_model"),
        "cloud_provider": tech.get("cloud_provider"),
        "orchestration": tech.get("orchestration"),
        "identity_provider": tech.get("identity_provider"),
        "observability_stack": tech.get("observability_stack", []),
        "required_frameworks": sec.get("required_frameworks", []),
        "sbom_required": bool(sec.get("sbom_required")),
        "audit_logs_required": bool(sec.get("audit_logs_required")),
        "data_residency_restrictions": sec.get("data_residency_restrictions"),
        "maturity_preference": risk.get("maturity_preference", "Stable only"),
        "require_commercial_support": bool(risk.get("require_commercial_support")),
        "risk_tolerance": risk.get("risk_tolerance", "Low"),
    }
    return constraints


def context_multiplier_for_finding(
    company_profile: Dict[str, Any],
    category: str,
    missing_controls: List[str],
) -> Tuple[float, str]:
    sec = company_profile.get("security_compliance", {})
    risk = company_profile.get("risk_appetite", {})
    multiplier = 1.0
    reasons = []
    frameworks = [x.upper() for x in sec.get("required_frameworks", [])]
    if category == "security":
        if sec.get("audit_logs_required") and "audit_logs" in missing_controls:
            multiplier += 0.35
            reasons.append("Audit logs are required by company policy.")
        if sec.get("sbom_required") and "sbom" in missing_controls:
            multiplier += 0.25
            reasons.append("SBOM is required by compliance posture.")
        if any(x in frameworks for x in ["SOC2", "HIPAA", "ISO27001", "GDPR"]):
            multiplier += 0.15
            reasons.append("Regulated frameworks increase security severity.")
    if category == "community" and risk.get("require_commercial_support"):
        multiplier += 0.25
        reasons.append("Commercial support is required.")
    tolerance = (risk.get("risk_tolerance") or "Low").lower()
    if tolerance == "low":
        multiplier += 0.15
        reasons.append("Low risk tolerance.")
    elif tolerance == "high":
        multiplier -= 0.1
        reasons.append("High risk tolerance allows lower severity.")
    return max(0.7, round(multiplier, 2)), "; ".join(reasons) or "Default context weighting."


def citation(
    repo_url: str,
    file_path: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    doc_section: str | None = None,
) -> Dict[str, Any]:
    return {
        "repo_url": repo_url,
        "file_path": file_path,
        "start_line": start_line,
        "end_line": end_line,
        "doc_section": doc_section,
    }


def ensure_grounded_claim(finding: Dict[str, Any]) -> Dict[str, Any]:
    if finding.get("citations"):
        return finding
    finding["claim"] = "Not confirmed from repo evidence"
    finding["confidence"] = "Low"
    finding["next_checks"] = finding.get("next_checks") or [
        "Inspect README and docs/ for related feature mentions",
        "Search repository for key integration keywords",
    ]
    return finding


def dedupe_findings(findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for f in findings:
        key = hashlib.sha1(
            f"{normalize_text(f.get('claim',''))}|{f.get('category')}|{f.get('owner')}".encode("utf-8")
        ).hexdigest()
        prev = merged.get(key)
        if not prev:
            merged[key] = f
            continue
        if float(f.get("adjusted_severity", 0.0)) > float(prev.get("adjusted_severity", 0.0)):
            merged[key] = f
    return sorted(merged.values(), key=lambda x: x.get("adjusted_severity", 0.0), reverse=True)


@dataclass
class AgentOutput:
    agent_name: str
    findings: List[Dict[str, Any]]
    notes: Dict[str, Any]


class CrewRuntime:
    """Light CrewAI integration for orchestration metadata."""

    def __init__(self) -> None:
        self.enabled = bool(
            Agent
            and Crew
            and Task
            and os.getenv("OPENAI_API_KEY")
            and os.getenv("ENABLE_CREWAI_RUNTIME", "0") == "1"
        )

    def kickoff(self, summary: str) -> Dict[str, Any]:
        if not self.enabled:
            return {"mode": "deterministic", "summary": summary}
        try:
            scout = Agent(role="ScoutingAgent", goal="Rank repos", backstory="Due diligence scout")
            task = Task(description=summary, agent=scout, expected_output="A concise summary")
            crew = Crew(agents=[scout], tasks=[task], verbose=False)
            _ = crew.kickoff()
            return {"mode": "crewai", "summary": summary}
        except Exception as exc:
            return {"mode": "deterministic", "summary": f"CrewAI fallback: {exc}"}


class ScoutingAgent:
    name = "ScoutingAgent"
    DOMAIN_HINTS = {
        "durable": ["temporal", "cadence", "argo workflows", "workflow engine", "orchestration"],
        "workflow": ["temporal", "cadence", "argo", "workflow"],
        "orchestration": ["temporal", "cadence", "argo workflows", "workflow engine"],
        "event": ["event-driven", "message queue", "kafka"],
        "payment": ["payment gateway", "payment processor", "checkout", "billing", "subscription", "fintech"],
        "payments": ["payment gateway", "payment processor", "checkout", "billing", "subscription", "fintech"],
        "fintech": ["payment gateway", "ledger", "billing", "pci", "fraud"],
        "provider": ["gateway", "processor", "api"],
    }
    DOMAIN_CANONICAL_REPOS = {
        "durable_execution": [
            "temporalio/temporal",
            "cadence-workflow/cadence",
            "Netflix/conductor",
            "argoproj/argo-workflows",
            "uber/cadence",
        ],
        "payments": [
            "juspay/hyperswitch",
            "killbill/killbill",
            "medusajs/medusa",
            "solidusio/solidus",
        ],
    }
    EXCLUDE_TOKENS = {"awesome", "trending", "list", "collections", "boilerplate", "example", "tutorial"}
    BLOCKLIST_TOKENS = {
        "brute",
        "cracker",
        "hacking",
        "hack",
        "phishing",
        "malware",
        "stealer",
        "ransomware",
        "ddos",
        "exploit",
        "keylogger",
        "facebook-cracker",
    }
    STOPWORDS = {
        "i",
        "want",
        "to",
        "use",
        "an",
        "a",
        "the",
        "open",
        "source",
        "for",
        "with",
        "and",
        "of",
        "on",
        "in",
        "my",
        "our",
        "we",
        "need",
        "platform",
        "provider",
        "tool",
        "tools",
        "system",
    }

    def __init__(self, github: GitHubClient) -> None:
        self.github = github
        self.min_stars = int(os.getenv("SCOUT_MIN_STARS", "200"))

    def run(self, company_profile: Dict[str, Any], requirements: str) -> Dict[str, Any]:
        constraints = build_company_constraints(company_profile)
        intent_tokens = self._intent_terms(requirements)
        queries = self._build_queries(requirements, constraints, intent_tokens)
        repos = self.github.search_repositories(
            queries,
            top_k=int(os.getenv("SCOUT_TOP_K", "25")),
            max_queries=min(int(os.getenv("SCOUT_MAX_QUERIES", "4")), 6),
            per_page=int(os.getenv("SCOUT_PER_PAGE", "15")),
            include_contributors=False,
        )
        repos = [r for r in repos if int(r.get("stargazers_count", 0) or 0) >= self.min_stars]
        repos = self._augment_with_canonical_repos(repos, requirements)
        # Progressive fallback: broaden search if strict queries yielded nothing.
        if not repos:
            broad_queries = self._build_broad_queries(requirements, constraints, intent_tokens)
            repos = self.github.search_repositories(
                broad_queries,
                top_k=int(os.getenv("SCOUT_TOP_K", "25")),
                max_queries=min(int(os.getenv("SCOUT_MAX_QUERIES", "4")), 6),
                per_page=int(os.getenv("SCOUT_PER_PAGE", "15")),
                include_contributors=False,
            )
            repos = [r for r in repos if int(r.get("stargazers_count", 0) or 0) >= self.min_stars]
            repos = self._augment_with_canonical_repos(repos, requirements)
            queries = queries + broad_queries
        # Seeded fallback for known durable/orchestration domains.
        if not repos:
            seeds = self._seeded_repositories(requirements)
            seeded = []
            for full_name in seeds:
                item = self.github.fetch_repository(full_name)
                if item:
                    seeded.append(item)
            repos = [r for r in seeded if int(r.get("stargazers_count", 0) or 0) >= self.min_stars]
        top3 = self._rank_top3(repos, constraints, requirements)
        return {
            "constraints": constraints,
            "query_pack": {"repo_queries": queries},
            "top3": top3,
            "candidate_count": len(repos),
            "min_stars": self.min_stars,
        }

    def _build_queries(self, requirements: str, constraints: Dict[str, Any], req_tokens: List[str] | None = None) -> List[str]:
        req_tokens = req_tokens or self._intent_terms(requirements)
        intent = " ".join(req_tokens[:8]) or "open source platform"
        lang = (constraints.get("primary_languages") or [None])[0]
        lang_q = f" language:{lang}" if lang else ""
        extra = []
        if constraints.get("orchestration"):
            extra.append(str(constraints["orchestration"]))
        if constraints.get("identity_provider"):
            extra.append(str(constraints["identity_provider"]))
        hints = []
        for t in req_tokens:
            hints.extend(self.DOMAIN_HINTS.get(t, []))
        hint_phrase = " ".join(list(dict.fromkeys(hints))[:5])
        plus = " ".join(extra + ([hint_phrase] if hint_phrase else []))
        exclude = "-awesome -trending -tutorial -boilerplate -example -hacking -cracker -brute -phishing -malware -exploit"
        keywords = self._extract_keyword_phrases(requirements, req_tokens)
        keyword_blob = " ".join(keywords[:6])
        queries = [
            f"{intent} {plus} in:readme stars:>{self.min_stars} pushed:>2024-01-01 {exclude}{lang_q}".strip(),
            f"{intent} {plus} in:name,description stars:>{self.min_stars} pushed:>2024-01-01 {exclude}{lang_q}".strip(),
            f"{hint_phrase or intent} in:description stars:>{max(self.min_stars, 120)} pushed:>2024-01-01 {exclude}{lang_q}".strip(),
            f"\"{keyword_blob}\" in:name,description,readme stars:>{self.min_stars} pushed:>2024-01-01 {exclude}{lang_q}".strip(),
        ]
        if any(t in {"durable", "workflow", "orchestration", "execution"} for t in req_tokens):
            queries.append(
                f"\"durable execution\" OR \"workflow orchestration\" OR temporal OR cadence OR conductor in:readme,description stars:>{self.min_stars} pushed:>2024-01-01 {exclude}{lang_q}".strip()
            )
        if any(t in {"payment", "payments", "fintech"} for t in req_tokens):
            queries.append(
                f"topic:payments OR topic:payment-gateway OR topic:fintech stars:>{self.min_stars} pushed:>2024-01-01 {lang_q}".strip()
            )
        return queries

    def _build_broad_queries(self, requirements: str, constraints: Dict[str, Any], req_tokens: List[str] | None = None) -> List[str]:
        req_tokens = req_tokens or self._intent_terms(requirements)
        lang = (constraints.get("primary_languages") or [None])[0]
        lang_q = f" language:{lang}" if lang else ""
        anchor = " ".join(req_tokens[:4]) or "workflow orchestration"
        return [
            f"{anchor} in:name,description stars:>{self.min_stars} {lang_q}".strip(),
            f"{anchor} workflow engine in:readme stars:>{self.min_stars} {lang_q}".strip(),
        ]

    def _extract_keyword_phrases(self, requirements: str, req_tokens: List[str]) -> List[str]:
        req = " ".join(req_tokens)
        phrases: List[str] = []
        if "durable" in req and "execution" in req:
            phrases.append("durable execution")
        if "workflow" in req and "orchestration" in req:
            phrases.append("workflow orchestration")
        if "order" in req and "processing" in req:
            phrases.append("order processing")
        phrases.extend(req_tokens[:8])
        return list(dict.fromkeys([p for p in phrases if p]))

    def _seeded_repositories(self, requirements: str) -> List[str]:
        req = " ".join(tokenize(requirements))
        if any(x in req for x in ["payment", "payments", "fintech", "checkout", "billing"]):
            return self.DOMAIN_CANONICAL_REPOS["payments"]
        if any(x in req for x in ["durable", "workflow", "orchestration", "execution"]):
            return self.DOMAIN_CANONICAL_REPOS["durable_execution"]
        return [
            "apache/airflow",
            "argoproj/argo-workflows",
            "kedacore/keda",
        ]

    def _augment_with_canonical_repos(self, repos: List[Dict[str, Any]], requirements: str) -> List[Dict[str, Any]]:
        req = " ".join(tokenize(requirements))
        seeds: List[str] = []
        if any(x in req for x in ["durable", "workflow", "orchestration", "execution"]):
            seeds.extend(self.DOMAIN_CANONICAL_REPOS["durable_execution"])
        if any(x in req for x in ["payment", "payments", "fintech", "checkout", "billing"]):
            seeds.extend(self.DOMAIN_CANONICAL_REPOS["payments"])
        if not seeds:
            return repos
        by_name: Dict[str, Dict[str, Any]] = {str(r.get("full_name")): r for r in repos if r.get("full_name")}
        for full_name in seeds:
            if full_name in by_name:
                continue
            item = self.github.fetch_repository(full_name)
            if item and int(item.get("stargazers_count", 0) or 0) >= self.min_stars:
                by_name[full_name] = item
        return list(by_name.values())

    def _rank_top3(self, repos: List[Dict[str, Any]], constraints: Dict[str, Any], requirements: str) -> List[Dict[str, Any]]:
        ranked = []
        req_terms = set(self._intent_terms(requirements))
        filtered = []
        allow_listy = bool(req_terms.intersection({"awesome", "list", "tutorial", "catalog"}))
        for repo in repos:
            full = (repo.get("full_name") or "").lower()
            desc = (repo.get("description") or "").lower()
            topics = " ".join((repo.get("topics") or [])).lower()
            blob = f"{full} {desc} {topics}"
            if any(tok in blob for tok in self.BLOCKLIST_TOKENS):
                continue
            if not allow_listy:
                if any(tok in full.split("/")[-1] for tok in self.EXCLUDE_TOKENS):
                    continue
                if "awesome" in desc or "curated" in desc:
                    continue
            filtered.append(repo)
        source = filtered or repos
        for repo in source:
            stars = int(repo.get("stargazers_count", 0) or 0)
            if stars < self.min_stars:
                continue
            scores = score_repo(repo)
            name = (repo.get("full_name") or "").lower()
            text = f"{name} {repo.get('description','')} {' '.join(repo.get('topics') or [])}".lower()
            overlap_terms = [t for t in req_terms if t in text]
            overlap = len(overlap_terms)
            fit = overlap / max(len(req_terms), 1)
            # Boost highly meaningful requirement terms.
            if any(t in req_terms for t in ["payment", "payments", "checkout", "billing", "gateway"]):
                payment_hits = sum(1 for t in ["payment", "payments", "checkout", "billing", "gateway", "processor", "merchant"] if t in text)
                fit = min(1.0, fit + 0.08 * payment_hits)
            if any(t in req_terms for t in ["durable", "workflow", "orchestration", "execution"]):
                durable_hits = sum(1 for t in ["durable", "workflow", "orchestration", "execution", "temporal", "cadence", "conductor"] if t in text)
                fit = min(1.0, fit + 0.09 * durable_hits)
            maturity = min(1.0, scores["activity_score"] * 0.6 + scores["adoption_score"] * 0.4)
            context_alignment = 0.7 if not constraints.get("primary_languages") else (
                1.0 if (repo.get("language") in constraints.get("primary_languages", [])) else 0.35
            )
            # Closest-match first: fit dominates; health/maturity break ties.
            final = 0.55 * fit + 0.15 * maturity + 0.1 * scores["health_score"] + 0.2 * context_alignment
            if fit < 0.15:
                final *= 0.55
            warnings = []
            if constraints.get("maturity_preference") == "Stable only" and maturity < 0.5:
                warnings.append("Maturity may be below Stable-only preference")
            if repo.get("license") in {None, "NOASSERTION"}:
                warnings.append("License not clearly declared")
            if fit < 0.2:
                warnings.append("Closest match found, but intent alignment is limited")
            pros = []
            cons = []
            if scores["activity_score"] > 0.7:
                pros.append({"text": "Recent commit activity", "citations": [citation(repo.get("html_url", ""), doc_section="GitHub API: /repos/{owner}/{repo}") ]})
            if scores["adoption_score"] > 0.7:
                pros.append({"text": "Strong adoption signal (stars/forks)", "citations": [citation(repo.get("html_url", ""), doc_section="GitHub API: /repos/{owner}/{repo}") ]})
            if fit > 0.4:
                pros.append({"text": "Strong keyword alignment with requirements", "citations": [citation(repo.get("html_url", ""), doc_section="Repository description/topics")]})
            if scores["maintenance_score"] < 0.4:
                cons.append({"text": "High open issue ratio", "citations": [citation(repo.get("html_url", ""), doc_section="GitHub API: open_issues_count")]})
            if context_alignment < 0.5:
                cons.append({"text": "Language mismatch with company stack", "citations": [citation(repo.get("html_url", ""), doc_section="GitHub API: language")]})
            if fit < 0.2:
                cons.append({"text": "Weak direct intent match", "citations": [citation(repo.get("html_url", ""), doc_section="description/topic overlap")]})
            ranked.append(
                {
                    "full_name": repo.get("full_name"),
                    "html_url": repo.get("html_url"),
                    "description": repo.get("description"),
                    "license": repo.get("license"),
                    "score_breakdown": {
                        "fit": round(fit, 4),
                        "maturity": round(maturity, 4),
                        "health": scores["health_score"],
                        "risk": scores["risk_score"],
                        "context_alignment": round(context_alignment, 4),
                    },
                    "final_score": round(final, 4),
                    "pros": pros or [{"text": "Good topical match", "citations": [citation(repo.get("html_url", ""), doc_section="metadata overview")]}],
                    "cons": cons or [{"text": "No critical weaknesses observed in metadata", "citations": [citation(repo.get("html_url", ""), doc_section="metadata overview")]}],
                    "constraint_warnings": warnings,
                    "evidence": [
                        citation(repo.get("html_url", ""), doc_section="GitHub repository metadata")
                    ],
                }
            )
        ranked.sort(key=lambda x: (x["final_score"], x["score_breakdown"]["fit"]), reverse=True)
        return ranked[:3]

    def _intent_terms(self, requirements: str) -> List[str]:
        terms = []
        for token in tokenize(requirements):
            if token in self.STOPWORDS:
                continue
            terms.append(token)
        if not terms:
            return tokenize(requirements)[:8]
        return list(dict.fromkeys(terms))


class RepoLibrarianAgent:
    name = "RepoLibrarianAgent"

    def run(
        self,
        company_profile: Dict[str, Any],
        repo_url: str,
        full_name: str,
        ref: str | None,
        index_profile: str = "standard",
        progress_cb=None,
        chunk_sink=None,
    ) -> Dict[str, Any]:
        # Signal-based evidence collection only (no full code clone/indexing).
        os.environ["MAX_FILES"] = str(int(os.getenv("INDEX_STANDARD_MAX_FILES", "50")) if index_profile == "standard" else int(os.getenv("INDEX_MINIMAL_MAX_FILES", "20")))
        collector = RepoEvidenceCollector(GitHubClient())
        collected = collector.collect(
            repo_full_name=full_name,
            repo_url=repo_url,
            ref=ref,
            progress_cb=progress_cb,
        )
        raw_evidence = collected["evidence"]
        # Compatibility adapter for downstream agents expecting chunk_text format.
        chunks = []
        for e in raw_evidence:
            chunks.append(
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
                    "evidence_id": e.get("evidence_id"),
                    "source_type": e.get("source_type"),
                    "api_ref": e.get("api_ref"),
                    "line_map": e.get("line_map"),
                    "sha": e.get("sha"),
                }
            )
        if chunk_sink and chunks:
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for ch in chunks:
                key = str(ch.get("file_path") or ch.get("source_type") or "evidence")
                grouped.setdefault(key, []).append(ch)
            for path_key, batch in grouped.items():
                chunk_sink(batch, path_key)
        files = sorted({c.get("file_path") for c in chunks if c.get("file_path")})
        repo_map = {
            "docs_files": [f for f in files if f and (str(f).lower().startswith("docs/") or str(f).lower().endswith(".md"))][:25],
            "config_files": [f for f in files if f and any(str(f).lower().endswith(x) for x in [".yaml", ".yml", ".toml", ".env"])][:25],
            "metadata_items": [c.get("source_type") for c in chunks if c.get("source_type", "").startswith("github_")][:20],
        }
        return {
            "repo_id": f"evidence_{uuid.uuid4().hex[:10]}",
            "commit_sha": collected.get("commit_or_ref"),
            "chunks": chunks,
            "raw_evidence": raw_evidence,
            "file_manifest": files,
            "repo_map": repo_map,
            "budget_usage": collected.get("budget_usage", {}),
            "repo_full_name": full_name,
            "repo_url": repo_url,
            "signals": collected.get("signals", []),
        }


def _search_evidence(chunks: List[Dict[str, Any]], keywords: List[str], k: int = 5) -> List[Dict[str, Any]]:
    scored = []
    for ch in chunks:
        text = ((ch.get("chunk_text") or ch.get("content") or "")).lower()
        file_path = (ch.get("file_path") or "").lower()
        score = sum(1 for kw in keywords if kw in text or kw in file_path)
        if score > 0:
            scored.append((score, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:k]]


def _mk_finding(
    agent_name: str,
    category: str,
    claim: str,
    confidence: str,
    base_severity: float,
    context_multiplier: float,
    rationale: str,
    citations: List[Dict[str, Any]],
    next_checks: List[str],
    owner: str,
    effort: str,
) -> Dict[str, Any]:
    return {
        "finding_id": f"f_{uuid.uuid4().hex[:10]}",
        "agent_name": agent_name,
        "category": category,
        "claim": claim,
        "confidence": confidence,
        "base_severity": round(base_severity, 3),
        "context_multiplier": round(context_multiplier, 3),
        "adjusted_severity": round(min(1.0, base_severity * context_multiplier), 3),
        "multiplier_rationale": rationale,
        "citations": citations,
        "next_checks": next_checks,
        "owner": owner,
        "effort": effort,
    }


class SecuritySupplyChainAgent:
    name = "SecuritySupplyChainAgent"

    def run(self, company_profile: Dict[str, Any], chunks: List[Dict[str, Any]], repo_url: str) -> AgentOutput:
        sec_hits = _search_evidence(chunks, ["security", "audit", "sbom", "vulnerability", "cve", "encryption"], k=6)
        missing = []
        if not _search_evidence(chunks, ["audit log", "audit"], k=1):
            missing.append("audit_logs")
        if not _search_evidence(chunks, ["sbom", "cyclonedx", "spdx"], k=1):
            missing.append("sbom")
        mult, reason = context_multiplier_for_finding(company_profile, "security", missing)
        cites = [
            citation(repo_url, s.get("file_path"), s.get("start_line"), s.get("end_line"), s.get("heading"))
            for s in sec_hits[:3]
        ]
        finding = _mk_finding(
            self.name,
            "security",
            "Security and supply-chain controls require verification for enterprise compliance.",
            "Medium" if cites else "Low",
            0.62,
            mult,
            reason,
            cites,
            ["Inspect SECURITY.md", "Search for SBOM generation instructions", "Verify audit logging capabilities"],
            owner="Security",
            effort="M",
        )
        return AgentOutput(self.name, [ensure_grounded_claim(finding)], {"matched_chunks": len(sec_hits)})


class LicenseComplianceAgent:
    name = "LicenseComplianceAgent"

    def run(self, company_profile: Dict[str, Any], chunks: List[Dict[str, Any]], repo_url: str) -> AgentOutput:
        license_hits = _search_evidence(chunks, ["license", "mit", "apache", "gpl", "agpl"], k=4)
        sec = company_profile.get("security_compliance", {})
        frameworks = sec.get("required_frameworks", [])
        mult = 1.15 if frameworks else 1.0
        cites = [
            citation(repo_url, s.get("file_path"), s.get("start_line"), s.get("end_line"), s.get("heading"))
            for s in license_hits[:2]
        ]
        finding = _mk_finding(
            self.name,
            "license",
            "License compatibility must be confirmed against internal legal policy.",
            "High" if cites else "Low",
            0.55,
            mult,
            "Compliance frameworks increase legal review priority." if frameworks else "Standard legal validation.",
            cites,
            ["Validate license obligations with Legal", "Confirm third-party dependencies' licenses"],
            owner="Legal",
            effort="S",
        )
        return AgentOutput(self.name, [ensure_grounded_claim(finding)], {"frameworks": frameworks})


class ReliabilityOpsAgent:
    name = "ReliabilityOpsAgent"

    def run(self, company_profile: Dict[str, Any], chunks: List[Dict[str, Any]], repo_url: str) -> AgentOutput:
        tech = company_profile.get("technical_environment", {})
        obs = [x.lower() for x in tech.get("observability_stack", [])]
        keywords = ["kubernetes", "helm", "docker", "prometheus", "opentelemetry", "datadog", "slo"]
        rel_hits = _search_evidence(chunks, keywords, k=6)
        missing = []
        if "datadog" in obs and not _search_evidence(chunks, ["datadog"], k=1):
            missing.append("observability")
        mult = 1.25 if missing else 1.0
        cites = [citation(repo_url, s.get("file_path"), s.get("start_line"), s.get("end_line")) for s in rel_hits[:3]]
        finding = _mk_finding(
            self.name,
            "reliability",
            "Operational readiness and runbook coverage may require additional SRE work.",
            "Medium" if cites else "Low",
            0.58,
            mult,
            "Observability stack mismatch increases reliability risk." if missing else "Baseline reliability review.",
            cites,
            ["Confirm deployment manifests", "Validate monitoring/alerting integration"],
            owner="SRE",
            effort="M",
        )
        return AgentOutput(self.name, [ensure_grounded_claim(finding)], {"missing": missing})


class ArchitectureIntegrationAgent:
    name = "ArchitectureIntegrationAgent"

    def run(self, company_profile: Dict[str, Any], chunks: List[Dict[str, Any]], repo_url: str) -> AgentOutput:
        tech = company_profile.get("technical_environment", {})
        db_terms = [str(x).lower() for x in tech.get("databases", [])]
        mq_terms = [str(x).lower() for x in tech.get("messaging", [])]
        idp = str(tech.get("identity_provider") or "").lower()
        keywords = db_terms + mq_terms + ([idp] if idp else []) + ["oauth", "saml", "postgres", "kafka", "mysql"]
        hits = _search_evidence(chunks, keywords, k=8)
        mult = 1.2 if not hits else 1.0
        cites = [citation(repo_url, s.get("file_path"), s.get("start_line"), s.get("end_line")) for s in hits[:4]]
        finding = _mk_finding(
            self.name,
            "architecture",
            "Integration surfaces require validation against company DB, messaging, and identity stack.",
            "High" if len(cites) >= 2 else "Low",
            0.6,
            mult,
            "Limited direct integration evidence found." if mult > 1.0 else "Integration touchpoints partially evidenced.",
            cites,
            ["Locate extension points/plugins", "Validate auth and persistence adapters"],
            owner="Eng",
            effort="L",
        )
        return AgentOutput(self.name, [ensure_grounded_claim(finding)], {"integration_hits": len(hits)})


class CommunityMaintenanceAgent:
    name = "CommunityMaintenanceAgent"

    def run(self, company_profile: Dict[str, Any], repo_meta: Dict[str, Any], repo_url: str) -> AgentOutput:
        risk = company_profile.get("risk_appetite", {})
        require_support = bool(risk.get("require_commercial_support"))
        stars = float(repo_meta.get("stargazers_count", 0) or 0)
        open_issues = float(repo_meta.get("open_issues_count", 0) or 0)
        pushed = repo_meta.get("pushed_at")
        base = 0.5 + (0.2 if open_issues > stars * 0.3 else 0.0)
        mult = 1.3 if require_support else 1.0
        rationale = "Commercial support requirement increases community risk." if require_support else "Standard maintenance assessment."
        finding = _mk_finding(
            self.name,
            "community",
            "Maintenance and support model should be confirmed with maintainers.",
            "Medium",
            min(base, 0.9),
            mult,
            rationale,
            [citation(repo_url, doc_section="GitHub repository metadata")],
            ["Ask maintainers about support SLAs", "Review release cadence and issue response time"],
            owner="Eng",
            effort="S",
        )
        return AgentOutput(self.name, [ensure_grounded_claim(finding)], {"stars": stars, "open_issues": open_issues, "pushed_at": pushed})


class JudgeVerifierAgent:
    name = "JudgeVerifierAgent"

    def run(
        self,
        company_profile: Dict[str, Any],
        findings: List[Dict[str, Any]],
        repo_full_name: str,
        requirements: str = "",
    ) -> Dict[str, Any]:
        normalized = [ensure_grounded_claim(dict(f)) for f in findings]
        deduped = dedupe_findings(normalized)

        for f in deduped:
            f["adjusted_severity"] = round(min(1.0, float(f["base_severity"]) * float(f["context_multiplier"])), 3)

        top_risk = max([f["adjusted_severity"] for f in deduped], default=0.0)
        decision = "GO"
        if top_risk >= 0.8:
            decision = "NO_GO"
        elif top_risk >= 0.55:
            decision = "CONDITIONAL_GO"

        next_steps = []
        for f in deduped[:8]:
            next_steps.append(
                {
                    "owner": f.get("owner", "Eng"),
                    "task": f"Address: {f.get('claim')}",
                    "effort": f.get("effort", "M"),
                    "priority": "P1" if f.get("adjusted_severity", 0) >= 0.75 else ("P2" if f.get("adjusted_severity", 0) >= 0.5 else "P3"),
                }
            )
        tech = company_profile.get("technical_environment", {})
        sec = company_profile.get("security_compliance", {})
        req_tokens = set(tokenize(requirements))
        db_targets = [str(x) for x in tech.get("databases", [])[:2]]
        mq_targets = [str(x) for x in tech.get("messaging", [])[:2]]
        idp = str(tech.get("identity_provider") or "our IdP")

        maintainer_questions = [
            f"For our requirement '{requirements or 'enterprise adoption'}', which production references most closely match this use case?",
            "What is your release and deprecation policy?",
            "Which integrations are officially supported and tested?",
        ]
        if db_targets or mq_targets:
            maintainer_questions.append(
                f"Do you have validated integration patterns for {', '.join(db_targets + mq_targets)}?"
            )
        maintainer_questions.append(f"How should we integrate authentication with {idp}?")
        if sec.get("audit_logs_required"):
            maintainer_questions.append("Where is audit logging documented and what events are guaranteed?")
        if sec.get("sbom_required"):
            maintainer_questions.append("Do you publish SBOM artifacts, and how do you handle vulnerability disclosure timelines?")
        if {"durable", "workflow", "orchestration", "execution"} & req_tokens:
            maintainer_questions.append(
                "How do you guarantee workflow durability, idempotency, and replay safety during partial outages?"
            )
        if {"payment", "payments", "checkout", "billing"} & req_tokens:
            maintainer_questions.append(
                "What payment reliability controls exist (idempotency keys, reconciliation, retry safety, and double-charge prevention)?"
            )
        # Keep a concise prioritized set.
        maintainer_questions = list(dict.fromkeys(maintainer_questions))[:8]
        report = {
            "decision": decision,
            "repo": repo_full_name,
            "generated_at": now_iso(),
            "risk_register": deduped,
            "prioritized_next_steps": next_steps,
            "maintainer_questions": maintainer_questions,
        }
        validate(instance=report, schema=REPORT_SCHEMA)
        return report


def report_to_markdown(report: Dict[str, Any]) -> str:
    lines = [
        f"# OSS Due Diligence Report: {report.get('repo', '')}",
        "",
        f"**Decision:** {report.get('decision')}",
        "",
        "## Risk Register",
    ]
    for f in report.get("risk_register", []):
        lines.append(f"- **[{f.get('category')}]** {f.get('claim')}")
        lines.append(
            f"  - Severity: base={f.get('base_severity')} x context={f.get('context_multiplier')} => adjusted={f.get('adjusted_severity')}"
        )
        lines.append(f"  - Rationale: {f.get('multiplier_rationale')}")
        lines.append(f"  - Confidence: {f.get('confidence')}")
        cites = f.get("citations", [])
        if cites:
            for c in cites[:2]:
                lines.append(
                    f"  - Citation: {c.get('repo_url')} | {c.get('file_path') or c.get('doc_section')}:{c.get('start_line')}-{c.get('end_line')}"
                )
        else:
            lines.append("  - Citation: Not confirmed from repo evidence")
        checks = f.get("next_checks", [])
        if checks:
            lines.append(f"  - Next checks: {', '.join(checks[:3])}")
    lines.append("")
    lines.append("## Prioritized Next Steps")
    for s in report.get("prioritized_next_steps", []):
        lines.append(f"- [{s.get('priority')}] {s.get('owner')}: {s.get('task')} (effort={s.get('effort')})")
    lines.append("")
    lines.append("## Maintainer Questions")
    for q in report.get("maintainer_questions", []):
        lines.append(f"- {q}")
    return "\n".join(lines)
