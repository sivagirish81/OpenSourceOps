from __future__ import annotations

from src.due_diligence_agents import (
    JudgeVerifierAgent,
    context_multiplier_for_finding,
    dedupe_findings,
    ensure_grounded_claim,
)


def _profile() -> dict:
    return {
        "security_compliance": {
            "required_frameworks": ["SOC2"],
            "audit_logs_required": True,
            "sbom_required": True,
        },
        "risk_appetite": {
            "risk_tolerance": "Low",
            "require_commercial_support": True,
        },
    }


def test_missing_evidence_enforces_not_confirmed_phrase():
    finding = {
        "claim": "Audit log support exists",
        "confidence": "High",
        "citations": [],
    }
    out = ensure_grounded_claim(finding)
    assert out["claim"] == "Not confirmed from repo evidence"
    assert out["confidence"] == "Low"
    assert isinstance(out["next_checks"], list)


def test_context_multiplier_soc2_increases_severity():
    m, rationale = context_multiplier_for_finding(
        _profile(),
        "security",
        missing_controls=["audit_logs", "sbom"],
    )
    assert m > 1.0
    assert "Audit logs are required" in rationale


def test_deduplication_keeps_highest_adjusted_severity():
    findings = [
        {
            "claim": "Missing SBOM guidance",
            "category": "security",
            "owner": "Security",
            "adjusted_severity": 0.4,
        },
        {
            "claim": "Missing sbom guidance",
            "category": "security",
            "owner": "Security",
            "adjusted_severity": 0.8,
        },
    ]
    out = dedupe_findings(findings)
    assert len(out) == 1
    assert out[0]["adjusted_severity"] == 0.8


def test_judge_report_schema_validation_and_decision():
    judge = JudgeVerifierAgent()
    findings = [
        {
            "finding_id": "f1",
            "agent_name": "SecuritySupplyChainAgent",
            "category": "security",
            "claim": "Critical compliance gap found",
            "confidence": "High",
            "citations": [{"repo_url": "https://github.com/a/b", "doc_section": "SECURITY"}],
            "base_severity": 0.9,
            "context_multiplier": 1.0,
            "adjusted_severity": 0.9,
            "multiplier_rationale": "critical",
            "next_checks": [],
            "owner": "Security",
            "effort": "M",
        }
    ]
    report = judge.run(_profile(), findings, "owner/repo")
    assert report["decision"] in {"GO", "CONDITIONAL_GO", "NO_GO"}
    assert isinstance(report["risk_register"], list)
    assert isinstance(report["prioritized_next_steps"], list)
    assert isinstance(report["maintainer_questions"], list)

