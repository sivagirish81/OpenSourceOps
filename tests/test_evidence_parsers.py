from __future__ import annotations

from src.evidence_parsers import parse_ci_workflow, parse_readme_docs, parse_security


def _ev():
    return {"evidence_id": "ev1"}


def _cite(e, s=None, en=None):
    return {"file_path": "README.md", "start_line": s, "end_line": en}


def test_readme_parser_extracts_features_with_citations():
    text = "This project runs on Kubernetes and supports OAuth with Prometheus metrics."
    sigs = parse_readme_docs(text, _ev(), _cite)
    names = {s["signal_name"] for s in sigs}
    assert "deployment_kubernetes" in names
    assert "auth_oauth_oidc" in names
    assert "obs_prometheus" in names
    assert all(s["citations"] for s in sigs)


def test_security_parser_extracts_reporting_signals():
    text = "Please report vulnerabilities to security@example.com. Response timeline within 7 days."
    sigs = parse_security(text, _ev(), _cite)
    by = {s["signal_name"]: s["value"] for s in sigs}
    assert by["security_contact_present"] is True
    assert by["disclosure_timeline_present"] is True


def test_ci_parser_extracts_ci_posture():
    text = "jobs: test, lint, codeql, dependency-review"
    sigs = parse_ci_workflow(text, _ev(), _cite)
    by = {s["signal_name"]: s["value"] for s in sigs}
    assert by["ci_tests_present"] is True
    assert by["ci_lint_present"] is True
    assert by["ci_sast_present"] is True

