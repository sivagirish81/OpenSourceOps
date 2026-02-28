"""Deterministic signal extractors for lightweight repo evidence indexing."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def _line_range_for_pattern(text: str, pattern: str) -> Tuple[int | None, int | None]:
    lines = text.splitlines()
    rx = re.compile(pattern, re.IGNORECASE)
    for idx, ln in enumerate(lines, start=1):
        if rx.search(ln):
            return idx, idx
    return None, None


def _mk_signal(
    category: str,
    name: str,
    value: Any,
    evidence_id: str,
    citation: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "category": category,
        "signal_name": name,
        "value": value,
        "evidence_id": evidence_id,
        "citations": [citation],
    }


def parse_readme_docs(content: str, evidence: Dict[str, Any], make_citation) -> List[Dict[str, Any]]:
    checks = {
        "deployment_kubernetes": r"\bkubernetes\b|\bk8s\b",
        "deployment_helm": r"\bhelm\b|chart\.yaml",
        "deployment_docker": r"\bdocker\b|docker-compose",
        "integration_postgres": r"\bpostgres\b|\bpostgresql\b",
        "integration_mysql": r"\bmysql\b",
        "integration_redis": r"\bredis\b",
        "integration_kafka": r"\bkafka\b",
        "auth_oauth_oidc": r"\boauth\b|\boidc\b",
        "auth_saml": r"\bsaml\b",
        "auth_ldap": r"\bldap\b",
        "obs_prometheus": r"\bprometheus\b",
        "obs_otel": r"\bopentelemetry\b|\botel\b",
        "obs_datadog": r"\bdatadog\b",
        "ha_claim": r"high availability|multi[- ]region|horizontal scal",
        "maturity_language": r"production[- ]ready|beta|experimental",
    }
    out: List[Dict[str, Any]] = []
    for name, pat in checks.items():
        s, e = _line_range_for_pattern(content, pat)
        if s is None:
            continue
        out.append(
            _mk_signal(
                "docs",
                name,
                True,
                evidence["evidence_id"],
                make_citation(evidence, s, e),
            )
        )
    return out


def parse_security(content: str, evidence: Dict[str, Any], make_citation) -> List[Dict[str, Any]]:
    items = {
        "vuln_reporting_process": r"report|vulnerability|security issue",
        "security_contact_present": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "disclosure_timeline_present": r"timeline|response time|within \d+ days",
    }
    out = []
    for name, pat in items.items():
        s, e = _line_range_for_pattern(content, pat)
        out.append(
            _mk_signal(
                "security",
                name,
                bool(s),
                evidence["evidence_id"],
                make_citation(evidence, s, e) if s else make_citation(evidence),
            )
        )
    return out


def parse_license(content: str, evidence: Dict[str, Any], make_citation) -> List[Dict[str, Any]]:
    types = {
        "MIT": r"\bMIT License\b",
        "Apache-2.0": r"Apache License|Version 2\.0",
        "GPL": r"GNU GENERAL PUBLIC LICENSE",
        "AGPL": r"AFFERO GENERAL PUBLIC LICENSE",
        "MPL": r"Mozilla Public License",
        "BSD": r"BSD",
    }
    detected = "Unknown"
    line = None
    for name, pat in types.items():
        s, _ = _line_range_for_pattern(content, pat)
        if s:
            detected = name
            line = s
            break
    notice_s, notice_e = _line_range_for_pattern(content, r"\bNOTICE\b")
    return [
        _mk_signal("license", "license_text_type", detected, evidence["evidence_id"], make_citation(evidence, line, line)),
        _mk_signal("license", "notice_present", bool(notice_s), evidence["evidence_id"], make_citation(evidence, notice_s, notice_e)),
    ]


def parse_ci_workflow(content: str, evidence: Dict[str, Any], make_citation) -> List[Dict[str, Any]]:
    checks = {
        "ci_tests_present": r"\b(test|pytest|go test|unit[-_ ]test)\b",
        "ci_lint_present": r"\b(lint|flake8|golangci|eslint|ruff)\b",
        "ci_build_present": r"\b(build|compile|docker build)\b",
        "ci_sast_present": r"\b(codeql|semgrep|sast)\b",
        "ci_dep_scan_present": r"\b(dependabot|dependency-review|trivy|grype)\b",
    }
    out = []
    for name, pat in checks.items():
        s, e = _line_range_for_pattern(content, pat)
        out.append(_mk_signal("ci", name, bool(s), evidence["evidence_id"], make_citation(evidence, s, e)))
    return out


def parse_deploy_config(content: str, evidence: Dict[str, Any], make_citation) -> List[Dict[str, Any]]:
    checks = {
        "docker_user_non_root": r"^\s*USER\s+\S+",
        "docker_healthcheck": r"^\s*HEALTHCHECK",
        "exposes_ports": r"\bEXPOSE\b|containerPort|servicePort|port:",
        "k8s_readiness_probe": r"readinessProbe",
        "k8s_liveness_probe": r"livenessProbe",
        "secrets_env_pattern": r"env:|ENV\s+\w+|secret|valueFrom",
    }
    out = []
    for name, pat in checks.items():
        s, e = _line_range_for_pattern(content, pat)
        out.append(_mk_signal("ops", name, bool(s), evidence["evidence_id"], make_citation(evidence, s, e)))
    return out


def parse_release_changelog(content: str, evidence: Dict[str, Any], make_citation) -> List[Dict[str, Any]]:
    changelog_s, changelog_e = _line_range_for_pattern(content, r"changelog|release notes")
    migration_s, migration_e = _line_range_for_pattern(content, r"migration|upgrade")
    semver_s, semver_e = _line_range_for_pattern(content, r"\bv?\d+\.\d+\.\d+\b")
    return [
        _mk_signal("release", "changelog_present", bool(changelog_s), evidence["evidence_id"], make_citation(evidence, changelog_s, changelog_e)),
        _mk_signal("release", "migration_guide_present", bool(migration_s), evidence["evidence_id"], make_citation(evidence, migration_s, migration_e)),
        _mk_signal("release", "semver_pattern_present", bool(semver_s), evidence["evidence_id"], make_citation(evidence, semver_s, semver_e)),
    ]

