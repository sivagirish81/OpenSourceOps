"""Deterministic, explainable scoring functions for repository and issue ranking."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, Iterable, List


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Bound a numeric value to [low, high]."""
    return max(low, min(high, value))


def normalize_log(value: float, cap: float = 100_000.0) -> float:
    """Log normalization with cap for long-tail GitHub counts."""
    if value <= 0:
        return 0.0
    return clamp(math.log1p(value) / math.log1p(cap))


def days_since(timestamp: str | None) -> float:
    """Compute age in days from ISO timestamp string."""
    if not timestamp:
        return 9999.0
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    return max((now - parsed).total_seconds() / 86_400.0, 0.0)


def score_repo(repo: Dict) -> Dict[str, float]:
    """Compute deterministic repo quality and risk scores."""
    stars = float(repo.get("stargazers_count", 0) or 0)
    forks = float(repo.get("forks_count", 0) or 0)
    open_issues = float(repo.get("open_issues_count", 0) or 0)
    watchers = float(repo.get("watchers_count", 0) or 0)
    pushed_at = repo.get("pushed_at")

    activity_score = clamp(1 - (days_since(pushed_at) / 120.0))
    adoption_score = normalize_log(stars) * 0.7 + normalize_log(forks) * 0.3
    maintenance_score = clamp(1 - (open_issues / (stars + 1.0)))
    community_score = normalize_log(watchers + 1.0)

    health_score = (
        0.35 * activity_score
        + 0.35 * adoption_score
        + 0.2 * maintenance_score
        + 0.1 * community_score
    )
    risk_score = 1 - (
        0.5 * activity_score + 0.3 * maintenance_score + 0.2 * community_score
    )

    return {
        "activity_score": round(activity_score, 4),
        "adoption_score": round(adoption_score, 4),
        "maintenance_score": round(maintenance_score, 4),
        "community_score": round(community_score, 4),
        "health_score": round(clamp(health_score), 4),
        "risk_score": round(clamp(risk_score), 4),
    }


def score_issues(issues: Iterable[Dict]) -> List[Dict]:
    """Compute simple contributor-centric issue ranking metrics."""
    scored: List[Dict] = []
    for issue in issues:
        labels = [lbl.lower() for lbl in issue.get("labels", [])]
        comments = float(issue.get("comments", 0) or 0)
        age_days = days_since(issue.get("created_at"))

        impact_score = clamp(0.6 * (1 - clamp(age_days / 120.0)) + 0.4 * normalize_log(comments + 1, cap=500))
        difficulty_score = clamp(
            0.2
            + (0.5 if "good first issue" in labels or "good-first-issue" in labels else 0.0)
            + (0.2 if "help-wanted" in labels else 0.0)
            - (0.2 if "bug" in labels else 0.0)
        )
        reputation_score = clamp(0.55 * impact_score + 0.45 * (1 - difficulty_score))

        scored.append(
            {
                "issue_id": issue.get("id"),
                "impact_score": round(impact_score, 4),
                "difficulty_score": round(difficulty_score, 4),
                "reputation_score": round(reputation_score, 4),
            }
        )
    return scored
