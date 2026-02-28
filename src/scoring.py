"""Deterministic scoring functions for repositories and contribution opportunities."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, Iterable, List


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def normalize_log(value: float, cap: float = 100_000.0) -> float:
    if value <= 0:
        return 0.0
    return clamp(math.log1p(value) / math.log1p(cap))


def days_since(timestamp: str | None) -> float:
    if not timestamp:
        return 9999.0
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    return max((now - parsed).total_seconds() / 86_400.0, 0.0)


def keyword_overlap_score(text: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    lower = (text or "").lower()
    matched = sum(1 for k in keywords if k.lower() in lower)
    return clamp(matched / max(len(keywords), 1))


def score_repo(repo: Dict, intent_keywords: List[str]) -> Dict[str, float]:
    stars = float(repo.get("stargazers_count", 0) or 0)
    forks = float(repo.get("forks_count", 0) or 0)
    open_issues = float(repo.get("open_issues_count", 0) or 0)
    watchers = float(repo.get("watchers_count", 0) or 0)
    pushed_at = repo.get("pushed_at")

    activity_score = clamp(1 - (days_since(pushed_at) / 120.0))
    adoption_score = normalize_log(stars) * 0.7 + normalize_log(forks) * 0.3
    maintenance_score = clamp(1 - (open_issues / (stars + 1.0)))
    community_score = normalize_log(watchers + 1.0)
    intent_match_score = keyword_overlap_score(
        " ".join([
            str(repo.get("full_name", "")),
            str(repo.get("description", "")),
            " ".join(repo.get("topics", []) or []),
        ]),
        intent_keywords,
    )

    health_score = (
        0.35 * activity_score
        + 0.35 * adoption_score
        + 0.2 * maintenance_score
        + 0.1 * community_score
    )
    risk_score = 1 - (
        0.5 * activity_score + 0.3 * maintenance_score + 0.2 * community_score
    )
    hint_boost = 0.08 if repo.get("feedback_hint_match") else 0.0
    final_score = clamp(0.45 * health_score + 0.35 * intent_match_score + 0.2 * (1 - risk_score) + hint_boost)

    return {
        "intent_match_score": round(intent_match_score, 4),
        "activity_score": round(activity_score, 4),
        "adoption_score": round(adoption_score, 4),
        "maintenance_score": round(maintenance_score, 4),
        "community_score": round(community_score, 4),
        "health_score": round(clamp(health_score), 4),
        "risk_score": round(clamp(risk_score), 4),
        "final_score": round(final_score, 4),
    }


def _label_weight(labels: List[str]) -> float:
    labels_l = [l.lower() for l in labels]
    if any(k in labels_l for k in ["feature", "enhancement", "feature-request"]):
        return 1.0
    if any(k in labels_l for k in ["good-first-issue", "good first issue", "help-wanted"]):
        return 0.95
    if "bug" in labels_l:
        return 0.7
    if any(k in labels_l for k in ["docs", "documentation"]):
        return 0.5
    return 0.6


def _difficulty_bucket(ease_score: float) -> str:
    if ease_score >= 0.75:
        return "Easy"
    if ease_score >= 0.45:
        return "Medium"
    return "Hard"


def score_opportunities(items: Iterable[Dict], intent_keywords: List[str]) -> List[Dict]:
    """Score combined issue/discussion opportunities for contributor mode."""
    results: List[Dict] = []
    for item in items:
        comments = float(item.get("comments", 0) or 0)
        recency = clamp(1 - (days_since(item.get("updated_at")) / 90.0))
        engagement = clamp(0.55 * normalize_log(comments + 1, cap=300) + 0.45 * recency)

        body = item.get("body", "") or ""
        title = item.get("title", "") or ""
        text = f"{title}\n{body}"
        relevance = keyword_overlap_score(text, intent_keywords)

        scope_keywords = ["acceptance criteria", "steps", "expected", "reproduce", "proposal"]
        scope_hint = sum(1 for k in scope_keywords if k in text.lower())
        scope_score = clamp(0.45 * normalize_log(len(body), cap=2500) + 0.55 * clamp(scope_hint / 3.0))

        labels = item.get("labels", []) or []
        ease_score = clamp(0.6 * _label_weight(labels) + 0.4 * (1 - normalize_log(comments + 1, cap=200)))
        item_type = (item.get("item_type") or "ISSUE").upper()
        type_factor = 1.0 if item_type == "ISSUE" else 0.55
        code_signal = 1.0 if any(k in text.lower() for k in ["implement", "api", "backend", "frontend", "tests", "refactor"]) else 0.75
        final_score = clamp((0.35 * engagement + 0.3 * relevance + 0.2 * scope_score + 0.15 * ease_score) * type_factor * code_signal)

        difficulty = _difficulty_bucket(ease_score)
        next_action = (
            "Comment with a scoped implementation plan and ask maintainer for alignment."
            if item.get("item_type") == "DISCUSSION"
            else "Acknowledge issue, propose approach, and request assignment/feedback."
        )

        why = (
            f"High signal due to engagement={engagement:.2f}, relevance={relevance:.2f}, "
            f"scope={scope_score:.2f}."
        )

        results.append(
            {
                **item,
                "engagement_score": round(engagement, 4),
                "scope_score": round(scope_score, 4),
                "relevance_score": round(relevance, 4),
                "ease_score": round(ease_score, 4),
                "final_score": round(final_score, 4),
                "difficulty": difficulty,
                "why": why,
                "suggested_next_action": next_action,
                "contact_handles": sorted({h for h in [item.get("author_login"), *(item.get("assignees") or [])] if h}),
            }
        )

    results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    return results
