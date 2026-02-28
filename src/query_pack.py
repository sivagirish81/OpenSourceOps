"""Generate GitHub query packs from domain, intent, constraints, and optional feedback."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

STOPWORDS = {
    "and", "the", "for", "with", "from", "into", "using", "open", "source",
    "project", "projects", "tool", "tools", "build", "want", "need", "that",
    "server", "servers", "deploy", "deployment", "running", "without", "issue",
    "issues", "worry", "about", "long", "not", "or",
}

SEED_TOPICS = {
    "rag": ["rag", "llm", "retrieval-augmented-generation"],
    "vector": ["vector-database", "embeddings", "ann"],
    "mlops": ["mlops", "machine-learning", "model-serving"],
    "observability": ["observability", "opentelemetry", "monitoring"],
    "dbt": ["dbt", "analytics-engineering", "data-transformation"],
    "security": ["security", "appsec", "threat-detection"],
    "evaluation": ["evaluation", "benchmark", "testing"],
    "durable": ["durable-execution", "workflow-orchestration", "distributed-systems"],
    "execution": ["workflow-orchestration", "job-queue", "distributed-systems"],
    "workflow": ["workflow-engine", "orchestration", "state-machine"],
    "orchestration": ["orchestration", "workflow-engine", "state-machine"],
    "cadence": ["cadence-workflow", "workflow-orchestration"],
    "temporal": ["temporal", "workflow-orchestration", "durable-execution"],
}


@dataclass
class QueryPack:
    """Structured query inputs for GitHub repository search."""

    domain: str
    intent: str
    language: str | None
    keywords: List[str]
    topics: List[str]
    queries: List[str]
    ranking_weights: Dict[str, float]


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9\-\+]+", (text or "").lower())
    filtered = [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1]
    out: List[str] = []
    for tok in filtered:
        if tok not in out:
            out.append(tok)
    return out


def _seed_topics(tokens: List[str]) -> List[str]:
    topics: List[str] = []
    for token in tokens:
        if token in SEED_TOPICS:
            for topic in SEED_TOPICS[token]:
                if topic not in topics:
                    topics.append(topic)
    return topics[:8]


def default_ranking_weights(persona: str) -> Dict[str, float]:
    if persona == "Contributor":
        return {"health": 0.3, "intent": 0.4, "community": 0.3}
    return {"health": 0.45, "intent": 0.35, "risk": 0.2}


def generate_query_pack(
    domain: str,
    intent: str,
    constraints: Dict[str, Any] | None = None,
    persona: str = "Adopter",
    feedback_text: str | None = None,
) -> QueryPack:
    """Create query templates from domain + intent + constraints (+ feedback)."""
    constraints = constraints or {}
    language = constraints.get("language") or None
    keyword_limit = 6

    tokens = _tokenize(" ".join([domain, intent, feedback_text or ""]))
    keywords = tokens[:keyword_limit] if tokens else ["opensource"]
    topics = _seed_topics(keywords)

    keyword_expr = " ".join(keywords[:4])
    language_suffix = f" language:{language}" if language else ""

    base_filters = f"stars:>50 pushed:>2024-01-01 archived:false fork:false{language_suffix}"
    queries = [
        f"{keyword_expr} in:readme {base_filters}",
        f"{keyword_expr} in:name,description {base_filters}",
        f"{keyword_expr} in:name,description,readme {base_filters}",
    ]

    if topics:
        topic_expr = " OR ".join([f"topic:{t}" for t in topics[:2]])
        queries.append(f"{topic_expr} {base_filters}")
    else:
        queries.append(f"{keyword_expr} {base_filters}")

    if len(keywords) >= 5:
        queries.append(
            f"{keywords[0]} {keywords[-1]} in:description {base_filters}"
        )

    return QueryPack(
        domain=domain,
        intent=intent,
        language=language,
        keywords=keywords,
        topics=topics,
        queries=queries[:6],
        ranking_weights=default_ranking_weights(persona),
    )
