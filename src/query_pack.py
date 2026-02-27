"""Generate GitHub query packs from free-text domains."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

STOPWORDS = {
    "and",
    "the",
    "for",
    "with",
    "from",
    "into",
    "using",
    "open",
    "source",
    "project",
    "projects",
    "tool",
    "tools",
}

SEED_TOPICS = {
    "rag": ["rag", "llm", "retrieval-augmented-generation"],
    "vector": ["vector-database", "embeddings", "ann"],
    "mlops": ["mlops", "machine-learning", "model-serving"],
    "observability": ["observability", "opentelemetry", "monitoring"],
    "dbt": ["dbt", "analytics-engineering", "data-transformation"],
    "security": ["security", "appsec", "threat-detection"],
    "evaluation": ["evaluation", "benchmark", "testing"],
}


@dataclass
class QueryPack:
    """Structured query inputs for GitHub repository search."""

    domain: str
    language: str | None
    keywords: List[str]
    topics: List[str]
    queries: List[str]


def _tokenize_domain(domain: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9\-\+]+", domain.lower())
    filtered = [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1]
    deduped: List[str] = []
    for token in filtered:
        if token not in deduped:
            deduped.append(token)
    return deduped[:6]


def _seed_topics(tokens: List[str]) -> List[str]:
    topics: List[str] = []
    for token in tokens:
        if token in SEED_TOPICS:
            for topic in SEED_TOPICS[token]:
                if topic not in topics:
                    topics.append(topic)
    return topics[:6]


def generate_query_pack(domain: str, language: str | None = None) -> QueryPack:
    """Create 3 GitHub query templates with optional language filter."""
    tokens = _tokenize_domain(domain)
    keywords = tokens[:6] if tokens else ["opensource"]
    topics = _seed_topics(keywords)

    keyword_expr = " ".join(keywords[:4])
    language_suffix = f" language:{language}" if language else ""

    query_a = (
        f"{keyword_expr} in:readme pushed:>2024-01-01 stars:>50{language_suffix}"
    )
    query_b = (
        f"{keyword_expr} in:name,description pushed:>2024-01-01 stars:>50{language_suffix}"
    )

    if topics:
        topic_expr = " OR ".join([f"topic:{t}" for t in topics[:2]])
    else:
        topic_expr = " ".join([f"{k}" for k in keywords[:2]])
    query_c = f"{topic_expr} stars:>50 pushed:>2024-01-01{language_suffix}"

    queries = [query_a, query_b, query_c]
    if len(keywords) >= 5:
        query_d = (
            f"{keywords[0]} {keywords[-1]} in:description stars:>50 pushed:>2024-01-01"
            f"{language_suffix}"
        )
        queries.append(query_d)

    return QueryPack(
        domain=domain,
        language=language,
        keywords=keywords[:6],
        topics=topics,
        queries=queries[:5],
    )
