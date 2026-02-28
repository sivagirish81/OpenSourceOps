import os

from src.github_client import GitHubClient
from src.query_pack import generate_query_pack


def test_query_pack_generation():
    pack = generate_query_pack(
        domain="RAG evaluation",
        intent="Need robust evaluation pipeline for enterprise adoption",
        constraints={"language": "Python"},
        persona="Adopter",
    )
    assert pack.domain == "RAG evaluation"
    assert len(pack.queries) >= 3
    assert any("pushed:>2024-01-01" in q for q in pack.queries)


def test_github_token_presence_or_skip():
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        assert True, "Skipping live GitHub calls; token not configured"
        return
    client = GitHubClient(token=token)
    assert client.is_configured() is True
