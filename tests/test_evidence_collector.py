from __future__ import annotations

from src.evidence_collector import RepoEvidenceCollector


class FakeGitHub:
    def __init__(self):
        self.token = "x"

    def fetch_repository(self, full_name):
        return {
            "full_name": full_name,
            "html_url": f"https://github.com/{full_name}",
            "description": "Durable workflow orchestration with Kubernetes and OAuth",
            "topics": ["workflow", "orchestration"],
            "language": "Go",
            "stargazers_count": 1000,
            "forks_count": 100,
            "open_issues_count": 20,
            "watchers_count": 80,
            "license": "MIT",
            "pushed_at": "2026-02-01T00:00:00Z",
            "created_at": "2022-01-01T00:00:00Z",
        }

    def fetch_repo_releases(self, full_name, per_page=4):
        return [{"title": "v1.2.0", "body": "release notes", "url": f"https://github.com/{full_name}/releases/tag/v1.2.0"}]

    def fetch_repo_recent_issues(self, full_name, per_page=10):
        return [{"id": 1, "title": "Audit logs?", "body": "Need audit logs", "labels": ["security"], "url": f"https://github.com/{full_name}/issues/1"}]

    def fetch_repo_discussions(self, full_name, first=5):
        return [{"id": "d1", "title": "Support roadmap", "body": "enterprise support", "category": "Q&A", "url": f"https://github.com/{full_name}/discussions/1"}]

    def _request(self, method, path, params=None):
        if path.endswith("/contents/"):
            return [
                {"type": "file", "name": "README.md", "path": "README.md"},
                {"type": "file", "name": "SECURITY.md", "path": "SECURITY.md"},
                {"type": "file", "name": "LICENSE", "path": "LICENSE"},
            ]
        if path.endswith("/contents/.github/workflows"):
            return [{"type": "file", "path": ".github/workflows/ci.yml"}]
        if path.endswith("/contents/docs"):
            return [{"type": "file", "path": "docs/intro.md"}]
        if "/contents/" in path:
            p = path.split("/contents/")[1]
            content = "IyBUaXRsZQoKIyMgU2VjdXJpdHkKQXVkaXQgbG9ncyBhbmQgT0F1dGg="
            return {"encoding": "base64", "content": content, "path": p, "html_url": f"https://github.com/acme/repo/blob/main/{p}", "sha": "abc"}
        return {}


def test_collector_respects_caps_and_allowlist(monkeypatch):
    c = RepoEvidenceCollector(FakeGitHub())
    c.max_files = 3
    c.max_bytes = 100000
    out = c.collect("acme/repo", "https://github.com/acme/repo")
    file_evidence = [e for e in out["evidence"] if e.get("source_type") in {"github_file", "github_workflow"}]
    assert len(file_evidence) <= 3
    assert out["budget_usage"]["files_fetched"] <= 3
    assert all(
        (not e.get("file_path")) or e["file_path"].endswith((".md", ".yml", ".yaml", ".toml", ".env")) or "Dockerfile" in e["file_path"] or e["file_path"] == "LICENSE"
        for e in file_evidence
    )


def test_line_number_citation_generation():
    c = RepoEvidenceCollector(FakeGitHub())
    ev = {
        "repo_url": "https://github.com/acme/repo",
        "file_path": "README.md",
        "url": "https://github.com/acme/repo/blob/main/README.md",
        "source_type": "github_file",
        "api_ref": "/repos/acme/repo/contents/README.md",
        "retrieved_at": "2026-02-28T00:00:00Z",
    }
    cite = c.make_citation(ev, 10, 12, "example")
    assert cite["start_line"] == 10
    assert cite["end_line"] == 12
    assert cite["file_path"] == "README.md"

