"""GitHub REST/GraphQL client for discovery, opportunities, and grounded repo indexing."""

from __future__ import annotations

import os
import re
from base64 import b64decode
from typing import Any, Dict, List, Tuple

import requests


class GitHubClient:
    """Authenticated GitHub API helper with call caps for rate-limit safety."""

    BASE_URL = "https://api.github.com"

    def __init__(self, token: str | None = None, timeout: int = 20) -> None:
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "User-Agent": "OpenSourceOps/1.0",
            }
        )
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})

    def is_configured(self) -> bool:
        return bool(self.token)

    def _request(self, method: str, path: str, params: Dict | None = None) -> Dict | List:
        url = f"{self.BASE_URL}{path}"
        response = self.session.request(method, url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def search_topics(self, query: str, per_page: int = 8) -> List[str]:
        """Use GitHub topic search to enrich repository topic qualifiers."""
        try:
            payload = self._request(
                "GET",
                "/search/topics",
                {"q": query, "per_page": min(per_page, 30)},
            )
            items = payload.get("items", []) if isinstance(payload, dict) else []
            names = [i.get("name") for i in items if i.get("name")]
            return list(dict.fromkeys(names))[:per_page]
        except Exception:
            return []

    def _graphql(self, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/graphql"
        response = self.session.post(url, json={"query": query, "variables": variables}, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            raise RuntimeError(str(payload["errors"]))
        return payload.get("data", {})

    @staticmethod
    def _parse_last_page(link_header: str | None) -> int:
        if not link_header:
            return 0
        match = re.search(r"[?&]page=(\d+)>; rel=\"last\"", link_header)
        return int(match.group(1)) if match else 1

    def get_contributors_count(self, full_name: str) -> int:
        try:
            url = f"{self.BASE_URL}/repos/{full_name}/contributors"
            resp = self.session.get(url, params={"per_page": 1, "anon": "true"}, timeout=self.timeout)
            if resp.status_code >= 400:
                return 0
            return self._parse_last_page(resp.headers.get("Link"))
        except Exception:
            return 0

    def search_repositories(
        self,
        queries: List[str],
        top_k: int = 25,
        max_queries: int = 3,
        per_page: int = 15,
        include_contributors: bool = False,
    ) -> List[Dict]:
        merged: Dict[str, Dict] = {}
        for query in queries[: max(1, min(max_queries, 6))]:
            try:
                payload = self._request(
                    "GET",
                    "/search/repositories",
                    {"q": query, "sort": "stars", "order": "desc", "per_page": min(per_page, 30)},
                )
                for item in payload.get("items", []):
                    full_name = item.get("full_name")
                    if not full_name or full_name in merged:
                        continue
                    merged[full_name] = {
                        "full_name": full_name,
                        "html_url": item.get("html_url"),
                        "description": item.get("description"),
                        "topics": item.get("topics", []),
                        "language": item.get("language"),
                        "stargazers_count": item.get("stargazers_count", 0),
                        "forks_count": item.get("forks_count", 0),
                        "open_issues_count": item.get("open_issues_count", 0),
                        "watchers_count": item.get("watchers_count", 0),
                        "license": (item.get("license") or {}).get("spdx_id"),
                        "pushed_at": item.get("pushed_at"),
                        "created_at": item.get("created_at"),
                    }
            except Exception:
                continue

        repos = list(merged.values())[:top_k]
        if include_contributors:
            for repo in repos[:10]:
                repo["contributors_count"] = self.get_contributors_count(repo["full_name"])
        for repo in repos:
            repo.setdefault("contributors_count", 0)
        return repos

    def fetch_repository(self, full_name: str) -> Dict[str, Any] | None:
        """Fetch one repository by full name for explicit feedback-based inclusion."""
        try:
            item = self._request("GET", f"/repos/{full_name}")
            return {
                "full_name": item.get("full_name"),
                "html_url": item.get("html_url"),
                "description": item.get("description"),
                "topics": item.get("topics", []),
                "language": item.get("language"),
                "stargazers_count": item.get("stargazers_count", 0),
                "forks_count": item.get("forks_count", 0),
                "open_issues_count": item.get("open_issues_count", 0),
                "watchers_count": item.get("watchers_count", 0),
                "license": (item.get("license") or {}).get("spdx_id"),
                "pushed_at": item.get("pushed_at"),
                "created_at": item.get("created_at"),
                "contributors_count": 0,
            }
        except Exception:
            return None

    @staticmethod
    def _split_owner_repo(full_name: str) -> Tuple[str, str]:
        if "/" not in full_name:
            return "", ""
        owner, repo = full_name.split("/", 1)
        return owner, repo

    def _repo_default_branch(self, full_name: str) -> str:
        owner, repo = self._split_owner_repo(full_name)
        if not owner:
            return "main"
        try:
            data = self._request("GET", f"/repos/{owner}/{repo}")
            return data.get("default_branch") or "main"
        except Exception:
            return "main"

    def fetch_repo_readme(self, full_name: str) -> Dict[str, Any] | None:
        owner, repo = self._split_owner_repo(full_name)
        if not owner:
            return None
        try:
            data = self._request("GET", f"/repos/{owner}/{repo}/readme")
            content = b64decode((data.get("content") or "").encode("utf-8")).decode("utf-8", errors="ignore")
            path = data.get("path") or "README.md"
            branch = self._repo_default_branch(full_name)
            return {
                "type": "README",
                "path": path,
                "text": content[:12000],
                "url": f"https://github.com/{full_name}/blob/{branch}/{path}",
            }
        except Exception:
            return None

    def _list_repo_contents_recursive(
        self, full_name: str, max_files: int = 40, max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        owner, repo = self._split_owner_repo(full_name)
        if not owner:
            return []

        allowed_ext = {
            ".md", ".rst", ".txt", ".go", ".py", ".ts", ".tsx", ".js", ".java", ".kt",
            ".rs", ".yaml", ".yml", ".json", ".toml", ".sh",
        }
        preferred_dirs = ("docs/", "doc/", "examples/", "example/", "pkg/", "internal/", "src/")
        branch = self._repo_default_branch(full_name)
        collected: List[Dict[str, Any]] = []

        def walk(path: str, depth: int) -> None:
            if len(collected) >= max_files or depth > max_depth:
                return
            try:
                entries = self._request("GET", f"/repos/{owner}/{repo}/contents/{path}")
            except Exception:
                return
            if not isinstance(entries, list):
                return
            for entry in entries:
                if len(collected) >= max_files:
                    break
                etype = entry.get("type")
                ep = entry.get("path", "")
                if etype == "dir":
                    if depth < max_depth and (
                        depth == 0 or ep.lower().startswith(preferred_dirs) or "docs" in ep.lower()
                    ):
                        walk(ep, depth + 1)
                elif etype == "file":
                    lower = ep.lower()
                    if not any(lower.endswith(ext) for ext in allowed_ext):
                        continue
                    try:
                        blob = self._request("GET", f"/repos/{owner}/{repo}/contents/{ep}")
                        raw = b64decode((blob.get("content") or "").encode("utf-8")).decode("utf-8", errors="ignore")
                    except Exception:
                        continue
                    collected.append(
                        {
                            "type": "FILE",
                            "path": ep,
                            "text": raw[:8000],
                            "url": f"https://github.com/{full_name}/blob/{branch}/{ep}",
                        }
                    )

        walk("", 0)
        return collected

    def fetch_repo_releases(self, full_name: str, per_page: int = 6) -> List[Dict[str, Any]]:
        owner, repo = self._split_owner_repo(full_name)
        if not owner:
            return []
        try:
            rows = self._request("GET", f"/repos/{owner}/{repo}/releases", {"per_page": per_page})
            out = []
            for r in rows if isinstance(rows, list) else []:
                out.append(
                    {
                        "type": "RELEASE",
                        "path": f"release:{r.get('tag_name') or r.get('name')}",
                        "text": f"{r.get('name') or ''}\n{r.get('body') or ''}"[:8000],
                        "url": r.get("html_url"),
                    }
                )
            return out
        except Exception:
            return []

    def fetch_repo_recent_issues(self, full_name: str, per_page: int = 12) -> List[Dict[str, Any]]:
        owner, repo = self._split_owner_repo(full_name)
        if not owner:
            return []
        try:
            rows = self._request(
                "GET",
                f"/repos/{owner}/{repo}/issues",
                {"state": "all", "sort": "updated", "direction": "desc", "per_page": per_page},
            )
            out = []
            for it in rows if isinstance(rows, list) else []:
                if it.get("pull_request"):
                    continue
                out.append(
                    {
                        "type": "ISSUE",
                        "path": f"issue:{it.get('number')}",
                        "text": f"{it.get('title') or ''}\n{it.get('body') or ''}"[:7000],
                        "url": it.get("html_url"),
                    }
                )
            return out
        except Exception:
            return []

    def build_repo_evidence_index(self, full_name: str) -> List[Dict[str, Any]]:
        """Build evidence index for grounded repo chat answers with citations."""
        max_files = int(os.getenv("GITHUB_REPO_INDEX_MAX_FILES", "40"))
        chunks: List[Dict[str, Any]] = []
        readme = self.fetch_repo_readme(full_name)
        if readme:
            chunks.append(readme)
        chunks.extend(self._list_repo_contents_recursive(full_name, max_files=max_files))
        chunks.extend(self.fetch_repo_recent_issues(full_name, per_page=12))
        chunks.extend(self.fetch_repo_releases(full_name, per_page=6))
        return chunks

    def fetch_issues_for_repos(self, repos: List[Dict], per_repo: int = 10, max_repos: int = 5) -> List[Dict]:
        all_issues: List[Dict] = []
        labels_filter = {
            "good first issue",
            "good-first-issue",
            "help-wanted",
            "enhancement",
            "feature",
            "feature-request",
            "docs",
        }

        for repo in repos[:max_repos]:
            full_name = repo.get("full_name")
            if not full_name:
                continue
            try:
                issues = self._request(
                    "GET",
                    f"/repos/{full_name}/issues",
                    {
                        "state": "open",
                        "sort": "updated",
                        "direction": "desc",
                        "per_page": per_repo,
                    },
                )
            except Exception:
                continue

            for issue in issues:
                if issue.get("pull_request"):
                    continue
                label_names = [lbl.get("name", "").strip().lower() for lbl in issue.get("labels", [])]
                normalized = [lbl.replace("_", "-") for lbl in label_names]
                if labels_filter and not labels_filter.intersection(set(normalized) | set(label_names)):
                    continue
                all_issues.append(
                    {
                        "repo_full_name": full_name,
                        "item_type": "ISSUE",
                        "item_id": str(issue.get("id")),
                        "title": issue.get("title"),
                        "body": (issue.get("body") or "")[:3500],
                        "labels": normalized,
                        "comments": issue.get("comments", 0),
                        "updated_at": issue.get("updated_at"),
                        "url": issue.get("html_url"),
                        "author_login": ((issue.get("user") or {}).get("login")),
                        "assignees": [a.get("login") for a in (issue.get("assignees") or []) if a.get("login")],
                    }
                )
        return all_issues

    def fetch_discussions_for_repos(
        self, repos: List[Dict], per_repo: int = 8, max_repos: int = 4
    ) -> Tuple[List[Dict], bool]:
        """Fetch discussions via GraphQL. Returns (items, available_flag)."""
        if not self.is_configured():
            return [], False

        query = """
        query($owner: String!, $name: String!, $first: Int!) {
          repository(owner: $owner, name: $name) {
            discussions(first: $first, orderBy: {field: UPDATED_AT, direction: DESC}) {
              nodes {
                id
                title
                bodyText
                url
                updatedAt
                author { login }
                comments { totalCount }
                category { name }
              }
            }
          }
        }
        """

        items: List[Dict] = []
        for repo in repos[:max_repos]:
            full_name = repo.get("full_name", "")
            if "/" not in full_name:
                continue
            owner, name = full_name.split("/", 1)
            try:
                data = self._graphql(query, {"owner": owner, "name": name, "first": per_repo})
                nodes = (((data or {}).get("repository") or {}).get("discussions") or {}).get("nodes") or []
                for node in nodes:
                    category = (((node or {}).get("category") or {}).get("name") or "discussion").lower()
                    items.append(
                        {
                            "repo_full_name": full_name,
                            "item_type": "DISCUSSION",
                            "item_id": str(node.get("id")),
                            "title": node.get("title"),
                            "body": (node.get("bodyText") or "")[:3500],
                            "labels": [category],
                            "comments": ((node.get("comments") or {}).get("totalCount") or 0),
                            "updated_at": node.get("updatedAt"),
                            "url": node.get("url"),
                            "author_login": (((node or {}).get("author") or {}).get("login")),
                            "assignees": [],
                        }
                    )
            except Exception:
                return [], False

        return items, True
