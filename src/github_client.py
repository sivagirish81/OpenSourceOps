"""GitHub REST API client for repository and issue discovery."""

from __future__ import annotations

import os
import re
from typing import Dict, List

import requests


class GitHubClient:
    """Lightweight authenticated GitHub REST client with safe fallbacks."""

    BASE_URL = "https://api.github.com"

    def __init__(self, token: str | None = None, timeout: int = 20) -> None:
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "User-Agent": "OpenSourceOps/0.1",
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

    @staticmethod
    def _parse_last_page(link_header: str | None) -> int:
        """Extract total count from link header when using `per_page=1`."""
        if not link_header:
            return 0
        match = re.search(r"[?&]page=(\d+)>; rel=\"last\"", link_header)
        return int(match.group(1)) if match else 1

    def get_contributors_count(self, full_name: str) -> int:
        """Best-effort contributor count by checking final pagination page."""
        try:
            url = f"{self.BASE_URL}/repos/{full_name}/contributors"
            resp = self.session.get(
                url,
                params={"per_page": 1, "anon": "true"},
                timeout=self.timeout,
            )
            if resp.status_code >= 400:
                return 0
            return self._parse_last_page(resp.headers.get("Link"))
        except Exception:
            return 0

    def search_repositories(
        self,
        queries: List[str],
        top_k: int = 25,
        max_queries: int = 2,
        per_page: int = 15,
        include_contributors: bool = False,
    ) -> List[Dict]:
        """Run a capped number of queries and deduplicate repositories."""
        merged: Dict[str, Dict] = {}
        for query in queries[:max_queries]:
            try:
                payload = self._request(
                    "GET",
                    "/search/repositories",
                    {"q": query, "sort": "stars", "order": "desc", "per_page": per_page},
                )
                for item in payload.get("items", []):
                    full_name = item.get("full_name")
                    if not full_name:
                        continue
                    if full_name in merged:
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

    def fetch_issues_for_repos(self, repos: List[Dict], per_repo: int = 10, max_repos: int = 5) -> List[Dict]:
        """Fetch candidate contribution issues for top repositories."""
        all_issues: List[Dict] = []
        labels_filter = {"good first issue", "good-first-issue", "help-wanted", "bug"}

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
                if not labels_filter.intersection(set(normalized) | set(label_names)):
                    continue
                all_issues.append(
                    {
                        "full_name": full_name,
                        "id": issue.get("id"),
                        "title": issue.get("title"),
                        "body": (issue.get("body") or "")[:1500],
                        "labels": normalized,
                        "comments": issue.get("comments", 0),
                        "created_at": issue.get("created_at"),
                        "updated_at": issue.get("updated_at"),
                        "html_url": issue.get("html_url"),
                    }
                )
        return all_issues
