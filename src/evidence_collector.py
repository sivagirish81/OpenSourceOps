"""Signal-based evidence collector using GitHub APIs only (no full codebase indexing)."""

from __future__ import annotations

import base64
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from src.evidence_parsers import (
    parse_ci_workflow,
    parse_deploy_config,
    parse_license,
    parse_readme_docs,
    parse_release_changelog,
    parse_security,
)
from src.github_client import GitHubClient


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RepoEvidenceCollector:
    """Collects lightweight, high-signal repository evidence within strict budgets."""

    ALLOWED_EXT = (".md", ".markdown", ".yaml", ".yml", ".toml", ".env")
    ROOT_PATTERNS = (
        r"^README",
        r"^LICENSE",
        r"^SECURITY",
        r"^CODE_OF_CONDUCT",
        r"^CONTRIBUTING",
        r"^CHANGELOG",
        r"^MIGRATION",
        r"^UPGRADING",
    )

    def __init__(self, github: GitHubClient | None = None) -> None:
        self.github = github or GitHubClient()
        self.max_files = int(os.getenv("MAX_FILES", "80"))
        self.max_bytes = int(os.getenv("MAX_BYTES", str(8 * 1024 * 1024)))
        self.max_doc_files = int(os.getenv("MAX_DOC_FILES", "35"))
        self.max_workflow_files = int(os.getenv("MAX_WORKFLOW_FILES", "20"))

    def make_citation(
        self,
        evidence: Dict[str, Any],
        start_line: int | None = None,
        end_line: int | None = None,
        snippet: str | None = None,
    ) -> Dict[str, Any]:
        return {
            "repo_url": evidence.get("repo_url"),
            "file_path": evidence.get("file_path"),
            "start_line": start_line,
            "end_line": end_line,
            "url": evidence.get("url"),
            "source_type": evidence.get("source_type"),
            "api_ref": evidence.get("api_ref"),
            "retrieved_at": evidence.get("retrieved_at"),
            "snippet": snippet,
        }

    @staticmethod
    def _line_map(content: str) -> List[Dict[str, Any]]:
        out = []
        for idx, line in enumerate((content or "").splitlines(), start=1):
            out.append({"line": idx, "text": line[:400]})
        return out

    @staticmethod
    def _decode_content(entry: Dict[str, Any]) -> str:
        enc = entry.get("encoding")
        raw = entry.get("content", "")
        if enc == "base64":
            try:
                return base64.b64decode(raw.encode("utf-8")).decode("utf-8", errors="ignore")
            except Exception:
                return ""
        return raw or ""

    def _evidence(
        self,
        repo_full_name: str,
        repo_url: str,
        ref: str,
        source_type: str,
        file_path: str | None,
        url: str,
        sha: str | None,
        content: str | None = None,
        structured: Dict[str, Any] | None = None,
        api_ref: str | None = None,
    ) -> Dict[str, Any]:
        return {
            "evidence_id": f"ev_{uuid.uuid4().hex[:12]}",
            "repo_url": repo_url,
            "repo_full_name": repo_full_name,
            "commit_or_ref": ref,
            "source_type": source_type,
            "file_path": file_path,
            "url": url,
            "retrieved_at": now_iso(),
            "content": content,
            "structured": structured or {},
            "line_map": self._line_map(content or "") if content else [],
            "sha": sha,
            "api_ref": api_ref,
        }

    def _list_dir(self, full_name: str, path: str = "") -> List[Dict[str, Any]]:
        try:
            res = self.github._request("GET", f"/repos/{full_name}/contents/{path}")
            return res if isinstance(res, list) else []
        except Exception:
            return []

    def _get_file(self, full_name: str, path: str) -> Dict[str, Any] | None:
        try:
            res = self.github._request("GET", f"/repos/{full_name}/contents/{path}")
            return res if isinstance(res, dict) else None
        except Exception:
            return None

    @staticmethod
    def _matches_root(name: str) -> bool:
        up = name.upper()
        for pat in RepoEvidenceCollector.ROOT_PATTERNS:
            if re.match(pat, up):
                return True
        return False

    def _allow_file(self, path: str) -> bool:
        p = path.lower()
        if any(p.endswith(ext) for ext in self.ALLOWED_EXT):
            return True
        if p.startswith(".github/workflows/") and p.endswith((".yml", ".yaml")):
            return True
        if p.startswith("charts/") and p.endswith((".yaml", ".yml")):
            return True
        if p.endswith("dockerfile") or "docker-compose" in p:
            return True
        return False

    def _target_paths(self, full_name: str) -> List[str]:
        root = self._list_dir(full_name, "")
        paths: List[str] = []
        for e in root:
            if e.get("type") != "file":
                continue
            n = e.get("name", "")
            p = e.get("path", "")
            if self._matches_root(n):
                paths.append(p)
            if n in {"Dockerfile", ".env.example"} or n.endswith((".yaml", ".yml", ".toml")):
                paths.append(p)

        # Workflows
        wf = self._list_dir(full_name, ".github/workflows")
        for e in wf[: self.max_workflow_files]:
            if e.get("type") == "file" and e.get("path", "").endswith((".yml", ".yaml")):
                paths.append(e["path"])

        # docs markdown only (limited recursive)
        doc_dirs = ["docs", "doc"]
        doc_files = 0
        queue = [(d, 0) for d in doc_dirs]
        while queue and doc_files < self.max_doc_files:
            d, depth = queue.pop(0)
            if depth > 2:
                continue
            for e in self._list_dir(full_name, d):
                p = e.get("path", "")
                if e.get("type") == "dir":
                    queue.append((p, depth + 1))
                elif p.lower().endswith(".md"):
                    paths.append(p)
                    doc_files += 1
                    if doc_files >= self.max_doc_files:
                        break

        # Deploy/config dirs
        for d in ["deploy", "k8s", "manifests", "charts"]:
            for e in self._list_dir(full_name, d):
                p = e.get("path", "")
                if e.get("type") == "file" and p.endswith((".yaml", ".yml", ".toml")):
                    paths.append(p)

        # Dedup preserve order
        seen = set()
        out = []
        for p in paths:
            if p in seen:
                continue
            seen.add(p)
            if self._allow_file(p):
                out.append(p)
        return out[: self.max_files]

    def collect(
        self,
        repo_full_name: str,
        repo_url: str,
        ref: str | None = None,
        progress_cb=None,
    ) -> Dict[str, Any]:
        evidence: List[Dict[str, Any]] = []
        signals: List[Dict[str, Any]] = []
        files_fetched = 0
        bytes_fetched = 0

        # Metadata
        repo = self.github.fetch_repository(repo_full_name) or {}
        commit_ref = ref or repo.get("pushed_at") or "latest"
        evidence.append(
            self._evidence(
                repo_full_name, repo_url, commit_ref, "github_repo", None, repo_url, None,
                structured=repo,
                api_ref=f"/repos/{repo_full_name}",
            )
        )
        if progress_cb:
            progress_cb(0.1, "Fetched repository metadata")

        releases = self.github.fetch_repo_releases(repo_full_name, per_page=4)
        for r in releases:
            evidence.append(
                self._evidence(
                    repo_full_name, repo_url, commit_ref, "github_release", None, r.get("url") or repo_url, None,
                    content=f"{r.get('title')}\n{r.get('body')}",
                    structured=r,
                    api_ref=f"/repos/{repo_full_name}/releases",
                )
            )
        issues = self.github.fetch_repo_recent_issues(repo_full_name, per_page=10)
        for i in issues[:8]:
            evidence.append(
                self._evidence(
                    repo_full_name, repo_url, commit_ref, "github_issue_sample", None, i.get("url") or repo_url, None,
                    content=f"{i.get('title')}\n{i.get('body')}",
                    structured=i,
                    api_ref=f"/repos/{repo_full_name}/issues",
                )
            )
        discussions = self.github.fetch_repo_discussions(repo_full_name, first=5)
        for d in discussions:
            evidence.append(
                self._evidence(
                    repo_full_name, repo_url, commit_ref, "github_issue_sample", None, d.get("url") or repo_url, None,
                    content=f"{d.get('title')}\n{d.get('body')}",
                    structured=d,
                    api_ref=f"/repos/{repo_full_name}/discussions",
                )
            )

        # Targeted files
        paths = self._target_paths(repo_full_name)
        total = max(len(paths), 1)
        for idx, path in enumerate(paths, start=1):
            if files_fetched >= self.max_files or bytes_fetched >= self.max_bytes:
                break
            node = self._get_file(repo_full_name, path)
            if not node:
                continue
            content = self._decode_content(node)
            if not content:
                continue
            b = len(content.encode("utf-8", errors="ignore"))
            if bytes_fetched + b > self.max_bytes:
                continue
            bytes_fetched += b
            files_fetched += 1
            url = node.get("html_url") or f"{repo_url}/blob/{commit_ref}/{path}"
            ev = self._evidence(
                repo_full_name,
                repo_url,
                commit_ref,
                "github_workflow" if path.startswith(".github/workflows/") else "github_file",
                path,
                url,
                node.get("sha"),
                content=content[:12000],
                structured={"size_bytes": b},
                api_ref=f"/repos/{repo_full_name}/contents/{path}",
            )
            evidence.append(ev)

            p_low = path.lower()
            if p_low.endswith(".md"):
                signals.extend(parse_readme_docs(content, ev, self.make_citation))
                if "security" in p_low:
                    signals.extend(parse_security(content, ev, self.make_citation))
                if any(x in p_low for x in ["changelog", "migration", "upgrading"]):
                    signals.extend(parse_release_changelog(content, ev, self.make_citation))
            if "license" in p_low:
                signals.extend(parse_license(content, ev, self.make_citation))
            if p_low.startswith(".github/workflows/"):
                signals.extend(parse_ci_workflow(content, ev, self.make_citation))
            if any(x in p_low for x in ["docker", "helm", "k8s", "deploy", "manifests"]) or p_low.endswith((".yaml", ".yml", ".toml", ".env")):
                signals.extend(parse_deploy_config(content, ev, self.make_citation))

            if progress_cb and (idx == 1 or idx % 10 == 0 or idx == total):
                progress_cb(0.2 + 0.75 * (idx / total), f"Collected {idx}/{total} targeted files")

        if progress_cb:
            progress_cb(1.0, f"Evidence collection complete ({files_fetched} files, {bytes_fetched} bytes)")

        return {
            "commit_or_ref": commit_ref,
            "evidence": evidence,
            "signals": signals,
            "budget_usage": {
                "max_files": self.max_files,
                "max_bytes": self.max_bytes,
                "files_fetched": files_fetched,
                "bytes_fetched": bytes_fetched,
            },
            "file_manifest": [e.get("file_path") for e in evidence if e.get("file_path")],
        }

