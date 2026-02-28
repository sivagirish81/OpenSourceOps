"""Repository ingestion and semantic chunking for due diligence indexing."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _git(args: List[str], cwd: Path) -> str:
    out = subprocess.check_output(["git", *args], cwd=str(cwd), stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="ignore").strip()


def _is_text_file(path: Path) -> bool:
    ext = path.suffix.lower()
    return ext in {
        ".md", ".rst", ".txt", ".go", ".py", ".ts", ".tsx", ".js", ".java", ".kt", ".rs",
        ".yaml", ".yml", ".toml", ".json", ".sh", ".sql",
    } or path.name.lower() in {"dockerfile", ".env.example", "makefile"}


def _priority_rank(rel_path: str) -> int:
    """Lower rank means higher indexing priority."""
    p = rel_path.lower()
    name = Path(rel_path).name.lower()
    if name.startswith(("readme", "license", "security", "changelog", "migration", "upgrading", "code_of_conduct")):
        return 0
    if any(k in p for k in ["architecture", "adr", "design", "runbook", "operat", "deploy"]):
        return 1
    if p.startswith("docs/") or p.startswith("examples/"):
        return 2
    if any(x in p for x in ["dockerfile", "docker-compose", ".env.example", "helm", "values.yaml"]):
        return 3
    if any(k in p for k in ["security", "auth", "oauth", "saml", "rbac", "permission", "policy"]):
        return 3
    if any(k in p for k in ["observability", "metrics", "monitor", "logging", "otel", "prometheus", "datadog"]):
        return 3
    if any(p.endswith(ext) for ext in [".yaml", ".yml", ".toml", ".json"]):
        return 4
    if any(p.endswith(ext) for ext in [".py", ".go", ".ts", ".js", ".java", ".rs"]):
        return 5
    return 6


def _chunk_markdown(content: str, file_path: str) -> List[Dict[str, Any]]:
    lines = content.splitlines()
    if not lines:
        return []
    chunks: List[Dict[str, Any]] = []
    heading = "root"
    start = 1
    buf: List[str] = []

    def flush(end_line: int):
        nonlocal buf, start
        text = "\n".join(buf).strip()
        if text:
            chunks.append(
                {
                    "file_path": file_path,
                    "start_line": start,
                    "end_line": end_line,
                    "heading": heading,
                    "symbol_name": None,
                    "content_type": "doc",
                    "chunk_text": text[:8000],
                }
            )
        buf = []

    for i, ln in enumerate(lines, start=1):
        if re.match(r"^#{1,6}\s+", ln):
            if buf:
                flush(i - 1)
            heading = re.sub(r"^#{1,6}\s+", "", ln).strip()
            start = i
            buf = [ln]
        else:
            buf.append(ln)
    if buf:
        flush(len(lines))
    return chunks


def _chunk_code(content: str, file_path: str) -> List[Dict[str, Any]]:
    lines = content.splitlines()
    if not lines:
        return []
    symbol_re = re.compile(r"^\s*(def\s+\w+|class\s+\w+|func\s+\w+|type\s+\w+|interface\s+\w+)")
    boundaries = [i for i, ln in enumerate(lines, start=1) if symbol_re.search(ln)]
    if not boundaries:
        return [
            {
                "file_path": file_path,
                "start_line": 1,
                "end_line": min(len(lines), 200),
                "heading": None,
                "symbol_name": None,
                "content_type": "code",
                "chunk_text": "\n".join(lines[:200]),
            }
        ]
    boundaries.append(len(lines) + 1)
    chunks: List[Dict[str, Any]] = []
    for i in range(len(boundaries) - 1):
        s = boundaries[i]
        e = min(boundaries[i + 1] - 1, s + 220)
        symbol = lines[s - 1].strip().split("(")[0]
        chunks.append(
            {
                "file_path": file_path,
                "start_line": s,
                "end_line": e,
                "heading": None,
                "symbol_name": symbol[:200],
                "content_type": "code",
                "chunk_text": "\n".join(lines[s - 1 : e]),
            }
        )
    return chunks


def _chunk_config(content: str, file_path: str) -> List[Dict[str, Any]]:
    lines = content.splitlines()
    if not lines:
        return []
    step = 120
    chunks: List[Dict[str, Any]] = []
    for i in range(0, len(lines), step):
        s = i + 1
        e = min(i + step, len(lines))
        chunks.append(
            {
                "file_path": file_path,
                "start_line": s,
                "end_line": e,
                "heading": None,
                "symbol_name": None,
                "content_type": "config",
                "chunk_text": "\n".join(lines[i:e]),
            }
        )
    return chunks


def _chunk_file(content: str, file_path: str) -> List[Dict[str, Any]]:
    lower = file_path.lower()
    if lower.endswith((".md", ".rst", ".txt")):
        return _chunk_markdown(content, file_path)
    if lower.endswith((".yaml", ".yml", ".toml", ".json")) or lower in {"dockerfile", ".env.example"}:
        return _chunk_config(content, file_path)
    return _chunk_code(content, file_path)


class RepoIndexer:
    """Shallow clone + chunking with metadata for Snowflake persistence."""

    def __init__(self, max_files: int = 160) -> None:
        self.max_files = max_files

    def _select_files(self, repo_dir: Path, files: List[Path]) -> List[Path]:
        """Select most relevant files for due diligence; keep indexing small and meaningful."""
        scored: List[tuple[int, int, Path]] = []
        for p in files:
            rel = str(p.relative_to(repo_dir))
            rank = _priority_rank(rel)
            scored.append((rank, len(rel), p))
        scored.sort(key=lambda x: (x[0], x[1]))

        # Keep a focused set: key docs/config first, then minimal code representative files.
        selected = [p for _, _, p in scored[: self.max_files]]
        return selected

    def build(
        self,
        repo_url: str,
        ref: str | None = None,
        progress_cb=None,
        chunk_sink=None,
    ) -> Tuple[str, str, List[Dict[str, Any]]]:
        tmp = Path(tempfile.mkdtemp(prefix="ossops_repo_"))
        try:
            if progress_cb:
                progress_cb(0.05, "Cloning repository...")
            cmd = ["git", "clone", "--depth", "1"]
            if ref:
                cmd.extend(["--branch", ref])
            cmd.extend([repo_url, str(tmp / "repo")])
            subprocess.check_call(cmd)
            repo_dir = tmp / "repo"
            commit_sha = _git(["rev-parse", "HEAD"], repo_dir)
            if progress_cb:
                progress_cb(0.22, f"Clone complete at commit {commit_sha[:12]}")

            files: List[Path] = []
            for p in repo_dir.rglob("*"):
                if p.is_file() and _is_text_file(p):
                    rel = p.relative_to(repo_dir)
                    if ".git" in rel.parts:
                        continue
                    files.append(p)

            files = self._select_files(repo_dir, files)
            if progress_cb:
                progress_cb(0.32, f"Selected {len(files)} files for indexing")

            chunks: List[Dict[str, Any]] = []
            total = max(len(files), 1)
            for idx, p in enumerate(files, start=1):
                rel = str(p.relative_to(repo_dir))
                try:
                    content = p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                new_chunks = []
                for chunk in _chunk_file(content, rel):
                    chunk["repo_url"] = repo_url
                    chunk["commit_sha"] = commit_sha
                    chunk["url"] = f"{repo_url}/blob/{commit_sha}/{rel}#L{chunk['start_line']}"
                    chunks.append(chunk)
                    new_chunks.append(chunk)
                if chunk_sink and new_chunks:
                    chunk_sink(new_chunks, rel)
                if progress_cb and (idx == 1 or idx % 10 == 0 or idx == total):
                    frac = 0.32 + 0.66 * (idx / total)
                    progress_cb(frac, f"Indexed {idx}/{total} files")

            repo_id = hashlib.sha1(f"{repo_url}@{commit_sha}".encode("utf-8")).hexdigest()[:16]
            if progress_cb:
                progress_cb(1.0, f"Index complete: {len(chunks)} chunks")
            return repo_id, commit_sha, chunks
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
