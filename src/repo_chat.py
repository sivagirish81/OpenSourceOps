"""Single-repo grounded chat with hybrid retrieval and verifier-enforced answer contract."""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.store import Store


ALLOWED_DOC_PREFIXES = (
    "README", "LICENSE", "SECURITY", "CODE_OF_CONDUCT", "CHANGELOG", "MIGRATION", "UPGRADING",
)


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9_\-]+", (text or "").lower()) if len(t) > 2]


def _embed_hash(text: str, dim: int = 256) -> List[float]:
    vec = [0.0] * dim
    for tok in _tokenize(text):
        idx = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def _git(args: List[str], cwd: Path) -> str:
    out = subprocess.check_output(["git", *args], cwd=str(cwd), stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="ignore").strip()


def _is_tier0(path: Path) -> bool:
    name_upper = path.name.upper()
    if any(name_upper.startswith(pfx) for pfx in ALLOWED_DOC_PREFIXES):
        return True
    if "docs" in path.parts or "examples" in path.parts:
        return True
    lower = path.name.lower()
    if lower in {"dockerfile", "docker-compose.yml", "docker-compose.yaml", ".env.example", "values.yaml"}:
        return True
    if lower.endswith((".yaml", ".yml", ".toml")) and any(k in lower for k in ["helm", "deploy", "config", "values"]):
        return True
    return False


def _is_text_file(path: Path) -> bool:
    lower = path.suffix.lower()
    return lower in {
        ".md", ".rst", ".txt", ".go", ".py", ".ts", ".tsx", ".js", ".java", ".kt", ".rs",
        ".yaml", ".yml", ".toml", ".json", ".sh", ".sql",
    } or path.name.lower() in {"dockerfile", ".env.example"}


def _chunk_markdown(content: str, file_path: str) -> List[Dict[str, Any]]:
    lines = content.splitlines()
    chunks: List[Dict[str, Any]] = []
    current_heading = "root"
    start = 1
    buffer: List[str] = []

    def flush(end_line: int):
        nonlocal start, buffer
        if not buffer:
            return
        text = "\n".join(buffer).strip()
        if text:
            chunks.append(
                {
                    "file_path": file_path,
                    "start_line": start,
                    "end_line": end_line,
                    "heading": current_heading,
                    "symbol_name": None,
                    "content_type": "doc",
                    "chunk_text": text,
                }
            )
        buffer = []

    for i, ln in enumerate(lines, start=1):
        if re.match(r"^#{1,6}\s+", ln):
            flush(i - 1)
            current_heading = re.sub(r"^#{1,6}\s+", "", ln).strip()
            start = i
            buffer = [ln]
        else:
            buffer.append(ln)
    flush(len(lines))
    return chunks


def _chunk_code(content: str, file_path: str) -> List[Dict[str, Any]]:
    lines = content.splitlines()
    pattern = re.compile(r"^\s*(def\s+\w+|class\s+\w+|func\s+\w+|type\s+\w+|interface\s+\w+)")
    chunks: List[Dict[str, Any]] = []

    boundaries = [i for i, ln in enumerate(lines, start=1) if pattern.search(ln)]
    if not boundaries:
        return [
            {
                "file_path": file_path,
                "start_line": 1,
                "end_line": len(lines),
                "heading": None,
                "symbol_name": None,
                "content_type": "code",
                "chunk_text": "\n".join(lines[:300]),
            }
        ]

    boundaries.append(len(lines) + 1)
    for idx in range(len(boundaries) - 1):
        s = boundaries[idx]
        e = min(boundaries[idx + 1] - 1, s + 200)
        symbol_ln = lines[s - 1].strip()
        sym = symbol_ln.split("(")[0].replace("{", "").strip()
        chunks.append(
            {
                "file_path": file_path,
                "start_line": s,
                "end_line": e,
                "heading": None,
                "symbol_name": sym,
                "content_type": "code",
                "chunk_text": "\n".join(lines[s - 1 : e]),
            }
        )
    return chunks


def _chunk_config(content: str, file_path: str) -> List[Dict[str, Any]]:
    lines = content.splitlines()
    chunks = []
    step = 120
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


def _chunk_file(content: str, rel_path: str) -> List[Dict[str, Any]]:
    lower = rel_path.lower()
    if lower.endswith((".md", ".rst", ".txt")):
        return _chunk_markdown(content, rel_path)
    if lower.endswith((".yaml", ".yml", ".toml", ".json")) or rel_path in {"Dockerfile", ".env.example"}:
        return _chunk_config(content, rel_path)
    return _chunk_code(content, rel_path)


@dataclass
class RetrieverAgent:
    def retrieve(self, chunks: List[Dict[str, Any]], question: str, k: int = 8) -> List[Dict[str, Any]]:
        q_terms = _tokenize(question)
        if not chunks:
            return []

        docs = [c.get("chunk_text", "") for c in chunks]
        doc_tokens = [_tokenize(d) for d in docs]
        n = len(docs)

        df = defaultdict(int)
        for toks in doc_tokens:
            for t in set(toks):
                df[t] += 1

        avgdl = sum(len(t) for t in doc_tokens) / max(n, 1)
        k1 = 1.5
        b = 0.75

        q_vec = _embed_hash(question)
        scored = []
        for i, c in enumerate(chunks):
            toks = doc_tokens[i]
            tf = Counter(toks)
            dl = len(toks) or 1

            bm25 = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                freq = tf[term]
                denom = freq + k1 * (1 - b + b * dl / max(avgdl, 1))
                bm25 += idf * ((freq * (k1 + 1)) / denom)

            emb = c.get("embedding") or _embed_hash(c.get("chunk_text", ""))
            sem = _cosine(q_vec, emb)

            exact_boost = 0.0
            file_path = (c.get("file_path") or "").lower()
            if any(t in file_path for t in q_terms):
                exact_boost += 0.4
            if any(t in (c.get("symbol_name") or "").lower() for t in q_terms):
                exact_boost += 0.6

            score = 0.55 * bm25 + 0.35 * sem + 0.10 * exact_boost
            if score > 0:
                scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:k]]


@dataclass
class VerifierAgent:
    min_chunks: int = 2

    def classify_question(self, question: str) -> str:
        q = question.lower()
        if any(x in q for x in ["where", "implemented", "code path", "symbol"]):
            return "A"
        if any(x in q for x in ["integrate", "configure", "setup", "deploy", "how do i"]):
            return "B"
        if any(x in q for x in ["does it support", "support", "oauth", "saml", "kafka", "postgres"]):
            return "C"
        if any(x in q for x in ["risk", "challenge", "pitfall", "hidden cost"]):
            return "D"
        if any(x in q for x in ["breaking change", "upgrade", "migration", "between versions"]):
            return "E"
        return "B"

    def evidence_strength(self, evidence: List[Dict[str, Any]]) -> float:
        if not evidence:
            return 0.0
        score = 0.0
        for e in evidence[:4]:
            ctype = (e.get("content_type") or "").lower()
            score += 0.45 if ctype == "code" else 0.35
            if e.get("symbol_name"):
                score += 0.2
        return min(score / 2.0, 1.0)

    def grounded_gate(self, question_type: str, evidence: List[Dict[str, Any]]) -> Tuple[bool, str]:
        if len(evidence) < self.min_chunks:
            return False, "Not enough relevant chunks"
        strength = self.evidence_strength(evidence)
        if question_type in {"A", "C", "E"} and strength < 0.45:
            return False, "Evidence too weak for strict capability/location answer"
        return True, "ok"


@dataclass
class WriterAgent:
    def _citations(self, evidence: List[Dict[str, Any]], limit: int = 6) -> List[Dict[str, Any]]:
        cites = []
        for e in evidence[:limit]:
            snippet = (e.get("chunk_text") or "")[:220].replace("\n", " ")
            cites.append(
                {
                    "file_path": e.get("file_path"),
                    "start_line": int(e.get("start_line") or 1),
                    "end_line": int(e.get("end_line") or 1),
                    "url": e.get("url"),
                    "snippet": snippet,
                }
            )
        return cites

    def _next_checks(self, evidence: List[Dict[str, Any]], question: str) -> List[str]:
        q_terms = _tokenize(question)
        file_hints = []
        for e in evidence[:5]:
            fp = e.get("file_path")
            if fp and fp not in file_hints:
                file_hints.append(fp)
        out = [f"Inspect {fp}" for fp in file_hints[:4]]
        if q_terms:
            out.append(f"Search repo for keywords: {', '.join(q_terms[:5])}")
        return out[:5]

    def answer(
        self,
        repo_id: str,
        question_type: str,
        question: str,
        mode: str,
        evidence: List[Dict[str, Any]],
        grounded_ok: bool,
    ) -> Dict[str, Any]:
        cites = self._citations(evidence)
        if not grounded_ok:
            return {
                "answer": "Not confirmed from repo evidence",
                "citations": cites,
                "confidence": "Low",
                "next_checks": self._next_checks(evidence, question),
                "question_type": question_type,
            }

        confidence = "High" if len(cites) >= 4 else "Medium"

        if question_type == "A":
            answer = "Implementation appears in the cited code locations; follow symbol/file references for call-path details."
        elif question_type == "B":
            answer = "Integration/configuration can be derived from the cited docs/config/code entry points below."
        elif question_type == "C":
            answer = "Capability is supported only if explicitly shown in the cited evidence below; otherwise treat as unconfirmed."
            if not any(any(k in (c.get("snippet") or "").lower() for k in ["oauth", "saml", "kafka", "postgres"]) for c in cites):
                return {
                    "answer": "Not confirmed from repo evidence",
                    "citations": cites,
                    "confidence": "Low",
                    "next_checks": self._next_checks(evidence, question),
                    "question_type": question_type,
                }
        elif question_type == "D":
            answer = (
                "Primary integration risks and mitigations can be inferred from the cited operational/configuration surfaces. "
                "Uncited items should be treated as general adoption risks."
            )
        else:
            answer = (
                "Breaking-change guidance is based on cited CHANGELOG/MIGRATION/UPGRADING/release notes. "
                "If sparse, compare tags and run compatibility tests."
            )

        if mode == "risk_review":
            answer = (
                "Risk register: 1) configuration drift; 2) upgrade regressions; 3) operational visibility gaps. "
                "Use cited files/releases to anchor mitigations."
            )
        elif mode == "scenario_brainstorm":
            answer += " Assumptions are marked and tied to cited repo areas."

        return {
            "answer": answer,
            "citations": cites,
            "confidence": confidence,
            "next_checks": self._next_checks(evidence, question),
            "question_type": question_type,
        }


class RepoIndexer:
    """Clone a single repo, capture commit SHA, and produce semantic evidence chunks."""

    def __init__(self, max_files: int = 180) -> None:
        self.max_files = max_files

    def build(self, repo_url: str, ref: str | None = None) -> Tuple[str, str, List[Dict[str, Any]]]:
        tmp = Path(tempfile.mkdtemp(prefix="ossops_repo_"))
        try:
            clone_cmd = ["git", "clone", "--depth", "1"]
            if ref:
                clone_cmd += ["--branch", ref]
            clone_cmd += [repo_url, str(tmp / "repo")]
            subprocess.check_call(clone_cmd)
            repo_dir = tmp / "repo"
            commit_sha = _git(["rev-parse", "HEAD"], repo_dir)

            files: List[Path] = []
            for p in repo_dir.rglob("*"):
                if p.is_file() and _is_text_file(p):
                    rel = p.relative_to(repo_dir)
                    if ".git" in rel.parts:
                        continue
                    files.append(p)

            files.sort(key=lambda x: (0 if _is_tier0(x.relative_to(repo_dir)) else 1, len(str(x))))
            files = files[: self.max_files]

            chunks: List[Dict[str, Any]] = []
            for p in files:
                rel = str(p.relative_to(repo_dir))
                try:
                    content = p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                for ch in _chunk_file(content, rel):
                    ch["repo_url"] = repo_url
                    ch["commit_sha"] = commit_sha
                    ch["url"] = f"{repo_url}/blob/{commit_sha}/{rel}#L{ch['start_line']}"
                    ch["embedding"] = _embed_hash(ch.get("chunk_text") or "")
                    chunks.append(ch)
            repo_id = hashlib.sha1(f"{repo_url}@{commit_sha}".encode("utf-8")).hexdigest()[:16]
            return repo_id, commit_sha, chunks
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class RepoChatService:
    """Orchestrates Retriever/Verifier/Writer agents and persists traces/eval logs."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self.retriever = RetrieverAgent()
        self.verifier = VerifierAgent()
        self.writer = WriterAgent()

    def build_index(self, repo_url: str, ref: str | None = None) -> Dict[str, Any]:
        idx = RepoIndexer(max_files=int(os.getenv("REPO_INDEX_MAX_FILES", "180")))
        repo_id, commit_sha, chunks = idx.build(repo_url, ref)
        self.store.save_evidence_chunks(repo_id, chunks)
        return {"repo_id": repo_id, "commit_sha": commit_sha, "chunk_count": len(chunks)}

    def retrieve(self, repo_id: str, question: str, k: int = 8) -> List[Dict[str, Any]]:
        chunks = self.store.load_evidence_chunks(repo_id)
        return self.retriever.retrieve(chunks, question, k=k)

    def chat(self, repo_id: str, question: str, mode: str) -> Dict[str, Any]:
        mode_map = {
            "grounded_qa": "grounded_qa",
            "scenario_brainstorm": "scenario_brainstorm",
            "risk_review": "risk_review",
        }
        mode = mode_map.get(mode, "grounded_qa")

        qtype = self.verifier.classify_question(question)
        evidence = self.retrieve(repo_id, question, k=8)
        grounded_ok, gate_reason = self.verifier.grounded_gate(qtype, evidence)
        result = self.writer.answer(repo_id, qtype, question, mode, evidence, grounded_ok)

        trace = {
            "trace_id": str(uuid.uuid4()),
            "repo_id": repo_id,
            "mode": mode,
            "question": question,
            "answer": result.get("answer"),
            "confidence": result.get("confidence"),
            "citations": result.get("citations", []),
            "next_checks": result.get("next_checks", []),
            "question_type": qtype,
        }
        self.store.save_chat_trace(trace)

        self.store.save_eval_log(
            {
                "eval_id": str(uuid.uuid4()),
                "repo_id": repo_id,
                "metric": "grounded_gate",
                "value": 1.0 if grounded_ok else 0.0,
                "detail": gate_reason,
            }
        )

        return result
