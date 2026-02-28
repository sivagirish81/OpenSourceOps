from __future__ import annotations

import json

from storage.snowflake_store import SnowflakeConfig, SnowflakeStore


def test_repo_evidence_and_signals_store_roundtrip(monkeypatch):
    cfg = SnowflakeConfig(
        account="acct",
        user="user",
        warehouse="wh",
        database="OPENSOURCEOPS",
        schema="PUBLIC",
        role="APP_ROLE",
        password="pwd",
        private_key=None,
    )
    store = SnowflakeStore(cfg)

    captured = {"evidence": [], "signals": []}

    def fake_execute(sql, params=None):
        su = sql.upper()
        if "INSERT INTO OPENSOURCEOPS.PUBLIC.REPO_EVIDENCE" in su:
            captured["evidence"].append(params)
        if "INSERT INTO OPENSOURCEOPS.PUBLIC.REPO_SIGNALS" in su:
            captured["signals"].append(params)

    def fake_query(sql, params=None):
        su = sql.upper()
        if "FROM OPENSOURCEOPS.PUBLIC.REPO_EVIDENCE" in su:
            return [
                (
                    "ev_1",
                    "https://github.com/acme/repo",
                    "main",
                    "github_file",
                    "README.md",
                    "https://github.com/acme/repo/blob/main/README.md",
                    "/repos/acme/repo/contents/README.md",
                    "2026-02-28T00:00:00Z",
                    "hello",
                    json.dumps([{"line": 1, "text": "hello"}]),
                    "sha1",
                    json.dumps({"k": "v"}),
                )
            ]
        if "FROM OPENSOURCEOPS.PUBLIC.REPO_SIGNALS" in su:
            return [("docs", "deployment_kubernetes", json.dumps(True), "ev_1", json.dumps([{"file_path": "README.md"}]))]
        return []

    monkeypatch.setattr(store, "execute", fake_execute)
    monkeypatch.setattr(store, "query", fake_query)

    store.save_repo_evidence_batch(
        ingest_run_id="ing1",
        repo_full_name="acme/repo",
        evidence_batch=[
            {
                "evidence_id": "ev_1",
                "repo_url": "https://github.com/acme/repo",
                "commit_or_ref": "main",
                "source_type": "github_file",
                "file_path": "README.md",
                "url": "https://github.com/acme/repo/blob/main/README.md",
                "api_ref": "/repos/acme/repo/contents/README.md",
                "retrieved_at": "2026-02-28T00:00:00Z",
                "content": "hello",
                "line_map": [{"line": 1, "text": "hello"}],
                "sha": "sha1",
                "structured": {"k": "v"},
            }
        ],
    )
    store.save_repo_signals(
        ingest_run_id="ing1",
        repo_full_name="acme/repo",
        signals=[{"category": "docs", "signal_name": "deployment_kubernetes", "value": True, "evidence_id": "ev_1", "citations": [{"file_path": "README.md"}]}],
    )

    ev = store.load_repo_evidence("ing1")
    sig = store.load_repo_signals("ing1")
    assert captured["evidence"]
    assert captured["signals"]
    assert ev and ev[0]["file_path"] == "README.md"
    assert sig and sig[0]["signal_name"] == "deployment_kubernetes"

