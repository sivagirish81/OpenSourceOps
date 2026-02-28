from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from storage.snowflake_store import SnowflakeConfig, SnowflakeStore


class FakeCursor:
    def __init__(self, conn: "FakeConn"):
        self.conn = conn
        self.last_sql = ""
        self.last_params: list[Any] = []

    def execute(self, sql: str, params=()):
        self.last_sql = sql
        self.last_params = list(params or [])
        self.conn.executed.append((sql, list(params or [])))
        return self

    def fetchall(self):
        sql_u = self.last_sql.upper()
        if "CURRENT_ACCOUNT()" in sql_u:
            return [("ACCT", "USER", "ROLE")]
        if "FROM OPENSOURCEOPS.PUBLIC.COMPANY_PROFILES" in sql_u and "WHERE COMPANY_ID" in sql_u:
            cid = self.last_params[0]
            profile = self.conn.company_profiles.get(cid)
            return [(json.dumps(profile),)] if profile else []
        if "FROM OPENSOURCEOPS.PUBLIC.COMPANY_PROFILES" in sql_u:
            out = []
            for cid, profile in self.conn.company_profiles.items():
                out.append((cid, "2026-01-01", "2026-01-01", json.dumps(profile)))
            return out
        return []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConn:
    def __init__(self):
        self.executed: list[tuple[str, list[Any]]] = []
        self.company_profiles: dict[str, dict] = {}

    def cursor(self):
        return FakeCursor(self)


def test_schema_creation_and_company_profile_write_read(monkeypatch):
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
    fake_conn = FakeConn()

    monkeypatch.setattr(store, "connect", lambda: fake_conn)

    schema_path = Path("storage/schema.sql")
    assert schema_path.exists()
    store.ensure_schema(str(schema_path))
    assert any("CREATE TABLE IF NOT EXISTS COMPANY_PROFILES" in sql for sql, _ in fake_conn.executed)

    profile = {"organization": {"name": "Acme"}, "risk_appetite": {"risk_tolerance": "Low"}}

    def fake_execute(sql: str, params=None):
        if "INSERT INTO OPENSOURCEOPS.PUBLIC.COMPANY_PROFILES" in sql.upper():
            cid = params[0]
            payload = json.loads(params[1])
            fake_conn.company_profiles[cid] = payload

    monkeypatch.setattr(store, "execute", fake_execute)

    cid = store.save_company_profile(profile, company_id="co_test")
    assert cid == "co_test"

    # Restore query path via fake cursor-backed query.
    def fake_query(sql, params=None):
        sql_u = sql.upper()
        if "WHERE COMPANY_ID" in sql_u:
            return [(json.dumps(profile),)]
        if "ORDER BY UPDATED_AT DESC" in sql_u:
            return [("co_test", "2026-01-01", "2026-01-01", json.dumps(profile))]
        return []

    monkeypatch.setattr(store, "query", fake_query)

    rows = store.list_company_profiles()
    assert rows and rows[0]["company_id"] == "co_test"
    got = store.get_company_profile("co_test")
    assert got["organization"]["name"] == "Acme"
