import os

import pytest

from src.snowflake_client import SnowflakeClient


def test_snowflake_connection_or_skip():
    account = os.getenv("SNOWFLAKE_ACCOUNT")
    user = os.getenv("SNOWFLAKE_USER")
    auth = os.getenv("SNOWFLAKE_PASSWORD") or os.getenv("SNOWFLAKE_PRIVATE_KEY")

    client = SnowflakeClient()

    if not (account and user and auth):
        assert client.is_configured() is False
        return

    try:
        conn = client.connect()
    except Exception as exc:
        pytest.skip(f"Snowflake configured but not reachable/valid in this environment: {exc}")
    assert conn is not None
