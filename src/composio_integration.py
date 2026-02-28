"""Composio integration for Notion playbook creation with dry-run support and action logging."""

from __future__ import annotations

import os
from typing import Any, Dict

try:
    from composio import Composio
except Exception:  # pragma: no cover
    Composio = None

from src.run_logger import RunLogger
from src.snowflake_client import SnowflakeClient


def _to_notion_blocks(markdown_text: str) -> list[dict]:
    blocks = []
    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("# "):
            blocks.append(
                {
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {"rich_text": [{"type": "text", "text": {"content": line[2:]}}]},
                }
            )
        elif line.startswith("## "):
            blocks.append(
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"type": "text", "text": {"content": line[3:]}}]},
                }
            )
        elif line.startswith("- "):
            blocks.append(
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": line[2:]}}]
                    },
                }
            )
        else:
            blocks.append(
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"type": "text", "text": {"content": line}}]},
                }
            )
    return blocks


def create_notion_playbook(
    composio_client: Any,
    run_id: str,
    domain: str,
    persona: str,
    playbook_md: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Create or preview a Notion page via Composio; logs RUN_ACTIONS and RUN_LOG."""
    sf = SnowflakeClient()
    logger = RunLogger(sf)

    notion_database_id = os.getenv("NOTION_DATABASE_ID")
    notion_parent_page_id = os.getenv("NOTION_PARENT_PAGE_ID")
    payload = {
        "title": f"OpenSourceOps {persona} Output - {domain}",
        "database_id": notion_database_id,
        "parent_page_id": notion_parent_page_id,
        "children": _to_notion_blocks(playbook_md),
    }

    if dry_run:
        sf.save_action(run_id, "CREATE_NOTION_PLAYBOOK", "DRY_RUN", None, "Payload preview only")
        logger.log(run_id, "CoordinatorAgent", "notion_dry_run", "INFO", "Notion payload generated in dry-run mode")
        return {
            "ok": True,
            "dry_run": True,
            "message": "Dry-run mode: payload generated without API call.",
            "payload": payload,
            "url": None,
        }

    if not composio_client:
        sf.save_action(run_id, "CREATE_NOTION_PLAYBOOK", "FAILED", None, "Composio client not available")
        logger.log(run_id, "CoordinatorAgent", "notion_create", "FAILED", "Composio client not available")
        return {
            "ok": False,
            "dry_run": False,
            "message": "Composio client not available.",
            "payload": payload,
            "url": None,
        }

    try:
        response = composio_client.tools.execute(
            "NOTION_CREATE_PAGE",
            {
                "title": payload["title"],
                "database_id": payload["database_id"],
                "parent_page_id": payload["parent_page_id"],
                "children": payload["children"],
            },
        )
        page_url = (response or {}).get("data", {}).get("url") if isinstance(response, dict) else None
        sf.save_action(run_id, "CREATE_NOTION_PLAYBOOK", "SUCCESS", page_url, "Notion page created")
        logger.log(run_id, "CoordinatorAgent", "notion_create", "SUCCESS", f"Notion page created: {page_url}")
        return {
            "ok": True,
            "dry_run": False,
            "message": "Notion page created.",
            "payload": payload,
            "url": page_url,
            "raw": response,
        }
    except Exception as exc:
        sf.save_action(run_id, "CREATE_NOTION_PLAYBOOK", "FAILED", None, str(exc))
        logger.log(run_id, "CoordinatorAgent", "notion_create", "FAILED", f"Notion create failed: {exc}")
        return {
            "ok": False,
            "dry_run": False,
            "message": f"Notion creation failed: {exc}",
            "payload": payload,
            "url": None,
        }


def get_composio_client() -> Any:
    api_key = os.getenv("COMPOSIO_API_KEY")
    if not api_key or Composio is None:
        return None
    try:
        return Composio(api_key=api_key)
    except Exception:
        return None
