"""Composio integration for Notion playbook creation with mandatory dry-run support."""

from __future__ import annotations

import os
from typing import Any, Dict

try:
    from composio import Composio
except Exception:  # pragma: no cover
    Composio = None


def _to_notion_blocks(markdown_text: str) -> list[dict]:
    """Simple markdown-to-Notion paragraph blocks conversion."""
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
    playbook_md: str,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Create or preview a Notion page creation payload via Composio SDK."""
    notion_database_id = os.getenv("NOTION_DATABASE_ID")
    notion_parent_page_id = os.getenv("NOTION_PARENT_PAGE_ID")

    payload = {
        "title": f"OpenSourceOps Playbook - {domain}",
        "run_id": run_id,
        "database_id": notion_database_id,
        "parent_page_id": notion_parent_page_id,
        "children": _to_notion_blocks(playbook_md),
    }

    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "message": "Dry-run mode: payload generated without API call.",
            "payload": payload,
            "url": None,
        }

    if not composio_client:
        return {
            "ok": False,
            "dry_run": False,
            "message": "Composio client not available.",
            "payload": payload,
            "url": None,
        }

    try:
        if hasattr(composio_client, "tools") and hasattr(composio_client.tools, "execute"):
            response = composio_client.tools.execute(
                "NOTION_CREATE_PAGE",
                {
                    "title": payload["title"],
                    "database_id": payload["database_id"],
                    "parent_page_id": payload["parent_page_id"],
                    "children": payload["children"],
                },
            )
        else:
            response = {"data": {"url": None}}

        page_url = (
            (response or {}).get("data", {}).get("url")
            if isinstance(response, dict)
            else None
        )

        return {
            "ok": True,
            "dry_run": False,
            "message": "Notion page created.",
            "payload": payload,
            "url": page_url,
            "raw": response,
        }
    except Exception as exc:
        return {
            "ok": False,
            "dry_run": False,
            "message": f"Notion creation failed: {exc}",
            "payload": payload,
            "url": None,
        }


def get_composio_client() -> Any:
    """Build Composio SDK client if available and configured."""
    api_key = os.getenv("COMPOSIO_API_KEY")
    if not api_key or Composio is None:
        return None
    try:
        return Composio(api_key=api_key)
    except Exception:
        return None


def create_internal_github_artifact(
    composio_client: Any,
    repo: str,
    title: str,
    body: str,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Optional Composio helper to create an internal issue/discussion reference."""
    payload = {"repo": repo, "title": title, "body": body}
    if dry_run:
        return {"ok": True, "dry_run": True, "message": "Dry-run only", "payload": payload}
    if not composio_client:
        return {"ok": False, "dry_run": False, "message": "Composio client unavailable", "payload": payload}
    try:
        if hasattr(composio_client, "tools") and hasattr(composio_client.tools, "execute"):
            result = composio_client.tools.execute(
                "GITHUB_CREATE_ISSUE",
                {"repo": repo, "title": title, "body": body},
            )
        else:
            result = {}
        return {"ok": True, "dry_run": False, "message": "GitHub artifact created", "payload": payload, "raw": result}
    except Exception as exc:
        return {"ok": False, "dry_run": False, "message": f"GitHub artifact failed: {exc}", "payload": payload}
