"""LLM router for OpenAI/xAI/Snowflake Cortex with strict JSON parsing."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests


class LLMRouter:
    """Routes model calls to configured provider with deterministic fallback support."""

    def __init__(self, snowflake_store: Any | None = None, timeout: int = 45) -> None:
        self.store = snowflake_store
        self.timeout = timeout
        self.provider = os.getenv("LLM_PROVIDER", "auto").lower()

    def enabled(self) -> bool:
        if self.provider in {"openai", "auto"} and os.getenv("OPENAI_API_KEY"):
            return True
        if self.provider in {"xai", "grok", "auto"} and os.getenv("XAI_API_KEY"):
            return True
        if self.provider in {"cortex", "snowflake", "auto"} and self.store is not None:
            return True
        return False

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        text = (raw or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json", "", 1).strip()
        return json.loads(text)

    def _call_openai_compatible(self, api_key: str, base_url: str, model: str, system: str, user: str) -> str:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return str((payload.get("choices") or [{}])[0].get("message", {}).get("content", ""))

    def _call_cortex(self, system: str, user: str) -> str:
        if self.store is None:
            raise RuntimeError("Snowflake store unavailable for Cortex call")
        model = os.getenv("CORTEX_MODEL_PRIMARY", "snowflake-arctic")
        prompt = f"{system}\n\n{user}"
        rows = self.store.query("SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s)", [model, prompt])
        if not rows:
            raise RuntimeError("Empty response from Snowflake Cortex")
        return str(rows[0][0])

    def complete_json(self, system: str, user: str) -> Dict[str, Any]:
        """Get strict JSON output from best-available model provider."""
        errors: List[str] = []
        order: List[str]
        if self.provider == "openai":
            order = ["openai"]
        elif self.provider in {"xai", "grok"}:
            order = ["xai"]
        elif self.provider in {"cortex", "snowflake"}:
            order = ["cortex"]
        else:
            order = ["openai", "xai", "cortex"]

        for p in order:
            try:
                if p == "openai" and os.getenv("OPENAI_API_KEY"):
                    raw = self._call_openai_compatible(
                        api_key=os.getenv("OPENAI_API_KEY", ""),
                        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                        system=system,
                        user=user,
                    )
                    return self._parse_json(raw)
                if p == "xai" and os.getenv("XAI_API_KEY"):
                    raw = self._call_openai_compatible(
                        api_key=os.getenv("XAI_API_KEY", ""),
                        base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
                        model=os.getenv("XAI_MODEL", "grok-2-latest"),
                        system=system,
                        user=user,
                    )
                    return self._parse_json(raw)
                if p == "cortex":
                    raw = self._call_cortex(system=system, user=user)
                    return self._parse_json(raw)
            except Exception as exc:
                errors.append(f"{p}: {exc}")
                continue
        raise RuntimeError("No LLM provider succeeded: " + " | ".join(errors))

