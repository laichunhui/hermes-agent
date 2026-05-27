"""Bocha AI Search (博查) — plugin form.

Bocha is a China-domestic web search API optimised for Chinese content and
low-latency access from cn-* networks. Search-only — pair with
Firecrawl / Tavily / Exa for ``web_extract``.

Config keys this provider responds to::

    web:
      search_backend: "bocha"     # explicit per-capability
      backend: "bocha"            # shared fallback

Auth env var::

    BOCHA_API_KEY=...   # https://open.bochaai.com (free trial quota available)

API reference: https://docs.bochaai.com/api-reference/endpoint/web-search
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_BOCHA_ENDPOINT = "https://api.bochaai.com/v1/web-search"


class BochaWebSearchProvider(WebSearchProvider):
    """Search-only Bocha provider using the open.bochaai.com Web Search API."""

    @property
    def name(self) -> str:
        return "bocha"

    @property
    def display_name(self) -> str:
        return "Bocha AI Search"

    def is_available(self) -> bool:
        """Return True when ``BOCHA_API_KEY`` is set to a non-empty value."""
        return bool(os.getenv("BOCHA_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a search against the Bocha Web Search API.

        Returns ``{"success": True, "data": {"web": [{"title", "url", "description", "position"}]}}``
        on success, or ``{"success": False, "error": str}`` on failure.
        """
        import httpx

        api_key = os.getenv("BOCHA_API_KEY", "").strip()
        if not api_key:
            return {"success": False, "error": "BOCHA_API_KEY is not set"}

        # Bocha accepts up to 50 results per query; clamp defensively.
        count = max(1, min(int(limit), 50))

        payload = {
            "query": query,
            "freshness": "noLimit",
            "summary": True,
            "count": count,
        }

        try:
            resp = httpx.post(
                _BOCHA_ENDPOINT,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=20,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("Bocha Search HTTP error: %s", exc)
            return {
                "success": False,
                "error": f"Bocha Search returned HTTP {exc.response.status_code}",
            }
        except httpx.RequestError as exc:
            logger.warning("Bocha Search request error: %s", exc)
            return {"success": False, "error": f"Could not reach Bocha Search: {exc}"}

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Bocha Search response parse error: %s", exc)
            return {"success": False, "error": "Could not parse Bocha Search response as JSON"}

        # Bocha surfaces application-level errors via top-level ``code`` /
        # ``msg`` fields (HTTP 200 with non-200 ``code``).
        bocha_code = data.get("code")
        if bocha_code is not None and int(bocha_code) != 200:
            err_msg = str(data.get("msg") or data.get("message") or f"Bocha code {bocha_code}")
            logger.warning("Bocha Search application error: %s", err_msg)
            return {"success": False, "error": f"Bocha Search error: {err_msg}"}

        # Response shape: { "code": 200, "data": { "webPages": { "value": [...] } } }
        body = data.get("data") if isinstance(data.get("data"), dict) else data
        web_pages = (body.get("webPages") or {}) if isinstance(body, dict) else {}
        raw_results = web_pages.get("value", []) or []
        truncated = raw_results[:limit]

        web_results = []
        for i, r in enumerate(truncated):
            # Prefer the longer AI-generated summary when present, fall back
            # to the snippet so we always have a non-empty description.
            description = str(r.get("summary") or r.get("snippet") or "")
            web_results.append({
                "title": str(r.get("name", "")),
                "url": str(r.get("url", "")),
                "description": description,
                "position": i + 1,
            })

        logger.info(
            "Bocha Search '%s': %d results (from %d raw, limit %d)",
            query,
            len(web_results),
            len(raw_results),
            limit,
        )

        return {"success": True, "data": {"web": web_results}}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Bocha AI Search",
            "badge": "cn",
            "tag": "China-domestic web search API. Free trial quota; search only.",
            "env_vars": [
                {
                    "key": "BOCHA_API_KEY",
                    "prompt": "Bocha API key",
                    "url": "https://open.bochaai.com",
                },
            ],
        }
