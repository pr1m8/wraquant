"""Generated upstream client adapter.

Purpose:
    Encapsulate upstream HTTP calls for the generated wrapper.

Design:
    The adapter keeps request construction in one place and supports dry-run
    mode when proxying is disabled.

Attributes:
    None.

Examples:
    ::
        >>> callable(get_upstream_client)
        True
"""

from __future__ import annotations

from typing import Any

import httpx

from .settings import Settings, get_settings


class FMPUpstreamClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def call(self, path: str, query: dict[str, Any]) -> dict[str, Any]:
        filtered_query = {
            key: value for key, value in query.items() if value is not None
        }
        if self.settings.api_key:
            filtered_query.setdefault("apikey", self.settings.api_key)
        if not self.settings.proxy_enabled:
            return {"mode": "dry_run", "path": path, "query": filtered_query}
        async with httpx.AsyncClient(
            base_url=self.settings.upstream_base_url, follow_redirects=True
        ) as client:
            response = await client.get(path, params=filtered_query)
            response.raise_for_status()
            return response.json()


def get_upstream_client(settings: Settings | None = None) -> FMPUpstreamClient:
    return FMPUpstreamClient(settings or get_settings())
