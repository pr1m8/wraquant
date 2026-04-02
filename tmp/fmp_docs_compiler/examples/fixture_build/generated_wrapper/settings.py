"""Generated runtime settings.

Purpose:
    Provide a minimal settings object for the generated wrapper.

Design:
    The settings are intentionally plain so users can replace them with
    Pydantic Settings or another configuration system later.

Attributes:
    None.

Examples:
    ::
        >>> get_settings().upstream_base_url.startswith('https://')
        True
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Settings:
    upstream_base_url: str = "https://financialmodelingprep.com"
    api_key: str | None = None
    proxy_enabled: bool = False


def get_settings() -> Settings:
    return Settings()
