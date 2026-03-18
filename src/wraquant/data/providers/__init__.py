"""Built-in data providers."""

from __future__ import annotations

from wraquant._lazy import is_available
from wraquant.data.base import registry
from wraquant.data.providers.csv import CSVProvider


def register_builtin_providers() -> None:
    """Register all available built-in providers."""
    # CSV is always available
    registry.register(CSVProvider())

    # Yahoo Finance (yfinance)
    if is_available("yfinance"):
        from wraquant.data.providers.yahoo import YahooProvider

        registry.register(YahooProvider(), default=True)

    # FRED
    if is_available("fredapi"):
        from wraquant.data.providers.fred import FREDProvider

        registry.register(FREDProvider())

    # NASDAQ Data Link
    if is_available("nasdaqdatalink"):
        from wraquant.data.providers.nasdaq import NasdaqProvider

        registry.register(NasdaqProvider())
