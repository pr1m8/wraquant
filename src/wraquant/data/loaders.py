"""High-level data loading API.

Convenience functions that delegate to the provider registry.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from wraquant.core.exceptions import DataFetchError
from wraquant.core.types import DateLike
from wraquant.data.base import registry


def _ensure_providers_loaded() -> None:
    """Lazily load built-in providers on first use."""
    if not registry.list_providers():
        from wraquant.data.providers import register_builtin_providers

        register_builtin_providers()


def fetch_prices(
    symbol: str,
    start: DateLike | None = None,
    end: DateLike | None = None,
    source: str | None = None,
    **kwargs: Any,
) -> pd.Series:
    """Fetch closing prices for a symbol.

    Parameters:
        symbol: Ticker symbol (e.g., 'AAPL', 'EURUSD=X').
        start: Start date.
        end: End date.
        source: Provider name (e.g., 'yahoo', 'fred'). None uses default.

    Returns:
        Price series with DatetimeIndex.

    Example:
        >>> from wraquant.data import fetch_prices
        >>> prices = fetch_prices("AAPL", start="2020-01-01")  # doctest: +SKIP
    """
    _ensure_providers_loaded()
    provider = registry.get(source)
    try:
        return provider.fetch_prices(symbol, start=start, end=end, **kwargs)
    except Exception as e:
        raise DataFetchError(provider.name, symbol, str(e)) from e


def fetch_ohlcv(
    symbol: str,
    start: DateLike | None = None,
    end: DateLike | None = None,
    source: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Fetch OHLCV data for a symbol.

    Parameters:
        symbol: Ticker symbol.
        start: Start date.
        end: End date.
        source: Provider name. None uses default.

    Returns:
        DataFrame with open, high, low, close, volume columns.

    Example:
        >>> from wraquant.data import fetch_ohlcv
        >>> df = fetch_ohlcv("AAPL", start="2020-01-01")  # doctest: +SKIP
    """
    _ensure_providers_loaded()
    provider = registry.get(source)
    try:
        return provider.fetch_ohlcv(symbol, start=start, end=end, **kwargs)
    except Exception as e:
        raise DataFetchError(provider.name, symbol, str(e)) from e


def fetch_macro(
    series_id: str,
    start: DateLike | None = None,
    end: DateLike | None = None,
    source: str = "fred",
    **kwargs: Any,
) -> pd.Series:
    """Fetch macroeconomic data series.

    Parameters:
        series_id: Series identifier (e.g., 'GDP', 'UNRATE', 'DFF').
        start: Start date.
        end: End date.
        source: Provider name. Defaults to 'fred'.

    Returns:
        Macro data series with DatetimeIndex.

    Example:
        >>> from wraquant.data import fetch_macro
        >>> gdp = fetch_macro("GDP", source="fred")  # doctest: +SKIP
    """
    _ensure_providers_loaded()
    provider = registry.get(source)
    try:
        return provider.fetch_macro(series_id, start=start, end=end, **kwargs)
    except Exception as e:
        raise DataFetchError(provider.name, series_id, str(e)) from e


def list_providers() -> list[str]:
    """List all available data providers.

    Returns:
        List of registered provider names.
    """
    _ensure_providers_loaded()
    return registry.list_providers()
