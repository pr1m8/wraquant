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

    Retrieves a daily close price series from the specified data
    provider.  The default provider is determined by the registry
    (typically Yahoo Finance for equities).

    Parameters:
        symbol (str): Ticker symbol (e.g., ``'AAPL'``, ``'EURUSD=X'``,
            ``'BTC-USD'``).
        start (DateLike | None): Start date (string, datetime, or
            pandas Timestamp).  ``None`` fetches from the earliest
            available date.
        end (DateLike | None): End date.  ``None`` fetches up to today.
        source (str | None): Provider name (e.g., ``'yahoo'``,
            ``'fred'``).  ``None`` uses the default provider.
        **kwargs: Additional keyword arguments forwarded to the
            provider's ``fetch_prices`` method.

    Returns:
        pd.Series: Price series with a DatetimeIndex.

    Raises:
        DataFetchError: If the provider fails to fetch the data.

    Example:
        >>> prices = fetch_prices("AAPL", start="2020-01-01")  # doctest: +SKIP

    See Also:
        fetch_ohlcv: Fetch full OHLCV data.
        fetch_macro: Fetch macroeconomic series from FRED.
    """
    _ensure_providers_loaded()
    provider = registry.get(source)
    try:
        prices = provider.fetch_prices(symbol, start=start, end=end, **kwargs)
    except Exception as e:
        raise DataFetchError(provider.name, symbol, str(e)) from e

    # Wrap in PriceSeries for rich metadata (frequency, currency, to_returns)
    from wraquant.frame.base import PriceSeries

    if not isinstance(prices, PriceSeries) and len(prices) > 0:
        prices = PriceSeries(prices, currency="USD")
    return prices


def fetch_ohlcv(
    symbol: str,
    start: DateLike | None = None,
    end: DateLike | None = None,
    source: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Fetch OHLCV (Open, High, Low, Close, Volume) data for a symbol.

    Returns a DataFrame with standard column names suitable for
    backtesting, technical analysis, and charting.

    Parameters:
        symbol (str): Ticker symbol (e.g., ``'AAPL'``).
        start (DateLike | None): Start date.  ``None`` fetches from the
            earliest available date.
        end (DateLike | None): End date.  ``None`` fetches up to today.
        source (str | None): Provider name.  ``None`` uses the default
            provider.
        **kwargs: Additional keyword arguments forwarded to the
            provider.

    Returns:
        pd.DataFrame: DataFrame with columns ``open``, ``high``,
            ``low``, ``close``, ``volume`` and a DatetimeIndex.

    Raises:
        DataFetchError: If the provider fails to fetch the data.

    Example:
        >>> df = fetch_ohlcv("AAPL", start="2020-01-01")  # doctest: +SKIP

    See Also:
        fetch_prices: Fetch close prices only (lighter weight).
        fetch_macro: Fetch macroeconomic series.
    """
    _ensure_providers_loaded()
    provider = registry.get(source)
    try:
        ohlcv = provider.fetch_ohlcv(symbol, start=start, end=end, **kwargs)
    except Exception as e:
        raise DataFetchError(provider.name, symbol, str(e)) from e

    # Wrap in OHLCVFrame for rich metadata (frequency, currency, typed accessors)
    from wraquant.frame.base import OHLCVFrame

    if not isinstance(ohlcv, OHLCVFrame) and len(ohlcv) > 0:
        ohlcv = OHLCVFrame(ohlcv, currency="USD")
    return ohlcv


def fetch_macro(
    series_id: str,
    start: DateLike | None = None,
    end: DateLike | None = None,
    source: str = "fred",
    **kwargs: Any,
) -> pd.Series:
    """Fetch macroeconomic data series.

    Retrieves economic indicators from FRED (Federal Reserve Economic
    Data) or other macro data providers.  Common series include GDP,
    unemployment rate (UNRATE), federal funds rate (DFF), CPI, and
    Treasury yields.

    Parameters:
        series_id (str): Series identifier (e.g., ``'GDP'``,
            ``'UNRATE'``, ``'DFF'``, ``'T10Y2Y'``).
        start (DateLike | None): Start date.  ``None`` fetches the
            full history.
        end (DateLike | None): End date.  ``None`` fetches up to the
            latest available release.
        source (str): Provider name (default ``'fred'``).
        **kwargs: Additional keyword arguments forwarded to the
            provider.

    Returns:
        pd.Series: Macro data series with a DatetimeIndex.

    Raises:
        DataFetchError: If the provider fails to fetch the data.

    Example:
        >>> gdp = fetch_macro("GDP", source="fred")  # doctest: +SKIP

    See Also:
        fetch_prices: Fetch asset prices.
        fetch_ohlcv: Fetch OHLCV bar data.
    """
    _ensure_providers_loaded()
    provider = registry.get(source)
    try:
        return provider.fetch_macro(series_id, start=start, end=end, **kwargs)
    except Exception as e:
        raise DataFetchError(provider.name, series_id, str(e)) from e


def list_providers() -> list[str]:
    """List all available data providers.

    Returns the names of all registered providers (e.g., ``'yahoo'``,
    ``'fred'``, ``'nasdaq'``, ``'csv'``).  The list depends on which
    optional dependencies are installed.

    Returns:
        list[str]: List of registered provider names.

    Example:
        >>> providers = list_providers()  # doctest: +SKIP
        >>> "yahoo" in providers  # doctest: +SKIP
        True
    """
    _ensure_providers_loaded()
    return registry.list_providers()
