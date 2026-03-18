"""Data fetching, provider registry, and caching for financial data.

Provides a unified API for fetching prices, macroeconomic data, and
other financial time series from multiple sources (Yahoo Finance,
FRED, NASDAQ Data Link, CSV files).

Example:
    >>> from wraquant.data import fetch_prices, fetch_macro
    >>> prices = fetch_prices("AAPL", start="2020-01-01")
    >>> gdp = fetch_macro("GDP", source="fred")
"""

from wraquant.data.base import DataProvider, ProviderRegistry, registry
from wraquant.data.loaders import fetch_macro, fetch_ohlcv, fetch_prices, list_providers

__all__ = [
    "DataProvider",
    "ProviderRegistry",
    "registry",
    "fetch_prices",
    "fetch_ohlcv",
    "fetch_macro",
    "list_providers",
]
