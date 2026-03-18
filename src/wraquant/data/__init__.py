"""Data fetching, cleaning, validation, transforms, and caching for financial data.

Provides a unified API for fetching prices, macroeconomic data, and
other financial time series from multiple sources (Yahoo Finance,
FRED, NASDAQ Data Link, CSV files), plus cleaning, validation, and
transformation utilities.

Example:
    >>> from wraquant.data import fetch_prices, fetch_macro
    >>> prices = fetch_prices("AAPL", start="2020-01-01")
    >>> gdp = fetch_macro("GDP", source="fred")
"""

from wraquant.data.base import DataProvider, ProviderRegistry, registry
from wraquant.data.cleaning import (
    align_series,
    detect_outliers,
    fill_missing,
    handle_splits_dividends,
    remove_duplicates,
    remove_outliers,
    resample_ohlcv,
    winsorize,
)
from wraquant.data.loaders import fetch_macro, fetch_ohlcv, fetch_prices, list_providers
from wraquant.data.transforms import (
    expanding_zscore,
    normalize_prices,
    percentile_rank,
    rank_transform,
    rolling_zscore,
    to_excess_returns,
    to_prices,
    to_returns,
)
from wraquant.data.validation import (
    check_completeness,
    check_staleness,
    data_quality_report,
    validate_ohlcv,
    validate_returns,
)

__all__ = [
    # providers
    "DataProvider",
    "ProviderRegistry",
    "registry",
    "fetch_prices",
    "fetch_ohlcv",
    "fetch_macro",
    "list_providers",
    # cleaning
    "align_series",
    "detect_outliers",
    "fill_missing",
    "handle_splits_dividends",
    "remove_duplicates",
    "remove_outliers",
    "resample_ohlcv",
    "winsorize",
    # transforms
    "expanding_zscore",
    "normalize_prices",
    "percentile_rank",
    "rank_transform",
    "rolling_zscore",
    "to_excess_returns",
    "to_prices",
    "to_returns",
    # validation
    "check_completeness",
    "check_staleness",
    "data_quality_report",
    "validate_ohlcv",
    "validate_returns",
]
