"""Data fetching, cleaning, validation, transforms, and caching for financial data.

Provides a unified API for fetching prices, macroeconomic data, and
other financial time series from multiple sources (Yahoo Finance,
FRED, NASDAQ Data Link, CSV files), plus cleaning, validation, and
transformation utilities.  This module is the primary entry point for
getting data into wraquant and preparing it for downstream analysis.

Key sub-modules:

- **Fetching** (``loaders``) -- ``fetch_prices`` retrieves OHLCV data,
  ``fetch_macro`` retrieves macroeconomic indicators, and ``fetch_ohlcv``
  returns full OHLCV bars.  A pluggable ``ProviderRegistry`` allows
  custom data sources.
- **Cleaning** (``cleaning``, ``cleaning_advanced``) -- Handle missing
  values (``fill_missing``), outliers (``detect_outliers``,
  ``remove_outliers``, ``winsorize``), duplicates, split/dividend
  adjustments, OHLCV resampling, fuzzy merges, and date parsing.
- **Validation** (``validation``, ``validation_advanced``) -- Check data
  quality with ``validate_ohlcv``, ``validate_returns``,
  ``check_completeness``, ``check_staleness``, and ``data_quality_report``.
  Schema-based validation via pandera is available through
  ``pandera_validate``.
- **Transforms** (``transforms``) -- Convert between prices and returns
  (``to_returns``, ``to_prices``), compute excess returns, normalize
  prices, and calculate rolling/expanding z-scores.

Example:
    >>> from wraquant.data import fetch_prices, validate_ohlcv, to_returns
    >>> prices = fetch_prices("AAPL", start="2020-01-01")
    >>> report = validate_ohlcv(prices)
    >>> returns = to_returns(prices["close"])

Use ``wraquant.data`` for all data ingestion and preparation.  For file-
and database-based I/O (Parquet, HDF5, SQL, cloud storage), use
``wraquant.io`` instead.  The data module feeds cleaned data into
``wraquant.stats``, ``wraquant.ts``, ``wraquant.backtest``, and all
other analytical modules.
"""

from wraquant.data.base import DataProvider, ProviderRegistry, registry
from wraquant.data.cleaning_advanced import (
    fix_text,
    fuzzy_merge,
    janitor_clean_names,
    janitor_remove_empty,
    normalize_countries,
    parse_dates_flexible,
    parse_prices,
)
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
from wraquant.data.validation_advanced import (
    create_ohlcv_schema,
    create_returns_schema,
    pandera_validate,
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
    # cleaning_advanced
    "janitor_clean_names",
    "janitor_remove_empty",
    "fuzzy_merge",
    "parse_dates_flexible",
    "parse_prices",
    "normalize_countries",
    "fix_text",
    # validation_advanced
    "pandera_validate",
    "create_ohlcv_schema",
    "create_returns_schema",
]
