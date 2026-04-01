"""Unified DataFrame/Series abstraction over pandas, polars, and NumPy.

Provides a consistent API for financial time series operations regardless
of the underlying backend.  The core financial types -- ``PriceSeries``,
``ReturnSeries``, ``OHLCVFrame``, ``ReturnFrame`` -- are pandas subclasses
that carry frequency, currency, and return-type metadata through all
operations, ensuring that downstream functions always receive properly
typed and annotated data.

Key components:

- **PriceSeries / ReturnSeries** -- Metadata-aware Series subclasses that
  preserve ``frequency`` and ``currency`` through slicing, arithmetic, and
  resampling.  Use ``PriceSeries`` for raw price data and ``ReturnSeries``
  for percentage or log returns.
- **OHLCVFrame / ReturnFrame** -- DataFrame subclasses for OHLCV bars and
  multi-asset return panels, respectively.
- **frame / series** -- Factory functions that auto-detect input type
  (list, numpy array, pandas, polars) and produce the appropriate
  financial type with metadata attached.
- **returns / log_returns / cumulative_returns** -- Core return computation
  functions that work across backends.
- **rolling_mean / rolling_std / ewm_mean / drawdowns / resample** --
  Backend-agnostic time series operations.

Example:
    >>> from wraquant.frame import series, returns, rolling_std
    >>> prices = series([100, 102, 101, 105, 103], name="AAPL")
    >>> rets = returns(prices)
    >>> vol = rolling_std(rets, window=3)

Use ``wraquant.frame`` when you need to construct typed financial objects
from raw data or perform backend-agnostic operations.  For data fetching
from external sources, use ``wraquant.data`` instead; the data module
returns ``PriceSeries`` and ``OHLCVFrame`` objects built on this module.
"""

from wraquant.frame.base import (
    AbstractFrame,
    AbstractSeries,
    OHLCVFrame,
    PriceSeries,
    ReturnFrame,
    ReturnSeries,
)
from wraquant.frame.factory import frame, series
from wraquant.frame.ops import (
    cumulative_returns,
    drawdowns,
    ewm_mean,
    log_returns,
    resample,
    returns,
    rolling_mean,
    rolling_std,
)

__all__ = [
    # Financial types
    "PriceSeries",
    "ReturnSeries",
    "OHLCVFrame",
    "ReturnFrame",
    # Legacy protocols
    "AbstractFrame",
    "AbstractSeries",
    # Factory
    "frame",
    "series",
    # Ops
    "returns",
    "log_returns",
    "cumulative_returns",
    "drawdowns",
    "rolling_mean",
    "rolling_std",
    "ewm_mean",
    "resample",
]
