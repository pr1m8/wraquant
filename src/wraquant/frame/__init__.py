"""Unified DataFrame/Series abstraction over pandas, polars, and NumPy.

Provides a consistent API for financial time series operations regardless
of the underlying backend.  The core financial types -- ``PriceSeries``,
``ReturnSeries``, ``OHLCVFrame``, ``ReturnFrame`` -- are pandas subclasses
that carry frequency, currency, and return-type metadata through all
operations.
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
