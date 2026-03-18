"""Unified DataFrame/Series abstraction over pandas, polars, and NumPy.

Provides a consistent API for financial time series operations regardless
of the underlying backend.
"""

from wraquant.frame.base import AbstractFrame, AbstractSeries
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
    "AbstractFrame",
    "AbstractSeries",
    "frame",
    "series",
    "returns",
    "log_returns",
    "cumulative_returns",
    "drawdowns",
    "rolling_mean",
    "rolling_std",
    "ewm_mean",
    "resample",
]
