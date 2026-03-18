"""Signal generation utilities.

Helper functions for detecting crossovers, comparing series, and
computing rolling extremes. These are the building blocks used to
translate indicator values into actionable trading signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "crossover",
    "crossunder",
    "above",
    "below",
    "rising",
    "falling",
    "highest",
    "lowest",
    "normalize",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_series(data: pd.Series, name: str = "data") -> pd.Series:
    if not isinstance(data, pd.Series):
        raise TypeError(f"{name} must be a pd.Series, got {type(data).__name__}")
    return data


def _validate_period(period: int, name: str = "period") -> int:
    if period < 1:
        raise ValueError(f"{name} must be >= 1, got {period}")
    return period


def _to_series(data: pd.Series | float | int, ref: pd.Series) -> pd.Series:
    """Coerce a scalar to a pd.Series aligned with *ref*."""
    if isinstance(data, (int, float)):
        return pd.Series(data, index=ref.index)
    return data


# ---------------------------------------------------------------------------
# Crossover / Crossunder
# ---------------------------------------------------------------------------


def crossover(series1: pd.Series, series2: pd.Series | float | int) -> pd.Series:
    """Detect when *series1* crosses above *series2*.

    Parameters
    ----------
    series1 : pd.Series
        First data series.
    series2 : pd.Series | float | int
        Second data series or constant level.

    Returns
    -------
    pd.Series
        Boolean series — ``True`` on bars where *series1* crosses above
        *series2*.
    """
    _validate_series(series1, "series1")
    s2 = _to_series(series2, series1)

    crossed = (series1 > s2) & (series1.shift(1) <= s2.shift(1))
    crossed.name = "crossover"
    return crossed


def crossunder(series1: pd.Series, series2: pd.Series | float | int) -> pd.Series:
    """Detect when *series1* crosses below *series2*.

    Parameters
    ----------
    series1 : pd.Series
        First data series.
    series2 : pd.Series | float | int
        Second data series or constant level.

    Returns
    -------
    pd.Series
        Boolean series — ``True`` on bars where *series1* crosses below
        *series2*.
    """
    _validate_series(series1, "series1")
    s2 = _to_series(series2, series1)

    crossed = (series1 < s2) & (series1.shift(1) >= s2.shift(1))
    crossed.name = "crossunder"
    return crossed


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def above(series1: pd.Series, series2: pd.Series | float | int) -> pd.Series:
    """Element-wise ``series1 > series2``.

    Parameters
    ----------
    series1 : pd.Series
        First data series.
    series2 : pd.Series | float | int
        Second data series or constant level.

    Returns
    -------
    pd.Series
        Boolean series.
    """
    _validate_series(series1, "series1")
    s2 = _to_series(series2, series1)
    result = series1 > s2
    result.name = "above"
    return result


def below(series1: pd.Series, series2: pd.Series | float | int) -> pd.Series:
    """Element-wise ``series1 < series2``.

    Parameters
    ----------
    series1 : pd.Series
        First data series.
    series2 : pd.Series | float | int
        Second data series or constant level.

    Returns
    -------
    pd.Series
        Boolean series.
    """
    _validate_series(series1, "series1")
    s2 = _to_series(series2, series1)
    result = series1 < s2
    result.name = "below"
    return result


# ---------------------------------------------------------------------------
# Rising / Falling
# ---------------------------------------------------------------------------


def rising(data: pd.Series, period: int = 1) -> pd.Series:
    """Detect whether *data* is rising (each bar higher than *period* bars ago).

    Parameters
    ----------
    data : pd.Series
        Data series.
    period : int, default 1
        Number of bars to look back.

    Returns
    -------
    pd.Series
        Boolean series.
    """
    _validate_series(data)
    _validate_period(period)
    result = data > data.shift(period)
    result.name = "rising"
    return result


def falling(data: pd.Series, period: int = 1) -> pd.Series:
    """Detect whether *data* is falling (each bar lower than *period* bars ago).

    Parameters
    ----------
    data : pd.Series
        Data series.
    period : int, default 1
        Number of bars to look back.

    Returns
    -------
    pd.Series
        Boolean series.
    """
    _validate_series(data)
    _validate_period(period)
    result = data < data.shift(period)
    result.name = "falling"
    return result


# ---------------------------------------------------------------------------
# Rolling Extremes
# ---------------------------------------------------------------------------


def highest(data: pd.Series, period: int = 14) -> pd.Series:
    """Rolling highest value over *period* bars.

    Parameters
    ----------
    data : pd.Series
        Data series.
    period : int, default 14
        Rolling window size.

    Returns
    -------
    pd.Series
        Rolling maximum.
    """
    _validate_series(data)
    _validate_period(period)
    result = data.rolling(window=period, min_periods=period).max()
    result.name = "highest"
    return result


def lowest(data: pd.Series, period: int = 14) -> pd.Series:
    """Rolling lowest value over *period* bars.

    Parameters
    ----------
    data : pd.Series
        Data series.
    period : int, default 14
        Rolling window size.

    Returns
    -------
    pd.Series
        Rolling minimum.
    """
    _validate_series(data)
    _validate_period(period)
    result = data.rolling(window=period, min_periods=period).min()
    result.name = "lowest"
    return result


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize(data: pd.Series, period: int | None = None) -> pd.Series:
    """Z-score normalization.

    When *period* is ``None``, the full-series mean and standard deviation
    are used. When *period* is given, a rolling z-score is computed.

    Parameters
    ----------
    data : pd.Series
        Data series.
    period : int or None, default None
        Rolling window for mean/std. ``None`` uses the entire series.

    Returns
    -------
    pd.Series
        Z-score normalized values.
    """
    _validate_series(data)

    if period is None:
        mean = data.mean()
        std = data.std(ddof=0)
        if std == 0:
            result = pd.Series(0.0, index=data.index)
        else:
            result = (data - mean) / std
    else:
        _validate_period(period)
        mean = data.rolling(window=period, min_periods=period).mean()
        std = data.rolling(window=period, min_periods=period).std(ddof=0)
        result = (data - mean) / std
        # Where std is 0, set to 0 rather than inf/nan
        result = result.fillna(0.0)
        result = result.replace([np.inf, -np.inf], 0.0)

    result.name = "normalize"
    return result
