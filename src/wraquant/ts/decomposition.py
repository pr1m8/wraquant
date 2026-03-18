"""Time series decomposition methods."""

from __future__ import annotations

from typing import Any

import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import seasonal_decompose as sm_seasonal_decompose


def seasonal_decompose(
    data: pd.Series,
    period: int | None = None,
    model: str = "additive",
) -> Any:
    """Decompose a time series into trend, seasonal, and residual components.

    Wraps ``statsmodels.tsa.seasonal.seasonal_decompose``.

    Parameters:
        data: Time series to decompose.
        period: Seasonal period. Inferred from the index frequency when
            *None*.
        model: ``"additive"`` or ``"multiplicative"``.

    Returns:
        ``DecomposeResult`` with ``trend``, ``seasonal``, and ``resid``
        attributes.
    """
    return sm_seasonal_decompose(data, model=model, period=period)


def stl_decompose(
    data: pd.Series,
    period: int | None = None,
) -> Any:
    """STL (Seasonal and Trend decomposition using Loess) decomposition.

    Parameters:
        data: Time series to decompose.
        period: Seasonal period. Uses 7 when *None* and no index
            frequency is available.

    Returns:
        ``STL`` result with ``trend``, ``seasonal``, and ``resid``
        attributes.
    """
    if period is None:
        period = 7
    return STL(data, period=period).fit()


def trend_filter(
    data: pd.Series,
    method: str = "hp",
    lamb: float = 1600,
) -> pd.Series:
    """Extract the trend component from a time series.

    Parameters:
        data: Time series.
        method: Filter method — ``"hp"`` (Hodrick-Prescott, default).
        lamb: Smoothing parameter for the HP filter.

    Returns:
        Trend component as a Series.

    Raises:
        ValueError: If *method* is not recognized.
    """
    if method == "hp":
        _cycle, trend = hpfilter(data.dropna(), lamb=lamb)
        return trend
    msg = f"Unknown trend filter method: {method!r}"
    raise ValueError(msg)
