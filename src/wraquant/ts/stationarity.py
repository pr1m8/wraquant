"""Stationarity transformations for time series."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import detrend as sp_detrend


def difference(data: pd.Series, order: int = 1) -> pd.Series:
    """Apply integer differencing to a time series.

    Parameters:
        data: Time series.
        order: Differencing order (1 = first difference).

    Returns:
        Differenced series with NaN values dropped.
    """
    result = data
    for _ in range(order):
        result = result.diff()
    return result.dropna()


def fractional_difference(
    data: pd.Series,
    d: float = 0.5,
    threshold: float = 1e-5,
) -> pd.Series:
    """Apply fractional differencing to preserve long-memory information.

    Implements the fixed-width window fracdiff method from
    *Advances in Financial Machine Learning* (Lopez de Prado).

    Parameters:
        data: Time series.
        d: Fractional differencing parameter (0 < d < 1).
        threshold: Weight cutoff threshold for the window.

    Returns:
        Fractionally differenced series.
    """
    # Compute binomial weights, capped at the length of the data
    n = len(data)
    weights = [1.0]
    k = 1
    while k < n:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1

    weights = np.array(weights[::-1])
    width = len(weights)

    result = {}
    values = data.values
    for i in range(width - 1, len(values)):
        result[data.index[i]] = np.dot(weights, values[i - width + 1 : i + 1])

    return pd.Series(result, dtype=float)


def detrend(data: pd.Series, method: str = "linear") -> pd.Series:
    """Remove trend from a time series.

    Parameters:
        data: Time series.
        method: Detrending method — ``"linear"`` (default) or
            ``"constant"`` (demean).

    Returns:
        Detrended series.

    Raises:
        ValueError: If *method* is not recognized.
    """
    clean = data.dropna()

    if method in ("linear", "constant"):
        detrended = sp_detrend(clean.values, type=method)
        return pd.Series(detrended, index=clean.index, name=data.name)
    msg = f"Unknown detrend method: {method!r}"
    raise ValueError(msg)
