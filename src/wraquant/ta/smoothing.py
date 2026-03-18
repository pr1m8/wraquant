"""Advanced smoothing and filtering indicators.

This module provides sophisticated moving averages and digital filters used
in technical analysis. All functions accept ``pd.Series`` inputs and return
``pd.Series``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "alma",
    "lsma",
    "swma",
    "sinema",
    "trima",
    "jma",
    "gaussian_filter",
    "butterworth_filter",
    "supersmoother",
    "hann_window_ma",
    "hamming_window_ma",
    "kaufman_efficiency_ratio",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_series(data: pd.Series, name: str = "data") -> pd.Series:
    """Ensure *data* is a ``pd.Series``; raise ``TypeError`` otherwise."""
    if not isinstance(data, pd.Series):
        raise TypeError(f"{name} must be a pd.Series, got {type(data).__name__}")
    return data


def _validate_period(period: int, name: str = "period") -> int:
    if period < 1:
        raise ValueError(f"{name} must be >= 1, got {period}")
    return period


# ---------------------------------------------------------------------------
# ALMA — Arnaud Legoux Moving Average
# ---------------------------------------------------------------------------


def alma(
    data: pd.Series,
    period: int = 9,
    offset: float = 0.85,
    sigma: float = 6.0,
) -> pd.Series:
    """Arnaud Legoux Moving Average (ALMA).

    A Gaussian-weighted moving average that allows the user to control the
    position of the bell curve along the window via *offset* and the width
    via *sigma*.

    Parameters
    ----------
    data : pd.Series
        Price series (typically close).
    period : int, default 9
        Window length.
    offset : float, default 0.85
        Controls the position of the Gaussian peak within the window.
        0 = far left (oldest), 1 = far right (newest).
    sigma : float, default 6.0
        Controls the width of the Gaussian bell curve. Higher values
        produce a broader, smoother curve.

    Returns
    -------
    pd.Series
        ALMA values. The first ``period - 1`` entries are ``NaN``.

    Example
    -------
    >>> result = alma(close, period=9, offset=0.85, sigma=6.0)
    """
    _validate_series(data)
    _validate_period(period)

    m = offset * (period - 1)
    s = period / sigma

    weights = np.array(
        [np.exp(-((i - m) ** 2) / (2.0 * s * s)) for i in range(period)]
    )
    weights = weights / weights.sum()

    def _alma(window: np.ndarray) -> float:
        return np.dot(window, weights)

    result = data.rolling(window=period, min_periods=period).apply(_alma, raw=True)
    result.name = "alma"
    return result


# ---------------------------------------------------------------------------
# LSMA — Least Squares Moving Average
# ---------------------------------------------------------------------------


def lsma(data: pd.Series, period: int = 25) -> pd.Series:
    """Least Squares Moving Average (LSMA).

    Also known as the Linear Regression Value or End Point Moving Average.
    At each bar, a least-squares line is fit over the window and the
    endpoint of the line is returned.

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 25
        Window length for the linear regression.

    Returns
    -------
    pd.Series
        LSMA values.

    Example
    -------
    >>> result = lsma(close, period=25)
    """
    _validate_series(data)
    _validate_period(period)

    def _linreg_endpoint(window: np.ndarray) -> float:
        n = len(window)
        x = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(x, window, 1)
        return intercept + slope * (n - 1)

    result = data.rolling(window=period, min_periods=period).apply(
        _linreg_endpoint, raw=True
    )
    result.name = "lsma"
    return result


# ---------------------------------------------------------------------------
# SWMA — Symmetrically Weighted Moving Average
# ---------------------------------------------------------------------------


def swma(data: pd.Series) -> pd.Series:
    """Symmetrically Weighted Moving Average (SWMA).

    A 4-bar weighted average using weights ``[1, 2, 2, 1] / 6``.

    Parameters
    ----------
    data : pd.Series
        Price series.

    Returns
    -------
    pd.Series
        SWMA values. The first 3 entries are ``NaN``.

    Example
    -------
    >>> result = swma(close)
    """
    _validate_series(data)

    weights = np.array([1.0, 2.0, 2.0, 1.0]) / 6.0

    def _swma(window: np.ndarray) -> float:
        return np.dot(window, weights)

    result = data.rolling(window=4, min_periods=4).apply(_swma, raw=True)
    result.name = "swma"
    return result


# ---------------------------------------------------------------------------
# SineMA — Sine-Weighted Moving Average
# ---------------------------------------------------------------------------


def sinema(data: pd.Series, period: int = 14) -> pd.Series:
    """Sine-Weighted Moving Average.

    Each element in the window is weighted by the sine of its proportional
    position within a half-period (pi), giving the most weight to the
    center of the window.

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 14
        Window length.

    Returns
    -------
    pd.Series
        Sine-weighted moving average values.

    Example
    -------
    >>> result = sinema(close, period=14)
    """
    _validate_series(data)
    _validate_period(period)

    weights = np.array(
        [np.sin(np.pi * (i + 1) / (period + 1)) for i in range(period)]
    )
    weights = weights / weights.sum()

    def _sinema(window: np.ndarray) -> float:
        return np.dot(window, weights)

    result = data.rolling(window=period, min_periods=period).apply(_sinema, raw=True)
    result.name = "sinema"
    return result


# ---------------------------------------------------------------------------
# TRIMA — Triangular Moving Average
# ---------------------------------------------------------------------------


def trima(data: pd.Series, period: int = 20) -> pd.Series:
    """Triangular Moving Average (TRIMA).

    Equivalent to a double SMA: ``SMA(SMA(data, ceil((period+1)/2)),
    floor((period+1)/2))``. This produces a smoother curve than a single
    SMA by effectively giving the most weight to the center of the window.

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 20
        Overall window length.

    Returns
    -------
    pd.Series
        TRIMA values.

    Example
    -------
    >>> result = trima(close, period=20)
    """
    _validate_series(data)
    _validate_period(period)

    # Determine the two sub-periods
    half1 = int(np.ceil((period + 1) / 2))
    half2 = int(np.floor((period + 1) / 2))

    sma1 = data.rolling(window=half1, min_periods=half1).mean()
    result = sma1.rolling(window=half2, min_periods=half2).mean()
    result.name = "trima"
    return result


# ---------------------------------------------------------------------------
# JMA — Jurik Moving Average (approximation)
# ---------------------------------------------------------------------------


def jma(
    data: pd.Series,
    period: int = 7,
    phase: float = 50.0,
    power: int = 2,
) -> pd.Series:
    """Jurik Moving Average approximation (JMA).

    An adaptive moving average that attempts to minimize lag and overshoot.
    This is an approximation of the proprietary Jurik algorithm using an
    adaptive EMA with phase and power controls.

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 7
        Smoothing period.
    phase : float, default 50.0
        Phase parameter in the range [-100, 100]. Controls the tradeoff
        between lag and overshoot. 0 is balanced, positive reduces lag.
    power : int, default 2
        Power parameter controlling the smoothing curve shape.

    Returns
    -------
    pd.Series
        JMA values.

    Example
    -------
    >>> result = jma(close, period=7, phase=50, power=2)
    """
    _validate_series(data)
    _validate_period(period)

    # Compute beta from period
    beta = 0.45 * (period - 1) / (0.45 * (period - 1) + 2.0)

    # Compute phase ratio
    if phase < -100:
        phase_ratio = 0.5
    elif phase > 100:
        phase_ratio = 2.5
    else:
        phase_ratio = phase / 100.0 + 1.5

    alpha = beta**power

    values = data.values.astype(float)
    n = len(values)
    result = np.full(n, np.nan)

    if n == 0:
        return pd.Series(result, index=data.index, name="jma")

    # Find first valid index
    first_valid = 0
    while first_valid < n and np.isnan(values[first_valid]):
        first_valid += 1

    if first_valid >= n:
        return pd.Series(result, index=data.index, name="jma")

    # Initialize
    e0 = values[first_valid]
    e1 = 0.0
    e2 = 0.0
    jma_val = values[first_valid]
    result[first_valid] = jma_val

    for i in range(first_valid + 1, n):
        if np.isnan(values[i]):
            result[i] = np.nan
            continue

        e0 = (1.0 - alpha) * values[i] + alpha * e0
        e1 = (values[i] - e0) * (1.0 - beta) + beta * e1
        e2 = (e0 + phase_ratio * e1 - jma_val) * (1.0 - alpha) ** 2 + (alpha**2) * e2
        jma_val = jma_val + e2
        result[i] = jma_val

    return pd.Series(result, index=data.index, name="jma")


# ---------------------------------------------------------------------------
# Gaussian Filter
# ---------------------------------------------------------------------------


def gaussian_filter(data: pd.Series, period: int = 14, poles: int = 2) -> pd.Series:
    """Gaussian-weighted rolling filter.

    Applies a discrete Gaussian kernel over the rolling window. The
    standard deviation of the kernel is set to ``period / 4`` so that
    the window captures approximately two standard deviations.

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 14
        Window length.
    poles : int, default 2
        Number of standard deviations captured within the window. Used
        to set ``sigma = period / (2 * poles)``.

    Returns
    -------
    pd.Series
        Gaussian-filtered values.

    Example
    -------
    >>> result = gaussian_filter(close, period=14)
    """
    _validate_series(data)
    _validate_period(period)

    sigma = period / (2.0 * poles)
    center = (period - 1) / 2.0
    weights = np.array(
        [np.exp(-0.5 * ((i - center) / sigma) ** 2) for i in range(period)]
    )
    weights = weights / weights.sum()

    def _gauss(window: np.ndarray) -> float:
        return np.dot(window, weights)

    result = data.rolling(window=period, min_periods=period).apply(_gauss, raw=True)
    result.name = "gaussian_filter"
    return result


# ---------------------------------------------------------------------------
# Butterworth Filter
# ---------------------------------------------------------------------------


def butterworth_filter(data: pd.Series, period: int = 14) -> pd.Series:
    """2nd-order Butterworth low-pass filter (IIR).

    This implements the classic Ehlers two-pole Butterworth filter, which
    provides smooth output with minimal lag relative to its degree of
    smoothing.

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 14
        Cut-off period in bars.

    Returns
    -------
    pd.Series
        Butterworth-filtered values.

    Example
    -------
    >>> result = butterworth_filter(close, period=14)
    """
    _validate_series(data)
    _validate_period(period)

    # Butterworth coefficients (2-pole)
    a = np.exp(-np.sqrt(2.0) * np.pi / period)
    b = 2.0 * a * np.cos(np.sqrt(2.0) * np.pi / period)
    c2 = b
    c3 = -(a * a)
    c1 = 1.0 - c2 - c3

    values = data.values.astype(float)
    n = len(values)
    result = np.full(n, np.nan)

    # Seed first two valid values
    first_valid = 0
    while first_valid < n and np.isnan(values[first_valid]):
        first_valid += 1

    if first_valid >= n:
        return pd.Series(result, index=data.index, name="butterworth")

    result[first_valid] = values[first_valid]
    if first_valid + 1 < n:
        result[first_valid + 1] = values[first_valid + 1]

    for i in range(first_valid + 2, n):
        if np.isnan(values[i]):
            result[i] = np.nan
            continue
        result[i] = c1 * values[i] + c2 * result[i - 1] + c3 * result[i - 2]

    return pd.Series(result, index=data.index, name="butterworth")


# ---------------------------------------------------------------------------
# Super Smoother
# ---------------------------------------------------------------------------


def supersmoother(data: pd.Series, period: int = 14) -> pd.Series:
    """Ehlers Super Smoother (2-pole Butterworth variant).

    A modified Butterworth filter by John Ehlers that removes aliasing
    noise while retaining a smooth, low-lag response.

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 14
        Cut-off period in bars.

    Returns
    -------
    pd.Series
        Super-smoothed values.

    Example
    -------
    >>> result = supersmoother(close, period=14)
    """
    _validate_series(data)
    _validate_period(period)

    # Ehlers super smoother coefficients
    f = 1.414 * np.pi / period
    a1 = np.exp(-f)
    b1 = 2.0 * a1 * np.cos(f)
    c3 = -(a1 * a1)
    c2 = b1
    c1 = 1.0 - c2 - c3

    values = data.values.astype(float)
    n = len(values)
    result = np.full(n, np.nan)

    first_valid = 0
    while first_valid < n and np.isnan(values[first_valid]):
        first_valid += 1

    if first_valid >= n:
        return pd.Series(result, index=data.index, name="supersmoother")

    result[first_valid] = values[first_valid]
    if first_valid + 1 < n:
        result[first_valid + 1] = values[first_valid + 1]

    for i in range(first_valid + 2, n):
        if np.isnan(values[i]):
            result[i] = np.nan
            continue
        result[i] = c1 * values[i] + c2 * result[i - 1] + c3 * result[i - 2]

    return pd.Series(result, index=data.index, name="supersmoother")


# ---------------------------------------------------------------------------
# Hann Window Moving Average
# ---------------------------------------------------------------------------


def hann_window_ma(data: pd.Series, period: int = 14) -> pd.Series:
    """Hann (raised cosine) windowed moving average.

    Each element in the window is weighted by the Hann function:
    ``w(i) = 0.5 * (1 - cos(2 * pi * i / (N - 1)))``

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 14
        Window length.

    Returns
    -------
    pd.Series
        Hann-windowed moving average values.

    Example
    -------
    >>> result = hann_window_ma(close, period=14)
    """
    _validate_series(data)
    _validate_period(period)

    if period == 1:
        result = data.copy()
        result.name = "hann_ma"
        return result

    weights = np.array(
        [0.5 * (1.0 - np.cos(2.0 * np.pi * i / (period - 1))) for i in range(period)]
    )
    weights = weights / weights.sum()

    def _hann(window: np.ndarray) -> float:
        return np.dot(window, weights)

    result = data.rolling(window=period, min_periods=period).apply(_hann, raw=True)
    result.name = "hann_ma"
    return result


# ---------------------------------------------------------------------------
# Hamming Window Moving Average
# ---------------------------------------------------------------------------


def hamming_window_ma(data: pd.Series, period: int = 14) -> pd.Series:
    """Hamming windowed moving average.

    Each element in the window is weighted by the Hamming function:
    ``w(i) = 0.54 - 0.46 * cos(2 * pi * i / (N - 1))``

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 14
        Window length.

    Returns
    -------
    pd.Series
        Hamming-windowed moving average values.

    Example
    -------
    >>> result = hamming_window_ma(close, period=14)
    """
    _validate_series(data)
    _validate_period(period)

    if period == 1:
        result = data.copy()
        result.name = "hamming_ma"
        return result

    weights = np.array(
        [
            0.54 - 0.46 * np.cos(2.0 * np.pi * i / (period - 1))
            for i in range(period)
        ]
    )
    weights = weights / weights.sum()

    def _hamming(window: np.ndarray) -> float:
        return np.dot(window, weights)

    result = data.rolling(window=period, min_periods=period).apply(_hamming, raw=True)
    result.name = "hamming_ma"
    return result


# ---------------------------------------------------------------------------
# Kaufman Efficiency Ratio
# ---------------------------------------------------------------------------


def kaufman_efficiency_ratio(data: pd.Series, period: int = 10) -> pd.Series:
    """Kaufman Efficiency Ratio (ER).

    Measures the efficiency of price movement as the ratio of directional
    change to total path length. This is the core component of the Kaufman
    Adaptive Moving Average (KAMA).

    ``ER = |close - close[period]| / sum(|close - close[1]|, period)``

    Values near 1.0 indicate strong trending; values near 0.0 indicate
    choppy / mean-reverting markets.

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 10
        Look-back period.

    Returns
    -------
    pd.Series
        Efficiency ratio values in [0, 1].

    Example
    -------
    >>> result = kaufman_efficiency_ratio(close, period=10)
    """
    _validate_series(data)
    _validate_period(period)

    direction = (data - data.shift(period)).abs()
    volatility = data.diff().abs().rolling(window=period, min_periods=period).sum()

    result = direction / volatility.replace(0.0, np.nan)
    result.name = "efficiency_ratio"
    return result
