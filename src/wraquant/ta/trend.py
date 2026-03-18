"""Trend indicators.

This module provides indicators that identify and measure the direction
and strength of price trends. All functions accept ``pd.Series`` inputs
and return ``pd.Series`` (or ``dict[str, pd.Series]`` for multi-output
indicators).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "adx",
    "aroon",
    "psar",
    "vortex",
    "trix",
    "linear_regression",
    "linear_regression_slope",
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


def _ema(data: pd.Series, period: int) -> pd.Series:
    """Internal EMA helper."""
    return data.ewm(span=period, adjust=False, min_periods=period).mean()


def _wilder_smooth(data: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing (equivalent to ``ewm(alpha=1/period)``)."""
    return data.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# ADX (Average Directional Index)
# ---------------------------------------------------------------------------


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> dict[str, pd.Series]:
    """Average Directional Index (ADX) with +DI and -DI.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    period : int, default 14
        Smoothing period.

    Returns
    -------
    dict[str, pd.Series]
        ``adx``, ``plus_di``, ``minus_di`` — all in [0, 100].
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")
    _validate_period(period)

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )

    # Wilder smoothing
    smoothed_tr = _wilder_smooth(tr, period)
    smoothed_plus_dm = _wilder_smooth(plus_dm, period)
    smoothed_minus_dm = _wilder_smooth(minus_dm, period)

    plus_di = 100.0 * smoothed_plus_dm / smoothed_tr
    minus_di = 100.0 * smoothed_minus_dm / smoothed_tr

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_val = _wilder_smooth(dx, period)

    return {
        "adx": adx_val.rename("adx"),
        "plus_di": plus_di.rename("plus_di"),
        "minus_di": minus_di.rename("minus_di"),
    }


# ---------------------------------------------------------------------------
# Aroon
# ---------------------------------------------------------------------------


def aroon(
    high: pd.Series,
    low: pd.Series,
    period: int = 25,
) -> dict[str, pd.Series]:
    """Aroon Indicator.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    period : int, default 25
        Look-back period.

    Returns
    -------
    dict[str, pd.Series]
        ``aroon_up``, ``aroon_down``, ``oscillator`` — up/down in [0, 100],
        oscillator in [-100, 100].
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_period(period)

    def _bars_since_high(window: np.ndarray) -> float:
        return float(period - (period - np.argmax(window)))

    def _bars_since_low(window: np.ndarray) -> float:
        return float(period - (period - np.argmin(window)))

    rolling_high_idx = high.rolling(window=period + 1, min_periods=period + 1).apply(
        _bars_since_high, raw=True
    )
    rolling_low_idx = low.rolling(window=period + 1, min_periods=period + 1).apply(
        _bars_since_low, raw=True
    )

    aroon_up = (rolling_high_idx / period) * 100.0
    aroon_down = (rolling_low_idx / period) * 100.0
    oscillator = aroon_up - aroon_down

    return {
        "aroon_up": aroon_up.rename("aroon_up"),
        "aroon_down": aroon_down.rename("aroon_down"),
        "oscillator": oscillator.rename("aroon_oscillator"),
    }


# ---------------------------------------------------------------------------
# Parabolic SAR
# ---------------------------------------------------------------------------


def psar(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.2,
) -> pd.Series:
    """Parabolic SAR (Stop and Reverse).

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    af_start : float, default 0.02
        Initial acceleration factor.
    af_step : float, default 0.02
        Acceleration factor increment.
    af_max : float, default 0.2
        Maximum acceleration factor.

    Returns
    -------
    pd.Series
        Parabolic SAR values. Values above price indicate downtrend;
        values below price indicate uptrend.
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    h = high.values.astype(float)
    l = low.values.astype(float)  # noqa: E741
    n = len(close)

    sar = np.full(n, np.nan)
    if n < 2:
        return pd.Series(sar, index=close.index, name="psar")

    # Initialize
    is_uptrend = True
    af = af_start
    ep = h[0]
    sar[0] = l[0]

    for i in range(1, n):
        prev_sar = sar[i - 1]

        if is_uptrend:
            sar[i] = prev_sar + af * (ep - prev_sar)
            # SAR must not be above the prior two lows
            sar[i] = min(sar[i], l[i - 1])
            if i >= 2:
                sar[i] = min(sar[i], l[i - 2])

            if l[i] < sar[i]:
                # Switch to downtrend
                is_uptrend = False
                sar[i] = ep
                ep = l[i]
                af = af_start
            else:
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + af_step, af_max)
        else:
            sar[i] = prev_sar + af * (ep - prev_sar)
            # SAR must not be below the prior two highs
            sar[i] = max(sar[i], h[i - 1])
            if i >= 2:
                sar[i] = max(sar[i], h[i - 2])

            if h[i] > sar[i]:
                # Switch to uptrend
                is_uptrend = True
                sar[i] = ep
                ep = h[i]
                af = af_start
            else:
                if l[i] < ep:
                    ep = l[i]
                    af = min(af + af_step, af_max)

    return pd.Series(sar, index=close.index, name="psar")


# ---------------------------------------------------------------------------
# Vortex Indicator
# ---------------------------------------------------------------------------


def vortex(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> dict[str, pd.Series]:
    """Vortex Indicator.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    period : int, default 14
        Look-back period.

    Returns
    -------
    dict[str, pd.Series]
        ``plus_vi`` and ``minus_vi``.
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")
    _validate_period(period)

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    plus_vm = (high - low.shift(1)).abs()
    minus_vm = (low - high.shift(1)).abs()

    sum_tr = tr.rolling(window=period, min_periods=period).sum()
    plus_vi_val = plus_vm.rolling(window=period, min_periods=period).sum() / sum_tr
    minus_vi_val = minus_vm.rolling(window=period, min_periods=period).sum() / sum_tr

    return {
        "plus_vi": plus_vi_val.rename("plus_vi"),
        "minus_vi": minus_vi_val.rename("minus_vi"),
    }


# ---------------------------------------------------------------------------
# TRIX
# ---------------------------------------------------------------------------


def trix(data: pd.Series, period: int = 15) -> pd.Series:
    """TRIX — Triple-smoothed EMA rate of change.

    ``TRIX = 100 * ROC(EMA(EMA(EMA(data))))``

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 15
        EMA period for each smoothing pass.

    Returns
    -------
    pd.Series
        TRIX values.
    """
    _validate_series(data)
    _validate_period(period)

    ema1 = _ema(data, period)
    ema2 = _ema(ema1, period)
    ema3 = _ema(ema2, period)
    result = ema3.pct_change() * 100.0
    result.name = "trix"
    return result


# ---------------------------------------------------------------------------
# Linear Regression
# ---------------------------------------------------------------------------


def linear_regression(
    data: pd.Series,
    period: int = 14,
) -> dict[str, pd.Series]:
    """Rolling Linear Regression.

    Fits an OLS regression line to the last *period* values at each bar.

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 14
        Look-back window.

    Returns
    -------
    dict[str, pd.Series]
        ``value`` (end-of-line prediction), ``slope``, ``intercept``,
        ``r_squared``.
    """
    _validate_series(data)
    _validate_period(period)

    n = len(data)
    values = data.values.astype(float)
    slope_arr = np.full(n, np.nan)
    intercept_arr = np.full(n, np.nan)
    r_squared_arr = np.full(n, np.nan)
    value_arr = np.full(n, np.nan)

    x = np.arange(period, dtype=float)
    x_mean = x.mean()
    ss_x = np.sum((x - x_mean) ** 2)

    for i in range(period - 1, n):
        y = values[i - period + 1 : i + 1]
        if np.any(np.isnan(y)):
            continue
        y_mean = y.mean()
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        ss_y = np.sum((y - y_mean) ** 2)

        slope_arr[i] = ss_xy / ss_x
        intercept_arr[i] = y_mean - slope_arr[i] * x_mean
        value_arr[i] = intercept_arr[i] + slope_arr[i] * (period - 1)

        if ss_y == 0:
            r_squared_arr[i] = 1.0
        else:
            r_squared_arr[i] = (ss_xy**2) / (ss_x * ss_y)

    idx = data.index
    return {
        "value": pd.Series(value_arr, index=idx, name="linreg_value"),
        "slope": pd.Series(slope_arr, index=idx, name="linreg_slope"),
        "intercept": pd.Series(intercept_arr, index=idx, name="linreg_intercept"),
        "r_squared": pd.Series(r_squared_arr, index=idx, name="linreg_r_squared"),
    }


def linear_regression_slope(data: pd.Series, period: int = 14) -> pd.Series:
    """Rolling Linear Regression Slope.

    A convenience wrapper around :func:`linear_regression` that returns
    only the slope component.

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 14
        Look-back window.

    Returns
    -------
    pd.Series
        Slope values.
    """
    return linear_regression(data, period)["slope"]
