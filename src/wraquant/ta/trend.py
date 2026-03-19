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
    "zigzag",
    "heikin_ashi",
    "mcginley_dynamic",
    "schaff_trend_cycle",
    "guppy_mma",
    "rainbow_ma",
    "hull_ma",
    "zero_lag_ema",
    "vidya",
    "tilson_t3",
    "fractal_adaptive_ma",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


from wraquant.ta._validators import validate_period as _validate_period
from wraquant.ta._validators import validate_series as _validate_series


def _ema(data: pd.Series, period: int) -> pd.Series:
    """Internal EMA helper."""
    return data.ewm(span=period, adjust=False, min_periods=period).mean()


def _wma(data: pd.Series, period: int) -> pd.Series:
    """Internal Weighted Moving Average helper."""
    weights = np.arange(1, period + 1, dtype=float)

    def _apply_wma(window: np.ndarray) -> float:
        return np.dot(window, weights) / weights.sum()

    return data.rolling(window=period, min_periods=period).apply(_apply_wma, raw=True)


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
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")
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
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
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
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")

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
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")
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
    data = _validate_series(data)
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
    data = _validate_series(data)
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


# ---------------------------------------------------------------------------
# ZigZag
# ---------------------------------------------------------------------------


def zigzag(
    close: pd.Series,
    pct_change: float = 5.0,
) -> pd.Series:
    """ZigZag indicator — connects swing highs and lows.

    Identifies pivots where price reverses by at least *pct_change* percent,
    then linearly interpolates between them.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    pct_change : float, default 5.0
        Minimum percentage change to register a new pivot.

    Returns
    -------
    pd.Series
        ZigZag line (interpolated between pivots). Non-pivot bars are
        filled via linear interpolation; leading/trailing NaNs remain where
        no pivot has been established.

    Example
    -------
    >>> import pandas as pd
    >>> zz = zigzag(pd.Series([100, 110, 105, 95, 100, 90]), pct_change=5.0)
    """
    close = _validate_series(close, "close")
    if pct_change <= 0:
        raise ValueError(f"pct_change must be > 0, got {pct_change}")

    values = close.values.astype(float)
    n = len(values)
    result = np.full(n, np.nan)

    if n < 2:
        return pd.Series(result, index=close.index, name="zigzag")

    threshold = pct_change / 100.0

    # Find first valid (non-NaN) index
    start = 0
    while start < n and np.isnan(values[start]):
        start += 1
    if start >= n - 1:
        return pd.Series(result, index=close.index, name="zigzag")

    pivots: list[tuple[int, float]] = [(start, values[start])]
    last_pivot_val = values[start]
    direction = 0  # 0 = unknown, 1 = up, -1 = down

    for i in range(start + 1, n):
        if np.isnan(values[i]):
            continue
        change = (values[i] - last_pivot_val) / abs(last_pivot_val)

        if direction == 0:
            if change >= threshold:
                direction = 1
                pivots.append((i, values[i]))
                last_pivot_val = values[i]
            elif change <= -threshold:
                direction = -1
                pivots.append((i, values[i]))
                last_pivot_val = values[i]
        elif direction == 1:
            if values[i] > last_pivot_val:
                # Extend the up-move — update the last pivot
                pivots[-1] = (i, values[i])
                last_pivot_val = values[i]
            elif (values[i] - last_pivot_val) / abs(last_pivot_val) <= -threshold:
                direction = -1
                pivots.append((i, values[i]))
                last_pivot_val = values[i]
        else:  # direction == -1
            if values[i] < last_pivot_val:
                pivots[-1] = (i, values[i])
                last_pivot_val = values[i]
            elif (values[i] - last_pivot_val) / abs(last_pivot_val) >= threshold:
                direction = 1
                pivots.append((i, values[i]))
                last_pivot_val = values[i]

    # Place pivot values
    for idx, val in pivots:
        result[idx] = val

    # Interpolate between pivots
    if len(pivots) >= 2:
        pivot_indices = [p[0] for p in pivots]
        pivot_values = [p[1] for p in pivots]
        for k in range(len(pivot_indices) - 1):
            i0, i1 = pivot_indices[k], pivot_indices[k + 1]
            v0, v1 = pivot_values[k], pivot_values[k + 1]
            for j in range(i0, i1 + 1):
                result[j] = v0 + (v1 - v0) * (j - i0) / (i1 - i0)

    return pd.Series(result, index=close.index, name="zigzag")


# ---------------------------------------------------------------------------
# Heikin-Ashi
# ---------------------------------------------------------------------------


def heikin_ashi(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> dict[str, pd.Series]:
    """Heikin-Ashi modified OHLC candles.

    Parameters
    ----------
    open_ : pd.Series
        Open prices.
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.

    Returns
    -------
    dict[str, pd.Series]
        ``ha_open``, ``ha_high``, ``ha_low``, ``ha_close``.

    Example
    -------
    >>> import pandas as pd
    >>> ha = heikin_ashi(
    ...     pd.Series([100, 102]),
    ...     pd.Series([105, 106]),
    ...     pd.Series([99, 101]),
    ...     pd.Series([104, 103]),
    ... )
    """
    open_ = _validate_series(open_, "open_")
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")

    o = open_.values.astype(float)
    h = high.values.astype(float)
    l = low.values.astype(float)  # noqa: E741
    c = close.values.astype(float)
    n = len(close)

    ha_close_arr = (o + h + l + c) / 4.0
    ha_open_arr = np.empty(n)
    ha_open_arr[0] = (o[0] + c[0]) / 2.0
    for i in range(1, n):
        ha_open_arr[i] = (ha_open_arr[i - 1] + ha_close_arr[i - 1]) / 2.0

    ha_high_arr = np.maximum(h, np.maximum(ha_open_arr, ha_close_arr))
    ha_low_arr = np.minimum(l, np.minimum(ha_open_arr, ha_close_arr))

    idx = close.index
    return {
        "ha_open": pd.Series(ha_open_arr, index=idx, name="ha_open"),
        "ha_high": pd.Series(ha_high_arr, index=idx, name="ha_high"),
        "ha_low": pd.Series(ha_low_arr, index=idx, name="ha_low"),
        "ha_close": pd.Series(ha_close_arr, index=idx, name="ha_close"),
    }


# ---------------------------------------------------------------------------
# McGinley Dynamic
# ---------------------------------------------------------------------------


def mcginley_dynamic(
    data: pd.Series,
    period: int = 14,
) -> pd.Series:
    """McGinley Dynamic — adaptive moving average.

    Adjusts its speed based on market velocity, reducing whipsaws compared
    to a standard EMA.

    ``MD_t = MD_{t-1} + (price - MD_{t-1}) / (N * (price / MD_{t-1})^4)``

    Parameters
    ----------
    data : pd.Series
        Price series (typically close).
    period : int, default 14
        Smoothing period.

    Returns
    -------
    pd.Series
        McGinley Dynamic values.

    Example
    -------
    >>> import pandas as pd
    >>> md = mcginley_dynamic(pd.Series([100, 102, 101, 103, 105]))
    """
    data = _validate_series(data)
    _validate_period(period)

    values = data.values.astype(float)
    n = len(values)
    result = np.full(n, np.nan)

    # Seed with first non-NaN value
    start = 0
    while start < n and np.isnan(values[start]):
        start += 1
    if start >= n:
        return pd.Series(result, index=data.index, name="mcginley_dynamic")

    result[start] = values[start]
    for i in range(start + 1, n):
        if np.isnan(values[i]):
            result[i] = result[i - 1]
            continue
        prev = result[i - 1]
        if prev == 0:
            result[i] = values[i]
            continue
        ratio = values[i] / prev
        denom = period * (ratio**4)
        result[i] = prev + (values[i] - prev) / denom

    return pd.Series(result, index=data.index, name="mcginley_dynamic")


# ---------------------------------------------------------------------------
# Schaff Trend Cycle
# ---------------------------------------------------------------------------


def schaff_trend_cycle(
    close: pd.Series,
    period: int = 10,
    fast: int = 23,
    slow: int = 50,
) -> pd.Series:
    """Schaff Trend Cycle — MACD passed through a double stochastic.

    Combines the MACD histogram with stochastic smoothing for a faster,
    smoother oscillator bounded between 0 and 100.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    period : int, default 10
        Stochastic look-back period.
    fast : int, default 23
        Fast EMA period for MACD.
    slow : int, default 50
        Slow EMA period for MACD.

    Returns
    -------
    pd.Series
        STC values in [0, 100].

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> stc = schaff_trend_cycle(pd.Series(np.random.randn(100).cumsum() + 100))
    """
    close = _validate_series(close, "close")
    _validate_period(period)
    _validate_period(fast, "fast")
    _validate_period(slow, "slow")

    macd_line = _ema(close, fast) - _ema(close, slow)

    # First stochastic of MACD
    lowest_macd = macd_line.rolling(window=period, min_periods=1).min()
    highest_macd = macd_line.rolling(window=period, min_periods=1).max()
    denom1 = highest_macd - lowest_macd
    stoch1 = pd.Series(
        np.where(denom1 != 0, (macd_line - lowest_macd) / denom1 * 100.0, 50.0),
        index=close.index,
    )
    # Smooth with EMA
    pf = _ema(stoch1, period)

    # Second stochastic
    lowest_pf = pf.rolling(window=period, min_periods=1).min()
    highest_pf = pf.rolling(window=period, min_periods=1).max()
    denom2 = highest_pf - lowest_pf
    stoch2 = pd.Series(
        np.where(denom2 != 0, (pf - lowest_pf) / denom2 * 100.0, 50.0),
        index=close.index,
    )
    result = _ema(stoch2, period)
    result.name = "schaff_trend_cycle"
    return result


# ---------------------------------------------------------------------------
# Guppy Multiple Moving Average
# ---------------------------------------------------------------------------


def guppy_mma(
    data: pd.Series,
) -> dict[str, pd.Series]:
    """Guppy Multiple Moving Average (GMMA).

    Returns two groups of EMAs:

    - **Short-term group**: 3, 5, 8, 10, 12, 15
    - **Long-term group**: 30, 35, 40, 45, 50, 60

    Parameters
    ----------
    data : pd.Series
        Price series (typically close).

    Returns
    -------
    dict[str, pd.Series]
        Keys ``short_3``, ``short_5``, ..., ``short_15``,
        ``long_30``, ``long_35``, ..., ``long_60``.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> g = guppy_mma(pd.Series(np.random.randn(100).cumsum() + 100))
    """
    data = _validate_series(data)

    short_periods = [3, 5, 8, 10, 12, 15]
    long_periods = [30, 35, 40, 45, 50, 60]

    result: dict[str, pd.Series] = {}
    for p in short_periods:
        key = f"short_{p}"
        result[key] = _ema(data, p).rename(key)
    for p in long_periods:
        key = f"long_{p}"
        result[key] = _ema(data, p).rename(key)
    return result


# ---------------------------------------------------------------------------
# Rainbow Moving Average
# ---------------------------------------------------------------------------


def rainbow_ma(
    data: pd.Series,
    period: int = 10,
    levels: int = 10,
) -> dict[str, pd.Series]:
    """Rainbow Moving Average — recursive SMAs.

    Each level is an SMA of the previous level.  Level 1 is SMA of *data*,
    level 2 is SMA of level 1, and so on.

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 10
        SMA period applied at each level.
    levels : int, default 10
        Number of SMA recursions (typically 10).

    Returns
    -------
    dict[str, pd.Series]
        Keys ``sma_1`` through ``sma_{levels}``.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> rb = rainbow_ma(pd.Series(np.random.randn(100).cumsum() + 100))
    """
    data = _validate_series(data)
    _validate_period(period)
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")

    result: dict[str, pd.Series] = {}
    current = data
    for i in range(1, levels + 1):
        current = current.rolling(window=period, min_periods=period).mean()
        key = f"sma_{i}"
        result[key] = current.rename(key)
    return result


# ---------------------------------------------------------------------------
# Hull Moving Average
# ---------------------------------------------------------------------------


def hull_ma(data: pd.Series, period: int = 16) -> pd.Series:
    """Hull Moving Average (HMA).

    ``HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))``

    Provides a fast, smooth moving average with reduced lag.

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 16
        HMA period.

    Returns
    -------
    pd.Series
        Hull Moving Average values.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> hma = hull_ma(pd.Series(np.random.randn(100).cumsum() + 100), period=16)
    """
    data = _validate_series(data)
    _validate_period(period)

    half_period = max(int(period / 2), 1)
    sqrt_period = max(int(np.sqrt(period)), 1)

    wma_half = _wma(data, half_period)
    wma_full = _wma(data, period)
    diff = 2.0 * wma_half - wma_full
    result = _wma(diff, sqrt_period)
    result.name = "hull_ma"
    return result


# ---------------------------------------------------------------------------
# Zero-Lag EMA
# ---------------------------------------------------------------------------


def zero_lag_ema(data: pd.Series, period: int = 21) -> pd.Series:
    """Zero-Lag Exponential Moving Average (ZLEMA).

    Compensates for inherent EMA lag by applying the EMA to a
    de-lagged series: ``zlema_input = data + (data - data.shift(lag))``
    where ``lag = (period - 1) / 2``.

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 21
        EMA period.

    Returns
    -------
    pd.Series
        ZLEMA values.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> zl = zero_lag_ema(pd.Series(np.random.randn(100).cumsum() + 100))
    """
    data = _validate_series(data)
    _validate_period(period)

    lag = int((period - 1) / 2)
    adjusted = data + (data - data.shift(lag))
    result = _ema(adjusted, period)
    result.name = "zero_lag_ema"
    return result


# ---------------------------------------------------------------------------
# VIDYA (Variable Index Dynamic Average)
# ---------------------------------------------------------------------------


def vidya(
    data: pd.Series,
    period: int = 14,
    smooth: int = 5,
) -> pd.Series:
    """Variable Index Dynamic Average (VIDYA).

    Uses the Chande Momentum Oscillator (CMO) as a volatility index to
    dynamically adjust the smoothing constant of an EMA.

    ``VIDYA_t = alpha * |CMO_t| * price_t + (1 - alpha * |CMO_t|) * VIDYA_{t-1}``

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 14
        CMO look-back period.
    smooth : int, default 5
        Smoothing period to derive the base alpha (``2 / (smooth + 1)``).

    Returns
    -------
    pd.Series
        VIDYA values.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> v = vidya(pd.Series(np.random.randn(100).cumsum() + 100))
    """
    data = _validate_series(data)
    _validate_period(period)
    _validate_period(smooth, "smooth")

    values = data.values.astype(float)
    n = len(values)
    result = np.full(n, np.nan)

    # Compute CMO
    diff = np.diff(values, prepend=np.nan)
    gains = np.where(diff > 0, diff, 0.0)
    losses = np.where(diff < 0, -diff, 0.0)

    gain_sum = pd.Series(gains).rolling(window=period, min_periods=period).sum().values
    loss_sum = pd.Series(losses).rolling(window=period, min_periods=period).sum().values

    total = gain_sum + loss_sum
    cmo = np.where(total != 0, (gain_sum - loss_sum) / total, 0.0)
    abs_cmo = np.abs(cmo)

    alpha = 2.0 / (smooth + 1)

    # Seed: first bar where CMO is valid
    start = period  # period bars needed for CMO
    if start >= n:
        return pd.Series(result, index=data.index, name="vidya")

    result[start] = values[start]
    for i in range(start + 1, n):
        if np.isnan(values[i]):
            result[i] = result[i - 1]
            continue
        sc = alpha * abs_cmo[i]
        result[i] = sc * values[i] + (1.0 - sc) * result[i - 1]

    return pd.Series(result, index=data.index, name="vidya")


# ---------------------------------------------------------------------------
# Tilson T3
# ---------------------------------------------------------------------------


def tilson_t3(
    data: pd.Series,
    period: int = 5,
    volume_factor: float = 0.7,
) -> pd.Series:
    """Tilson T3 — triple-smoothed exponential moving average.

    Applies six sequential EMAs with Tilson coefficients derived from the
    *volume_factor* to produce an ultra-smooth, low-lag average.

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 5
        EMA period for each pass.
    volume_factor : float, default 0.7
        Volume factor (commonly 0.7). Controls the overshoot reduction.

    Returns
    -------
    pd.Series
        Tilson T3 values.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> t3 = tilson_t3(pd.Series(np.random.randn(100).cumsum() + 100))
    """
    data = _validate_series(data)
    _validate_period(period)

    vf = volume_factor
    c1 = -(vf**3)
    c2 = 3 * vf**2 + 3 * vf**3
    c3 = -6 * vf**2 - 3 * vf - 3 * vf**3
    c4 = 1 + 3 * vf + vf**3 + 3 * vf**2

    e1 = _ema(data, period)
    e2 = _ema(e1, period)
    e3 = _ema(e2, period)
    e4 = _ema(e3, period)
    e5 = _ema(e4, period)
    e6 = _ema(e5, period)

    result = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
    result.name = "tilson_t3"
    return result


# ---------------------------------------------------------------------------
# Fractal Adaptive Moving Average (FRAMA)
# ---------------------------------------------------------------------------


def fractal_adaptive_ma(
    data: pd.Series,
    period: int = 16,
) -> pd.Series:
    """Fractal Adaptive Moving Average (FRAMA).

    Uses the fractal dimension of the price series to dynamically adjust
    the EMA smoothing factor.  More responsive in trending markets and
    slower during consolidation.

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 16
        Look-back period (should be even; if odd, it is rounded up).

    Returns
    -------
    pd.Series
        FRAMA values.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> f = fractal_adaptive_ma(pd.Series(np.random.randn(200).cumsum() + 100))
    """
    data = _validate_series(data)
    _validate_period(period)

    # Ensure period is even
    if period % 2 != 0:
        period += 1

    half = period // 2
    values = data.values.astype(float)
    n = len(values)
    result = np.full(n, np.nan)

    if n < period:
        return pd.Series(result, index=data.index, name="frama")

    # Seed FRAMA with the value at the start of the window
    result[period - 1] = values[period - 1]

    for i in range(period, n):
        # High-low range of first half, second half, and full window
        window = values[i - period + 1 : i + 1]
        first_half = window[:half]
        second_half = window[half:]

        n1 = (np.max(first_half) - np.min(first_half)) / half
        n2 = (np.max(second_half) - np.min(second_half)) / half
        n3 = (np.max(window) - np.min(window)) / period

        if n1 + n2 > 0 and n3 > 0:
            d = (np.log(n1 + n2) - np.log(n3)) / np.log(2)
        else:
            d = 1.0

        alpha = np.exp(-4.6 * (d - 1.0))
        alpha = max(0.01, min(alpha, 1.0))

        result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1]

    return pd.Series(result, index=data.index, name="frama")
