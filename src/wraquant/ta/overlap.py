"""Overlap / moving average technical analysis studies.

This module provides a comprehensive set of moving average and overlay
indicators used in technical analysis. All functions accept ``pd.Series``
inputs and return ``pd.Series`` (or ``dict[str, pd.Series]`` for
multi-output indicators).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "sma",
    "ema",
    "wma",
    "dema",
    "tema",
    "kama",
    "vwap",
    "supertrend",
    "ichimoku",
    "bollinger_bands",
    "keltner_channel",
    "donchian_channel",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


from wraquant.ta._validators import validate_period as _validate_period
from wraquant.ta._validators import validate_series as _validate_series

# ---------------------------------------------------------------------------
# Moving Averages
# ---------------------------------------------------------------------------


def sma(data: pd.Series, period: int = 20) -> pd.Series:
    """Simple Moving Average.

    Parameters
    ----------
    data : pd.Series
        Price series (typically close).
    period : int, default 20
        Window length.

    Returns
    -------
    pd.Series
        Simple moving average values. The first ``period - 1`` entries are
        ``NaN``.
    """
    data = _validate_series(data)
    _validate_period(period)
    return data.rolling(window=period, min_periods=period).mean()


def ema(data: pd.Series, period: int = 20) -> pd.Series:
    """Exponential Moving Average.

    Uses the standard *span*-based smoothing factor ``2 / (period + 1)``.

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 20
        Span for the EMA.

    Returns
    -------
    pd.Series
        Exponential moving average values.
    """
    data = _validate_series(data)
    _validate_period(period)
    return data.ewm(span=period, adjust=False, min_periods=period).mean()


def wma(data: pd.Series, period: int = 20) -> pd.Series:
    """Weighted Moving Average.

    Weights increase linearly so that the most recent observation receives
    the highest weight.

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 20
        Window length.

    Returns
    -------
    pd.Series
        Weighted moving average values.
    """
    data = _validate_series(data)
    _validate_period(period)
    weights = np.arange(1, period + 1, dtype=float)

    def _wma(window: np.ndarray) -> float:
        return np.dot(window, weights) / weights.sum()

    return data.rolling(window=period, min_periods=period).apply(_wma, raw=True)


def dema(data: pd.Series, period: int = 20) -> pd.Series:
    """Double Exponential Moving Average.

    ``DEMA = 2 * EMA(data) - EMA(EMA(data))``

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 20
        Span for each EMA component.

    Returns
    -------
    pd.Series
        DEMA values.
    """
    data = _validate_series(data)
    _validate_period(period)
    ema1 = ema(data, period)
    ema2 = ema(ema1, period)
    return 2 * ema1 - ema2


def tema(data: pd.Series, period: int = 20) -> pd.Series:
    """Triple Exponential Moving Average.

    ``TEMA = 3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))``

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 20
        Span for each EMA component.

    Returns
    -------
    pd.Series
        TEMA values.
    """
    data = _validate_series(data)
    _validate_period(period)
    ema1 = ema(data, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    return 3 * ema1 - 3 * ema2 + ema3


def kama(
    data: pd.Series,
    period: int = 10,
    fast: int = 2,
    slow: int = 30,
) -> pd.Series:
    """Kaufman Adaptive Moving Average (KAMA).

    KAMA adapts its smoothing constant based on the efficiency ratio of the
    price movement.

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 10
        Efficiency ratio look-back period.
    fast : int, default 2
        Fast smoothing constant period.
    slow : int, default 30
        Slow smoothing constant period.

    Returns
    -------
    pd.Series
        KAMA values.
    """
    data = _validate_series(data)
    _validate_period(period)

    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)

    values = data.values.astype(float).copy()
    result = np.full_like(values, np.nan)

    # Seed with first available non-NaN after enough data
    start = period
    if start >= len(values):
        return pd.Series(result, index=data.index, name="kama")

    result[start] = values[start]

    for i in range(start + 1, len(values)):
        direction = abs(values[i] - values[i - period])
        volatility = np.nansum(np.abs(np.diff(values[i - period : i + 1])))
        if volatility == 0:
            er = 0.0
        else:
            er = direction / volatility
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        result[i] = result[i - 1] + sc * (values[i] - result[i - 1])

    return pd.Series(result, index=data.index, name="kama")


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Volume Weighted Average Price (VWAP).

    Computed as the cumulative sum of ``typical_price * volume`` divided by
    cumulative volume. This is the *intraday running* VWAP; for session-reset
    VWAP, pre-group your data by session.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume data.

    Returns
    -------
    pd.Series
        Cumulative VWAP values.
    """
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")
    volume = _validate_series(volume, "volume")

    typical_price = (high + low + close) / 3.0
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    result = cum_tp_vol / cum_vol
    result.name = "vwap"
    return result


# ---------------------------------------------------------------------------
# Supertrend
# ---------------------------------------------------------------------------


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> dict[str, pd.Series]:
    """Supertrend indicator.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    period : int, default 10
        ATR look-back period.
    multiplier : float, default 3.0
        ATR multiplier for bands.

    Returns
    -------
    dict[str, pd.Series]
        ``supertrend`` — the indicator line, and ``direction`` (1 for
        uptrend / bullish, -1 for downtrend / bearish).
    """
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")
    _validate_period(period)

    hl2 = (high + low) / 2.0

    # ATR calculation (inlined to avoid circular import)
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()

    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    n = len(close)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    st = np.full(n, np.nan)
    direction = np.full(n, np.nan)

    c = close.values.astype(float)
    ub = upper_basic.values.astype(float)
    lb = lower_basic.values.astype(float)

    # Seed
    first_valid = period - 1
    upper_band[first_valid] = ub[first_valid]
    lower_band[first_valid] = lb[first_valid]
    st[first_valid] = ub[first_valid]
    direction[first_valid] = 1.0

    for i in range(first_valid + 1, n):
        if np.isnan(ub[i]):
            continue

        # Final upper band
        if ub[i] < upper_band[i - 1] or c[i - 1] > upper_band[i - 1]:
            upper_band[i] = ub[i]
        else:
            upper_band[i] = upper_band[i - 1]

        # Final lower band
        if lb[i] > lower_band[i - 1] or c[i - 1] < lower_band[i - 1]:
            lower_band[i] = lb[i]
        else:
            lower_band[i] = lower_band[i - 1]

        # Direction & supertrend value
        if np.isnan(st[i - 1]):
            direction[i] = 1.0
            st[i] = lower_band[i]
        elif st[i - 1] == upper_band[i - 1]:
            if c[i] <= upper_band[i]:
                direction[i] = -1.0
                st[i] = upper_band[i]
            else:
                direction[i] = 1.0
                st[i] = lower_band[i]
        else:
            if c[i] >= lower_band[i]:
                direction[i] = 1.0
                st[i] = lower_band[i]
            else:
                direction[i] = -1.0
                st[i] = upper_band[i]

    idx = close.index
    return {
        "supertrend": pd.Series(st, index=idx, name="supertrend"),
        "direction": pd.Series(direction, index=idx, name="direction"),
    }


# ---------------------------------------------------------------------------
# Ichimoku
# ---------------------------------------------------------------------------


def ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
) -> dict[str, pd.Series]:
    """Ichimoku Kinko Hyo (Ichimoku Cloud).

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    tenkan : int, default 9
        Tenkan-sen (conversion line) period.
    kijun : int, default 26
        Kijun-sen (base line) period.
    senkou_b : int, default 52
        Senkou Span B period.

    Returns
    -------
    dict[str, pd.Series]
        Keys: ``tenkan_sen``, ``kijun_sen``, ``senkou_span_a``,
        ``senkou_span_b``, ``chikou_span``.

    Notes
    -----
    Senkou Span A and B are shifted forward by ``kijun`` periods, and the
    Chikou Span is shifted backward by ``kijun`` periods, matching
    traditional charting convention.
    """
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")

    tenkan_sen = (
        high.rolling(window=tenkan, min_periods=tenkan).max()
        + low.rolling(window=tenkan, min_periods=tenkan).min()
    ) / 2.0
    tenkan_sen.name = "tenkan_sen"

    kijun_sen = (
        high.rolling(window=kijun, min_periods=kijun).max()
        + low.rolling(window=kijun, min_periods=kijun).min()
    ) / 2.0
    kijun_sen.name = "kijun_sen"

    senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0).shift(kijun)
    senkou_span_a.name = "senkou_span_a"

    senkou_span_b_raw = (
        high.rolling(window=senkou_b, min_periods=senkou_b).max()
        + low.rolling(window=senkou_b, min_periods=senkou_b).min()
    ) / 2.0
    senkou_span_b_val = senkou_span_b_raw.shift(kijun)
    senkou_span_b_val.name = "senkou_span_b"

    chikou_span = close.shift(-kijun)
    chikou_span.name = "chikou_span"

    return {
        "tenkan_sen": tenkan_sen,
        "kijun_sen": kijun_sen,
        "senkou_span_a": senkou_span_a,
        "senkou_span_b": senkou_span_b_val,
        "chikou_span": chikou_span,
    }


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------


def bollinger_bands(
    data: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> dict[str, pd.Series]:
    """Bollinger Bands.

    Parameters
    ----------
    data : pd.Series
        Price series (typically close).
    period : int, default 20
        SMA window length.
    std_dev : float, default 2.0
        Number of standard deviations for the bands.

    Returns
    -------
    dict[str, pd.Series]
        ``upper``, ``middle``, ``lower``, ``bandwidth``, ``percent_b``.
    """
    data = _validate_series(data)
    _validate_period(period)

    middle = sma(data, period)
    rolling_std = data.rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std
    bandwidth = (upper - lower) / middle
    percent_b = (data - lower) / (upper - lower)

    return {
        "upper": upper.rename("bb_upper"),
        "middle": middle.rename("bb_middle"),
        "lower": lower.rename("bb_lower"),
        "bandwidth": bandwidth.rename("bb_bandwidth"),
        "percent_b": percent_b.rename("bb_percent_b"),
    }


# ---------------------------------------------------------------------------
# Keltner Channel
# ---------------------------------------------------------------------------


def keltner_channel(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    multiplier: float = 1.5,
) -> dict[str, pd.Series]:
    """Keltner Channel.

    The middle line is an EMA of the close; upper and lower bands are
    offset by a multiple of the Average True Range.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    period : int, default 20
        EMA / ATR period.
    multiplier : float, default 1.5
        ATR multiplier.

    Returns
    -------
    dict[str, pd.Series]
        ``upper``, ``middle``, ``lower``.
    """
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")
    _validate_period(period)

    middle = ema(close, period)

    # True Range
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr_val = tr.rolling(window=period, min_periods=period).mean()

    upper = middle + multiplier * atr_val
    lower = middle - multiplier * atr_val

    return {
        "upper": upper.rename("kc_upper"),
        "middle": middle.rename("kc_middle"),
        "lower": lower.rename("kc_lower"),
    }


# ---------------------------------------------------------------------------
# Donchian Channel
# ---------------------------------------------------------------------------


def donchian_channel(
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
) -> dict[str, pd.Series]:
    """Donchian Channel.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    period : int, default 20
        Look-back period.

    Returns
    -------
    dict[str, pd.Series]
        ``upper``, ``lower``, ``middle``.
    """
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    _validate_period(period)

    upper = high.rolling(window=period, min_periods=period).max()
    lower = low.rolling(window=period, min_periods=period).min()
    middle = (upper + lower) / 2.0

    return {
        "upper": upper.rename("dc_upper"),
        "lower": lower.rename("dc_lower"),
        "middle": middle.rename("dc_middle"),
    }
