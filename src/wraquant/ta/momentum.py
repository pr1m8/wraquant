"""Momentum oscillator indicators.

This module contains oscillators that measure the speed and magnitude of
price movements. All functions accept ``pd.Series`` inputs and return
``pd.Series`` (or ``dict[str, pd.Series]`` for multi-output indicators).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "rsi",
    "stochastic",
    "stochastic_rsi",
    "macd",
    "williams_r",
    "cci",
    "roc",
    "momentum",
    "tsi",
    "awesome_oscillator",
    "ppo",
    "ultimate_oscillator",
    "cmo",
    "dpo",
    "kst",
    "connors_rsi",
    "fisher_transform",
    "elder_ray",
    "aroon_oscillator",
    "chande_forecast_oscillator",
    "balance_of_power",
    "qstick",
    "coppock_curve",
    "relative_vigor_index",
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
    """Internal EMA helper to avoid circular import."""
    return data.ewm(span=period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


def rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI).

    Uses the Wilder smoothing method (equivalent to ``ewm(alpha=1/period)``).

    Parameters
    ----------
    data : pd.Series
        Price series (typically close).
    period : int, default 14
        Look-back period.

    Returns
    -------
    pd.Series
        RSI values in the range [0, 100].
    """
    _validate_series(data)
    _validate_period(period)

    delta = data.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    result = 100.0 - (100.0 / (1.0 + rs))
    result.name = "rsi"
    return result


# ---------------------------------------------------------------------------
# Stochastic
# ---------------------------------------------------------------------------


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> dict[str, pd.Series]:
    """Stochastic Oscillator (%K / %D).

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    k_period : int, default 14
        Look-back for %K.
    d_period : int, default 3
        SMA smoothing period for %D.

    Returns
    -------
    dict[str, pd.Series]
        ``k`` (%K) and ``d`` (%D), both in [0, 100].
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()

    k = 100.0 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period, min_periods=d_period).mean()

    return {
        "k": k.rename("stoch_k"),
        "d": d.rename("stoch_d"),
    }


# ---------------------------------------------------------------------------
# Stochastic RSI
# ---------------------------------------------------------------------------


def stochastic_rsi(
    data: pd.Series,
    period: int = 14,
    k_period: int = 3,
    d_period: int = 3,
) -> dict[str, pd.Series]:
    """Stochastic RSI.

    Applies the Stochastic formula to the RSI output.

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 14
        RSI period.
    k_period : int, default 3
        Smoothing for %K.
    d_period : int, default 3
        Smoothing for %D.

    Returns
    -------
    dict[str, pd.Series]
        ``k`` and ``d``, both in [0, 100].
    """
    _validate_series(data)
    _validate_period(period)

    rsi_val = rsi(data, period)
    lowest_rsi = rsi_val.rolling(window=period, min_periods=period).min()
    highest_rsi = rsi_val.rolling(window=period, min_periods=period).max()

    stoch_rsi = (rsi_val - lowest_rsi) / (highest_rsi - lowest_rsi)
    k = stoch_rsi.rolling(window=k_period, min_periods=k_period).mean() * 100.0
    d = k.rolling(window=d_period, min_periods=d_period).mean()

    return {
        "k": k.rename("stochrsi_k"),
        "d": d.rename("stochrsi_d"),
    }


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


def macd(
    data: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, pd.Series]:
    """Moving Average Convergence Divergence (MACD).

    Parameters
    ----------
    data : pd.Series
        Price series.
    fast : int, default 12
        Fast EMA period.
    slow : int, default 26
        Slow EMA period.
    signal : int, default 9
        Signal EMA period.

    Returns
    -------
    dict[str, pd.Series]
        ``macd``, ``signal``, ``histogram``.
    """
    _validate_series(data)

    fast_ema = _ema(data, fast)
    slow_ema = _ema(data, slow)
    macd_line = fast_ema - slow_ema
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line

    return {
        "macd": macd_line.rename("macd"),
        "signal": signal_line.rename("macd_signal"),
        "histogram": hist.rename("macd_histogram"),
    }


# ---------------------------------------------------------------------------
# Williams %R
# ---------------------------------------------------------------------------


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Williams %R.

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
    pd.Series
        Williams %R values in [-100, 0].
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")
    _validate_period(period)

    highest_high = high.rolling(window=period, min_periods=period).max()
    lowest_low = low.rolling(window=period, min_periods=period).min()

    result = -100.0 * (highest_high - close) / (highest_high - lowest_low)
    result.name = "williams_r"
    return result


# ---------------------------------------------------------------------------
# CCI
# ---------------------------------------------------------------------------


def cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Commodity Channel Index (CCI).

    Uses Lambert's constant of 0.015.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    period : int, default 20
        Look-back period.

    Returns
    -------
    pd.Series
        CCI values (unbounded).
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")
    _validate_period(period)

    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(window=period, min_periods=period).mean()
    mean_dev = tp.rolling(window=period, min_periods=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    result = (tp - sma_tp) / (0.015 * mean_dev)
    result.name = "cci"
    return result


# ---------------------------------------------------------------------------
# ROC
# ---------------------------------------------------------------------------


def roc(data: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change (ROC) — percentage change over *period* bars.

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 10
        Look-back period.

    Returns
    -------
    pd.Series
        Percentage rate of change.
    """
    _validate_series(data)
    _validate_period(period)

    result = ((data - data.shift(period)) / data.shift(period)) * 100.0
    result.name = "roc"
    return result


# ---------------------------------------------------------------------------
# Momentum (simple difference)
# ---------------------------------------------------------------------------


def momentum(data: pd.Series, period: int = 10) -> pd.Series:
    """Price Momentum (difference over *period* bars).

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 10
        Look-back period.

    Returns
    -------
    pd.Series
        Momentum values.
    """
    _validate_series(data)
    _validate_period(period)

    result = data - data.shift(period)
    result.name = "momentum"
    return result


# ---------------------------------------------------------------------------
# TSI
# ---------------------------------------------------------------------------


def tsi(
    data: pd.Series,
    long: int = 25,
    short: int = 13,
    signal: int = 13,
) -> dict[str, pd.Series]:
    """True Strength Index (TSI).

    Parameters
    ----------
    data : pd.Series
        Price series.
    long : int, default 25
        Long EMA period.
    short : int, default 13
        Short EMA period.
    signal : int, default 13
        Signal line EMA period.

    Returns
    -------
    dict[str, pd.Series]
        ``tsi`` and ``signal``.
    """
    _validate_series(data)

    diff = data.diff()
    double_smoothed = _ema(_ema(diff, long), short)
    double_smoothed_abs = _ema(_ema(diff.abs(), long), short)
    tsi_line = 100.0 * double_smoothed / double_smoothed_abs
    signal_line = _ema(tsi_line, signal)

    return {
        "tsi": tsi_line.rename("tsi"),
        "signal": signal_line.rename("tsi_signal"),
    }


# ---------------------------------------------------------------------------
# Awesome Oscillator
# ---------------------------------------------------------------------------


def awesome_oscillator(
    high: pd.Series,
    low: pd.Series,
    fast: int = 5,
    slow: int = 34,
) -> pd.Series:
    """Awesome Oscillator (AO).

    ``AO = SMA(median_price, fast) - SMA(median_price, slow)``

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    fast : int, default 5
        Fast SMA period.
    slow : int, default 34
        Slow SMA period.

    Returns
    -------
    pd.Series
        AO values.
    """
    _validate_series(high, "high")
    _validate_series(low, "low")

    median_price = (high + low) / 2.0
    result = (
        median_price.rolling(window=fast, min_periods=fast).mean()
        - median_price.rolling(window=slow, min_periods=slow).mean()
    )
    result.name = "ao"
    return result


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------


def ppo(
    data: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, pd.Series]:
    """Percentage Price Oscillator (PPO).

    Like MACD but expressed as a percentage of the slow EMA.

    Parameters
    ----------
    data : pd.Series
        Price series.
    fast : int, default 12
        Fast EMA period.
    slow : int, default 26
        Slow EMA period.
    signal : int, default 9
        Signal EMA period.

    Returns
    -------
    dict[str, pd.Series]
        ``ppo``, ``signal``, ``histogram``.
    """
    _validate_series(data)

    fast_ema = _ema(data, fast)
    slow_ema = _ema(data, slow)
    ppo_line = ((fast_ema - slow_ema) / slow_ema) * 100.0
    signal_line = _ema(ppo_line, signal)
    hist = ppo_line - signal_line

    return {
        "ppo": ppo_line.rename("ppo"),
        "signal": signal_line.rename("ppo_signal"),
        "histogram": hist.rename("ppo_histogram"),
    }


# ---------------------------------------------------------------------------
# Ultimate Oscillator
# ---------------------------------------------------------------------------


def ultimate_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period1: int = 7,
    period2: int = 14,
    period3: int = 28,
) -> pd.Series:
    """Ultimate Oscillator.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    period1 : int, default 7
        First (shortest) period.
    period2 : int, default 14
        Second period.
    period3 : int, default 28
        Third (longest) period.

    Returns
    -------
    pd.Series
        Ultimate Oscillator values in [0, 100].
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    prev_close = close.shift(1)
    buying_pressure = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    avg1 = (
        buying_pressure.rolling(period1, min_periods=period1).sum()
        / true_range.rolling(period1, min_periods=period1).sum()
    )
    avg2 = (
        buying_pressure.rolling(period2, min_periods=period2).sum()
        / true_range.rolling(period2, min_periods=period2).sum()
    )
    avg3 = (
        buying_pressure.rolling(period3, min_periods=period3).sum()
        / true_range.rolling(period3, min_periods=period3).sum()
    )

    result = 100.0 * (4 * avg1 + 2 * avg2 + avg3) / 7.0
    result.name = "ultimate_oscillator"
    return result


# ---------------------------------------------------------------------------
# CMO
# ---------------------------------------------------------------------------


def cmo(data: pd.Series, period: int = 14) -> pd.Series:
    """Chande Momentum Oscillator (CMO).

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 14
        Look-back period.

    Returns
    -------
    pd.Series
        CMO values in [-100, 100].
    """
    _validate_series(data)
    _validate_period(period)

    delta = data.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    sum_gain = gain.rolling(window=period, min_periods=period).sum()
    sum_loss = loss.rolling(window=period, min_periods=period).sum()

    result = 100.0 * (sum_gain - sum_loss) / (sum_gain + sum_loss)
    result.name = "cmo"
    return result


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------


def dpo(data: pd.Series, period: int = 20) -> pd.Series:
    """Detrended Price Oscillator (DPO).

    ``DPO = close - SMA(close, period).shift(period // 2 + 1)``

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 20
        SMA period.

    Returns
    -------
    pd.Series
        DPO values (unbounded).
    """
    _validate_series(data)
    _validate_period(period)

    shift_amt = period // 2 + 1
    sma_val = data.rolling(window=period, min_periods=period).mean()
    result = data - sma_val.shift(shift_amt)
    result.name = "dpo"
    return result


# ---------------------------------------------------------------------------
# KST (Know Sure Thing)
# ---------------------------------------------------------------------------


def kst(
    data: pd.Series,
    roc1: int = 10,
    roc2: int = 15,
    roc3: int = 20,
    roc4: int = 30,
    sma1: int = 10,
    sma2: int = 10,
    sma3: int = 10,
    sma4: int = 15,
    signal_period: int = 9,
) -> dict[str, pd.Series]:
    """Know Sure Thing (KST) oscillator.

    Weighted sum of four smoothed rate-of-change values with weights 1, 2, 3, 4.

    Parameters
    ----------
    data : pd.Series
        Price series (typically close).
    roc1 : int, default 10
        First ROC period.
    roc2 : int, default 15
        Second ROC period.
    roc3 : int, default 20
        Third ROC period.
    roc4 : int, default 30
        Fourth ROC period.
    sma1 : int, default 10
        SMA smoothing for first ROC.
    sma2 : int, default 10
        SMA smoothing for second ROC.
    sma3 : int, default 10
        SMA smoothing for third ROC.
    sma4 : int, default 15
        SMA smoothing for fourth ROC.
    signal_period : int, default 9
        SMA period for the signal line.

    Returns
    -------
    dict[str, pd.Series]
        ``kst`` and ``signal``.

    Example
    -------
    >>> result = kst(close)
    >>> result["kst"]
    """
    _validate_series(data)

    r1 = data.pct_change(periods=roc1) * 100.0
    r2 = data.pct_change(periods=roc2) * 100.0
    r3 = data.pct_change(periods=roc3) * 100.0
    r4 = data.pct_change(periods=roc4) * 100.0

    s1 = r1.rolling(window=sma1, min_periods=sma1).mean()
    s2 = r2.rolling(window=sma2, min_periods=sma2).mean()
    s3 = r3.rolling(window=sma3, min_periods=sma3).mean()
    s4 = r4.rolling(window=sma4, min_periods=sma4).mean()

    kst_line = 1.0 * s1 + 2.0 * s2 + 3.0 * s3 + 4.0 * s4
    signal_line = kst_line.rolling(window=signal_period, min_periods=signal_period).mean()

    return {
        "kst": kst_line.rename("kst"),
        "signal": signal_line.rename("kst_signal"),
    }


# ---------------------------------------------------------------------------
# Connors RSI
# ---------------------------------------------------------------------------


def connors_rsi(
    data: pd.Series,
    rsi_period: int = 3,
    streak_period: int = 2,
    rank_period: int = 100,
) -> pd.Series:
    """Connors RSI — composite of RSI, up/down streak RSI, and percentile rank.

    ``ConnorsRSI = (RSI(close, rsi_period) + RSI(streak, streak_period)
    + PercentRank(pct_change, rank_period)) / 3``

    Parameters
    ----------
    data : pd.Series
        Price series (typically close).
    rsi_period : int, default 3
        Look-back for the standard RSI component.
    streak_period : int, default 2
        Look-back for the streak RSI component.
    rank_period : int, default 100
        Look-back for the percentile rank of the one-bar ROC.

    Returns
    -------
    pd.Series
        Connors RSI values in [0, 100].

    Example
    -------
    >>> result = connors_rsi(close)
    """
    _validate_series(data)
    _validate_period(rsi_period, "rsi_period")
    _validate_period(streak_period, "streak_period")
    _validate_period(rank_period, "rank_period")

    # Component 1: standard RSI
    rsi_component = rsi(data, rsi_period)

    # Component 2: streak RSI
    # Build up/down streak series
    diff = data.diff()
    streak = pd.Series(0.0, index=data.index)
    for i in range(1, len(data)):
        if diff.iloc[i] > 0:
            streak.iloc[i] = max(streak.iloc[i - 1], 0) + 1
        elif diff.iloc[i] < 0:
            streak.iloc[i] = min(streak.iloc[i - 1], 0) - 1
        else:
            streak.iloc[i] = 0.0
    streak_rsi_component = rsi(streak, streak_period)

    # Component 3: percentile rank of one-bar ROC
    pct_chg = data.pct_change() * 100.0
    pct_rank = pct_chg.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: np.sum(x[-1] >= x[:-1]) / (len(x) - 1) * 100.0, raw=True
    )

    result = (rsi_component + streak_rsi_component + pct_rank) / 3.0
    result.name = "connors_rsi"
    return result


# ---------------------------------------------------------------------------
# Fisher Transform
# ---------------------------------------------------------------------------


def fisher_transform(
    high: pd.Series,
    low: pd.Series,
    period: int = 9,
) -> dict[str, pd.Series]:
    """Fisher Transform — normalizes prices to a Gaussian distribution.

    Uses the midpoint ``(high + low) / 2``, normalizes to [-1, 1] over the
    look-back window, then applies the inverse hyperbolic tangent (Fisher
    Transform).

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    period : int, default 9
        Look-back period for normalization.

    Returns
    -------
    dict[str, pd.Series]
        ``fisher`` (current value) and ``signal`` (one-bar lag).

    Example
    -------
    >>> result = fisher_transform(high, low)
    >>> result["fisher"]
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_period(period)

    midpoint = (high + low) / 2.0
    lowest = midpoint.rolling(window=period, min_periods=period).min()
    highest = midpoint.rolling(window=period, min_periods=period).max()

    # Normalize to [-1, 1], clamp to avoid atanh domain errors
    raw = 2.0 * (midpoint - lowest) / (highest - lowest) - 1.0
    raw = raw.clip(lower=-0.999, upper=0.999)

    # Iterative EMA-style smoothing (Ehlers uses 0.5 factor)
    value = pd.Series(0.0, index=high.index)
    for i in range(len(raw)):
        if np.isnan(raw.iloc[i]):
            value.iloc[i] = np.nan
        else:
            prev = 0.0 if i == 0 or np.isnan(value.iloc[i - 1]) else value.iloc[i - 1]
            value.iloc[i] = 0.5 * raw.iloc[i] + 0.5 * prev

    fisher_line = pd.Series(np.nan, index=high.index)
    for i in range(len(value)):
        if not np.isnan(value.iloc[i]):
            fisher_line.iloc[i] = np.log((1.0 + value.iloc[i]) / (1.0 - value.iloc[i]))

    signal_line = fisher_line.shift(1)

    return {
        "fisher": fisher_line.rename("fisher"),
        "signal": signal_line.rename("fisher_signal"),
    }


# ---------------------------------------------------------------------------
# Elder Ray
# ---------------------------------------------------------------------------


def elder_ray(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 13,
) -> dict[str, pd.Series]:
    """Elder Ray Index — bull power and bear power.

    ``bull_power = high - EMA(close, period)``
    ``bear_power = low - EMA(close, period)``

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    period : int, default 13
        EMA period.

    Returns
    -------
    dict[str, pd.Series]
        ``bull_power`` and ``bear_power``.

    Example
    -------
    >>> result = elder_ray(high, low, close)
    >>> result["bull_power"]
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")
    _validate_period(period)

    ema_close = _ema(close, period)
    bull = high - ema_close
    bear = low - ema_close

    return {
        "bull_power": bull.rename("bull_power"),
        "bear_power": bear.rename("bear_power"),
    }


# ---------------------------------------------------------------------------
# Aroon Oscillator
# ---------------------------------------------------------------------------


def aroon_oscillator(
    high: pd.Series,
    low: pd.Series,
    period: int = 25,
) -> pd.Series:
    """Aroon Oscillator — difference between Aroon Up and Aroon Down.

    ``Aroon Up = 100 * (period - bars_since_high) / period``
    ``Aroon Down = 100 * (period - bars_since_low) / period``
    ``Aroon Oscillator = Aroon Up - Aroon Down``

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
    pd.Series
        Aroon Oscillator values in [-100, 100].

    Example
    -------
    >>> result = aroon_oscillator(high, low, period=25)
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_period(period)

    aroon_up = high.rolling(window=period + 1, min_periods=period + 1).apply(
        lambda x: 100.0 * (period - (period - np.argmax(x))) / period, raw=True
    )
    aroon_down = low.rolling(window=period + 1, min_periods=period + 1).apply(
        lambda x: 100.0 * (period - (period - np.argmin(x))) / period, raw=True
    )

    result = aroon_up - aroon_down
    result.name = "aroon_oscillator"
    return result


# ---------------------------------------------------------------------------
# Chande Forecast Oscillator
# ---------------------------------------------------------------------------


def chande_forecast_oscillator(
    data: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Chande Forecast Oscillator (CFO).

    Percentage difference between the close and the *period*-bar linear
    regression forecast value.

    ``CFO = ((close - linreg_forecast) / close) * 100``

    Parameters
    ----------
    data : pd.Series
        Price series (typically close).
    period : int, default 14
        Linear regression look-back period.

    Returns
    -------
    pd.Series
        CFO values (percentage, unbounded).

    Example
    -------
    >>> result = chande_forecast_oscillator(close, period=14)
    """
    _validate_series(data)
    _validate_period(period)

    def _linreg_forecast(window: np.ndarray) -> float:
        """Return the linear-regression value at the end of *window*."""
        n = len(window)
        x = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(x, window, 1)
        return intercept + slope * (n - 1)

    forecast = data.rolling(window=period, min_periods=period).apply(
        _linreg_forecast, raw=True
    )
    result = ((data - forecast) / data) * 100.0
    result.name = "cfo"
    return result


# ---------------------------------------------------------------------------
# Balance of Power
# ---------------------------------------------------------------------------


def balance_of_power(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Balance of Power (BOP).

    ``BOP = SMA((close - open) / (high - low), period)``

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
    period : int, default 14
        SMA smoothing period.

    Returns
    -------
    pd.Series
        BOP values in [-1, 1] (when smoothed, may slightly exceed bounds).

    Example
    -------
    >>> result = balance_of_power(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")
    _validate_period(period)

    hl_range = high - low
    raw_bop = (close - open_) / hl_range.replace(0, np.nan)
    result = raw_bop.rolling(window=period, min_periods=period).mean()
    result.name = "bop"
    return result


# ---------------------------------------------------------------------------
# QStick
# ---------------------------------------------------------------------------


def qstick(
    open_: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """QStick indicator — moving average of ``(close - open)``.

    A positive QStick indicates more bullish bars; a negative value indicates
    more bearish bars.

    Parameters
    ----------
    open_ : pd.Series
        Open prices.
    close : pd.Series
        Close prices.
    period : int, default 14
        SMA period.

    Returns
    -------
    pd.Series
        QStick values.

    Example
    -------
    >>> result = qstick(open_, close, period=14)
    """
    _validate_series(open_, "open_")
    _validate_series(close, "close")
    _validate_period(period)

    co_diff = close - open_
    result = co_diff.rolling(window=period, min_periods=period).mean()
    result.name = "qstick"
    return result


# ---------------------------------------------------------------------------
# Coppock Curve
# ---------------------------------------------------------------------------


def coppock_curve(
    data: pd.Series,
    wma_period: int = 10,
    long_roc: int = 14,
    short_roc: int = 11,
) -> pd.Series:
    """Coppock Curve — weighted moving average of the sum of two ROCs.

    ``Coppock = WMA(ROC(long_roc) + ROC(short_roc), wma_period)``

    Parameters
    ----------
    data : pd.Series
        Price series (typically close).
    wma_period : int, default 10
        Weighted moving average period.
    long_roc : int, default 14
        Long rate-of-change period.
    short_roc : int, default 11
        Short rate-of-change period.

    Returns
    -------
    pd.Series
        Coppock Curve values (unbounded).

    Example
    -------
    >>> result = coppock_curve(close)
    """
    _validate_series(data)
    _validate_period(wma_period, "wma_period")
    _validate_period(long_roc, "long_roc")
    _validate_period(short_roc, "short_roc")

    roc_long = data.pct_change(periods=long_roc) * 100.0
    roc_short = data.pct_change(periods=short_roc) * 100.0
    roc_sum = roc_long + roc_short

    # Weighted moving average (linearly weighted)
    weights = np.arange(1, wma_period + 1, dtype=float)
    result = roc_sum.rolling(window=wma_period, min_periods=wma_period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )
    result.name = "coppock"
    return result


# ---------------------------------------------------------------------------
# Relative Vigor Index
# ---------------------------------------------------------------------------


def relative_vigor_index(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
) -> dict[str, pd.Series]:
    """Relative Vigor Index (RVI).

    Measures the conviction of a recent price move by comparing the close-open
    range to the high-low range, smoothed with a symmetric-weighted moving
    average and then a simple moving average.

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
    period : int, default 10
        SMA smoothing period.

    Returns
    -------
    dict[str, pd.Series]
        ``rvi`` and ``signal``.

    Example
    -------
    >>> result = relative_vigor_index(open_, high, low, close)
    >>> result["rvi"]
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")
    _validate_period(period)

    co = close - open_
    hl = high - low

    # Symmetric weighted moving average (SWMA) with weights [1, 2, 2, 1] / 6
    def _swma(s: pd.Series) -> pd.Series:
        return (
            s + 2.0 * s.shift(1) + 2.0 * s.shift(2) + s.shift(3)
        ) / 6.0

    co_swma = _swma(co)
    hl_swma = _swma(hl)

    # Smooth numerator and denominator separately with SMA
    num = co_swma.rolling(window=period, min_periods=period).sum()
    den = hl_swma.rolling(window=period, min_periods=period).sum()

    rvi_line = num / den.replace(0, np.nan)

    # Signal line: SWMA of RVI
    signal_line = _swma(rvi_line)

    return {
        "rvi": rvi_line.rename("rvi"),
        "signal": signal_line.rename("rvi_signal"),
    }
