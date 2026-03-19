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
    "schaff_momentum",
    "price_momentum_oscillator",
    "klinger_oscillator",
    "stochastic_momentum_index",
    "inertia",
    "squeeze_histogram",
    "center_of_gravity",
    "psychological_line",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


from wraquant.ta._validators import validate_period as _validate_period
from wraquant.ta._validators import validate_series as _validate_series


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
    signal_line = kst_line.rolling(
        window=signal_period, min_periods=signal_period
    ).mean()

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
        return (s + 2.0 * s.shift(1) + 2.0 * s.shift(2) + s.shift(3)) / 6.0

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


# ---------------------------------------------------------------------------
# Schaff Momentum
# ---------------------------------------------------------------------------


def schaff_momentum(
    data: pd.Series,
    period: int = 10,
    fast: int = 23,
    slow: int = 50,
) -> pd.Series:
    """Schaff Trend Cycle applied to momentum (Schaff Momentum).

    Applies two rounds of Stochastic smoothing to the difference between
    fast and slow EMAs, producing a bounded oscillator in [0, 100].

    Parameters
    ----------
    data : pd.Series
        Price series (typically close).
    period : int, default 10
        Stochastic look-back period.
    fast : int, default 23
        Fast EMA period.
    slow : int, default 50
        Slow EMA period.

    Returns
    -------
    pd.Series
        Schaff Momentum values in [0, 100].

    Example
    -------
    >>> result = schaff_momentum(close, period=10)
    """
    _validate_series(data)
    _validate_period(period)

    macd_line = _ema(data, fast) - _ema(data, slow)

    # First Stochastic of MACD
    lowest = macd_line.rolling(window=period, min_periods=period).min()
    highest = macd_line.rolling(window=period, min_periods=period).max()
    hl_range = (highest - lowest).replace(0, np.nan)
    frac1 = ((macd_line - lowest) / hl_range) * 100.0

    # Smooth with EMA-like factor (Schaff uses 0.5)
    pf = pd.Series(np.nan, index=data.index)
    for i in range(len(frac1)):
        if np.isnan(frac1.iloc[i]):
            continue
        prev = (
            pf.iloc[i - 1] if i > 0 and not np.isnan(pf.iloc[i - 1]) else frac1.iloc[i]
        )
        pf.iloc[i] = prev + 0.5 * (frac1.iloc[i] - prev)

    # Second Stochastic of the smoothed result
    lowest2 = pf.rolling(window=period, min_periods=period).min()
    highest2 = pf.rolling(window=period, min_periods=period).max()
    hl_range2 = (highest2 - lowest2).replace(0, np.nan)
    frac2 = ((pf - lowest2) / hl_range2) * 100.0

    result = pd.Series(np.nan, index=data.index)
    for i in range(len(frac2)):
        if np.isnan(frac2.iloc[i]):
            continue
        prev = (
            result.iloc[i - 1]
            if i > 0 and not np.isnan(result.iloc[i - 1])
            else frac2.iloc[i]
        )
        result.iloc[i] = prev + 0.5 * (frac2.iloc[i] - prev)

    result.name = "schaff_momentum"
    return result


# ---------------------------------------------------------------------------
# Price Momentum Oscillator
# ---------------------------------------------------------------------------


def price_momentum_oscillator(
    data: pd.Series,
    short: int = 35,
    long: int = 20,
    signal: int = 10,
) -> dict[str, pd.Series]:
    """Price Momentum Oscillator (PMO) — double-smoothed ROC.

    ``PMO = EMA(EMA(ROC(1), short), long)``

    Developed by Carl Swenlin, the PMO is a double-smoothed one-bar
    rate-of-change, scaled by a factor of 10 for visibility.

    Parameters
    ----------
    data : pd.Series
        Price series (typically close).
    short : int, default 35
        First smoothing EMA period.
    long : int, default 20
        Second smoothing EMA period.
    signal : int, default 10
        Signal line EMA period.

    Returns
    -------
    dict[str, pd.Series]
        ``pmo`` and ``signal``.

    Example
    -------
    >>> result = price_momentum_oscillator(close)
    >>> result["pmo"]
    """
    _validate_series(data)
    _validate_period(short, "short")
    _validate_period(long, "long")
    _validate_period(signal, "signal")

    roc_1 = ((data - data.shift(1)) / data.shift(1)) * 100.0
    smoothed_1 = _ema(roc_1 * 10.0, short)
    pmo_line = _ema(smoothed_1, long)
    signal_line = _ema(pmo_line, signal)

    return {
        "pmo": pmo_line.rename("pmo"),
        "signal": signal_line.rename("pmo_signal"),
    }


# ---------------------------------------------------------------------------
# Klinger Oscillator (momentum view)
# ---------------------------------------------------------------------------


def klinger_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    fast: int = 34,
    slow: int = 55,
    signal: int = 13,
) -> dict[str, pd.Series]:
    """Klinger Volume Oscillator — momentum-oriented view.

    Uses the relationship between price trend direction and volume to
    predict price reversals.

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
    fast : int, default 34
        Fast EMA period.
    slow : int, default 55
        Slow EMA period.
    signal : int, default 13
        Signal line EMA period.

    Returns
    -------
    dict[str, pd.Series]
        ``kvo`` and ``signal``.

    Example
    -------
    >>> result = klinger_oscillator(high, low, close, volume)
    >>> result["kvo"]
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")
    _validate_series(volume, "volume")

    hlc3 = (high + low + close) / 3.0
    trend = np.sign(hlc3 - hlc3.shift(1))
    trend.iloc[0] = 0

    dm = high - low
    dm_arr = dm.values.copy()
    trend_arr = trend.values

    cm = np.empty(len(dm))
    cm[0] = dm_arr[0]
    for i in range(1, len(dm)):
        if trend_arr[i] == trend_arr[i - 1]:
            cm[i] = cm[i - 1] + dm_arr[i]
        else:
            cm[i] = dm_arr[i]

    cm_series = pd.Series(cm, index=close.index)
    ratio = pd.Series(
        np.where(cm_series != 0, 2.0 * dm / cm_series - 1.0, 0.0),
        index=close.index,
    )
    vf = volume * np.abs(ratio) * trend * 100.0

    kvo = _ema(vf, fast) - _ema(vf, slow)
    signal_line = _ema(kvo, signal)

    return {
        "kvo": kvo.rename("kvo"),
        "signal": signal_line.rename("kvo_signal"),
    }


# ---------------------------------------------------------------------------
# Stochastic Momentum Index
# ---------------------------------------------------------------------------


def stochastic_momentum_index(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> dict[str, pd.Series]:
    """Stochastic Momentum Index (SMI).

    The SMI is a refinement of the Stochastic Oscillator that measures the
    distance of the close relative to the midpoint of the high-low range,
    double-smoothed with EMAs.

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
    smooth_k : int, default 3
        First smoothing EMA period.
    smooth_d : int, default 3
        Signal line EMA period.

    Returns
    -------
    dict[str, pd.Series]
        ``smi`` and ``signal``.

    Example
    -------
    >>> result = stochastic_momentum_index(high, low, close)
    >>> result["smi"]
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")
    _validate_period(period)

    highest_high = high.rolling(window=period, min_periods=period).max()
    lowest_low = low.rolling(window=period, min_periods=period).min()

    midpoint = (highest_high + lowest_low) / 2.0
    diff = close - midpoint
    hl_range = highest_high - lowest_low

    # Double smooth both numerator and denominator
    smoothed_diff = _ema(_ema(diff, smooth_k), smooth_k)
    smoothed_range = _ema(_ema(hl_range, smooth_k), smooth_k)

    smi = (smoothed_diff / (smoothed_range / 2.0).replace(0, np.nan)) * 100.0
    signal_line = _ema(smi, smooth_d)

    return {
        "smi": smi.rename("smi"),
        "signal": signal_line.rename("smi_signal"),
    }


# ---------------------------------------------------------------------------
# Inertia
# ---------------------------------------------------------------------------


def inertia(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    rvi_period: int = 10,
    linreg_period: int = 20,
) -> pd.Series:
    """Ehlers Inertia Indicator — RVI smoothed by linear regression.

    Applies a linear regression (moving regression value) to the Relative
    Volatility Index (RVI) to produce a momentum-like indicator.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    rvi_period : int, default 10
        Period for computing the RVI.
    linreg_period : int, default 20
        Linear regression look-back period.

    Returns
    -------
    pd.Series
        Inertia values. Values above 50 are bullish; below 50 are bearish.

    Example
    -------
    >>> result = inertia(close, high, low)
    """
    _validate_series(close, "close")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_period(rvi_period, "rvi_period")
    _validate_period(linreg_period, "linreg_period")

    # Compute RVI (relative volatility index) using std dev
    delta = close.diff()
    up_vol = pd.Series(
        np.where(delta > 0, close.rolling(rvi_period).std(), 0.0),
        index=close.index,
    )
    down_vol = pd.Series(
        np.where(delta <= 0, close.rolling(rvi_period).std(), 0.0),
        index=close.index,
    )

    avg_up = _ema(up_vol, rvi_period)
    avg_down = _ema(down_vol, rvi_period)
    rvi_values = 100.0 * avg_up / (avg_up + avg_down).replace(0, np.nan)

    # Apply linear regression smoothing
    def _linreg_value(window: np.ndarray) -> float:
        n = len(window)
        x = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(x, window, 1)
        return intercept + slope * (n - 1)

    result = rvi_values.rolling(window=linreg_period, min_periods=linreg_period).apply(
        _linreg_value, raw=True
    )
    result.name = "inertia"
    return result


# ---------------------------------------------------------------------------
# Squeeze Histogram
# ---------------------------------------------------------------------------


def squeeze_histogram(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    linreg_period: int = 20,
) -> pd.Series:
    """Squeeze Histogram — momentum component of the TTM Squeeze.

    Computes the linear regression value of ``close - midline`` where
    midline is the average of the highest high and lowest low over the
    period, combined with the SMA.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    period : int, default 20
        Donchian / Bollinger look-back period.
    linreg_period : int, default 20
        Linear regression period for the momentum value.

    Returns
    -------
    pd.Series
        Squeeze momentum histogram values.

    Example
    -------
    >>> result = squeeze_histogram(high, low, close)
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")
    _validate_period(period)
    _validate_period(linreg_period, "linreg_period")

    highest_high = high.rolling(window=period, min_periods=period).max()
    lowest_low = low.rolling(window=period, min_periods=period).min()
    sma = close.rolling(window=period, min_periods=period).mean()

    midline = (highest_high + lowest_low) / 2.0
    delta = close - (midline + sma) / 2.0

    def _linreg_value(window: np.ndarray) -> float:
        n = len(window)
        x = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(x, window, 1)
        return intercept + slope * (n - 1)

    result = delta.rolling(window=linreg_period, min_periods=linreg_period).apply(
        _linreg_value, raw=True
    )
    result.name = "squeeze_histogram"
    return result


# ---------------------------------------------------------------------------
# Center of Gravity
# ---------------------------------------------------------------------------


def center_of_gravity(
    data: pd.Series,
    period: int = 10,
) -> pd.Series:
    """Ehlers Center of Gravity Oscillator.

    A weighted sum of recent prices where more recent bars receive higher
    weight, normalized by a simple sum. This produces a leading oscillator.

    ``CoG = -sum(price[i] * (i+1), i=0..period-1) / sum(price[i], i=0..period-1)``

    Parameters
    ----------
    data : pd.Series
        Price series (typically close).
    period : int, default 10
        Look-back period.

    Returns
    -------
    pd.Series
        Center of Gravity values (oscillates around zero).

    Example
    -------
    >>> result = center_of_gravity(close, period=10)
    """
    _validate_series(data)
    _validate_period(period)

    weights = np.arange(1, period + 1, dtype=float)

    def _cog(window: np.ndarray) -> float:
        denom = np.sum(window)
        if denom == 0:
            return np.nan
        return -np.dot(window, weights) / denom

    result = data.rolling(window=period, min_periods=period).apply(_cog, raw=True)
    result.name = "center_of_gravity"
    return result


# ---------------------------------------------------------------------------
# Psychological Line
# ---------------------------------------------------------------------------


def psychological_line(
    data: pd.Series,
    period: int = 12,
) -> pd.Series:
    """Psychological Line — percentage of up days over N periods.

    ``PSY = (number of up bars in period) / period * 100``

    Values above 50 suggest bullish sentiment; below 50 suggest bearish.

    Parameters
    ----------
    data : pd.Series
        Price series (typically close).
    period : int, default 12
        Look-back period.

    Returns
    -------
    pd.Series
        Psychological Line values in [0, 100].

    Example
    -------
    >>> result = psychological_line(close, period=12)
    """
    _validate_series(data)
    _validate_period(period)

    up = (data.diff() > 0).astype(float)
    result = up.rolling(window=period, min_periods=period).sum() / period * 100.0
    result.name = "psychological_line"
    return result
