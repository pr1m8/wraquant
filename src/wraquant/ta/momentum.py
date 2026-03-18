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
