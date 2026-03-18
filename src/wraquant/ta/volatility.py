"""Volatility indicators.

This module provides indicators that measure the degree of price
variation over time. All functions accept ``pd.Series`` inputs and
return ``pd.Series``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "atr",
    "true_range",
    "natr",
    "bbwidth",
    "kc_width",
    "chaikin_volatility",
    "historical_volatility",
    "mass_index",
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


# ---------------------------------------------------------------------------
# True Range / ATR
# ---------------------------------------------------------------------------


def true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """True Range.

    ``TR = max(H - L, |H - C_prev|, |L - C_prev|)``

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.

    Returns
    -------
    pd.Series
        True Range values (always >= 0).
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    tr.name = "true_range"
    return tr


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range (ATR).

    Uses the Wilder smoothing method (``ewm(alpha=1/period)``).

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
    pd.Series
        ATR values (always >= 0).
    """
    _validate_period(period)

    tr_val = true_range(high, low, close)
    result = tr_val.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    result.name = "atr"
    return result


def natr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Normalized Average True Range (NATR).

    ``NATR = (ATR / close) * 100``

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    period : int, default 14
        ATR period.

    Returns
    -------
    pd.Series
        NATR as a percentage.
    """
    _validate_series(close, "close")
    atr_val = atr(high, low, close, period)
    result = (atr_val / close) * 100.0
    result.name = "natr"
    return result


# ---------------------------------------------------------------------------
# Bollinger Band Width
# ---------------------------------------------------------------------------


def bbwidth(
    data: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.Series:
    """Bollinger Band Width.

    ``BBWidth = (upper - lower) / middle``

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 20
        SMA period.
    std_dev : float, default 2.0
        Number of standard deviations.

    Returns
    -------
    pd.Series
        Bandwidth values.
    """
    _validate_series(data)
    _validate_period(period)

    middle = data.rolling(window=period, min_periods=period).mean()
    rolling_std = data.rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std
    result = (upper - lower) / middle
    result.name = "bbwidth"
    return result


# ---------------------------------------------------------------------------
# Keltner Channel Width
# ---------------------------------------------------------------------------


def kc_width(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    multiplier: float = 1.5,
) -> pd.Series:
    """Keltner Channel Width.

    ``KC_Width = (upper - lower) / middle``

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
    pd.Series
        Keltner Channel width values.
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")
    _validate_period(period)

    middle = _ema(close, period)
    atr_val = atr(high, low, close, period)
    upper = middle + multiplier * atr_val
    lower = middle - multiplier * atr_val
    result = (upper - lower) / middle
    result.name = "kc_width"
    return result


# ---------------------------------------------------------------------------
# Chaikin Volatility
# ---------------------------------------------------------------------------


def chaikin_volatility(
    high: pd.Series,
    low: pd.Series,
    period: int = 10,
    smoothing: int = 10,
) -> pd.Series:
    """Chaikin Volatility.

    Measures the rate of change of the EMA of the high-low spread.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    period : int, default 10
        EMA smoothing period for the H-L spread.
    smoothing : int, default 10
        ROC period applied to the smoothed spread.

    Returns
    -------
    pd.Series
        Chaikin Volatility as a percentage.
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_period(period)
    _validate_period(smoothing, "smoothing")

    hl_spread = high - low
    ema_spread = _ema(hl_spread, period)
    result = (
        (ema_spread - ema_spread.shift(smoothing)) / ema_spread.shift(smoothing)
    ) * 100.0
    result.name = "chaikin_volatility"
    return result


# ---------------------------------------------------------------------------
# Historical (Close-to-Close) Volatility
# ---------------------------------------------------------------------------


def historical_volatility(
    data: pd.Series,
    period: int = 21,
    annualize: bool = True,
) -> pd.Series:
    """Historical Volatility (close-to-close).

    Computes the rolling standard deviation of log returns. When
    *annualize* is ``True`` the result is scaled by ``sqrt(252)``.

    Parameters
    ----------
    data : pd.Series
        Price series (close).
    period : int, default 21
        Rolling window.
    annualize : bool, default True
        If ``True``, multiply by ``sqrt(252)``.

    Returns
    -------
    pd.Series
        Historical volatility (annualized if requested).
    """
    _validate_series(data)
    _validate_period(period)

    log_returns = np.log(data / data.shift(1))
    vol = log_returns.rolling(window=period, min_periods=period).std(ddof=1)
    if annualize:
        vol = vol * np.sqrt(252)
    vol.name = "historical_volatility"
    return vol


# ---------------------------------------------------------------------------
# Mass Index
# ---------------------------------------------------------------------------


def mass_index(
    high: pd.Series,
    low: pd.Series,
    period: int = 9,
    trigger: int = 25,
) -> pd.Series:
    """Mass Index.

    The Mass Index uses the high-low range to identify trend reversals
    based on range expansions. It accumulates the ratio of two EMAs of
    the range over the *trigger* period.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    period : int, default 9
        EMA period for the range.
    trigger : int, default 25
        Summation (rolling sum) window.

    Returns
    -------
    pd.Series
        Mass Index values. A *reversal bulge* occurs when the index
        rises above 27 and then falls below 26.5.
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_period(period)
    _validate_period(trigger, "trigger")

    hl_range = high - low
    single_ema = _ema(hl_range, period)
    double_ema = _ema(single_ema, period)
    ratio = single_ema / double_ema
    result = ratio.rolling(window=trigger, min_periods=trigger).sum()
    result.name = "mass_index"
    return result
