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
    "garman_klass",
    "parkinson",
    "rogers_satchell",
    "yang_zhang",
    "close_to_close",
    "ulcer_index",
    "relative_volatility_index",
    "acceleration_bands",
    "standard_deviation",
    "variance",
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


# ---------------------------------------------------------------------------
# Garman-Klass Volatility
# ---------------------------------------------------------------------------


def garman_klass(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    open_: pd.Series,
    period: int = 21,
    annualize: bool = True,
) -> pd.Series:
    """Garman-Klass volatility estimator.

    An efficient OHLC volatility estimator that uses open, high, low, and
    close prices. More efficient than close-to-close because it uses
    intraday range information.

    ``GK = sqrt((1/n) * sum(0.5*(ln(H/L))^2 - (2*ln(2)-1)*(ln(C/O))^2))``

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    open_ : pd.Series
        Open prices.
    period : int, default 21
        Rolling window.
    annualize : bool, default True
        If ``True``, multiply by ``sqrt(252)``.

    Returns
    -------
    pd.Series
        Garman-Klass volatility estimate.

    Example
    -------
    >>> gk = garman_klass(high, low, close, open_, period=21)
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")
    _validate_series(open_, "open_")
    _validate_period(period)

    log_hl = np.log(high / low)
    log_co = np.log(close / open_)

    term = 0.5 * log_hl**2 - (2.0 * np.log(2.0) - 1.0) * log_co**2
    vol = term.rolling(window=period, min_periods=period).mean().apply(
        lambda x: np.sqrt(x) if x >= 0 else np.nan
    )
    if annualize:
        vol = vol * np.sqrt(252)
    vol.name = "garman_klass"
    return vol


# ---------------------------------------------------------------------------
# Parkinson Volatility
# ---------------------------------------------------------------------------


def parkinson(
    high: pd.Series,
    low: pd.Series,
    period: int = 21,
    annualize: bool = True,
) -> pd.Series:
    """Parkinson volatility estimator.

    Uses the high-low range to estimate volatility, which is more efficient
    than close-to-close since it captures intraday extremes.

    ``Parkinson = sqrt((1 / (4 * n * ln(2))) * sum((ln(H/L))^2))``

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    period : int, default 21
        Rolling window.
    annualize : bool, default True
        If ``True``, multiply by ``sqrt(252)``.

    Returns
    -------
    pd.Series
        Parkinson volatility estimate.

    Example
    -------
    >>> pk = parkinson(high, low, period=21)
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_period(period)

    log_hl_sq = np.log(high / low) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    vol = (
        factor * log_hl_sq.rolling(window=period, min_periods=period).mean()
    ).apply(lambda x: np.sqrt(x) if x >= 0 else np.nan)
    if annualize:
        vol = vol * np.sqrt(252)
    vol.name = "parkinson"
    return vol


# ---------------------------------------------------------------------------
# Rogers-Satchell Volatility
# ---------------------------------------------------------------------------


def rogers_satchell(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    open_: pd.Series,
    period: int = 21,
    annualize: bool = True,
) -> pd.Series:
    """Rogers-Satchell volatility estimator.

    Accounts for non-zero drift (trending markets), making it more robust
    than Parkinson or Garman-Klass for trending assets.

    ``RS = sqrt((1/n) * sum(ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)))``

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    open_ : pd.Series
        Open prices.
    period : int, default 21
        Rolling window.
    annualize : bool, default True
        If ``True``, multiply by ``sqrt(252)``.

    Returns
    -------
    pd.Series
        Rogers-Satchell volatility estimate.

    Example
    -------
    >>> rs = rogers_satchell(high, low, close, open_, period=21)
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")
    _validate_series(open_, "open_")
    _validate_period(period)

    term = (
        np.log(high / close) * np.log(high / open_)
        + np.log(low / close) * np.log(low / open_)
    )
    vol = term.rolling(window=period, min_periods=period).mean().apply(
        lambda x: np.sqrt(x) if x >= 0 else np.nan
    )
    if annualize:
        vol = vol * np.sqrt(252)
    vol.name = "rogers_satchell"
    return vol


# ---------------------------------------------------------------------------
# Yang-Zhang Volatility
# ---------------------------------------------------------------------------


def yang_zhang(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    open_: pd.Series,
    period: int = 21,
    annualize: bool = True,
) -> pd.Series:
    """Yang-Zhang volatility estimator.

    The most efficient OHLC volatility estimator. Combines overnight
    (close-to-open) volatility, open-to-close volatility, and the
    Rogers-Satchell estimator.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    open_ : pd.Series
        Open prices.
    period : int, default 21
        Rolling window.
    annualize : bool, default True
        If ``True``, multiply by ``sqrt(252)``.

    Returns
    -------
    pd.Series
        Yang-Zhang volatility estimate.

    Example
    -------
    >>> yz = yang_zhang(high, low, close, open_, period=21)
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")
    _validate_series(open_, "open_")
    _validate_period(period)

    k = 0.34 / (1.34 + (period + 1) / (period - 1))

    # Overnight volatility: log(open / prev_close)
    log_oc = np.log(open_ / close.shift(1))
    overnight_var = log_oc.rolling(window=period, min_periods=period).var(ddof=1)

    # Open-to-close volatility: log(close / open)
    log_co = np.log(close / open_)
    openclose_var = log_co.rolling(window=period, min_periods=period).var(ddof=1)

    # Rogers-Satchell component
    rs_term = (
        np.log(high / close) * np.log(high / open_)
        + np.log(low / close) * np.log(low / open_)
    )
    rs_var = rs_term.rolling(window=period, min_periods=period).mean()

    vol_sq = overnight_var + k * openclose_var + (1.0 - k) * rs_var
    vol = vol_sq.apply(lambda x: np.sqrt(x) if x >= 0 else np.nan)
    if annualize:
        vol = vol * np.sqrt(252)
    vol.name = "yang_zhang"
    return vol


# ---------------------------------------------------------------------------
# Close-to-Close Volatility
# ---------------------------------------------------------------------------


def close_to_close(
    data: pd.Series,
    period: int = 21,
    annualize: bool = True,
) -> pd.Series:
    """Close-to-close volatility (standard deviation of log returns).

    The simplest volatility estimator based on daily log returns. This is
    equivalent to :func:`historical_volatility` but named explicitly to
    distinguish it from range-based estimators.

    Parameters
    ----------
    data : pd.Series
        Close price series.
    period : int, default 21
        Rolling window.
    annualize : bool, default True
        If ``True``, multiply by ``sqrt(252)``.

    Returns
    -------
    pd.Series
        Close-to-close volatility.

    Example
    -------
    >>> cc = close_to_close(close, period=21)
    """
    _validate_series(data)
    _validate_period(period)

    log_returns = np.log(data / data.shift(1))
    vol = log_returns.rolling(window=period, min_periods=period).std(ddof=1)
    if annualize:
        vol = vol * np.sqrt(252)
    vol.name = "close_to_close"
    return vol


# ---------------------------------------------------------------------------
# Ulcer Index
# ---------------------------------------------------------------------------


def ulcer_index(
    data: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Ulcer Index.

    Measures downside volatility by computing the quadratic mean of
    percentage drawdowns from the rolling maximum over the given period.
    Higher values indicate greater drawdown depth and duration.

    ``UI = sqrt(mean(R^2))`` where ``R = 100 * (C - max(C, n)) / max(C, n)``

    Parameters
    ----------
    data : pd.Series
        Close price series.
    period : int, default 14
        Rolling window.

    Returns
    -------
    pd.Series
        Ulcer Index values (always >= 0).

    Example
    -------
    >>> ui = ulcer_index(close, period=14)
    """
    _validate_series(data)
    _validate_period(period)

    rolling_max = data.rolling(window=period, min_periods=period).max()
    pct_drawdown = 100.0 * (data - rolling_max) / rolling_max
    result = (
        (pct_drawdown**2)
        .rolling(window=period, min_periods=period)
        .mean()
        .apply(lambda x: np.sqrt(x) if x >= 0 else np.nan)
    )
    result.name = "ulcer_index"
    return result


# ---------------------------------------------------------------------------
# Relative Volatility Index
# ---------------------------------------------------------------------------


def relative_volatility_index(
    data: pd.Series,
    period: int = 10,
    smoothing: int = 14,
) -> pd.Series:
    """Relative Volatility Index (RVI).

    Applies the RSI formula to the rolling standard deviation of closes
    rather than to price changes. Values above 50 suggest increasing
    volatility, below 50 decreasing.

    Parameters
    ----------
    data : pd.Series
        Close price series.
    period : int, default 10
        Standard deviation lookback.
    smoothing : int, default 14
        RSI-style smoothing period applied to the std-dev changes.

    Returns
    -------
    pd.Series
        RVI values oscillating between 0 and 100.

    Example
    -------
    >>> rvi = relative_volatility_index(close, period=10, smoothing=14)
    """
    _validate_series(data)
    _validate_period(period)
    _validate_period(smoothing, "smoothing")

    std_dev = data.rolling(window=period, min_periods=period).std(ddof=1)
    delta = std_dev.diff()

    gain = pd.Series(np.where(delta > 0, delta, 0.0), index=data.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0.0), index=data.index)

    avg_gain = gain.ewm(
        alpha=1.0 / smoothing, min_periods=smoothing, adjust=False
    ).mean()
    avg_loss = loss.ewm(
        alpha=1.0 / smoothing, min_periods=smoothing, adjust=False
    ).mean()

    rs = avg_gain / avg_loss
    result = 100.0 - (100.0 / (1.0 + rs))
    result.name = "relative_volatility_index"
    return result


# ---------------------------------------------------------------------------
# Acceleration Bands
# ---------------------------------------------------------------------------


def acceleration_bands(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    factor: float = 0.001,
) -> dict[str, pd.Series]:
    """Acceleration Bands.

    Bands that widen with high-low range acceleration, narrowing during
    consolidation. Uses ``factor * (high - low) / (high + low)`` as the
    width multiplier.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    period : int, default 20
        SMA period.
    factor : float, default 0.001
        Width factor applied to range acceleration.

    Returns
    -------
    dict[str, pd.Series]
        Dictionary with keys ``"upper"``, ``"middle"``, ``"lower"``.

    Example
    -------
    >>> bands = acceleration_bands(high, low, close, period=20)
    >>> upper = bands["upper"]
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")
    _validate_period(period)

    hl_ratio = factor * (high - low) / ((high + low) / 2.0)
    upper_band = (high * (1.0 + hl_ratio)).rolling(
        window=period, min_periods=period
    ).mean()
    lower_band = (low * (1.0 - hl_ratio)).rolling(
        window=period, min_periods=period
    ).mean()
    middle = close.rolling(window=period, min_periods=period).mean()

    upper_band.name = "acceleration_bands_upper"
    middle.name = "acceleration_bands_middle"
    lower_band.name = "acceleration_bands_lower"

    return {"upper": upper_band, "middle": middle, "lower": lower_band}


# ---------------------------------------------------------------------------
# Standard Deviation (Rolling)
# ---------------------------------------------------------------------------


def standard_deviation(
    data: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Rolling standard deviation.

    Computes the rolling sample standard deviation over the given period.

    Parameters
    ----------
    data : pd.Series
        Input price series.
    period : int, default 20
        Rolling window.

    Returns
    -------
    pd.Series
        Rolling standard deviation values.

    Example
    -------
    >>> sd = standard_deviation(close, period=20)
    """
    _validate_series(data)
    _validate_period(period)

    result = data.rolling(window=period, min_periods=period).std(ddof=1)
    result.name = "standard_deviation"
    return result


# ---------------------------------------------------------------------------
# Variance (Rolling)
# ---------------------------------------------------------------------------


def variance(
    data: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Rolling variance.

    Computes the rolling sample variance over the given period.

    Parameters
    ----------
    data : pd.Series
        Input price series.
    period : int, default 20
        Rolling window.

    Returns
    -------
    pd.Series
        Rolling variance values.

    Example
    -------
    >>> v = variance(close, period=20)
    """
    _validate_series(data)
    _validate_period(period)

    result = data.rolling(window=period, min_periods=period).var(ddof=1)
    result.name = "variance"
    return result
