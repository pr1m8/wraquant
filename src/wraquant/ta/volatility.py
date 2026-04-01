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


from wraquant.ta._validators import validate_period as _validate_period
from wraquant.ta._validators import validate_series as _validate_series


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

    The single-bar volatility measure that accounts for gaps.

    Interpretation:
        - **High TR**: Large price movement -- high volatility bar.
        - **Low TR**: Small price movement -- low volatility bar.
        - Spikes in TR often occur at trend changes or breakouts.
        - TR forms the basis of ATR and many other volatility indicators.

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
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")

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
    The standard measure of market volatility.

    Interpretation:
        - **Higher ATR**: More volatile market -- wider price swings.
        - **Lower ATR**: Less volatile -- tight price action.
        - **Rising ATR**: Volatility is increasing (often at trend
          beginnings or during strong moves).
        - **Falling ATR**: Volatility is decreasing (often during
          consolidation, before a breakout).
        - ATR does not indicate direction, only the magnitude of
          price movement.

    Trading rules:
        - **Stop placement**: Set stop-loss at 2x or 3x ATR from
          entry to account for normal market noise.
        - **Position sizing**: Risk a fixed dollar amount per trade;
          divide by ATR to determine share count.
        - **Breakout confirmation**: A breakout with rising ATR is
          more likely to sustain than one with falling ATR.
        - **Trailing stop**: Trail by 2-3x ATR below the highest
          high (for longs).

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

    Interpretation:
        - Same as ATR but expressed as a percentage of price, making
          it comparable across different assets and price levels.
        - **Higher NATR**: More volatile (in percentage terms).
        - **Lower NATR**: Less volatile.
        - Useful for ranking assets by volatility or for building
          volatility-weighted portfolios.

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
    close = _validate_series(close, "close")
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

    Interpretation:
        - **Low BBWidth (squeeze)**: Bollinger Bands are narrow --
          volatility is low. A breakout is imminent. This is the
          key signal: low volatility precedes high volatility.
        - **High BBWidth**: Bollinger Bands are wide -- volatility
          is high. The move may be overextended.
        - **BBWidth at 6-month low**: Strong squeeze -- prepare
          for a significant breakout.
        - **BBWidth expanding**: Breakout in progress.

    Trading rules:
        - Look for BBWidth at historical lows (squeeze).
        - When the squeeze releases (BBWidth starts expanding),
          trade the breakout direction.
        - Combine with momentum indicators to determine direction.

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
    data = _validate_series(data)
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

    Interpretation:
        - Same concept as BBWidth but based on ATR instead of
          standard deviation.
        - **Narrow KC Width**: Low ATR volatility, potential squeeze.
        - **Wide KC Width**: High ATR volatility, extended move.
        - Used with BBWidth for the TTM Squeeze: when BB is inside
          KC, a squeeze is active.

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
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")
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

    Interpretation:
        - **Rising**: Volatility is increasing (range is expanding).
        - **Falling**: Volatility is decreasing (range is contracting).
        - **Spike up**: Can indicate a market top (panic/climax).
        - **Spike down**: Can indicate a market bottom (capitulation
          followed by quiet).

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
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
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

    Interpretation:
        - **High HV** (e.g. > 30% annualized for equities): Asset is
          highly volatile. Wider stops and smaller positions needed.
        - **Low HV** (e.g. < 15% annualized): Asset is calm. Tighter
          stops and larger positions possible.
        - **HV vs Implied Volatility**: If HV < IV, options are
          relatively expensive (sell premium). If HV > IV, options
          are cheap (buy premium).
        - **Rising HV**: Uncertainty increasing. Often accompanies
          selloffs.
        - **Falling HV**: Market calming down. Often accompanies
          gradual rallies.

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
    data = _validate_series(data)
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

    Interpretation:
        - **Reversal bulge**: The key signal. When Mass Index rises
          above 27 and then falls back below 26.5, a trend reversal
          is likely (regardless of direction).
        - The indicator does not tell you the direction of the
          reversal, only that one is coming.
        - Combine with a trend indicator to determine which direction
          the reversal will take.

    Trading rules:
        - When Mass Index crosses above 27 then back below 26.5
          (reversal bulge), prepare for a trend change.
        - Use a 9-period EMA crossover or similar to determine the
          new trend direction.

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
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
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

    Interpretation:
        - Values are directly comparable to historical volatility.
        - **Higher efficiency**: Uses the same data as close-to-close
          but extracts more information, producing tighter estimates.
        - Compare with Parkinson and Yang-Zhang to assess which
          estimator best suits your data.
        - Does not handle overnight gaps well; use Yang-Zhang for
          assets with significant overnight risk.

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
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")
    open_ = _validate_series(open_, "open_")
    _validate_period(period)

    from wraquant.vol.realized import garman_klass as _garman_klass

    return _garman_klass(
        open_=open_, high=high, low=low, close=close, window=period, annualize=annualize
    )


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

    Interpretation:
        - Approximately 5x more efficient than close-to-close.
        - Tends to underestimate true volatility when there are
          overnight gaps (since it ignores open/close).
        - Best suited for assets that trade continuously (e.g. forex).

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
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    _validate_period(period)

    from wraquant.vol.realized import parkinson as _parkinson

    return _parkinson(high, low, window=period, annualize=annualize)


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

    Interpretation:
        - Better than Garman-Klass for assets with strong trends,
          because it does not assume zero drift.
        - Still does not handle overnight gaps; for that, use
          Yang-Zhang.

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
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")
    open_ = _validate_series(open_, "open_")
    _validate_period(period)

    from wraquant.vol.realized import rogers_satchell as _rogers_satchell

    return _rogers_satchell(
        open_=open_, high=high, low=low, close=close, window=period, annualize=annualize
    )


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

    Interpretation:
        - **The gold standard** for OHLC volatility estimation.
        - Handles both overnight gaps and intraday drift.
        - Use this as the default volatility estimator when you have
          full OHLC data.
        - Compare with close-to-close: if Yang-Zhang is significantly
          higher, overnight/gap risk is material.

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
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")
    open_ = _validate_series(open_, "open_")
    _validate_period(period)

    from wraquant.vol.realized import yang_zhang as _yang_zhang

    return _yang_zhang(
        open_=open_, high=high, low=low, close=close, window=period, annualize=annualize
    )


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

    Interpretation:
        - The baseline volatility measure. All other estimators
          (Parkinson, Garman-Klass, Yang-Zhang) should be compared
          against this.
        - See :func:`historical_volatility` for full interpretation.

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
    data = _validate_series(data)
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

    Interpretation:
        - **Low values (< 5)**: Stable asset with shallow drawdowns.
        - **High values (> 10)**: Asset is experiencing significant
          drawdowns.
        - Unlike standard deviation, only measures downside risk.
        - Used in the Martin Ratio (return / Ulcer Index) as a
          risk-adjusted performance metric.
        - Rising Ulcer Index = drawdowns are deepening = trouble.

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
    data = _validate_series(data)
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
    rather than to price changes.

    Interpretation:
        - **> 50**: Volatility is increasing (standard deviation is
          rising) -- directional moves are more likely.
        - **< 50**: Volatility is decreasing (standard deviation is
          falling) -- consolidation / range-bound.
        - Not a standalone indicator; best used as a filter.

    Trading rules:
        - Confirm RSI signals: only take RSI buy signals when RVI > 50.
        - Avoid breakout trades when RVI < 50 (low volatility = false
          breakout risk).

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
    data = _validate_series(data)
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

    Interpretation:
        - **Price above upper band**: Strong uptrend / breakout.
        - **Price below lower band**: Strong downtrend / breakdown.
        - **Bands narrowing**: Consolidation -- breakout imminent.
        - **Bands widening**: Trend accelerating.
        - Similar concept to Bollinger Bands but based on range
          acceleration rather than standard deviation.

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
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")
    _validate_period(period)

    hl_ratio = factor * (high - low) / ((high + low) / 2.0)
    upper_band = (
        (high * (1.0 + hl_ratio)).rolling(window=period, min_periods=period).mean()
    )
    lower_band = (
        (low * (1.0 - hl_ratio)).rolling(window=period, min_periods=period).mean()
    )
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

    Interpretation:
        - **Rising**: Volatility increasing -- larger price swings.
        - **Falling**: Volatility decreasing -- tighter price action.
        - Low standard deviation often precedes a breakout.
        - Used to compute Bollinger Bands (middle +/- N * std_dev).

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
    data = _validate_series(data)
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

    Interpretation:
        - The square of standard deviation. Same directional
          interpretation as standard deviation.
        - Useful in mathematical contexts where variance is preferred
          (e.g. portfolio optimization, risk decomposition).

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
    data = _validate_series(data)
    _validate_period(period)

    result = data.rolling(window=period, min_periods=period).var(ddof=1)
    result.name = "variance"
    return result
