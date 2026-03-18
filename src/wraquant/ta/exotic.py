"""Lesser-known and exotic technical analysis indicators.

This module provides uncommon or specialized indicators that measure
various aspects of market behaviour. All functions accept ``pd.Series``
inputs and return ``pd.Series`` (or ``dict[str, pd.Series]`` for
multi-output indicators).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "choppiness_index",
    "random_walk_index",
    "polarized_fractal_efficiency",
    "price_zone_oscillator",
    "ergodic_oscillator",
    "elder_thermometer",
    "market_facilitation_index",
    "efficiency_ratio",
    "trend_intensity_index",
    "directional_movement_index",
    "kairi",
    "gopalakrishnan_range",
    "pretty_good_oscillator",
    "connors_tps",
    "relative_momentum_index",
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


def _ema(data: pd.Series, period: int) -> pd.Series:
    """Internal EMA helper to avoid circular import."""
    return data.ewm(span=period, adjust=False, min_periods=period).mean()


def _sma(data: pd.Series, period: int) -> pd.Series:
    """Internal SMA helper to avoid circular import."""
    return data.rolling(window=period, min_periods=period).mean()


def _true_range(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Internal True Range helper."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def _atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> pd.Series:
    """Internal ATR helper (SMA of True Range)."""
    tr = _true_range(high, low, close)
    return tr.rolling(window=period, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Choppiness Index
# ---------------------------------------------------------------------------


def choppiness_index(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Choppiness Index.

    Measures whether the market is trending or range-bound. Values near
    100 indicate a choppy, consolidating market; values near 0 indicate
    a strong trend.

    ``CI = 100 * log10(sum(ATR(1), n) / (highest_high - lowest_low)) / log10(n)``

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
        Choppiness Index values, typically in [0, 100].

    Example
    -------
    >>> result = choppiness_index(high, low, close, period=14)
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")
    _validate_period(period)

    tr = _true_range(high, low, close)
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    highest = high.rolling(window=period, min_periods=period).max()
    lowest = low.rolling(window=period, min_periods=period).min()

    hl_range = highest - lowest
    hl_range = hl_range.replace(0.0, np.nan)

    result = 100.0 * np.log10(atr_sum / hl_range) / np.log10(period)
    result.name = "choppiness_index"
    return result


# ---------------------------------------------------------------------------
# Random Walk Index
# ---------------------------------------------------------------------------


def random_walk_index(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> dict[str, pd.Series]:
    """Random Walk Index (RWI).

    Compares the range of directional price moves to the expected range
    of a random walk. Values above 1.0 suggest trending behavior.

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
        ``rwi_high`` and ``rwi_low``.

    Example
    -------
    >>> result = random_walk_index(high, low, close, period=14)
    >>> result["rwi_high"]
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")
    _validate_period(period)

    tr = _true_range(high, low, close)
    atr_val = tr.rolling(window=period, min_periods=period).mean()

    h = high.values.astype(float)
    l = low.values.astype(float)
    atr_arr = atr_val.values.astype(float)
    n = len(h)

    rwi_high = np.full(n, np.nan)
    rwi_low = np.full(n, np.nan)

    for i in range(period, n):
        max_rwi_h = 0.0
        max_rwi_l = 0.0
        for j in range(1, period + 1):
            denom = atr_arr[i] * np.sqrt(j)
            if denom > 0 and not np.isnan(denom):
                rwi_h = (h[i] - l[i - j]) / denom
                rwi_l = (h[i - j] - l[i]) / denom
                max_rwi_h = max(max_rwi_h, rwi_h)
                max_rwi_l = max(max_rwi_l, rwi_l)
        rwi_high[i] = max_rwi_h
        rwi_low[i] = max_rwi_l

    idx = high.index
    return {
        "rwi_high": pd.Series(rwi_high, index=idx, name="rwi_high"),
        "rwi_low": pd.Series(rwi_low, index=idx, name="rwi_low"),
    }


# ---------------------------------------------------------------------------
# Polarized Fractal Efficiency
# ---------------------------------------------------------------------------


def polarized_fractal_efficiency(
    close: pd.Series,
    period: int = 10,
    smoothing: int = 5,
) -> pd.Series:
    """Polarized Fractal Efficiency (PFE).

    Measures the efficiency of the price path using fractal geometry.
    A straight-line move yields +/- 100; a random walk yields ~0.

    ``PFE = sign(direction) * sqrt(sum_sq_diff + direction^2) / sum_single_diff * 100``

    The raw PFE is then smoothed with an EMA.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    period : int, default 10
        Look-back period.
    smoothing : int, default 5
        EMA smoothing period applied to the raw PFE.

    Returns
    -------
    pd.Series
        PFE values, bounded roughly in [-100, 100].

    Example
    -------
    >>> result = polarized_fractal_efficiency(close, period=10)
    """
    _validate_series(close, "close")
    _validate_period(period)

    values = close.values.astype(float)
    n = len(values)
    pfe_raw = np.full(n, np.nan)

    for i in range(period, n):
        # Total direction (end-to-end)
        direction = values[i] - values[i - period]

        # Sum of individual bar-to-bar distances (Euclidean with unit x-step)
        path_length = 0.0
        for j in range(i - period + 1, i + 1):
            diff = values[j] - values[j - 1]
            path_length += np.sqrt(1.0 + diff * diff)

        # Fractal dimension of the path
        fractal_length = np.sqrt(period * period + direction * direction)

        if path_length > 0:
            sign = 1.0 if direction > 0 else -1.0
            pfe_raw[i] = sign * (fractal_length / path_length) * 100.0

    raw_series = pd.Series(pfe_raw, index=close.index)
    result = _ema(raw_series, smoothing)
    result.name = "pfe"
    return result


# ---------------------------------------------------------------------------
# Price Zone Oscillator
# ---------------------------------------------------------------------------


def price_zone_oscillator(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Price Zone Oscillator (PZO).

    An EMA-based oscillator that classifies price action into zones.
    A positive close change gets +close, negative gets -close.

    ``PZO = 100 * EMA(signed_close, period) / EMA(close, period)``

    Parameters
    ----------
    close : pd.Series
        Close prices.
    period : int, default 14
        EMA period.

    Returns
    -------
    pd.Series
        PZO values, typically in [-100, 100].

    Example
    -------
    >>> result = price_zone_oscillator(close, period=14)
    """
    _validate_series(close, "close")
    _validate_period(period)

    diff = close.diff()
    sign = diff.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
    signed_close = close * sign

    ema_signed = _ema(signed_close, period)
    ema_close = _ema(close, period)

    result = 100.0 * ema_signed / ema_close.replace(0.0, np.nan)
    result.name = "pzo"
    return result


# ---------------------------------------------------------------------------
# Ergodic Oscillator
# ---------------------------------------------------------------------------


def ergodic_oscillator(
    close: pd.Series,
    fast: int = 5,
    slow: int = 20,
    signal: int = 5,
) -> dict[str, pd.Series]:
    """Ergodic Oscillator.

    A True Strength Index variant that produces a histogram (oscillator
    minus signal line) for identifying momentum shifts.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    fast : int, default 5
        Fast double-smoothing EMA period.
    slow : int, default 20
        Slow double-smoothing EMA period.
    signal : int, default 5
        Signal line EMA period.

    Returns
    -------
    dict[str, pd.Series]
        ``ergodic`` (the TSI line), ``signal``, and ``histogram``.

    Example
    -------
    >>> result = ergodic_oscillator(close)
    >>> result["ergodic"]
    """
    _validate_series(close, "close")
    _validate_period(fast, "fast")
    _validate_period(slow, "slow")
    _validate_period(signal, "signal")

    diff = close.diff()
    double_smoothed = _ema(_ema(diff, slow), fast)
    double_smoothed_abs = _ema(_ema(diff.abs(), slow), fast)

    ergodic_line = 100.0 * double_smoothed / double_smoothed_abs.replace(0.0, np.nan)
    signal_line = _ema(ergodic_line, signal)
    histogram = ergodic_line - signal_line

    return {
        "ergodic": ergodic_line.rename("ergodic"),
        "signal": signal_line.rename("ergodic_signal"),
        "histogram": histogram.rename("ergodic_histogram"),
    }


# ---------------------------------------------------------------------------
# Elder Thermometer
# ---------------------------------------------------------------------------


def elder_thermometer(
    high: pd.Series,
    low: pd.Series,
    period: int = 22,
) -> pd.Series:
    """Elder Thermometer.

    Measures the distance of the current bar from the previous bar's
    range, indicating how far price has moved beyond the prior bar.

    ``Thermo = max(high - prev_high, prev_low - low, 0)``

    The result is smoothed with an EMA.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    period : int, default 22
        EMA smoothing period.

    Returns
    -------
    pd.Series
        Elder Thermometer values (non-negative).

    Example
    -------
    >>> result = elder_thermometer(high, low, period=22)
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_period(period)

    up_move = (high - high.shift(1)).clip(lower=0.0)
    down_move = (low.shift(1) - low).clip(lower=0.0)

    thermo_raw = pd.concat([up_move, down_move], axis=1).max(axis=1)
    result = _ema(thermo_raw, period)
    result.name = "elder_thermometer"
    return result


# ---------------------------------------------------------------------------
# Market Facilitation Index
# ---------------------------------------------------------------------------


def market_facilitation_index(
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Market Facilitation Index (MFI / BW MFI).

    Also known as the Bill Williams Market Facilitation Index. Measures
    the efficiency of price movement per unit of volume.

    ``MFI = (high - low) / volume``

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    volume : pd.Series
        Volume data.

    Returns
    -------
    pd.Series
        Market Facilitation Index values.

    Example
    -------
    >>> result = market_facilitation_index(high, low, volume)
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(volume, "volume")

    result = (high - low) / volume.replace(0.0, np.nan)
    result.name = "mfi_bw"
    return result


# ---------------------------------------------------------------------------
# Efficiency Ratio
# ---------------------------------------------------------------------------


def efficiency_ratio(data: pd.Series, period: int = 10) -> pd.Series:
    """Efficiency Ratio (ER).

    Measures the efficiency of price movement as the ratio of net
    directional change to the total path traveled.

    ``ER = |close - close[n]| / sum(|close - close[1]|, n)``

    Values near 1.0 indicate a strong trend; values near 0.0 indicate
    choppy, range-bound price action.

    Parameters
    ----------
    data : pd.Series
        Price series (typically close).
    period : int, default 10
        Look-back period.

    Returns
    -------
    pd.Series
        Efficiency ratio values in [0, 1].

    Example
    -------
    >>> result = efficiency_ratio(close, period=10)
    """
    _validate_series(data)
    _validate_period(period)

    direction = (data - data.shift(period)).abs()
    volatility = data.diff().abs().rolling(window=period, min_periods=period).sum()

    result = direction / volatility.replace(0.0, np.nan)
    result.name = "efficiency_ratio"
    return result


# ---------------------------------------------------------------------------
# Trend Intensity Index
# ---------------------------------------------------------------------------


def trend_intensity_index(
    data: pd.Series,
    period: int = 30,
) -> pd.Series:
    """Trend Intensity Index (TII).

    Measures the percentage of closes above or below the SMA over the
    look-back period.

    ``TII = 100 * (count_above - count_below) / period``

    Parameters
    ----------
    data : pd.Series
        Price series (typically close).
    period : int, default 30
        Look-back period.

    Returns
    -------
    pd.Series
        TII values in [-100, 100]. Positive indicates bullish bias;
        negative indicates bearish bias.

    Example
    -------
    >>> result = trend_intensity_index(close, period=30)
    """
    _validate_series(data)
    _validate_period(period)

    sma_val = _sma(data, period)
    deviation = data - sma_val

    def _tii(window: np.ndarray) -> float:
        above = np.sum(window > 0)
        below = np.sum(window < 0)
        return 100.0 * (above - below) / len(window)

    result = deviation.rolling(window=period, min_periods=period).apply(_tii, raw=True)
    result.name = "tii"
    return result


# ---------------------------------------------------------------------------
# Directional Movement Index (simplified)
# ---------------------------------------------------------------------------


def directional_movement_index(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> dict[str, pd.Series]:
    """Simplified Directional Movement Index (DMI).

    Computes +DI and -DI without the full ADX smoothing, providing raw
    directional movement readings.

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
        ``plus_di`` (+DI) and ``minus_di`` (-DI), both in [0, 100].

    Example
    -------
    >>> result = directional_movement_index(high, low, close, period=14)
    >>> result["plus_di"]
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")
    _validate_period(period)

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

    tr = _true_range(high, low, close)
    atr_val = tr.rolling(window=period, min_periods=period).sum()

    plus_dm_sum = plus_dm.rolling(window=period, min_periods=period).sum()
    minus_dm_sum = minus_dm.rolling(window=period, min_periods=period).sum()

    plus_di = 100.0 * plus_dm_sum / atr_val.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm_sum / atr_val.replace(0.0, np.nan)

    return {
        "plus_di": plus_di.rename("plus_di"),
        "minus_di": minus_di.rename("minus_di"),
    }


# ---------------------------------------------------------------------------
# KAIRI — % deviation from SMA
# ---------------------------------------------------------------------------


def kairi(data: pd.Series, period: int = 14) -> pd.Series:
    """KAIRI — percentage deviation from the SMA.

    ``KAIRI = ((close - SMA(close, period)) / SMA(close, period)) * 100``

    Parameters
    ----------
    data : pd.Series
        Price series (typically close).
    period : int, default 14
        SMA period.

    Returns
    -------
    pd.Series
        KAIRI values (percentage, unbounded).

    Example
    -------
    >>> result = kairi(close, period=14)
    """
    _validate_series(data)
    _validate_period(period)

    sma_val = _sma(data, period)
    result = ((data - sma_val) / sma_val.replace(0.0, np.nan)) * 100.0
    result.name = "kairi"
    return result


# ---------------------------------------------------------------------------
# Gopalakrishnan Range Index
# ---------------------------------------------------------------------------


def gopalakrishnan_range(
    high: pd.Series,
    low: pd.Series,
    period: int = 5,
) -> pd.Series:
    """Gopalakrishnan Range Index (GAPO).

    ``GAPO = log(max_high - min_low) / log(period)``

    Measures the log of the high-low range relative to the log of the
    look-back period. Higher values indicate larger ranges.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    period : int, default 5
        Look-back period.

    Returns
    -------
    pd.Series
        GAPO values (non-negative).

    Example
    -------
    >>> result = gopalakrishnan_range(high, low, period=5)
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_period(period)

    max_high = high.rolling(window=period, min_periods=period).max()
    min_low = low.rolling(window=period, min_periods=period).min()

    hl_range = (max_high - min_low).clip(lower=1e-10)
    result = np.log(hl_range) / np.log(period)
    result.name = "gapo"
    return result


# ---------------------------------------------------------------------------
# Pretty Good Oscillator
# ---------------------------------------------------------------------------


def pretty_good_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Pretty Good Oscillator (PGO).

    ``PGO = (close - SMA(close, period)) / ATR(period)``

    Normalizes the deviation from the SMA by the ATR, producing a
    volatility-adjusted momentum reading.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    period : int, default 14
        SMA / ATR period.

    Returns
    -------
    pd.Series
        PGO values (unbounded).

    Example
    -------
    >>> result = pretty_good_oscillator(high, low, close, period=14)
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")
    _validate_period(period)

    sma_val = _sma(close, period)
    atr_val = _atr(high, low, close, period)

    result = (close - sma_val) / atr_val.replace(0.0, np.nan)
    result.name = "pgo"
    return result


# ---------------------------------------------------------------------------
# Connors TPS (Trend/Percentile/Streak)
# ---------------------------------------------------------------------------


def connors_tps(data: pd.Series, period: int = 2) -> pd.Series:
    """ConnorsRSI TPS component — cumulative streak RSI.

    Computes an up/down streak series, then applies RSI to the streak
    values. This isolates the streak component of ConnorsRSI.

    Parameters
    ----------
    data : pd.Series
        Price series (typically close).
    period : int, default 2
        RSI period applied to the streak series.

    Returns
    -------
    pd.Series
        Streak RSI values in [0, 100].

    Example
    -------
    >>> result = connors_tps(close, period=2)
    """
    _validate_series(data)
    _validate_period(period)

    diff = data.diff()
    values = diff.values.astype(float)
    n = len(values)
    streak = np.zeros(n)

    for i in range(1, n):
        if np.isnan(values[i]):
            streak[i] = 0.0
        elif values[i] > 0:
            streak[i] = max(streak[i - 1], 0) + 1
        elif values[i] < 0:
            streak[i] = min(streak[i - 1], 0) - 1
        else:
            streak[i] = 0.0

    streak_series = pd.Series(streak, index=data.index)

    # Apply RSI to the streak
    delta = streak_series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100.0 - (100.0 / (1.0 + rs))
    result.name = "connors_tps"
    return result


# ---------------------------------------------------------------------------
# Relative Momentum Index
# ---------------------------------------------------------------------------


def relative_momentum_index(
    data: pd.Series,
    period: int = 14,
    momentum_period: int = 4,
) -> pd.Series:
    """Relative Momentum Index (RMI).

    RSI applied to momentum (``close - close[momentum_period]``) instead
    of the simple one-bar change. This produces a smoother oscillator
    that reacts to the direction and magnitude of price swings over
    *momentum_period* bars.

    Parameters
    ----------
    data : pd.Series
        Price series (typically close).
    period : int, default 14
        RSI smoothing period.
    momentum_period : int, default 4
        Look-back for the momentum (difference) calculation.

    Returns
    -------
    pd.Series
        RMI values in [0, 100].

    Example
    -------
    >>> result = relative_momentum_index(close, period=14, momentum_period=4)
    """
    _validate_series(data)
    _validate_period(period)
    _validate_period(momentum_period, "momentum_period")

    delta = data - data.shift(momentum_period)
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100.0 - (100.0 / (1.0 + rs))
    result.name = "rmi"
    return result
