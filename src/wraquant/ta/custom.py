"""Modern and advanced technical analysis indicators.

This module provides contemporary indicators including squeeze detection,
anchored VWAP, market structure analysis, and adaptive oscillators.
All functions accept ``pd.Series`` inputs and return ``pd.Series``
(or ``dict[str, pd.Series]`` for multi-output indicators).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "squeeze_momentum",
    "anchored_vwap",
    "linear_regression_channel",
    "pivot_points",
    "market_structure",
    "swing_points",
    "volume_weighted_macd",
    "ehlers_fisher",
    "adaptive_rsi",
    "relative_strength",
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


def _sma(data: pd.Series, period: int) -> pd.Series:
    """Internal SMA helper."""
    return data.rolling(window=period, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Squeeze Momentum (TTM Squeeze)
# ---------------------------------------------------------------------------


def squeeze_momentum(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    bb_period: int = 20,
    bb_std: float = 2.0,
    kc_period: int = 20,
    kc_mult: float = 1.5,
    mom_period: int = 12,
) -> dict[str, pd.Series]:
    """TTM Squeeze Momentum indicator.

    Detects when Bollinger Bands are inside Keltner Channels (the
    "squeeze") and measures momentum via a linear regression of price.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    bb_period : int, default 20
        Bollinger Bands SMA period.
    bb_std : float, default 2.0
        Bollinger Bands standard deviation multiplier.
    kc_period : int, default 20
        Keltner Channel EMA period.
    kc_mult : float, default 1.5
        Keltner Channel ATR multiplier.
    mom_period : int, default 12
        Momentum linear regression period.

    Returns
    -------
    dict[str, pd.Series]
        ``squeeze_on`` (bool: 1 = squeeze active), ``momentum`` (momentum
        histogram values).

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> n = 100
    >>> c = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5))
    >>> h = c + abs(np.random.randn(n) * 0.3)
    >>> lo = c - abs(np.random.randn(n) * 0.3)
    >>> result = squeeze_momentum(h, lo, c)  # doctest: +SKIP
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    # Bollinger Bands
    bb_mid = _sma(close, bb_period)
    bb_rolling_std = close.rolling(window=bb_period, min_periods=bb_period).std(ddof=0)
    bb_upper = bb_mid + bb_std * bb_rolling_std
    bb_lower = bb_mid - bb_std * bb_rolling_std

    # Keltner Channels (using ATR via true range)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_val = tr.ewm(alpha=1.0 / kc_period, min_periods=kc_period, adjust=False).mean()
    kc_mid = _ema(close, kc_period)
    kc_upper = kc_mid + kc_mult * atr_val
    kc_lower = kc_mid - kc_mult * atr_val

    # Squeeze: BB inside KC
    squeeze_on = ((bb_lower > kc_lower) & (bb_upper < kc_upper)).astype(float)

    # Momentum: linear regression value of (close - midline)
    midline = (
        high.rolling(window=kc_period, min_periods=kc_period).max()
        + low.rolling(window=kc_period, min_periods=kc_period).min()
    ) / 2
    delta = close - (midline + bb_mid) / 2

    def _linreg_value(window: np.ndarray) -> float:
        n = len(window)
        x = np.arange(n, dtype=float)
        slope = (n * np.dot(x, window) - x.sum() * window.sum()) / (
            n * np.dot(x, x) - x.sum() ** 2
        )
        intercept = (window.sum() - slope * x.sum()) / n
        return float(slope * (n - 1) + intercept)

    mom = delta.rolling(window=mom_period, min_periods=mom_period).apply(
        _linreg_value, raw=True
    )

    return {
        "squeeze_on": squeeze_on.rename("squeeze_on"),
        "momentum": mom.rename("squeeze_momentum"),
    }


# ---------------------------------------------------------------------------
# Anchored VWAP
# ---------------------------------------------------------------------------


def anchored_vwap(
    close: pd.Series,
    volume: pd.Series,
    anchor_index: int = 0,
) -> pd.Series:
    """VWAP anchored from a specific bar index.

    Computes the Volume Weighted Average Price starting from
    *anchor_index* onwards. Values before the anchor are ``NaN``.

    Parameters
    ----------
    close : pd.Series
        Close (or typical) prices.
    volume : pd.Series
        Volume series.
    anchor_index : int, default 0
        The integer position index to begin the VWAP calculation from.

    Returns
    -------
    pd.Series
        Anchored VWAP values.

    Example
    -------
    >>> import pandas as pd
    >>> close = pd.Series([10, 11, 12, 11, 10, 9, 10, 11, 12, 13], dtype=float)
    >>> volume = pd.Series([100, 200, 150, 300, 250, 100, 200, 150, 300, 250], dtype=float)
    >>> anchored_vwap(close, volume, anchor_index=3)  # doctest: +SKIP
    """
    _validate_series(close, "close")
    _validate_series(volume, "volume")

    pv = close * volume
    result = pd.Series(np.nan, index=close.index, name="anchored_vwap")

    cum_pv = pv.iloc[anchor_index:].cumsum()
    cum_vol = volume.iloc[anchor_index:].cumsum()

    result.iloc[anchor_index:] = (cum_pv / cum_vol).values

    return result


# ---------------------------------------------------------------------------
# Linear Regression Channel
# ---------------------------------------------------------------------------


def linear_regression_channel(
    data: pd.Series,
    period: int = 100,
    std_dev: float = 2.0,
) -> dict[str, pd.Series]:
    """Linear regression channel with standard deviation bands.

    Fits a rolling linear regression and constructs upper/lower
    channel lines based on the standard error.

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 100
        Rolling window length for the regression.
    std_dev : float, default 2.0
        Number of standard deviations for the channel width.

    Returns
    -------
    dict[str, pd.Series]
        ``middle`` (regression value), ``upper``, ``lower``.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> close = pd.Series(100 + np.arange(120, dtype=float) * 0.5)
    >>> result = linear_regression_channel(close, period=50)  # doctest: +SKIP
    """
    _validate_series(data)
    _validate_period(period)
    values = data.values.astype(float)
    n = len(values)

    mid = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)

    for i in range(period - 1, n):
        window = values[i - period + 1 : i + 1]
        x = np.arange(period, dtype=float)
        slope = (period * np.dot(x, window) - x.sum() * window.sum()) / (
            period * np.dot(x, x) - x.sum() ** 2
        )
        intercept = (window.sum() - slope * x.sum()) / period
        reg_val = slope * (period - 1) + intercept
        predicted = slope * x + intercept
        residuals = window - predicted
        std_err = np.std(residuals, ddof=1)

        mid[i] = reg_val
        upper[i] = reg_val + std_dev * std_err
        lower[i] = reg_val - std_dev * std_err

    return {
        "middle": pd.Series(mid, index=data.index, name="lr_middle"),
        "upper": pd.Series(upper, index=data.index, name="lr_upper"),
        "lower": pd.Series(lower, index=data.index, name="lr_lower"),
    }


# ---------------------------------------------------------------------------
# Pivot Points
# ---------------------------------------------------------------------------


def pivot_points(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    method: str = "standard",
) -> dict[str, pd.Series]:
    """Pivot points with support and resistance levels.

    Computes pivot point and two levels of support/resistance using
    the prior bar's high, low, and close.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    method : str, default "standard"
        Calculation method: ``"standard"``, ``"fibonacci"``, or ``"woodie"``.

    Returns
    -------
    dict[str, pd.Series]
        ``pivot``, ``s1``, ``s2``, ``r1``, ``r2``.

    Example
    -------
    >>> import pandas as pd
    >>> h = pd.Series([12, 13, 14, 13, 12], dtype=float)
    >>> lo = pd.Series([10, 11, 12, 11, 10], dtype=float)
    >>> c = pd.Series([11, 12, 13, 12, 11], dtype=float)
    >>> result = pivot_points(h, lo, c)  # doctest: +SKIP
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    h_prev = high.shift(1)
    l_prev = low.shift(1)
    c_prev = close.shift(1)

    if method == "standard":
        pp = (h_prev + l_prev + c_prev) / 3.0
        r1 = 2 * pp - l_prev
        s1 = 2 * pp - h_prev
        r2 = pp + (h_prev - l_prev)
        s2 = pp - (h_prev - l_prev)
    elif method == "fibonacci":
        pp = (h_prev + l_prev + c_prev) / 3.0
        diff = h_prev - l_prev
        r1 = pp + 0.382 * diff
        r2 = pp + 0.618 * diff
        s1 = pp - 0.382 * diff
        s2 = pp - 0.618 * diff
    elif method == "woodie":
        pp = (h_prev + l_prev + 2 * close) / 4.0
        r1 = 2 * pp - l_prev
        s1 = 2 * pp - h_prev
        r2 = pp + (h_prev - l_prev)
        s2 = pp - (h_prev - l_prev)
    else:
        raise ValueError(
            f"method must be 'standard', 'fibonacci', or 'woodie', got {method!r}"
        )

    return {
        "pivot": pp.rename("pivot"),
        "r1": r1.rename("r1"),
        "r2": r2.rename("r2"),
        "s1": s1.rename("s1"),
        "s2": s2.rename("s2"),
    }


# ---------------------------------------------------------------------------
# Market Structure
# ---------------------------------------------------------------------------


def market_structure(
    high: pd.Series,
    low: pd.Series,
    lookback: int = 5,
) -> dict[str, pd.Series]:
    """Higher highs / lower lows market structure detection.

    Identifies swing highs and lows using a *lookback* window, then
    labels each swing as higher-high (HH), lower-high (LH),
    higher-low (HL), or lower-low (LL).

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    lookback : int, default 5
        Number of bars on each side to confirm a swing point.

    Returns
    -------
    dict[str, pd.Series]
        ``swing_high`` (high values at swing highs, else NaN),
        ``swing_low`` (low values at swing lows, else NaN),
        ``structure`` (1 = bullish / HH+HL, -1 = bearish / LH+LL, 0 = neutral).

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> h = pd.Series(100 + np.cumsum(np.random.randn(50) * 0.5) + 1)
    >>> lo = h - 2
    >>> result = market_structure(h, lo, lookback=3)  # doctest: +SKIP
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_period(lookback, "lookback")

    n = len(high)
    h_vals = high.values.astype(float)
    l_vals = low.values.astype(float)

    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)

    # Detect swing points
    for i in range(lookback, n - lookback):
        # Swing high: highest in both left and right windows
        left_h = h_vals[i - lookback : i]
        right_h = h_vals[i + 1 : i + lookback + 1]
        if h_vals[i] >= np.max(left_h) and h_vals[i] >= np.max(right_h):
            sh[i] = h_vals[i]

        # Swing low: lowest in both left and right windows
        left_l = l_vals[i - lookback : i]
        right_l = l_vals[i + 1 : i + lookback + 1]
        if l_vals[i] <= np.min(left_l) and l_vals[i] <= np.min(right_l):
            sl[i] = l_vals[i]

    # Determine structure
    structure = np.zeros(n)
    last_sh = np.nan
    last_sl = np.nan

    for i in range(n):
        if not np.isnan(sh[i]):
            if not np.isnan(last_sh):
                if sh[i] > last_sh:
                    structure[i] = 1.0  # Higher high
                elif sh[i] < last_sh:
                    structure[i] = -1.0  # Lower high
            last_sh = sh[i]
        elif not np.isnan(sl[i]):
            if not np.isnan(last_sl):
                if sl[i] > last_sl:
                    structure[i] = 1.0  # Higher low
                elif sl[i] < last_sl:
                    structure[i] = -1.0  # Lower low
            last_sl = sl[i]

    # Forward-fill structure labels
    struct_series = pd.Series(structure, index=high.index)
    struct_series = struct_series.replace(0.0, np.nan).ffill().fillna(0.0)

    return {
        "swing_high": pd.Series(sh, index=high.index, name="swing_high"),
        "swing_low": pd.Series(sl, index=high.index, name="swing_low"),
        "structure": struct_series.rename("structure"),
    }


# ---------------------------------------------------------------------------
# Swing Points
# ---------------------------------------------------------------------------


def swing_points(
    high: pd.Series,
    low: pd.Series,
    lookback: int = 5,
) -> dict[str, pd.Series]:
    """Swing high and low detection.

    A swing high occurs when the high is the maximum of
    ``2 * lookback + 1`` bars centred on the pivot bar. Symmetrically
    for swing lows.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    lookback : int, default 5
        Number of bars on each side of the pivot.

    Returns
    -------
    dict[str, pd.Series]
        ``swing_high`` (high values at swing highs, else NaN),
        ``swing_low`` (low values at swing lows, else NaN).

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> h = pd.Series(100 + np.cumsum(np.random.randn(50) * 0.5) + 1)
    >>> lo = h - 2
    >>> result = swing_points(h, lo, lookback=3)  # doctest: +SKIP
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_period(lookback, "lookback")

    n = len(high)
    h_vals = high.values.astype(float)
    l_vals = low.values.astype(float)

    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)

    for i in range(lookback, n - lookback):
        window_h = h_vals[i - lookback : i + lookback + 1]
        if h_vals[i] == np.max(window_h):
            sh[i] = h_vals[i]

        window_l = l_vals[i - lookback : i + lookback + 1]
        if l_vals[i] == np.min(window_l):
            sl[i] = l_vals[i]

    return {
        "swing_high": pd.Series(sh, index=high.index, name="swing_high"),
        "swing_low": pd.Series(sl, index=low.index, name="swing_low"),
    }


# ---------------------------------------------------------------------------
# Volume-Weighted MACD
# ---------------------------------------------------------------------------


def volume_weighted_macd(
    close: pd.Series,
    volume: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, pd.Series]:
    """MACD weighted by volume.

    Uses volume-weighted moving averages instead of standard EMAs
    for the fast and slow lines.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume series.
    fast : int, default 12
        Fast VWMA period.
    slow : int, default 26
        Slow VWMA period.
    signal : int, default 9
        Signal EMA period.

    Returns
    -------
    dict[str, pd.Series]
        ``macd``, ``signal``, ``histogram``.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> c = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5))
    >>> v = pd.Series(np.random.randint(1000, 10000, 100), dtype=float)
    >>> result = volume_weighted_macd(c, v)  # doctest: +SKIP
    """
    _validate_series(close, "close")
    _validate_series(volume, "volume")

    def _vwma(price: pd.Series, vol: pd.Series, period: int) -> pd.Series:
        pv = price * vol
        return pv.rolling(window=period, min_periods=period).sum() / vol.rolling(
            window=period, min_periods=period
        ).sum()

    fast_vwma = _vwma(close, volume, fast)
    slow_vwma = _vwma(close, volume, slow)
    macd_line = fast_vwma - slow_vwma
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line

    return {
        "macd": macd_line.rename("vwmacd"),
        "signal": signal_line.rename("vwmacd_signal"),
        "histogram": hist.rename("vwmacd_histogram"),
    }


# ---------------------------------------------------------------------------
# Ehlers Fisher Transform
# ---------------------------------------------------------------------------


def ehlers_fisher(
    high: pd.Series,
    low: pd.Series,
    period: int = 10,
) -> dict[str, pd.Series]:
    """Ehlers Fisher Transform.

    Converts prices into a Gaussian normal distribution to create
    sharp turning points, making it easier to identify reversals.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    period : int, default 10
        Look-back period for the normalisation.

    Returns
    -------
    dict[str, pd.Series]
        ``fisher`` and ``trigger`` (one-bar lag of fisher).

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> h = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5) + 1)
    >>> lo = h - 2
    >>> result = ehlers_fisher(h, lo)  # doctest: +SKIP
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_period(period)

    hl2 = (high + low) / 2.0
    n = len(hl2)

    highest = hl2.rolling(window=period, min_periods=period).max()
    lowest = hl2.rolling(window=period, min_periods=period).min()

    # Normalise to [-1, 1]
    raw = 2.0 * (hl2 - lowest) / (highest - lowest) - 1.0
    # Clamp to avoid atanh(1) = inf
    raw = raw.clip(-0.999, 0.999)

    # Smooth
    val = np.zeros(n)
    fisher_out = np.zeros(n)
    trigger_out = np.zeros(n)

    for i in range(n):
        if np.isnan(raw.iloc[i]):
            val[i] = 0.0
            fisher_out[i] = np.nan
            trigger_out[i] = np.nan
            continue
        val[i] = 0.5 * raw.iloc[i] + 0.5 * val[i - 1] if i > 0 else 0.5 * raw.iloc[i]
        val[i] = np.clip(val[i], -0.999, 0.999)
        fisher_out[i] = (
            0.5 * np.log((1 + val[i]) / (1 - val[i]))
            + 0.5 * (fisher_out[i - 1] if i > 0 else 0.0)
        )
        trigger_out[i] = fisher_out[i - 1] if i > 0 else np.nan

    # Set pre-warmup to NaN
    fisher_out[: period - 1] = np.nan
    trigger_out[: period] = np.nan

    return {
        "fisher": pd.Series(fisher_out, index=high.index, name="fisher"),
        "trigger": pd.Series(trigger_out, index=high.index, name="fisher_trigger"),
    }


# ---------------------------------------------------------------------------
# Adaptive RSI
# ---------------------------------------------------------------------------


def adaptive_rsi(
    data: pd.Series,
    base_period: int = 14,
    vol_period: int = 10,
    min_period: int = 5,
    max_period: int = 50,
) -> pd.Series:
    """RSI with an adaptive period based on volatility.

    The look-back period expands in low-volatility regimes and
    contracts in high-volatility regimes, improving responsiveness.

    Parameters
    ----------
    data : pd.Series
        Price series (typically close).
    base_period : int, default 14
        Base RSI period.
    vol_period : int, default 10
        Period for the volatility (standard deviation) calculation.
    min_period : int, default 5
        Minimum allowed RSI period.
    max_period : int, default 50
        Maximum allowed RSI period.

    Returns
    -------
    pd.Series
        Adaptive RSI values in [0, 100].

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> close = pd.Series(100 + np.cumsum(np.random.randn(200) * 0.5))
    >>> adaptive_rsi(close)  # doctest: +SKIP
    """
    _validate_series(data)
    _validate_period(base_period)

    n = len(data)
    values = data.values.astype(float)
    returns = np.diff(values, prepend=values[0])

    # Rolling volatility
    vol = data.pct_change().rolling(window=vol_period, min_periods=vol_period).std()
    # Normalise volatility to [0, 1]
    vol_min = vol.rolling(window=max_period, min_periods=1).min()
    vol_max = vol.rolling(window=max_period, min_periods=1).max()
    vol_norm = (vol - vol_min) / (vol_max - vol_min)
    vol_norm = vol_norm.fillna(0.5)

    # Adaptive period: high vol -> short period, low vol -> long period
    adaptive_period = (
        max_period - vol_norm * (max_period - min_period)
    ).clip(min_period, max_period)

    result = np.full(n, np.nan)
    delta = data.diff()
    gain = delta.clip(lower=0.0).values
    loss = (-delta).clip(lower=0.0).values

    for i in range(max_period, n):
        period = int(round(adaptive_period.iloc[i]))
        window_gain = gain[i - period + 1 : i + 1]
        window_loss = loss[i - period + 1 : i + 1]
        avg_gain = np.mean(window_gain)
        avg_loss = np.mean(window_loss)
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - 100.0 / (1.0 + rs)

    out = pd.Series(result, index=data.index, name="adaptive_rsi")
    return out


# ---------------------------------------------------------------------------
# Relative Strength (Ratio)
# ---------------------------------------------------------------------------


def relative_strength(
    data: pd.Series,
    benchmark: pd.Series,
) -> pd.Series:
    """Relative strength ratio of one series to another.

    Commonly used for pair analysis or sector rotation. A rising
    ratio indicates *data* is outperforming *benchmark*.

    Parameters
    ----------
    data : pd.Series
        Numerator price series (e.g., individual stock).
    benchmark : pd.Series
        Denominator price series (e.g., index or sector ETF).

    Returns
    -------
    pd.Series
        Ratio of data / benchmark.

    Example
    -------
    >>> import pandas as pd
    >>> stock = pd.Series([100, 105, 110, 108, 112], dtype=float)
    >>> index = pd.Series([1000, 1010, 1005, 1015, 1020], dtype=float)
    >>> relative_strength(stock, index)  # doctest: +SKIP
    """
    _validate_series(data, "data")
    _validate_series(benchmark, "benchmark")

    result = data / benchmark
    result.name = "relative_strength"
    return result
