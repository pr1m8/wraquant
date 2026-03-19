"""Candlestick analytics (structural metrics, not pattern recognition).

Functions in this module quantify the *shape* of individual candlesticks —
body size, shadow ratios, gaps, and structural bar classifications such as
inside bars, outside bars, and pin bars.  All functions accept ``pd.Series``
inputs and return ``pd.Series``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "candle_body_size",
    "candle_range",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "body_to_range_ratio",
    "candle_direction",
    "average_candle_body",
    "candle_momentum",
    "body_gap",
    "inside_bar",
    "outside_bar",
    "pin_bar",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


from wraquant.ta._validators import validate_series as _validate_series


def _body(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Absolute body size."""
    return (close - open_).abs()


def _range(high: pd.Series, low: pd.Series) -> pd.Series:
    """Full candle range (high - low)."""
    return high - low


def _upper_shadow(open_: pd.Series, high: pd.Series, close: pd.Series) -> pd.Series:
    return high - pd.concat([open_, close], axis=1).max(axis=1)


def _lower_shadow(open_: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return pd.concat([open_, close], axis=1).min(axis=1) - low


# ---------------------------------------------------------------------------
# candle_body_size
# ---------------------------------------------------------------------------


def candle_body_size(
    open_: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Absolute candle body size.

    Computed as ``abs(close - open)``.

    Parameters:
        open_: Open prices.
        close: Close prices.

    Returns:
        Absolute body size for each bar.

    Example:
        >>> body = candle_body_size(open_, close)
    """
    open_ = _validate_series(open_, "open_")
    close = _validate_series(close, "close")

    result = _body(open_, close)
    result.name = "candle_body_size"
    return result


# ---------------------------------------------------------------------------
# candle_range
# ---------------------------------------------------------------------------


def candle_range(
    high: pd.Series,
    low: pd.Series,
) -> pd.Series:
    """Full candle range (high minus low).

    Parameters:
        high: High prices.
        low: Low prices.

    Returns:
        Range for each bar.

    Example:
        >>> rng = candle_range(high, low)
    """
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")

    result = _range(high, low)
    result.name = "candle_range"
    return result


# ---------------------------------------------------------------------------
# upper_shadow_ratio
# ---------------------------------------------------------------------------


def upper_shadow_ratio(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Upper shadow as a fraction of the total candle range.

    ``upper_shadow / (high - low)``

    Parameters:
        open_: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.

    Returns:
        Upper shadow ratio in [0, 1].  Returns 0 when the range is zero.

    Example:
        >>> ratio = upper_shadow_ratio(open_, high, low, close)
    """
    open_ = _validate_series(open_, "open_")
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")

    upper = _upper_shadow(open_, high, close)
    rng = _range(high, low)
    result = pd.Series(
        np.where(rng != 0, upper / rng, 0.0),
        index=close.index,
        name="upper_shadow_ratio",
    )
    return result


# ---------------------------------------------------------------------------
# lower_shadow_ratio
# ---------------------------------------------------------------------------


def lower_shadow_ratio(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Lower shadow as a fraction of the total candle range.

    ``lower_shadow / (high - low)``

    Parameters:
        open_: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.

    Returns:
        Lower shadow ratio in [0, 1].  Returns 0 when the range is zero.

    Example:
        >>> ratio = lower_shadow_ratio(open_, high, low, close)
    """
    open_ = _validate_series(open_, "open_")
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")

    lower = _lower_shadow(open_, low, close)
    rng = _range(high, low)
    result = pd.Series(
        np.where(rng != 0, lower / rng, 0.0),
        index=close.index,
        name="lower_shadow_ratio",
    )
    return result


# ---------------------------------------------------------------------------
# body_to_range_ratio
# ---------------------------------------------------------------------------


def body_to_range_ratio(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Body size as a fraction of the total candle range.

    ``abs(close - open) / (high - low)``

    Parameters:
        open_: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.

    Returns:
        Body-to-range ratio in [0, 1].  Returns 0 when the range is zero.

    Example:
        >>> ratio = body_to_range_ratio(open_, high, low, close)
    """
    open_ = _validate_series(open_, "open_")
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")

    body = _body(open_, close)
    rng = _range(high, low)
    result = pd.Series(
        np.where(rng != 0, body / rng, 0.0),
        index=close.index,
        name="body_to_range_ratio",
    )
    return result


# ---------------------------------------------------------------------------
# candle_direction
# ---------------------------------------------------------------------------


def candle_direction(
    open_: pd.Series,
    close: pd.Series,
    doji_threshold: float = 0.0,
) -> pd.Series:
    """Candle direction: 1 (bullish), -1 (bearish), 0 (doji).

    Parameters:
        open_: Open prices.
        close: Close prices.
        doji_threshold: Maximum absolute difference between close and open
            to classify the candle as a doji (default 0.0 — exact equality).

    Returns:
        Direction for each bar.

    Example:
        >>> direction = candle_direction(open_, close)
    """
    open_ = _validate_series(open_, "open_")
    close = _validate_series(close, "close")

    diff = close - open_
    result = np.where(
        diff > doji_threshold,
        1,
        np.where(diff < -doji_threshold, -1, 0),
    )
    return pd.Series(result, index=close.index, name="candle_direction", dtype=int)


# ---------------------------------------------------------------------------
# average_candle_body
# ---------------------------------------------------------------------------


def average_candle_body(
    open_: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Simple moving average of absolute body sizes.

    Parameters:
        open_: Open prices.
        close: Close prices.
        period: SMA look-back period.

    Returns:
        Smoothed average body size.

    Example:
        >>> avg_body = average_candle_body(open_, close, period=14)
    """
    open_ = _validate_series(open_, "open_")
    close = _validate_series(close, "close")
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    body = _body(open_, close)
    result = body.rolling(window=period, min_periods=period).mean()
    result.name = "average_candle_body"
    return result


# ---------------------------------------------------------------------------
# candle_momentum
# ---------------------------------------------------------------------------


def candle_momentum(
    open_: pd.Series,
    close: pd.Series,
    period: int = 5,
) -> pd.Series:
    """Sum of (close - open) over the last *period* bars.

    A positive value indicates net bullish momentum; negative indicates
    bearish momentum.

    Parameters:
        open_: Open prices.
        close: Close prices.
        period: Number of bars to sum.

    Returns:
        Cumulative close-minus-open over the window.

    Example:
        >>> mom = candle_momentum(open_, close, period=5)
    """
    open_ = _validate_series(open_, "open_")
    close = _validate_series(close, "close")
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    diff = close - open_
    result = diff.rolling(window=period, min_periods=period).sum()
    result.name = "candle_momentum"
    return result


# ---------------------------------------------------------------------------
# body_gap
# ---------------------------------------------------------------------------


def body_gap(
    open_: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Gap between consecutive candle bodies.

    Computed as ``open[i] - close[i-1]``, i.e. the distance between the
    current open and the previous close.

    Parameters:
        open_: Open prices.
        close: Close prices.

    Returns:
        Body gap for each bar (positive = gap up, negative = gap down).

    Example:
        >>> gap = body_gap(open_, close)
    """
    open_ = _validate_series(open_, "open_")
    close = _validate_series(close, "close")

    result = open_ - close.shift(1)
    result.name = "body_gap"
    return result


# ---------------------------------------------------------------------------
# inside_bar
# ---------------------------------------------------------------------------


def inside_bar(
    high: pd.Series,
    low: pd.Series,
) -> pd.Series:
    """Detect inside bars.

    An inside bar has a high below the previous high **and** a low above
    the previous low (the bar is fully contained within the prior bar's
    range).

    Parameters:
        high: High prices.
        low: Low prices.

    Returns:
        Boolean series (True where an inside bar is detected).

    Example:
        >>> ib = inside_bar(high, low)
    """
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")

    result = (high < high.shift(1)) & (low > low.shift(1))
    return pd.Series(result, index=high.index, name="inside_bar", dtype=bool)


# ---------------------------------------------------------------------------
# outside_bar
# ---------------------------------------------------------------------------


def outside_bar(
    high: pd.Series,
    low: pd.Series,
) -> pd.Series:
    """Detect outside bars (engulfing range).

    An outside bar has a high above the previous high **and** a low below
    the previous low (the bar's range completely engulfs the prior bar's
    range).

    Parameters:
        high: High prices.
        low: Low prices.

    Returns:
        Boolean series (True where an outside bar is detected).

    Example:
        >>> ob = outside_bar(high, low)
    """
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")

    result = (high > high.shift(1)) & (low < low.shift(1))
    return pd.Series(result, index=high.index, name="outside_bar", dtype=bool)


# ---------------------------------------------------------------------------
# pin_bar
# ---------------------------------------------------------------------------


def pin_bar(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    shadow_ratio: float = 2.0,
) -> pd.Series:
    """Detect pin bars.

    A pin bar has a long shadow that is at least *shadow_ratio* times the
    body size.  A bullish pin bar (1) has a long lower shadow; a bearish
    pin bar (-1) has a long upper shadow.

    Parameters:
        open_: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.
        shadow_ratio: Minimum shadow-to-body ratio to qualify (default 2.0).

    Returns:
        1 (bullish pin bar), -1 (bearish pin bar), or 0.

    Example:
        >>> pb = pin_bar(open_, high, low, close)
    """
    open_ = _validate_series(open_, "open_")
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")

    body = _body(open_, close)
    upper = _upper_shadow(open_, high, close)
    lower = _lower_shadow(open_, low, close)
    rng = _range(high, low)

    # Avoid division by zero: only consider bars with non-zero body
    safe_body = body.replace(0, np.nan)

    bullish = (lower / safe_body >= shadow_ratio) & (upper < lower) & (rng > 0)
    bearish = (upper / safe_body >= shadow_ratio) & (lower < upper) & (rng > 0)

    result = np.where(bullish, 1, np.where(bearish, -1, 0))
    return pd.Series(result, index=close.index, name="pin_bar", dtype=int)
