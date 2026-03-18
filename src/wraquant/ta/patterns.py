"""Candlestick pattern recognition.

Each function returns a ``pd.Series`` of integers:

- **1** — bullish signal
- **-1** — bearish signal
- **0** — no pattern detected

Some patterns are inherently one-directional (e.g. Morning Star is always
bullish), but where a pattern has both bullish and bearish variants the
sign distinguishes them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "doji",
    "hammer",
    "engulfing",
    "morning_star",
    "evening_star",
    "three_white_soldiers",
    "three_black_crows",
    "harami",
    "spinning_top",
    "marubozu",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_series(data: pd.Series, name: str = "data") -> pd.Series:
    if not isinstance(data, pd.Series):
        raise TypeError(f"{name} must be a pd.Series, got {type(data).__name__}")
    return data


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
# Doji
# ---------------------------------------------------------------------------


def doji(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    threshold: float = 0.05,
) -> pd.Series:
    """Doji pattern.

    A Doji occurs when the body is very small relative to the total range.

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
    threshold : float, default 0.05
        Maximum body-to-range ratio to qualify as a Doji.

    Returns
    -------
    pd.Series
        1 where a Doji is detected, 0 otherwise.
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    body = _body(open_, close)
    rng = _range(high, low)

    ratio = pd.Series(
        np.where(rng != 0, body / rng, 0.0),
        index=close.index,
    )
    result = pd.Series(
        np.where(ratio <= threshold, 1, 0),
        index=close.index,
        name="doji",
        dtype=int,
    )
    return result


# ---------------------------------------------------------------------------
# Hammer / Hanging Man
# ---------------------------------------------------------------------------


def hammer(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Hammer and Hanging Man pattern.

    A hammer has a small body near the top and a long lower shadow (at
    least 2x the body). Returns 1 for bullish hammer (after a downtrend
    proxy: prior close < close), -1 for hanging man (after an uptrend
    proxy: prior close > close), 0 otherwise.

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

    Returns
    -------
    pd.Series
        1 (bullish hammer), -1 (hanging man), or 0.
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    body = _body(open_, close)
    lower = _lower_shadow(open_, low, close)
    upper = _upper_shadow(open_, high, close)
    rng = _range(high, low)

    # Conditions: lower shadow >= 2*body, upper shadow small, body exists
    is_hammer_shape = (lower >= 2 * body) & (upper <= body * 0.5) & (rng > 0)

    prev_close = close.shift(1)
    direction = np.where(
        is_hammer_shape & (prev_close > close),
        1,  # bullish (prior downtrend)
        np.where(is_hammer_shape & (prev_close < close), -1, 0),  # bearish
    )

    # Default: if shape matches but no clear prior trend, mark bullish
    direction = np.where(is_hammer_shape & (direction == 0), 1, direction)

    return pd.Series(direction, index=close.index, name="hammer", dtype=int)


# ---------------------------------------------------------------------------
# Engulfing
# ---------------------------------------------------------------------------


def engulfing(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Bullish/Bearish Engulfing pattern.

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

    Returns
    -------
    pd.Series
        1 (bullish engulfing), -1 (bearish engulfing), or 0.
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    prev_open = open_.shift(1)
    prev_close = close.shift(1)

    # Current candle body must engulf previous candle body
    curr_bullish = close > open_
    curr_bearish = close < open_
    prev_bearish = prev_close < prev_open
    prev_bullish = prev_close > prev_open

    bullish_engulf = (
        curr_bullish & prev_bearish & (open_ <= prev_close) & (close >= prev_open)
    )

    bearish_engulf = (
        curr_bearish & prev_bullish & (open_ >= prev_close) & (close <= prev_open)
    )

    result = np.where(bullish_engulf, 1, np.where(bearish_engulf, -1, 0))
    return pd.Series(result, index=close.index, name="engulfing", dtype=int)


# ---------------------------------------------------------------------------
# Morning Star
# ---------------------------------------------------------------------------


def morning_star(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Morning Star (3-candle bullish reversal).

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

    Returns
    -------
    pd.Series
        1 where a Morning Star is detected, 0 otherwise.
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    body = _body(open_, close)
    rng = _range(high, low)

    # Day 1: large bearish candle
    d1_bearish = (open_.shift(2) > close.shift(2)) & (
        body.shift(2) > rng.shift(2) * 0.5
    )
    # Day 2: small body (star) — gap down optional
    d2_small = body.shift(1) < rng.shift(1) * 0.3
    # Day 3: large bullish candle closing above midpoint of Day 1 body
    d3_bullish = close > open_
    midpoint_d1 = (open_.shift(2) + close.shift(2)) / 2.0
    d3_above_mid = close > midpoint_d1

    signal = d1_bearish & d2_small & d3_bullish & d3_above_mid
    return pd.Series(signal.astype(int), index=close.index, name="morning_star")


# ---------------------------------------------------------------------------
# Evening Star
# ---------------------------------------------------------------------------


def evening_star(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Evening Star (3-candle bearish reversal).

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

    Returns
    -------
    pd.Series
        -1 where an Evening Star is detected, 0 otherwise.
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    body = _body(open_, close)
    rng = _range(high, low)

    # Day 1: large bullish candle
    d1_bullish = (close.shift(2) > open_.shift(2)) & (
        body.shift(2) > rng.shift(2) * 0.5
    )
    # Day 2: small body (star)
    d2_small = body.shift(1) < rng.shift(1) * 0.3
    # Day 3: large bearish candle closing below midpoint of Day 1 body
    d3_bearish = close < open_
    midpoint_d1 = (open_.shift(2) + close.shift(2)) / 2.0
    d3_below_mid = close < midpoint_d1

    signal = d1_bullish & d2_small & d3_bearish & d3_below_mid
    result = np.where(signal, -1, 0)
    return pd.Series(result, index=close.index, name="evening_star", dtype=int)


# ---------------------------------------------------------------------------
# Three White Soldiers
# ---------------------------------------------------------------------------


def three_white_soldiers(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Three White Soldiers (strong bullish continuation).

    Three consecutive bullish candles, each opening within the prior body
    and closing at a new high.

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

    Returns
    -------
    pd.Series
        1 where the pattern is detected, 0 otherwise.
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    # All three days bullish
    d1_bull = close.shift(2) > open_.shift(2)
    d2_bull = close.shift(1) > open_.shift(1)
    d3_bull = close > open_

    # Each opens within previous body
    d2_opens_in_d1 = (open_.shift(1) >= open_.shift(2)) & (
        open_.shift(1) <= close.shift(2)
    )
    d3_opens_in_d2 = (open_ >= open_.shift(1)) & (open_ <= close.shift(1))

    # Each closes higher
    higher_closes = (close.shift(1) > close.shift(2)) & (close > close.shift(1))

    signal = (
        d1_bull & d2_bull & d3_bull & d2_opens_in_d1 & d3_opens_in_d2 & higher_closes
    )
    return pd.Series(signal.astype(int), index=close.index, name="three_white_soldiers")


# ---------------------------------------------------------------------------
# Three Black Crows
# ---------------------------------------------------------------------------


def three_black_crows(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Three Black Crows (strong bearish continuation).

    Three consecutive bearish candles, each opening within the prior body
    and closing at a new low.

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

    Returns
    -------
    pd.Series
        -1 where the pattern is detected, 0 otherwise.
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    # All three days bearish
    d1_bear = close.shift(2) < open_.shift(2)
    d2_bear = close.shift(1) < open_.shift(1)
    d3_bear = close < open_

    # Each opens within previous body
    d2_opens_in_d1 = (open_.shift(1) <= open_.shift(2)) & (
        open_.shift(1) >= close.shift(2)
    )
    d3_opens_in_d2 = (open_ <= open_.shift(1)) & (open_ >= close.shift(1))

    # Each closes lower
    lower_closes = (close.shift(1) < close.shift(2)) & (close < close.shift(1))

    signal = (
        d1_bear & d2_bear & d3_bear & d2_opens_in_d1 & d3_opens_in_d2 & lower_closes
    )
    result = np.where(signal, -1, 0)
    return pd.Series(result, index=close.index, name="three_black_crows", dtype=int)


# ---------------------------------------------------------------------------
# Harami
# ---------------------------------------------------------------------------


def harami(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Harami pattern (bullish and bearish).

    The second candle's body is entirely contained within the first
    candle's body.

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

    Returns
    -------
    pd.Series
        1 (bullish harami), -1 (bearish harami), or 0.
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    prev_body_high = pd.concat([open_.shift(1), close.shift(1)], axis=1).max(axis=1)
    prev_body_low = pd.concat([open_.shift(1), close.shift(1)], axis=1).min(axis=1)
    curr_body_high = pd.concat([open_, close], axis=1).max(axis=1)
    curr_body_low = pd.concat([open_, close], axis=1).min(axis=1)

    # Current body within previous body
    contained = (curr_body_high <= prev_body_high) & (curr_body_low >= prev_body_low)

    prev_bearish = close.shift(1) < open_.shift(1)
    prev_bullish = close.shift(1) > open_.shift(1)

    bullish_harami = contained & prev_bearish  # bullish reversal
    bearish_harami = contained & prev_bullish  # bearish reversal

    result = np.where(bullish_harami, 1, np.where(bearish_harami, -1, 0))
    return pd.Series(result, index=close.index, name="harami", dtype=int)


# ---------------------------------------------------------------------------
# Spinning Top
# ---------------------------------------------------------------------------


def spinning_top(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    body_threshold: float = 0.3,
) -> pd.Series:
    """Spinning Top (indecision candle).

    A candle with a small body relative to its range and roughly equal
    upper and lower shadows.

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
    body_threshold : float, default 0.3
        Maximum body-to-range ratio.

    Returns
    -------
    pd.Series
        1 where a Spinning Top is detected, 0 otherwise.
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    body = _body(open_, close)
    rng = _range(high, low)
    upper = _upper_shadow(open_, high, close)
    lower = _lower_shadow(open_, low, close)

    body_ratio = pd.Series(
        np.where(rng != 0, body / rng, 0.0),
        index=close.index,
    )
    small_body = body_ratio <= body_threshold

    # Both shadows should be meaningfully present
    has_shadows = (upper > body * 0.3) & (lower > body * 0.3)

    signal = small_body & has_shadows & (rng > 0)
    return pd.Series(signal.astype(int), index=close.index, name="spinning_top")


# ---------------------------------------------------------------------------
# Marubozu
# ---------------------------------------------------------------------------


def marubozu(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    threshold: float = 0.01,
) -> pd.Series:
    """Marubozu (full-body candle with no/tiny wicks).

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
    threshold : float, default 0.01
        Maximum shadow-to-range ratio for each shadow.

    Returns
    -------
    pd.Series
        1 (bullish marubozu), -1 (bearish marubozu), or 0.
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    rng = _range(high, low)
    upper = _upper_shadow(open_, high, close)
    lower = _lower_shadow(open_, low, close)

    upper_ratio = pd.Series(np.where(rng != 0, upper / rng, 0.0), index=close.index)
    lower_ratio = pd.Series(np.where(rng != 0, lower / rng, 0.0), index=close.index)

    tiny_shadows = (upper_ratio <= threshold) & (lower_ratio <= threshold) & (rng > 0)

    bullish = close > open_
    bearish = close < open_

    result = np.where(
        tiny_shadows & bullish,
        1,
        np.where(tiny_shadows & bearish, -1, 0),
    )
    return pd.Series(result, index=close.index, name="marubozu", dtype=int)
