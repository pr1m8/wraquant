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
    "piercing_pattern",
    "dark_cloud_cover",
    "hanging_man",
    "inverted_hammer",
    "shooting_star",
    "tweezer_top",
    "tweezer_bottom",
    "three_inside_up",
    "three_inside_down",
    "abandoned_baby",
    "kicking",
    "belt_hold",
    "rising_three_methods",
    "falling_three_methods",
    "tasuki_gap",
    "on_neck",
    "in_neck",
    "thrusting",
    "separating_lines",
    "closing_marubozu",
    "rickshaw_man",
    "long_legged_doji",
    "dragonfly_doji",
    "gravestone_doji",
    "tri_star",
    "unique_three_river",
    "concealing_baby_swallow",
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


# ---------------------------------------------------------------------------
# Piercing Pattern
# ---------------------------------------------------------------------------


def piercing_pattern(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Piercing Pattern (bullish reversal).

    A two-candle pattern: Day 1 is bearish, Day 2 opens below Day 1's
    low and closes above the midpoint of Day 1's body.

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
        1 where a Piercing Pattern is detected, 0 otherwise.

    Example
    -------
    >>> signal = piercing_pattern(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    prev_bearish = close.shift(1) < open_.shift(1)
    curr_bullish = close > open_

    # Current opens below previous low
    opens_below = open_ < low.shift(1)

    # Closes above midpoint of previous body
    prev_midpoint = (open_.shift(1) + close.shift(1)) / 2.0
    closes_above_mid = close > prev_midpoint

    # Does not close above previous open (otherwise it's engulfing)
    not_engulfing = close < open_.shift(1)

    signal = prev_bearish & curr_bullish & opens_below & closes_above_mid & not_engulfing
    return pd.Series(signal.astype(int), index=close.index, name="piercing_pattern")


# ---------------------------------------------------------------------------
# Dark Cloud Cover
# ---------------------------------------------------------------------------


def dark_cloud_cover(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Dark Cloud Cover (bearish reversal).

    The bearish counterpart of the Piercing Pattern. Day 1 is bullish,
    Day 2 opens above Day 1's high and closes below the midpoint of
    Day 1's body.

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
        -1 where a Dark Cloud Cover is detected, 0 otherwise.

    Example
    -------
    >>> signal = dark_cloud_cover(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    prev_bullish = close.shift(1) > open_.shift(1)
    curr_bearish = close < open_

    # Current opens above previous high
    opens_above = open_ > high.shift(1)

    # Closes below midpoint of previous body
    prev_midpoint = (open_.shift(1) + close.shift(1)) / 2.0
    closes_below_mid = close < prev_midpoint

    # Does not close below previous open (otherwise it's engulfing)
    not_engulfing = close > open_.shift(1)

    signal = prev_bullish & curr_bearish & opens_above & closes_below_mid & not_engulfing
    result = np.where(signal, -1, 0)
    return pd.Series(result, index=close.index, name="dark_cloud_cover", dtype=int)


# ---------------------------------------------------------------------------
# Hanging Man
# ---------------------------------------------------------------------------


def hanging_man(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    trend_period: int = 5,
) -> pd.Series:
    """Hanging Man (bearish reversal at top).

    Same hammer shape (small body near top, long lower shadow) but appears
    after an uptrend, signalling potential reversal.

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
    trend_period : int, default 5
        Number of bars to assess prior uptrend.

    Returns
    -------
    pd.Series
        -1 where a Hanging Man is detected, 0 otherwise.

    Example
    -------
    >>> signal = hanging_man(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    body = _body(open_, close)
    lower = _lower_shadow(open_, low, close)
    upper = _upper_shadow(open_, high, close)
    rng = _range(high, low)

    is_hammer_shape = (lower >= 2 * body) & (upper <= body * 0.5) & (rng > 0)

    # Prior uptrend: close higher than close N bars ago
    prior_uptrend = close.shift(1) > close.shift(trend_period)

    signal = is_hammer_shape & prior_uptrend
    result = np.where(signal, -1, 0)
    return pd.Series(result, index=close.index, name="hanging_man", dtype=int)


# ---------------------------------------------------------------------------
# Inverted Hammer
# ---------------------------------------------------------------------------


def inverted_hammer(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    trend_period: int = 5,
) -> pd.Series:
    """Inverted Hammer (bullish reversal at bottom).

    A candle with a long upper shadow (at least 2x the body) and a
    small lower shadow, appearing after a downtrend.

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
    trend_period : int, default 5
        Number of bars to assess prior downtrend.

    Returns
    -------
    pd.Series
        1 where an Inverted Hammer is detected, 0 otherwise.

    Example
    -------
    >>> signal = inverted_hammer(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    body = _body(open_, close)
    upper = _upper_shadow(open_, high, close)
    lower = _lower_shadow(open_, low, close)
    rng = _range(high, low)

    is_inverted_shape = (upper >= 2 * body) & (lower <= body * 0.5) & (rng > 0)

    # Prior downtrend: close lower than close N bars ago
    prior_downtrend = close.shift(1) < close.shift(trend_period)

    signal = is_inverted_shape & prior_downtrend
    return pd.Series(signal.astype(int), index=close.index, name="inverted_hammer")


# ---------------------------------------------------------------------------
# Shooting Star
# ---------------------------------------------------------------------------


def shooting_star(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    trend_period: int = 5,
) -> pd.Series:
    """Shooting Star (bearish reversal at top).

    Same shape as inverted hammer (long upper shadow, small lower shadow)
    but appears after an uptrend.

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
    trend_period : int, default 5
        Number of bars to assess prior uptrend.

    Returns
    -------
    pd.Series
        -1 where a Shooting Star is detected, 0 otherwise.

    Example
    -------
    >>> signal = shooting_star(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    body = _body(open_, close)
    upper = _upper_shadow(open_, high, close)
    lower = _lower_shadow(open_, low, close)
    rng = _range(high, low)

    is_shooting_shape = (upper >= 2 * body) & (lower <= body * 0.5) & (rng > 0)

    # Prior uptrend
    prior_uptrend = close.shift(1) > close.shift(trend_period)

    signal = is_shooting_shape & prior_uptrend
    result = np.where(signal, -1, 0)
    return pd.Series(result, index=close.index, name="shooting_star", dtype=int)


# ---------------------------------------------------------------------------
# Tweezer Top
# ---------------------------------------------------------------------------


def tweezer_top(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tolerance: float = 0.001,
) -> pd.Series:
    """Tweezer Top (bearish reversal).

    Two consecutive candles with nearly the same highs. The first candle
    is bullish and the second is bearish.

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
    tolerance : float, default 0.001
        Maximum relative difference between highs to be considered equal.

    Returns
    -------
    pd.Series
        -1 where a Tweezer Top is detected, 0 otherwise.

    Example
    -------
    >>> signal = tweezer_top(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    prev_bullish = close.shift(1) > open_.shift(1)
    curr_bearish = close < open_

    # Same highs (within tolerance)
    high_diff = ((high - high.shift(1)).abs() / high.shift(1))
    same_highs = high_diff <= tolerance

    signal = prev_bullish & curr_bearish & same_highs
    result = np.where(signal, -1, 0)
    return pd.Series(result, index=close.index, name="tweezer_top", dtype=int)


# ---------------------------------------------------------------------------
# Tweezer Bottom
# ---------------------------------------------------------------------------


def tweezer_bottom(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tolerance: float = 0.001,
) -> pd.Series:
    """Tweezer Bottom (bullish reversal).

    Two consecutive candles with nearly the same lows. The first candle
    is bearish and the second is bullish.

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
    tolerance : float, default 0.001
        Maximum relative difference between lows to be considered equal.

    Returns
    -------
    pd.Series
        1 where a Tweezer Bottom is detected, 0 otherwise.

    Example
    -------
    >>> signal = tweezer_bottom(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    prev_bearish = close.shift(1) < open_.shift(1)
    curr_bullish = close > open_

    # Same lows (within tolerance)
    low_diff = ((low - low.shift(1)).abs() / low.shift(1))
    same_lows = low_diff <= tolerance

    signal = prev_bearish & curr_bullish & same_lows
    return pd.Series(signal.astype(int), index=close.index, name="tweezer_bottom")


# ---------------------------------------------------------------------------
# Three Inside Up
# ---------------------------------------------------------------------------


def three_inside_up(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Three Inside Up (bullish reversal).

    A three-candle pattern: bullish harami (Days 1-2) confirmed by a
    third bullish candle closing above Day 1's open.

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

    Example
    -------
    >>> signal = three_inside_up(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    # Day 1: large bearish candle
    d1_bearish = close.shift(2) < open_.shift(2)

    # Day 2: bullish candle contained within Day 1 body (harami)
    d2_bullish = close.shift(1) > open_.shift(1)
    d2_body_high = pd.concat([open_.shift(1), close.shift(1)], axis=1).max(axis=1)
    d2_body_low = pd.concat([open_.shift(1), close.shift(1)], axis=1).min(axis=1)
    d2_contained = (d2_body_high <= open_.shift(2)) & (d2_body_low >= close.shift(2))

    # Day 3: bullish candle closing above Day 1 open
    d3_bullish = close > open_
    d3_above_d1 = close > open_.shift(2)

    signal = d1_bearish & d2_bullish & d2_contained & d3_bullish & d3_above_d1
    return pd.Series(signal.astype(int), index=close.index, name="three_inside_up")


# ---------------------------------------------------------------------------
# Three Inside Down
# ---------------------------------------------------------------------------


def three_inside_down(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Three Inside Down (bearish reversal).

    A three-candle pattern: bearish harami (Days 1-2) confirmed by a
    third bearish candle closing below Day 1's open.

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

    Example
    -------
    >>> signal = three_inside_down(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    # Day 1: large bullish candle
    d1_bullish = close.shift(2) > open_.shift(2)

    # Day 2: bearish candle contained within Day 1 body (harami)
    d2_bearish = close.shift(1) < open_.shift(1)
    d2_body_high = pd.concat([open_.shift(1), close.shift(1)], axis=1).max(axis=1)
    d2_body_low = pd.concat([open_.shift(1), close.shift(1)], axis=1).min(axis=1)
    d2_contained = (d2_body_high <= close.shift(2)) & (d2_body_low >= open_.shift(2))

    # Day 3: bearish candle closing below Day 1 open
    d3_bearish = close < open_
    d3_below_d1 = close < open_.shift(2)

    signal = d1_bullish & d2_bearish & d2_contained & d3_bearish & d3_below_d1
    result = np.where(signal, -1, 0)
    return pd.Series(result, index=close.index, name="three_inside_down", dtype=int)


# ---------------------------------------------------------------------------
# Abandoned Baby
# ---------------------------------------------------------------------------


def abandoned_baby(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Abandoned Baby (reversal pattern).

    A rare three-candle pattern with gaps. A Doji star gaps away from the
    first candle and the third candle gaps in the opposite direction.

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
        1 (bullish abandoned baby), -1 (bearish abandoned baby), or 0.

    Example
    -------
    >>> signal = abandoned_baby(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    body = _body(open_, close)
    rng = _range(high, low)

    # Day 2 is a doji-like candle
    ratio_d2 = pd.Series(
        np.where(rng.shift(1) != 0, body.shift(1) / rng.shift(1), 0.0),
        index=close.index,
    )
    d2_doji = ratio_d2 <= 0.1

    # Bullish: Day 1 bearish, gap down to Day 2, gap up to Day 3 bullish
    d1_bearish = close.shift(2) < open_.shift(2)
    gap_down = high.shift(1) < low.shift(2)  # Day 2 high < Day 1 low
    d3_bullish = close > open_
    gap_up = low > high.shift(1)  # Day 3 low > Day 2 high

    bullish_baby = d1_bearish & d2_doji & gap_down & d3_bullish & gap_up

    # Bearish: Day 1 bullish, gap up to Day 2, gap down to Day 3 bearish
    d1_bullish = close.shift(2) > open_.shift(2)
    gap_up_d2 = low.shift(1) > high.shift(2)  # Day 2 low > Day 1 high
    d3_bearish = close < open_
    gap_down_d3 = high < low.shift(1)  # Day 3 high < Day 2 low

    bearish_baby = d1_bullish & d2_doji & gap_up_d2 & d3_bearish & gap_down_d3

    result = np.where(bullish_baby, 1, np.where(bearish_baby, -1, 0))
    return pd.Series(result, index=close.index, name="abandoned_baby", dtype=int)


# ---------------------------------------------------------------------------
# Kicking
# ---------------------------------------------------------------------------


def kicking(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    threshold: float = 0.01,
) -> pd.Series:
    """Kicking pattern.

    Two consecutive marubozu candles in opposite directions with a gap
    between them. One of the strongest reversal signals.

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
        Maximum shadow-to-range ratio for marubozu qualification.

    Returns
    -------
    pd.Series
        1 (bullish kicking), -1 (bearish kicking), or 0.

    Example
    -------
    >>> signal = kicking(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    rng = _range(high, low)
    upper = _upper_shadow(open_, high, close)
    lower = _lower_shadow(open_, low, close)

    prev_rng = _range(high.shift(1), low.shift(1))
    prev_upper = _upper_shadow(open_.shift(1), high.shift(1), close.shift(1))
    prev_lower = _lower_shadow(open_.shift(1), low.shift(1), close.shift(1))

    # Current candle is marubozu
    curr_upper_ratio = pd.Series(
        np.where(rng != 0, upper / rng, 0.0), index=close.index
    )
    curr_lower_ratio = pd.Series(
        np.where(rng != 0, lower / rng, 0.0), index=close.index
    )
    curr_maru = (curr_upper_ratio <= threshold) & (curr_lower_ratio <= threshold) & (rng > 0)

    # Previous candle is marubozu
    prev_upper_ratio = pd.Series(
        np.where(prev_rng != 0, prev_upper / prev_rng, 0.0), index=close.index
    )
    prev_lower_ratio = pd.Series(
        np.where(prev_rng != 0, prev_lower / prev_rng, 0.0), index=close.index
    )
    prev_maru = (prev_upper_ratio <= threshold) & (prev_lower_ratio <= threshold) & (prev_rng > 0)

    # Bullish kicking: prev bearish marubozu, curr bullish marubozu with gap up
    prev_bear = close.shift(1) < open_.shift(1)
    curr_bull = close > open_
    gap_up = open_ > open_.shift(1)  # gap up from previous open

    # Bearish kicking: prev bullish marubozu, curr bearish marubozu with gap down
    prev_bull = close.shift(1) > open_.shift(1)
    curr_bear = close < open_
    gap_down = open_ < open_.shift(1)  # gap down from previous open

    bullish_kick = prev_maru & curr_maru & prev_bear & curr_bull & gap_up
    bearish_kick = prev_maru & curr_maru & prev_bull & curr_bear & gap_down

    result = np.where(bullish_kick, 1, np.where(bearish_kick, -1, 0))
    return pd.Series(result, index=close.index, name="kicking", dtype=int)


# ---------------------------------------------------------------------------
# Belt Hold
# ---------------------------------------------------------------------------


def belt_hold(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    threshold: float = 0.01,
) -> pd.Series:
    """Belt Hold pattern.

    A long marubozu candle that opens with a gap in the direction of the
    prior trend. A bullish belt hold gaps down and opens at the low; a
    bearish belt hold gaps up and opens at the high.

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
        Maximum shadow-to-range ratio for the relevant shadow.

    Returns
    -------
    pd.Series
        1 (bullish belt hold), -1 (bearish belt hold), or 0.

    Example
    -------
    >>> signal = belt_hold(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    rng = _range(high, low)
    lower = _lower_shadow(open_, low, close)
    upper = _upper_shadow(open_, high, close)

    lower_ratio = pd.Series(
        np.where(rng != 0, lower / rng, 0.0), index=close.index
    )
    upper_ratio = pd.Series(
        np.where(rng != 0, upper / rng, 0.0), index=close.index
    )

    body = _body(open_, close)
    large_body = body > rng * 0.6

    # Bullish belt hold: gaps down, opens at low (tiny lower shadow), bullish
    curr_bullish = close > open_
    gap_down = open_ < close.shift(1)
    bullish_belt = curr_bullish & gap_down & (lower_ratio <= threshold) & large_body & (rng > 0)

    # Bearish belt hold: gaps up, opens at high (tiny upper shadow), bearish
    curr_bearish = close < open_
    gap_up = open_ > close.shift(1)
    bearish_belt = curr_bearish & gap_up & (upper_ratio <= threshold) & large_body & (rng > 0)

    result = np.where(bullish_belt, 1, np.where(bearish_belt, -1, 0))
    return pd.Series(result, index=close.index, name="belt_hold", dtype=int)


# ---------------------------------------------------------------------------
# Rising Three Methods
# ---------------------------------------------------------------------------


def rising_three_methods(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Rising Three Methods (bullish continuation).

    A five-candle pattern: a long bullish candle, followed by three small
    bearish candles that stay within the first candle's range, then a
    final long bullish candle that closes above the first candle's close.

    Parameters:
        open_: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.

    Returns:
        1 where the pattern is detected, 0 otherwise.

    Example:
        >>> signal = rising_three_methods(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    result = pd.Series(0, index=close.index, name="rising_three_methods", dtype=int)

    for i in range(4, len(close)):
        # Day 1 (i-4): long bullish candle
        d1_open = open_.iloc[i - 4]
        d1_close = close.iloc[i - 4]
        d1_high = high.iloc[i - 4]
        d1_low = low.iloc[i - 4]
        d1_body = abs(d1_close - d1_open)
        d1_range = d1_high - d1_low
        if d1_close <= d1_open or d1_range == 0 or d1_body < d1_range * 0.5:
            continue

        # Days 2-4 (i-3, i-2, i-1): small candles within Day 1 range
        middle_ok = True
        for j in range(i - 3, i):
            if high.iloc[j] > d1_high or low.iloc[j] < d1_low:
                middle_ok = False
                break
            if close.iloc[j] >= open_.iloc[j]:  # must be bearish or small
                mid_body = abs(close.iloc[j] - open_.iloc[j])
                if mid_body > d1_body * 0.5:
                    middle_ok = False
                    break

        if not middle_ok:
            continue

        # Day 5 (i): long bullish candle closing above Day 1 close
        d5_body = abs(close.iloc[i] - open_.iloc[i])
        d5_range = high.iloc[i] - low.iloc[i]
        if (
            close.iloc[i] > open_.iloc[i]
            and close.iloc[i] > d1_close
            and d5_range > 0
            and d5_body > d5_range * 0.5
        ):
            result.iloc[i] = 1

    return result


# ---------------------------------------------------------------------------
# Falling Three Methods
# ---------------------------------------------------------------------------


def falling_three_methods(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Falling Three Methods (bearish continuation).

    The bearish counterpart of Rising Three Methods.  A long bearish candle,
    three small bullish candles inside its range, then a final long bearish
    candle that closes below the first candle's close.

    Parameters:
        open_: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.

    Returns:
        -1 where the pattern is detected, 0 otherwise.

    Example:
        >>> signal = falling_three_methods(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    result = pd.Series(0, index=close.index, name="falling_three_methods", dtype=int)

    for i in range(4, len(close)):
        # Day 1 (i-4): long bearish candle
        d1_open = open_.iloc[i - 4]
        d1_close = close.iloc[i - 4]
        d1_high = high.iloc[i - 4]
        d1_low = low.iloc[i - 4]
        d1_body = abs(d1_close - d1_open)
        d1_range = d1_high - d1_low
        if d1_close >= d1_open or d1_range == 0 or d1_body < d1_range * 0.5:
            continue

        # Days 2-4: small candles within Day 1 range
        middle_ok = True
        for j in range(i - 3, i):
            if high.iloc[j] > d1_high or low.iloc[j] < d1_low:
                middle_ok = False
                break
            if close.iloc[j] <= open_.iloc[j]:  # must be bullish or small
                mid_body = abs(close.iloc[j] - open_.iloc[j])
                if mid_body > d1_body * 0.5:
                    middle_ok = False
                    break

        if not middle_ok:
            continue

        # Day 5: long bearish candle closing below Day 1 close
        d5_body = abs(close.iloc[i] - open_.iloc[i])
        d5_range = high.iloc[i] - low.iloc[i]
        if (
            close.iloc[i] < open_.iloc[i]
            and close.iloc[i] < d1_close
            and d5_range > 0
            and d5_body > d5_range * 0.5
        ):
            result.iloc[i] = -1

    return result


# ---------------------------------------------------------------------------
# Tasuki Gap
# ---------------------------------------------------------------------------


def tasuki_gap(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Upside/Downside Tasuki Gap.

    **Upside** (1): two bullish candles with a gap up, followed by a bearish
    candle that opens within the second body and closes within the gap but
    does not fill it.

    **Downside** (-1): two bearish candles with a gap down, followed by a
    bullish candle that opens within the second body and closes within the
    gap but does not fill it.

    Parameters:
        open_: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.

    Returns:
        1 (upside Tasuki gap), -1 (downside Tasuki gap), or 0.

    Example:
        >>> signal = tasuki_gap(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    # Upside Tasuki gap
    d1_bull = close.shift(2) > open_.shift(2)
    d2_bull = close.shift(1) > open_.shift(1)
    gap_up = open_.shift(1) > close.shift(2)  # gap up between d1 and d2
    d3_bear = close < open_
    d3_opens_in_d2 = (open_ >= open_.shift(1)) & (open_ <= close.shift(1))
    d3_closes_in_gap = (close < open_.shift(1)) & (close > close.shift(2))

    upside = d1_bull & d2_bull & gap_up & d3_bear & d3_opens_in_d2 & d3_closes_in_gap

    # Downside Tasuki gap
    d1_bear = close.shift(2) < open_.shift(2)
    d2_bear = close.shift(1) < open_.shift(1)
    gap_down = open_.shift(1) < close.shift(2)  # gap down between d1 and d2
    d3_bull = close > open_
    d3_opens_in_d2_dn = (open_ <= open_.shift(1)) & (open_ >= close.shift(1))
    d3_closes_in_gap_dn = (close > open_.shift(1)) & (close < close.shift(2))

    downside = (
        d1_bear & d2_bear & gap_down & d3_bull & d3_opens_in_d2_dn & d3_closes_in_gap_dn
    )

    result = np.where(upside, 1, np.where(downside, -1, 0))
    return pd.Series(result, index=close.index, name="tasuki_gap", dtype=int)


# ---------------------------------------------------------------------------
# On Neck
# ---------------------------------------------------------------------------


def on_neck(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tolerance: float = 0.001,
) -> pd.Series:
    """On Neck pattern (bearish continuation).

    A bearish candle followed by a small bullish candle that opens below
    the previous low and closes at or near the previous low.

    Parameters:
        open_: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.
        tolerance: Maximum relative difference between close and previous
            low.

    Returns:
        -1 where the pattern is detected, 0 otherwise.

    Example:
        >>> signal = on_neck(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    prev_bearish = close.shift(1) < open_.shift(1)
    curr_bullish = close > open_
    opens_below = open_ < low.shift(1)

    # Close at or near previous low
    prev_low = low.shift(1)
    close_near_prev_low = ((close - prev_low).abs() / prev_low) <= tolerance

    signal = prev_bearish & curr_bullish & opens_below & close_near_prev_low
    result = np.where(signal, -1, 0)
    return pd.Series(result, index=close.index, name="on_neck", dtype=int)


# ---------------------------------------------------------------------------
# In Neck
# ---------------------------------------------------------------------------


def in_neck(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tolerance: float = 0.003,
) -> pd.Series:
    """In Neck pattern (slight bullish variant of On Neck).

    A bearish candle followed by a small bullish candle that opens below
    the previous low and closes slightly above (but near) the previous
    close.

    Parameters:
        open_: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.
        tolerance: Maximum relative distance above the previous close.

    Returns:
        -1 where the pattern is detected, 0 otherwise.

    Example:
        >>> signal = in_neck(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    prev_bearish = close.shift(1) < open_.shift(1)
    curr_bullish = close > open_
    opens_below = open_ < low.shift(1)

    prev_close = close.shift(1)
    # Close slightly above previous close
    close_near_prev_close = (close >= prev_close) & (
        ((close - prev_close) / prev_close) <= tolerance
    )

    signal = prev_bearish & curr_bullish & opens_below & close_near_prev_close
    result = np.where(signal, -1, 0)
    return pd.Series(result, index=close.index, name="in_neck", dtype=int)


# ---------------------------------------------------------------------------
# Thrusting
# ---------------------------------------------------------------------------


def thrusting(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Thrusting pattern (moderate bearish continuation).

    A bearish candle followed by a bullish candle that opens below the
    previous low and closes above the previous close but below the midpoint
    of the previous body.

    Parameters:
        open_: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.

    Returns:
        -1 where the pattern is detected, 0 otherwise.

    Example:
        >>> signal = thrusting(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    prev_bearish = close.shift(1) < open_.shift(1)
    curr_bullish = close > open_
    opens_below = open_ < low.shift(1)

    prev_midpoint = (open_.shift(1) + close.shift(1)) / 2.0
    closes_above_prev_close = close > close.shift(1)
    closes_below_mid = close < prev_midpoint

    signal = prev_bearish & curr_bullish & opens_below & closes_above_prev_close & closes_below_mid
    result = np.where(signal, -1, 0)
    return pd.Series(result, index=close.index, name="thrusting", dtype=int)


# ---------------------------------------------------------------------------
# Separating Lines
# ---------------------------------------------------------------------------


def separating_lines(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tolerance: float = 0.001,
) -> pd.Series:
    """Bullish/Bearish Separating Lines.

    **Bullish** (1): a bearish candle followed by a bullish candle that
    opens at the same level as the previous open.

    **Bearish** (-1): a bullish candle followed by a bearish candle that
    opens at the same level as the previous open.

    Parameters:
        open_: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.
        tolerance: Maximum relative difference between opens.

    Returns:
        1 (bullish), -1 (bearish), or 0.

    Example:
        >>> signal = separating_lines(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    prev_open = open_.shift(1)
    same_open = ((open_ - prev_open).abs() / prev_open) <= tolerance

    prev_bearish = close.shift(1) < open_.shift(1)
    prev_bullish = close.shift(1) > open_.shift(1)
    curr_bullish = close > open_
    curr_bearish = close < open_

    bullish = prev_bearish & curr_bullish & same_open
    bearish = prev_bullish & curr_bearish & same_open

    result = np.where(bullish, 1, np.where(bearish, -1, 0))
    return pd.Series(result, index=close.index, name="separating_lines", dtype=int)


# ---------------------------------------------------------------------------
# Closing Marubozu
# ---------------------------------------------------------------------------


def closing_marubozu(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    threshold: float = 0.01,
) -> pd.Series:
    """Closing Marubozu — no shadow on the closing side only.

    A **bullish closing marubozu** (1) has no upper shadow (close == high)
    but may have a lower shadow.  A **bearish closing marubozu** (-1) has
    no lower shadow (close == low) but may have an upper shadow.

    Parameters:
        open_: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.
        threshold: Maximum shadow-to-range ratio for the closing side.

    Returns:
        1 (bullish), -1 (bearish), or 0.

    Example:
        >>> signal = closing_marubozu(open_, high, low, close)
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

    body = _body(open_, close)
    has_body = body > rng * 0.5

    # Bullish: close near high (tiny upper shadow), bullish candle
    bullish = (close > open_) & (upper_ratio <= threshold) & has_body & (rng > 0)

    # Bearish: close near low (tiny lower shadow), bearish candle
    bearish = (close < open_) & (lower_ratio <= threshold) & has_body & (rng > 0)

    result = np.where(bullish, 1, np.where(bearish, -1, 0))
    return pd.Series(result, index=close.index, name="closing_marubozu", dtype=int)


# ---------------------------------------------------------------------------
# Rickshaw Man
# ---------------------------------------------------------------------------


def rickshaw_man(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    body_threshold: float = 0.05,
    shadow_threshold: float = 0.3,
) -> pd.Series:
    """Rickshaw Man — a Doji with very long shadows and tiny body near center.

    Parameters:
        open_: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.
        body_threshold: Maximum body-to-range ratio to qualify.
        shadow_threshold: Minimum shadow-to-range ratio for each shadow.

    Returns:
        1 where detected, 0 otherwise.

    Example:
        >>> signal = rickshaw_man(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    body = _body(open_, close)
    rng = _range(high, low)
    upper = _upper_shadow(open_, high, close)
    lower = _lower_shadow(open_, low, close)

    body_ratio = pd.Series(np.where(rng != 0, body / rng, 0.0), index=close.index)
    upper_ratio = pd.Series(np.where(rng != 0, upper / rng, 0.0), index=close.index)
    lower_ratio = pd.Series(np.where(rng != 0, lower / rng, 0.0), index=close.index)

    # Body near center: midpoint of body near midpoint of range
    body_mid = (open_ + close) / 2.0
    range_mid = (high + low) / 2.0
    near_center = pd.Series(
        np.where(rng != 0, (body_mid - range_mid).abs() / rng, 0.0),
        index=close.index,
    )

    signal = (
        (body_ratio <= body_threshold)
        & (upper_ratio >= shadow_threshold)
        & (lower_ratio >= shadow_threshold)
        & (near_center <= 0.1)
        & (rng > 0)
    )
    return pd.Series(signal.astype(int), index=close.index, name="rickshaw_man")


# ---------------------------------------------------------------------------
# Long Legged Doji
# ---------------------------------------------------------------------------


def long_legged_doji(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    body_threshold: float = 0.05,
    shadow_threshold: float = 0.3,
) -> pd.Series:
    """Long Legged Doji — Doji with unusually long upper and lower shadows.

    Parameters:
        open_: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.
        body_threshold: Maximum body-to-range ratio.
        shadow_threshold: Minimum shadow-to-range ratio for each shadow.

    Returns:
        1 where detected, 0 otherwise.

    Example:
        >>> signal = long_legged_doji(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    body = _body(open_, close)
    rng = _range(high, low)
    upper = _upper_shadow(open_, high, close)
    lower = _lower_shadow(open_, low, close)

    body_ratio = pd.Series(np.where(rng != 0, body / rng, 0.0), index=close.index)
    upper_ratio = pd.Series(np.where(rng != 0, upper / rng, 0.0), index=close.index)
    lower_ratio = pd.Series(np.where(rng != 0, lower / rng, 0.0), index=close.index)

    signal = (
        (body_ratio <= body_threshold)
        & (upper_ratio >= shadow_threshold)
        & (lower_ratio >= shadow_threshold)
        & (rng > 0)
    )
    return pd.Series(signal.astype(int), index=close.index, name="long_legged_doji")


# ---------------------------------------------------------------------------
# Dragonfly Doji
# ---------------------------------------------------------------------------


def dragonfly_doji(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    body_threshold: float = 0.05,
    upper_threshold: float = 0.05,
    lower_min: float = 0.3,
) -> pd.Series:
    """Dragonfly Doji — Doji with a long lower shadow and no upper shadow.

    Parameters:
        open_: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.
        body_threshold: Maximum body-to-range ratio.
        upper_threshold: Maximum upper-shadow-to-range ratio.
        lower_min: Minimum lower-shadow-to-range ratio.

    Returns:
        1 where detected, 0 otherwise.

    Example:
        >>> signal = dragonfly_doji(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    body = _body(open_, close)
    rng = _range(high, low)
    upper = _upper_shadow(open_, high, close)
    lower = _lower_shadow(open_, low, close)

    body_ratio = pd.Series(np.where(rng != 0, body / rng, 0.0), index=close.index)
    upper_ratio = pd.Series(np.where(rng != 0, upper / rng, 0.0), index=close.index)
    lower_ratio = pd.Series(np.where(rng != 0, lower / rng, 0.0), index=close.index)

    signal = (
        (body_ratio <= body_threshold)
        & (upper_ratio <= upper_threshold)
        & (lower_ratio >= lower_min)
        & (rng > 0)
    )
    return pd.Series(signal.astype(int), index=close.index, name="dragonfly_doji")


# ---------------------------------------------------------------------------
# Gravestone Doji
# ---------------------------------------------------------------------------


def gravestone_doji(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    body_threshold: float = 0.05,
    lower_threshold: float = 0.05,
    upper_min: float = 0.3,
) -> pd.Series:
    """Gravestone Doji — Doji with a long upper shadow and no lower shadow.

    Parameters:
        open_: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.
        body_threshold: Maximum body-to-range ratio.
        lower_threshold: Maximum lower-shadow-to-range ratio.
        upper_min: Minimum upper-shadow-to-range ratio.

    Returns:
        1 where detected, 0 otherwise.

    Example:
        >>> signal = gravestone_doji(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    body = _body(open_, close)
    rng = _range(high, low)
    upper = _upper_shadow(open_, high, close)
    lower = _lower_shadow(open_, low, close)

    body_ratio = pd.Series(np.where(rng != 0, body / rng, 0.0), index=close.index)
    upper_ratio = pd.Series(np.where(rng != 0, upper / rng, 0.0), index=close.index)
    lower_ratio = pd.Series(np.where(rng != 0, lower / rng, 0.0), index=close.index)

    signal = (
        (body_ratio <= body_threshold)
        & (lower_ratio <= lower_threshold)
        & (upper_ratio >= upper_min)
        & (rng > 0)
    )
    return pd.Series(signal.astype(int), index=close.index, name="gravestone_doji")


# ---------------------------------------------------------------------------
# Tri Star
# ---------------------------------------------------------------------------


def tri_star(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    doji_threshold: float = 0.05,
) -> pd.Series:
    """Tri-Star pattern — three consecutive dojis with gaps.

    **Bullish** (1): three dojis where the middle gaps below the other two.
    **Bearish** (-1): three dojis where the middle gaps above the other two.

    Parameters:
        open_: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.
        doji_threshold: Maximum body-to-range ratio for doji qualification.

    Returns:
        1 (bullish tri-star), -1 (bearish tri-star), or 0.

    Example:
        >>> signal = tri_star(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    body = _body(open_, close)
    rng = _range(high, low)

    ratio = pd.Series(np.where(rng != 0, body / rng, 0.0), index=close.index)

    is_doji = ratio <= doji_threshold
    d1_doji = is_doji.shift(2)
    d2_doji = is_doji.shift(1)
    d3_doji = is_doji

    # Middle candle gaps
    d2_mid = (open_.shift(1) + close.shift(1)) / 2.0
    d1_mid = (open_.shift(2) + close.shift(2)) / 2.0
    d3_mid = (open_ + close) / 2.0

    bullish = d1_doji & d2_doji & d3_doji & (d2_mid < d1_mid) & (d2_mid < d3_mid)
    bearish = d1_doji & d2_doji & d3_doji & (d2_mid > d1_mid) & (d2_mid > d3_mid)

    result = np.where(bullish, 1, np.where(bearish, -1, 0))
    return pd.Series(result, index=close.index, name="tri_star", dtype=int)


# ---------------------------------------------------------------------------
# Unique Three River
# ---------------------------------------------------------------------------


def unique_three_river(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Unique Three River Bottom (rare bullish reversal).

    Day 1: long bearish candle.  Day 2: harami-like bearish candle with a
    lower low.  Day 3: small bullish candle that closes below Day 2's close.

    Parameters:
        open_: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.

    Returns:
        1 where detected, 0 otherwise.

    Example:
        >>> signal = unique_three_river(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    body = _body(open_, close)
    rng = _range(high, low)

    # Day 1: long bearish
    d1_bearish = close.shift(2) < open_.shift(2)
    d1_body = body.shift(2)
    d1_range = rng.shift(2)
    d1_large = d1_body > d1_range * 0.5

    # Day 2: bearish candle inside Day 1 body but with a lower low
    d2_bearish = close.shift(1) < open_.shift(1)
    d2_inside = (close.shift(1) >= close.shift(2)) & (open_.shift(1) <= open_.shift(2))
    d2_lower_low = low.shift(1) < low.shift(2)

    # Day 3: small bullish candle closing below Day 2 close
    d3_bullish = close > open_
    d3_small = body < body.shift(1)
    d3_below_d2 = close < close.shift(1)

    signal = d1_bearish & d1_large & d2_bearish & d2_inside & d2_lower_low & d3_bullish & d3_small & d3_below_d2
    return pd.Series(signal.astype(int), index=close.index, name="unique_three_river")


# ---------------------------------------------------------------------------
# Concealing Baby Swallow
# ---------------------------------------------------------------------------


def concealing_baby_swallow(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    marubozu_threshold: float = 0.02,
) -> pd.Series:
    """Concealing Baby Swallow (four-candle bearish pattern).

    Four bearish candles: Days 1-2 are bearish marubozus.  Day 3 gaps down,
    has a long upper shadow into Day 2's body.  Day 4 engulfs Day 3
    including the shadow.

    Parameters:
        open_: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.
        marubozu_threshold: Maximum shadow-to-range ratio for marubozu.

    Returns:
        -1 where detected, 0 otherwise.

    Example:
        >>> signal = concealing_baby_swallow(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    rng = _range(high, low)
    upper = _upper_shadow(open_, high, close)
    lower = _lower_shadow(open_, low, close)

    def _is_bear_maru(shift: int) -> pd.Series:
        s_rng = rng.shift(shift) if shift > 0 else rng
        s_upper = upper.shift(shift) if shift > 0 else upper
        s_lower = lower.shift(shift) if shift > 0 else lower
        s_close = close.shift(shift) if shift > 0 else close
        s_open = open_.shift(shift) if shift > 0 else open_
        ur = pd.Series(np.where(s_rng != 0, s_upper / s_rng, 0.0), index=close.index)
        lr = pd.Series(np.where(s_rng != 0, s_lower / s_rng, 0.0), index=close.index)
        return (s_close < s_open) & (ur <= marubozu_threshold) & (lr <= marubozu_threshold) & (s_rng > 0)

    # Days 1 and 2: bearish marubozus
    d1_maru = _is_bear_maru(3)
    d2_maru = _is_bear_maru(2)

    # Day 3: bearish, gaps down, upper shadow reaches into Day 2 body
    d3_bearish = close.shift(1) < open_.shift(1)
    d3_gap_down = open_.shift(1) < close.shift(2)
    d3_upper_into_d2 = high.shift(1) > close.shift(2)

    # Day 4: bearish, engulfs Day 3 (including shadow)
    d4_bearish = close < open_
    d4_engulfs = (open_ >= high.shift(1)) & (close <= low.shift(1))

    signal = d1_maru & d2_maru & d3_bearish & d3_gap_down & d3_upper_into_d2 & d4_bearish & d4_engulfs
    result = np.where(signal, -1, 0)
    return pd.Series(result, index=close.index, name="concealing_baby_swallow", dtype=int)
