"""Price action analysis.

Functions for detecting structural price action features such as swing
points, trend bars, gaps, range expansion/contraction, and reversal bars.
All functions accept ``pd.Series`` inputs and return ``pd.Series`` (or
``dict`` where noted).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "higher_highs_lows",
    "swing_high",
    "swing_low",
    "trend_bars",
    "gap_analysis",
    "range_expansion",
    "narrow_range",
    "wide_range_bar",
    "key_reversal",
    "pivot_reversal",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


from wraquant.ta._validators import validate_period as _validate_period
from wraquant.ta._validators import validate_series as _validate_series

# ---------------------------------------------------------------------------
# higher_highs_lows
# ---------------------------------------------------------------------------


def higher_highs_lows(
    high: pd.Series,
    low: pd.Series,
    period: int = 5,
) -> pd.Series:
    """Detect sequences of HH/HL (uptrend) or LH/LL (downtrend).

    Compares rolling highest-high and lowest-low over *period* bars to
    determine whether the market is making higher highs & higher lows
    (uptrend = 1), lower highs & lower lows (downtrend = -1), or neither
    (0).

    Parameters:
        high: High prices.
        low: Low prices.
        period: Look-back window for swing comparison.

    Returns:
        1 (uptrend / HH+HL), -1 (downtrend / LH+LL), or 0.

    Example:
        >>> trend = higher_highs_lows(high, low, period=5)
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_period(period)

    prev_high = high.shift(1).rolling(window=period, min_periods=period).max()
    prev_low = low.shift(1).rolling(window=period, min_periods=period).min()

    curr_high = high.rolling(window=period, min_periods=period).max()
    curr_low = low.rolling(window=period, min_periods=period).min()

    hh = curr_high > prev_high  # higher high
    hl = curr_low > prev_low  # higher low
    lh = curr_high < prev_high  # lower high
    ll = curr_low < prev_low  # lower low

    uptrend = hh & hl
    downtrend = lh & ll

    result = np.where(uptrend, 1, np.where(downtrend, -1, 0))
    return pd.Series(result, index=high.index, name="higher_highs_lows", dtype=int)


# ---------------------------------------------------------------------------
# swing_high
# ---------------------------------------------------------------------------


def swing_high(
    high: pd.Series,
    lookback: int = 2,
    lookahead: int = 2,
) -> pd.Series:
    """Detect swing highs.

    A swing high is a bar whose high is greater than the highs of the
    *lookback* bars before it **and** the *lookahead* bars after it.

    Parameters:
        high: High prices.
        lookback: Number of bars to look back.
        lookahead: Number of bars to look ahead.

    Returns:
        Boolean series (True at swing highs).

    Example:
        >>> sh = swing_high(high, lookback=2, lookahead=2)
    """
    _validate_series(high, "high")
    _validate_period(lookback, "lookback")
    _validate_period(lookahead, "lookahead")

    result = pd.Series(False, index=high.index, name="swing_high", dtype=bool)

    for i in range(lookback, len(high) - lookahead):
        left = high.iloc[i - lookback : i]
        right = high.iloc[i + 1 : i + 1 + lookahead]
        if (high.iloc[i] > left.max()) and (high.iloc[i] > right.max()):
            result.iloc[i] = True

    return result


# ---------------------------------------------------------------------------
# swing_low
# ---------------------------------------------------------------------------


def swing_low(
    low: pd.Series,
    lookback: int = 2,
    lookahead: int = 2,
) -> pd.Series:
    """Detect swing lows.

    A swing low is a bar whose low is less than the lows of the
    *lookback* bars before it **and** the *lookahead* bars after it.

    Parameters:
        low: Low prices.
        lookback: Number of bars to look back.
        lookahead: Number of bars to look ahead.

    Returns:
        Boolean series (True at swing lows).

    Example:
        >>> sl = swing_low(low, lookback=2, lookahead=2)
    """
    _validate_series(low, "low")
    _validate_period(lookback, "lookback")
    _validate_period(lookahead, "lookahead")

    result = pd.Series(False, index=low.index, name="swing_low", dtype=bool)

    for i in range(lookback, len(low) - lookahead):
        left = low.iloc[i - lookback : i]
        right = low.iloc[i + 1 : i + 1 + lookahead]
        if (low.iloc[i] < left.min()) and (low.iloc[i] < right.min()):
            result.iloc[i] = True

    return result


# ---------------------------------------------------------------------------
# trend_bars
# ---------------------------------------------------------------------------


def trend_bars(
    close: pd.Series,
) -> pd.Series:
    """Count consecutive up/down bars.

    Returns a running count: positive values for consecutive up bars
    (close > previous close), negative values for consecutive down bars
    (close < previous close).  The count resets to 0 on a flat bar.

    Parameters:
        close: Close prices.

    Returns:
        Consecutive bar count (positive = up streak, negative = down streak).

    Example:
        >>> streaks = trend_bars(close)
    """
    _validate_series(close, "close")

    diff = close.diff()
    result = pd.Series(0, index=close.index, name="trend_bars", dtype=int)

    for i in range(1, len(close)):
        if diff.iloc[i] > 0:
            prev = result.iloc[i - 1]
            result.iloc[i] = max(prev, 0) + 1
        elif diff.iloc[i] < 0:
            prev = result.iloc[i - 1]
            result.iloc[i] = min(prev, 0) - 1
        else:
            result.iloc[i] = 0

    return result


# ---------------------------------------------------------------------------
# gap_analysis
# ---------------------------------------------------------------------------


def gap_analysis(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    avg_range_period: int = 20,
    breakaway_threshold: float = 1.5,
) -> dict[str, pd.Series]:
    """Detect and classify gaps.

    Gaps are classified as:

    - **common** — gap size is below the average range
    - **breakaway** — gap size exceeds *breakaway_threshold* times the
      average range
    - **exhaustion** — gap that occurs after a sustained move
      (approximated by comparing the current close to a look-back SMA)

    Parameters:
        open_: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.
        avg_range_period: Period for computing the average range.
        breakaway_threshold: Multiplier of average range for breakaway
            classification.

    Returns:
        Dictionary with keys ``gap_size``, ``gap_direction``, and
        ``gap_type``.

        - ``gap_size`` — absolute gap size
        - ``gap_direction`` — 1 (gap up), -1 (gap down), 0 (no gap)
        - ``gap_type`` — categorical string (``"common"``,
          ``"breakaway"``, ``"exhaustion"``, or ``""`` for no gap)

    Example:
        >>> result = gap_analysis(open_, high, low, close)
        >>> result["gap_type"]
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    prev_high = high.shift(1)
    prev_low = low.shift(1)

    gap_up = open_ > prev_high
    gap_down = open_ < prev_low

    gap_size = pd.Series(0.0, index=close.index, name="gap_size")
    gap_size = gap_size.where(~gap_up, open_ - prev_high)
    gap_size = gap_size.where(~gap_down, prev_low - open_)
    gap_size = gap_size.abs()

    direction = np.where(gap_up, 1, np.where(gap_down, -1, 0))
    gap_direction = pd.Series(
        direction, index=close.index, name="gap_direction", dtype=int
    )

    avg_range = (
        (high - low)
        .rolling(window=avg_range_period, min_periods=avg_range_period)
        .mean()
    )

    sma_close = close.rolling(
        window=avg_range_period, min_periods=avg_range_period
    ).mean()

    has_gap = gap_up | gap_down

    # Exhaustion: gap in direction of sustained move (close far from SMA)
    sustained_up = close.shift(1) > sma_close.shift(1)
    sustained_down = close.shift(1) < sma_close.shift(1)
    is_exhaustion = has_gap & ((gap_up & sustained_up) | (gap_down & sustained_down))

    is_breakaway = (
        has_gap & (gap_size > breakaway_threshold * avg_range) & ~is_exhaustion
    )

    gap_type = pd.Series("", index=close.index, name="gap_type")
    gap_type = gap_type.where(~has_gap, "common")
    gap_type = gap_type.where(~is_breakaway, "breakaway")
    gap_type = gap_type.where(~is_exhaustion, "exhaustion")

    return {
        "gap_size": gap_size,
        "gap_direction": gap_direction,
        "gap_type": gap_type,
    }


# ---------------------------------------------------------------------------
# range_expansion
# ---------------------------------------------------------------------------


def range_expansion(
    high: pd.Series,
    low: pd.Series,
    period: int = 14,
    threshold: float = 1.5,
) -> pd.Series:
    """Detect range expansion (current range significantly above average).

    Returns True when ``(high - low) > threshold * avg_range``.

    Parameters:
        high: High prices.
        low: Low prices.
        period: Look-back for average range.
        threshold: Multiplier of average range to trigger expansion.

    Returns:
        Boolean series (True where range is expanded).

    Example:
        >>> expanded = range_expansion(high, low)
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_period(period)

    rng = high - low
    avg_rng = rng.rolling(window=period, min_periods=period).mean()
    result = rng > threshold * avg_rng
    return pd.Series(result, index=high.index, name="range_expansion", dtype=bool)


# ---------------------------------------------------------------------------
# narrow_range
# ---------------------------------------------------------------------------


def narrow_range(
    high: pd.Series,
    low: pd.Series,
    period: int = 4,
) -> pd.Series:
    """NR4/NR7 detection — narrowest range in *period* bars.

    Returns True when the current bar's range is the smallest in the last
    *period* bars (including itself).  ``period=4`` gives NR4; ``period=7``
    gives NR7.

    Parameters:
        high: High prices.
        low: Low prices.
        period: Look-back window (default 4 for NR4).

    Returns:
        Boolean series (True at narrow-range bars).

    Example:
        >>> nr4 = narrow_range(high, low, period=4)
        >>> nr7 = narrow_range(high, low, period=7)
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_period(period)

    rng = high - low
    min_rng = rng.rolling(window=period, min_periods=period).min()
    result = rng == min_rng
    return pd.Series(result, index=high.index, name="narrow_range", dtype=bool)


# ---------------------------------------------------------------------------
# wide_range_bar
# ---------------------------------------------------------------------------


def wide_range_bar(
    high: pd.Series,
    low: pd.Series,
    period: int = 14,
    threshold: float = 1.5,
) -> pd.Series:
    """Wide Range Bar (WRB): range > threshold * average range.

    Parameters:
        high: High prices.
        low: Low prices.
        period: Look-back for average range.
        threshold: Multiplier (default 1.5).

    Returns:
        Boolean series (True at wide-range bars).

    Example:
        >>> wrb = wide_range_bar(high, low)
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_period(period)

    rng = high - low
    avg_rng = rng.rolling(window=period, min_periods=period).mean()
    result = rng > threshold * avg_rng
    return pd.Series(result, index=high.index, name="wide_range_bar", dtype=bool)


# ---------------------------------------------------------------------------
# key_reversal
# ---------------------------------------------------------------------------


def key_reversal(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Detect key reversal bars.

    A **bullish key reversal** (1) makes a new low (below the previous low)
    but closes above the previous close.

    A **bearish key reversal** (-1) makes a new high (above the previous
    high) but closes below the previous close.

    Parameters:
        open_: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.

    Returns:
        1 (bullish key reversal), -1 (bearish key reversal), or 0.

    Example:
        >>> kr = key_reversal(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    # Bullish: new low but closes above previous close
    bullish = (low < prev_low) & (close > prev_close)

    # Bearish: new high but closes below previous close
    bearish = (high > prev_high) & (close < prev_close)

    result = np.where(bullish, 1, np.where(bearish, -1, 0))
    return pd.Series(result, index=close.index, name="key_reversal", dtype=int)


# ---------------------------------------------------------------------------
# pivot_reversal
# ---------------------------------------------------------------------------


def pivot_reversal(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Detect two-bar pivot reversal patterns at swing points.

    A **bullish pivot reversal** (1): the previous bar makes a lower low
    than its predecessor, and the current bar closes above the previous
    bar's high.

    A **bearish pivot reversal** (-1): the previous bar makes a higher high
    than its predecessor, and the current bar closes below the previous
    bar's low.

    Parameters:
        open_: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.

    Returns:
        1 (bullish pivot reversal), -1 (bearish pivot reversal), or 0.

    Example:
        >>> pr = pivot_reversal(open_, high, low, close)
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    # Previous bar made a lower low (swing low candidate)
    prev_lower_low = low.shift(1) < low.shift(2)
    # Previous bar made a higher high (swing high candidate)
    prev_higher_high = high.shift(1) > high.shift(2)

    # Current bar closes above previous high (bullish reversal)
    bullish = prev_lower_low & (close > high.shift(1))

    # Current bar closes below previous low (bearish reversal)
    bearish = prev_higher_high & (close < low.shift(1))

    result = np.where(bullish, 1, np.where(bearish, -1, 0))
    return pd.Series(result, index=close.index, name="pivot_reversal", dtype=int)
