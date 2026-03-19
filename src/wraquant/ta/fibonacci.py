"""Fibonacci-based technical analysis indicators.

This module provides Fibonacci retracement, extension, fan, time zone,
pivot point, and auto-detection indicators. All functions accept
``pd.Series`` inputs and return ``dict`` or ``list`` outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "fibonacci_retracements",
    "fibonacci_extensions",
    "fibonacci_fans",
    "fibonacci_time_zones",
    "fibonacci_pivot_points",
    "auto_fibonacci",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


from wraquant.ta._validators import validate_series as _validate_series

# ---------------------------------------------------------------------------
# Fibonacci Retracements
# ---------------------------------------------------------------------------


def fibonacci_retracements(
    swing_high: float,
    swing_low: float,
    direction: str = "up",
) -> dict[str, float]:
    """Compute Fibonacci retracement levels from a swing high/low pair.

    Given a swing high and swing low, computes the standard Fibonacci
    retracement levels at 23.6%, 38.2%, 50%, 61.8%, and 78.6%.

    Parameters
    ----------
    swing_high : float
        The swing high price.
    swing_low : float
        The swing low price.
    direction : str, default "up"
        If ``"up"``, retracements are measured from the high downward
        (pullback in an uptrend). If ``"down"``, retracements are
        measured from the low upward (pullback in a downtrend).

    Returns
    -------
    dict[str, float]
        Level names (e.g. ``"23.6%"``) as keys and price values.

    Example
    -------
    >>> result = fibonacci_retracements(swing_high=110.0, swing_low=100.0)
    >>> result["50.0%"]
    105.0
    """
    if swing_high <= swing_low:
        raise ValueError(f"swing_high ({swing_high}) must be > swing_low ({swing_low})")

    diff = swing_high - swing_low
    ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    labels = ["0.0%", "23.6%", "38.2%", "50.0%", "61.8%", "78.6%", "100.0%"]

    if direction == "up":
        # Retracing down from the high
        levels = {label: swing_high - r * diff for label, r in zip(labels, ratios)}
    elif direction == "down":
        # Retracing up from the low
        levels = {label: swing_low + r * diff for label, r in zip(labels, ratios)}
    else:
        raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")

    return levels


# ---------------------------------------------------------------------------
# Fibonacci Extensions
# ---------------------------------------------------------------------------


def fibonacci_extensions(
    swing_low: float,
    swing_high: float,
    pullback_low: float,
) -> dict[str, float]:
    """Compute Fibonacci extension levels from three price points.

    Uses a swing low, swing high, and pullback low to project
    extension levels at 100%, 127.2%, 161.8%, 200%, and 261.8%.

    Parameters
    ----------
    swing_low : float
        The initial swing low price.
    swing_high : float
        The swing high price.
    pullback_low : float
        The pullback low price (retracement point).

    Returns
    -------
    dict[str, float]
        Extension level names as keys and projected price values.

    Example
    -------
    >>> result = fibonacci_extensions(100.0, 110.0, 105.0)
    >>> result["100.0%"]
    115.0
    """
    if swing_high <= swing_low:
        raise ValueError(f"swing_high ({swing_high}) must be > swing_low ({swing_low})")

    diff = swing_high - swing_low
    ratios = [1.0, 1.272, 1.618, 2.0, 2.618]
    labels = ["100.0%", "127.2%", "161.8%", "200.0%", "261.8%"]

    levels = {label: pullback_low + r * diff for label, r in zip(labels, ratios)}
    return levels


# ---------------------------------------------------------------------------
# Fibonacci Fans
# ---------------------------------------------------------------------------


def fibonacci_fans(
    pivot_x: int,
    pivot_y: float,
    target_x: int,
    target_y: float,
) -> dict[str, float]:
    """Compute Fibonacci fan line slopes from two pivot points.

    Draws fan lines from ``(pivot_x, pivot_y)`` through Fibonacci
    retracement levels of the vertical distance to
    ``(target_x, target_y)``.

    Parameters
    ----------
    pivot_x : int
        Bar index of the pivot (start) point.
    pivot_y : float
        Price at the pivot point.
    target_x : int
        Bar index of the target (end) point.
    target_y : float
        Price at the target point.

    Returns
    -------
    dict[str, float]
        Fan line labels as keys and slope values.

    Example
    -------
    >>> result = fibonacci_fans(0, 100.0, 10, 110.0)
    >>> abs(result["50.0%"] - 0.5) < 1e-10
    True
    """
    dx = target_x - pivot_x
    if dx == 0:
        raise ValueError("pivot_x and target_x must be different")

    dy = target_y - pivot_y
    ratios = {"38.2%": 0.382, "50.0%": 0.5, "61.8%": 0.618}

    slopes: dict[str, float] = {}
    for label, ratio in ratios.items():
        fan_y = pivot_y + ratio * dy
        slopes[label] = (fan_y - pivot_y) / dx

    return slopes


# ---------------------------------------------------------------------------
# Fibonacci Time Zones
# ---------------------------------------------------------------------------


def fibonacci_time_zones(
    start_index: int,
    max_index: int,
) -> list[int]:
    """Compute Fibonacci time zone indices from a start bar.

    Generates a sequence of bar indices at Fibonacci intervals
    (1, 1, 2, 3, 5, 8, 13, 21, ...) from the given start index,
    up to ``max_index``.

    Parameters
    ----------
    start_index : int
        The bar index to begin the Fibonacci time zones from.
    max_index : int
        The maximum bar index (exclusive) to generate zones up to.

    Returns
    -------
    list[int]
        List of bar indices at Fibonacci time intervals.

    Example
    -------
    >>> fibonacci_time_zones(0, 50)
    [1, 2, 3, 5, 8, 13, 21, 34]
    """
    if max_index <= start_index:
        raise ValueError(
            f"max_index ({max_index}) must be > start_index ({start_index})"
        )

    zones: list[int] = []
    a, b = 1, 1
    while start_index + a < max_index:
        idx = start_index + a
        if not zones or zones[-1] != idx:
            zones.append(idx)
        a, b = b, a + b

    return zones


# ---------------------------------------------------------------------------
# Fibonacci Pivot Points
# ---------------------------------------------------------------------------


def fibonacci_pivot_points(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> dict[str, pd.Series]:
    """Pivot points using Fibonacci ratios.

    Computes the standard pivot ``P = (H + L + C) / 3`` and derives
    support/resistance using Fibonacci ratios applied to the
    prior bar's range.

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
    dict[str, pd.Series]
        ``pivot``, ``s1``, ``s2``, ``s3``, ``r1``, ``r2``, ``r3``.

    Example
    -------
    >>> import pandas as pd
    >>> h = pd.Series([12, 13, 14, 13, 12], dtype=float)
    >>> lo = pd.Series([10, 11, 12, 11, 10], dtype=float)
    >>> c = pd.Series([11, 12, 13, 12, 11], dtype=float)
    >>> result = fibonacci_pivot_points(h, lo, c)  # doctest: +SKIP
    """
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")

    h_prev = high.shift(1)
    l_prev = low.shift(1)
    c_prev = close.shift(1)

    pp = (h_prev + l_prev + c_prev) / 3.0
    diff = h_prev - l_prev

    r1 = pp + 0.382 * diff
    r2 = pp + 0.618 * diff
    r3 = pp + 1.000 * diff
    s1 = pp - 0.382 * diff
    s2 = pp - 0.618 * diff
    s3 = pp - 1.000 * diff

    return {
        "pivot": pp.rename("fib_pivot"),
        "r1": r1.rename("fib_r1"),
        "r2": r2.rename("fib_r2"),
        "r3": r3.rename("fib_r3"),
        "s1": s1.rename("fib_s1"),
        "s2": s2.rename("fib_s2"),
        "s3": s3.rename("fib_s3"),
    }


# ---------------------------------------------------------------------------
# Auto Fibonacci
# ---------------------------------------------------------------------------


def auto_fibonacci(
    data: pd.Series,
    lookback: int = 50,
    direction: str = "up",
) -> dict[str, object]:
    """Automatically detect swing high/low and compute Fibonacci retracements.

    Scans the most recent *lookback* bars to find the highest high
    and lowest low, then computes Fibonacci retracement levels.

    Parameters
    ----------
    data : pd.Series
        Price series (typically close).
    lookback : int, default 50
        Number of recent bars to scan for swing points.
    direction : str, default "up"
        Trend direction assumption: ``"up"`` retraces from high
        downward, ``"down"`` retraces from low upward.

    Returns
    -------
    dict[str, object]
        ``swing_high`` (float), ``swing_high_idx`` (index label),
        ``swing_low`` (float), ``swing_low_idx`` (index label),
        ``levels`` (dict of retracement levels).

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> close = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5))
    >>> result = auto_fibonacci(close, lookback=30)  # doctest: +SKIP
    """
    data = _validate_series(data)
    if lookback < 2:
        raise ValueError(f"lookback must be >= 2, got {lookback}")

    window = data.iloc[-lookback:]
    swing_high_idx = window.idxmax()
    swing_low_idx = window.idxmin()
    swing_high_val = float(window.loc[swing_high_idx])
    swing_low_val = float(window.loc[swing_low_idx])

    if swing_high_val <= swing_low_val:
        # Flat price — return degenerate levels
        levels = {
            "0.0%": swing_high_val,
            "23.6%": swing_high_val,
            "38.2%": swing_high_val,
            "50.0%": swing_high_val,
            "61.8%": swing_high_val,
            "78.6%": swing_high_val,
            "100.0%": swing_high_val,
        }
    else:
        levels = fibonacci_retracements(swing_high_val, swing_low_val, direction)

    return {
        "swing_high": swing_high_val,
        "swing_high_idx": swing_high_idx,
        "swing_low": swing_low_val,
        "swing_low_idx": swing_low_idx,
        "levels": levels,
    }
