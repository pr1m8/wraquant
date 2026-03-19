"""Support and resistance detection indicators.

This module provides tools for identifying horizontal support/resistance
levels, fractals, supply/demand zones, and trendlines from price data.
All functions accept ``pd.Series`` inputs and return ``dict``, ``list``,
or ``pd.Series`` outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "find_support_resistance",
    "price_clustering",
    "fractal_levels",
    "round_number_levels",
    "supply_demand_zones",
    "trendline_detection",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


from wraquant.ta._validators import validate_period as _validate_period
from wraquant.ta._validators import validate_series as _validate_series

# ---------------------------------------------------------------------------
# Find Support / Resistance
# ---------------------------------------------------------------------------


def find_support_resistance(
    high: pd.Series,
    low: pd.Series,
    lookback: int = 5,
    num_levels: int = 5,
    tolerance: float = 0.02,
) -> dict[str, list[float]]:
    """Detect horizontal support and resistance levels via clustering.

    Identifies local swing highs and swing lows, then clusters nearby
    levels within *tolerance* (as a fraction of price) to produce
    consolidated S/R levels.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    lookback : int, default 5
        Number of bars on each side to confirm a swing point.
    num_levels : int, default 5
        Maximum number of S/R levels to return per side.
    tolerance : float, default 0.02
        Fraction of price within which nearby levels are merged.

    Returns
    -------
    dict[str, list[float]]
        ``support`` and ``resistance`` lists of price levels,
        sorted ascending.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> h = pd.Series(100 + np.cumsum(np.random.randn(200) * 0.5) + 1)
    >>> lo = h - 2
    >>> result = find_support_resistance(h, lo)  # doctest: +SKIP
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_period(lookback, "lookback")

    n = len(high)
    h_vals = high.values.astype(float)
    l_vals = low.values.astype(float)

    swing_highs: list[float] = []
    swing_lows: list[float] = []

    for i in range(lookback, n - lookback):
        left_h = h_vals[i - lookback : i]
        right_h = h_vals[i + 1 : i + lookback + 1]
        if h_vals[i] >= np.max(left_h) and h_vals[i] >= np.max(right_h):
            swing_highs.append(float(h_vals[i]))

        left_l = l_vals[i - lookback : i]
        right_l = l_vals[i + 1 : i + lookback + 1]
        if l_vals[i] <= np.min(left_l) and l_vals[i] <= np.min(right_l):
            swing_lows.append(float(l_vals[i]))

    def _cluster(levels: list[float], max_levels: int) -> list[float]:
        if not levels:
            return []
        sorted_levels = sorted(levels)
        clusters: list[list[float]] = [[sorted_levels[0]]]
        for lvl in sorted_levels[1:]:
            cluster_mean = np.mean(clusters[-1])
            if abs(lvl - cluster_mean) / max(cluster_mean, 1e-10) <= tolerance:
                clusters[-1].append(lvl)
            else:
                clusters.append([lvl])

        # Sort clusters by number of touches (descending), take top N
        clusters.sort(key=len, reverse=True)
        result = sorted(float(np.mean(c)) for c in clusters[:max_levels])
        return result

    return {
        "support": _cluster(swing_lows, num_levels),
        "resistance": _cluster(swing_highs, num_levels),
    }


# ---------------------------------------------------------------------------
# Price Clustering
# ---------------------------------------------------------------------------


def price_clustering(
    data: pd.Series,
    num_levels: int = 5,
    bins: int = 100,
) -> np.ndarray:
    """Find price levels where price has spent the most time.

    Builds a histogram of price values and returns the bin centres
    with the highest counts, analogous to a simplified volume profile.

    Parameters
    ----------
    data : pd.Series
        Price series (typically close).
    num_levels : int, default 5
        Number of key price levels to return.
    bins : int, default 100
        Number of histogram bins.

    Returns
    -------
    np.ndarray
        Array of key price levels sorted ascending.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> close = pd.Series([100, 101, 100, 99, 100, 101, 102, 100], dtype=float)
    >>> price_clustering(close, num_levels=3)  # doctest: +SKIP
    """
    _validate_series(data)

    values = data.dropna().values.astype(float)
    if len(values) == 0:
        return np.array([])

    counts, bin_edges = np.histogram(values, bins=bins)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    top_indices = np.argsort(counts)[::-1][:num_levels]
    key_levels = np.sort(bin_centres[top_indices])

    return key_levels


# ---------------------------------------------------------------------------
# Fractal Levels (Williams Fractals)
# ---------------------------------------------------------------------------


def fractal_levels(
    high: pd.Series,
    low: pd.Series,
    period: int = 2,
) -> dict[str, pd.Series]:
    """Williams fractal swing high/low identification.

    An up-fractal occurs when the high is the highest of
    ``2 * period + 1`` bars. A down-fractal occurs when the low
    is the lowest.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    period : int, default 2
        Number of bars on each side of the fractal pivot.

    Returns
    -------
    dict[str, pd.Series]
        ``up_fractals`` (boolean Series, True at up-fractal bars),
        ``down_fractals`` (boolean Series, True at down-fractal bars).

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> h = pd.Series(100 + np.cumsum(np.random.randn(50) * 0.5) + 1)
    >>> lo = h - 2
    >>> result = fractal_levels(h, lo, period=2)  # doctest: +SKIP
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_period(period, "period")

    n = len(high)
    h_vals = high.values.astype(float)
    l_vals = low.values.astype(float)

    up_fractals = np.zeros(n, dtype=bool)
    down_fractals = np.zeros(n, dtype=bool)

    for i in range(period, n - period):
        left_h = h_vals[i - period : i]
        right_h = h_vals[i + 1 : i + period + 1]
        if h_vals[i] > np.max(left_h) and h_vals[i] > np.max(right_h):
            up_fractals[i] = True

        left_l = l_vals[i - period : i]
        right_l = l_vals[i + 1 : i + period + 1]
        if l_vals[i] < np.min(left_l) and l_vals[i] < np.min(right_l):
            down_fractals[i] = True

    return {
        "up_fractals": pd.Series(up_fractals, index=high.index, name="up_fractals"),
        "down_fractals": pd.Series(
            down_fractals, index=low.index, name="down_fractals"
        ),
    }


# ---------------------------------------------------------------------------
# Round Number Levels
# ---------------------------------------------------------------------------


def round_number_levels(
    current_price: float,
    num_levels: int = 5,
    step: float | None = None,
) -> list[float]:
    """Generate psychological round number levels near the current price.

    Computes evenly spaced round numbers above and below the given
    price. Useful for identifying potential support/resistance at
    psychologically significant prices.

    Parameters
    ----------
    current_price : float
        The current (or reference) price.
    num_levels : int, default 5
        Number of levels to return on each side (above and below).
    step : float or None, default None
        Step size between levels. If ``None``, automatically determined
        from the magnitude of *current_price*.

    Returns
    -------
    list[float]
        Sorted list of round number price levels.

    Example
    -------
    >>> round_number_levels(105.3, num_levels=3, step=10.0)
    [80.0, 90.0, 100.0, 110.0, 120.0, 130.0]
    """
    if current_price <= 0:
        raise ValueError(f"current_price must be > 0, got {current_price}")

    if step is None:
        magnitude = 10 ** (int(np.log10(max(current_price, 1))) - 1)
        step = float(max(magnitude, 1.0))

    # Find the nearest round number below
    base = np.floor(current_price / step) * step

    levels: list[float] = []
    for i in range(-num_levels, num_levels + 1):
        level = base + i * step
        if level > 0:
            levels.append(float(round(level, 10)))

    return sorted(levels)


# ---------------------------------------------------------------------------
# Supply / Demand Zones
# ---------------------------------------------------------------------------


def supply_demand_zones(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    body_pct: float = 0.6,
    consolidation_bars: int = 3,
) -> dict[str, list[dict[str, float]]]:
    """Detect supply and demand zones from price action.

    A demand zone forms when a large bullish candle follows a period
    of basing (small bodies). A supply zone forms similarly for
    bearish moves. Zones are defined by the basing candles' range.

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
    body_pct : float, default 0.6
        Minimum body-to-range ratio to qualify as a "large" candle.
    consolidation_bars : int, default 3
        Number of preceding small-body bars required for basing.

    Returns
    -------
    dict[str, list[dict[str, float]]]
        ``demand`` and ``supply`` lists, each containing dicts with
        ``zone_low``, ``zone_high``, and ``index`` keys.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> n = 100
    >>> c = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5))
    >>> o = c.shift(1).fillna(c.iloc[0])
    >>> h = pd.concat([o, c], axis=1).max(axis=1) + 0.5
    >>> lo = pd.concat([o, c], axis=1).min(axis=1) - 0.5
    >>> result = supply_demand_zones(o, h, lo, c)  # doctest: +SKIP
    """
    _validate_series(open_, "open_")
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")

    n = len(close)
    o_vals = open_.values.astype(float)
    h_vals = high.values.astype(float)
    l_vals = low.values.astype(float)
    c_vals = close.values.astype(float)

    body = np.abs(c_vals - o_vals)
    candle_range = h_vals - l_vals
    # Avoid division by zero
    safe_range = np.where(candle_range > 0, candle_range, np.nan)
    body_ratio = body / safe_range

    # Median body size for classifying small candles
    median_body = np.nanmedian(body)

    demand_zones: list[dict[str, float]] = []
    supply_zones: list[dict[str, float]] = []

    for i in range(consolidation_bars + 1, n):
        # Check if current bar is a large candle
        if np.isnan(body_ratio[i]) or body_ratio[i] < body_pct:
            continue

        # Check if preceding bars are small (consolidation/basing)
        preceding_bodies = body[i - consolidation_bars : i]
        if np.any(np.isnan(preceding_bodies)):
            continue
        if not np.all(preceding_bodies < median_body):
            continue

        # Zone is defined by the range of the basing candles
        zone_low = float(np.min(l_vals[i - consolidation_bars : i]))
        zone_high = float(np.max(h_vals[i - consolidation_bars : i]))

        if c_vals[i] > o_vals[i]:
            # Bullish breakout -> demand zone
            demand_zones.append(
                {"zone_low": zone_low, "zone_high": zone_high, "index": float(i)}
            )
        elif c_vals[i] < o_vals[i]:
            # Bearish breakout -> supply zone
            supply_zones.append(
                {"zone_low": zone_low, "zone_high": zone_high, "index": float(i)}
            )

    return {
        "demand": demand_zones,
        "supply": supply_zones,
    }


# ---------------------------------------------------------------------------
# Trendline Detection
# ---------------------------------------------------------------------------


def trendline_detection(
    high: pd.Series,
    low: pd.Series,
    lookback: int = 5,
    min_touches: int = 2,
) -> dict[str, list[dict[str, float]]]:
    """Fit linear trendlines to swing high and swing low points.

    Detects swing points, then fits lines through the most recent
    swing highs (resistance trendline) and swing lows (support
    trendline) using least-squares regression.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    lookback : int, default 5
        Number of bars on each side for swing point detection.
    min_touches : int, default 2
        Minimum number of swing points required to fit a trendline.

    Returns
    -------
    dict[str, list[dict[str, float]]]
        ``resistance_lines`` and ``support_lines``, each containing
        dicts with ``slope``, ``intercept``, and ``num_touches`` keys.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> h = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5) + 1)
    >>> lo = h - 2
    >>> result = trendline_detection(h, lo, lookback=5)  # doctest: +SKIP
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_period(lookback, "lookback")

    n = len(high)
    h_vals = high.values.astype(float)
    l_vals = low.values.astype(float)

    swing_high_indices: list[int] = []
    swing_low_indices: list[int] = []

    for i in range(lookback, n - lookback):
        left_h = h_vals[i - lookback : i]
        right_h = h_vals[i + 1 : i + lookback + 1]
        if h_vals[i] >= np.max(left_h) and h_vals[i] >= np.max(right_h):
            swing_high_indices.append(i)

        left_l = l_vals[i - lookback : i]
        right_l = l_vals[i + 1 : i + lookback + 1]
        if l_vals[i] <= np.min(left_l) and l_vals[i] <= np.min(right_l):
            swing_low_indices.append(i)

    def _fit_line(indices: list[int], values: np.ndarray) -> list[dict[str, float]]:
        if len(indices) < min_touches:
            return []
        x = np.array(indices, dtype=float)
        y = values[indices]
        slope, intercept = np.polyfit(x, y, 1)
        return [
            {
                "slope": float(slope),
                "intercept": float(intercept),
                "num_touches": float(len(indices)),
            }
        ]

    resistance_lines = _fit_line(swing_high_indices, h_vals)
    support_lines = _fit_line(swing_low_indices, l_vals)

    return {
        "resistance_lines": resistance_lines,
        "support_lines": support_lines,
    }
