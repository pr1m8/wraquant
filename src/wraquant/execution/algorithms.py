"""Execution algorithm schedules and transaction cost analysis.

Execution algorithms control *how* a large order is sliced into smaller
child orders and distributed across time to minimise market impact and
execution cost. This module provides scheduling logic for the three
most common algorithms and tools for measuring execution quality.

Algorithms:
    - ``twap_schedule``: Time-Weighted Average Price. Splits the order
      evenly across intervals. The simplest algorithm; minimises timing
      risk but ignores volume patterns. Use for low-urgency orders in
      liquid markets.

    - ``vwap_schedule``: Volume-Weighted Average Price. Allocates
      proportionally to expected volume. Tracks the market's VWAP,
      which is the most common benchmark for institutional execution.
      Use when you want to trade "at the market's pace."

    - ``participation_rate_schedule``: Participation-of-Volume (POV).
      Targets a fixed fraction of market volume in each interval. Use
      when you want to limit market impact to a known participation
      rate (e.g., "never trade more than 10% of volume").

Execution quality analytics:
    - ``implementation_shortfall``: decomposes total execution cost
      into delay cost (waiting), trading impact (price movement during
      execution), and opportunity cost (unexecuted portion). The most
      comprehensive cost metric.

    - ``arrival_price_benchmark``: measures execution cost relative to
      the price when the order arrived. Simpler than IS but widely
      used for benchmarking.

How to choose:
    - **Low urgency, liquid market**: TWAP or VWAP.
    - **Volume-sensitive market**: VWAP (tracks volume profile).
    - **Need to limit participation**: POV.
    - **High urgency**: aggressive POV with high target rate.
    - **Measuring execution quality**: implementation shortfall for
      attribution, arrival price for quick benchmarking.

References:
    - Almgren & Chriss (2001), "Optimal Execution of Portfolio
      Transactions"
    - Perold (1988), "The Implementation Shortfall"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def twap_schedule(
    total_qty: float,
    n_intervals: int,
) -> NDArray[np.floating]:
    """Time-weighted average price (TWAP) schedule.

    Use TWAP for low-urgency orders in liquid markets where you want to
    minimise timing risk by spreading execution evenly over time.  TWAP
    is the simplest algorithm and is often used as a benchmark for more
    sophisticated strategies.

    Splits *total_qty* evenly across *n_intervals*.

    Parameters:
        total_qty: Total quantity to execute (positive for buys).
        n_intervals: Number of time intervals (e.g., 78 for 5-min
            bars in a 6.5-hour trading day).

    Returns:
        Array of length *n_intervals* with equal quantities.  Each
        element is ``total_qty / n_intervals``.

    Example:
        >>> schedule = twap_schedule(10_000, n_intervals=10)
        >>> schedule[0]
        1000.0
        >>> len(schedule)
        10

    See Also:
        vwap_schedule: Volume-proportional scheduling.
        participation_rate_schedule: Participation-of-volume scheduling.
    """
    if n_intervals <= 0:
        raise ValueError("n_intervals must be positive")
    qty_per_interval = total_qty / n_intervals
    return np.full(n_intervals, qty_per_interval, dtype=np.float64)


def vwap_schedule(
    total_qty: float,
    historical_volume_profile: pd.Series | NDArray[np.floating],
) -> NDArray[np.floating]:
    """Volume-weighted average price (VWAP) schedule.

    Use VWAP when you want to trade "at the market's pace" and your
    benchmark is the market VWAP.  Allocating proportionally to
    historical volume minimises deviation from VWAP and is the most
    common institutional execution benchmark.

    Allocates *total_qty* proportionally to the historical volume
    profile so that execution tracks the expected VWAP.

    Parameters:
        total_qty: Total quantity to execute.
        historical_volume_profile: Expected volume in each interval
            (e.g., average intraday volume by 5-min bucket).
            The shape determines the number of intervals.

    Returns:
        Quantity to execute in each interval, proportional to expected
        volume.  Sum equals *total_qty*.

    Example:
        >>> import numpy as np
        >>> # U-shaped intraday volume profile (high at open/close)
        >>> volume = np.array([500, 300, 200, 200, 300, 500])
        >>> schedule = vwap_schedule(10_000, volume)
        >>> schedule[0]  # most volume at open
        2500.0
        >>> np.isclose(schedule.sum(), 10_000)
        True

    See Also:
        twap_schedule: Equal-time scheduling (ignores volume).
        participation_rate_schedule: Fixed participation rate.
    """
    profile = np.asarray(historical_volume_profile, dtype=np.float64)
    total_vol = np.sum(profile)
    if total_vol <= 0:
        raise ValueError("historical_volume_profile must have positive total")
    weights = profile / total_vol
    return total_qty * weights


def implementation_shortfall(
    execution_prices: pd.Series | NDArray[np.floating],
    decision_price: float,
    benchmark_price: float,
    quantities: pd.Series | NDArray[np.floating] | None = None,
) -> dict[str, float]:
    """Implementation shortfall decomposition.

    Use implementation shortfall (IS) for the most comprehensive
    attribution of execution costs.  IS decomposes the gap between the
    paper portfolio (executed at the decision price) and the actual
    portfolio into three components, allowing you to identify whether
    costs come from delay, market impact, or missed opportunities.

    Decomposes execution cost into delay cost, trading impact, and
    opportunity cost relative to the decision and benchmark prices.

    Parameters:
        execution_prices: Price of each fill (chronologically ordered).
        decision_price: Price at the time the decision was made (e.g.,
            previous close or signal generation price).
        benchmark_price: Closing or end-of-day benchmark price.
        quantities: Quantity per fill. If *None*, assumed uniform across
            fills.

    Returns:
        Dictionary with:

        - ``'total_is'``: Total implementation shortfall (avg exec
          price - decision price).  Positive = paid more than decision.
        - ``'delay_cost'``: Cost from waiting between decision and
          first execution (first fill - decision price).
        - ``'trading_impact'``: Cost from executing in the market
          (avg exec price - first fill price).
        - ``'opportunity_cost'``: Benchmark price vs decision price;
          the cost of not executing instantly at the decision price.

    Example:
        >>> import numpy as np
        >>> fills = np.array([100.10, 100.15, 100.20, 100.18])
        >>> result = implementation_shortfall(fills, decision_price=100.0,
        ...                                  benchmark_price=100.25)
        >>> result['total_is'] > 0  # paid more than decision price
        True
        >>> result['delay_cost']  # first fill - decision
        0.1

    References:
        - Perold (1988), "The Implementation Shortfall"

    See Also:
        arrival_price_benchmark: Simpler benchmark-relative cost.
    """
    exec_p = np.asarray(execution_prices, dtype=np.float64)
    n = len(exec_p)

    if quantities is None:
        qty = np.ones(n, dtype=np.float64) / n
    else:
        qty = np.asarray(quantities, dtype=np.float64)
        qty = qty / np.sum(qty)

    avg_exec_price = float(np.sum(exec_p * qty))

    # Delay cost: cost of waiting between decision and first execution
    delay_cost = exec_p[0] - decision_price

    # Trading impact: cost from execution relative to arrival
    trading_impact = avg_exec_price - exec_p[0]

    # Opportunity cost: benchmark vs decision for unexecuted portion
    opportunity_cost = benchmark_price - decision_price

    # Total IS: avg execution price vs decision price
    total_is = avg_exec_price - decision_price

    return {
        "total_is": float(total_is),
        "delay_cost": float(delay_cost),
        "trading_impact": float(trading_impact),
        "opportunity_cost": float(opportunity_cost),
    }


def participation_rate_schedule(
    total_qty: float,
    target_rate: float,
    expected_volume: pd.Series | NDArray[np.floating],
) -> NDArray[np.floating]:
    """Participation-of-volume (POV) execution schedule.

    Use POV when you need to limit your market footprint to a known
    fraction of market volume.  This is common for large institutional
    orders where exceeding 10-20% of volume would cause excessive
    market impact.

    In each interval, the algorithm participates at *target_rate* of the
    expected market volume, up to the remaining quantity.

    Parameters:
        total_qty: Total quantity to execute.
        target_rate: Target participation rate in (0, 1].  For example,
            0.10 means "trade no more than 10% of market volume per
            interval."
        expected_volume: Expected market volume per interval.

    Returns:
        Quantity to execute in each interval (may not fully exhaust
        *total_qty* if cumulative expected volume is too low).

    Example:
        >>> import numpy as np
        >>> volume = np.array([1000, 2000, 1500, 3000, 2500])
        >>> schedule = participation_rate_schedule(500, 0.10, volume)
        >>> schedule[0]  # 10% of 1000
        100.0
        >>> schedule.sum() <= 500
        True

    See Also:
        twap_schedule: Time-based scheduling.
        vwap_schedule: Volume-proportional scheduling.
    """
    if not 0 < target_rate <= 1:
        raise ValueError("target_rate must be in (0, 1]")
    vol = np.asarray(expected_volume, dtype=np.float64)
    schedule = np.zeros_like(vol)
    remaining = total_qty

    for i in range(len(vol)):
        desired = target_rate * vol[i]
        fill = min(desired, remaining)
        schedule[i] = fill
        remaining -= fill
        if remaining <= 0:
            break

    return schedule


def arrival_price_benchmark(
    execution_prices: pd.Series | NDArray[np.floating],
    volumes: pd.Series | NDArray[np.floating],
    arrival_price: float,
) -> dict[str, float]:
    """Arrival price cost analysis.

    Use arrival price benchmarking for a quick, intuitive measure of
    how much your execution cost relative to the market price when the
    order was submitted.  Simpler than full implementation shortfall
    but widely used in practice.

    Measures execution quality relative to the price at the time the
    order was submitted (arrival price).

    Parameters:
        execution_prices: Price per fill.
        volumes: Volume per fill.
        arrival_price: Market price when the order arrived (typically
            mid-quote at time of order entry).

    Returns:
        Dictionary with:

        - ``'vwap'``: Volume-weighted average execution price.
        - ``'arrival_cost'``: VWAP minus arrival price.  Positive
          means you paid more than the arrival price (slippage).
        - ``'arrival_cost_bps'``: Arrival cost in basis points
          (10,000 bps = 100%).  Typical equity execution costs
          are 5-30 bps.

    Example:
        >>> import numpy as np
        >>> prices = np.array([100.05, 100.10, 100.08])
        >>> volumes = np.array([1000, 2000, 1500])
        >>> result = arrival_price_benchmark(prices, volumes, arrival_price=100.0)
        >>> result['arrival_cost_bps'] > 0  # paid more than arrival
        True

    See Also:
        implementation_shortfall: Full cost attribution.
    """
    exec_p = np.asarray(execution_prices, dtype=np.float64)
    vol = np.asarray(volumes, dtype=np.float64)

    total_vol = np.sum(vol)
    if total_vol <= 0:
        raise ValueError("volumes must have positive total")

    vwap = float(np.sum(exec_p * vol) / total_vol)
    arrival_cost = vwap - arrival_price
    arrival_cost_bps = (
        (arrival_cost / arrival_price) * 10_000 if arrival_price != 0 else np.nan
    )

    return {
        "vwap": vwap,
        "arrival_cost": float(arrival_cost),
        "arrival_cost_bps": float(arrival_cost_bps),
    }
