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


def adaptive_schedule(
    total_quantity: float,
    market_volumes: pd.Series | NDArray[np.floating],
    spread_series: pd.Series | NDArray[np.floating],
    urgency: float = 0.5,
) -> NDArray[np.floating]:
    """Adapt execution schedule based on real-time microstructure signals.

    Uses bid-ask spread and volume from the microstructure module to
    dynamically adjust VWAP/TWAP schedules.  The core idea: trade more
    aggressively when spreads are tight and volume is high (low execution
    cost), and slow down when spreads widen or volume dries up.

    This bridges ``execution`` and ``microstructure`` -- the schedule
    responds to liquidity conditions rather than following a static plan.

    The allocation for each interval is proportional to:

        score_i = volume_i * (1 / spread_i)^urgency

    where *urgency* controls how much the schedule responds to spread
    changes.  Higher urgency means the algorithm shifts more volume
    into tight-spread intervals.

    Parameters:
        total_quantity: Total quantity to execute (positive for buys).
        market_volumes: Expected or observed market volume per interval.
            Shape determines the number of intervals.
        spread_series: Bid-ask spread (or effective spread) per interval.
            Must have the same length as *market_volumes*.  Wider
            spreads cause the algorithm to defer volume.
        urgency: Urgency parameter in [0, 1].  Controls the trade-off
            between execution speed and cost minimisation:

            - ``0.0`` -- ignore spreads entirely (pure VWAP schedule).
            - ``0.5`` -- balanced (default).  Moderate reallocation.
            - ``1.0`` -- maximum spread sensitivity.  Concentrates
              volume in the tightest-spread intervals.

    Returns:
        Quantity to execute in each interval.  Sum equals
        *total_quantity*.  Intervals with wider spreads receive less
        volume; intervals with higher volume and tighter spreads
        receive more.

    Example:
        >>> import numpy as np
        >>> volumes = np.array([1000, 2000, 1500, 3000, 2500])
        >>> spreads = np.array([0.05, 0.02, 0.08, 0.03, 0.04])
        >>> schedule = adaptive_schedule(10_000, volumes, spreads, urgency=0.5)
        >>> np.isclose(schedule.sum(), 10_000)
        True
        >>> schedule[1] > schedule[2]  # more volume when spread is tight
        True

    Notes:
        In production, feed live spread and volume data from
        ``wraquant.microstructure.liquidity`` to dynamically re-plan
        the remaining execution at each interval.

    See Also:
        vwap_schedule: Static volume-proportional scheduling.
        twap_schedule: Equal-time scheduling.
        wraquant.microstructure.liquidity.effective_spread: Spread input.
    """
    if not 0 <= urgency <= 1:
        raise ValueError("urgency must be in [0, 1]")

    vol = np.asarray(market_volumes, dtype=np.float64)
    spr = np.asarray(spread_series, dtype=np.float64)

    if len(vol) != len(spr):
        raise ValueError(
            f"market_volumes ({len(vol)}) and spread_series ({len(spr)}) "
            "must have the same length"
        )

    # Avoid division by zero in spread
    spr_safe = np.maximum(spr, 1e-10)

    # Score: volume * inverse-spread^urgency
    inv_spread_score = (1.0 / spr_safe) ** urgency
    scores = vol * inv_spread_score

    total_score = np.sum(scores)
    if total_score <= 0:
        # Fallback to uniform
        return np.full(len(vol), total_quantity / len(vol), dtype=np.float64)

    weights = scores / total_score
    return total_quantity * weights


def is_schedule(
    total_quantity: float,
    market_volumes: pd.Series | NDArray[np.floating],
    alpha: float = 0.5,
) -> NDArray[np.floating]:
    """Implementation Shortfall (IS) optimal schedule.

    Use this when you want to balance execution urgency against market
    impact, following the simplified Almgren-Chriss intuition.  The
    *alpha* parameter controls the trade-off: higher alpha front-loads
    execution (reduces timing risk but increases impact); lower alpha
    spreads execution more evenly (reduces impact but increases timing
    risk).

    The schedule allocates volume proportionally to a blend of uniform
    (TWAP-like) and volume-proportional (VWAP-like) components, with
    *alpha* controlling the blend:

        allocation_i = alpha * (1/N) + (1 - alpha) * (V_i / sum(V))

    Parameters:
        total_quantity: Total quantity to execute (positive for buys).
        market_volumes: Expected market volume per interval.  Shape
            determines the number of intervals.
        alpha: Urgency parameter in [0, 1].  Controls the balance
            between front-loading (high alpha, TWAP-like urgency) and
            volume-tracking (low alpha, VWAP-like patience):

            - ``0.0`` -- pure VWAP (follow volume exactly).
            - ``0.5`` -- balanced (default).
            - ``1.0`` -- pure TWAP (ignore volume, equal slices).

    Returns:
        Quantity to execute in each interval.  Sum equals
        *total_quantity*.

    Example:
        >>> import numpy as np
        >>> volumes = np.array([1000, 2000, 1500, 3000, 2500])
        >>> schedule = is_schedule(10_000, volumes, alpha=0.5)
        >>> np.isclose(schedule.sum(), 10_000)
        True
        >>> schedule_urgent = is_schedule(10_000, volumes, alpha=0.9)
        >>> np.std(schedule_urgent) < np.std(is_schedule(10_000, volumes, alpha=0.1))
        True

    References:
        - Almgren & Chriss (2001), "Optimal Execution of Portfolio
          Transactions"

    See Also:
        twap_schedule: Pure time-weighted scheduling.
        vwap_schedule: Pure volume-weighted scheduling.
        adaptive_schedule: Spread-aware dynamic scheduling.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")

    vol = np.asarray(market_volumes, dtype=np.float64)
    n = len(vol)

    # Uniform (TWAP) component
    uniform = np.ones(n, dtype=np.float64) / n

    # Volume-proportional (VWAP) component
    total_vol = np.sum(vol)
    if total_vol <= 0:
        vol_weights = uniform.copy()
    else:
        vol_weights = vol / total_vol

    # Blend
    blended = alpha * uniform + (1 - alpha) * vol_weights
    blended = blended / np.sum(blended)  # normalise

    return total_quantity * blended


def pov_schedule(
    total_quantity: float,
    market_volumes: pd.Series | NDArray[np.floating],
    pov_rate: float = 0.1,
) -> NDArray[np.floating]:
    """Percentage of Volume (POV) schedule with constant participation.

    Use this when you need to maintain a constant participation rate
    across all intervals.  Unlike :func:`participation_rate_schedule`
    which caps at *total_quantity*, this function explicitly models
    constant-rate participation and returns the quantity per interval.

    Each interval's allocation is ``pov_rate * market_volume_i``, with
    the constraint that the cumulative allocation does not exceed
    *total_quantity*.

    Parameters:
        total_quantity: Total quantity to execute.
        market_volumes: Expected market volume per interval.
        pov_rate: Target participation rate in (0, 1].  Common values:

            - ``0.05`` -- 5% (very passive, minimal impact).
            - ``0.10`` -- 10% (standard institutional).
            - ``0.20`` -- 20% (moderately aggressive).
            - ``0.30``+ -- aggressive (significant impact risk).

    Returns:
        Quantity to execute in each interval.  Each entry is at most
        ``pov_rate * market_volume_i``.  Total may be less than
        *total_quantity* if cumulative volume is insufficient.

    Example:
        >>> import numpy as np
        >>> volumes = np.array([5000, 8000, 6000, 10000, 7000])
        >>> schedule = pov_schedule(2000, volumes, pov_rate=0.10)
        >>> schedule[0]  # 10% of 5000
        500.0
        >>> schedule.sum() <= 2000
        True

    See Also:
        participation_rate_schedule: Similar but from the algorithms module.
        is_schedule: Urgency-based IS schedule.
    """
    if not 0 < pov_rate <= 1:
        raise ValueError("pov_rate must be in (0, 1]")

    vol = np.asarray(market_volumes, dtype=np.float64)
    schedule = np.zeros_like(vol)
    remaining = total_quantity

    for i in range(len(vol)):
        desired = pov_rate * vol[i]
        fill = min(desired, remaining)
        schedule[i] = fill
        remaining -= fill
        if remaining <= 0:
            break

    return schedule


def close_auction_allocation(
    total_quantity: float,
    historical_close_volume_pct: float = 0.2,
) -> dict[str, float]:
    """Reserve a portion of the order for the closing auction.

    Use this when you want to participate in the closing auction
    (MOC -- Market on Close) to benefit from the closing price, which
    is the most common benchmark for fund NAV calculations and index
    rebalancing.

    Splits the order into a continuous-market portion (to be executed
    throughout the day using TWAP/VWAP/IS) and a closing-auction
    portion.  The close allocation is based on the historical fraction
    of daily volume that occurs at the close.

    Parameters:
        total_quantity: Total quantity to execute.
        historical_close_volume_pct: Historical fraction of daily volume
            that trades at the close (default 0.20 = 20%).  This varies
            by market:

            - US equities: ~7-15% (higher on rebalance days).
            - European equities: ~15-25%.
            - Index constituents on rebalance: ~30-50%.

    Returns:
        Dictionary containing:

        - **continuous_quantity** (*float*) -- Quantity to execute during
          the continuous trading session.
        - **close_quantity** (*float*) -- Quantity reserved for the
          closing auction.
        - **close_pct** (*float*) -- Fraction allocated to close.

    Example:
        >>> result = close_auction_allocation(10_000, historical_close_volume_pct=0.15)
        >>> result['close_quantity']
        1500.0
        >>> result['continuous_quantity']
        8500.0

    See Also:
        vwap_schedule: Schedule the continuous portion.
        is_schedule: Schedule with urgency control.
    """
    if not 0 <= historical_close_volume_pct <= 1:
        raise ValueError("historical_close_volume_pct must be in [0, 1]")

    close_qty = total_quantity * historical_close_volume_pct
    continuous_qty = total_quantity - close_qty

    return {
        "continuous_quantity": float(continuous_qty),
        "close_quantity": float(close_qty),
        "close_pct": float(historical_close_volume_pct),
    }
