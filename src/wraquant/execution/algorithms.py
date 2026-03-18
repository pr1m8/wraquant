"""Execution algorithm schedules.

Provides scheduling functions for common execution algorithms:
TWAP, VWAP, participation-of-volume (POV), and benchmark analytics
such as implementation shortfall and arrival price cost.
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

    Splits *total_qty* evenly across *n_intervals*.

    Parameters:
        total_qty: Total quantity to execute.
        n_intervals: Number of time intervals.

    Returns:
        Array of length *n_intervals* with equal quantities.
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

    Allocates *total_qty* proportionally to the historical volume
    profile so that execution tracks the expected VWAP.

    Parameters:
        total_qty: Total quantity to execute.
        historical_volume_profile: Expected volume in each interval.
            The shape determines the number of intervals.

    Returns:
        Quantity to execute in each interval.
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

    Decomposes execution cost into delay cost, trading impact, and
    opportunity cost relative to the decision and benchmark prices.

    Parameters:
        execution_prices: Price of each fill.
        decision_price: Price at the time the decision was made.
        benchmark_price: Closing or end-of-day benchmark price.
        quantities: Quantity per fill. If *None*, assumed uniform.

    Returns:
        Dictionary with ``'total_is'``, ``'delay_cost'``,
        ``'trading_impact'``, and ``'opportunity_cost'``.
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

    In each interval, the algorithm participates at *target_rate* of the
    expected market volume, up to the remaining quantity.

    Parameters:
        total_qty: Total quantity to execute.
        target_rate: Target participation rate in (0, 1].
        expected_volume: Expected market volume per interval.

    Returns:
        Quantity to execute in each interval (may not fully exhaust
        *total_qty* if cumulative expected volume is too low).
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

    Measures execution quality relative to the price at the time the
    order was submitted (arrival price).

    Parameters:
        execution_prices: Price per fill.
        volumes: Volume per fill.
        arrival_price: Market price when the order arrived.

    Returns:
        Dictionary with ``'vwap'``, ``'arrival_cost'``, and
        ``'arrival_cost_bps'``.
    """
    exec_p = np.asarray(execution_prices, dtype=np.float64)
    vol = np.asarray(volumes, dtype=np.float64)

    total_vol = np.sum(vol)
    if total_vol <= 0:
        raise ValueError("volumes must have positive total")

    vwap = float(np.sum(exec_p * vol) / total_vol)
    arrival_cost = vwap - arrival_price
    arrival_cost_bps = (arrival_cost / arrival_price) * 10_000 if arrival_price != 0 else np.nan

    return {
        "vwap": vwap,
        "arrival_cost": float(arrival_cost),
        "arrival_cost_bps": float(arrival_cost_bps),
    }
