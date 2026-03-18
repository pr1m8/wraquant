"""Market microstructure and execution algorithms with wraquant.

Demonstrates liquidity metrics, toxicity measures, market quality,
and optimal execution algorithms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1000

# --- Simulate market data ---
mid_prices = 100 + np.cumsum(rng.normal(0, 0.1, n))
spreads = np.abs(rng.normal(0.05, 0.02, n))
bid = mid_prices - spreads / 2
ask = mid_prices + spreads / 2
volumes = rng.lognormal(10, 1, n).astype(int)
trade_prices = mid_prices + rng.normal(0, 0.03, n)

# --- Liquidity metrics ---
from wraquant.microstructure.liquidity import (
    amihud_illiquidity,
    kyle_lambda,
    roll_spread,
    effective_spread,
    price_impact,
    turnover_ratio,
)

returns = np.diff(mid_prices) / mid_prices[:-1]
dollar_volume = np.abs(returns) * volumes[1:]

print("=== Liquidity Metrics ===")
amihud = amihud_illiquidity(returns, volumes[1:].astype(float))
print(f"  Amihud illiquidity: {amihud:.6f}")

kyle = kyle_lambda(returns, volumes[1:].astype(float))
print(f"  Kyle's lambda: {kyle:.6f}")

roll = roll_spread(mid_prices)
print(f"  Roll spread: {roll:.4f}")

eff_spread = effective_spread(trade_prices, mid_prices)
print(f"  Mean effective spread: {np.mean(eff_spread):.4f}")

impact = price_impact(trade_prices, mid_prices, volumes.astype(float))
print(f"  Mean price impact: {np.mean(impact):.6f}")

turnover = turnover_ratio(volumes.astype(float), total_shares=1_000_000.0)
print(f"  Mean turnover ratio: {np.mean(turnover):.6f}")

# --- Toxicity metrics ---
from wraquant.microstructure.toxicity import (
    vpin,
    order_flow_imbalance,
    trade_classification,
)

print(f"\n=== Toxicity Metrics ===")
vpin_result = vpin(trade_prices, volumes.astype(float), n_buckets=50)
print(f"  VPIN (last bucket): {vpin_result[-1]:.4f}")
print(f"  VPIN range: [{vpin_result.min():.4f}, {vpin_result.max():.4f}]")

ofi = order_flow_imbalance(bid, ask, volumes.astype(float))
print(f"  Order flow imbalance std: {ofi.std():.4f}")

# Lee-Ready trade classification
signs = trade_classification(trade_prices, mid_prices)
print(f"  Buy-initiated: {(signs > 0).sum()}, Sell-initiated: {(signs < 0).sum()}")

# --- Market quality ---
from wraquant.microstructure.market_quality import (
    quoted_spread,
    relative_spread,
    variance_ratio,
)

print(f"\n=== Market Quality ===")
qs = quoted_spread(bid, ask)
print(f"  Mean quoted spread: {np.mean(qs):.4f}")

rs = relative_spread(bid, ask)
print(f"  Mean relative spread: {np.mean(rs):.6f}")

prices_series = pd.Series(mid_prices)
vr = variance_ratio(prices_series, short_period=5, long_period=20)
print(f"  Variance ratio (5/20): {vr['vr']:.4f} (z={vr['z_stat']:.2f}, p={vr['p_value']:.4f})")

# --- Execution algorithms ---
from wraquant.execution.algorithms import (
    twap_schedule,
    vwap_schedule,
    participation_rate_schedule,
)

print(f"\n=== Execution Algorithms ===")
total_qty = 10_000

twap = twap_schedule(total_quantity=total_qty, n_periods=10)
print(f"  TWAP schedule (10 periods): {twap}")

hist_volumes = rng.lognormal(8, 0.5, 10)
vwap_sched = vwap_schedule(total_quantity=total_qty, historical_volumes=hist_volumes)
print(f"  VWAP schedule: {np.round(vwap_sched).astype(int)}")

participation = participation_rate_schedule(
    total_quantity=total_qty, market_volumes=hist_volumes, participation_rate=0.1,
)
print(f"  Participation (10%): {np.round(participation).astype(int)}")

# --- Optimal execution ---
from wraquant.execution.optimal import almgren_chriss

print(f"\n=== Almgren-Chriss Optimal Execution ===")
ac = almgren_chriss(
    total_shares=total_qty, n_periods=10,
    volatility=0.02, permanent_impact=1e-5, temporary_impact=1e-4,
    risk_aversion=1e-3,
)
print(f"  Optimal trajectory: {np.round(ac['trajectory']).astype(int)}")
print(f"  Expected cost: {ac['expected_cost']:.2f}")
print(f"  Cost variance: {ac['cost_variance']:.2f}")

# --- Transaction costs ---
from wraquant.execution.cost import slippage, commission_cost, total_cost

print(f"\n=== Transaction Costs ===")
slip = slippage(price=100, volume=1000, avg_daily_volume=100_000, volatility=0.02)
print(f"  Estimated slippage: {slip:.4f}")

comm = commission_cost(quantity=1000, price=100, rate_bps=5.0)
print(f"  Commission: ${comm:.2f}")

tc = total_cost(price=100, quantity=1000, spread=0.05, commission_bps=5.0, slippage_bps=2.0)
print(f"  Total cost: ${tc:.2f}")
