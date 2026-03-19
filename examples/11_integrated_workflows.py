"""Example 11 — Integrated workflows using wraquant recipes.

Demonstrates how wraquant's 27+ modules work together as a cohesive
framework through the ``recipes`` module.  Each recipe chains multiple
subsystems into a complete pipeline.

Recipes shown:
    1. analyze()                        -- quick comprehensive analysis
    2. regime_aware_backtest()          -- regime -> position sizing -> tearsheet
    3. garch_risk_pipeline()            -- GARCH -> VaR/CVaR -> stress testing
    4. portfolio_construction_pipeline() -- optimize -> regime adjust -> risk decompose
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Synthetic data (replace with real market data in practice)
# ---------------------------------------------------------------------------

np.random.seed(42)
N = 600  # ~2.4 years of daily data

# Simulate a price series with regime-like behavior
bull_returns = np.random.normal(0.0005, 0.008, N // 2)
bear_returns = np.random.normal(-0.0002, 0.018, N // 2)
market_returns = np.concatenate([bull_returns, bear_returns])
np.random.shuffle(market_returns)  # mix regimes

prices = pd.Series(
    100 * np.cumprod(1 + market_returns),
    index=pd.bdate_range("2022-01-03", periods=N),
    name="SPY",
)
returns = prices.pct_change().dropna()

# Multi-asset returns for portfolio recipes
multi_returns = pd.DataFrame(
    {
        "SPY": returns.values,
        "TLT": np.random.normal(0.0001, 0.005, len(returns)),
        "GLD": np.random.normal(0.0002, 0.008, len(returns)),
    },
    index=returns.index,
)

multi_prices = (1 + multi_returns).cumprod() * 100


# ---------------------------------------------------------------------------
# 1. analyze() — Quick comprehensive analysis
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. analyze() — Comprehensive Analysis")
print("=" * 60)

import wraquant as wq

report = wq.analyze(returns)

print(f"\nDescriptive stats:")
for k, v in report["descriptive"].items():
    print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

print(f"\nRisk metrics:")
for k, v in report["risk"].items():
    print(f"  {k}: {v:.4f}")

print(f"\nStationarity (ADF):")
print(f"  p-value: {report['stationarity']['p_value']:.4f}")
print(f"  is_stationary: {report['stationarity']['is_stationary']}")

if "regime" in report:
    print(f"\nRegime detection:")
    print(f"  current regime: {report['regime']['current']}")
    print(f"  probabilities: {report['regime']['probabilities']}")

if "volatility" in report:
    print(f"\nGARCH volatility:")
    print(f"  persistence: {report['volatility']['persistence']:.4f}")
    print(f"  half_life: {report['volatility']['half_life']:.1f} days")
    print(f"  current vol: {report['volatility']['current_vol']:.4f}")


# ---------------------------------------------------------------------------
# 2. regime_aware_backtest() — Regime-Aware Backtest
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. regime_aware_backtest() — Regime-Aware Backtest")
print("=" * 60)

from wraquant.recipes import regime_aware_backtest

bt_result = regime_aware_backtest(
    prices,
    n_regimes=2,
    bull_weight=1.0,  # fully invested in low-vol regime
    bear_weight=0.3,  # 30% invested in high-vol regime
)

print(f"\nRegime statistics:")
print(bt_result["regime_stats"].to_string())

print(f"\nRisk metrics:")
for k, v in bt_result["risk_metrics"].items():
    print(f"  {k}: {v:.4f}")

ts = bt_result["tearsheet"]["summary"]
print(f"\nTearsheet summary:")
for k, v in ts.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")


# ---------------------------------------------------------------------------
# 3. garch_risk_pipeline() — GARCH Risk Pipeline
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. garch_risk_pipeline() — GARCH Risk Pipeline")
print("=" * 60)

from wraquant.recipes import garch_risk_pipeline

try:
    risk_result = garch_risk_pipeline(
        returns,
        vol_model="GJR",
        dist="t",
        var_alpha=0.05,
    )

    diag = risk_result["diagnostics"]
    print(f"\nGARCH diagnostics:")
    print(f"  persistence: {diag['persistence']:.4f}")
    print(f"  half_life: {diag['half_life']:.1f} days")
    print(f"  current vol: {diag['current_vol']:.6f}")
    print(f"  VaR breach rate: {diag['breach_rate']:.3f} (target: 0.050)")
except Exception as e:
    print(f"\nGARCH pipeline skipped (optional dep): {e}")


# ---------------------------------------------------------------------------
# 4. portfolio_construction_pipeline() — Portfolio Construction
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. portfolio_construction_pipeline() — Portfolio Construction")
print("=" * 60)

from wraquant.recipes import portfolio_construction_pipeline

port_result = portfolio_construction_pipeline(
    multi_returns,
    method="risk_parity",
    regime_aware=True,
)

print(f"\nOptimal weights:")
for asset, w in port_result["weights"].items():
    print(f"  {asset}: {w:.4f}")

print(f"\nDiversification ratio: {port_result['diversification_ratio']:.4f}")
print(f"Regime adjusted: {port_result['regime_adjusted']}")

print(f"\nBetas (vs {multi_returns.columns[0]}):")
for asset, beta in port_result["betas"].items():
    print(f"  {asset}: {beta:.4f}")

print(f"\nComponent VaR:")
for asset, cvar in port_result["component_var"].items():
    print(f"  {asset}: {cvar:.6f}")

print("\n" + "=" * 60)
print("All integrated workflows completed successfully.")
print("=" * 60)
