"""Risk management with wraquant.

Demonstrates VaR/CVaR, stress testing, copulas, DCC correlation,
credit risk, and Monte Carlo methods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 500
returns = rng.normal(0.0003, 0.02, n)

# --- Value at Risk ---
from wraquant.risk.var import historical_var, parametric_var, monte_carlo_var

h_var = historical_var(returns, alpha=0.05)
p_var = parametric_var(returns, alpha=0.05)
mc_var = monte_carlo_var(returns, alpha=0.05, n_simulations=10_000, seed=42)

print("=== Value at Risk (95%) ===")
print(f"  Historical VaR: {h_var['var']:.4f}")
print(f"  Parametric VaR: {p_var['var']:.4f}")
print(f"  Monte Carlo VaR: {mc_var['var']:.4f}")
print(f"  Historical CVaR: {h_var['cvar']:.4f}")

# --- Risk metrics ---
from wraquant.risk.metrics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    information_ratio,
)

print(f"\n=== Risk Metrics ===")
print(f"  Sharpe: {sharpe_ratio(returns):.4f}")
print(f"  Sortino: {sortino_ratio(returns):.4f}")
print(f"  Max Drawdown: {max_drawdown(returns):.4f}")
print(f"  Calmar: {calmar_ratio(returns):.4f}")

benchmark = rng.normal(0.0002, 0.018, n)
print(f"  Info Ratio: {information_ratio(returns, benchmark):.4f}")

# --- Stress testing ---
from wraquant.risk.stress import (
    stress_test_returns,
    vol_stress_test,
    spot_stress_test,
    sensitivity_ladder,
)

print(f"\n=== Stress Testing ===")
stressed = stress_test_returns(returns, shock_mean=-0.01, shock_vol=1.5)
print(f"  Original mean: {returns.mean():.6f}")
print(f"  Stressed mean: {stressed.mean():.6f}")

vol_stress = vol_stress_test(returns, vol_multipliers=[1.0, 1.5, 2.0, 3.0])
for scenario in vol_stress:
    print(f"  Vol x{scenario['vol_multiplier']}: VaR={scenario['var_95']:.4f}")

# --- Copula fitting ---
from wraquant.risk.copulas import fit_gaussian_copula, copula_simulate

bivariate = np.column_stack([returns, benchmark])
cop = fit_gaussian_copula(bivariate)
print(f"\n=== Gaussian Copula ===")
print(f"  Correlation matrix:\n{cop['correlation']}")

sim = copula_simulate(cop["correlation"], n_samples=1000, seed=42)
print(f"  Simulated shape: {sim.shape}")

# --- DCC-GARCH ---
from wraquant.risk.dcc import dcc_garch

multi_returns = np.column_stack([returns, benchmark, rng.normal(0, 0.02, n)])
dcc_result = dcc_garch(multi_returns)
print(f"\n=== DCC-GARCH ===")
print(f"  Time-varying correlations shape: {dcc_result['correlations'].shape}")
print(f"  Final correlation matrix:\n{dcc_result['correlations'][-1]}")

# --- Credit risk ---
from wraquant.risk.credit import merton_model, altman_z_score

merton = merton_model(asset_value=100, debt_face=80, volatility=0.3, rf=0.05, maturity=1.0)
print(f"\n=== Merton Model ===")
print(f"  Default probability: {merton['default_probability']:.4f}")
print(f"  Credit spread: {merton['credit_spread']:.4f}")

z = altman_z_score(
    working_capital=50, total_assets=200, retained_earnings=30,
    ebit=25, market_equity=150, total_liabilities=80, sales=300,
)
print(f"  Altman Z-score: {z['z_score']:.2f} ({z['zone']})")

# --- Monte Carlo methods ---
from wraquant.risk.monte_carlo import block_bootstrap, importance_sampling_var

bb = block_bootstrap(returns, block_size=10, n_bootstrap=1000, seed=42)
print(f"\n=== Block Bootstrap ===")
print(f"  Bootstrap VaR 95%: {np.percentile(bb['bootstrap_vars'], 5):.4f}")

is_var = importance_sampling_var(returns, alpha=0.01, n_simulations=50_000, seed=42)
print(f"  Importance sampling VaR 99%: {is_var['var']:.4f}")
