"""Portfolio optimization with wraquant.

Demonstrates mean-variance optimization, risk parity,
Black-Litterman, and hierarchical risk parity.
"""

from __future__ import annotations

import numpy as np

rng = np.random.default_rng(42)

# --- Generate multi-asset return data ---
n_assets = 5
n_obs = 500
asset_names = ["US_Equity", "Intl_Equity", "Bonds", "Gold", "REITs"]

# Correlated returns via Cholesky
true_means = np.array([0.08, 0.06, 0.03, 0.04, 0.07]) / 252
true_vols = np.array([0.18, 0.22, 0.06, 0.15, 0.20]) / np.sqrt(252)
corr = np.array([
    [1.0, 0.7, -0.2, 0.1, 0.5],
    [0.7, 1.0, -0.1, 0.15, 0.4],
    [-0.2, -0.1, 1.0, 0.3, 0.0],
    [0.1, 0.15, 0.3, 1.0, 0.1],
    [0.5, 0.4, 0.0, 0.1, 1.0],
])
cov = np.outer(true_vols, true_vols) * corr
L = np.linalg.cholesky(cov)
returns = rng.normal(size=(n_obs, n_assets)) @ L.T + true_means

# --- Mean-Variance Optimization ---
from wraquant.opt.portfolio import (
    mean_variance_optimize,
    risk_parity,
    black_litterman,
    hierarchical_risk_parity,
)

print("=== Mean-Variance Optimization ===")
mv = mean_variance_optimize(returns, target_return=0.0003)
print(f"  Weights: {dict(zip(asset_names, np.round(mv['weights'], 4)))}")
print(f"  Expected return: {mv['expected_return']:.6f}")
print(f"  Portfolio vol: {mv['portfolio_volatility']:.6f}")

# --- Minimum Variance ---
mv_min = mean_variance_optimize(returns)
print(f"\n  Min-variance weights: {dict(zip(asset_names, np.round(mv_min['weights'], 4)))}")

# --- Risk Parity ---
print(f"\n=== Risk Parity ===")
rp = risk_parity(returns)
print(f"  Weights: {dict(zip(asset_names, np.round(rp['weights'], 4)))}")
print(f"  Risk contributions: {np.round(rp['risk_contributions'], 4)}")

# --- Black-Litterman ---
print(f"\n=== Black-Litterman ===")
market_weights = np.array([0.4, 0.2, 0.2, 0.1, 0.1])
# View: US Equity will outperform Intl Equity by 2% annualized
P = np.array([[1, -1, 0, 0, 0]])
Q = np.array([0.02 / 252])
bl = black_litterman(returns, market_weights, P, Q, tau=0.05)
print(f"  BL weights: {dict(zip(asset_names, np.round(bl['weights'], 4)))}")
print(f"  BL expected returns: {np.round(bl['expected_returns'] * 252, 4)}")

# --- Hierarchical Risk Parity ---
print(f"\n=== Hierarchical Risk Parity ===")
hrp = hierarchical_risk_parity(returns)
print(f"  HRP weights: {dict(zip(asset_names, np.round(hrp['weights'], 4)))}")

# --- Efficient frontier ---
from wraquant.opt.portfolio import efficient_frontier

ef = efficient_frontier(returns, n_points=20)
print(f"\n=== Efficient Frontier ===")
print(f"  {len(ef['returns'])} points computed")
print(f"  Return range: [{min(ef['returns']):.6f}, {max(ef['returns']):.6f}]")
print(f"  Vol range: [{min(ef['volatilities']):.6f}, {max(ef['volatilities']):.6f}]")
