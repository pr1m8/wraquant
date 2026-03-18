"""Options pricing and Greeks with wraquant.

Demonstrates Black-Scholes, binomial trees, Monte Carlo pricing,
Greeks computation, implied volatility, and stochastic processes.
"""

from __future__ import annotations

import numpy as np

# --- Black-Scholes pricing ---
from wraquant.price.options import black_scholes, binomial_tree, monte_carlo_option

print("=== Black-Scholes ===")
bs = black_scholes(spot=100, strike=100, vol=0.2, rf=0.05, maturity=1.0, option_type="call")
print(f"  Call price: {bs['price']:.4f}")
print(f"  d1: {bs['d1']:.4f}, d2: {bs['d2']:.4f}")

bs_put = black_scholes(spot=100, strike=100, vol=0.2, rf=0.05, maturity=1.0, option_type="put")
print(f"  Put price: {bs_put['price']:.4f}")
print(f"  Put-call parity check: {bs['price'] - bs_put['price']:.4f} vs {100 - 100*np.exp(-0.05):.4f}")

# --- Binomial tree ---
print(f"\n=== Binomial Tree ===")
bt = binomial_tree(spot=100, strike=100, vol=0.2, rf=0.05, maturity=1.0, steps=200, option_type="call")
print(f"  Call price (200 steps): {bt['price']:.4f}")
print(f"  BS vs Binomial diff: {abs(bs['price'] - bt['price']):.4f}")

# --- Monte Carlo ---
print(f"\n=== Monte Carlo ===")
mc = monte_carlo_option(
    spot=100, strike=100, vol=0.2, rf=0.05, maturity=1.0,
    option_type="call", n_paths=100_000, seed=42,
)
print(f"  Call price: {mc['price']:.4f}")
print(f"  Std error: {mc['std_error']:.4f}")
print(f"  95% CI: [{mc['ci_lower']:.4f}, {mc['ci_upper']:.4f}]")

# --- Greeks ---
from wraquant.price.greeks import compute_greeks, greek_surface

print(f"\n=== Greeks ===")
greeks = compute_greeks(spot=100, strike=100, vol=0.2, rf=0.05, maturity=1.0, option_type="call")
for greek, value in greeks.items():
    print(f"  {greek}: {value:.6f}")

# --- Volatility surface ---
from wraquant.price.volatility import sabr_vol, implied_vol_newton

print(f"\n=== SABR Volatility ===")
sabr = sabr_vol(forward=100, strike=95, maturity=1.0, alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
print(f"  SABR implied vol (K=95): {sabr:.4f}")

sabr_atm = sabr_vol(forward=100, strike=100, maturity=1.0, alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
print(f"  SABR implied vol (ATM): {sabr_atm:.4f}")

# --- Implied vol from price ---
iv = implied_vol_newton(price=bs["price"], spot=100, strike=100, rf=0.05, maturity=1.0, option_type="call")
print(f"\n=== Implied Volatility ===")
print(f"  Recovered IV: {iv:.4f} (input was 0.2000)")

# --- Stochastic processes ---
from wraquant.price.stochastic import simulate_gbm, simulate_heston, simulate_jump_diffusion

print(f"\n=== Stochastic Processes ===")
gbm = simulate_gbm(s0=100, mu=0.05, sigma=0.2, T=1.0, n_steps=252, n_paths=5, seed=42)
print(f"  GBM final prices: {gbm[-1, :]}")

heston = simulate_heston(
    s0=100, v0=0.04, mu=0.05, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7,
    T=1.0, n_steps=252, n_paths=3, seed=42,
)
print(f"  Heston final prices: {heston['prices'][-1, :]}")

jd = simulate_jump_diffusion(
    s0=100, mu=0.05, sigma=0.15, lam=0.5, jump_mean=-0.02, jump_std=0.05,
    T=1.0, n_steps=252, n_paths=3, seed=42,
)
print(f"  Jump-diffusion final prices: {jd[-1, :]}")
