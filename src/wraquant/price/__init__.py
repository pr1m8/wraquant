"""Options pricing, fixed income, volatility surfaces, and stochastic processes.

This module provides pure numpy/scipy implementations of:

- **Options pricing**: Black-Scholes, binomial tree, Monte Carlo
- **Greeks**: Delta, gamma, theta, vega, rho (analytical BS)
- **Volatility**: Implied vol, vol smiles and surfaces
- **Fixed income**: Bond pricing, yields, duration, convexity
- **Stochastic processes**: GBM, Heston, jump-diffusion, OU, CIR
- **Yield curves**: Bootstrapping, interpolation, forward rates
"""

from __future__ import annotations

from wraquant.price.curves import (
    bootstrap_zero_curve,
    discount_factor,
    forward_rate,
    interpolate_curve,
)
from wraquant.price.fixed_income import (
    bond_price,
    bond_yield,
    convexity,
    duration,
    modified_duration,
    zero_rate,
)
from wraquant.price.greeks import (
    all_greeks,
    delta,
    gamma,
    rho,
    theta,
    vega,
)
from wraquant.price.options import (
    binomial_tree,
    black_scholes,
    monte_carlo_option,
)
from wraquant.price.stochastic import (
    cir_process,
    geometric_brownian_motion,
    heston,
    jump_diffusion,
    ornstein_uhlenbeck,
)
from wraquant.price.volatility import (
    implied_volatility,
    vol_smile,
    vol_surface,
)

__all__ = [
    # Options pricing
    "black_scholes",
    "binomial_tree",
    "monte_carlo_option",
    # Greeks
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "all_greeks",
    # Volatility
    "implied_volatility",
    "vol_smile",
    "vol_surface",
    # Fixed income
    "bond_price",
    "bond_yield",
    "duration",
    "modified_duration",
    "convexity",
    "zero_rate",
    # Stochastic processes
    "geometric_brownian_motion",
    "heston",
    "jump_diffusion",
    "ornstein_uhlenbeck",
    "cir_process",
    # Yield curves
    "bootstrap_zero_curve",
    "interpolate_curve",
    "forward_rate",
    "discount_factor",
]
