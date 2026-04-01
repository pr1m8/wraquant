"""Options pricing, fixed income, stochastic models, and FBSDE solvers.

Provides a unified pricing library spanning equity derivatives, fixed
income instruments, and exotic stochastic models.  Includes both closed-
form solutions (Black-Scholes, bond pricing) and numerical methods
(binomial trees, Monte Carlo, FFT, COS method, deep BSDE solvers),
along with stochastic process simulators for scenario generation and
model calibration.

Key sub-modules:

- **Options** (``options``) -- ``black_scholes`` (closed-form European),
  ``binomial_tree`` (American/European via CRR lattice), and
  ``monte_carlo_option`` (path-dependent and exotic payoffs).
- **Greeks** (``greeks``) -- Analytical sensitivities: ``delta``,
  ``gamma``, ``theta``, ``vega``, ``rho``, and ``all_greeks`` (compute
  all at once).
- **Volatility** (``volatility``) -- ``implied_volatility`` (Newton/
  bisection solver), ``vol_smile`` (strike-space IV), and ``vol_surface``
  (strike x maturity IV grid).
- **Fixed income** (``fixed_income``) -- ``bond_price``, ``bond_yield``
  (YTM solver), ``duration``, ``modified_duration``, ``convexity``, and
  ``zero_rate``.
- **Curves** (``curves``) -- ``bootstrap_zero_curve``,
  ``interpolate_curve`` (linear, cubic, Nelson-Siegel), ``forward_rate``,
  and ``discount_factor``.
- **Levy pricing** (``levy_pricing``) -- ``fft_option_price`` (Carr-Madan
  FFT), ``cos_method`` (Fang-Oosterlee COS), ``vg_european_fft``, and
  ``nig_european_fft`` for pricing under fat-tailed Levy dynamics.
- **Characteristic functions** (``characteristic``) --
  ``heston_characteristic``, ``vg_characteristic``,
  ``nig_characteristic``, ``cgmy_characteristic``, and
  ``characteristic_function_price`` (unified pricing from any CF).
- **FBSDE solvers** (``fbsde``) -- ``fbsde_european`` (classical FBSDE),
  ``deep_bsde`` (neural-network BSDE solver for high-dimensional PDEs),
  and ``reflected_bsde`` (American-style early exercise).
- **Stochastic processes** (``stochastic``) -- Path simulators:
  ``geometric_brownian_motion``, ``heston``, ``jump_diffusion``,
  ``ornstein_uhlenbeck``, ``simulate_sabr``, ``simulate_rough_bergomi``,
  ``simulate_3_2_model``, ``simulate_cir``, and ``simulate_vasicek``.
- **Integrations** (``integrations``) -- Wrappers for QuantLib
  (``quantlib_option``, ``quantlib_bond``, ``quantlib_yield_curve``),
  FinancePy, rateslib, and py_vollib.

Example:
    >>> from wraquant.price import black_scholes, implied_volatility
    >>> price = black_scholes(S=100, K=105, T=0.25, r=0.05, sigma=0.2)
    >>> iv = implied_volatility(market_price=3.50, S=100, K=105, T=0.25, r=0.05)

Use ``wraquant.price`` for derivative pricing, fixed income analytics,
and stochastic simulation.  For realized and conditional volatility
modeling (GARCH, realized estimators), see ``wraquant.vol``.  For
implied vol surface fitting (SVI), see ``wraquant.vol.vol_surface_svi``.
"""

from __future__ import annotations

from wraquant.price.characteristic import (
    cgmy_characteristic,
    characteristic_function_price,
    heston_characteristic,
    nig_characteristic,
    vg_characteristic,
)
from wraquant.price.curves import (
    bootstrap_zero_curve,
    discount_factor,
    forward_rate,
    interpolate_curve,
)
from wraquant.price.fbsde import (
    deep_bsde,
    fbsde_european,
    reflected_bsde,
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
from wraquant.price.integrations import (
    financepy_option,
    quantlib_bond,
    quantlib_option,
    quantlib_yield_curve,
    rateslib_swap,
    sdeint_solve,
    vollib_implied_vol,
)
from wraquant.price.levy_pricing import (
    cos_method,
    fft_option_price,
    nig_european_fft,
    vg_european_fft,
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
    simulate_3_2_model,
    simulate_cir,
    simulate_rough_bergomi,
    simulate_sabr,
    simulate_vasicek,
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
    # Curves
    "bootstrap_zero_curve",
    "interpolate_curve",
    "forward_rate",
    "discount_factor",
    # Levy pricing
    "vg_european_fft",
    "nig_european_fft",
    "fft_option_price",
    "cos_method",
    # Characteristic functions
    "characteristic_function_price",
    "heston_characteristic",
    "vg_characteristic",
    "nig_characteristic",
    "cgmy_characteristic",
    # FBSDE solvers
    "fbsde_european",
    "deep_bsde",
    "reflected_bsde",
    # Stochastic models
    "geometric_brownian_motion",
    "heston",
    "jump_diffusion",
    "ornstein_uhlenbeck",
    "cir_process",
    "simulate_sabr",
    "simulate_rough_bergomi",
    "simulate_3_2_model",
    "simulate_cir",
    "simulate_vasicek",
    # Integrations
    "quantlib_bond",
    "quantlib_option",
    "quantlib_yield_curve",
    "financepy_option",
    "rateslib_swap",
    "vollib_implied_vol",
    "sdeint_solve",
]
