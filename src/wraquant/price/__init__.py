"""Options pricing, fixed income, stochastic models, and FBSDE solvers.

Submodules
----------
options
    Black-Scholes, binomial tree, and Monte Carlo option pricing.
greeks
    Analytical Black-Scholes Greeks (delta, gamma, theta, vega, rho).
volatility
    Implied volatility solvers and volatility surface construction.
fixed_income
    Bond pricing, yield-to-maturity, duration, and convexity.
curves
    Yield curve bootstrapping, interpolation, and forward rates.
levy_pricing
    FFT and COS-method pricing for VG, NIG, and generic characteristic
    function models.
characteristic
    Characteristic function constructors for Heston, VG, NIG, CGMY models
    and a unified pricing interface.
fbsde
    Forward-Backward SDE solvers for European and American derivatives.
stochastic
    Stochastic process simulators including GBM, Heston, jump-diffusion,
    SABR, rough Bergomi, 3/2, CIR, and Vasicek models.
integrations
    Wrappers for QuantLib, FinancePy, rateslib, and py-vollib.
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
]
