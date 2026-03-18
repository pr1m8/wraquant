"""Options pricing, stochastic models, and FBSDE solvers.

Submodules
----------
levy_pricing
    FFT and COS-method pricing for VG, NIG, and generic characteristic
    function models.
characteristic
    Characteristic function constructors for Heston, VG, NIG, CGMY models
    and a unified pricing interface.
fbsde
    Forward-Backward SDE solvers for European and American derivatives.
stochastic
    Stochastic process simulators including SABR, rough Bergomi, 3/2,
    CIR, and Vasicek models.
"""

from __future__ import annotations

from wraquant.price.characteristic import (
    cgmy_characteristic,
    characteristic_function_price,
    heston_characteristic,
    nig_characteristic,
    vg_characteristic,
)
from wraquant.price.fbsde import (
    deep_bsde,
    fbsde_european,
    reflected_bsde,
)
from wraquant.price.levy_pricing import (
    cos_method,
    fft_option_price,
    nig_european_fft,
    vg_european_fft,
)
from wraquant.price.stochastic import (
    simulate_3_2_model,
    simulate_cir,
    simulate_rough_bergomi,
    simulate_sabr,
    simulate_vasicek,
)

__all__ = [
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
    "simulate_sabr",
    "simulate_rough_bergomi",
    "simulate_3_2_model",
    "simulate_cir",
    "simulate_vasicek",
]
