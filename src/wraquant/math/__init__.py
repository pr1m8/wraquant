"""Advanced mathematical tools for quantitative finance.

Submodules
----------
spectral
    Spectral analysis / FFT tools for financial data.
hawkes
    Hawkes process modelling for event clustering.
information
    Information-theoretic measures for financial analysis.
ergodicity
    Ergodicity economics (Ole Peters framework).
numerical
    General-purpose numerical methods.
signals
    Signal processing filters for financial time series.
"""

from __future__ import annotations

from wraquant.math.ergodicity import (
    ensemble_average,
    ergodicity_gap,
    ergodicity_ratio,
    growth_optimal_leverage,
    kelly_fraction,
    time_average,
)
from wraquant.math.hawkes import (
    fit_hawkes,
    hawkes_branching_ratio,
    hawkes_intensity,
    simulate_hawkes,
)
from wraquant.math.information import (
    conditional_entropy,
    entropy,
    fisher_information,
    kl_divergence,
    mutual_information,
    transfer_entropy,
)
from wraquant.math.numerical import (
    bisection,
    finite_difference_gradient,
    finite_difference_hessian,
    monte_carlo_integration,
    newton_raphson,
    trapezoidal_integration,
)
from wraquant.math.signals import (
    exponential_smooth,
    hodrick_prescott,
    kalman_smooth,
    median_filter,
    savitzky_golay,
    wavelet_denoise,
)
from wraquant.math.spectral import (
    bandpass_filter,
    dominant_frequencies,
    fft_decompose,
    spectral_density,
    spectral_entropy,
)

__all__ = [
    # spectral
    "fft_decompose",
    "dominant_frequencies",
    "spectral_density",
    "bandpass_filter",
    "spectral_entropy",
    # hawkes
    "hawkes_intensity",
    "simulate_hawkes",
    "fit_hawkes",
    "hawkes_branching_ratio",
    # information
    "fisher_information",
    "mutual_information",
    "transfer_entropy",
    "entropy",
    "kl_divergence",
    "conditional_entropy",
    # ergodicity
    "ensemble_average",
    "time_average",
    "ergodicity_gap",
    "kelly_fraction",
    "growth_optimal_leverage",
    "ergodicity_ratio",
    # numerical
    "finite_difference_gradient",
    "finite_difference_hessian",
    "newton_raphson",
    "bisection",
    "trapezoidal_integration",
    "monte_carlo_integration",
    # signals
    "savitzky_golay",
    "kalman_smooth",
    "wavelet_denoise",
    "median_filter",
    "exponential_smooth",
    "hodrick_prescott",
]
