"""Volatility modeling and forecasting.

Covers realized volatility estimators, EWMA, GARCH-family models (GARCH,
EGARCH, GJR-GARCH, FIGARCH, HARCH), stochastic volatility, Hawkes
processes, Gaussian mixture regime-based volatility, SVI implied vol
surfaces, and variance risk premium computation.
"""

from wraquant.vol.models import (
    dcc_fit,
    egarch_fit,
    ewma_volatility,
    figarch_fit,
    garch_fit,
    garch_forecast,
    gaussian_mixture_vol,
    gjr_garch_fit,
    harch_fit,
    hawkes_process,
    news_impact_curve,
    realized_garch,
    stochastic_vol_sv,
    variance_risk_premium,
    vol_surface_svi,
    volatility_persistence,
)
from wraquant.vol.realized import (
    garman_klass,
    parkinson,
    realized_volatility,
    rogers_satchell,
    yang_zhang,
)

__all__ = [
    # Realized volatility estimators
    "realized_volatility",
    "parkinson",
    "garman_klass",
    "rogers_satchell",
    "yang_zhang",
    # EWMA
    "ewma_volatility",
    # GARCH family
    "garch_fit",
    "egarch_fit",
    "gjr_garch_fit",
    "figarch_fit",
    "harch_fit",
    "garch_forecast",
    "dcc_fit",
    "realized_garch",
    # Diagnostics
    "news_impact_curve",
    "volatility_persistence",
    # Self-exciting and stochastic vol
    "hawkes_process",
    "stochastic_vol_sv",
    "gaussian_mixture_vol",
    # Implied vol and VRP
    "vol_surface_svi",
    "variance_risk_premium",
]
