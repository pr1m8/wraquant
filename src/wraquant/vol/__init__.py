"""Volatility modeling and forecasting.

This module provides a complete toolkit for measuring, modeling, and
forecasting financial volatility -- the most critical input to risk
management, option pricing, and portfolio construction.

Key concepts
------------
Volatility is the degree of variation in asset returns over time. Unlike
prices, volatility is not directly observable and must be estimated. The
methods here fall into four families:

1. **Realized volatility estimators** -- non-parametric measures computed
   from observed prices. Use these for ex-post volatility measurement.

   - ``realized_volatility`` -- close-to-close standard deviation, the
     simplest estimator. Works well with daily data but ignores intraday
     information.
   - ``parkinson`` -- uses high and low prices, ~5x more efficient than
     close-to-close for continuous diffusions.
   - ``garman_klass`` -- uses open/high/low/close, the most efficient
     single-day estimator (~8x close-to-close).
   - ``rogers_satchell`` -- handles drift (trending markets) unlike
     Parkinson; preferred when the asset has a non-zero mean return.
   - ``yang_zhang`` -- combines overnight (open-to-close) and intraday
     (Rogers-Satchell) components; the best general-purpose estimator for
     daily OHLC data.

2. **GARCH family** -- parametric conditional volatility models. Use these
   when you need volatility *forecasts* and want to capture clustering
   (large moves beget large moves).

   - ``garch_fit`` -- standard GARCH(1,1). The workhorse model; start here.
   - ``egarch_fit`` -- Exponential GARCH. Captures the *leverage effect*
     (negative returns increase vol more than positive returns) without
     requiring positivity constraints.
   - ``gjr_garch_fit`` -- GJR-GARCH (Glosten-Jagannathan-Runkle). Also
     captures leverage via an asymmetric threshold term.  Simpler than
     EGARCH, often fits equity indices well.
   - ``figarch_fit`` -- Fractionally Integrated GARCH. Models long-memory
     in volatility (slow hyperbolic decay of shocks).  Use for assets
     where vol persistence is extremely high (e.g., FX, commodities).
   - ``harch_fit`` -- Heterogeneous ARCH. Mixes multiple time horizons
     (daily, weekly, monthly) reflecting heterogeneous market participants.
   - ``garch_forecast`` -- multi-step ahead volatility forecast from any
     fitted GARCH model.
   - ``dcc_fit`` -- Dynamic Conditional Correlation GARCH for multivariate
     volatility and time-varying correlations across assets.

3. **Stochastic and self-exciting volatility** -- models where volatility
   itself is a latent random process or event-driven.

   - ``stochastic_vol_sv`` -- Heston-style stochastic volatility estimated
     via particle filter or MCMC.  Use when GARCH is too rigid, e.g., for
     option-implied dynamics.
   - ``hawkes_process`` -- self-exciting point process for modeling
     volatility clustering through event arrivals (jumps, flash crashes).
   - ``gaussian_mixture_vol`` -- regime-switching volatility using
     Gaussian mixtures.  Use when you suspect distinct market regimes
     (calm vs. crisis).

4. **Implied volatility and variance risk premium** -- market-derived
   measures.

   - ``vol_surface_svi`` -- fit a Stochastic Volatility Inspired (SVI)
     parameterisation to an implied volatility surface. Use for option
     pricing and vol arbitrage.
   - ``variance_risk_premium`` -- difference between implied and realized
     variance.  Positive VRP means options are "expensive" relative to
     realized moves.

Diagnostics
-----------
- ``news_impact_curve`` -- visualise the asymmetric response of
  conditional variance to return shocks (leverage effect).
- ``volatility_persistence`` -- measure how long volatility shocks take
  to decay (sum of GARCH alpha + beta).

How to choose
-------------
- **Daily OHLC data, one asset**: start with ``yang_zhang`` for realized
  vol, then fit ``garch_fit`` for forecasts. If residuals show leverage,
  switch to ``egarch_fit`` or ``gjr_garch_fit``.
- **Multi-asset portfolio**: use ``dcc_fit`` for time-varying correlation
  and covariance.
- **Options / derivatives**: use ``vol_surface_svi`` for the smile, and
  ``variance_risk_premium`` to gauge richness.
- **Regime-aware strategies**: combine ``gaussian_mixture_vol`` with the
  ``regimes`` module.

References
----------
- Bollerslev (1986), "Generalized Autoregressive Conditional
  Heteroskedasticity"
- Engle (2002), "Dynamic Conditional Correlation"
- Gatheral (2004), "A parsimonious arbitrage-free implied volatility
  parameterization" (SVI)
- Yang & Zhang (2000), "Drift Independent Volatility Estimation"
"""

from wraquant.vol.models import (
    aparch_fit,
    dcc_fit,
    egarch_fit,
    ewma_volatility,
    figarch_fit,
    garch_fit,
    garch_forecast,
    garch_model_selection,
    garch_rolling_forecast,
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
    bipower_variation,
    garman_klass,
    jump_test_bns,
    parkinson,
    realized_kernel,
    realized_volatility,
    rogers_satchell,
    two_scale_realized_variance,
    yang_zhang,
)

__all__ = [
    # Realized volatility estimators
    "realized_volatility",
    "parkinson",
    "garman_klass",
    "rogers_satchell",
    "yang_zhang",
    # Jump-robust and noise-robust estimators
    "bipower_variation",
    "jump_test_bns",
    "two_scale_realized_variance",
    "realized_kernel",
    # EWMA
    "ewma_volatility",
    # GARCH family
    "garch_fit",
    "egarch_fit",
    "gjr_garch_fit",
    "figarch_fit",
    "aparch_fit",
    "harch_fit",
    "garch_forecast",
    "garch_rolling_forecast",
    "garch_model_selection",
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
