"""Bayesian analysis methods for quantitative finance.

Provides conjugate Bayesian regression, Bayesian Sharpe ratio estimation,
portfolio allocation via posterior sampling, Bayesian VaR, MCMC samplers,
convergence diagnostics, and wrappers for external Bayesian packages
(PyMC, ArviZ, NumPyro).
"""

from wraquant.bayes.mcmc import (
    gelman_rubin,
    gibbs_sampler,
    metropolis_hastings,
    nuts_diagnostic,
    trace_summary,
)
from wraquant.bayes.models import (
    bayes_factor,
    bayesian_portfolio,
    bayesian_regression,
    bayesian_sharpe,
    bayesian_var,
    credible_interval,
    posterior_predictive,
)

__all__ = [
    # models.py — pure numpy/scipy Bayesian methods
    "bayesian_regression",
    "bayesian_sharpe",
    "bayesian_portfolio",
    "bayesian_var",
    "credible_interval",
    "bayes_factor",
    "posterior_predictive",
    # mcmc.py — MCMC utilities
    "metropolis_hastings",
    "gibbs_sampler",
    "nuts_diagnostic",
    "trace_summary",
    "gelman_rubin",
]
