"""Bayesian analysis methods for quantitative finance.

Provides conjugate Bayesian regression, Bayesian Sharpe ratio estimation,
portfolio allocation via posterior sampling, Bayesian VaR, MCMC samplers,
convergence diagnostics, and wrappers for external Bayesian packages
(PyMC, ArviZ, NumPyro, Bambi, emcee, BlackJAX).
"""

from wraquant.bayes.mcmc import (
    convergence_diagnostics,
    gelman_rubin,
    gibbs_sampler,
    hamiltonian_monte_carlo,
    metropolis_hastings,
    nuts_diagnostic,
    slice_sampler,
    trace_summary,
)
from wraquant.bayes.models import (
    bayes_factor,
    bayesian_changepoint,
    bayesian_cointegration,
    bayesian_factor_model,
    bayesian_linear_regression,
    bayesian_portfolio,
    bayesian_portfolio_bl,
    bayesian_regime_inference,
    bayesian_regression,
    bayesian_sharpe,
    bayesian_var,
    bayesian_volatility,
    credible_interval,
    model_comparison,
    posterior_predictive,
)

__all__ = [
    # models.py — pure numpy/scipy Bayesian methods (original)
    "bayesian_regression",
    "bayesian_sharpe",
    "bayesian_portfolio",
    "bayesian_var",
    "credible_interval",
    "bayes_factor",
    "posterior_predictive",
    # models.py — enhanced deep implementations
    "bayesian_linear_regression",
    "bayesian_factor_model",
    "bayesian_changepoint",
    "bayesian_portfolio_bl",
    "bayesian_volatility",
    "bayesian_cointegration",
    "bayesian_regime_inference",
    "model_comparison",
    # mcmc.py — MCMC utilities (original)
    "metropolis_hastings",
    "gibbs_sampler",
    "nuts_diagnostic",
    "trace_summary",
    "gelman_rubin",
    # mcmc.py — enhanced MCMC
    "hamiltonian_monte_carlo",
    "slice_sampler",
    "convergence_diagnostics",
]


def __getattr__(name: str):
    """Lazy-load integration functions to avoid importing optional deps at module level."""
    _integration_names = {
        "pymc_regression",
        "arviz_summary",
        "numpyro_regression",
        "bambi_regression",
        "emcee_sample",
        "blackjax_nuts",
    }
    if name in _integration_names:
        from wraquant.bayes import integrations

        return getattr(integrations, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
