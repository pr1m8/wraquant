"""Bayesian analysis methods for quantitative finance.

Provides a complete Bayesian inference toolkit for financial modeling,
from conjugate closed-form solutions through custom MCMC samplers to
wrappers for modern probabilistic programming frameworks.  Bayesian
methods are particularly valuable in finance for incorporating prior
beliefs (e.g., market equilibrium), quantifying parameter uncertainty
(e.g., "how confident is this Sharpe ratio?"), and producing full
posterior distributions rather than point estimates.

Key sub-modules:

- **Models** (``models``) -- Pure numpy/scipy Bayesian methods:
  ``bayesian_regression`` (conjugate normal-inverse-gamma),
  ``bayesian_sharpe`` (posterior distribution of the Sharpe ratio),
  ``bayesian_portfolio`` (posterior-sampled allocation),
  ``bayesian_var`` (Bayesian Value-at-Risk with parameter uncertainty),
  ``bayesian_linear_regression`` (full posterior with credible intervals),
  ``bayesian_factor_model``, ``bayesian_changepoint`` (Bayesian change-
  point detection), ``bayesian_portfolio_bl`` (Bayesian Black-Litterman),
  ``bayesian_volatility``, ``bayesian_cointegration``,
  ``bayesian_regime_inference``, ``model_comparison`` (Bayes factors and
  WAIC/LOO), ``credible_interval``, and ``posterior_predictive``.
- **MCMC** (``mcmc``) -- Sampling algorithms and diagnostics:
  ``metropolis_hastings``, ``hamiltonian_monte_carlo``,
  ``gibbs_sampler``, ``slice_sampler``, ``gelman_rubin`` (R-hat
  convergence diagnostic), ``nuts_diagnostic``, ``trace_summary``,
  and ``convergence_diagnostics``.
- **Integrations** (lazy-loaded) -- Wrappers for external packages:
  ``pymc_regression`` (PyMC), ``arviz_summary`` (ArviZ),
  ``numpyro_regression`` (NumPyro/JAX), ``bambi_regression`` (Bambi),
  ``emcee_sample`` (emcee), and ``blackjax_nuts`` (BlackJAX).

Example:
    >>> from wraquant.bayes import bayesian_sharpe, bayesian_regression
    >>> posterior = bayesian_sharpe(returns, n_samples=10_000)
    >>> print(f"Sharpe 95% CI: [{posterior['ci_lower']:.2f}, {posterior['ci_upper']:.2f}]")
    >>> reg = bayesian_regression(X, y, prior_precision=0.1)

Use ``wraquant.bayes`` when you need uncertainty quantification around
financial estimates, want to incorporate prior information (views,
equilibrium), or need full posterior distributions for downstream
decisions.  For frequentist regression and hypothesis testing, see
``wraquant.stats``.  For Black-Litterman portfolio optimization, see
``wraquant.opt.black_litterman`` or ``wraquant.bayes.bayesian_portfolio_bl``.
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
