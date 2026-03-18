"""External package wrappers for Bayesian analysis.

Functions in this module require the ``bayes`` optional dependency group
(PyMC, ArviZ, NumPyro) and are guarded by ``@requires_extra('bayes')``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wraquant.core.decorators import requires_extra

__all__ = [
    "pymc_regression",
    "arviz_summary",
    "numpyro_regression",
    "bambi_regression",
    "emcee_sample",
    "blackjax_nuts",
]


# ---------------------------------------------------------------------------
# PyMC Bayesian regression
# ---------------------------------------------------------------------------


@requires_extra("bayes")
def pymc_regression(
    y: np.ndarray | pd.Series,
    X: np.ndarray | pd.DataFrame,
    samples: int = 2_000,
    chains: int = 2,
    target_accept: float = 0.9,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Bayesian linear regression using PyMC.

    Fits the model y ~ Normal(X @ beta, sigma) with weakly informative
    priors.

    Parameters
    ----------
    y : array-like
        Response variable.
    X : array-like
        Design matrix. An intercept column is added automatically.
    samples : int
        Number of posterior samples per chain.
    chains : int
        Number of MCMC chains.
    target_accept : float
        Target acceptance rate for NUTS.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``trace``: PyMC InferenceData object,
        ``coefficients_mean``: np.ndarray of posterior mean coefficients,
        ``coefficients_std``: np.ndarray of posterior std coefficients,
        ``sigma_mean``: float — posterior mean of noise std,
        ``model``: PyMC Model object.
    """
    import pymc as pm

    y_arr = np.asarray(y, dtype=float).ravel()
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    n, k = X_arr.shape

    # Add intercept
    X_arr = np.column_stack([np.ones(n), X_arr])
    k_total = k + 1

    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, sigma=10, shape=k_total)
        sigma = pm.HalfNormal("sigma", sigma=5)
        mu = pm.math.dot(X_arr, beta)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_arr)

        trace = pm.sample(
            draws=samples,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
        )

    beta_samples = trace.posterior["beta"].values.reshape(-1, k_total)
    sigma_samples = trace.posterior["sigma"].values.ravel()

    return {
        "trace": trace,
        "coefficients_mean": np.mean(beta_samples, axis=0),
        "coefficients_std": np.std(beta_samples, axis=0),
        "sigma_mean": float(np.mean(sigma_samples)),
        "model": model,
    }


# ---------------------------------------------------------------------------
# ArviZ summary
# ---------------------------------------------------------------------------


@requires_extra("bayes")
def arviz_summary(
    trace: Any,
    var_names: list[str] | None = None,
    hdi_prob: float = 0.94,
) -> pd.DataFrame:
    """Generate a summary table from a trace using ArviZ.

    Parameters
    ----------
    trace : InferenceData or dict
        ArviZ InferenceData object or a dict of arrays.
    var_names : list[str] or None
        Variables to include. If None, includes all.
    hdi_prob : float
        Probability mass for the HDI interval. Default is 0.94.

    Returns
    -------
    pd.DataFrame
        Summary table with mean, sd, HDI, ESS, R-hat.
    """
    import arviz as az

    # ArviZ >= 0.20 renamed hdi_prob to ci_prob
    try:
        summary = az.summary(trace, var_names=var_names, hdi_prob=hdi_prob)
    except TypeError:
        summary = az.summary(trace, var_names=var_names, ci_prob=hdi_prob)
    return summary


# ---------------------------------------------------------------------------
# NumPyro regression
# ---------------------------------------------------------------------------


@requires_extra("bayes")
def numpyro_regression(
    y: np.ndarray | pd.Series,
    X: np.ndarray | pd.DataFrame,
    samples: int = 2_000,
    warmup: int = 500,
    chains: int = 1,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Bayesian linear regression using NumPyro.

    Fits the model y ~ Normal(X @ beta, sigma) using NUTS sampling.

    Parameters
    ----------
    y : array-like
        Response variable.
    X : array-like
        Design matrix. An intercept column is added automatically.
    samples : int
        Number of posterior samples.
    warmup : int
        Number of warmup (burn-in) samples.
    chains : int
        Number of MCMC chains.
    rng_seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``samples``: dict of posterior samples by parameter name,
        ``coefficients_mean``: np.ndarray of posterior mean coefficients,
        ``coefficients_std``: np.ndarray of posterior std coefficients,
        ``sigma_mean``: float — posterior mean of noise std.
    """
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    y_arr = np.asarray(y, dtype=float).ravel()
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    n, k = X_arr.shape

    # Add intercept
    X_arr = np.column_stack([np.ones(n), X_arr])
    k_total = k + 1

    X_jnp = jnp.array(X_arr)
    y_jnp = jnp.array(y_arr)

    def model(X: jnp.ndarray, y: jnp.ndarray | None = None) -> None:
        beta = numpyro.sample("beta", dist.Normal(0, 10).expand([k_total]))
        sigma = numpyro.sample("sigma", dist.HalfNormal(5))
        mu = jnp.dot(X, beta)
        numpyro.sample("y_obs", dist.Normal(mu, sigma), obs=y)

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=chains)
    rng_key = jax.random.PRNGKey(rng_seed)
    mcmc.run(rng_key, X_jnp, y_jnp)

    posterior_samples = mcmc.get_samples()
    beta_samples = np.asarray(posterior_samples["beta"])
    sigma_samples = np.asarray(posterior_samples["sigma"])

    return {
        "samples": {k: np.asarray(v) for k, v in posterior_samples.items()},
        "coefficients_mean": np.mean(beta_samples, axis=0),
        "coefficients_std": np.std(beta_samples, axis=0),
        "sigma_mean": float(np.mean(sigma_samples)),
    }


# ---------------------------------------------------------------------------
# Bambi regression
# ---------------------------------------------------------------------------


@requires_extra("bayes")
def bambi_regression(
    formula: str,
    data: pd.DataFrame,
    family: str = "gaussian",
    draws: int = 1000,
    chains: int = 2,
    seed: int | None = None,
) -> dict[str, Any]:
    """Fit a Bayesian regression using Bambi's formula interface.

    Bambi provides R-style formula syntax for Bayesian models built on PyMC.

    Parameters
    ----------
    formula : str
        Model formula string (e.g., ``"y ~ x1 + x2"``).
    data : pd.DataFrame
        DataFrame with variables referenced in formula.
    family : str, default "gaussian"
        Likelihood family (``"gaussian"``, ``"bernoulli"``, ``"poisson"``).
    draws : int, default 1000
        Number of posterior draws per chain.
    chains : int, default 2
        Number of MCMC chains.
    seed : int or None
        Random seed.

    Returns
    -------
    dict
        ``model``: Bambi Model object,
        ``trace``: InferenceData posterior samples,
        ``summary``: pd.DataFrame ArviZ summary table.
    """
    import bambi as bmb

    model = bmb.Model(formula, data, family=family)
    trace = model.fit(draws=draws, chains=chains, random_seed=seed)
    import arviz as az

    try:
        summary = az.summary(trace, hdi_prob=0.94)
    except TypeError:
        summary = az.summary(trace, ci_prob=0.94)
    return {"model": model, "trace": trace, "summary": summary}


# ---------------------------------------------------------------------------
# emcee ensemble sampling
# ---------------------------------------------------------------------------


@requires_extra("bayes")
def emcee_sample(
    log_prob_fn: Any,
    n_walkers: int,
    n_dim: int,
    n_steps: int = 1000,
    initial: np.ndarray | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run ensemble MCMC sampling using emcee.

    Parameters
    ----------
    log_prob_fn : callable
        Log probability function accepting a parameter array.
    n_walkers : int
        Number of walkers (must be >= 2 * n_dim).
    n_dim : int
        Number of parameters.
    n_steps : int, default 1000
        Number of MCMC steps.
    initial : np.ndarray or None
        Initial walker positions of shape ``(n_walkers, n_dim)``.
        If None, drawn from a standard normal distribution.
    seed : int or None
        Random seed.

    Returns
    -------
    dict
        ``samples``: array of shape ``(n_walkers * n_steps, n_dim)``,
        ``log_prob``: array of log probabilities,
        ``acceptance_fraction``: mean acceptance fraction.
    """
    import emcee

    rng = np.random.default_rng(seed)
    if initial is None:
        initial = rng.normal(0, 1, (n_walkers, n_dim))
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob_fn)
    sampler.run_mcmc(initial, n_steps, progress=False)
    return {
        "samples": sampler.get_chain(flat=True),
        "log_prob": sampler.get_log_prob(flat=True),
        "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
    }


# ---------------------------------------------------------------------------
# BlackJAX NUTS sampling
# ---------------------------------------------------------------------------


@requires_extra("bayes")
def blackjax_nuts(
    log_prob_fn: Any,
    initial_position: Any,
    n_samples: int = 1000,
    step_size: float = 0.01,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run NUTS sampling using BlackJAX (JAX-based).

    Parameters
    ----------
    log_prob_fn : callable
        Log probability function (must be JAX-compatible).
    initial_position : jax.Array
        Initial parameter values.
    n_samples : int, default 1000
        Number of samples.
    step_size : float, default 0.01
        NUTS step size.
    seed : int or None
        Random seed integer.

    Returns
    -------
    dict
        ``samples``: array of posterior samples,
        ``divergences``: number of divergent transitions.
    """
    import jax
    import jax.numpy as jnp
    import blackjax

    key = jax.random.PRNGKey(seed or 0)
    # BlackJAX >= 1.0 requires inverse_mass_matrix; default to identity.
    if isinstance(initial_position, (int, float)):
        n_dim = 1
    elif hasattr(initial_position, "shape"):
        n_dim = initial_position.shape[0] if initial_position.ndim >= 1 else 1
    else:
        n_dim = len(initial_position)
    inv_mass = jnp.ones(n_dim)
    kernel = blackjax.nuts(log_prob_fn, step_size=step_size, inverse_mass_matrix=inv_mass)
    initial_state = kernel.init(initial_position)

    def one_step(state, key):
        state, info = kernel.step(key, state)
        return state, (state, info)

    keys = jax.random.split(key, n_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

    return {
        "samples": np.asarray(states.position),
        "divergences": int(np.sum(np.asarray(infos.is_divergent))),
    }
