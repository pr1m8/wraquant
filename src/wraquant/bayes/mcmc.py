"""MCMC sampling utilities using pure numpy/scipy.

Includes Metropolis-Hastings sampler, Gibbs sampler, NUTS diagnostics
(ESS, R-hat), trace summary tables, and the Gelman-Rubin convergence
diagnostic.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "metropolis_hastings",
    "gibbs_sampler",
    "nuts_diagnostic",
    "trace_summary",
    "gelman_rubin",
]


# ---------------------------------------------------------------------------
# Metropolis-Hastings sampler
# ---------------------------------------------------------------------------


def metropolis_hastings(
    log_posterior: Callable[[np.ndarray], float],
    initial: np.ndarray,
    n_samples: int = 10_000,
    proposal_std: float | np.ndarray = 1.0,
    burn_in: int = 1_000,
    thin: int = 1,
    rng_seed: int = 42,
) -> dict[str, np.ndarray | float]:
    """Random-walk Metropolis-Hastings sampler.

    Parameters
    ----------
    log_posterior : callable
        Function that takes a parameter vector and returns the log
        posterior density (up to a normalizing constant).
    initial : np.ndarray
        Initial parameter vector.
    n_samples : int
        Total number of samples to draw (before burn-in and thinning).
    proposal_std : float or np.ndarray
        Standard deviation(s) for the Gaussian proposal distribution.
        Can be a scalar or a vector of the same length as initial.
    burn_in : int
        Number of initial samples to discard.
    thin : int
        Keep every ``thin``-th sample.
    rng_seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``samples``: np.ndarray of shape (n_kept, n_params) — posterior samples,
        ``acceptance_rate``: float — fraction of proposals accepted,
        ``log_posteriors``: np.ndarray — log posterior values for kept samples.
    """
    rng = np.random.default_rng(rng_seed)
    initial = np.asarray(initial, dtype=float).ravel()
    n_params = len(initial)

    if np.isscalar(proposal_std):
        proposal_std = np.full(n_params, float(proposal_std))
    else:
        proposal_std = np.asarray(proposal_std, dtype=float).ravel()

    total_samples = n_samples + burn_in
    all_samples = np.zeros((total_samples, n_params))
    all_log_post = np.zeros(total_samples)

    current = initial.copy()
    current_lp = log_posterior(current)
    n_accepted = 0

    for i in range(total_samples):
        proposal = current + rng.normal(0, proposal_std)
        proposal_lp = log_posterior(proposal)

        log_alpha = proposal_lp - current_lp
        if np.log(rng.uniform()) < log_alpha:
            current = proposal
            current_lp = proposal_lp
            n_accepted += 1

        all_samples[i] = current
        all_log_post[i] = current_lp

    # Discard burn-in and thin
    kept_samples = all_samples[burn_in::thin]
    kept_log_post = all_log_post[burn_in::thin]

    return {
        "samples": kept_samples,
        "acceptance_rate": n_accepted / total_samples,
        "log_posteriors": kept_log_post,
    }


# ---------------------------------------------------------------------------
# Gibbs sampler
# ---------------------------------------------------------------------------


def gibbs_sampler(
    conditionals: Sequence[Callable[[np.ndarray, np.random.Generator], float]],
    initial: np.ndarray,
    n_samples: int = 10_000,
    burn_in: int = 1_000,
    thin: int = 1,
    rng_seed: int = 42,
) -> np.ndarray:
    """Gibbs sampler for a model specified via full conditional distributions.

    Parameters
    ----------
    conditionals : sequence of callables
        A list of functions, one per parameter. Each function takes
        (current_params, rng) and returns a single sample from the full
        conditional distribution of that parameter given all others.
    initial : np.ndarray
        Initial parameter vector.
    n_samples : int
        Total number of samples to draw (before burn-in and thinning).
    burn_in : int
        Number of initial samples to discard.
    thin : int
        Keep every ``thin``-th sample.
    rng_seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Posterior samples of shape (n_kept, n_params).
    """
    rng = np.random.default_rng(rng_seed)
    current = np.asarray(initial, dtype=float).ravel().copy()
    n_params = len(current)

    if len(conditionals) != n_params:
        raise ValueError(
            f"Number of conditionals ({len(conditionals)}) must match "
            f"number of parameters ({n_params})."
        )

    total_samples = n_samples + burn_in
    all_samples = np.zeros((total_samples, n_params))

    for i in range(total_samples):
        for j in range(n_params):
            current[j] = conditionals[j](current, rng)
        all_samples[i] = current.copy()

    return all_samples[burn_in::thin]


# ---------------------------------------------------------------------------
# Effective sample size
# ---------------------------------------------------------------------------


def _ess_single(x: np.ndarray) -> float:
    """Compute effective sample size for a single chain using autocorrelation."""
    n = len(x)
    if n < 4:
        return float(n)

    x_centered = x - np.mean(x)
    var = np.var(x_centered, ddof=0)
    if var < 1e-15:
        return float(n)

    # Use FFT-based autocorrelation
    fft_x = np.fft.fft(x_centered, n=2 * n)
    acf = np.real(np.fft.ifft(fft_x * np.conj(fft_x)))[:n]
    acf = acf / acf[0]

    # Sum autocorrelations until they become negative (initial monotone sequence)
    tau = 1.0
    for lag in range(1, n):
        if acf[lag] < 0:
            break
        tau += 2.0 * acf[lag]

    return max(1.0, n / tau)


# ---------------------------------------------------------------------------
# Gelman-Rubin diagnostic
# ---------------------------------------------------------------------------


def gelman_rubin(chains: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """Compute the Gelman-Rubin (R-hat) convergence diagnostic.

    Parameters
    ----------
    chains : np.ndarray or list of np.ndarray
        Either a 3D array of shape (n_chains, n_samples, n_params) or a
        list of 2D arrays each of shape (n_samples, n_params).

    Returns
    -------
    np.ndarray
        R-hat values for each parameter. Values close to 1.0 indicate
        convergence.
    """
    if isinstance(chains, list):
        chains = np.array(chains)

    chains = np.asarray(chains, dtype=float)
    if chains.ndim == 2:
        # Single parameter, multiple chains: (n_chains, n_samples)
        chains = chains[:, :, np.newaxis]

    m, n, k = chains.shape  # m chains, n samples, k params

    r_hat = np.zeros(k)
    for j in range(k):
        chain_means = np.mean(chains[:, :, j], axis=1)  # (m,)
        chain_vars = np.var(chains[:, :, j], axis=1, ddof=1)  # (m,)

        grand_mean = np.mean(chain_means)
        B = n * np.var(chain_means, ddof=1)  # between-chain variance
        W = np.mean(chain_vars)  # within-chain variance

        if W < 1e-15:
            r_hat[j] = 1.0
        else:
            var_hat = (1 - 1.0 / n) * W + B / n
            r_hat[j] = np.sqrt(var_hat / W)

    return r_hat


# ---------------------------------------------------------------------------
# NUTS diagnostic statistics
# ---------------------------------------------------------------------------


def nuts_diagnostic(
    samples: np.ndarray,
    chains: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute NUTS-style diagnostic statistics (ESS, R-hat, etc.).

    Can be used on output from any sampler, not just NUTS.

    Parameters
    ----------
    samples : np.ndarray
        Posterior samples. If 2D (n_samples, n_params), treated as a
        single chain. If 3D (n_chains, n_samples, n_params), R-hat is
        computed across chains.
    chains : np.ndarray or None
        Optional multi-chain samples for R-hat computation. If provided,
        must be 3D (n_chains, n_samples, n_params).

    Returns
    -------
    dict
        ``ess``: np.ndarray — effective sample size per parameter,
        ``r_hat``: np.ndarray — R-hat per parameter (NaN if single chain),
        ``mean``: np.ndarray — posterior mean,
        ``std``: np.ndarray — posterior standard deviation.
    """
    samples = np.asarray(samples, dtype=float)

    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    if samples.ndim == 3:
        # Multi-chain: compute R-hat, then flatten for ESS
        r_hat = gelman_rubin(samples)
        flat = samples.reshape(-1, samples.shape[2])
        ess = np.array([_ess_single(flat[:, j]) for j in range(flat.shape[1])])
        mean = np.mean(flat, axis=0)
        std = np.std(flat, axis=0, ddof=1)
    elif samples.ndim == 2:
        n_params = samples.shape[1]
        ess = np.array([_ess_single(samples[:, j]) for j in range(n_params)])
        mean = np.mean(samples, axis=0)
        std = np.std(samples, axis=0, ddof=1)

        if chains is not None:
            r_hat = gelman_rubin(chains)
        else:
            r_hat = np.full(n_params, np.nan)
    else:
        raise ValueError(f"samples must be 2D or 3D, got {samples.ndim}D.")

    return {
        "ess": ess,
        "r_hat": r_hat,
        "mean": mean,
        "std": std,
    }


# ---------------------------------------------------------------------------
# Trace summary
# ---------------------------------------------------------------------------


def trace_summary(
    samples: np.ndarray,
    param_names: Sequence[str] | None = None,
    quantiles: tuple[float, ...] = (0.025, 0.25, 0.5, 0.75, 0.975),
    chains: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute a summary table for MCMC samples.

    Parameters
    ----------
    samples : np.ndarray
        Posterior samples of shape (n_samples, n_params) or
        (n_chains, n_samples, n_params).
    param_names : sequence of str or None
        Parameter names. Defaults to ``['param_0', 'param_1', ...]``.
    quantiles : tuple of float
        Quantiles to compute.
    chains : np.ndarray or None
        Optional multi-chain samples for R-hat. If None and samples is
        3D, R-hat is computed from the 3D array.

    Returns
    -------
    pd.DataFrame
        Summary table with columns: mean, std, quantiles, ESS, R-hat.
    """
    samples = np.asarray(samples, dtype=float)

    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    if samples.ndim == 3:
        diag = nuts_diagnostic(samples)
        flat = samples.reshape(-1, samples.shape[2])
    else:
        diag = nuts_diagnostic(samples, chains=chains)
        flat = samples

    n_params = flat.shape[1]

    if param_names is None:
        param_names = [f"param_{i}" for i in range(n_params)]

    data: dict[str, list[float]] = {
        "mean": diag["mean"].tolist(),
        "std": diag["std"].tolist(),
    }

    for q in quantiles:
        data[f"{q:.1%}"] = [float(np.quantile(flat[:, j], q)) for j in range(n_params)]

    data["ess"] = diag["ess"].tolist()
    data["r_hat"] = diag["r_hat"].tolist()

    return pd.DataFrame(data, index=param_names)
