"""Advanced Monte Carlo methods for risk measurement.

Variance reduction techniques, bootstrap methods, and filtered
historical simulation for improved VaR/ES estimation.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm as _norm

__all__ = [
    "antithetic_variates",
    "block_bootstrap",
    "filtered_historical_simulation",
    "importance_sampling_var",
    "stationary_bootstrap",
    "stratified_sampling",
]


def importance_sampling_var(
    returns: np.ndarray,
    n_sims: int = 10_000,
    target_quantile: float = 0.01,
    shift: float | None = None,
    seed: int | None = None,
) -> dict[str, float]:
    """Estimate VaR via importance sampling.

    Shifts the sampling distribution toward the tail to obtain more
    accurate estimates of extreme quantiles with fewer simulations.

    Parameters:
        returns: 1-D array of historical returns.
        n_sims: Number of Monte Carlo draws.
        target_quantile: Quantile level for VaR (e.g., 0.01 for 1%).
        shift: Mean shift for the importance sampling distribution.
            If ``None``, automatically set to the empirical quantile.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys:

        - ``var``: Estimated VaR (positive number = loss).
        - ``effective_sample_size``: Effective sample size after
          reweighting, as a fraction of ``n_sims``.
    """
    returns = np.asarray(returns, dtype=float)
    mu = np.mean(returns)
    sigma = np.std(returns)

    if sigma < 1e-15:
        return {"var": 0.0, "effective_sample_size": 1.0}

    if shift is None:
        shift = np.quantile(returns, target_quantile) - mu

    rng = np.random.default_rng(seed)

    # Draw from shifted distribution (importance distribution)
    samples = rng.normal(loc=mu + shift, scale=sigma, size=n_sims)

    # Likelihood ratios (original / importance)
    log_ratios = (
        -0.5 * ((samples - mu) / sigma) ** 2
        + 0.5 * ((samples - mu - shift) / sigma) ** 2
    )
    weights = np.exp(log_ratios)

    # Normalise weights
    weights /= weights.sum()

    # Sort samples and compute weighted quantile
    order = np.argsort(samples)
    sorted_samples = samples[order]
    sorted_weights = weights[order]
    cum_weights = np.cumsum(sorted_weights)

    idx = np.searchsorted(cum_weights, target_quantile)
    idx = min(idx, len(sorted_samples) - 1)
    var_estimate = -sorted_samples[idx]

    # Effective sample size
    ess = 1.0 / np.sum(weights**2) / n_sims

    return {
        "var": float(var_estimate),
        "effective_sample_size": float(ess),
    }


def antithetic_variates(
    mu: float | np.ndarray,
    sigma: float | np.ndarray,
    n_sims: int,
    n_assets: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    """Generate antithetic variate samples for variance reduction.

    Produces ``n_sims`` paired samples (original + antithetic) from a
    normal distribution.  The antithetic counterpart mirrors each draw
    about the mean, reducing variance for monotone functions.

    Parameters:
        mu: Mean(s) of the distribution.  Scalar or array of length
            ``n_assets``.
        sigma: Standard deviation(s).  Scalar or array of length
            ``n_assets``.
        n_sims: Number of *pairs* to generate (total output = 2 * n_sims
            rows).
        n_assets: Number of assets / dimensions.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape ``(2 * n_sims, n_assets)`` containing the
        original and antithetic draws interleaved.
    """
    rng = np.random.default_rng(seed)
    mu = np.broadcast_to(np.asarray(mu, dtype=float), (n_assets,))
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (n_assets,))

    z = rng.standard_normal((n_sims, n_assets))
    original = mu + sigma * z
    antithetic = mu - sigma * z

    # Interleave original and antithetic
    result = np.empty((2 * n_sims, n_assets))
    result[0::2] = original
    result[1::2] = antithetic

    return result


def stratified_sampling(
    returns: np.ndarray,
    n_strata: int = 10,
    n_sims: int = 10_000,
    seed: int | None = None,
) -> np.ndarray:
    """Stratified sampling for VaR estimation.

    Divides the probability space into equal strata and draws uniformly
    within each stratum, ensuring better coverage of the tails.

    Parameters:
        returns: 1-D array of historical returns (used to fit a normal
            distribution).
        n_strata: Number of strata.
        n_sims: Total number of draws (distributed evenly across strata).
        seed: Random seed for reproducibility.

    Returns:
        1-D array of ``n_sims`` stratified samples drawn from the fitted
        normal distribution.
    """
    returns = np.asarray(returns, dtype=float)
    mu = np.mean(returns)
    sigma = np.std(returns)

    rng = np.random.default_rng(seed)

    sims_per_stratum = n_sims // n_strata
    remainder = n_sims - sims_per_stratum * n_strata

    samples = []
    for i in range(n_strata):
        n_this = sims_per_stratum + (1 if i < remainder else 0)
        lo = i / n_strata
        hi = (i + 1) / n_strata
        # Uniform within the stratum
        u = rng.uniform(lo, hi, size=n_this)
        samples.append(_norm.ppf(u, loc=mu, scale=sigma))

    return np.concatenate(samples)


def block_bootstrap(
    returns: np.ndarray,
    block_size: int,
    n_sims: int = 1_000,
    seed: int | None = None,
) -> np.ndarray:
    """Block bootstrap for autocorrelated time series.

    Resamples contiguous blocks of the input series to preserve serial
    dependence (Kunsch 1989, Liu & Singh 1992).

    Parameters:
        returns: 1-D array of return observations.
        block_size: Length of each block.
        n_sims: Number of bootstrap replications.
        seed: Random seed for reproducibility.

    Returns:
        2-D array of shape ``(n_sims, len(returns))`` where each row is
        one bootstrap replicate.
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    if block_size < 1 or block_size > n:
        raise ValueError("block_size must be between 1 and len(returns)")

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))

    result = np.empty((n_sims, n))
    for i in range(n_sims):
        # Draw random block starting points
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        blocks = [returns[s : s + block_size] for s in starts]
        replicate = np.concatenate(blocks)[:n]
        result[i] = replicate

    return result


def stationary_bootstrap(
    returns: np.ndarray,
    avg_block_size: float = 10.0,
    n_sims: int = 1_000,
    seed: int | None = None,
) -> np.ndarray:
    """Stationary bootstrap with random block sizes (Politis & Romano 1994).

    Block lengths follow a geometric distribution with mean
    ``avg_block_size``, producing a strictly stationary resampled series.

    Parameters:
        returns: 1-D array of return observations.
        avg_block_size: Expected block size (must be > 1).
        n_sims: Number of bootstrap replications.
        seed: Random seed for reproducibility.

    Returns:
        2-D array of shape ``(n_sims, len(returns))`` where each row is
        one bootstrap replicate.
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    if avg_block_size < 1:
        raise ValueError("avg_block_size must be >= 1")

    rng = np.random.default_rng(seed)
    p = 1.0 / avg_block_size  # probability of starting new block

    result = np.empty((n_sims, n))
    for i in range(n_sims):
        idx = rng.integers(0, n)  # random starting point
        for j in range(n):
            result[i, j] = returns[idx % n]
            # With probability p, jump to a new random position
            if rng.random() < p:
                idx = rng.integers(0, n)
            else:
                idx += 1

    return result


def filtered_historical_simulation(
    returns: np.ndarray,
    vol_model: str = "ewma",
    decay: float = 0.94,
    n_sims: int = 1_000,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Filtered historical simulation (FHS).

    Combines a volatility model (EWMA or simple GARCH(1,1)) with
    historical bootstrap of standardised residuals to produce
    volatility-adjusted scenario returns.

    Parameters:
        returns: 1-D array of historical returns.
        vol_model: Volatility model — ``"ewma"`` (default) or ``"garch"``.
        decay: EWMA decay factor (lambda) or, for ``"garch"``, the
            persistence parameter beta.
        n_sims: Number of simulated next-period returns.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys:

        - ``simulated_returns``: 1-D array of ``n_sims`` simulated
          next-period returns.
        - ``current_vol``: Estimated current volatility used for
          rescaling.
        - ``standardised_residuals``: Standardised residuals from
          the volatility model.
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)

    if n < 3:
        raise ValueError("Need at least 3 return observations")

    if vol_model == "ewma":
        # EWMA variance
        var_t = np.zeros(n)
        var_t[0] = returns[0] ** 2
        for i in range(1, n):
            var_t[i] = decay * var_t[i - 1] + (1.0 - decay) * returns[i - 1] ** 2
    elif vol_model == "garch":
        # Simple GARCH(1,1): sigma^2_t = omega + alpha * r_{t-1}^2 + beta * sigma^2_{t-1}
        # Estimate omega and alpha from the data
        alpha = 1.0 - decay  # alpha = 1 - beta for a simple parameterisation
        omega = (
            np.var(returns) * (1.0 - decay - alpha)
            if (1.0 - decay - alpha) > 0
            else 1e-6
        )
        omega = max(omega, 1e-10)

        var_t = np.zeros(n)
        var_t[0] = np.var(returns)
        for i in range(1, n):
            var_t[i] = omega + alpha * returns[i - 1] ** 2 + decay * var_t[i - 1]
    else:
        msg = f"Unknown vol_model: {vol_model!r}"
        raise ValueError(msg)

    sigma_t = np.sqrt(np.maximum(var_t, 1e-15))
    standardised = returns / sigma_t
    current_vol = sigma_t[-1]

    # One-step-ahead volatility forecast
    if vol_model == "ewma":
        next_var = decay * var_t[-1] + (1.0 - decay) * returns[-1] ** 2
    else:
        next_var = omega + alpha * returns[-1] ** 2 + decay * var_t[-1]

    forecast_vol = np.sqrt(max(next_var, 1e-15))

    # Bootstrap standardised residuals and rescale
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=n_sims)
    simulated = forecast_vol * standardised[boot_idx]

    return {
        "simulated_returns": simulated,
        "current_vol": float(current_vol),
        "standardised_residuals": standardised,
    }
