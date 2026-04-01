"""MCMC sampling utilities using pure numpy/scipy.

Includes Metropolis-Hastings sampler, Gibbs sampler, NUTS diagnostics
(ESS, R-hat), trace summary tables, and the Gelman-Rubin convergence
diagnostic.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from wraquant.core._coerce import coerce_array

__all__ = [
    "metropolis_hastings",
    "gibbs_sampler",
    "nuts_diagnostic",
    "trace_summary",
    "gelman_rubin",
    "hamiltonian_monte_carlo",
    "slice_sampler",
    "convergence_diagnostics",
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

    The Metropolis-Hastings algorithm is the foundational MCMC method
    for sampling from an arbitrary (unnormalised) posterior distribution.
    At each step it proposes a new point from a symmetric Gaussian
    proposal, accepts or rejects based on the posterior ratio, and
    thereby generates a Markov chain whose stationary distribution is
    the target posterior.

    Use this when the posterior does not have a convenient closed form
    and you need posterior samples for uncertainty quantification,
    credible intervals, or posterior predictive checks.

    **Tuning**: Aim for an acceptance rate of 20--50%.  If the rate is
    too low, reduce ``proposal_std``; if too high, increase it.

    Parameters:
        log_posterior (callable): Function mapping a parameter vector
            (1-D array) to the log posterior density (up to a
            normalising constant).
        initial (np.ndarray): Initial parameter vector of shape
            ``(d,)``.
        n_samples (int): Total number of samples to draw (before
            burn-in and thinning).
        proposal_std (float | np.ndarray): Standard deviation(s) for
            the isotropic Gaussian proposal.  A scalar applies the
            same std to all dimensions; an array allows per-dimension
            tuning.
        burn_in (int): Number of initial samples to discard as
            warm-up.
        thin (int): Keep every *thin*-th sample to reduce
            autocorrelation.
        rng_seed (int): Random seed for reproducibility.

    Returns:
        dict: Dictionary with keys:

        - ``samples`` -- np.ndarray of shape ``(n_kept, d)``,
          posterior samples after burn-in and thinning.
        - ``acceptance_rate`` -- float, fraction of proposals accepted.
        - ``log_posteriors`` -- np.ndarray, log posterior values for
          each kept sample.

    Example:
        >>> import numpy as np
        >>> log_p = lambda q: -0.5 * np.sum((q - 2.0) ** 2)
        >>> result = metropolis_hastings(log_p, np.zeros(2),
        ...     n_samples=5000, burn_in=500, proposal_std=1.0)
        >>> result['samples'].shape[1]
        2

    See Also:
        hamiltonian_monte_carlo: Gradient-based sampler with higher
            acceptance rates in high dimensions.
        slice_sampler: Tuning-free univariate sampler.
        convergence_diagnostics: Assess MCMC convergence.
    """
    rng = np.random.default_rng(rng_seed)
    initial = coerce_array(initial, name="initial")
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
        "acceptance_rate": float(n_accepted / total_samples),
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

    Gibbs sampling is a special case of MCMC where each parameter is
    updated by drawing from its **full conditional** distribution
    (the distribution of that parameter given all others and the data).
    Because every draw is accepted, Gibbs sampling avoids the tuning
    burden of Metropolis-Hastings and is efficient when the full
    conditionals have known forms (e.g., conjugate models).

    Use Gibbs when you can analytically derive each conditional (common
    in Bayesian linear regression, mixture models, and hierarchical
    models).

    Parameters:
        conditionals (Sequence[callable]): A list of functions, one per
            parameter.  Each callable has signature
            ``fn(current_params, rng) -> float`` and returns a single
            draw from the full conditional of that parameter.
        initial (np.ndarray): Initial parameter vector of shape
            ``(d,)``.
        n_samples (int): Total number of samples to draw (before
            burn-in and thinning).
        burn_in (int): Number of initial samples to discard.
        thin (int): Keep every *thin*-th sample.
        rng_seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Posterior samples of shape ``(n_kept, d)``.

    Raises:
        ValueError: If the number of conditionals does not match the
            number of parameters.

    Example:
        >>> import numpy as np
        >>> # Gibbs for bivariate normal with rho=0
        >>> cond_0 = lambda p, rng: rng.normal(0, 1)
        >>> cond_1 = lambda p, rng: rng.normal(0, 1)
        >>> samples = gibbs_sampler([cond_0, cond_1], np.zeros(2),
        ...     n_samples=2000, burn_in=200)
        >>> samples.shape[1]
        2

    See Also:
        metropolis_hastings: General-purpose MCMC when conditionals
            are not available.
        hamiltonian_monte_carlo: Gradient-based MCMC.
    """
    rng = np.random.default_rng(rng_seed)
    current = coerce_array(initial, name="initial").copy()
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

    R-hat compares the between-chain variance to the within-chain
    variance.  If chains have converged to the same stationary
    distribution, these should be approximately equal (R-hat ~ 1).
    Large R-hat means the chains disagree, indicating they have not
    yet explored the same region of parameter space.

    Interpretation:
        - **R-hat < 1.01**: Excellent convergence. Safe to use samples.
        - **R-hat 1.01 - 1.05**: Acceptable, but consider running longer.
        - **R-hat 1.05 - 1.1**: Marginal. Results may be unreliable.
        - **R-hat > 1.1**: NOT converged. Do not use these samples for
          inference. Run longer, tune the sampler, or reparameterise.

    Requires at least 2 chains. Running multiple chains from different
    starting points is the gold standard for diagnosing convergence.

    Parameters:
        chains: Either a 3D array of shape (n_chains, n_samples,
            n_params) or a list of 2D arrays each of shape
            (n_samples, n_params).

    Returns:
        R-hat values for each parameter, shape (n_params,).

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> chains = rng.normal(size=(4, 2000, 3))
        >>> r_hat = gelman_rubin(chains)
        >>> print(f"R-hat: {r_hat}")  # Should be ~1.0

    See Also:
        convergence_diagnostics: Full battery including split R-hat,
            ESS, and MCSE.
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

    Provides the minimum set of diagnostics that every MCMC analysis
    should report.  Can be used on output from any sampler, not just
    NUTS.

    Interpretation:
        - **ESS** (Effective Sample Size): How many independent samples
          your chain is worth, accounting for autocorrelation.  If you
          drew 10,000 samples but ESS = 500, your estimates are only
          as precise as 500 independent draws.  Rule of thumb: ESS >
          400 for reliable posterior summaries, ESS > 1000 for tail
          quantiles.
        - **R-hat**: Convergence diagnostic (see ``gelman_rubin``).
          Only meaningful with multiple chains.
        - **mean/std**: Posterior summaries.  MCSE = std / sqrt(ESS)
          gives the precision of the posterior mean estimate.

    When to use:
        - After any MCMC run, before using the samples for inference.
        - To decide whether to run the sampler longer.

    Parameters:
        samples: Posterior samples. If 2D (n_samples, n_params),
            treated as a single chain. If 3D (n_chains, n_samples,
            n_params), R-hat is computed across chains.
        chains: Optional multi-chain samples for R-hat computation.
            If provided, must be 3D (n_chains, n_samples, n_params).

    Returns:
        Dictionary containing:

        - **ess** (*ndarray*) -- Effective sample size per parameter.
        - **r_hat** (*ndarray*) -- R-hat per parameter (NaN if single
          chain).
        - **mean** (*ndarray*) -- Posterior mean per parameter.
        - **std** (*ndarray*) -- Posterior standard deviation.

    Example:
        >>> import numpy as np
        >>> samples = np.random.default_rng(0).normal(size=(5000, 3))
        >>> diag = nuts_diagnostic(samples)
        >>> print(f"ESS: {diag['ess']}")  # Should be ~5000 for iid
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
    """Compute a publication-ready summary table for MCMC samples.

    This is the standard output table that every Bayesian analysis
    should report.  It combines posterior summaries (mean, std,
    quantiles) with convergence diagnostics (ESS, R-hat) in a single
    DataFrame.

    Interpretation:
        - **mean** and **std**: point estimate and uncertainty.
        - **2.5% and 97.5%**: bounds of the 95% credible interval.
          If this interval excludes 0, the parameter is "significant"
          in a Bayesian sense.
        - **50%** (median): more robust than mean for skewed posteriors.
        - **ESS**: effective sample size. If ESS < 400, the quantile
          estimates (especially the tails) may be unreliable.
        - **R-hat**: convergence diagnostic. R-hat > 1.05 is a red
          flag; do not trust the results.

    Parameters:
        samples: Posterior samples of shape (n_samples, n_params) or
            (n_chains, n_samples, n_params).
        param_names: Parameter names. Defaults to
            ``['param_0', 'param_1', ...]``.
        quantiles: Quantiles to compute. Default includes the 95%
            credible interval endpoints and median.
        chains: Optional multi-chain samples for R-hat. If None and
            samples is 3D, R-hat is computed from the 3D array.

    Returns:
        pd.DataFrame with columns: mean, std, quantiles, ESS, R-hat.
        One row per parameter.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> samples = rng.normal(size=(5000, 3))
        >>> summary = trace_summary(samples, param_names=['alpha', 'beta', 'sigma'])
        >>> print(summary)
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


# ---------------------------------------------------------------------------
# Hamiltonian Monte Carlo (pure numpy)
# ---------------------------------------------------------------------------


def hamiltonian_monte_carlo(
    log_posterior: Callable[[np.ndarray], float],
    grad_log_posterior: Callable[[np.ndarray], np.ndarray],
    initial: np.ndarray,
    n_samples: int = 5_000,
    step_size: float = 0.01,
    n_leapfrog: int = 20,
    burn_in: int = 1_000,
    mass_matrix: np.ndarray | None = None,
    rng_seed: int = 42,
) -> dict[str, np.ndarray | float]:
    """Hamiltonian Monte Carlo sampler (pure numpy, no JAX/PyMC needed).

    HMC uses Hamiltonian dynamics to propose distant points in parameter
    space that are likely to be accepted, dramatically reducing the
    random-walk behaviour of Metropolis-Hastings.  This makes it much
    more efficient for correlated, high-dimensional posteriors.

    The algorithm works by augmenting the parameter space with a
    "momentum" variable and simulating Hamiltonian dynamics using the
    leapfrog integrator.  The total energy (Hamiltonian) is approximately
    conserved, leading to high acceptance rates even for large jumps.

    **When to use HMC**: Use HMC when Metropolis-Hastings mixes poorly
    (high autocorrelation, low effective sample size) or when the
    posterior has strong correlations between parameters.  HMC requires
    the gradient of the log-posterior, which is the main practical
    barrier.

    **Tuning**: The two key parameters are ``step_size`` (too large
    causes rejections; too small wastes computation) and ``n_leapfrog``
    (too few gives random-walk behaviour; too many wastes computation).
    A good heuristic is to aim for acceptance rates of 60--80 %.

    Args:
        log_posterior: Function mapping parameter vector to log posterior
            density (up to a normalising constant).
        grad_log_posterior: Function mapping parameter vector to the
            gradient of the log posterior (a vector of the same length).
        initial: Initial parameter vector of shape ``(d,)``.
        n_samples: Number of samples to draw (before burn-in).
        step_size: Leapfrog step size (epsilon).  Typical range
            0.001 -- 0.1 depending on the scale of the problem.
        n_leapfrog: Number of leapfrog steps per proposal (L).
        burn_in: Number of initial samples to discard.
        mass_matrix: Diagonal mass matrix of shape ``(d,)``.  If None,
            uses identity (unit mass for all parameters).  Set this to
            the inverse of the marginal posterior variances for better
            performance.
        rng_seed: Random seed for reproducibility.

    Returns:
        Dictionary with:
            - ``samples``: np.ndarray of shape ``(n_samples, d)`` --
              posterior samples after burn-in.
            - ``acceptance_rate``: float -- fraction of proposals
              accepted.
            - ``log_posteriors``: np.ndarray -- log posterior values for
              each kept sample.

    Example:
        >>> import numpy as np
        >>> # Sample from N(3, 1)
        >>> log_p = lambda q: -0.5 * (q[0] - 3.0) ** 2
        >>> grad_log_p = lambda q: np.array([-(q[0] - 3.0)])
        >>> result = hamiltonian_monte_carlo(log_p, grad_log_p,
        ...     initial=np.array([0.0]), n_samples=5000, burn_in=500)
        >>> print(f"Mean: {result['samples'].mean():.1f}")  # ~3.0
    """
    rng = np.random.default_rng(rng_seed)
    q = coerce_array(initial, name="initial").copy()
    d = len(q)

    if mass_matrix is None:
        mass_matrix = np.ones(d)
    else:
        mass_matrix = np.asarray(mass_matrix, dtype=float).ravel()

    inv_mass = 1.0 / mass_matrix

    total = n_samples + burn_in
    samples = np.zeros((total, d))
    log_posts = np.zeros(total)
    n_accepted = 0

    current_lp = log_posterior(q)

    for i in range(total):
        # Sample momentum
        p = rng.normal(0, np.sqrt(mass_matrix))
        current_p = p.copy()
        current_q = q.copy()

        # Leapfrog integration
        grad = grad_log_posterior(q)
        p = p + 0.5 * step_size * grad  # half step for momentum

        for step in range(n_leapfrog - 1):
            q = q + step_size * inv_mass * p  # full step for position
            grad = grad_log_posterior(q)
            p = p + step_size * grad  # full step for momentum

        q = q + step_size * inv_mass * p  # final full step for position
        grad = grad_log_posterior(q)
        p = p + 0.5 * step_size * grad  # half step for momentum

        # Negate momentum (for reversibility, though not needed for MH)
        p = -p

        # Compute Hamiltonian
        proposed_lp = log_posterior(q)
        current_K = 0.5 * np.sum(current_p**2 * inv_mass)
        proposed_K = 0.5 * np.sum(p**2 * inv_mass)

        # Metropolis acceptance
        log_alpha = (proposed_lp - proposed_K) - (current_lp - current_K)

        if np.isfinite(log_alpha) and np.log(rng.uniform()) < log_alpha:
            # Accept
            current_lp = proposed_lp
            n_accepted += 1
        else:
            # Reject: revert
            q = current_q

        samples[i] = q.copy()
        log_posts[i] = current_lp

    return {
        "samples": samples[burn_in:],
        "acceptance_rate": float(n_accepted / total),
        "log_posteriors": log_posts[burn_in:],
    }


# ---------------------------------------------------------------------------
# Slice sampler
# ---------------------------------------------------------------------------


def slice_sampler(
    log_posterior: Callable[[float], float],
    initial: float = 0.0,
    n_samples: int = 5_000,
    burn_in: int = 500,
    w: float = 1.0,
    rng_seed: int = 42,
) -> np.ndarray:
    """Univariate slice sampler (Neal, 2003).

    Slice sampling is an adaptive MCMC method that requires **no tuning**
    of a proposal distribution.  It works by:

    1. Drawing a horizontal "slice" under the (unnormalised) density at
       the current point.
    2. Finding an interval around the current point that contains the
       slice.
    3. Sampling uniformly from that interval, shrinking it if the
       proposed point falls outside the slice.

    This makes it more robust than Metropolis-Hastings (no rejected
    samples, no proposal std to tune) while being simple to implement.

    **When to use this**: Use slice sampling for univariate conditional
    distributions inside a Gibbs sampler, or for simple 1D posteriors
    where you don't want to worry about tuning.

    Args:
        log_posterior: Function mapping a scalar to the log (unnormalised)
            posterior density.
        initial: Starting value.
        n_samples: Number of samples to keep (after burn-in).
        burn_in: Number of initial samples to discard.
        w: Initial bracket width.  The algorithm adapts from this, so
            the exact value is not critical.  A rough estimate of the
            posterior std is a good choice.
        rng_seed: Random seed.

    Returns:
        np.ndarray of shape ``(n_samples,)`` -- posterior samples.

    Example:
        >>> import numpy as np
        >>> # Sample from N(2, 1)
        >>> log_p = lambda x: -0.5 * (x - 2.0) ** 2
        >>> samples = slice_sampler(log_p, initial=0.0, n_samples=5000)
        >>> print(f"Mean: {samples.mean():.1f}")  # ~2.0
    """
    rng = np.random.default_rng(rng_seed)
    total = n_samples + burn_in
    samples = np.zeros(total)

    x = float(initial)
    lp_x = log_posterior(x)

    for i in range(total):
        # Draw the slice level
        log_y = lp_x + np.log(rng.uniform())

        # Step out: find the bracket [L, R]
        u = rng.uniform()
        L = x - w * u
        R = L + w

        # Expand bracket
        max_steps = 100
        j = 0
        while log_posterior(L) > log_y and j < max_steps:
            L -= w
            j += 1
        j = 0
        while log_posterior(R) > log_y and j < max_steps:
            R += w
            j += 1

        # Shrink and sample
        for _ in range(200):  # safety limit
            x_prime = L + rng.uniform() * (R - L)
            lp_prime = log_posterior(x_prime)

            if lp_prime > log_y:
                x = x_prime
                lp_x = lp_prime
                break

            if x_prime < x:
                L = x_prime
            else:
                R = x_prime
        else:
            # Fallback: keep current sample
            pass

        samples[i] = x

    return samples[burn_in:]


# ---------------------------------------------------------------------------
# Enhanced convergence diagnostics
# ---------------------------------------------------------------------------


def convergence_diagnostics(
    chains: np.ndarray | list[np.ndarray],
    param_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Comprehensive MCMC convergence diagnostics for multi-chain output.

    Computes a battery of diagnostics that every Bayesian analysis should
    report:

    - **Split R-hat**: The Gelman-Rubin statistic computed on split
      chains (each chain is split in half), which is more sensitive to
      non-stationarity than the original R-hat.  Values below 1.01 are
      considered good; below 1.05 is acceptable.

    - **Effective Sample Size (ESS)**: The number of effectively
      independent samples, accounting for autocorrelation.  Low ESS
      means your inferences are based on fewer independent data points
      than you might think.

    - **ESS per second**: Not computed here (would need timing info),
      but ESS alone tells you if you need to run longer.

    - **Autocorrelation time**: The number of steps between effectively
      independent samples.  ``tau = n / ESS``.

    - **MCSE** (Monte Carlo Standard Error): The standard error of the
      posterior mean estimate due to finite sampling.
      ``MCSE = posterior_std / sqrt(ESS)``.

    **Interpreting the results**: Your MCMC has converged if *all* of:
    (1) R-hat < 1.05, (2) ESS > 400, (3) trace plots show no trends.
    If R-hat > 1.1, you need more samples or better tuning.

    Args:
        chains: Multi-chain samples, either a 3D array of shape
            ``(n_chains, n_samples, n_params)`` or a list of 2D arrays.
        param_names: Optional parameter names.  Defaults to
            ``['param_0', ...]``.

    Returns:
        pd.DataFrame with columns: ``mean``, ``std``, ``r_hat``,
        ``split_r_hat``, ``ess``, ``autocorrelation_time``, ``mcse``.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> chains = rng.normal(size=(4, 2000, 3))
        >>> diag = convergence_diagnostics(chains)
        >>> print(diag[['split_r_hat', 'ess']])
    """
    if isinstance(chains, list):
        chains = np.array(chains)
    chains = np.asarray(chains, dtype=float)

    if chains.ndim == 2:
        # Single chain: split into two halves for split-R-hat
        n_half = chains.shape[0] // 2
        chains = np.stack([chains[:n_half], chains[n_half : 2 * n_half]])

    if chains.ndim != 3:
        raise ValueError(f"chains must be 2D or 3D, got {chains.ndim}D.")

    m, n, k = chains.shape

    if param_names is None:
        param_names = [f"param_{j}" for j in range(k)]

    # Standard R-hat
    r_hat = gelman_rubin(chains)

    # Split R-hat: split each chain in half
    n_half = n // 2
    split_chains = np.zeros((2 * m, n_half, k))
    for c in range(m):
        split_chains[2 * c] = chains[c, :n_half, :]
        split_chains[2 * c + 1] = chains[c, n_half : 2 * n_half, :]
    split_r_hat = gelman_rubin(split_chains)

    # Flatten for ESS and summary stats
    flat = chains.reshape(-1, k)
    means = np.mean(flat, axis=0)
    stds = np.std(flat, axis=0, ddof=1)

    ess = np.array([_ess_single(flat[:, j]) for j in range(k)])
    autocorr_time = (m * n) / np.maximum(ess, 1.0)
    mcse = stds / np.sqrt(np.maximum(ess, 1.0))

    data = {
        "mean": means.tolist(),
        "std": stds.tolist(),
        "r_hat": r_hat.tolist(),
        "split_r_hat": split_r_hat.tolist(),
        "ess": ess.tolist(),
        "autocorrelation_time": autocorr_time.tolist(),
        "mcse": mcse.tolist(),
    }

    return pd.DataFrame(data, index=param_names)
