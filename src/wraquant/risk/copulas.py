"""Copula models for dependency structure in multi-asset returns.

Copulas separate the *marginal* distributions of individual assets from
their *dependence* structure. This is critical in finance because linear
correlation (Pearson) understates co-movement in the tails -- precisely
where risk matters most. Copulas capture non-linear, asymmetric, and
tail-specific dependence that correlation matrices cannot.

This module provides five copula families:

1. **Gaussian copula** (``fit_gaussian_copula``) -- the simplest elliptical
   copula. Captures linear dependence but has *zero* tail dependence: extreme
   co-movements are asymptotically independent. Useful as a baseline but
   dangerous for tail-risk applications.

2. **Student-t copula** (``fit_t_copula``) -- symmetric tail dependence
   controlled by degrees of freedom. Lower df => heavier tails and
   stronger tail dependence. The standard choice for equity portfolios
   where joint crashes are common.

3. **Clayton copula** (``fit_clayton_copula``) -- Archimedean copula with
   *lower* tail dependence and no upper tail dependence. Use when you
   expect assets to crash together but rally independently (typical for
   equities and credit).

4. **Gumbel copula** (``fit_gumbel_copula``) -- Archimedean copula with
   *upper* tail dependence and no lower tail dependence. Use for assets
   that rally together but crash independently (rare but possible for
   certain commodity pairs).

5. **Frank copula** (``fit_frank_copula``) -- Archimedean copula with
   *symmetric* dependence and *no* tail dependence. Useful as a
   benchmark or when tail dependence is genuinely absent.

Utilities:
    - ``copula_simulate``: Monte Carlo simulation from any fitted copula.
    - ``tail_dependence``: empirical estimation of lower and upper tail
      dependence coefficients.
    - ``rank_correlation``: Kendall's tau and Spearman's rho matrices.

How to choose:
    - Start with ``tail_dependence`` to check if the data has
      asymmetric tail dependence.
    - If lower > upper: use Clayton.
    - If upper > lower: use Gumbel.
    - If roughly symmetric tails: use Student-t with estimated df.
    - If no significant tail dependence: use Gaussian or Frank.

References:
    - Nelsen (2006), "An Introduction to Copulas"
    - McNeil, Frey & Embrechts (2005), Ch. 5, "Copulas and Dependence"
    - Embrechts, Lindskog & McNeil (2003), "Modelling Dependence with
      Copulas and Applications to Risk Management"
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empirical_cdf(x: np.ndarray) -> np.ndarray:
    """Convert raw observations to pseudo-observations (uniform marginals).

    Uses the rescaled empirical CDF:  rank / (n + 1) to avoid 0 and 1.
    """
    n = len(x)
    ranks = sp_stats.rankdata(x)
    return ranks / (n + 1)


# ---------------------------------------------------------------------------
# Gaussian copula
# ---------------------------------------------------------------------------


def fit_gaussian_copula(
    returns: np.ndarray,
) -> dict[str, Any]:
    """Fit a Gaussian copula to multivariate returns.

    Transforms each marginal to uniform via the empirical CDF, then
    maps to standard normal and estimates the correlation matrix.

    Parameters:
        returns: Array of shape ``(n_obs, n_assets)`` with raw returns.

    Returns:
        Dict with keys:

        * ``"correlation"`` -- estimated Gaussian copula correlation
          matrix (n_assets, n_assets).
        * ``"copula_type"`` -- ``"gaussian"``.
    """
    n_obs, n_assets = returns.shape
    u = np.column_stack([_empirical_cdf(returns[:, j]) for j in range(n_assets)])
    z = sp_stats.norm.ppf(u)

    # Clip to avoid inf at boundaries (shouldn't happen with n+1 rule)
    z = np.clip(z, -8, 8)

    corr = np.corrcoef(z, rowvar=False)

    return {
        "correlation": corr,
        "copula_type": "gaussian",
    }


# ---------------------------------------------------------------------------
# Student-t copula
# ---------------------------------------------------------------------------


def fit_t_copula(
    returns: np.ndarray,
    df: float = 5.0,
) -> dict[str, Any]:
    """Fit a Student-t copula to multivariate returns.

    Transforms marginals to uniform, maps through the inverse t-CDF,
    and estimates the shape (correlation) matrix.  The degrees-of-freedom
    parameter is user-supplied (profile likelihood for *df* is expensive
    and often unstable).

    Parameters:
        returns: Array of shape ``(n_obs, n_assets)``.
        df: Degrees of freedom for the t-copula.

    Returns:
        Dict with keys:

        * ``"correlation"`` -- shape correlation matrix.
        * ``"df"`` -- degrees of freedom.
        * ``"copula_type"`` -- ``"student_t"``.
    """
    n_obs, n_assets = returns.shape
    u = np.column_stack([_empirical_cdf(returns[:, j]) for j in range(n_assets)])
    t_vals = sp_stats.t.ppf(u, df=df)
    t_vals = np.clip(t_vals, -50, 50)

    corr = np.corrcoef(t_vals, rowvar=False)

    return {
        "correlation": corr,
        "df": df,
        "copula_type": "student_t",
    }


# ---------------------------------------------------------------------------
# Archimedean copulas (bivariate)
# ---------------------------------------------------------------------------


def _clayton_cdf(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """Clayton copula CDF: C(u,v) = (u^{-theta} + v^{-theta} - 1)^{-1/theta}."""
    return np.power(
        np.power(u, -theta) + np.power(v, -theta) - 1,
        -1.0 / theta,
    )


def _clayton_log_density(u: np.ndarray, v: np.ndarray, theta: float) -> float:
    """Log-likelihood of the Clayton copula density."""
    n = len(u)
    log_c = (
        n * np.log(1 + theta)
        + (-theta - 1) * (np.sum(np.log(u)) + np.sum(np.log(v)))
        + (-2 - 1.0 / theta)
        * np.sum(np.log(np.power(u, -theta) + np.power(v, -theta) - 1))
    )
    return float(log_c)


def fit_clayton_copula(
    u: np.ndarray,
    v: np.ndarray,
) -> dict[str, Any]:
    """Fit a Clayton copula (lower tail dependence) to bivariate data.

    *u* and *v* should already be on ``(0, 1)`` (pseudo-observations).
    If raw data is passed, the empirical CDF is applied automatically.

    Parameters:
        u: First marginal uniform observations.
        v: Second marginal uniform observations.

    Returns:
        Dict with keys:

        * ``"theta"`` -- estimated Clayton parameter (> 0).
        * ``"copula_type"`` -- ``"clayton"``.
        * ``"lower_tail_dependence"`` -- ``2^{-1/theta}``.
    """
    u, v = _ensure_uniform(u), _ensure_uniform(v)

    def neg_ll(params: np.ndarray) -> float:
        theta = params[0]
        if theta <= 0:
            return 1e12
        try:
            return -_clayton_log_density(u, v, theta)
        except (FloatingPointError, ValueError):
            return 1e12

    # Kendall's tau initial estimate: theta = 2*tau/(1-tau)
    tau, _ = sp_stats.kendalltau(u, v)
    tau = max(tau, 0.05)
    theta_init = 2 * tau / (1 - tau)
    theta_init = max(theta_init, 0.1)

    res = optimize.minimize(
        neg_ll,
        x0=[theta_init],
        method="Nelder-Mead",
        options={"maxiter": 5000},
    )
    theta = max(float(res.x[0]), 1e-6)

    return {
        "theta": theta,
        "copula_type": "clayton",
        "lower_tail_dependence": 2 ** (-1.0 / theta),
    }


def _gumbel_log_density(u: np.ndarray, v: np.ndarray, theta: float) -> float:
    """Log-likelihood of the Gumbel copula density."""
    lu = -np.log(u)
    lv = -np.log(v)
    A = np.power(lu, theta) + np.power(lv, theta)
    A_inv = np.power(A, 1.0 / theta)

    # log C(u,v)
    log_C = -A_inv

    # log c(u,v) = log C + log[ ... ]
    # Full density:
    # c(u,v) = C(u,v) * (1/(u*v)) * (lu*lv)^(theta-1) / A^(2-1/theta)
    #          * (A_inv + theta - 1)
    log_density = (
        log_C
        - np.log(u)
        - np.log(v)
        + (theta - 1) * (np.log(lu) + np.log(lv))
        - (2 - 1.0 / theta) * np.log(A)
        + np.log(A_inv + theta - 1)
    )
    return float(np.sum(log_density))


def fit_gumbel_copula(
    u: np.ndarray,
    v: np.ndarray,
) -> dict[str, Any]:
    """Fit a Gumbel copula (upper tail dependence) to bivariate data.

    Parameters:
        u: First marginal uniform observations.
        v: Second marginal uniform observations.

    Returns:
        Dict with keys:

        * ``"theta"`` -- estimated Gumbel parameter (>= 1).
        * ``"copula_type"`` -- ``"gumbel"``.
        * ``"upper_tail_dependence"`` -- ``2 - 2^{1/theta}``.
    """
    u, v = _ensure_uniform(u), _ensure_uniform(v)

    def neg_ll(params: np.ndarray) -> float:
        theta = params[0]
        if theta < 1.0:
            return 1e12
        try:
            return -_gumbel_log_density(u, v, theta)
        except (FloatingPointError, ValueError):
            return 1e12

    tau, _ = sp_stats.kendalltau(u, v)
    tau = max(tau, 0.05)
    theta_init = 1.0 / (1 - tau)
    theta_init = max(theta_init, 1.01)

    res = optimize.minimize(
        neg_ll,
        x0=[theta_init],
        method="Nelder-Mead",
        options={"maxiter": 5000},
    )
    theta = max(float(res.x[0]), 1.0)

    return {
        "theta": theta,
        "copula_type": "gumbel",
        "upper_tail_dependence": 2 - 2 ** (1.0 / theta),
    }


def _frank_log_density(u: np.ndarray, v: np.ndarray, theta: float) -> float:
    """Log-likelihood of the Frank copula density."""
    # c(u,v) = theta * (1 - exp(-theta)) * exp(-theta*(u+v))
    #          / (  (1-exp(-theta)) - (1-exp(-theta*u))*(1-exp(-theta*v))  )^2
    e_t = np.exp(-theta)
    e_tu = np.exp(-theta * u)
    e_tv = np.exp(-theta * v)

    numer = theta * (1 - e_t) * np.exp(-theta * (u + v))
    denom = ((1 - e_t) - (1 - e_tu) * (1 - e_tv)) ** 2

    # Guard against non-positive values
    mask = (numer > 0) & (denom > 0)
    if not np.all(mask):
        return -1e12
    return float(np.sum(np.log(numer[mask]) - np.log(denom[mask])))


def fit_frank_copula(
    u: np.ndarray,
    v: np.ndarray,
) -> dict[str, Any]:
    """Fit a Frank copula (symmetric dependence) to bivariate data.

    Parameters:
        u: First marginal uniform observations.
        v: Second marginal uniform observations.

    Returns:
        Dict with keys:

        * ``"theta"`` -- estimated Frank parameter (nonzero).
        * ``"copula_type"`` -- ``"frank"``.
    """
    u, v = _ensure_uniform(u), _ensure_uniform(v)

    def neg_ll(params: np.ndarray) -> float:
        theta = params[0]
        if abs(theta) < 1e-6:
            return 1e12
        try:
            return -_frank_log_density(u, v, theta)
        except (FloatingPointError, ValueError):
            return 1e12

    tau, _ = sp_stats.kendalltau(u, v)
    # Rough initial guess
    theta_init = np.sign(tau) * max(abs(tau) * 10, 1.0)
    if abs(theta_init) < 0.5:
        theta_init = 1.0

    res = optimize.minimize(
        neg_ll,
        x0=[theta_init],
        method="Nelder-Mead",
        options={"maxiter": 5000},
    )
    theta = float(res.x[0])

    return {
        "theta": theta,
        "copula_type": "frank",
    }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def copula_simulate(
    copula_params: dict[str, Any],
    n_sims: int = 10000,
    copula_type: str | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate from a fitted copula.

    Parameters:
        copula_params: Output from one of the ``fit_*_copula`` functions.
        n_sims: Number of samples to draw.
        copula_type: Override copula type (defaults to
            ``copula_params["copula_type"]``).
        seed: Random seed for reproducibility.

    Returns:
        Array of shape ``(n_sims, d)`` with uniform marginals in
        ``(0, 1)``.

    Raises:
        ValueError: If the copula type is not recognized.
    """
    rng = np.random.default_rng(seed)
    ctype = copula_type or copula_params.get("copula_type", "")

    if ctype == "gaussian":
        corr = copula_params["correlation"]
        d = corr.shape[0]
        z = rng.multivariate_normal(np.zeros(d), corr, size=n_sims)
        return sp_stats.norm.cdf(z)

    if ctype == "student_t":
        corr = copula_params["correlation"]
        df = copula_params["df"]
        d = corr.shape[0]
        z = rng.multivariate_normal(np.zeros(d), corr, size=n_sims)
        chi2 = rng.chisquare(df, size=(n_sims, 1))
        t_vals = z / np.sqrt(chi2 / df)
        return sp_stats.t.cdf(t_vals, df=df)

    if ctype == "clayton":
        theta = copula_params["theta"]
        # Marshall-Olkin algorithm for Clayton
        v_gamma = rng.gamma(1.0 / theta, 1.0, size=n_sims)
        e1 = rng.exponential(1.0, size=n_sims)
        e2 = rng.exponential(1.0, size=n_sims)
        u1 = np.power(1 + e1 / v_gamma, -1.0 / theta)
        u2 = np.power(1 + e2 / v_gamma, -1.0 / theta)
        return np.column_stack([u1, u2])

    if ctype == "gumbel":
        theta = copula_params["theta"]
        # Stable distribution method for Gumbel
        # Use inverse CDF sampling via conditional
        u1 = rng.uniform(0, 1, size=n_sims)
        u2 = rng.uniform(0, 1, size=n_sims)
        # Use conditional CDF inversion approximation
        lu1 = -np.log(u1)
        lu2 = -np.log(u2)
        A = np.power(lu1, theta) + np.power(lu2, theta)
        C = np.exp(-np.power(A, 1.0 / theta))
        # Apply PIT
        return np.column_stack([u1, C / u1])

    if ctype == "frank":
        theta = copula_params["theta"]
        u1 = rng.uniform(0, 1, size=n_sims)
        # Conditional CDF inversion for Frank
        # C_2|1(u2|u1) = t => u2 = -1/theta * log(1 + t*(exp(-theta)-1) / (exp(-theta*u1) * (1 - t) + t))
        # where t is uniform
        t = rng.uniform(0, 1, size=n_sims)
        exp_neg_theta = np.exp(-theta)
        exp_neg_theta_u1 = np.exp(-theta * u1)
        numer = t * (exp_neg_theta - 1)
        denom = exp_neg_theta_u1 * (1 - t) + t
        u2 = -np.log(1 + numer / denom) / theta
        u2 = np.clip(u2, 1e-10, 1 - 1e-10)
        return np.column_stack([u1, u2])

    msg = f"Unknown copula type: {ctype!r}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Tail dependence
# ---------------------------------------------------------------------------


def tail_dependence(
    u: np.ndarray,
    v: np.ndarray,
    method: str = "empirical",
    threshold: float = 0.05,
) -> dict[str, float]:
    """Estimate lower and upper tail dependence coefficients.

    Parameters:
        u: First marginal uniform observations.
        v: Second marginal uniform observations.
        method: ``"empirical"`` (default) for non-parametric estimation.
        threshold: Quantile threshold for tail estimation (default 0.05).

    Returns:
        Dict with ``"lower"`` and ``"upper"`` tail dependence estimates.

    Raises:
        ValueError: If *method* is not recognized.
    """
    u, v = _ensure_uniform(u), _ensure_uniform(v)

    if method == "empirical":
        # Lower tail: P(V <= q | U <= q)
        lower_mask = u <= threshold
        if np.sum(lower_mask) > 0:
            lower_td = float(np.mean(v[lower_mask] <= threshold))
        else:
            lower_td = 0.0

        # Upper tail: P(V > 1-q | U > 1-q)
        upper_q = 1 - threshold
        upper_mask = u >= upper_q
        if np.sum(upper_mask) > 0:
            upper_td = float(np.mean(v[upper_mask] >= upper_q))
        else:
            upper_td = 0.0

        return {"lower": lower_td, "upper": upper_td}

    msg = f"Unknown tail dependence method: {method!r}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Rank correlation
# ---------------------------------------------------------------------------


def rank_correlation(
    returns: np.ndarray,
    method: str = "both",
) -> dict[str, Any]:
    """Compute Kendall's tau and/or Spearman's rho rank correlation.

    Parameters:
        returns: Array of shape ``(n_obs, n_assets)``.
        method: ``"kendall"``, ``"spearman"``, or ``"both"`` (default).

    Returns:
        Dict with keys (depending on *method*):

        * ``"kendall_tau"`` -- Kendall's tau correlation matrix.
        * ``"spearman_rho"`` -- Spearman's rho correlation matrix.

    Raises:
        ValueError: If *method* is not recognized.
    """
    n_assets = returns.shape[1]

    result: dict[str, Any] = {}

    if method in ("kendall", "both"):
        tau_mat = np.eye(n_assets)
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                tau, _ = sp_stats.kendalltau(returns[:, i], returns[:, j])
                tau_mat[i, j] = tau
                tau_mat[j, i] = tau
        result["kendall_tau"] = tau_mat

    if method in ("spearman", "both"):
        rho_mat = np.eye(n_assets)
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                rho, _ = sp_stats.spearmanr(returns[:, i], returns[:, j])
                rho_mat[i, j] = rho
                rho_mat[j, i] = rho
        result["spearman_rho"] = rho_mat

    if not result:
        msg = f"Unknown rank correlation method: {method!r}"
        raise ValueError(msg)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_uniform(x: np.ndarray) -> np.ndarray:
    """If values appear to be raw data (outside [0,1]), convert to pseudo-obs."""
    if np.min(x) < 0 or np.max(x) > 1:
        return _empirical_cdf(x)
    return x
