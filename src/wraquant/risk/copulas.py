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

    The Gaussian copula models dependence via a multivariate normal
    distribution applied to the rank-transformed data.  It captures
    linear dependence but has zero tail dependence: in the limit,
    extreme co-movements become independent.

    Interpretation:
        - The correlation matrix describes the "normal-like" dependence
          structure of the data.
        - The key limitation: the Gaussian copula implies that joint
          extreme events (crashes, rallies) are asymptotically
          independent.  This is dangerous for risk management -- the
          2008 crisis demonstrated that assets crash together more
          than the Gaussian copula predicts.
        - Use as a baseline for comparison.  If tail dependence is
          non-negligible (check with ``tail_dependence``), switch to
          the Student-t copula.

    When to use:
        - As a baseline for dependence modelling.
        - When you have evidence that tail dependence is genuinely
          absent.
        - For quick-and-dirty simulation of correlated returns.

    Parameters:
        returns: Array of shape ``(n_obs, n_assets)`` with raw returns.

    Returns:
        Dict with keys:

        * ``"correlation"`` -- estimated copula correlation matrix.
        * ``"copula_type"`` -- ``"gaussian"``.

    See Also:
        fit_t_copula: Student-t copula with symmetric tail dependence.
        tail_dependence: Check whether tail dependence is present.
    """
    from wraquant.core._coerce import coerce_array

    if returns.ndim == 1:
        returns = coerce_array(returns, name="returns").reshape(-1, 1)
    else:
        returns = np.asarray(returns, dtype=np.float64)
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

    The Student-t copula is the standard choice for equity portfolios
    because it captures symmetric tail dependence: the tendency of
    assets to experience extreme co-movements (both crashes and
    rallies) more often than a Gaussian model would predict.

    Interpretation:
        - **df** (degrees of freedom) controls tail heaviness:
          - df = 3-5: very heavy tails, strong tail dependence.
            Appropriate for equity portfolios during turbulent markets.
          - df = 10-20: moderate tails. Appropriate for investment-grade
            fixed income.
          - df -> infinity: converges to Gaussian copula (no tail
            dependence).
        - The tail dependence coefficient is approximately
          2 * t_{df+1}(-sqrt((df+1)(1-rho)/(1+rho))), where rho is
          the copula correlation.
        - Higher df means the copula becomes more "Gaussian-like" in
          the tails.

    When to use:
        - Equity risk modelling where joint crashes are a concern.
        - VaR/CVaR estimation for multi-asset portfolios.
        - As the default copula for portfolio risk (unless there is
          evidence of asymmetric tail dependence).

    Parameters:
        returns: Array of shape ``(n_obs, n_assets)``.
        df: Degrees of freedom. Lower = heavier tails. Typical values
            3-10 for equities. If unsure, start with 5.

    Returns:
        Dict with keys:

        * ``"correlation"`` -- shape correlation matrix.
        * ``"df"`` -- degrees of freedom.
        * ``"copula_type"`` -- ``"student_t"``.

    See Also:
        fit_gaussian_copula: No tail dependence (df -> infinity).
        fit_clayton_copula: Lower tail dependence only.
    """
    from wraquant.core._coerce import coerce_array

    if returns.ndim == 1:
        returns = coerce_array(returns, name="returns").reshape(-1, 1)
    else:
        returns = np.asarray(returns, dtype=np.float64)
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

    The Clayton copula has lower tail dependence but zero upper tail
    dependence.  This means assets modelled with a Clayton copula
    tend to crash together but rally independently -- a realistic
    pattern for equities and credit, where "risk-on/risk-off" dynamics
    create asymmetric co-movement.

    Interpretation:
        - **theta** > 0 controls dependence strength.  Higher theta
          = stronger lower tail dependence.  theta -> 0 = independence.
        - **lower_tail_dependence** = 2^{-1/theta} is the probability
          that both assets are in their lower tail simultaneously,
          given that one is.  Values above 0.3 indicate meaningful
          crash co-movement.
        - The Clayton copula is the canonical choice for modelling
          "contagion" and "flight to quality" dynamics.

    When to use:
        - Two equity assets or sectors where you expect crash
          contagion but independent rallies.
        - Credit portfolios where defaults cluster.
        - When ``tail_dependence`` shows lower > upper.

    Parameters:
        u: First marginal uniform observations.  Pass raw data and
            the empirical CDF is applied automatically.
        v: Second marginal uniform observations.

    Returns:
        Dict with keys:

        * ``"theta"`` -- Clayton parameter (> 0).
        * ``"copula_type"`` -- ``"clayton"``.
        * ``"lower_tail_dependence"`` -- 2^{-1/theta}. > 0.3 is
          meaningful.

    See Also:
        fit_gumbel_copula: Upper tail dependence (opposite asymmetry).
        fit_t_copula: Symmetric tail dependence.
    """
    from wraquant.core._coerce import coerce_array

    u = coerce_array(u, name="u")
    v = coerce_array(v, name="v")
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

    The Gumbel copula has upper tail dependence but zero lower tail
    dependence.  Assets rally together in extreme moves but crash
    independently.  This is less common than the Clayton pattern in
    equities but can apply to certain commodity pairs (e.g., two
    correlated energy commodities that spike together during supply
    shocks).

    Interpretation:
        - **theta** >= 1 controls dependence strength. theta = 1 is
          independence; larger theta = stronger upper tail dependence.
        - **upper_tail_dependence** = 2 - 2^{1/theta} measures the
          probability of joint extreme upper co-movement.

    When to use:
        - Commodity pairs with supply-shock co-movement.
        - When ``tail_dependence`` shows upper > lower.
        - When modelling "melt-up" contagion.

    Parameters:
        u: First marginal uniform observations.
        v: Second marginal uniform observations.

    Returns:
        Dict with keys:

        * ``"theta"`` -- Gumbel parameter (>= 1).
        * ``"copula_type"`` -- ``"gumbel"``.
        * ``"upper_tail_dependence"`` -- 2 - 2^{1/theta}.

    See Also:
        fit_clayton_copula: Lower tail dependence (more common in equities).
        fit_t_copula: Symmetric tail dependence.
    """
    from wraquant.core._coerce import coerce_array

    u = coerce_array(u, name="u")
    v = coerce_array(v, name="v")
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
    """Fit a Frank copula (symmetric dependence, no tail dependence).

    The Frank copula is symmetric with zero tail dependence in both
    directions.  It is useful as a benchmark: if neither the Clayton
    nor Gumbel copula fits significantly better than Frank, then
    there is no evidence of asymmetric tail dependence.

    Interpretation:
        - **theta** > 0: positive dependence; theta < 0: negative
          dependence; |theta| large = strong dependence.
        - The Frank copula allows negative dependence (unlike Clayton
          and Gumbel), making it useful for hedging pairs.
        - No tail dependence: extreme co-movements are modelled as
          asymptotically independent.

    When to use:
        - As a benchmark against Clayton/Gumbel.
        - When ``tail_dependence`` shows negligible tail dependence.
        - For pairs with negative dependence (hedging relationships).

    Parameters:
        u: First marginal uniform observations.
        v: Second marginal uniform observations.

    Returns:
        Dict with keys:

        * ``"theta"`` -- Frank parameter. Positive = positive
          dependence, negative = negative dependence.
        * ``"copula_type"`` -- ``"frank"``.

    See Also:
        fit_gaussian_copula: Multivariate with no tail dependence.
        fit_clayton_copula: Lower tail dependence.
        fit_gumbel_copula: Upper tail dependence.
    """
    from wraquant.core._coerce import coerce_array

    u = coerce_array(u, name="u")
    v = coerce_array(v, name="v")
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

    Generates samples from the fitted dependence structure with uniform
    marginals.  To get realistic return scenarios, transform each
    column through the inverse CDF of the desired marginal distribution
    (e.g., inverse normal, inverse t, or empirical quantile function).

    Interpretation:
        - Output columns are uniform on (0, 1) -- they represent the
          dependence structure only, not the marginal distributions.
        - Correlated low values in all columns = a joint crash scenario.
        - To convert to returns: ``returns[:, j] = norm.ppf(U[:, j],
          loc=mu_j, scale=sigma_j)`` or use empirical quantile
          functions for non-parametric marginals.
        - Use for Monte Carlo VaR/CVaR, portfolio simulation, or
          stress testing.

    Parameters:
        copula_params: Output from one of the ``fit_*_copula`` functions.
        n_sims: Number of samples to draw.
        copula_type: Override copula type (defaults to
            ``copula_params["copula_type"]``).
        seed: Random seed for reproducibility.

    Returns:
        Array of shape ``(n_sims, d)`` with uniform marginals in (0, 1).

    Raises:
        ValueError: If the copula type is not recognized.

    Example:
        >>> from scipy.stats import norm
        >>> cop = fit_gaussian_copula(returns)
        >>> U = copula_simulate(cop, n_sims=10000, seed=42)
        >>> # Transform to normal marginals:
        >>> sim_returns = norm.ppf(U, loc=mu, scale=sigma)
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

    Tail dependence measures the probability that one variable is in
    its extreme tail given that the other is.  This is the key quantity
    that copula selection hinges on.

    Interpretation:
        - **lower** ~ P(V <= q | U <= q): the probability of a joint
          crash.  Values above 0.2-0.3 are economically significant
          and indicate that the Gaussian copula is inappropriate.
        - **upper** ~ P(V > 1-q | U > 1-q): the probability of a
          joint rally.
        - If lower >> upper: use Clayton copula (crash contagion).
        - If upper >> lower: use Gumbel copula (rally contagion).
        - If both are similar: use Student-t copula (symmetric tails).
        - If both are near zero: Gaussian or Frank copula is adequate.

    Caveat:
        - Empirical tail dependence estimates are noisy with small
          samples.  Use threshold = 0.10 for more observations per
          tail (less extreme, more precise) or 0.05 for fewer
          observations (more extreme, noisier).

    Parameters:
        u: First marginal uniform observations.
        v: Second marginal uniform observations.
        method: ``"empirical"`` (default) for non-parametric estimation.
        threshold: Quantile threshold for tail estimation.
            0.05 = 5th/95th percentile (default). 0.10 = more data
            in the tail estimate but less extreme.

    Returns:
        Dict with ``"lower"`` and ``"upper"`` tail dependence estimates.

    Example:
        >>> import numpy as np
        >>> from scipy.stats import norm
        >>> rng = np.random.default_rng(0)
        >>> # Simulate from a t-copula (symmetric tail dependence)
        >>> z = rng.multivariate_normal([0,0], [[1,0.5],[0.5,1]], 5000)
        >>> u = norm.cdf(z[:, 0])
        >>> v = norm.cdf(z[:, 1])
        >>> td = tail_dependence(u, v)
        >>> print(f"Lower: {td['lower']:.2f}, Upper: {td['upper']:.2f}")
    """
    from wraquant.core._coerce import coerce_array

    u = coerce_array(u, name="u")
    v = coerce_array(v, name="v")
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

    Rank correlations measure monotonic association without assuming
    linearity.  Unlike Pearson correlation, they are invariant to
    monotonic transformations of the marginals and directly related
    to copula parameters, making them the correct correlation measure
    for copula modelling.

    Interpretation:
        - **Kendall's tau**: Probability of concordance minus
          probability of discordance.  tau = 0.5 means ~75% of pairs
          move in the same direction.
        - **Spearman's rho**: Pearson correlation of the ranks.
          Generally |rho| >= |tau| for the same data.
        - Both are robust to outliers (unlike Pearson).
        - Copula relationships: for the Gaussian copula,
          rho_pearson = 2*sin(pi*tau/6). For Clayton,
          theta = 2*tau/(1-tau).

    When to use:
        - Always prefer rank correlation over Pearson for copula
          modelling.
        - Use Kendall's tau for small samples (more robust).
        - Use Spearman's rho for comparison with Pearson.

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
    from wraquant.core._coerce import coerce_array

    if returns.ndim == 1:
        returns = coerce_array(returns, name="returns").reshape(-1, 1)
    else:
        returns = np.asarray(returns, dtype=np.float64)
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
