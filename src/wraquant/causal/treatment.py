"""Treatment effect estimation via pure numpy/scipy implementations.

Causal inference methods estimate the causal effect of a treatment or
intervention on an outcome, controlling for confounding variables. In
finance, these methods answer questions like:

- "Did the Fed rate cut cause the equity rally, or would it have
  happened anyway?"
- "What is the causal effect of share buyback announcements on stock
  returns?"
- "Did the new regulatory rule reduce trading costs, controlling for
  market conditions?"

This module provides six complementary causal inference estimators:

1. **Inverse Probability Weighting** (``ipw_ate``) -- reweights
   observations by the inverse of their propensity score to create a
   pseudo-population where treatment is independent of covariates.
   Simple but sensitive to extreme propensity scores.

2. **Nearest-neighbor matching** (``matching_ate``) -- matches each
   treated unit to the closest control unit(s) in covariate space.
   Intuitive and nonparametric, but can be biased in high dimensions.

3. **Doubly robust estimation** (``doubly_robust_ate``) -- combines IPW
   with outcome regression. Consistent if *either* the propensity score
   model *or* the outcome model is correctly specified. The recommended
   default when you are unsure about model specification.

4. **Regression discontinuity** (``regression_discontinuity``) -- exploits
   a sharp cutoff in a running variable to identify a local treatment
   effect. Example: index inclusion at a market-cap threshold.

5. **Synthetic control** (``synthetic_control``) -- constructs a
   weighted combination of control units that mimics the treated unit's
   pre-treatment trajectory. The treatment effect is the post-treatment
   gap. Ideal for single-unit studies (e.g., one country, one firm).

6. **Difference-in-differences** (``diff_in_diff``) -- compares the
   before-after change in the treatment group to the before-after change
   in the control group. Requires the parallel trends assumption.

Utilities:
    - ``propensity_score``: logistic regression for treatment assignment
      probabilities.

How to choose:
    - **Random treatment, confounders observed**: IPW or doubly robust.
    - **Sharp cutoff determines treatment**: regression discontinuity.
    - **One treated unit, many controls over time**: synthetic control.
    - **Two-period panel, treatment at a known time**: diff-in-diff.
    - **Uncertain model specification**: doubly robust (safest).

References:
    - Rubin (1974), "Estimating Causal Effects of Treatments"
    - Imbens & Rubin (2015), "Causal Inference for Statistics, Social,
      and Biomedical Sciences"
    - Abadie, Diamond & Hainmueller (2010), "Synthetic Control Methods
      for Comparative Case Studies"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, spatial, stats


def _ols_coefficients(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Get OLS coefficients using canonical wraquant implementation.

    Assumes X already contains an intercept column if needed.
    Uses the shared stats.regression.ols implementation as the
    single source of truth for OLS computation.
    """
    from wraquant.stats.regression import ols

    result = ols(y, X, add_constant=False)
    return result["coefficients"]


__all__ = [
    "propensity_score",
    "ipw_ate",
    "matching_ate",
    "doubly_robust_ate",
    "regression_discontinuity",
    "synthetic_control",
    "diff_in_diff",
    "granger_causality",
    "instrumental_variable",
    "event_study",
    "synthetic_control_weights",
    "causal_forest",
    "mediation_analysis",
    "regression_discontinuity_robust",
    "bounds_analysis",
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ATEResult:
    """Result container for average treatment effect estimators.

    Parameters
    ----------
    ate : float
        Average treatment effect estimate.
    se : float
        Standard error of the ATE estimate.
    ci_lower : float
        Lower bound of the 95% confidence interval.
    ci_upper : float
        Upper bound of the 95% confidence interval.
    n_treated : int
        Number of treated units.
    n_control : int
        Number of control units.
    details : dict
        Additional estimator-specific diagnostics.
    """

    ate: float
    se: float
    ci_lower: float
    ci_upper: float
    n_treated: int
    n_control: int
    details: dict = field(default_factory=dict)


@dataclass
class RDResult:
    """Result container for regression discontinuity design.

    Parameters
    ----------
    ate : float
        Estimated treatment effect at the cutoff.
    se : float
        Standard error.
    ci_lower : float
        Lower bound of the 95% confidence interval.
    ci_upper : float
        Upper bound of the 95% confidence interval.
    n_left : int
        Number of observations to the left of the cutoff.
    n_right : int
        Number of observations to the right of the cutoff.
    bandwidth : float
        Bandwidth used for the estimation.
    details : dict
        Additional diagnostics.
    """

    ate: float
    se: float
    ci_lower: float
    ci_upper: float
    n_left: int
    n_right: int
    bandwidth: float
    details: dict = field(default_factory=dict)


@dataclass
class SyntheticControlResult:
    """Result container for synthetic control method.

    Parameters
    ----------
    ate : float
        Estimated treatment effect (post-period average gap).
    weights : np.ndarray
        Donor weights for the synthetic control unit.
    treated_outcomes : np.ndarray
        Observed treated outcomes.
    synthetic_outcomes : np.ndarray
        Synthetic control outcomes.
    pre_rmse : float
        Pre-period root mean squared error.
    gaps : np.ndarray
        Period-by-period gaps (treated - synthetic).
    """

    ate: float
    weights: np.ndarray
    treated_outcomes: np.ndarray
    synthetic_outcomes: np.ndarray
    pre_rmse: float
    gaps: np.ndarray


@dataclass
class DIDResult:
    """Result container for difference-in-differences.

    Parameters
    ----------
    ate : float
        Difference-in-differences estimate.
    se : float
        Standard error.
    ci_lower : float
        Lower bound of the 95% confidence interval.
    ci_upper : float
        Upper bound of the 95% confidence interval.
    pre_treatment_mean : float
        Mean outcome for the treatment group pre-treatment.
    post_treatment_mean : float
        Mean outcome for the treatment group post-treatment.
    pre_control_mean : float
        Mean outcome for the control group pre-treatment.
    post_control_mean : float
        Mean outcome for the control group post-treatment.
    details : dict
        Additional diagnostics.
    """

    ate: float
    se: float
    ci_lower: float
    ci_upper: float
    pre_treatment_mean: float
    post_treatment_mean: float
    pre_control_mean: float
    post_control_mean: float
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Propensity score estimation
# ---------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )


def _log_likelihood_grad(
    beta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Negative log-likelihood and gradient for logistic regression."""
    z = X @ beta
    p = _sigmoid(z)
    # Clip to avoid log(0)
    p_clipped = np.clip(p, 1e-15, 1.0 - 1e-15)
    nll = -np.sum(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))
    grad = X.T @ (p - y)
    return float(nll), grad


def propensity_score(
    treatment: np.ndarray,
    covariates: np.ndarray,
) -> np.ndarray:
    """Estimate propensity scores using logistic regression.

    The propensity score e(X) = P(treatment=1 | covariates=X) is the
    probability that a unit receives treatment given its observed
    characteristics.  It is the foundation of all observational causal
    inference: by conditioning on (or weighting by) the propensity
    score, you can remove confounding bias, making the treated and
    control groups comparable.

    This function fits a logistic regression of treatment assignment on
    covariates.  Scores are clipped to [0.01, 0.99] to prevent extreme
    inverse-probability weights that would blow up downstream estimators.

    Interpretation:
        - A propensity score of 0.5 means the unit was equally likely
          to be treated or not -- ideal for causal inference.
        - Scores near 0 or 1 indicate units that were almost
          deterministically treated/untreated.  These are problematic:
          the overlap assumption fails, and IPW weights become extreme.
        - Check the overlap: plot histograms of propensity scores
          separately for treated and control groups.  If they don't
          overlap, causal inference is unreliable in the non-overlap
          region.

    When to use:
        - As input to ``ipw_ate`` or ``doubly_robust_ate``.
        - For propensity score matching (match treated to controls
          with similar scores).
        - To diagnose selection bias: if propensity scores are
          very different between groups, selection is strong.

    Red flags:
        - Most scores near 0 or 1: poor overlap, IPW will be unstable.
        - Treatment is nearly perfectly predicted: positivity violation.
        - Scores are all similar (~0.5): treatment is nearly random,
          which is actually ideal for causal inference.

    Parameters:
        treatment: Binary treatment indicator (1-D array of 0s and 1s).
        covariates: Covariate matrix of shape ``(n_samples, n_features)``.
            An intercept column is added automatically; do not include one.

    Returns:
        Estimated propensity scores (probabilities) of shape
        ``(n_samples,)``, each in [0.01, 0.99].

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X = rng.normal(size=(200, 3))
        >>> T = (X[:, 0] + rng.normal(size=200) > 0).astype(float)
        >>> ps = propensity_score(T, X)
        >>> # Check overlap: both groups should have scores in [0.2, 0.8]
        >>> print(f"Treated scores: [{ps[T==1].min():.2f}, {ps[T==1].max():.2f}]")
        >>> print(f"Control scores: [{ps[T==0].min():.2f}, {ps[T==0].max():.2f}]")

    See Also:
        ipw_ate: Use propensity scores to estimate the ATE.
        doubly_robust_ate: Combines propensity scores with outcome
            regression for robustness.
    """
    treatment = np.asarray(treatment, dtype=float).ravel()
    covariates = np.asarray(covariates, dtype=float)
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    n = len(treatment)
    # Add intercept
    X = np.column_stack([np.ones(n), covariates])
    k = X.shape[1]

    beta0 = np.zeros(k)
    result = optimize.minimize(
        _log_likelihood_grad,
        beta0,
        args=(X, treatment),
        method="L-BFGS-B",
        jac=True,
    )
    scores = _sigmoid(X @ result.x)
    # Clip to avoid extreme weights
    return np.clip(scores, 0.01, 0.99)


# ---------------------------------------------------------------------------
# Inverse probability weighting
# ---------------------------------------------------------------------------


def ipw_ate(
    outcome: np.ndarray,
    treatment: np.ndarray,
    propensity_scores: np.ndarray,
) -> ATEResult:
    r"""Estimate the Average Treatment Effect using Inverse Probability Weighting.

    The ATE (Average Treatment Effect) answers: "On average, how much
    does the treatment change the outcome?"  IPW estimates this by
    reweighting observations by the inverse of their propensity score,
    creating a pseudo-population where treatment assignment is
    independent of confounders.

    The Horvitz-Thompson estimator is:

    .. math::

        \widehat{\text{ATE}} = \frac{1}{n}\sum_{i=1}^{n}
        \frac{T_i\,Y_i}{e(X_i)}
        - \frac{1}{n}\sum_{i=1}^{n}
        \frac{(1 - T_i)\,Y_i}{1 - e(X_i)}

    Interpretation:
        - **ate > 0**: Treatment increases the outcome on average.
        - **ate < 0**: Treatment decreases the outcome on average.
        - The 95% CI is asymptotically valid under correct propensity
          score specification and the overlap assumption.
        - If CI includes 0: cannot conclude treatment has an effect.

    When to use:
        - You have a well-estimated propensity score model.
        - There is good overlap (propensity scores are not extreme).
        - You are willing to assume no unobserved confounders.

    When NOT to use:
        - Propensity scores near 0 or 1: weights explode, variance
          is huge.  Use ``doubly_robust_ate`` instead.
        - Strong model misspecification concerns: use
          ``doubly_robust_ate`` (consistent if either model is right).
        - Very few treated or control units: matching may be better.

    Parameters:
        outcome: Observed outcomes (1-D array).
        treatment: Binary treatment indicator (0s and 1s).
        propensity_scores: Estimated propensity scores from
            :func:`propensity_score` (1-D array).

    Returns:
        ATEResult with ATE estimate, standard error, 95% confidence
        interval, and sample sizes.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> n = 500
        >>> X = rng.normal(size=(n, 2))
        >>> T = (X[:, 0] > 0).astype(float)
        >>> Y = 2.0 * T + X[:, 0] + rng.normal(size=n)
        >>> ps = propensity_score(T, X)
        >>> result = ipw_ate(Y, T, ps)
        >>> print(f"ATE: {result.ate:.2f} [{result.ci_lower:.2f}, {result.ci_upper:.2f}]")

    See Also:
        doubly_robust_ate: More robust alternative (recommended default).
        matching_ate: Nonparametric matching estimator.
    """
    outcome = np.asarray(outcome, dtype=float).ravel()
    treatment = np.asarray(treatment, dtype=float).ravel()
    ps = np.asarray(propensity_scores, dtype=float).ravel()

    n = len(outcome)
    treated_mask = treatment == 1
    control_mask = treatment == 0
    n_treated = int(treated_mask.sum())
    n_control = int(control_mask.sum())

    # Horvitz-Thompson estimator
    y1_weighted = outcome * treatment / ps
    y0_weighted = outcome * (1 - treatment) / (1 - ps)

    ate = float(np.mean(y1_weighted) - np.mean(y0_weighted))

    # Influence function based standard error
    influence = y1_weighted - y0_weighted - ate
    se = float(np.std(influence, ddof=1) / np.sqrt(n))

    z = stats.norm.ppf(0.975)
    return ATEResult(
        ate=ate,
        se=se,
        ci_lower=ate - z * se,
        ci_upper=ate + z * se,
        n_treated=n_treated,
        n_control=n_control,
        details={"estimator": "ipw"},
    )


# ---------------------------------------------------------------------------
# Nearest-neighbor matching
# ---------------------------------------------------------------------------


def matching_ate(
    outcome: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    n_neighbors: int = 1,
) -> ATEResult:
    """Estimate the ATE using nearest-neighbor matching on covariates.

    For each treated unit, finds the closest control unit(s) in
    covariate space and uses the matched pair difference as the
    individual treatment effect estimate (and vice versa for controls).
    The ATE is the average of these pairwise differences.

    Matching is the most intuitive causal inference method: compare
    each treated unit to the most similar untreated unit.  It is
    nonparametric (no model assumptions) but suffers from the curse
    of dimensionality -- with many covariates, "nearest" neighbors
    can be far away.

    Interpretation:
        - **ate** is the average difference in outcomes between
          treated units and their matched controls, averaged over
          all units.
        - Good matching means the matched pairs are genuinely similar.
          Check the covariate balance after matching.
        - **n_neighbors = 1**: lowest bias, highest variance (each
          unit matched to exactly one partner).
        - **n_neighbors = 3-5**: reduces variance at the cost of
          some bias (averaging over less-similar matches).

    When to use:
        - Low-dimensional covariates (2-5 features).
        - When you want a transparent, model-free estimator.
        - When propensity score modelling is difficult.

    Limitations:
        - Curse of dimensionality: in high dimensions, matches become
          poor.  Use propensity score matching instead.
        - Biased with finite samples (Abadie & Imbens, 2006).
        - No extrapolation: cannot estimate effects outside the
          overlap region.

    Parameters:
        outcome: Observed outcomes (1-D array).
        treatment: Binary treatment indicator (0s and 1s).
        covariates: Covariate matrix of shape ``(n_samples, n_features)``.
        n_neighbors: Number of nearest neighbors to match with
            (default 1). More neighbors reduce variance but may
            increase bias.

    Returns:
        ATEResult with ATE estimate, standard error, and 95% CI.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X = rng.normal(size=(300, 2))
        >>> T = (X[:, 0] > 0).astype(float)
        >>> Y = 1.5 * T + X[:, 0] + rng.normal(size=300)
        >>> result = matching_ate(Y, T, X)
        >>> print(f"ATE: {result.ate:.2f} (se={result.se:.2f})")

    See Also:
        ipw_ate: Weighting-based estimator (no matching).
        doubly_robust_ate: Recommended default when uncertain about
            model specification.
    """
    outcome = np.asarray(outcome, dtype=float).ravel()
    treatment = np.asarray(treatment, dtype=float).ravel()
    covariates = np.asarray(covariates, dtype=float)
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    treated_mask = treatment == 1
    control_mask = treatment == 0
    n_treated = int(treated_mask.sum())
    n_control = int(control_mask.sum())

    X_treated = covariates[treated_mask]
    X_control = covariates[control_mask]
    y_treated = outcome[treated_mask]
    y_control = outcome[control_mask]

    # Build KD-trees for fast nearest neighbor lookup
    tree_control = spatial.KDTree(X_control)
    tree_treated = spatial.KDTree(X_treated)

    # For each treated unit, find matched control outcomes
    _, idx_control = tree_control.query(X_treated, k=n_neighbors)
    if n_neighbors == 1:
        idx_control = idx_control.reshape(-1, 1)
    matched_control = np.mean(y_control[idx_control], axis=1)

    # For each control unit, find matched treated outcomes
    _, idx_treated = tree_treated.query(X_control, k=n_neighbors)
    if n_neighbors == 1:
        idx_treated = idx_treated.reshape(-1, 1)
    matched_treated = np.mean(y_treated[idx_treated], axis=1)

    # ATE: average of (Y_treated - matched_control) and (matched_treated - Y_control)
    tau_treated = y_treated - matched_control
    tau_control = matched_treated - y_control
    ate = float(0.5 * (np.mean(tau_treated) + np.mean(tau_control)))

    # Abadie-Imbens variance estimator (simplified)
    all_tau = np.concatenate([tau_treated, tau_control])
    se = float(np.std(all_tau, ddof=1) / np.sqrt(len(all_tau)))

    z = stats.norm.ppf(0.975)
    return ATEResult(
        ate=ate,
        se=se,
        ci_lower=ate - z * se,
        ci_upper=ate + z * se,
        n_treated=n_treated,
        n_control=n_control,
        details={"estimator": "matching", "n_neighbors": n_neighbors},
    )


# ---------------------------------------------------------------------------
# Doubly robust estimator
# ---------------------------------------------------------------------------


def doubly_robust_ate(
    outcome: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
) -> ATEResult:
    r"""Estimate the ATE using the doubly robust (augmented IPW) estimator.

    The doubly robust estimator combines inverse probability weighting
    with outcome regression.  It is **consistent if either** the
    propensity score model **or** the outcome regression model is
    correctly specified, providing a valuable insurance against model
    misspecification.

    This is the **recommended default** when you are unsure about
    model specification.

    Internally, the function:

    1. Estimates propensity scores via logistic regression.
    2. Fits separate OLS outcome models for treated and control groups.
    3. Combines both in the augmented IPW formula.

    Parameters:
        outcome (np.ndarray): Observed outcomes (1-D array).
        treatment (np.ndarray): Binary treatment indicator (0s and 1s).
        covariates (np.ndarray): Covariate matrix of shape
            ``(n_samples, n_features)``.

    Returns:
        ATEResult: ATE estimate with standard error and 95% confidence
            interval.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X = rng.normal(size=(500, 3))
        >>> T = (X[:, 0] > 0).astype(float)
        >>> Y = 2.0 * T + X[:, 0] + rng.normal(size=500)
        >>> result = doubly_robust_ate(Y, T, X)
        >>> 0.5 < result.ate < 3.5
        True

    See Also:
        ipw_ate: Pure IPW estimator.
        matching_ate: Nonparametric matching estimator.
        propensity_score: Logistic regression propensity scores.

    References:
        Robins, J.M., Rotnitzky, A. & Zhao, L.P. (1994). *Estimation
        of Regression Coefficients When Some Regressors Are Not Always
        Observed.* JASA 89(427).
    """
    outcome = np.asarray(outcome, dtype=float).ravel()
    treatment = np.asarray(treatment, dtype=float).ravel()
    covariates = np.asarray(covariates, dtype=float)
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    n = len(outcome)
    treated_mask = treatment == 1
    control_mask = treatment == 0
    n_treated = int(treated_mask.sum())
    n_control = int(control_mask.sum())

    # Step 1: Estimate propensity scores
    ps = propensity_score(treatment, covariates)

    # Step 2: Outcome regression (OLS with intercept for each group)
    X = np.column_stack([np.ones(n), covariates])

    # Fit outcome model for treated group
    X_t = X[treated_mask]
    y_t = outcome[treated_mask]
    beta_t = _ols_coefficients(X_t, y_t)
    mu1_hat = X @ beta_t  # predicted E[Y(1)|X] for all units

    # Fit outcome model for control group
    X_c = X[control_mask]
    y_c = outcome[control_mask]
    beta_c = _ols_coefficients(X_c, y_c)
    mu0_hat = X @ beta_c  # predicted E[Y(0)|X] for all units

    # Step 3: Doubly robust estimator
    dr1 = mu1_hat + treatment * (outcome - mu1_hat) / ps
    dr0 = mu0_hat + (1 - treatment) * (outcome - mu0_hat) / (1 - ps)

    ate = float(np.mean(dr1 - dr0))

    # Influence function based SE
    influence = dr1 - dr0 - ate
    se = float(np.std(influence, ddof=1) / np.sqrt(n))

    z = stats.norm.ppf(0.975)
    return ATEResult(
        ate=ate,
        se=se,
        ci_lower=ate - z * se,
        ci_upper=ate + z * se,
        n_treated=n_treated,
        n_control=n_control,
        details={"estimator": "doubly_robust"},
    )


# ---------------------------------------------------------------------------
# Regression discontinuity
# ---------------------------------------------------------------------------


def regression_discontinuity(
    outcome: np.ndarray,
    running_var: np.ndarray,
    cutoff: float = 0.0,
    bandwidth: float | None = None,
) -> RDResult:
    """Estimate treatment effect using a sharp regression discontinuity design.

    RDD exploits a sharp cutoff in a running variable to identify a
    local causal effect.  The key insight: units just above and just
    below the cutoff are essentially identical except for treatment
    status, so any discontinuity in outcomes at the cutoff is
    attributable to the treatment.

    Financial examples:
        - Market-cap threshold for S&P 500 inclusion: does index
          inclusion cause a price premium?
        - Credit score cutoff for loan approval: does access to
          credit affect firm investment?
        - Regulatory thresholds: do firms just above a reporting
          threshold behave differently?

    Interpretation:
        - **ate** is the LOCAL average treatment effect at the cutoff.
          It tells you the effect for units right at the boundary,
          not the average effect for all units.
        - The validity depends on the assumption that units cannot
          precisely manipulate their position relative to the cutoff.
          If they can (e.g., firms just barely meet a threshold by
          manipulating data), the estimate is biased.  Use the
          McCrary test (in ``regression_discontinuity_robust``) to
          check for manipulation.
        - **bandwidth** matters: too wide includes units far from the
          cutoff (more bias); too narrow uses few observations (more
          variance).

    Red flags:
        - Very different n_left and n_right: asymmetric data.
        - Bandwidth is very small: few observations, imprecise.
        - Density of running variable has a spike at the cutoff:
          possible manipulation (run McCrary test).

    Parameters:
        outcome: Observed outcomes (1-D array).
        running_var: Running variable that determines treatment
            (1-D array). Units with ``running_var >= cutoff`` are
            treated.
        cutoff: Cutoff value (default 0.0).
        bandwidth: Bandwidth for local linear regression. If None,
            uses Silverman's rule. Narrower = less bias, more variance.

    Returns:
        RDResult with estimated local ATE at the cutoff, standard
        error, confidence interval, and bandwidth used.

    Raises:
        ValueError: If fewer than 2 observations on either side within
            bandwidth.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> n = 500
        >>> X = rng.uniform(-1, 1, n)
        >>> Y = 0.5 * (X >= 0).astype(float) + X + rng.normal(0, 0.2, n)
        >>> result = regression_discontinuity(Y, X, cutoff=0.0)
        >>> print(f"RD estimate: {result.ate:.3f} (se={result.se:.3f})")

    See Also:
        regression_discontinuity_robust: With IK bandwidth and McCrary test.
        diff_in_diff: Before/after comparison with control group.
    """
    outcome = np.asarray(outcome, dtype=float).ravel()
    running_var = np.asarray(running_var, dtype=float).ravel()

    if bandwidth is None:
        bandwidth = 1.06 * np.std(running_var) * len(running_var) ** (-0.2)

    # Select observations within bandwidth
    left_mask = (running_var >= cutoff - bandwidth) & (running_var < cutoff)
    right_mask = (running_var >= cutoff) & (running_var <= cutoff + bandwidth)

    n_left = int(left_mask.sum())
    n_right = int(right_mask.sum())

    if n_left < 2 or n_right < 2:
        raise ValueError(
            f"Insufficient observations within bandwidth: "
            f"n_left={n_left}, n_right={n_right}. "
            f"Try increasing the bandwidth (current={bandwidth:.4f})."
        )

    # Centered running variable
    r_left = running_var[left_mask] - cutoff
    r_right = running_var[right_mask] - cutoff
    y_left = outcome[left_mask]
    y_right = outcome[right_mask]

    # Local linear regression: left side
    X_left = np.column_stack([np.ones(n_left), r_left])
    beta_left = _ols_coefficients(X_left, y_left)

    # Local linear regression: right side
    X_right = np.column_stack([np.ones(n_right), r_right])
    beta_right = _ols_coefficients(X_right, y_right)

    # Treatment effect at cutoff
    ate = float(beta_right[0] - beta_left[0])

    # Standard errors
    resid_left = y_left - X_left @ beta_left
    resid_right = y_right - X_right @ beta_right
    se_left = np.sqrt(np.sum(resid_left**2) / (n_left - 2)) / np.sqrt(n_left)
    se_right = np.sqrt(np.sum(resid_right**2) / (n_right - 2)) / np.sqrt(n_right)
    se = float(np.sqrt(se_left**2 + se_right**2))

    z = stats.norm.ppf(0.975)
    return RDResult(
        ate=ate,
        se=se,
        ci_lower=ate - z * se,
        ci_upper=ate + z * se,
        n_left=n_left,
        n_right=n_right,
        bandwidth=bandwidth,
        details={
            "beta_left": beta_left.tolist(),
            "beta_right": beta_right.tolist(),
        },
    )


# ---------------------------------------------------------------------------
# Synthetic control
# ---------------------------------------------------------------------------


def synthetic_control(
    treated_outcomes: np.ndarray,
    donor_outcomes: np.ndarray,
    pre_period: int,
) -> SyntheticControlResult:
    """Estimate treatment effect using the synthetic control method.

    The synthetic control method constructs a data-driven counterfactual
    for a single treated unit by finding a weighted combination of
    untreated donor units that closely reproduces the treated unit's
    pre-treatment trajectory.  The treatment effect is the
    post-treatment gap between the observed and synthetic outcomes.

    Use synthetic control for single-unit case studies (e.g., the
    effect of a policy change on one country, or a management change
    at one firm) when a natural control group is unavailable.

    Parameters:
        treated_outcomes (np.ndarray): Outcomes for the treated unit
            over all time periods (1-D array of length T).
        donor_outcomes (np.ndarray): Outcomes for donor (control) units,
            shape ``(T, n_donors)``.
        pre_period (int): Number of pre-treatment time periods.  The
            first ``pre_period`` rows are used for fitting; the
            remainder is the post-treatment evaluation window.

    Returns:
        SyntheticControlResult: Estimated treatment effect (average
            post-period gap), donor weights, synthetic outcomes, and
            pre-period RMSE as a fit quality diagnostic.

    Raises:
        ValueError: If *pre_period* is out of range.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> T, n_donors = 20, 5
        >>> donors = rng.normal(size=(T, n_donors)).cumsum(axis=0)
        >>> treated = donors @ np.array([0.3, 0.5, 0.2, 0, 0])
        >>> treated[15:] += 2.0  # treatment effect of 2.0
        >>> result = synthetic_control(treated, donors, pre_period=15)
        >>> result.ate > 0
        True

    See Also:
        diff_in_diff: Two-group before/after comparison.
        regression_discontinuity: Cutoff-based causal identification.

    References:
        Abadie, A., Diamond, A. & Hainmueller, J. (2010). *Synthetic
        Control Methods for Comparative Case Studies.* JASA 105(490).
    """
    treated_outcomes = np.asarray(treated_outcomes, dtype=float).ravel()
    donor_outcomes = np.asarray(donor_outcomes, dtype=float)
    if donor_outcomes.ndim == 1:
        donor_outcomes = donor_outcomes.reshape(-1, 1)

    n_periods, n_donors = donor_outcomes.shape

    if pre_period < 1 or pre_period >= n_periods:
        raise ValueError(
            f"pre_period must be between 1 and {n_periods - 1}, got {pre_period}."
        )

    # Pre-treatment data
    y_pre = treated_outcomes[:pre_period]
    D_pre = donor_outcomes[:pre_period, :]

    # Find optimal weights by minimizing ||y_pre - D_pre @ w||^2
    # subject to w >= 0 and sum(w) = 1
    def objective(w: np.ndarray) -> float:
        return float(np.sum((y_pre - D_pre @ w) ** 2))

    def grad(w: np.ndarray) -> np.ndarray:
        return -2.0 * D_pre.T @ (y_pre - D_pre @ w)

    # Constraints: sum(w) = 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    # Bounds: w_i >= 0
    bounds = [(0.0, 1.0)] * n_donors
    w0 = np.ones(n_donors) / n_donors

    result = optimize.minimize(
        objective,
        w0,
        jac=grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    weights = result.x

    # Synthetic control outcomes
    synthetic = donor_outcomes @ weights
    gaps = treated_outcomes - synthetic

    # Pre-period fit
    pre_rmse = float(np.sqrt(np.mean(gaps[:pre_period] ** 2)))

    # Post-period treatment effect
    ate = float(np.mean(gaps[pre_period:]))

    return SyntheticControlResult(
        ate=ate,
        weights=weights,
        treated_outcomes=treated_outcomes,
        synthetic_outcomes=synthetic,
        pre_rmse=pre_rmse,
        gaps=gaps,
    )


# ---------------------------------------------------------------------------
# Difference-in-differences
# ---------------------------------------------------------------------------


def diff_in_diff(
    outcome: np.ndarray,
    treatment: np.ndarray,
    post: np.ndarray,
    entity: np.ndarray | None = None,
) -> DIDResult:
    r"""Estimate treatment effect using difference-in-differences (DID).

    DID is the workhorse of causal inference in finance and economics.
    It compares the before-after change in the treatment group to the
    before-after change in the control group, netting out common time
    trends:

    .. math::

        \hat{\delta} = (\bar{Y}_{T,\text{post}} - \bar{Y}_{T,\text{pre}})
        - (\bar{Y}_{C,\text{post}} - \bar{Y}_{C,\text{pre}})

    The key identifying assumption is **parallel trends**: absent the
    treatment, both groups would have followed the same trajectory.

    Interpretation:
        - **ate (delta)**: The causal effect of the treatment on
          the treated group, net of any common time trend.
        - Positive delta = treatment increased the outcome.
        - The group means (pre_treatment_mean, post_treatment_mean,
          pre_control_mean, post_control_mean) can be used to
          construct a "parallel trends" plot: plot the treated and
          control group means over time. If they are parallel
          pre-treatment, the assumption is plausible.

    How to check parallel trends:
        - Plot both group means in the pre-treatment period.
          They should move in parallel (not necessarily at the same
          level -- DID allows for level differences).
        - Run placebo DID tests at fake treatment dates in the
          pre-period. The estimated "effects" should be near zero.

    Financial examples:
        - Effect of a new regulation on trading costs (treated =
          affected firms, control = unaffected firms).
        - Impact of an IPO on a firm's operating performance
          (treated = IPO firms, control = matched private firms).
        - Did the COVID-19 lockdown affect commercial real estate
          values differently than residential?

    Parameters:
        outcome: Observed outcomes (1-D array).
        treatment: Binary group indicator: 1 = treatment group,
            0 = control group.
        post: Binary time indicator: 1 = post-treatment,
            0 = pre-treatment.
        entity: Entity identifiers for panel data. If provided,
            entity fixed effects are included.

    Returns:
        DIDResult with the DID estimate, standard error, 95% CI,
        and group means for diagnostics.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> n = 400
        >>> treat = np.repeat([0, 1], n // 2)
        >>> post_period = np.tile([0, 1], n // 2)
        >>> Y = 1.0 + 0.5 * treat + 0.3 * post_period + 2.0 * treat * post_period
        >>> Y += rng.normal(0, 0.5, n)
        >>> result = diff_in_diff(Y, treat, post_period)
        >>> print(f"DID: {result.ate:.2f} [{result.ci_lower:.2f}, {result.ci_upper:.2f}]")

    See Also:
        synthetic_control: For single-unit case studies.
        event_study: For measuring effects around specific dates.
    """
    outcome = np.asarray(outcome, dtype=float).ravel()
    treatment = np.asarray(treatment, dtype=float).ravel()
    post = np.asarray(post, dtype=float).ravel()

    n = len(outcome)

    # Group means
    pre_treat_mask = (treatment == 1) & (post == 0)
    post_treat_mask = (treatment == 1) & (post == 1)
    pre_ctrl_mask = (treatment == 0) & (post == 0)
    post_ctrl_mask = (treatment == 0) & (post == 1)

    pre_treat_mean = float(np.mean(outcome[pre_treat_mask]))
    post_treat_mean = float(np.mean(outcome[post_treat_mask]))
    pre_ctrl_mean = float(np.mean(outcome[pre_ctrl_mask]))
    post_ctrl_mean = float(np.mean(outcome[post_ctrl_mask]))

    # Build regression: Y = alpha + beta1*treatment + beta2*post + delta*(treatment*post)
    interaction = treatment * post
    X = np.column_stack([np.ones(n), treatment, post, interaction])

    if entity is not None:
        # With entity FE, use group means directly (canonical 2x2 DID)
        delta = (post_treat_mean - pre_treat_mean) - (post_ctrl_mean - pre_ctrl_mean)
        # Compute residuals from full OLS for SE estimation
        entity = np.asarray(entity)
        beta = _ols_coefficients(X, outcome)
        resid = outcome - X @ beta
    else:
        beta = _ols_coefficients(X, outcome)
        resid = outcome - X @ beta
        delta = float(beta[3])  # interaction coefficient

    # Standard error (HC1 robust)
    k = X.shape[1]
    sigma2 = np.sum(resid**2) / (n - k)
    XtX_inv = np.linalg.inv(X.T @ X)
    se_delta = float(np.sqrt(sigma2 * XtX_inv[3, 3]))

    z = stats.norm.ppf(0.975)
    return DIDResult(
        ate=delta,
        se=se_delta,
        ci_lower=delta - z * se_delta,
        ci_upper=delta + z * se_delta,
        pre_treatment_mean=pre_treat_mean,
        post_treatment_mean=post_treat_mean,
        pre_control_mean=pre_ctrl_mean,
        post_control_mean=post_ctrl_mean,
        details={"estimator": "did", "has_entity_fe": entity is not None},
    )


# ---------------------------------------------------------------------------
# Result dataclasses for new estimators
# ---------------------------------------------------------------------------


@dataclass
class GrangerResult:
    """Result container for Granger causality test.

    Parameters
    ----------
    f_statistic : float
        F-statistic from the Granger causality test.
    p_value : float
        P-value associated with the F-statistic.
    optimal_lag : int
        Lag order selected (either user-specified or chosen by BIC).
    direction : str
        String describing the causal direction tested
        (e.g., ``'x -> y'`` or ``'y -> x'``).
    all_lags : dict
        Mapping from each tested lag to ``{'f_stat': ..., 'p_value': ...}``.
    reject : bool
        Whether the null of no Granger causality is rejected at the given
        significance level.
    details : dict
        Additional diagnostics (AIC, BIC per lag if computed).
    """

    f_statistic: float
    p_value: float
    optimal_lag: int
    direction: str
    all_lags: dict = field(default_factory=dict)
    reject: bool = False
    details: dict = field(default_factory=dict)


@dataclass
class IVResult:
    """Result container for instrumental variable (2SLS) estimation.

    Parameters
    ----------
    coefficient : float
        2SLS coefficient on the endogenous regressor.
    se : float
        Standard error of the coefficient.
    ci_lower : float
        Lower bound of the 95% confidence interval.
    ci_upper : float
        Upper bound of the 95% confidence interval.
    first_stage_f : float
        First-stage F-statistic for instrument relevance. Values below 10
        suggest weak instruments (Staiger-Stock rule of thumb).
    hausman_stat : float or None
        Hausman test statistic comparing OLS and 2SLS (chi-squared).
    hausman_p : float or None
        P-value for the Hausman test. Small p rejects exogeneity of the
        regressor, supporting the use of IV.
    sargan_stat : float or None
        Sargan overidentification test statistic (chi-squared). Only
        available when there are more instruments than endogenous
        regressors.
    sargan_p : float or None
        P-value for the Sargan test. Small p rejects instrument validity.
    n_obs : int
        Number of observations.
    n_instruments : int
        Number of instruments.
    details : dict
        Additional diagnostics (OLS coefficient, first-stage coefficients).
    """

    coefficient: float
    se: float
    ci_lower: float
    ci_upper: float
    first_stage_f: float
    hausman_stat: float | None = None
    hausman_p: float | None = None
    sargan_stat: float | None = None
    sargan_p: float | None = None
    n_obs: int = 0
    n_instruments: int = 0
    details: dict = field(default_factory=dict)


@dataclass
class EventStudyResult:
    """Result container for event study analysis.

    Parameters
    ----------
    car : float
        Cumulative Abnormal Return over the event window.
    car_se : float
        Standard error of the CAR.
    car_t_stat : float
        t-statistic for the CAR (null: CAR = 0).
    car_p_value : float
        Two-sided p-value for the CAR t-test.
    abnormal_returns : np.ndarray
        Abnormal returns for each day in the event window.
    cumulative_ar : np.ndarray
        Running cumulative abnormal returns over the event window.
    estimation_alpha : float
        Intercept from the estimation window market model.
    estimation_beta : float
        Slope (market beta) from the estimation window market model.
    estimation_sigma : float
        Residual standard deviation from the estimation window.
    n_events : int
        Number of events analyzed.
    cross_sectional_t : float or None
        Cross-sectional t-statistic when multiple events are tested.
    cross_sectional_p : float or None
        P-value for the cross-sectional test.
    details : dict
        Additional diagnostics per event.
    """

    car: float
    car_se: float
    car_t_stat: float
    car_p_value: float
    abnormal_returns: np.ndarray
    cumulative_ar: np.ndarray
    estimation_alpha: float
    estimation_beta: float
    estimation_sigma: float
    n_events: int = 1
    cross_sectional_t: float | None = None
    cross_sectional_p: float | None = None
    details: dict = field(default_factory=dict)


@dataclass
class SyntheticControlWeightsResult:
    """Result container for enhanced synthetic control with inference.

    Parameters
    ----------
    ate : float
        Estimated treatment effect (post-period average gap).
    weights : np.ndarray
        Donor weights for the synthetic control unit.
    treated_outcomes : np.ndarray
        Observed treated outcomes.
    synthetic_outcomes : np.ndarray
        Synthetic control outcomes.
    pre_rmspe : float
        Pre-treatment root mean squared prediction error.
    post_rmspe : float
        Post-treatment root mean squared prediction error.
    rmspe_ratio : float
        Post/pre RMSPE ratio (key test statistic for inference).
    gaps : np.ndarray
        Period-by-period gaps (treated - synthetic).
    placebo_p_value : float or None
        P-value from placebo (permutation) test. Only computed if
        ``run_placebo=True``.
    placebo_ratios : np.ndarray or None
        RMSPE ratios from placebo runs.
    donor_names : list[str] | None
        Names of donor units for interpretability.
    details : dict
        Additional diagnostics.
    """

    ate: float
    weights: np.ndarray
    treated_outcomes: np.ndarray
    synthetic_outcomes: np.ndarray
    pre_rmspe: float
    post_rmspe: float
    rmspe_ratio: float
    gaps: np.ndarray
    placebo_p_value: float | None = None
    placebo_ratios: np.ndarray | None = None
    donor_names: list | None = None
    details: dict = field(default_factory=dict)


@dataclass
class CausalForestResult:
    """Result container for causal forest CATE estimation.

    Parameters
    ----------
    ate : float
        Average Treatment Effect (mean of CATE estimates).
    cate : np.ndarray
        Conditional Average Treatment Effect for each observation.
    ate_se : float
        Standard error of the ATE estimate.
    ci_lower : float
        Lower bound of the 95% confidence interval for the ATE.
    ci_upper : float
        Upper bound of the 95% confidence interval for the ATE.
    feature_importances : np.ndarray or None
        Feature importances from the forest (if available).
    details : dict
        Additional diagnostics.
    """

    ate: float
    cate: np.ndarray
    ate_se: float
    ci_lower: float
    ci_upper: float
    feature_importances: np.ndarray | None = None
    details: dict = field(default_factory=dict)


@dataclass
class MediationResult:
    """Result container for Baron-Kenny mediation analysis.

    Parameters
    ----------
    total_effect : float
        Total effect of treatment on outcome (path c).
    direct_effect : float
        Direct effect of treatment on outcome controlling for mediator
        (path c').
    indirect_effect : float
        Indirect effect through the mediator (a * b = c - c').
    sobel_stat : float
        Sobel test statistic for the indirect effect.
    sobel_p : float
        Two-sided p-value for the Sobel test.
    proportion_mediated : float
        Proportion of total effect explained by the mediator
        (indirect / total). Clamped to [0, 1].
    path_a : float
        Coefficient of treatment -> mediator regression.
    path_a_se : float
        Standard error of path a.
    path_b : float
        Coefficient of mediator -> outcome (controlling for treatment).
    path_b_se : float
        Standard error of path b.
    details : dict
        Additional diagnostics.
    """

    total_effect: float
    direct_effect: float
    indirect_effect: float
    sobel_stat: float
    sobel_p: float
    proportion_mediated: float
    path_a: float
    path_a_se: float
    path_b: float
    path_b_se: float
    details: dict = field(default_factory=dict)


@dataclass
class RDRobustResult:
    """Result container for robust regression discontinuity design.

    Parameters
    ----------
    ate : float
        Estimated treatment effect at the cutoff.
    se : float
        Standard error (robust, bias-corrected).
    ci_lower : float
        Lower bound of the 95% confidence interval.
    ci_upper : float
        Upper bound of the 95% confidence interval.
    bandwidth : float
        Bandwidth used for estimation.
    bandwidth_method : str
        Method used for bandwidth selection.
    n_left : int
        Number of observations to the left of the cutoff (within bandwidth).
    n_right : int
        Number of observations to the right of the cutoff (within bandwidth).
    poly_order : int
        Order of the local polynomial.
    mccrary_stat : float or None
        McCrary (2008) density test statistic. Large values indicate
        manipulation of the running variable.
    mccrary_p : float or None
        P-value for the McCrary test.
    rd_type : str
        ``'sharp'`` or ``'fuzzy'``.
    details : dict
        Additional diagnostics.
    """

    ate: float
    se: float
    ci_lower: float
    ci_upper: float
    bandwidth: float
    bandwidth_method: str
    n_left: int
    n_right: int
    poly_order: int
    mccrary_stat: float | None = None
    mccrary_p: float | None = None
    rd_type: str = "sharp"
    details: dict = field(default_factory=dict)


@dataclass
class BoundsResult:
    """Result container for Manski/Lee bounds analysis.

    Parameters
    ----------
    lower_bound : float
        Lower bound on the treatment effect.
    upper_bound : float
        Upper bound on the treatment effect.
    bound_type : str
        Type of bounds (``'manski'`` or ``'lee'``).
    identified : bool
        Whether the sign of the effect is identified (both bounds
        have the same sign).
    ci_lower : float
        Lower bound of the confidence interval for the lower bound.
    ci_upper : float
        Upper bound of the confidence interval for the upper bound.
    details : dict
        Additional diagnostics.
    """

    lower_bound: float
    upper_bound: float
    bound_type: str
    identified: bool
    ci_lower: float
    ci_upper: float
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Granger causality
# ---------------------------------------------------------------------------


def granger_causality(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = 10,
    significance: float = 0.05,
) -> GrangerResult:
    """Test whether time series *x* Granger-causes time series *y*.

    Granger causality tests whether past values of *x* contain information
    that helps predict *y* beyond what past values of *y* alone provide.
    The null hypothesis is that *x* does **not** Granger-cause *y*.

    **IMPORTANT**: Granger "causality" is really *predictive* causality,
    NOT true causality.  It means "x helps forecast y" -- this could be
    because x causes y, or because x responds faster to a common cause.
    The name is a historical misnomer.

    The test fits two VAR models for each candidate lag order *p*:

    .. math::

        \\text{Restricted:} \\quad y_t = \\alpha + \\sum_{i=1}^{p} \\beta_i y_{t-i} + \\varepsilon_t

        \\text{Unrestricted:} \\quad y_t = \\alpha + \\sum_{i=1}^{p} \\beta_i y_{t-i}
        + \\sum_{i=1}^{p} \\gamma_i x_{t-i} + \\varepsilon_t

    The F-statistic tests :math:`H_0: \\gamma_1 = \\cdots = \\gamma_p = 0`.

    Interpretation:
        - **reject = True**: Past x helps predict y beyond y's own
          history.  This is evidence of a predictive relationship.
        - **reject = False**: No evidence that x helps predict y.
        - **optimal_lag**: The lag order selected by BIC.  Economically,
          this tells you "how far back does x's influence on y extend?"
        - Check both directions: if x Granger-causes y AND y
          Granger-causes x, there may be a common driver (not a
          directional causal effect).
        - Both series should be stationary.  If they are I(1),
          difference first.  If they are cointegrated, use a VECM
          instead.

    Financial use cases:
        - Does the VIX Granger-cause S&P 500 returns?
        - Does order flow predict price changes?
        - Does the yield curve lead GDP growth?
        - Does sentiment data help forecast volatility?

    Parameters:
        x: Potential cause time series (1D array, length T).
        y: Potential effect time series (1D array, length T).
        max_lag: Maximum lag order to test. The optimal lag is selected
            by BIC. Default is 10.
        significance: Significance level for rejection. Default is 0.05.

    Returns:
        GrangerResult with F-statistic, p-value, optimal lag, direction,
        and per-lag results.

    Raises
    ------
    ValueError
        If the time series are too short for the requested ``max_lag``.

    Example
    -------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> x = rng.normal(size=200)
    >>> y = np.zeros(200)
    >>> for t in range(1, 200):
    ...     y[t] = 0.5 * y[t-1] + 0.3 * x[t-1] + rng.normal(0, 0.5)
    >>> result = granger_causality(x, y, max_lag=5)
    >>> result.reject  # should be True — x Granger-causes y
    True

    References
    ----------
    - Granger, C. W. J. (1969). "Investigating Causal Relations by
      Econometric Models and Cross-spectral Methods."
    - Toda, H. Y. & Yamamoto, T. (1995). "Statistical inference in
      vector autoregressions with possibly integrated processes."

    Notes
    -----
    Uses ``statsmodels.tsa.stattools.grangercausalitytests`` internally.
    The function tests :math:`x \\to y` (whether *x* helps predict *y*).
    To test the reverse direction, swap the arguments.
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if len(x) != len(y):
        raise ValueError(
            f"x and y must have the same length, got {len(x)} and {len(y)}."
        )
    if len(x) < max_lag + 3:
        raise ValueError(
            f"Time series too short ({len(x)}) for max_lag={max_lag}. "
            f"Need at least {max_lag + 3} observations."
        )

    # statsmodels expects data as [y, x] — column 0 is the "effect",
    # column 1 is the potential "cause"
    data = np.column_stack([y, x])

    # Run the test (verbose=False suppresses printing)
    results = grangercausalitytests(data, maxlag=max_lag, verbose=False)

    # Collect results for each lag and pick optimal by BIC
    all_lags: dict[int, dict] = {}
    best_lag = 1
    best_bic = np.inf

    for lag in range(1, max_lag + 1):
        lag_result = results[lag]
        # lag_result is a tuple: (test_dict, [ols_restricted, ols_unrestricted])
        test_dict = lag_result[0]
        ols_restricted = lag_result[1][0]
        ols_unrestricted = lag_result[1][1]

        f_stat = test_dict["ssr_ftest"][0]
        p_val = test_dict["ssr_ftest"][1]
        bic_unrestricted = ols_unrestricted.bic

        all_lags[lag] = {
            "f_stat": float(f_stat),
            "p_value": float(p_val),
            "bic": float(bic_unrestricted),
        }

        if bic_unrestricted < best_bic:
            best_bic = bic_unrestricted
            best_lag = lag

    optimal = all_lags[best_lag]

    return GrangerResult(
        f_statistic=optimal["f_stat"],
        p_value=optimal["p_value"],
        optimal_lag=best_lag,
        direction="x -> y",
        all_lags=all_lags,
        reject=optimal["p_value"] < significance,
        details={"max_lag_tested": max_lag, "significance": significance},
    )


# ---------------------------------------------------------------------------
# Instrumental variable (2SLS)
# ---------------------------------------------------------------------------


def instrumental_variable(
    outcome: np.ndarray,
    endogenous: np.ndarray,
    instruments: np.ndarray,
    exogenous: np.ndarray | None = None,
) -> IVResult:
    """Two-stage least squares (2SLS) instrumental variable estimation.

    IV estimation addresses endogeneity — the situation where an explanatory
    variable is correlated with the error term (e.g., due to omitted
    variables, simultaneity, or measurement error). An instrument *Z* must
    satisfy:

    1. **Relevance**: *Z* is correlated with the endogenous regressor *X*.
    2. **Exclusion**: *Z* affects the outcome *Y* only through *X*.

    .. math::

        \\text{First stage:}  \\quad X = Z \\pi + W \\delta + v

        \\text{Second stage:} \\quad Y = \\hat{X} \\beta + W \\gamma + \\varepsilon

    Diagnostics provided:
        - **First-stage F-statistic**: Tests instrument relevance.
          F < 10 indicates weak instruments (Staiger & Stock, 1997).
        - **Hausman test**: Compares OLS and 2SLS coefficients. Rejection
          supports the endogeneity of the regressor and the need for IV.
        - **Sargan overidentification test**: When there are more
          instruments than endogenous regressors, tests the joint validity
          of instruments. Rejection suggests at least one instrument is
          invalid.

    Financial use cases:
        - Estimating the effect of leverage on firm value using regulatory
          shocks as instruments.
        - Measuring the causal impact of analyst coverage on stock
          liquidity using index inclusion as an instrument.
        - Identifying the effect of order flow on prices using weather
          (affects trader mood) as an instrument.

    Parameters
    ----------
    outcome : np.ndarray
        Dependent variable *Y* (1D array, length n).
    endogenous : np.ndarray
        Endogenous regressor *X* (1D array, length n).
    instruments : np.ndarray
        Instrumental variables *Z* (n, k) matrix where k >= 1.
        More instruments than endogenous regressors allows the
        Sargan overidentification test.
    exogenous : np.ndarray or None
        Exogenous control variables *W* (n, m). An intercept is
        always added. Default is None (intercept only).

    Returns
    -------
    IVResult
        2SLS coefficient, standard error, confidence interval,
        and diagnostic tests.

    Raises
    ------
    ValueError
        If dimensions are incompatible or there are too few instruments.

    Example
    -------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n = 1000
    >>> z = rng.normal(size=(n, 1))  # instrument
    >>> u = rng.normal(size=n)       # unobserved confounder
    >>> x = z.ravel() + u + rng.normal(0, 0.5, n)  # endogenous
    >>> y = 2.0 * x + u + rng.normal(0, 0.5, n)    # outcome
    >>> result = instrumental_variable(y, x, z)
    >>> abs(result.coefficient - 2.0) < 0.5
    True

    References
    ----------
    - Angrist, J. D. & Pischke, J.-S. (2009). "Mostly Harmless
      Econometrics: An Empiricist's Companion."
    - Staiger, D. & Stock, J. H. (1997). "Instrumental Variables
      Regression with Weak Instruments."
    - Sargan, J. D. (1958). "The Estimation of Economic Relationships
      Using Instrumental Variables."
    """
    outcome = np.asarray(outcome, dtype=float).ravel()
    endogenous = np.asarray(endogenous, dtype=float).ravel()
    instruments = np.asarray(instruments, dtype=float)
    if instruments.ndim == 1:
        instruments = instruments.reshape(-1, 1)

    n = len(outcome)
    n_instruments = instruments.shape[1]

    if n_instruments < 1:
        raise ValueError("At least one instrument is required.")

    # Build the exogenous matrix (intercept + controls)
    if exogenous is not None:
        exogenous = np.asarray(exogenous, dtype=float)
        if exogenous.ndim == 1:
            exogenous = exogenous.reshape(-1, 1)
        W = np.column_stack([np.ones(n), exogenous])
    else:
        W = np.ones((n, 1))

    k_exog = W.shape[1]

    # Full instrument matrix: [W, Z]
    Z_full = np.column_stack([W, instruments])

    # --- First stage: regress endogenous on instruments + controls ---
    beta_first = _ols_coefficients(Z_full, endogenous)
    x_hat = Z_full @ beta_first
    first_stage_resid = endogenous - x_hat

    # First-stage F-statistic (test that instrument coefficients are
    # jointly zero)
    # Restricted model: endogenous ~ W only
    beta_restricted = _ols_coefficients(W, endogenous)
    x_hat_restricted = W @ beta_restricted
    ssr_restricted = np.sum((endogenous - x_hat_restricted) ** 2)
    ssr_unrestricted = np.sum(first_stage_resid**2)

    df_num = n_instruments
    df_den = n - Z_full.shape[1]
    first_stage_f = float(
        ((ssr_restricted - ssr_unrestricted) / df_num) / (ssr_unrestricted / df_den)
    )

    # --- Second stage: regress outcome on x_hat + controls ---
    X_second = np.column_stack([W, x_hat])
    beta_2sls = _ols_coefficients(X_second, outcome)

    # The 2SLS coefficient on the endogenous variable is the last one
    coef_2sls = float(beta_2sls[-1])

    # Standard errors: use original endogenous (not x_hat) for residuals
    X_orig = np.column_stack([W, endogenous])
    resid_2sls = outcome - X_orig @ beta_2sls  # not X_second @ beta_2sls
    sigma2_2sls = float(np.sum(resid_2sls**2) / (n - X_second.shape[1]))

    # Variance: sigma^2 * (X_hat'X_hat)^{-1}
    XhXh_inv = np.linalg.inv(X_second.T @ X_second)
    se_2sls = float(np.sqrt(sigma2_2sls * XhXh_inv[-1, -1]))

    z_crit = stats.norm.ppf(0.975)

    # --- OLS for comparison (Hausman test) ---
    beta_ols = _ols_coefficients(X_orig, outcome)
    coef_ols = float(beta_ols[-1])
    resid_ols = outcome - X_orig @ beta_ols
    sigma2_ols = float(np.sum(resid_ols**2) / (n - X_orig.shape[1]))
    XoXo_inv = np.linalg.inv(X_orig.T @ X_orig)
    se_ols = float(np.sqrt(sigma2_ols * XoXo_inv[-1, -1]))

    # Hausman test: H = (b_2SLS - b_OLS)^2 / (Var(b_2SLS) - Var(b_OLS))
    var_diff = sigma2_2sls * XhXh_inv[-1, -1] - sigma2_ols * XoXo_inv[-1, -1]
    if var_diff > 0:
        hausman_stat = float((coef_2sls - coef_ols) ** 2 / var_diff)
        hausman_p = float(1.0 - stats.chi2.cdf(hausman_stat, df=1))
    else:
        hausman_stat = None
        hausman_p = None

    # --- Sargan overidentification test (if over-identified) ---
    sargan_stat = None
    sargan_p = None
    if n_instruments > 1:
        # Regress 2SLS residuals on all instruments + exogenous
        beta_sargan = _ols_coefficients(Z_full, resid_2sls)
        resid_sargan = resid_2sls - Z_full @ beta_sargan
        r2_sargan = 1.0 - np.sum(resid_sargan**2) / np.sum(
            (resid_2sls - np.mean(resid_2sls)) ** 2
        )
        sargan_stat = float(n * r2_sargan)
        sargan_df = n_instruments - 1  # number of overidentifying restrictions
        sargan_p = float(1.0 - stats.chi2.cdf(sargan_stat, df=sargan_df))

    return IVResult(
        coefficient=coef_2sls,
        se=se_2sls,
        ci_lower=coef_2sls - z_crit * se_2sls,
        ci_upper=coef_2sls + z_crit * se_2sls,
        first_stage_f=first_stage_f,
        hausman_stat=hausman_stat,
        hausman_p=hausman_p,
        sargan_stat=sargan_stat,
        sargan_p=sargan_p,
        n_obs=n,
        n_instruments=n_instruments,
        details={
            "coef_ols": coef_ols,
            "se_ols": se_ols,
            "first_stage_coefficients": beta_first[k_exog:].tolist(),
        },
    )


# ---------------------------------------------------------------------------
# Event study
# ---------------------------------------------------------------------------


def event_study(
    returns: np.ndarray,
    market_returns: np.ndarray,
    event_indices: list[int] | np.ndarray,
    estimation_window: int = 120,
    event_window_pre: int = 5,
    event_window_post: int = 5,
    gap: int = 10,
) -> EventStudyResult:
    """Conduct a market-model event study for one or more events.

    The event study methodology estimates abnormal returns around an event
    by comparing realized returns to expected returns from a market model
    estimated during a pre-event estimation window.

    **Methodology**:

    1. **Estimation window**: Fit :math:`R_{i,t} = \\alpha + \\beta R_{m,t}
       + \\varepsilon_t` over a window ending ``gap`` days before the event.

    2. **Event window**: Compute abnormal returns
       :math:`AR_t = R_{i,t} - (\\hat{\\alpha} + \\hat{\\beta} R_{m,t})`
       for each day in ``[-event_window_pre, +event_window_post]``.

    3. **CAR**: Cumulative Abnormal Return :math:`= \\sum AR_t`.

    4. **Statistical test**: :math:`t = \\text{CAR} / (\\hat{\\sigma}
       \\sqrt{L})` where *L* is the length of the event window and
       :math:`\\hat{\\sigma}` is the residual std from the estimation
       window.

    5. **Cross-sectional test** (multiple events): Use the distribution
       of individual CARs across events.

    Financial use cases:
        - Impact of earnings announcements on stock prices.
        - Effect of M&A announcements on acquirer/target returns.
        - Regulatory changes (e.g., short-selling bans).
        - Central bank policy announcements.

    Parameters
    ----------
    returns : np.ndarray
        Asset return series (1D array, length T).
    market_returns : np.ndarray
        Market (benchmark) return series (1D array, length T).
    event_indices : list[int] or np.ndarray
        Index positions in the return series where each event occurs.
        For a single event, pass ``[idx]``.
    estimation_window : int
        Number of days in the estimation window. Default is 120.
    event_window_pre : int
        Number of days before the event in the event window. Default 5.
    event_window_post : int
        Number of days after the event in the event window. Default 5.
    gap : int
        Gap between the estimation window and the event window to avoid
        contamination. Default is 10.

    Returns
    -------
    EventStudyResult
        CAR, abnormal returns, t-statistics, and diagnostics.

    Raises
    ------
    ValueError
        If any event does not have enough data for the estimation or
        event windows.

    Example
    -------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> market = rng.normal(0.0005, 0.01, 500)
    >>> stock = 0.001 + 1.2 * market + rng.normal(0, 0.005, 500)
    >>> stock[250:256] += 0.02  # inject positive event at day 250
    >>> result = event_study(stock, market, [250], event_window_post=5)
    >>> result.car > 0  # should detect the positive abnormal return
    True

    References
    ----------
    - MacKinlay, A. C. (1997). "Event Studies in Economics and Finance."
      *Journal of Economic Literature*, 35(1), 13-39.
    - Brown, S. J. & Warner, J. B. (1985). "Using Daily Stock Returns."
      *Journal of Financial Economics*, 14(1), 3-31.
    """
    returns = np.asarray(returns, dtype=float).ravel()
    market_returns = np.asarray(market_returns, dtype=float).ravel()
    event_indices = np.asarray(event_indices, dtype=int).ravel()

    if len(returns) != len(market_returns):
        raise ValueError("returns and market_returns must have the same length.")

    event_window_len = event_window_pre + event_window_post + 1
    all_cars = []
    all_ars = []
    all_alphas = []
    all_betas = []
    all_sigmas = []

    for event_idx in event_indices:
        # Define windows
        est_end = event_idx - gap - event_window_pre
        est_start = est_end - estimation_window

        ew_start = event_idx - event_window_pre
        ew_end = event_idx + event_window_post + 1

        if est_start < 0:
            raise ValueError(
                f"Event at index {event_idx}: estimation window starts "
                f"before the series (need index {est_start})."
            )
        if ew_end > len(returns):
            raise ValueError(
                f"Event at index {event_idx}: event window extends "
                f"beyond the series (need index {ew_end - 1}, "
                f"series length {len(returns)})."
            )

        # Estimation window data
        r_est = returns[est_start:est_end]
        m_est = market_returns[est_start:est_end]

        # Fit market model: R_i = alpha + beta * R_m
        X_est = np.column_stack([np.ones(len(r_est)), m_est])
        beta_hat = _ols_coefficients(X_est, r_est)
        alpha_hat = float(beta_hat[0])
        beta_market = float(beta_hat[1])
        resid_est = r_est - X_est @ beta_hat
        sigma_est = float(np.std(resid_est, ddof=2))

        # Event window abnormal returns
        r_event = returns[ew_start:ew_end]
        m_event = market_returns[ew_start:ew_end]
        expected = alpha_hat + beta_market * m_event
        ar = r_event - expected

        car = float(np.sum(ar))
        all_cars.append(car)
        all_ars.append(ar)
        all_alphas.append(alpha_hat)
        all_betas.append(beta_market)
        all_sigmas.append(sigma_est)

    # Aggregate across events
    n_events = len(event_indices)

    if n_events == 1:
        ar_avg = all_ars[0]
        car_avg = all_cars[0]
        sigma = all_sigmas[0]
        car_se = sigma * np.sqrt(event_window_len)
        alpha_out = all_alphas[0]
        beta_out = all_betas[0]
        sigma_out = sigma

        if car_se > 0:
            car_t = car_avg / car_se
        else:
            car_t = 0.0
        car_p = float(2.0 * (1.0 - stats.norm.cdf(abs(car_t))))

        cross_t = None
        cross_p = None
    else:
        # Average abnormal returns across events
        ar_matrix = np.array(all_ars)  # (n_events, event_window_len)
        ar_avg = np.mean(ar_matrix, axis=0)
        car_avg = float(np.sum(ar_avg))

        # Standardized test: CAR / (sigma * sqrt(L))
        sigma = float(np.mean(all_sigmas))
        car_se = sigma * np.sqrt(event_window_len) / np.sqrt(n_events)
        alpha_out = float(np.mean(all_alphas))
        beta_out = float(np.mean(all_betas))
        sigma_out = sigma

        if car_se > 0:
            car_t = car_avg / car_se
        else:
            car_t = 0.0
        car_p = float(2.0 * (1.0 - stats.norm.cdf(abs(car_t))))

        # Cross-sectional test
        cars = np.array(all_cars)
        cs_mean = np.mean(cars)
        cs_se = np.std(cars, ddof=1) / np.sqrt(n_events)
        if cs_se > 0:
            cross_t = float(cs_mean / cs_se)
            cross_p = float(2.0 * (1.0 - stats.norm.cdf(abs(cross_t))))
        else:
            cross_t = 0.0
            cross_p = 1.0

    cumulative_ar = np.cumsum(ar_avg)

    return EventStudyResult(
        car=car_avg,
        car_se=car_se,
        car_t_stat=car_t,
        car_p_value=car_p,
        abnormal_returns=ar_avg,
        cumulative_ar=cumulative_ar,
        estimation_alpha=alpha_out,
        estimation_beta=beta_out,
        estimation_sigma=sigma_out,
        n_events=n_events,
        cross_sectional_t=cross_t,
        cross_sectional_p=cross_p,
        details={
            "estimation_window": estimation_window,
            "event_window_pre": event_window_pre,
            "event_window_post": event_window_post,
            "gap": gap,
            "individual_cars": all_cars,
        },
    )


# ---------------------------------------------------------------------------
# Enhanced synthetic control with placebo inference
# ---------------------------------------------------------------------------


def _fit_synthetic_weights(
    y_pre: np.ndarray,
    D_pre: np.ndarray,
) -> np.ndarray:
    """Find convex combination weights minimizing pre-treatment MSPE.

    Parameters
    ----------
    y_pre : np.ndarray
        Pre-treatment outcomes for the treated unit (1D).
    D_pre : np.ndarray
        Pre-treatment outcomes for donor units (n_pre, n_donors).

    Returns
    -------
    np.ndarray
        Optimal donor weights (sum to 1, non-negative).
    """
    n_donors = D_pre.shape[1]

    def objective(w: np.ndarray) -> float:
        return float(np.sum((y_pre - D_pre @ w) ** 2))

    def grad(w: np.ndarray) -> np.ndarray:
        return -2.0 * D_pre.T @ (y_pre - D_pre @ w)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * n_donors
    w0 = np.ones(n_donors) / n_donors

    result = optimize.minimize(
        objective,
        w0,
        jac=grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result.x


def synthetic_control_weights(
    treated_outcomes: np.ndarray,
    donor_outcomes: np.ndarray,
    pre_period: int,
    run_placebo: bool = False,
    donor_names: list[str] | None = None,
) -> SyntheticControlWeightsResult:
    """Enhanced synthetic control with donor selection and placebo inference.

    Extends the basic synthetic control method with:

    1. **Pre-treatment fit quality**: RMSPE (root mean squared prediction
       error) measures how well the synthetic unit matches the treated
       unit before treatment. Lower is better.

    2. **Placebo (permutation) tests**: Applies the synthetic control
       method iteratively, treating each donor as if it were the treated
       unit. The RMSPE ratio (post/pre) for the true treated unit is
       compared to the distribution of placebo ratios. A p-value is the
       fraction of placebo ratios at least as large as the treated unit's.

    3. **Gap plot data**: Period-by-period gaps and cumulative effects
       for visualization.

    .. math::

        \\text{RMSPE ratio} = \\frac{\\text{RMSPE}_{\\text{post}}}{\\text{RMSPE}_{\\text{pre}}}

    Financial use cases:
        - Impact of a merger/acquisition on the target firm's stock price
          relative to a synthetic peer.
        - Effect of a regulatory change on trading volume for a single
          exchange.
        - Impact of a central bank's unconventional monetary policy on
          a country's equity market.

    Parameters
    ----------
    treated_outcomes : np.ndarray
        Outcomes for the treated unit over all time periods (1D, length T).
    donor_outcomes : np.ndarray
        Outcomes for donor units (T, n_donors).
    pre_period : int
        Number of pre-treatment time periods.
    run_placebo : bool
        If True, runs placebo tests for each donor to compute
        permutation-based p-values. Default is False (expensive for
        many donors).
    donor_names : list[str] or None
        Optional names for donor units for interpretability.

    Returns
    -------
    SyntheticControlWeightsResult
        Weights, RMSPE, placebo p-value, gap data.

    Example
    -------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> T, J = 50, 10
    >>> common = np.cumsum(rng.normal(0, 0.5, T))
    >>> donors = np.column_stack([
    ...     common + rng.normal(0, 0.3, T) for _ in range(J)
    ... ])
    >>> true_w = rng.dirichlet(np.ones(J))
    >>> treated = donors @ true_w + rng.normal(0, 0.1, T)
    >>> treated[30:] += 5.0  # treatment effect of 5
    >>> result = synthetic_control_weights(treated, donors, 30,
    ...                                    run_placebo=True)
    >>> result.ate  # should be close to 5.0
    5.0...
    >>> result.placebo_p_value < 0.2  # significant
    True

    References
    ----------
    - Abadie, A., Diamond, A. & Hainmueller, J. (2010). "Synthetic
      Control Methods for Comparative Case Studies."
    - Abadie, A. (2021). "Using Synthetic Controls: Feasibility, Data
      Requirements, and Methodological Aspects."
    """
    treated_outcomes = np.asarray(treated_outcomes, dtype=float).ravel()
    donor_outcomes = np.asarray(donor_outcomes, dtype=float)
    if donor_outcomes.ndim == 1:
        donor_outcomes = donor_outcomes.reshape(-1, 1)

    n_periods, n_donors = donor_outcomes.shape

    if pre_period < 2 or pre_period >= n_periods:
        raise ValueError(
            f"pre_period must be between 2 and {n_periods - 1}, " f"got {pre_period}."
        )

    # Pre- and post-treatment splits
    y_pre = treated_outcomes[:pre_period]
    D_pre = donor_outcomes[:pre_period, :]

    # Fit weights
    weights = _fit_synthetic_weights(y_pre, D_pre)

    # Synthetic control outcomes
    synthetic = donor_outcomes @ weights
    gaps = treated_outcomes - synthetic

    # RMSPE
    pre_rmspe = float(np.sqrt(np.mean(gaps[:pre_period] ** 2)))
    post_rmspe = float(np.sqrt(np.mean(gaps[pre_period:] ** 2)))
    rmspe_ratio = post_rmspe / pre_rmspe if pre_rmspe > 1e-12 else np.inf

    # Post-period ATE
    ate = float(np.mean(gaps[pre_period:]))

    # Placebo tests
    placebo_p_value = None
    placebo_ratios = None

    if run_placebo and n_donors >= 2:
        ratios = [rmspe_ratio]
        for j in range(n_donors):
            # Treat donor j as the "treated" unit
            placebo_treated = donor_outcomes[:, j]
            # Remaining donors (including original treated as a donor)
            remaining_idx = [i for i in range(n_donors) if i != j]
            placebo_donors = np.column_stack(
                [donor_outcomes[:, remaining_idx], treated_outcomes.reshape(-1, 1)]
            )

            placebo_y_pre = placebo_treated[:pre_period]
            placebo_D_pre = placebo_donors[:pre_period, :]

            try:
                placebo_w = _fit_synthetic_weights(placebo_y_pre, placebo_D_pre)
                placebo_synth = placebo_donors @ placebo_w
                placebo_gaps = placebo_treated - placebo_synth
                p_pre = float(np.sqrt(np.mean(placebo_gaps[:pre_period] ** 2)))
                p_post = float(np.sqrt(np.mean(placebo_gaps[pre_period:] ** 2)))
                p_ratio = p_post / p_pre if p_pre > 1e-12 else np.inf
                ratios.append(p_ratio)
            except Exception:
                # Skip donors where optimization fails
                continue

        placebo_ratios = np.array(ratios)
        # p-value: fraction of placebo ratios >= treated ratio
        placebo_p_value = float(np.mean(placebo_ratios >= rmspe_ratio))

    return SyntheticControlWeightsResult(
        ate=ate,
        weights=weights,
        treated_outcomes=treated_outcomes,
        synthetic_outcomes=synthetic,
        pre_rmspe=pre_rmspe,
        post_rmspe=post_rmspe,
        rmspe_ratio=rmspe_ratio,
        gaps=gaps,
        placebo_p_value=placebo_p_value,
        placebo_ratios=placebo_ratios,
        donor_names=donor_names,
        details={"n_donors": n_donors, "pre_period": pre_period},
    )


# ---------------------------------------------------------------------------
# Causal forest (pure sklearn-based CATE estimation)
# ---------------------------------------------------------------------------


def causal_forest(
    outcome: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    n_estimators: int = 200,
    min_samples_leaf: int = 5,
    max_depth: int | None = None,
    honest: bool = True,
    seed: int = 42,
) -> CausalForestResult:
    """Estimate heterogeneous treatment effects using a causal forest.

    Implements the T-learner approach for CATE (Conditional Average
    Treatment Effect) estimation using random forests. Two separate
    forests are trained — one for the treated outcomes and one for the
    control outcomes — and the CATE is estimated as the difference in
    predictions.

    If ``honest=True``, the data is split: one half for building the
    tree structure, the other half for estimating leaf predictions.
    This reduces overfitting and provides valid confidence intervals.

    .. math::

        \\hat{\\tau}(x) = \\hat{\\mu}_1(x) - \\hat{\\mu}_0(x)

    where :math:`\\hat{\\mu}_1` is the treated outcome model and
    :math:`\\hat{\\mu}_0` is the control outcome model.

    Financial use cases:
        - Which firms benefit most from analyst coverage?
        - How does the effect of a tax change vary by firm size?
        - Heterogeneous impact of ESG scores on cost of capital.
        - Personalized treatment effects for portfolio tilts.

    Parameters
    ----------
    outcome : np.ndarray
        Observed outcomes (1D array, length n).
    treatment : np.ndarray
        Binary treatment indicator (1D array of 0s and 1s).
    covariates : np.ndarray
        Covariate matrix (n, p) for effect heterogeneity.
    n_estimators : int
        Number of trees in each forest. Default is 200.
    min_samples_leaf : int
        Minimum samples per leaf. Larger values give smoother estimates.
        Default is 5.
    max_depth : int or None
        Maximum tree depth. None means unlimited. Default is None.
    honest : bool
        If True, uses sample splitting for honest estimation.
        Default is True.
    seed : int
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    CausalForestResult
        ATE, CATE for each observation, standard error, confidence
        interval, and feature importances.

    Example
    -------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n = 1000
    >>> X = rng.normal(size=(n, 3))
    >>> T = (rng.uniform(size=n) > 0.5).astype(float)
    >>> # Heterogeneous effect: tau(x) = 2 + x[:,0]
    >>> tau = 2.0 + X[:, 0]
    >>> Y = X[:, 1] + tau * T + rng.normal(0, 0.5, n)
    >>> result = causal_forest(Y, T, X)
    >>> abs(result.ate - 2.0) < 1.0
    True

    References
    ----------
    - Wager, S. & Athey, S. (2018). "Estimation and Inference of
      Heterogeneous Treatment Effects using Random Forests."
    - Kunzel, S. R. et al. (2019). "Metalearners for Estimating
      Heterogeneous Treatment Effects using Machine Learning."
    """
    from sklearn.ensemble import RandomForestRegressor

    outcome = np.asarray(outcome, dtype=float).ravel()
    treatment = np.asarray(treatment, dtype=float).ravel()
    covariates = np.asarray(covariates, dtype=float)
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    rng = np.random.default_rng(seed)
    n = len(outcome)
    treated_mask = treatment == 1
    control_mask = treatment == 0

    if honest:
        # Split data into structure and estimation halves
        perm = rng.permutation(n)
        half = n // 2
        struct_idx = perm[:half]
        est_idx = perm[half:]

        # Build trees on structure half, predict on estimation half
        X_struct = covariates[struct_idx]
        y_struct = outcome[struct_idx]
        t_struct = treatment[struct_idx]

        X_est = covariates[est_idx]
        y_est = outcome[est_idx]
        t_est = treatment[est_idx]

        # Train forests on structure data
        t_mask_s = t_struct == 1
        c_mask_s = t_struct == 0

        rf1 = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            random_state=seed,
        )
        rf0 = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            random_state=seed + 1,
        )

        if t_mask_s.sum() < min_samples_leaf or c_mask_s.sum() < min_samples_leaf:
            raise ValueError(
                "Too few treated or control units in the structure half "
                "for honest estimation. Try honest=False or increase n."
            )

        rf1.fit(X_struct[t_mask_s], y_struct[t_mask_s])
        rf0.fit(X_struct[c_mask_s], y_struct[c_mask_s])

        # Predict CATE on ALL data
        mu1_all = rf1.predict(covariates)
        mu0_all = rf0.predict(covariates)
        cate = mu1_all - mu0_all

        # Feature importances (average across both forests)
        fi = 0.5 * (rf1.feature_importances_ + rf0.feature_importances_)
    else:
        X_t = covariates[treated_mask]
        y_t = outcome[treated_mask]
        X_c = covariates[control_mask]
        y_c = outcome[control_mask]

        rf1 = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            random_state=seed,
        )
        rf0 = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            random_state=seed + 1,
        )

        rf1.fit(X_t, y_t)
        rf0.fit(X_c, y_c)

        mu1_all = rf1.predict(covariates)
        mu0_all = rf0.predict(covariates)
        cate = mu1_all - mu0_all
        fi = 0.5 * (rf1.feature_importances_ + rf0.feature_importances_)

    ate = float(np.mean(cate))
    ate_se = float(np.std(cate, ddof=1) / np.sqrt(n))

    z_crit = stats.norm.ppf(0.975)

    return CausalForestResult(
        ate=ate,
        cate=cate,
        ate_se=ate_se,
        ci_lower=ate - z_crit * ate_se,
        ci_upper=ate + z_crit * ate_se,
        feature_importances=fi,
        details={
            "n_estimators": n_estimators,
            "honest": honest,
            "n_treated": int(treated_mask.sum()),
            "n_control": int(control_mask.sum()),
        },
    )


# ---------------------------------------------------------------------------
# Mediation analysis (Baron-Kenny)
# ---------------------------------------------------------------------------


def mediation_analysis(
    outcome: np.ndarray,
    treatment: np.ndarray,
    mediator: np.ndarray,
    covariates: np.ndarray | None = None,
) -> MediationResult:
    """Baron-Kenny mediation analysis with Sobel test.

    Decomposes the total effect of treatment on outcome into a direct
    effect and an indirect effect through a mediator variable:

    .. math::

        \\text{Path c (total):}  \\quad Y = \\alpha_1 + c \\cdot T + \\varepsilon_1

        \\text{Path a:}          \\quad M = \\alpha_2 + a \\cdot T + \\varepsilon_2

        \\text{Paths c', b:}     \\quad Y = \\alpha_3 + c' \\cdot T + b \\cdot M + \\varepsilon_3

    The indirect effect is :math:`a \\times b` (equivalently
    :math:`c - c'`). The Sobel test provides a z-test for the
    significance of the indirect effect:

    .. math::

        z = \\frac{a \\cdot b}{\\sqrt{b^2 \\cdot SE_a^2 + a^2 \\cdot SE_b^2}}

    Financial use cases:
        - Does analyst coverage affect stock returns *through* improved
          information environment (bid-ask spread)?
        - Does ESG performance impact firm value through the channel of
          reduced cost of capital?
        - Does monetary policy affect equity prices through the
          expectations channel or the risk premium channel?
        - Does order flow affect prices through inventory or information?

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable *Y* (1D array).
    treatment : np.ndarray
        Treatment variable *T* (1D array, can be continuous).
    mediator : np.ndarray
        Mediator variable *M* (1D array).
    covariates : np.ndarray or None
        Optional control variables (n, k). Default is None.

    Returns
    -------
    MediationResult
        Total, direct, indirect effects, Sobel test, and proportion
        mediated.

    Example
    -------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n = 1000
    >>> T = rng.binomial(1, 0.5, n).astype(float)
    >>> M = 0.8 * T + rng.normal(0, 0.5, n)     # path a = 0.8
    >>> Y = 0.5 * T + 0.6 * M + rng.normal(0, 0.5, n)  # c'=0.5, b=0.6
    >>> result = mediation_analysis(Y, T, M)
    >>> abs(result.indirect_effect - 0.48) < 0.2  # a*b ~ 0.48
    True
    >>> result.sobel_p < 0.05  # significant mediation
    True

    References
    ----------
    - Baron, R. M. & Kenny, D. A. (1986). "The Moderator-Mediator
      Variable Distinction."
    - Sobel, M. E. (1982). "Asymptotic Confidence Intervals for
      Indirect Effects in Structural Equation Models."
    - MacKinnon, D. P. (2008). "Introduction to Statistical Mediation
      Analysis."
    """
    outcome = np.asarray(outcome, dtype=float).ravel()
    treatment = np.asarray(treatment, dtype=float).ravel()
    mediator = np.asarray(mediator, dtype=float).ravel()

    n = len(outcome)

    # Build design matrices
    if covariates is not None:
        covariates = np.asarray(covariates, dtype=float)
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)
        W = np.column_stack([np.ones(n), covariates])
    else:
        W = np.ones((n, 1))

    # Path c (total effect): Y ~ T + W
    X_c = np.column_stack([W, treatment])
    beta_c = _ols_coefficients(X_c, outcome)
    total_effect = float(beta_c[-1])

    # Path a: M ~ T + W
    X_a = np.column_stack([W, treatment])
    beta_a = _ols_coefficients(X_a, mediator)
    path_a = float(beta_a[-1])
    resid_a = mediator - X_a @ beta_a
    sigma2_a = float(np.sum(resid_a**2) / (n - X_a.shape[1]))
    XaXa_inv = np.linalg.inv(X_a.T @ X_a)
    se_a = float(np.sqrt(sigma2_a * XaXa_inv[-1, -1]))

    # Paths c' and b: Y ~ T + M + W
    X_cb = np.column_stack([W, treatment, mediator])
    beta_cb = _ols_coefficients(X_cb, outcome)
    direct_effect = float(beta_cb[-2])  # c' (treatment coefficient)
    path_b = float(beta_cb[-1])  # b (mediator coefficient)
    resid_cb = outcome - X_cb @ beta_cb
    sigma2_cb = float(np.sum(resid_cb**2) / (n - X_cb.shape[1]))
    XcbXcb_inv = np.linalg.inv(X_cb.T @ X_cb)
    se_b = float(np.sqrt(sigma2_cb * XcbXcb_inv[-1, -1]))

    # Indirect effect
    indirect_effect = path_a * path_b

    # Sobel test
    sobel_se = np.sqrt(path_b**2 * se_a**2 + path_a**2 * se_b**2)
    if sobel_se > 0:
        sobel_stat = float(indirect_effect / sobel_se)
    else:
        sobel_stat = 0.0
    sobel_p = float(2.0 * (1.0 - stats.norm.cdf(abs(sobel_stat))))

    # Proportion mediated
    if abs(total_effect) > 1e-12:
        prop_med = indirect_effect / total_effect
        prop_med = float(np.clip(prop_med, 0.0, 1.0))
    else:
        prop_med = 0.0

    return MediationResult(
        total_effect=total_effect,
        direct_effect=direct_effect,
        indirect_effect=indirect_effect,
        sobel_stat=sobel_stat,
        sobel_p=sobel_p,
        proportion_mediated=prop_med,
        path_a=path_a,
        path_a_se=se_a,
        path_b=path_b,
        path_b_se=se_b,
        details={
            "total_effect_decomposition": {
                "total": total_effect,
                "direct": direct_effect,
                "indirect": indirect_effect,
                "check_c_minus_cprime": total_effect - direct_effect,
            }
        },
    )


# ---------------------------------------------------------------------------
# Robust regression discontinuity
# ---------------------------------------------------------------------------


def _ik_bandwidth(
    running_var: np.ndarray,
    outcome: np.ndarray,
    cutoff: float,
) -> float:
    """Imbens-Kalyanaraman (2012) optimal bandwidth selector for RDD.

    Implements a simplified version of the IK procedure that balances
    bias and variance in local linear regression at the cutoff.

    Parameters
    ----------
    running_var : np.ndarray
        Running variable (1D).
    outcome : np.ndarray
        Outcome variable (1D).
    cutoff : float
        RD cutoff value.

    Returns
    -------
    float
        Optimal bandwidth.
    """
    n = len(running_var)
    r = running_var - cutoff
    h_pilot = 1.84 * np.std(r) * n ** (-1.0 / 5.0)

    left = (r < 0) & (r >= -h_pilot)
    right = (r >= 0) & (r <= h_pilot)

    if left.sum() < 5 or right.sum() < 5:
        # Fallback to Silverman
        return 1.06 * np.std(running_var) * n ** (-0.2)

    # Estimate second derivative on each side using a cubic fit
    def _curvature(mask: np.ndarray) -> float:
        r_sub = r[mask]
        y_sub = outcome[mask]
        X = np.column_stack([np.ones(len(r_sub)), r_sub, r_sub**2, r_sub**3])
        beta = _ols_coefficients(X, y_sub)
        return abs(float(2.0 * beta[2]))

    m2_left = _curvature(left)
    m2_right = _curvature(right)

    # Regularization term
    reg_const = max(m2_left + m2_right, 1e-6)

    # Residual variance
    def _sigma2(mask: np.ndarray) -> float:
        r_sub = r[mask]
        y_sub = outcome[mask]
        X = np.column_stack([np.ones(len(r_sub)), r_sub])
        beta = _ols_coefficients(X, y_sub)
        resid = y_sub - X @ beta
        return float(np.var(resid, ddof=2))

    s2_left = _sigma2(left)
    s2_right = _sigma2(right)

    # IK formula (simplified)
    n_eff = left.sum() + right.sum()
    numerator = s2_left + s2_right
    h_opt = (numerator / (reg_const + 1e-10)) ** (1.0 / 5.0) * n_eff ** (-1.0 / 5.0)

    # Clamp to reasonable range
    r_range = np.ptp(running_var)
    h_opt = float(np.clip(h_opt, r_range * 0.01, r_range * 0.5))

    return h_opt


def _mccrary_test(
    running_var: np.ndarray,
    cutoff: float,
    n_bins: int = 50,
) -> tuple[float, float]:
    """McCrary (2008) density test for manipulation of the running variable.

    Tests whether there is a discontinuity in the density of the running
    variable at the cutoff. A significant discontinuity suggests units
    may have manipulated their running variable to receive (or avoid)
    treatment.

    Parameters
    ----------
    running_var : np.ndarray
        Running variable (1D).
    cutoff : float
        RD cutoff value.
    n_bins : int
        Number of bins for the histogram. Default is 50.

    Returns
    -------
    tuple[float, float]
        (test_statistic, p_value).
    """
    r = running_var - cutoff
    left = r[r < 0]
    right = r[r >= 0]

    # Bin widths
    r_range = np.ptp(r)
    bin_width = r_range / n_bins

    # Create bins
    bins = np.arange(r.min(), r.max() + bin_width, bin_width)
    counts, edges = np.histogram(r, bins=bins)

    # Find bins just below and above cutoff
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    left_bins = bin_centers < 0
    right_bins = bin_centers >= 0

    if left_bins.sum() < 3 or right_bins.sum() < 3:
        return 0.0, 1.0

    # Normalize counts to density
    density = counts / (len(r) * bin_width)

    # Fit local linear on each side
    def _fit_side(mask: np.ndarray) -> tuple[float, float]:
        bc = bin_centers[mask]
        d = density[mask]
        X = np.column_stack([np.ones(len(bc)), bc])
        beta = _ols_coefficients(X, d)
        pred_at_0 = float(beta[0])
        resid = d - X @ beta
        se = float(np.std(resid, ddof=2) / np.sqrt(len(bc)))
        return pred_at_0, max(se, 1e-12)

    f_left, se_left = _fit_side(left_bins)
    f_right, se_right = _fit_side(right_bins)

    # Test statistic
    log_diff = np.log(max(f_right, 1e-12)) - np.log(max(f_left, 1e-12))
    se_diff = np.sqrt(
        (se_right / max(f_right, 1e-12)) ** 2 + (se_left / max(f_left, 1e-12)) ** 2
    )

    if se_diff > 0:
        t_stat = float(log_diff / se_diff)
    else:
        t_stat = 0.0

    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))
    return t_stat, p_val


def regression_discontinuity_robust(
    outcome: np.ndarray,
    running_var: np.ndarray,
    cutoff: float = 0.0,
    bandwidth: float | None = None,
    poly_order: int = 1,
    fuzzy_treatment: np.ndarray | None = None,
    kernel: str = "triangular",
    run_mccrary: bool = True,
) -> RDRobustResult:
    """Robust regression discontinuity design with optimal bandwidth.

    Enhanced RDD implementation with:

    1. **IK bandwidth selection**: Imbens-Kalyanaraman (2012) optimal
       bandwidth that balances bias and variance.

    2. **Local polynomial regression**: Supports linear (order 1) and
       quadratic (order 2) local fits with kernel weighting.

    3. **McCrary density test**: Tests for manipulation of the running
       variable at the cutoff.

    4. **Sharp vs. fuzzy RD**: Sharp RD uses the cutoff directly. Fuzzy
       RD uses a two-stage approach when treatment is probabilistic at
       the cutoff (analogous to IV).

    .. math::

        \\text{Sharp RD:} \\quad \\hat{\\tau} = \\lim_{x \\downarrow c} E[Y|X=x]
        - \\lim_{x \\uparrow c} E[Y|X=x]

        \\text{Fuzzy RD:} \\quad \\hat{\\tau} = \\frac{\\lim_{x \\downarrow c} E[Y|X=x]
        - \\lim_{x \\uparrow c} E[Y|X=x]}{\\lim_{x \\downarrow c} E[D|X=x]
        - \\lim_{x \\uparrow c} E[D|X=x]}

    Financial use cases:
        - Effect of index inclusion on stock liquidity (market-cap
          threshold).
        - Impact of credit rating changes at rating boundaries.
        - Effect of SEC disclosure requirements at firm-size thresholds.
        - Impact of margin requirements at leverage cutoffs.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable (1D array).
    running_var : np.ndarray
        Running variable that determines treatment (1D array).
    cutoff : float
        Cutoff value. Default is 0.0.
    bandwidth : float or None
        Bandwidth for local regression. If None, uses the IK optimal
        bandwidth. Default is None.
    poly_order : int
        Order of the local polynomial (1 = linear, 2 = quadratic).
        Default is 1.
    fuzzy_treatment : np.ndarray or None
        Actual treatment indicator for fuzzy RD (1D array of 0s and 1s).
        If None, uses sharp RD. Default is None.
    kernel : str
        Kernel for weighting observations: ``'triangular'``,
        ``'uniform'``, or ``'epanechnikov'``. Default is
        ``'triangular'``.
    run_mccrary : bool
        Whether to run the McCrary manipulation test. Default is True.

    Returns
    -------
    RDRobustResult
        RD estimate with robust standard errors, bandwidth information,
        and diagnostic tests.

    Raises
    ------
    ValueError
        If insufficient observations or invalid parameters.

    Example
    -------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n = 2000
    >>> x = rng.uniform(-2, 2, n)
    >>> y = 0.5 * x + 1.5 * (x >= 0) + rng.normal(0, 0.3, n)
    >>> result = regression_discontinuity_robust(y, x, cutoff=0.0)
    >>> abs(result.ate - 1.5) < 0.5
    True
    >>> result.bandwidth_method
    'imbens_kalyanaraman'

    References
    ----------
    - Imbens, G. W. & Kalyanaraman, K. (2012). "Optimal Bandwidth
      Choice for the Regression Discontinuity Estimator."
    - McCrary, J. (2008). "Manipulation of the Running Variable in the
      Regression Discontinuity Design."
    - Cattaneo, M. D. et al. (2020). "A Practical Introduction to
      Regression Discontinuity Designs."
    """
    outcome = np.asarray(outcome, dtype=float).ravel()
    running_var = np.asarray(running_var, dtype=float).ravel()
    n = len(outcome)

    rd_type = "fuzzy" if fuzzy_treatment is not None else "sharp"

    # Bandwidth selection
    if bandwidth is None:
        bandwidth = _ik_bandwidth(running_var, outcome, cutoff)
        bw_method = "imbens_kalyanaraman"
    else:
        bw_method = "user_specified"

    # Select observations within bandwidth
    r = running_var - cutoff
    left_mask = (r >= -bandwidth) & (r < 0)
    right_mask = (r >= 0) & (r <= bandwidth)
    in_bw = left_mask | right_mask

    n_left = int(left_mask.sum())
    n_right = int(right_mask.sum())

    if n_left < poly_order + 2 or n_right < poly_order + 2:
        raise ValueError(
            f"Insufficient observations within bandwidth "
            f"(n_left={n_left}, n_right={n_right}). "
            f"Try increasing the bandwidth (current={bandwidth:.4f})."
        )

    # Kernel weights
    r_bw = r[in_bw]
    y_bw = outcome[in_bw]

    if kernel == "triangular":
        w = 1.0 - np.abs(r_bw) / bandwidth
    elif kernel == "epanechnikov":
        w = 0.75 * (1.0 - (r_bw / bandwidth) ** 2)
    else:  # uniform
        w = np.ones(len(r_bw))
    w = np.maximum(w, 0.0)

    # Indicator for right of cutoff
    d = (r_bw >= 0).astype(float)

    # Build local polynomial design matrix
    # Y = a0 + tau * D + b1 * r + b2 * D * r + [b3 * r^2 + b4 * D * r^2] + eps
    X_cols = [np.ones(len(r_bw)), d, r_bw, d * r_bw]
    if poly_order >= 2:
        X_cols.extend([r_bw**2, d * r_bw**2])
    X = np.column_stack(X_cols)

    # Weighted least squares
    W_diag = np.diag(np.sqrt(w))
    Xw = W_diag @ X
    yw = W_diag @ y_bw

    beta = _ols_coefficients(Xw, yw)
    ate_sharp = float(beta[1])  # coefficient on D

    # For fuzzy RD, also estimate the first stage
    if fuzzy_treatment is not None:
        fuzzy_treatment = np.asarray(fuzzy_treatment, dtype=float).ravel()
        d_actual = fuzzy_treatment[in_bw]

        # First stage: D_actual ~ same polynomial spec
        beta_fs = _ols_coefficients(Xw, W_diag @ d_actual)
        first_stage_jump = float(beta_fs[1])

        if abs(first_stage_jump) < 1e-10:
            raise ValueError(
                "First stage jump is near zero. The instrument (cutoff) "
                "does not predict treatment. Fuzzy RD is not applicable."
            )
        ate = ate_sharp / first_stage_jump
    else:
        ate = ate_sharp

    # Robust standard error (HC1)
    resid = y_bw - X @ beta
    w_resid = w * resid**2
    bread = np.linalg.inv(X.T @ np.diag(w) @ X)
    meat = X.T @ np.diag(w_resid) @ X
    sandwich = bread @ meat @ bread

    se = float(np.sqrt(sandwich[1, 1]))
    if fuzzy_treatment is not None and abs(first_stage_jump) > 1e-10:
        se = se / abs(first_stage_jump)

    z_crit = stats.norm.ppf(0.975)

    # McCrary test
    mc_stat = None
    mc_p = None
    if run_mccrary:
        mc_stat, mc_p = _mccrary_test(running_var, cutoff)

    return RDRobustResult(
        ate=ate,
        se=se,
        ci_lower=ate - z_crit * se,
        ci_upper=ate + z_crit * se,
        bandwidth=bandwidth,
        bandwidth_method=bw_method,
        n_left=n_left,
        n_right=n_right,
        poly_order=poly_order,
        mccrary_stat=mc_stat,
        mccrary_p=mc_p,
        rd_type=rd_type,
        details={
            "kernel": kernel,
            "coefficients": beta.tolist(),
        },
    )


# ---------------------------------------------------------------------------
# Bounds analysis (Manski / Lee)
# ---------------------------------------------------------------------------


def bounds_analysis(
    outcome: np.ndarray,
    treatment: np.ndarray,
    selection: np.ndarray | None = None,
    outcome_bounds: tuple[float, float] | None = None,
    method: str = "manski",
) -> BoundsResult:
    """Compute Manski or Lee bounds for partial identification of treatment effects.

    When standard identifying assumptions (unconfoundedness, exclusion
    restriction) are too strong, partial identification provides bounds
    on the treatment effect that hold under weaker assumptions.

    **Manski (1990) bounds** require only:
        - Known outcome support (bounded outcomes).
        - Random treatment assignment OR treatment is observed.

    The treatment effect is bounded by:

    .. math::

        E[Y|D=1] - y_{\\max} \\cdot P(D=0) - E[Y|D=0] \\cdot P(D=0)
        \\leq \\text{ATE} \\leq
        E[Y|D=1] - y_{\\min} \\cdot P(D=0) - E[Y|D=0] \\cdot P(D=0)

    **Lee (2009) bounds** handle sample selection (attrition):
        - When treatment affects whether the outcome is observed.
        - Tightens bounds by trimming the group with higher selection rate.

    Financial use cases:
        - Bounding the effect of financial literacy programs on savings
          when participation is endogenous.
        - Bounding returns to education when the outcome (wages) is only
          observed for the employed.
        - Partial identification of trading strategy effects when some
          trades are censored.

    Parameters
    ----------
    outcome : np.ndarray
        Observed outcomes (1D array). For Lee bounds, should only
        contain observed (non-missing) values.
    treatment : np.ndarray
        Binary treatment indicator (1D array of 0s and 1s).
    selection : np.ndarray or None
        Binary selection indicator (1D array). Required for Lee bounds.
        1 = outcome observed, 0 = outcome missing/attrited. Must be the
        same length as treatment. Default is None.
    outcome_bounds : tuple[float, float] or None
        (y_min, y_max) bounds on the outcome support. Required for
        Manski bounds. If None, uses the observed min and max.
    method : str
        ``'manski'`` for Manski worst-case bounds or ``'lee'`` for
        Lee trimming bounds. Default is ``'manski'``.

    Returns
    -------
    BoundsResult
        Lower and upper bounds on the treatment effect, whether the
        effect is sign-identified, and confidence intervals.

    Example
    -------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n = 1000
    >>> T = rng.binomial(1, 0.5, n).astype(float)
    >>> Y = 2.0 * T + rng.normal(0, 1, n)
    >>> result = bounds_analysis(Y, T, outcome_bounds=(-5, 10))
    >>> result.lower_bound < 2.0 < result.upper_bound
    True
    >>> result.identified  # both bounds positive -> sign identified
    True

    References
    ----------
    - Manski, C. F. (1990). "Nonparametric Bounds on Treatment Effects."
    - Lee, D. S. (2009). "Training, Wages, and Sample Selection:
      Estimating Sharp Bounds on Treatment Effects."
    - Horowitz, J. L. & Manski, C. F. (2000). "Nonparametric Analysis
      of Randomized Experiments with Missing Covariate and Outcome Data."
    """
    outcome = np.asarray(outcome, dtype=float).ravel()
    treatment = np.asarray(treatment, dtype=float).ravel()

    treated_mask = treatment == 1
    control_mask = treatment == 0

    if method == "manski":
        return _manski_bounds(
            outcome, treatment, treated_mask, control_mask, outcome_bounds
        )
    elif method == "lee":
        if selection is None:
            raise ValueError("Lee bounds require a 'selection' indicator.")
        return _lee_bounds(outcome, treatment, selection, treated_mask, control_mask)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'manski' or 'lee'.")


def _manski_bounds(
    outcome: np.ndarray,
    treatment: np.ndarray,
    treated_mask: np.ndarray,
    control_mask: np.ndarray,
    outcome_bounds: tuple[float, float] | None,
) -> BoundsResult:
    """Compute Manski worst-case bounds."""
    n = len(outcome)

    if outcome_bounds is None:
        y_min = float(np.min(outcome))
        y_max = float(np.max(outcome))
    else:
        y_min, y_max = outcome_bounds

    # E[Y|D=1] and E[Y|D=0]
    e_y1 = float(np.mean(outcome[treated_mask]))
    e_y0 = float(np.mean(outcome[control_mask]))
    p_treat = float(np.mean(treatment))
    p_control = 1.0 - p_treat

    # Manski bounds:
    # Lower: E[Y(1)] >= E[Y|D=1]*P(D=1) + y_min*P(D=0)
    #         E[Y(0)] <= E[Y|D=0]*P(D=0) + y_max*P(D=1)
    # ATE_lower = lower_Y1 - upper_Y0
    lower_y1 = e_y1 * p_treat + y_min * p_control
    upper_y0 = e_y0 * p_control + y_max * p_treat
    lower_bound = lower_y1 - upper_y0

    upper_y1 = e_y1 * p_treat + y_max * p_control
    lower_y0 = e_y0 * p_control + y_min * p_treat
    upper_bound = upper_y1 - lower_y0

    # Sign identification
    identified = (lower_bound > 0 and upper_bound > 0) or (
        lower_bound < 0 and upper_bound < 0
    )

    # Bootstrap-style CI using influence functions
    n_t = treated_mask.sum()
    n_c = control_mask.sum()
    se_e_y1 = float(np.std(outcome[treated_mask], ddof=1) / np.sqrt(n_t))
    se_e_y0 = float(np.std(outcome[control_mask], ddof=1) / np.sqrt(n_c))

    # Conservative: SE of bounds is at least SE of the means
    se_bound = np.sqrt(se_e_y1**2 + se_e_y0**2)
    z_crit = stats.norm.ppf(0.975)

    return BoundsResult(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        bound_type="manski",
        identified=identified,
        ci_lower=lower_bound - z_crit * se_bound,
        ci_upper=upper_bound + z_crit * se_bound,
        details={
            "e_y_treated": e_y1,
            "e_y_control": e_y0,
            "p_treated": p_treat,
            "outcome_bounds": (y_min, y_max),
        },
    )


def _lee_bounds(
    outcome: np.ndarray,
    treatment: np.ndarray,
    selection: np.ndarray,
    treated_mask: np.ndarray,
    control_mask: np.ndarray,
) -> BoundsResult:
    """Compute Lee (2009) trimming bounds for sample selection."""
    selection = np.asarray(selection, dtype=float).ravel()

    # Selection rates
    s1 = float(np.mean(selection[treated_mask]))  # P(S=1|D=1)
    s0 = float(np.mean(selection[control_mask]))  # P(S=1|D=0)

    if abs(s1 - s0) < 1e-12:
        # No differential selection — standard comparison
        y_t = outcome[treated_mask & (selection == 1)]
        y_c = outcome[control_mask & (selection == 1)]
        diff = float(np.mean(y_t) - np.mean(y_c))
        se = float(
            np.sqrt(np.var(y_t, ddof=1) / len(y_t) + np.var(y_c, ddof=1) / len(y_c))
        )
        z_crit = stats.norm.ppf(0.975)
        return BoundsResult(
            lower_bound=diff,
            upper_bound=diff,
            bound_type="lee",
            identified=True,
            ci_lower=diff - z_crit * se,
            ci_upper=diff + z_crit * se,
            details={"selection_rate_treated": s1, "selection_rate_control": s0},
        )

    # Determine which group to trim (the one with higher selection)
    if s1 > s0:
        # Trim treated group
        trim_frac = 1.0 - s0 / s1
        y_trim = outcome[treated_mask & (selection == 1)]
        y_other = outcome[control_mask & (selection == 1)]
    else:
        # Trim control group
        trim_frac = 1.0 - s1 / s0
        y_trim = outcome[control_mask & (selection == 1)]
        y_other = outcome[treated_mask & (selection == 1)]

    # Sort the group to trim
    y_sorted = np.sort(y_trim)
    n_trim = len(y_sorted)
    n_remove = int(np.ceil(trim_frac * n_trim))

    if n_remove >= n_trim:
        raise ValueError(
            "Lee bounds: trimming fraction too large. Not enough "
            "overlap between selection rates."
        )

    # Lower bound: trim from above (remove highest outcomes)
    y_lower = y_sorted[: n_trim - n_remove]
    # Upper bound: trim from below (remove lowest outcomes)
    y_upper = y_sorted[n_remove:]

    mean_other = float(np.mean(y_other))

    if s1 > s0:
        lower_bound = float(np.mean(y_lower)) - mean_other
        upper_bound = float(np.mean(y_upper)) - mean_other
    else:
        lower_bound = mean_other - float(np.mean(y_upper))
        upper_bound = mean_other - float(np.mean(y_lower))

    # Ensure proper ordering
    if lower_bound > upper_bound:
        lower_bound, upper_bound = upper_bound, lower_bound

    identified = (lower_bound > 0 and upper_bound > 0) or (
        lower_bound < 0 and upper_bound < 0
    )

    # SE via bootstrap approximation
    se_trim = float(np.std(y_trim, ddof=1) / np.sqrt(len(y_trim)))
    se_other = float(np.std(y_other, ddof=1) / np.sqrt(len(y_other)))
    se_bound = np.sqrt(se_trim**2 + se_other**2)
    z_crit = stats.norm.ppf(0.975)

    return BoundsResult(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        bound_type="lee",
        identified=identified,
        ci_lower=lower_bound - z_crit * se_bound,
        ci_upper=upper_bound + z_crit * se_bound,
        details={
            "selection_rate_treated": s1,
            "selection_rate_control": s0,
            "trim_fraction": trim_frac,
            "n_trimmed": n_remove,
        },
    )
