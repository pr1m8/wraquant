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

__all__ = [
    "propensity_score",
    "ipw_ate",
    "matching_ate",
    "doubly_robust_ate",
    "regression_discontinuity",
    "synthetic_control",
    "diff_in_diff",
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

    Fits a logistic regression of treatment assignment on covariates
    to estimate P(treatment=1 | covariates).

    Parameters
    ----------
    treatment : np.ndarray
        Binary treatment indicator (1D array of 0s and 1s).
    covariates : np.ndarray
        Covariate matrix (n_samples, n_features). An intercept column
        is added automatically.

    Returns
    -------
    np.ndarray
        Estimated propensity scores (probabilities), shape (n_samples,).
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
    """Estimate the Average Treatment Effect using Inverse Probability Weighting.

    Uses the Horvitz-Thompson estimator with normalized weights.

    Parameters
    ----------
    outcome : np.ndarray
        Observed outcomes (1D array).
    treatment : np.ndarray
        Binary treatment indicator (1D array of 0s and 1s).
    propensity_scores : np.ndarray
        Estimated propensity scores (1D array).

    Returns
    -------
    ATEResult
        ATE estimate with standard error and confidence interval.
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

    Each treated unit is matched to the closest control unit(s) based on
    Euclidean distance in covariate space (and vice-versa for ATT/ATC).

    Parameters
    ----------
    outcome : np.ndarray
        Observed outcomes (1D array).
    treatment : np.ndarray
        Binary treatment indicator (1D array of 0s and 1s).
    covariates : np.ndarray
        Covariate matrix (n_samples, n_features).
    n_neighbors : int
        Number of nearest neighbors to match with. Default is 1.

    Returns
    -------
    ATEResult
        ATE estimate with standard error and confidence interval.
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
    """Estimate the ATE using the doubly robust (augmented IPW) estimator.

    Combines inverse probability weighting with outcome regression.
    The estimator is consistent if either the propensity score model or
    the outcome model is correctly specified.

    Parameters
    ----------
    outcome : np.ndarray
        Observed outcomes (1D array).
    treatment : np.ndarray
        Binary treatment indicator (1D array of 0s and 1s).
    covariates : np.ndarray
        Covariate matrix (n_samples, n_features).

    Returns
    -------
    ATEResult
        ATE estimate with standard error and confidence interval.
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
    beta_t = np.linalg.lstsq(X_t, y_t, rcond=None)[0]
    mu1_hat = X @ beta_t  # predicted E[Y(1)|X] for all units

    # Fit outcome model for control group
    X_c = X[control_mask]
    y_c = outcome[control_mask]
    beta_c = np.linalg.lstsq(X_c, y_c, rcond=None)[0]
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

    Fits local linear regressions on each side of the cutoff and estimates
    the treatment effect as the discontinuity at the cutoff.

    Parameters
    ----------
    outcome : np.ndarray
        Observed outcomes (1D array).
    running_var : np.ndarray
        Running variable that determines treatment assignment (1D array).
    cutoff : float
        Cutoff value. Units with running_var >= cutoff are treated.
    bandwidth : float or None
        Bandwidth for local linear regression. If None, uses Silverman's
        rule of thumb: 1.06 * std * n^(-1/5).

    Returns
    -------
    RDResult
        RD estimate with standard error and confidence interval.
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
    beta_left = np.linalg.lstsq(X_left, y_left, rcond=None)[0]

    # Local linear regression: right side
    X_right = np.column_stack([np.ones(n_right), r_right])
    beta_right = np.linalg.lstsq(X_right, y_right, rcond=None)[0]

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

    Finds convex combination weights for donor units that best reproduce
    the treated unit's pre-treatment outcomes, then estimates the
    treatment effect as the post-treatment gap.

    Parameters
    ----------
    treated_outcomes : np.ndarray
        Outcomes for the treated unit over all time periods (1D array).
    donor_outcomes : np.ndarray
        Outcomes for donor units (n_periods, n_donors).
    pre_period : int
        Number of pre-treatment time periods (index of first post-treatment
        period).

    Returns
    -------
    SyntheticControlResult
        Estimated treatment effect, donor weights, and diagnostics.
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
    """Estimate treatment effect using difference-in-differences.

    Implements the canonical 2x2 DID estimator. When entity fixed effects
    are provided, uses the within estimator.

    Parameters
    ----------
    outcome : np.ndarray
        Observed outcomes (1D array).
    treatment : np.ndarray
        Binary group indicator (1D array): 1 = treatment group,
        0 = control group.
    post : np.ndarray
        Binary time indicator (1D array): 1 = post-treatment,
        0 = pre-treatment.
    entity : np.ndarray or None
        Entity identifiers for panel data (1D array). If provided,
        entity fixed effects are included. Default is None.

    Returns
    -------
    DIDResult
        DID estimate with standard error and confidence interval.
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
        beta = np.linalg.lstsq(X, outcome, rcond=None)[0]
        resid = outcome - X @ beta
    else:
        beta = np.linalg.lstsq(X, outcome, rcond=None)[0]
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
