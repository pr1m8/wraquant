"""Advanced dependence measures for financial data.

Beyond linear correlation, financial risk management requires measures
that capture tail dependence, nonlinear relationships, and the full
dependence structure between variables.  This module provides tools for
tail dependence estimation, copula-based dependence modelling, rank
correlation matrices, and concordance analysis.

Key concepts:
    - **Tail dependence** quantifies the probability that two variables
      jointly experience extreme values.  This is critical for portfolio
      risk during crises, where correlations spike.
    - **Copulas** separate the marginal distributions from the dependence
      structure, allowing flexible modelling of joint distributions.
    - **Rank correlation** is robust to monotonic transformations and
      outliers, making it suitable for heavy-tailed financial data.
    - **Concordance index** measures the agreement between two orderings,
      useful for model validation and survival analysis.

References:
    - Joe, H. (2014). *Dependence Modeling with Copulas*.
    - McNeil, A. J., Frey, R. & Embrechts, P. (2015). *Quantitative Risk
      Management: Concepts, Techniques and Tools*.
    - Harrell, F. E. et al. (1996). "Multivariable prognostic models."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


__all__ = [
    "tail_dependence_coefficient",
    "copula_selection",
    "rank_correlation_matrix",
    "concordance_index",
]


# ---------------------------------------------------------------------------
# Tail dependence coefficient
# ---------------------------------------------------------------------------


def tail_dependence_coefficient(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    threshold: float = 0.05,
) -> dict:
    """Estimate upper and lower tail dependence coefficients from empirical data.

    Tail dependence measures the probability that one variable is
    extremely large (small) given that the other is also extremely large
    (small).  This is crucial for understanding joint tail risk in
    portfolios --- standard correlation says nothing about co-movement
    in the tails.

    When to use:
        - To quantify the risk of joint extreme losses in a portfolio.
        - To assess whether diversification benefits disappear during
          market crashes (asymmetric tail dependence).
        - To select the appropriate copula family: Gaussian copulas have
          zero tail dependence, while Clayton (lower) and Gumbel (upper)
          copulas can model it.
        - To compare the tail behaviour of different asset pairs.

    Mathematical formulation:
        The upper tail dependence coefficient is:

        .. math::

            \\lambda_U = \\lim_{q \\to 1} P(Y > F_Y^{-1}(q) \\mid X > F_X^{-1}(q))

        The lower tail dependence coefficient is:

        .. math::

            \\lambda_L = \\lim_{q \\to 0} P(Y \\le F_Y^{-1}(q) \\mid X \\le F_X^{-1}(q))

        We estimate these empirically using the rank-transformed data
        (pseudo-observations) and counting joint exceedances.

    How to interpret:
        - ``lambda = 0``: no tail dependence (e.g., Gaussian copula).
          Diversification holds in the tails.
        - ``lambda > 0``: positive tail dependence.  Extreme events tend
          to happen together.
        - ``lambda_L > lambda_U``: lower tail dependence is stronger
          than upper (common in equity markets --- crashes are more
          contagious than rallies).
        - Values are bounded in [0, 1].

    Parameters:
        x: First variable (1-D array or Series).
        y: Second variable (1-D array or Series, same length).
        threshold: Quantile threshold for defining "extreme" (default
            0.05, meaning the top/bottom 5%).

    Returns:
        Dictionary with:
        - ``upper_lambda``: estimated upper tail dependence coefficient.
        - ``lower_lambda``: estimated lower tail dependence coefficient.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> # Gaussian data has zero tail dependence
        >>> x = rng.normal(0, 1, 5000)
        >>> y = 0.7 * x + rng.normal(0, 0.71, 5000)
        >>> result = tail_dependence_coefficient(x, y)
        >>> 0 <= result["upper_lambda"] <= 1
        True
        >>> 0 <= result["lower_lambda"] <= 1
        True

    References:
        - Joe, H. (2014). *Dependence Modeling with Copulas*, Ch. 2.
        - McNeil et al. (2015). *Quantitative Risk Management*, Ch. 7.

    See Also:
        copula_selection: Fit copulas that model tail dependence.
        rank_correlation_matrix: Rank-based dependence matrix.
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()

    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    n = len(x_arr)

    if n < 20:
        return {"upper_lambda": 0.0, "lower_lambda": 0.0}

    # Convert to pseudo-observations (uniform margins via ranks)
    u = sp_stats.rankdata(x_arr) / (n + 1)
    v = sp_stats.rankdata(y_arr) / (n + 1)

    # Upper tail: P(V > 1-t | U > 1-t)
    upper_threshold = 1.0 - threshold
    upper_joint = np.sum((u > upper_threshold) & (v > upper_threshold))
    upper_marginal = np.sum(u > upper_threshold)
    upper_lambda = float(upper_joint / upper_marginal) if upper_marginal > 0 else 0.0

    # Lower tail: P(V <= t | U <= t)
    lower_joint = np.sum((u <= threshold) & (v <= threshold))
    lower_marginal = np.sum(u <= threshold)
    lower_lambda = float(lower_joint / lower_marginal) if lower_marginal > 0 else 0.0

    return {
        "upper_lambda": float(min(max(upper_lambda, 0.0), 1.0)),
        "lower_lambda": float(min(max(lower_lambda, 0.0), 1.0)),
    }


# ---------------------------------------------------------------------------
# Copula selection
# ---------------------------------------------------------------------------


def copula_selection(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
) -> dict:
    """Fit multiple copula families and select the best by AIC.

    Copulas separate the marginal distributions from the dependence
    structure, allowing flexible modelling of how two variables move
    together.  This function fits several parametric copula families
    and ranks them by AIC to identify the best model for the data.

    When to use:
        - To model joint distributions for portfolio risk (e.g., joint
          simulation of asset returns for VaR/CVaR).
        - To capture tail dependence or asymmetric dependence that
          Gaussian models miss.
        - To select the appropriate copula for bivariate analysis
          before using it in a larger risk framework.

    Copula families fitted:
        - **Gaussian**: symmetric, zero tail dependence.
        - **Student-t** (approximated): symmetric, positive tail
          dependence in both tails.
        - **Clayton**: lower tail dependence (joint crashes).
        - **Gumbel**: upper tail dependence (joint rallies).

    Mathematical formulation:
        A copula ``C(u, v)`` is a joint CDF on [0,1]^2 whose marginals
        are uniform.  By Sklar's theorem, any joint distribution can be
        written as:

        .. math::

            F(x, y) = C(F_X(x), F_Y(y))

        Each copula family has a parameter ``theta`` that controls the
        strength and shape of dependence.  We estimate ``theta`` by
        maximum likelihood on the pseudo-observations.

    Parameters:
        x: First variable (1-D array or Series).
        y: Second variable (1-D array or Series, same length).

    Returns:
        Dictionary with:
        - ``best_copula``: name of the best-fitting copula.
        - ``all_fits``: DataFrame with columns ``copula``, ``parameter``,
          ``log_likelihood``, ``aic``, sorted by AIC ascending.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> x = rng.normal(0, 1, 500)
        >>> y = 0.7 * x + rng.normal(0, 0.71, 500)
        >>> result = copula_selection(x, y)
        >>> isinstance(result["all_fits"], pd.DataFrame)
        True
        >>> result["best_copula"] in ["gaussian", "student_t", "clayton", "gumbel"]
        True

    References:
        - Joe, H. (2014). *Dependence Modeling with Copulas*.
        - Nelsen, R. B. (2006). *An Introduction to Copulas*.

    See Also:
        tail_dependence_coefficient: Empirical tail dependence.
        rank_correlation_matrix: Rank-based dependence matrix.
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()

    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    n = len(x_arr)

    # Convert to pseudo-observations
    u = sp_stats.rankdata(x_arr) / (n + 1)
    v = sp_stats.rankdata(y_arr) / (n + 1)

    # Kendall's tau for parameter estimation via shared module
    from wraquant.stats.correlation import kendall_tau as _kendall_tau

    _tau_result = _kendall_tau(x_arr, y_arr)
    tau = _tau_result["tau"]
    tau = float(np.clip(tau, -0.99, 0.99))

    rows: list[dict] = []

    # --- Gaussian copula ---
    try:
        rho = float(np.sin(tau * np.pi / 2))  # inversion formula
        rho = np.clip(rho, -0.999, 0.999)
        # Log-likelihood of Gaussian copula
        z_u = sp_stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
        z_v = sp_stats.norm.ppf(np.clip(v, 1e-6, 1 - 1e-6))
        ll = float(np.sum(
            -0.5 * np.log(1 - rho ** 2)
            - (rho ** 2 * (z_u ** 2 + z_v ** 2) - 2 * rho * z_u * z_v)
            / (2 * (1 - rho ** 2))
        ))
        aic = float(2 * 1 - 2 * ll)
        rows.append({
            "copula": "gaussian",
            "parameter": float(rho),
            "log_likelihood": ll,
            "aic": aic,
        })
    except Exception:  # noqa: BLE001
        pass

    # --- Student-t copula (approximation via Gaussian with tail correction) ---
    try:
        # Use Gaussian rho as parameter, estimate df from tail behavior
        rho_t = rho
        # Approximate df via method of moments on squared z-residuals
        z_res = (z_u ** 2 + z_v ** 2 - 2 * rho_t * z_u * z_v) / (1 - rho_t ** 2)
        # For t-copula, z_res / 2 follows F(2, df)
        mean_z_res = float(np.mean(z_res))
        if mean_z_res > 2:
            df_est = max(2 * mean_z_res / (mean_z_res - 2), 2.5)
        else:
            df_est = 30.0
        df_est = min(df_est, 100.0)

        # t-copula log-likelihood approximation
        from scipy.special import gammaln
        half_df = df_est / 2
        t_u = sp_stats.t.ppf(np.clip(u, 1e-6, 1 - 1e-6), df=df_est)
        t_v = sp_stats.t.ppf(np.clip(v, 1e-6, 1 - 1e-6), df=df_est)

        ll_t = float(np.sum(
            gammaln((df_est + 2) / 2)
            - gammaln(df_est / 2)
            - np.log(np.pi * df_est)
            - 0.5 * np.log(1 - rho_t ** 2)
            - ((df_est + 2) / 2) * np.log(
                1 + (t_u ** 2 + t_v ** 2 - 2 * rho_t * t_u * t_v)
                / (df_est * (1 - rho_t ** 2))
            )
            - np.sum([
                -sp_stats.t.logpdf(t_u, df=df_est),
                -sp_stats.t.logpdf(t_v, df=df_est),
            ], axis=0)
        ))
        aic_t = float(2 * 2 - 2 * ll_t)  # 2 parameters: rho and df
        rows.append({
            "copula": "student_t",
            "parameter": float(rho_t),
            "log_likelihood": ll_t,
            "aic": aic_t,
        })
    except Exception:  # noqa: BLE001
        pass

    # --- Clayton copula (lower tail dependence) ---
    try:
        if tau > 0.01:
            theta_c = 2 * tau / (1 - tau)
            theta_c = max(theta_c, 0.01)
            # Clayton log-likelihood
            ll_c = float(np.sum(
                np.log(1 + theta_c)
                + (-1 - theta_c) * (np.log(u) + np.log(v))
                + (-2 - 1 / theta_c) * np.log(
                    u ** (-theta_c) + v ** (-theta_c) - 1
                )
            ))
            aic_c = float(2 * 1 - 2 * ll_c)
            rows.append({
                "copula": "clayton",
                "parameter": float(theta_c),
                "log_likelihood": ll_c,
                "aic": aic_c,
            })
    except Exception:  # noqa: BLE001
        pass

    # --- Gumbel copula (upper tail dependence) ---
    try:
        if tau > 0.01:
            theta_g = 1 / (1 - tau)
            theta_g = max(theta_g, 1.01)
            # Gumbel log-likelihood (simplified)
            neg_log_u = -np.log(np.clip(u, 1e-10, 1))
            neg_log_v = -np.log(np.clip(v, 1e-10, 1))
            A = (neg_log_u ** theta_g + neg_log_v ** theta_g) ** (1 / theta_g)

            ll_g = float(np.sum(
                -A
                + (theta_g - 1) * (np.log(neg_log_u) + np.log(neg_log_v))
                + np.log(A + theta_g - 1)
                - (2 - 1 / theta_g) * np.log(
                    neg_log_u ** theta_g + neg_log_v ** theta_g
                )
                - np.log(u) - np.log(v)
            ))
            aic_g = float(2 * 1 - 2 * ll_g)
            rows.append({
                "copula": "gumbel",
                "parameter": float(theta_g),
                "log_likelihood": ll_g,
                "aic": aic_g,
            })
    except Exception:  # noqa: BLE001
        pass

    if not rows:
        return {
            "best_copula": "gaussian",
            "all_fits": pd.DataFrame(
                columns=["copula", "parameter", "log_likelihood", "aic"]
            ),
        }

    df = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)

    return {
        "best_copula": str(df.iloc[0]["copula"]),
        "all_fits": df,
    }


# ---------------------------------------------------------------------------
# Spearman rank correlation matrix
# ---------------------------------------------------------------------------


def rank_correlation_matrix(
    data: pd.DataFrame,
    method: str = "spearman",
) -> pd.DataFrame:
    """Compute the Spearman rank correlation matrix.

    Spearman correlation assesses monotonic relationships by computing
    Pearson correlation on the rank-transformed data.  It is more robust
    to outliers and non-linearity than Pearson correlation, making it
    well-suited for financial data with heavy tails.

    When to use:
        - When the relationship between variables is monotonic but not
          necessarily linear.
        - When data contains outliers that would distort Pearson
          correlation.
        - For copula parameter estimation (Spearman's rho has a direct
          relationship to many copula parameters).
        - As a robustness check on Pearson correlation: if the two differ
          substantially, the relationship may be nonlinear.

    Mathematical formulation:
        .. math::

            \\rho_S(X, Y) = \\text{Pearson}(\\text{rank}(X), \\text{rank}(Y))

        Equivalently:

        .. math::

            \\rho_S = 1 - \\frac{6 \\sum d_i^2}{n(n^2 - 1)}

        where ``d_i`` is the difference in ranks.

    How to interpret:
        - Same scale as Pearson: [-1, 1].
        - ``rho_S > rho_P``: the relationship is stronger in the ranks
          than in the raw values (concave/convex relationship).
        - ``rho_S ≈ rho_P``: the relationship is approximately linear.

    Parameters:
        data: DataFrame with columns as variables and rows as
            observations.
        method: Rank correlation method -- ``"spearman"`` (default) or
            ``"kendall"``.

    Returns:
        Rank correlation matrix as a DataFrame (p x p).

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> data = pd.DataFrame(np.random.randn(100, 4), columns=list("ABCD"))
        >>> rcm = rank_correlation_matrix(data)
        >>> rcm.shape
        (4, 4)

    See Also:
        correlation_matrix: Pearson correlation matrix.
        partial_correlation: Partial correlation controlling for others.
    """
    from wraquant.stats.correlation import correlation_matrix as _correlation_matrix

    return _correlation_matrix(data, method=method)


# ---------------------------------------------------------------------------
# Concordance index (Harrell's C-index)
# ---------------------------------------------------------------------------


def concordance_index(
    predicted: pd.Series | np.ndarray,
    observed: pd.Series | np.ndarray,
) -> float:
    """Compute Harrell's concordance index (C-index).

    The C-index measures the probability that for a random pair of
    observations, the one with the higher predicted value also has the
    higher observed value.  It is a generalised rank correlation measure
    widely used in survival analysis, credit risk, and model validation.

    When to use:
        - To evaluate the discriminatory power of a predictive model
          (e.g., a credit scoring model, a default probability model,
          or an alpha signal).
        - When you care about ordinal accuracy (ranking) rather than
          calibration (magnitude).
        - As an alternative to AUC for continuous outcomes (the C-index
          generalises AUC to continuous data).

    Mathematical formulation:
        .. math::

            C = \\frac{\\text{concordant pairs}}{\\text{concordant + discordant pairs}}

        A pair ``(i, j)`` is **concordant** if the predicted and observed
        orderings agree: ``(\\hat{y}_i > \\hat{y}_j \\text{ and } y_i > y_j)``
        or ``(\\hat{y}_i < \\hat{y}_j \\text{ and } y_i < y_j)``.

    How to interpret:
        - ``C = 0.5``: random (no predictive power).
        - ``C = 1.0``: perfect concordance (perfect ranking).
        - ``C < 0.5``: worse than random (model has the sign wrong).
        - ``C > 0.7``: generally considered acceptable in finance.
        - ``C > 0.8``: strong discriminatory power.

    Parameters:
        predicted: Predicted values or scores.
        observed: Observed/actual values.

    Returns:
        Concordance index as a float in [0, 1].

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> predicted = rng.normal(0, 1, 100)
        >>> observed = predicted + rng.normal(0, 0.5, 100)
        >>> c = concordance_index(predicted, observed)
        >>> c > 0.7  # good concordance
        True

    References:
        Harrell, F. E., Lee, K. L. & Mark, D. B. (1996). "Multivariable
        prognostic models: issues in developing models, evaluating
        assumptions and adequacy, and measuring and reducing errors."
        *Statistics in Medicine*, 15(4), 361-387.

    See Also:
        kendall_tau: Rank correlation (related to C-index).
    """
    pred = np.asarray(predicted, dtype=float).ravel()
    obs = np.asarray(observed, dtype=float).ravel()

    mask = ~(np.isnan(pred) | np.isnan(obs))
    pred = pred[mask]
    obs = obs[mask]
    n = len(pred)

    if n < 2:
        return 0.5

    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            if obs[i] == obs[j]:
                continue  # skip tied observed values

            if (pred[i] > pred[j] and obs[i] > obs[j]) or (
                pred[i] < pred[j] and obs[i] < obs[j]
            ):
                concordant += 1
            elif pred[i] != pred[j]:
                discordant += 1

    total = concordant + discordant
    if total == 0:
        return 0.5

    return float(concordant / total)
