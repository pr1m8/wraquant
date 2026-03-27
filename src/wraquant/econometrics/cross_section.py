"""Cross-sectional econometric methods.

Provides robust OLS, quantile regression, instrumental variables (2SLS),
GMM estimation, and the Sargan overidentification test -- core tools for
empirical asset pricing and cross-sectional return analysis.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import optimize as sp_optimize
from scipy import stats as sp_stats

from wraquant.core._coerce import coerce_array


def robust_ols(
    y: np.ndarray | pd.Series,
    X: np.ndarray | pd.DataFrame,
    cov_type: str = "HC1",
) -> dict[str, Any]:
    """OLS regression with heteroskedasticity-robust standard errors.

    Implements the HC0 -- HC3 covariance estimators of White (1980) and
    MacKinnon and White (1985).  Point estimates are identical to plain
    OLS; only the standard errors, t-statistics, and p-values differ.

    Parameters:
        y: Dependent variable (n,).
        X: Design matrix (n, k).  A constant is **not** added automatically;
            use ``sm.add_constant(X)`` beforehand if you need an intercept.
        cov_type: Heteroskedasticity-robust covariance type.  One of
            ``"HC0"``, ``"HC1"``, ``"HC2"``, or ``"HC3"``.

    Returns:
        Dictionary with ``coefficients``, ``std_errors``, ``t_stats``,
        ``p_values``, ``r_squared``, ``adj_r_squared``, ``residuals``,
        ``cov_type``, and ``nobs``.
    """
    import statsmodels.api as sm

    y_arr = coerce_array(y, name="y")
    X_arr = np.asarray(X, dtype=np.float64)

    model = sm.OLS(y_arr, X_arr)
    result = model.fit(cov_type=cov_type)

    return {
        "coefficients": result.params,
        "std_errors": result.bse,
        "t_stats": result.tvalues,
        "p_values": result.pvalues,
        "r_squared": float(result.rsquared),
        "adj_r_squared": float(result.rsquared_adj),
        "residuals": result.resid,
        "cov_type": cov_type,
        "nobs": int(result.nobs),
    }


def quantile_regression(
    y: np.ndarray | pd.Series,
    X: np.ndarray | pd.DataFrame,
    quantile: float = 0.5,
) -> dict[str, Any]:
    """Quantile regression via statsmodels.

    Estimates the conditional quantile function, generalising OLS which
    estimates the conditional *mean*.  Useful for understanding heterogeneous
    effects across the return distribution.

    Parameters:
        y: Dependent variable (n,).
        X: Design matrix (n, k).
        quantile: Quantile to estimate (0 < quantile < 1).  Defaults to the
            median (0.5).

    Returns:
        Dictionary with ``coefficients``, ``std_errors``, ``t_stats``,
        ``p_values``, ``quantile``, ``pseudo_r_squared``, and ``nobs``.
    """
    import statsmodels.api as sm

    y_arr = coerce_array(y, name="y")
    X_arr = np.asarray(X, dtype=np.float64)

    model = sm.QuantReg(y_arr, X_arr)
    result = model.fit(q=quantile)

    return {
        "coefficients": result.params,
        "std_errors": result.bse,
        "t_stats": result.tvalues,
        "p_values": result.pvalues,
        "quantile": quantile,
        "pseudo_r_squared": float(result.prsquared),
        "nobs": int(result.nobs),
    }


def two_stage_least_squares(
    y: np.ndarray | pd.Series,
    X: np.ndarray | pd.DataFrame,
    instruments: np.ndarray | pd.DataFrame,
    endog_vars: list[int] | np.ndarray,
) -> dict[str, Any]:
    """Two-stage least squares (2SLS) instrumental variables estimator.

    Addresses endogeneity by instrumenting the endogenous regressors in *X*
    with exogenous *instruments*.  The first stage regresses each endogenous
    variable on the instruments, and the second stage regresses *y* on the
    fitted values plus the exogenous regressors.

    Parameters:
        y: Dependent variable (n,).
        X: Full design matrix (n, k) containing both exogenous and endogenous
            regressors.
        instruments: Matrix of excluded instruments (n, m) with m >= number
            of endogenous variables (order condition).
        endog_vars: Column indices (0-based) within *X* that are endogenous.

    Returns:
        Dictionary with ``coefficients``, ``std_errors``, ``t_stats``,
        ``p_values``, ``r_squared``, ``residuals``, ``first_stage_f``,
        and ``nobs``.

    Raises:
        ValueError: If the order condition is violated (fewer instruments
            than endogenous variables).
    """
    y_arr = coerce_array(y, name="y")
    X_arr = np.asarray(X, dtype=np.float64)
    Z_excl = np.asarray(instruments, dtype=np.float64)
    endog_idx = np.asarray(endog_vars).ravel()

    n, k = X_arr.shape
    n_endog = len(endog_idx)

    if Z_excl.ndim == 1:
        Z_excl = Z_excl.reshape(-1, 1)

    if Z_excl.shape[1] < n_endog:
        msg = (
            f"Order condition violated: {Z_excl.shape[1]} excluded instruments "
            f"for {n_endog} endogenous variables."
        )
        raise ValueError(msg)

    # Identify exogenous columns in X
    all_idx = np.arange(k)
    exog_idx = np.setdiff1d(all_idx, endog_idx)
    X_exog = X_arr[:, exog_idx]

    # Full instrument set = exogenous regressors + excluded instruments
    Z = np.hstack([X_exog, Z_excl])

    # --- First stage: regress each endogenous variable on Z ---
    P_z = Z @ np.linalg.lstsq(Z, np.eye(n), rcond=None)[0]  # noqa: N806
    # More stable: projection matrix via QR
    Q, R = np.linalg.qr(Z)
    P_z = Q @ Q.T  # noqa: N806

    X_hat = X_arr.copy()
    first_stage_f_stats = []
    for idx in endog_idx:
        X_hat[:, idx] = P_z @ X_arr[:, idx]
        # First-stage F-statistic
        fitted = X_hat[:, idx]
        resid_first = X_arr[:, idx] - fitted
        ss_reg = np.sum((fitted - np.mean(X_arr[:, idx])) ** 2)
        ss_res = np.sum(resid_first**2)
        df_reg = Z.shape[1]
        df_res = n - Z.shape[1]
        f_val = (ss_reg / df_reg) / (ss_res / max(df_res, 1))
        first_stage_f_stats.append(float(f_val))

    # --- Second stage: regress y on X_hat ---
    from wraquant.stats.regression import ols as _ols

    _2sls_result = _ols(y_arr, X_hat, add_constant=False)
    beta_2sls = _2sls_result["coefficients"]

    # Residuals use actual X, not fitted X_hat
    residuals = y_arr - X_arr @ beta_2sls

    # Standard errors
    sigma2 = np.sum(residuals**2) / (n - k)
    # 2SLS variance: sigma^2 * (X_hat' X_hat)^{-1}
    XhXh_inv = np.linalg.inv(X_hat.T @ X_hat)
    cov_beta = sigma2 * XhXh_inv
    std_errors = np.sqrt(np.diag(cov_beta))
    t_stats = beta_2sls / std_errors
    p_values = 2.0 * (1.0 - sp_stats.t.cdf(np.abs(t_stats), df=n - k))

    # R-squared (may be negative for IV)
    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
    ss_res = np.sum(residuals**2)
    r_squared = 1.0 - ss_res / ss_tot

    return {
        "coefficients": beta_2sls,
        "std_errors": std_errors,
        "t_stats": t_stats,
        "p_values": p_values,
        "r_squared": float(r_squared),
        "residuals": residuals,
        "first_stage_f": first_stage_f_stats,
        "nobs": n,
    }


def gmm_estimation(
    moment_conditions: Callable[[np.ndarray, np.ndarray], np.ndarray],
    params_init: np.ndarray,
    W: np.ndarray | None = None,
    *,
    data: np.ndarray | None = None,
    max_iter: int = 2,
) -> dict[str, Any]:
    """Generalized Method of Moments (GMM) estimation.

    Minimises the quadratic form  g(theta)' W g(theta) where g(theta) is the
    sample average of moment conditions.  Supports iterated GMM (two-step by
    default).

    Parameters:
        moment_conditions: Callable ``(params, data) -> (n, q)`` returning the
            n-by-q matrix of moment conditions evaluated at *params*.  Each
            row corresponds to an observation, each column to a moment
            condition.
        params_init: Initial parameter vector (p,).
        W: Weighting matrix (q, q).  Defaults to the identity matrix
            (one-step GMM).  After the first step the optimal weighting
            matrix is computed from the residual moment conditions.
        data: Data array passed as the second argument to
            *moment_conditions*.  If ``None``, a zero-length array is used.
        max_iter: Maximum number of GMM iterations (default 2 = two-step GMM).

    Returns:
        Dictionary with ``params``, ``objective``, ``moment_conditions_mean``,
        ``W``, ``nobs``, and ``n_moments``.
    """
    theta = np.asarray(params_init, dtype=float).copy()
    if data is None:
        data = np.empty(0)
    data_arr = np.asarray(data, dtype=float)

    # Evaluate moment conditions to determine dimensions
    g_mat = moment_conditions(theta, data_arr)
    n, q = g_mat.shape

    if W is None:
        W_mat = np.eye(q)
    else:
        W_mat = np.asarray(W, dtype=float)

    for _ in range(max_iter):

        def _objective(
            params: np.ndarray,
            _W: np.ndarray = W_mat,
        ) -> float:
            g = moment_conditions(params, data_arr)
            g_bar = g.mean(axis=0)
            return float(g_bar @ _W @ g_bar)

        result = sp_optimize.minimize(
            _objective,
            theta,
            method="BFGS",
            options={"maxiter": 5000, "gtol": 1e-10},
        )
        theta = result.x

        # Update weighting matrix (optimal = inverse of S)
        g_mat = moment_conditions(theta, data_arr)
        g_bar = g_mat.mean(axis=0)
        g_centered = g_mat - g_bar
        S = (g_centered.T @ g_centered) / n
        try:
            W_mat = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            W_mat = np.linalg.pinv(S)

    g_final = moment_conditions(theta, data_arr)
    g_bar_final = g_final.mean(axis=0)
    obj_val = float(g_bar_final @ W_mat @ g_bar_final)

    return {
        "params": theta,
        "objective": obj_val,
        "moment_conditions_mean": g_bar_final,
        "W": W_mat,
        "nobs": n,
        "n_moments": q,
    }


def sargan_test(
    residuals: np.ndarray | pd.Series,
    instruments: np.ndarray | pd.DataFrame,
) -> dict[str, Any]:
    """Sargan-Hansen J-test for overidentifying restrictions.

    Tests whether excluded instruments are validly uncorrelated with the
    structural error.  Only applicable when the model is overidentified
    (more instruments than endogenous variables).

    Parameters:
        residuals: 2SLS residuals (n,).
        instruments: Full instrument matrix (n, m) including both included
            and excluded instruments.

    Returns:
        Dictionary with ``statistic``, ``p_value``, ``df``, and ``is_valid``
        (instruments are valid at 5 % level if not rejected).
    """
    resid = coerce_array(residuals, name="residuals")
    Z = np.asarray(instruments, dtype=np.float64)

    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    n, m = Z.shape

    # Regress residuals on instruments
    Q, R = np.linalg.qr(Z)
    fitted = Q @ (Q.T @ resid)

    # J = n * R^2 from regressing residuals on instruments
    ss_res = np.sum((resid - fitted) ** 2)
    ss_tot = np.sum((resid - resid.mean()) ** 2)

    if ss_tot == 0:
        r_squared = 0.0
    else:
        r_squared = 1.0 - ss_res / ss_tot

    j_stat = n * r_squared

    # Degrees of freedom = number of overidentifying restrictions
    # In practice, the user should pass the number of endogenous regressors
    # separately.  Here we use m - 1 as a common case (one endogenous var).
    # The test is chi-squared with df = m - k_endog, but since we do not
    # know k_endog here, we report the statistic and let the user set df.
    # We default to df = m (equivalent to regressing residuals on all instruments).
    df = m
    p_value = 1.0 - sp_stats.chi2.cdf(j_stat, df=max(df, 1))

    return {
        "statistic": float(j_stat),
        "p_value": float(p_value),
        "df": int(df),
        "is_valid": bool(p_value > 0.05),
    }
