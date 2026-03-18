"""Panel data econometrics.

Provides pooled OLS, fixed effects (within estimator), random effects (GLS),
between effects, first-difference estimator, and the Hausman specification
test.  These are the workhorses of empirical asset pricing with panel data
(e.g. Fama-MacBeth regressions, firm-level regressions with firm and time
fixed effects).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def pooled_ols(
    y: np.ndarray | pd.Series,
    X: np.ndarray | pd.DataFrame,
) -> dict[str, Any]:
    """Pooled OLS regression ignoring panel structure.

    Treats all observations as independent cross-sectional data.  This is
    generally inconsistent when unobserved heterogeneity is present but
    serves as a baseline for comparison with panel estimators.

    Parameters:
        y: Dependent variable (n,).
        X: Design matrix (n, k).  Should include a constant column if an
            intercept is desired.

    Returns:
        Dictionary with ``coefficients``, ``std_errors``, ``t_stats``,
        ``p_values``, ``r_squared``, ``adj_r_squared``, ``residuals``,
        and ``nobs``.
    """
    import statsmodels.api as sm

    y_arr = np.asarray(y, dtype=float).ravel()
    X_arr = np.asarray(X, dtype=float)

    result = sm.OLS(y_arr, X_arr).fit()

    return {
        "coefficients": result.params,
        "std_errors": result.bse,
        "t_stats": result.tvalues,
        "p_values": result.pvalues,
        "r_squared": float(result.rsquared),
        "adj_r_squared": float(result.rsquared_adj),
        "residuals": result.resid,
        "nobs": int(result.nobs),
    }


def fixed_effects(
    y: pd.Series,
    X: pd.DataFrame,
    entity_col: str,
    time_col: str | None = None,
) -> dict[str, Any]:
    """Entity fixed effects regression (within estimator).

    Demeans all variables by entity (and optionally by time period) before
    running OLS, which is algebraically equivalent to including entity
    dummies.  This removes time-invariant unobserved heterogeneity.

    Parameters:
        y: Dependent variable.  Must share the same index as *X*.
        X: Regressor DataFrame.  Must contain *entity_col* (and optionally
            *time_col*).  All other columns are treated as regressors.
        entity_col: Column name identifying the cross-sectional unit.
        time_col: Optional column name identifying the time period for
            two-way fixed effects.

    Returns:
        Dictionary with ``coefficients``, ``std_errors``, ``t_stats``,
        ``p_values``, ``r_squared`` (within), ``entity_effects``,
        ``residuals``, ``nobs``, and ``n_entities``.
    """
    df = X.copy()
    df["__y__"] = np.asarray(y, dtype=float)

    # Identify regressor columns (exclude entity and time columns)
    drop_cols = {entity_col}
    if time_col is not None:
        drop_cols.add(time_col)
    reg_cols = [c for c in X.columns if c not in drop_cols]

    # Entity demeaning
    group_means = df.groupby(entity_col)[["__y__", *reg_cols]].transform("mean")
    df_demeaned = df[["__y__", *reg_cols]] - group_means

    # Time demeaning (two-way FE)
    if time_col is not None:
        time_means = df.copy()
        time_means[["__y__", *reg_cols]] = df_demeaned[["__y__", *reg_cols]]
        time_group_means = (
            time_means.groupby(time_col)[["__y__", *reg_cols]].transform("mean")
        )
        overall_means = df_demeaned[["__y__", *reg_cols]].mean()
        df_demeaned[["__y__", *reg_cols]] = (
            df_demeaned[["__y__", *reg_cols]] - time_group_means + overall_means
        )

    y_dm = df_demeaned["__y__"].values
    X_dm = df_demeaned[reg_cols].values

    n = len(y_dm)
    k = X_dm.shape[1]
    n_entities = df[entity_col].nunique()

    # OLS on demeaned data
    beta = np.linalg.lstsq(X_dm, y_dm, rcond=None)[0]
    residuals = y_dm - X_dm @ beta

    # Degrees of freedom: n - n_entities - k (for entity FE)
    df_resid = n - n_entities - k
    if time_col is not None:
        n_times = df[time_col].nunique()
        df_resid = n - n_entities - n_times - k + 1

    sigma2 = np.sum(residuals**2) / max(df_resid, 1)
    XtX_inv = np.linalg.inv(X_dm.T @ X_dm)
    cov_beta = sigma2 * XtX_inv
    std_errors = np.sqrt(np.diag(cov_beta))
    t_stats = beta / std_errors
    p_values = 2.0 * (1.0 - sp_stats.t.cdf(np.abs(t_stats), df=max(df_resid, 1)))

    # Within R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_dm - y_dm.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Entity effects (group mean of y - X @ beta)
    y_arr = np.asarray(y, dtype=float)
    X_reg = X[reg_cols].values
    entity_effects_raw = y_arr - X_reg @ beta
    entity_effects_series = pd.Series(entity_effects_raw, index=X.index)
    entity_effects = entity_effects_series.groupby(df[entity_col]).mean()

    coef_series = pd.Series(beta, index=reg_cols)
    se_series = pd.Series(std_errors, index=reg_cols)
    t_series = pd.Series(t_stats, index=reg_cols)
    p_series = pd.Series(p_values, index=reg_cols)

    return {
        "coefficients": coef_series,
        "std_errors": se_series,
        "t_stats": t_series,
        "p_values": p_series,
        "r_squared": float(r_squared),
        "entity_effects": entity_effects,
        "residuals": residuals,
        "nobs": n,
        "n_entities": n_entities,
    }


def random_effects(
    y: pd.Series,
    X: pd.DataFrame,
    entity_col: str,
) -> dict[str, Any]:
    """Random effects (GLS) panel regression.

    Assumes that the unobserved entity effect is uncorrelated with the
    regressors.  Applies a partial demeaning transformation using the
    estimated variance components, then runs GLS.

    Parameters:
        y: Dependent variable.
        X: Regressor DataFrame containing *entity_col*.
        entity_col: Column identifying the cross-sectional unit.

    Returns:
        Dictionary with ``coefficients``, ``std_errors``, ``t_stats``,
        ``p_values``, ``r_squared``, ``theta`` (partial-demeaning
        parameter), ``residuals``, and ``nobs``.
    """
    df = X.copy()
    df["__y__"] = np.asarray(y, dtype=float)

    reg_cols = [c for c in X.columns if c != entity_col]
    groups = df[entity_col]

    # --- Step 1: estimate variance components from FE residuals ---
    group_means_y = df.groupby(entity_col)["__y__"].transform("mean")
    group_means_X = df.groupby(entity_col)[reg_cols].transform("mean")

    y_dm = df["__y__"].values - group_means_y.values
    X_dm = df[reg_cols].values - group_means_X.values

    n = len(y_dm)
    n_entities = groups.nunique()
    k = X_dm.shape[1]

    beta_fe = np.linalg.lstsq(X_dm, y_dm, rcond=None)[0]
    resid_fe = y_dm - X_dm @ beta_fe

    sigma2_e = np.sum(resid_fe**2) / max(n - n_entities - k, 1)

    # Between estimator residuals for sigma2_u
    T_i = groups.value_counts()  # observations per entity
    T_bar = T_i.mean()

    # Total residual from pooled OLS for sigma2_total
    import statsmodels.api as sm

    X_with_const = sm.add_constant(df[reg_cols].values)
    pooled = sm.OLS(df["__y__"].values, X_with_const).fit()

    # Variance of group means of pooled residuals
    pooled_resid = pd.Series(pooled.resid, index=df.index)
    group_mean_resid = pooled_resid.groupby(groups).mean()
    sigma2_between = group_mean_resid.var()

    sigma2_u = max(float(sigma2_between) - sigma2_e / T_bar, 0.0)

    # --- Step 2: partial demeaning ---
    # theta_i = 1 - sqrt(sigma2_e / (T_i * sigma2_u + sigma2_e))
    theta_dict = {}
    for entity, Ti in T_i.items():
        denom = Ti * sigma2_u + sigma2_e
        if denom > 0:
            theta_dict[entity] = 1.0 - np.sqrt(sigma2_e / denom)
        else:
            theta_dict[entity] = 0.0

    theta_series = groups.map(theta_dict).values

    # Quasi-demeaned variables
    y_re = df["__y__"].values - theta_series * group_means_y.values
    X_re = df[reg_cols].values - theta_series[:, None] * group_means_X.values
    X_re_const = sm.add_constant(X_re)

    # GLS estimation
    result = sm.OLS(y_re, X_re_const).fit()

    # Use all coefficients (intercept + slopes)
    coefficients = result.params
    coef_names = ["const", *reg_cols]

    residuals = df["__y__"].values - sm.add_constant(df[reg_cols].values) @ coefficients

    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((df["__y__"].values - df["__y__"].mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    theta_avg = float(np.mean(list(theta_dict.values())))

    return {
        "coefficients": pd.Series(coefficients, index=coef_names),
        "std_errors": pd.Series(result.bse, index=coef_names),
        "t_stats": pd.Series(result.tvalues, index=coef_names),
        "p_values": pd.Series(result.pvalues, index=coef_names),
        "r_squared": float(r_squared),
        "theta": theta_avg,
        "residuals": residuals,
        "nobs": n,
    }


def hausman_test(
    fe_results: dict[str, Any],
    re_results: dict[str, Any],
) -> dict[str, Any]:
    """Hausman specification test for fixed vs. random effects.

    Under the null hypothesis, both FE and RE are consistent, but RE is
    efficient.  Rejection favours fixed effects.

    Parameters:
        fe_results: Output from :func:`fixed_effects`.
        re_results: Output from :func:`random_effects`.

    Returns:
        Dictionary with ``statistic``, ``p_value``, ``df``, and ``prefer``
        (``"fe"`` or ``"re"``).
    """
    beta_fe = np.asarray(fe_results["coefficients"])
    beta_re_full = np.asarray(re_results["coefficients"])

    # RE includes a constant; FE does not.  Align on the slope coefficients.
    fe_names = list(fe_results["coefficients"].index)
    re_names = list(re_results["coefficients"].index)

    # Find common slope names
    common = [n for n in fe_names if n in re_names and n != "const"]
    if not common:
        common = fe_names  # fallback

    fe_idx = [fe_names.index(n) for n in common]
    re_idx = [re_names.index(n) for n in common]

    b_fe = beta_fe[fe_idx]
    b_re = beta_re_full[re_idx]
    diff = b_fe - b_re

    # Variance of the difference
    var_fe = np.diag(
        np.asarray(fe_results["std_errors"])[fe_idx] ** 2
    ) * np.eye(len(common))
    var_re = np.diag(
        np.asarray(re_results["std_errors"])[re_idx] ** 2
    ) * np.eye(len(common))

    # Reconstruct covariance matrices from std errors (diagonal approx)
    cov_diff = var_fe - var_re

    # Ensure positive definiteness by taking absolute values on diagonal
    cov_diff = np.abs(cov_diff)

    try:
        cov_diff_inv = np.linalg.inv(cov_diff)
    except np.linalg.LinAlgError:
        cov_diff_inv = np.linalg.pinv(cov_diff)

    stat = float(diff @ cov_diff_inv @ diff)
    df = len(common)
    p_value = 1.0 - sp_stats.chi2.cdf(stat, df=df)

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "df": df,
        "prefer": "fe" if p_value < 0.05 else "re",
    }


def between_effects(
    y: pd.Series,
    X: pd.DataFrame,
    entity_col: str,
) -> dict[str, Any]:
    """Between estimator (regression on group means).

    Collapses the panel to entity-level means and runs OLS.  Exploits only
    the cross-sectional (between) variation and is inconsistent in the
    presence of entity-level omitted variables.

    Parameters:
        y: Dependent variable.
        X: Regressor DataFrame containing *entity_col*.
        entity_col: Column identifying the cross-sectional unit.

    Returns:
        Dictionary with ``coefficients``, ``std_errors``, ``t_stats``,
        ``p_values``, ``r_squared``, ``residuals``, and ``n_entities``.
    """
    import statsmodels.api as sm

    df = X.copy()
    df["__y__"] = np.asarray(y, dtype=float)

    reg_cols = [c for c in X.columns if c != entity_col]

    # Group means
    gm = df.groupby(entity_col)[["__y__", *reg_cols]].mean()

    # Drop columns that are constant across entities (e.g. time index)
    varying = [c for c in reg_cols if gm[c].std() > 1e-12]

    y_bar = gm["__y__"].values
    X_bar = sm.add_constant(gm[varying].values)

    result = sm.OLS(y_bar, X_bar).fit()

    coef_names = ["const", *varying]

    return {
        "coefficients": pd.Series(result.params, index=coef_names),
        "std_errors": pd.Series(result.bse, index=coef_names),
        "t_stats": pd.Series(result.tvalues, index=coef_names),
        "p_values": pd.Series(result.pvalues, index=coef_names),
        "r_squared": float(result.rsquared),
        "residuals": result.resid,
        "n_entities": len(gm),
    }


def first_difference(
    y: pd.Series,
    X: pd.DataFrame,
    entity_col: str,
    time_col: str,
) -> dict[str, Any]:
    """First-difference estimator.

    Differences all variables within each entity, eliminating the
    time-invariant entity effect.  Under strict exogeneity this is
    consistent; it is often preferred over FE when the errors are a
    random walk.

    Parameters:
        y: Dependent variable.
        X: Regressor DataFrame containing *entity_col* and *time_col*.
        entity_col: Column identifying the cross-sectional unit.
        time_col: Column identifying the time period.

    Returns:
        Dictionary with ``coefficients``, ``std_errors``, ``t_stats``,
        ``p_values``, ``r_squared``, ``residuals``, and ``nobs``.
    """
    import statsmodels.api as sm

    df = X.copy()
    df["__y__"] = np.asarray(y, dtype=float)

    reg_cols = [c for c in X.columns if c not in {entity_col, time_col}]

    # Sort and difference within entity
    df = df.sort_values([entity_col, time_col])
    diff_cols = ["__y__", *reg_cols]
    df_diff = df.groupby(entity_col)[diff_cols].diff().dropna()

    y_d = df_diff["__y__"].values
    X_d = df_diff[reg_cols].values

    result = sm.OLS(y_d, X_d).fit()

    return {
        "coefficients": pd.Series(result.params, index=reg_cols),
        "std_errors": pd.Series(result.bse, index=reg_cols),
        "t_stats": pd.Series(result.tvalues, index=reg_cols),
        "p_values": pd.Series(result.pvalues, index=reg_cols),
        "r_squared": float(result.rsquared),
        "residuals": result.resid,
        "nobs": len(y_d),
    }
