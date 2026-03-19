"""Regression diagnostics for econometric models.

Provides tests for serial correlation, heteroskedasticity, normality,
functional form misspecification, and multicollinearity detection.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey,
)


def durbin_watson(residuals: np.ndarray | pd.Series) -> float:
    """Compute the Durbin-Watson statistic for serial correlation.

    The statistic ranges from 0 to 4.  A value near 2 indicates no
    first-order autocorrelation, values near 0 indicate positive
    autocorrelation, and values near 4 indicate negative autocorrelation.

    Delegates to ``wraquant.stats.tests.durbin_watson`` for the core
    computation.

    Parameters:
        residuals: OLS residuals.

    Returns:
        Durbin-Watson statistic (float in [0, 4]).
    """
    from wraquant.stats.tests import durbin_watson as _stats_dw

    result = _stats_dw(residuals)
    return result["statistic"]


def breusch_godfrey(
    residuals: np.ndarray | pd.Series,
    X: np.ndarray | pd.DataFrame,
    nlags: int = 4,
) -> dict[str, Any]:
    """Breusch-Godfrey LM test for serial correlation.

    Regresses the residuals on the original regressors plus lagged residuals.
    A significant test statistic rejects the null of no serial correlation up
    to order *nlags*.

    Parameters:
        residuals: OLS residuals.
        X: Design matrix (with or without constant; a constant is added
            internally if needed).
        nlags: Number of lags to include.

    Returns:
        Dictionary with ``lm_statistic``, ``lm_p_value``, ``f_statistic``,
        ``f_p_value``, and ``is_autocorrelated`` (at 5 % level).
    """
    import statsmodels.api as sm

    resid = np.asarray(residuals).ravel()
    X_arr = np.asarray(X)

    # The BG test needs an OLS results object.  We reconstruct y so that
    # OLS(y, X).resid == resid.  Since OLS residuals are orthogonal to X,
    # any vector of the form y = X @ b + resid works; pick b = 0 for
    # simplicity, then add back the projection to make the fit non-trivial.
    y_reconstructed = X_arr @ np.linalg.lstsq(X_arr, resid, rcond=None)[0] + resid

    ols_result = sm.OLS(y_reconstructed, X_arr).fit()

    lm_stat, lm_p, f_stat, f_p = acorr_breusch_godfrey(ols_result, nlags=nlags)

    return {
        "lm_statistic": float(lm_stat),
        "lm_p_value": float(lm_p),
        "f_statistic": float(f_stat),
        "f_p_value": float(f_p),
        "is_autocorrelated": bool(lm_p < 0.05),
    }


def breusch_pagan(
    residuals: np.ndarray | pd.Series,
    X: np.ndarray | pd.DataFrame,
) -> dict[str, Any]:
    """Breusch-Pagan test for heteroskedasticity.

    Tests whether the variance of the residuals depends on the regressors.
    The null hypothesis is homoskedasticity.  Delegates to
    ``wraquant.stats.tests.breusch_pagan`` for the core computation.

    Parameters:
        residuals: OLS residuals.
        X: Design matrix used in the original regression.

    Returns:
        Dictionary with ``lm_statistic``, ``lm_p_value``, ``f_statistic``,
        ``f_p_value``, and ``is_heteroskedastic`` (at 5 % level).
    """
    from wraquant.stats.tests import breusch_pagan as _stats_bp

    result = _stats_bp(residuals, X)
    return {
        "lm_statistic": result["lm_stat"],
        "lm_p_value": result["p_value"],
        "f_statistic": result["f_stat"],
        "f_p_value": result["f_p_value"],
        "is_heteroskedastic": result["is_heteroskedastic"],
    }


def white_test(
    residuals: np.ndarray | pd.Series,
    X: np.ndarray | pd.DataFrame,
) -> dict[str, Any]:
    """White's general test for heteroskedasticity.

    Includes cross-product terms and squared regressors, making it more
    general than Breusch-Pagan.  The null hypothesis is homoskedasticity.
    Delegates to ``wraquant.stats.tests.white_test`` for the core
    computation.

    Parameters:
        residuals: OLS residuals.
        X: Design matrix used in the original regression.

    Returns:
        Dictionary with ``lm_statistic``, ``lm_p_value``, ``f_statistic``,
        ``f_p_value``, and ``is_heteroskedastic`` (at 5 % level).
    """
    from wraquant.stats.tests import white_test as _stats_white

    result = _stats_white(residuals, X)
    return {
        "lm_statistic": result["lm_stat"],
        "lm_p_value": result["p_value"],
        "f_statistic": result["f_stat"],
        "f_p_value": result["f_p_value"],
        "is_heteroskedastic": result["is_heteroskedastic"],
    }


def jarque_bera(
    residuals: np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Jarque-Bera test for normality of residuals.

    The null hypothesis is that the residuals are normally distributed.
    Delegates to ``wraquant.stats.tests.test_normality`` for the core
    computation.

    Parameters:
        residuals: Model residuals.

    Returns:
        Dictionary with ``statistic``, ``p_value``, ``skewness``,
        ``kurtosis``, and ``is_normal`` (at 5 % level).
    """
    from wraquant.stats.tests import test_normality as _test_normality

    resid_series = pd.Series(np.asarray(residuals).ravel())
    result = _test_normality(resid_series, method="jarque_bera")

    resid_clean = resid_series.dropna().values
    return {
        "statistic": result["statistic"],
        "p_value": result["p_value"],
        "skewness": float(sp_stats.skew(resid_clean)),
        "kurtosis": float(sp_stats.kurtosis(resid_clean, fisher=True)),
        "is_normal": result["is_normal"],
    }


def ramsey_reset(
    y: np.ndarray | pd.Series,
    X: np.ndarray | pd.DataFrame,
    power: int = 3,
) -> dict[str, Any]:
    """Ramsey RESET test for functional form misspecification.

    Adds powers of the fitted values to the regression and tests their
    joint significance.  Rejection suggests the linear specification is
    inadequate.

    Parameters:
        y: Dependent variable.
        X: Design matrix (should include a constant).
        power: Maximum power of fitted values to include (default 3 adds
            y_hat^2 and y_hat^3).

    Returns:
        Dictionary with ``f_statistic``, ``p_value``, ``df_num``,
        ``df_denom``, and ``is_misspecified`` (at 5 % level).
    """
    import statsmodels.api as sm

    y_arr = np.asarray(y).ravel()
    X_arr = np.asarray(X)

    # Original regression
    ols_orig = sm.OLS(y_arr, X_arr).fit()
    y_hat = ols_orig.fittedvalues

    # Build augmented regressors with y_hat^2, ..., y_hat^power
    augmented_cols = [X_arr]
    for p in range(2, power + 1):
        augmented_cols.append((y_hat**p).reshape(-1, 1))
    X_aug = np.hstack(augmented_cols)

    ols_aug = sm.OLS(y_arr, X_aug).fit()

    # F-test for the added variables
    n = len(y_arr)
    k_orig = X_arr.shape[1]
    k_aug = X_aug.shape[1]
    df_num = k_aug - k_orig
    df_denom = n - k_aug

    ssr_orig = np.sum(ols_orig.resid**2)
    ssr_aug = np.sum(ols_aug.resid**2)

    f_stat = ((ssr_orig - ssr_aug) / df_num) / (ssr_aug / df_denom)
    p_value = 1.0 - sp_stats.f.cdf(f_stat, df_num, df_denom)

    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "df_num": int(df_num),
        "df_denom": int(df_denom),
        "is_misspecified": bool(p_value < 0.05),
    }


def vif(X: np.ndarray | pd.DataFrame) -> pd.Series:
    """Compute Variance Inflation Factors for each regressor.

    VIF > 10 is a common rule-of-thumb threshold indicating problematic
    multicollinearity.  The input should *not* include a constant column.

    Parameters:
        X: Design matrix **without** an intercept.

    Returns:
        Series of VIF values indexed by column name (or integer index).
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    n_cols = X_arr.shape[1]

    if isinstance(X, pd.DataFrame):
        names = list(X.columns)
    else:
        names = list(range(n_cols))

    # Add constant for VIF calculation
    import statsmodels.api as sm

    X_with_const = sm.add_constant(X_arr)

    vif_values = [
        variance_inflation_factor(X_with_const, i + 1) for i in range(n_cols)
    ]

    return pd.Series(vif_values, index=names, name="VIF")


def condition_number(X: np.ndarray | pd.DataFrame) -> float:
    """Compute the condition number of the X'X matrix.

    A condition number exceeding 30 suggests harmful multicollinearity
    (Belsley, Kuh, and Welsch, 1980).

    Parameters:
        X: Design matrix.

    Returns:
        Condition number (ratio of largest to smallest singular value of X).
    """
    X_arr = np.asarray(X, dtype=float)
    sv = np.linalg.svd(X_arr, compute_uv=False)
    return float(sv.max() / sv.min())
