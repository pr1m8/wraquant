"""Statistical hypothesis tests for financial data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss


def test_normality(data: pd.Series, method: str = "jarque_bera") -> dict:
    """Test whether a series is normally distributed.

    Parameters:
        data: Data series to test.
        method: Test method — ``"jarque_bera"`` (default), ``"shapiro"``,
            or ``"dagostino"``.

    Returns:
        Dictionary with ``statistic``, ``p_value``, and ``is_normal``
        (at 5% significance level).

    Raises:
        ValueError: If *method* is not recognized.
    """
    from wraquant.core._coerce import coerce_series

    data = coerce_series(data, name="data")
    clean = data.dropna().values

    if method == "jarque_bera":
        stat, p = sp_stats.jarque_bera(clean)
    elif method == "shapiro":
        stat, p = sp_stats.shapiro(clean)
    elif method == "dagostino":
        stat, p = sp_stats.normaltest(clean)
    else:
        msg = f"Unknown normality test method: {method!r}"
        raise ValueError(msg)

    return {
        "statistic": float(stat),
        "p_value": float(p),
        "is_normal": bool(p > 0.05),
    }


def test_stationarity(data: pd.Series, method: str = "adf") -> dict:
    """Test whether a time series is stationary.

    Parameters:
        data: Time series to test.
        method: Test method — ``"adf"`` (Augmented Dickey-Fuller, default)
            or ``"kpss"``.

    Returns:
        Dictionary with ``statistic``, ``p_value``, and ``is_stationary``
        (at 5% significance level).

    Raises:
        ValueError: If *method* is not recognized.
    """
    from wraquant.core._coerce import coerce_series

    data = coerce_series(data, name="data")
    clean = data.dropna().values

    if method == "adf":
        result = adfuller(clean, autolag="AIC")
        stat, p = result[0], result[1]
        is_stationary = bool(p < 0.05)
    elif method == "kpss":
        stat, p, _lags, _crit = kpss(clean, regression="c", nlags="auto")
        # KPSS null hypothesis is stationarity, so reject means non-stationary
        is_stationary = bool(p > 0.05)
    else:
        msg = f"Unknown stationarity test method: {method!r}"
        raise ValueError(msg)

    return {
        "statistic": float(stat),
        "p_value": float(p),
        "is_stationary": is_stationary,
    }


def test_autocorrelation(data: pd.Series, nlags: int = 10) -> dict:
    """Ljung-Box test for autocorrelation.

    Parameters:
        data: Time series to test.
        nlags: Number of lags to test.

    Returns:
        Dictionary with ``statistic`` (at max lag), ``p_value``,
        ``is_autocorrelated`` (at 5% significance), and the full
        ``results`` DataFrame.
    """
    from wraquant.core._coerce import coerce_series

    data = coerce_series(data, name="data")
    clean = data.dropna()
    result = acorr_ljungbox(clean, lags=nlags, return_df=True)
    last_row = result.iloc[-1]
    return {
        "statistic": float(last_row["lb_stat"]),
        "p_value": float(last_row["lb_pvalue"]),
        "is_autocorrelated": bool(last_row["lb_pvalue"] < 0.05),
        "results": result,
    }


# ---------------------------------------------------------------------------
# Shapiro-Wilk normality test
# ---------------------------------------------------------------------------


def shapiro_wilk(data: pd.Series | np.ndarray) -> dict:
    """Shapiro-Wilk test for normality.

    The Shapiro-Wilk test is widely regarded as the most powerful normality
    test for small to moderate sample sizes (n < 5000).  It is more
    sensitive than the Jarque-Bera test, which relies only on skewness and
    kurtosis, because it considers the full empirical distribution.

    When to use:
        - When you have fewer than 2000 observations and need a reliable
          normality assessment (e.g., validating assumptions before
          parametric VaR, calibrating option pricing models).
        - As a complement to Jarque-Bera: Shapiro-Wilk catches departures
          in the center of the distribution that JB (which focuses on
          moments 3 and 4) may miss.
        - For validating regression residuals before using t-based
          confidence intervals.

    Mathematical formulation:
        .. math::

            W = \\frac{\\left(\\sum_{i=1}^n a_i x_{(i)}\\right)^2}{\\sum_{i=1}^n (x_i - \\bar{x})^2}

        where ``x_{(i)}`` are the order statistics and ``a_i`` are
        tabulated constants derived from the expected values and
        covariance matrix of order statistics from a normal distribution.

    How to interpret:
        - ``W`` is in (0, 1].  Values near 1 indicate normality.
        - Reject normality if ``p_value < 0.05``.
        - For financial returns, rejection is typical (fat tails),
          confirming that Gaussian-based risk measures are unreliable.

    Parameters:
        data: Data series or 1-D array.  Sample size should be between
            3 and 5000 (scipy limitation).

    Returns:
        Dictionary with:
        - ``statistic``: Shapiro-Wilk W statistic.
        - ``p_value``: p-value for H0: data is normally distributed.
        - ``is_normal``: bool, True if p > 0.05.

    Example:
        >>> import numpy as np
        >>> data = np.random.default_rng(42).normal(0, 1, 200)
        >>> result = shapiro_wilk(data)
        >>> result["is_normal"]
        True
    """
    from wraquant.core._coerce import coerce_array

    clean = coerce_array(data, name="data")
    clean = clean[~np.isnan(clean)]

    stat, p = sp_stats.shapiro(clean)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "is_normal": bool(p > 0.05),
    }


# ---------------------------------------------------------------------------
# Durbin-Watson test for autocorrelation
# ---------------------------------------------------------------------------


def durbin_watson(
    residuals: pd.Series | np.ndarray,
) -> dict:
    """Durbin-Watson test for first-order autocorrelation in residuals.

    The Durbin-Watson statistic tests whether the residuals of a
    regression model exhibit first-order serial correlation.  This is
    critical in financial econometrics where autocorrelated residuals
    invalidate standard OLS inference.

    When to use:
        - After fitting any OLS regression to time-series data (e.g.,
          CAPM beta estimation, factor models).  Autocorrelated residuals
          mean standard errors are biased and t-statistics are unreliable.
        - As a diagnostic before deciding whether to use Newey-West
          (HAC) standard errors.
        - For model validation: significant autocorrelation suggests a
          missing variable or incorrect functional form.

    Mathematical formulation:
        .. math::

            DW = \\frac{\\sum_{t=2}^T (e_t - e_{t-1})^2}{\\sum_{t=1}^T e_t^2}

    How to interpret:
        - ``DW ≈ 2.0``: no autocorrelation.
        - ``DW < 2.0``: positive autocorrelation (residuals tend to have
          the same sign as their predecessor).
        - ``DW > 2.0``: negative autocorrelation.
        - Rule of thumb: ``DW < 1.5`` or ``DW > 2.5`` indicates
          significant autocorrelation.  For precise inference, compare
          to the Durbin-Watson tables for dL and dU critical values.

    Parameters:
        residuals: Regression residuals (1-D array or Series).

    Returns:
        Dictionary with:
        - ``statistic``: Durbin-Watson statistic (range [0, 4]).
        - ``interpretation``: string describing the result.

    Example:
        >>> import numpy as np
        >>> residuals = np.random.default_rng(42).normal(0, 1, 100)
        >>> result = durbin_watson(residuals)
        >>> 1.5 < result["statistic"] < 2.5  # no autocorrelation
        True

    See Also:
        test_autocorrelation: Ljung-Box test for higher-order autocorrelation.
    """
    import numpy as np
    from statsmodels.stats.stattools import durbin_watson as _dw

    from wraquant.core._coerce import coerce_array

    clean = coerce_array(residuals, name="residuals")
    clean = clean[~np.isnan(clean)]

    dw_stat = float(_dw(clean))

    if dw_stat < 1.5:
        interpretation = "Positive autocorrelation detected (DW < 1.5)"
    elif dw_stat > 2.5:
        interpretation = "Negative autocorrelation detected (DW > 2.5)"
    else:
        interpretation = "No significant autocorrelation (1.5 <= DW <= 2.5)"

    return {
        "statistic": dw_stat,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Breusch-Pagan test for heteroskedasticity
# ---------------------------------------------------------------------------


def breusch_pagan(
    residuals: pd.Series | np.ndarray,
    exog: pd.DataFrame | np.ndarray,
) -> dict:
    """Breusch-Pagan Lagrange Multiplier test for heteroskedasticity.

    Tests whether the variance of regression residuals depends on the
    values of the independent variables.  If heteroskedasticity is
    present, OLS standard errors are biased and inference is invalid.

    When to use:
        - After OLS regression on financial data where the volatility of
          returns (and hence residuals) may depend on market conditions,
          firm size, or other regressors.
        - To decide between OLS and WLS, or whether to use White/HC
          robust standard errors.
        - For validating GARCH model residuals: after fitting a GARCH
          model, the standardized residuals should be homoskedastic.

    Mathematical formulation:
        1. Regress squared residuals ``e^2`` on the original regressors.
        2. The LM statistic is ``n * R^2`` from this auxiliary regression.
        3. Under H0 (homoskedasticity), ``LM ~ chi^2(k)`` where *k* is
           the number of regressors.

    How to interpret:
        - Low p-value (< 0.05): reject H0, heteroskedasticity is present.
          Use robust standard errors or WLS.
        - High p-value: no evidence of heteroskedasticity.  OLS inference
          is valid.

    Parameters:
        residuals: OLS regression residuals (1-D array or Series).
        exog: Design matrix of independent variables used in the original
            regression (should include constant if one was used).

    Returns:
        Dictionary with:
        - ``lm_stat``: Lagrange Multiplier statistic.
        - ``p_value``: p-value from chi-squared distribution.
        - ``f_stat``: F-statistic variant.
        - ``f_p_value``: p-value from F-distribution.
        - ``is_heteroskedastic``: bool, True if p_value < 0.05.

    Example:
        >>> import numpy as np, statsmodels.api as sm
        >>> rng = np.random.default_rng(42)
        >>> X = rng.normal(0, 1, (200, 2))
        >>> X = sm.add_constant(X)
        >>> y = X @ [1, 0.5, -0.3] + rng.normal(0, 1, 200)
        >>> resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        >>> result = breusch_pagan(resid, X)
        >>> "lm_stat" in result
        True

    See Also:
        white_test: More general heteroskedasticity test.
        durbin_watson: Test for autocorrelation instead.
    """
    import numpy as np
    from statsmodels.stats.diagnostic import het_breuschpagan

    from wraquant.core._coerce import coerce_array

    resid = coerce_array(residuals, name="residuals")
    X = np.asarray(exog, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    lm_stat, lm_pval, f_stat, f_pval = het_breuschpagan(resid, X)

    return {
        "lm_stat": float(lm_stat),
        "p_value": float(lm_pval),
        "f_stat": float(f_stat),
        "f_p_value": float(f_pval),
        "is_heteroskedastic": bool(lm_pval < 0.05),
    }


# ---------------------------------------------------------------------------
# White's test for heteroskedasticity
# ---------------------------------------------------------------------------


def white_test(
    residuals: pd.Series | np.ndarray,
    exog: pd.DataFrame | np.ndarray,
) -> dict:
    """White's test for heteroskedasticity.

    White's test is a more general heteroskedasticity test than
    Breusch-Pagan.  It does not assume a specific functional form for
    the heteroskedasticity --- it includes squares and cross-products of
    all regressors in the auxiliary regression, so it can detect
    non-linear forms of heteroskedasticity.

    When to use:
        - When you want a comprehensive heteroskedasticity diagnostic
          that does not assume the variance is a linear function of
          regressors (which Breusch-Pagan assumes).
        - When the Breusch-Pagan test fails to reject but you still
          suspect non-linear heteroskedasticity.
        - Note: White's test has lower power than BP when BP's
          assumptions are correct, and requires more observations
          because it estimates more parameters.

    Mathematical formulation:
        Regress squared residuals ``e^2`` on the original regressors,
        their squares, and all pairwise cross-products.  The test
        statistic is ``n * R^2`` from this auxiliary regression, which
        follows a chi-squared distribution under H0.

    Parameters:
        residuals: OLS regression residuals (1-D array or Series).
        exog: Design matrix of independent variables (should include
            constant if used in the original regression).

    Returns:
        Dictionary with:
        - ``lm_stat``: White LM statistic.
        - ``p_value``: p-value from chi-squared distribution.
        - ``f_stat``: F-statistic variant.
        - ``f_p_value``: p-value from F-distribution.
        - ``is_heteroskedastic``: bool, True if p_value < 0.05.

    Example:
        >>> import numpy as np, statsmodels.api as sm
        >>> rng = np.random.default_rng(42)
        >>> X = rng.normal(0, 1, (200, 2))
        >>> X = sm.add_constant(X)
        >>> # Heteroskedastic errors: variance depends on X
        >>> y = X @ [1, 0.5, -0.3] + rng.normal(0, 1, 200) * (1 + np.abs(X[:, 1]))
        >>> resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        >>> result = white_test(resid, X)
        >>> "lm_stat" in result
        True

    See Also:
        breusch_pagan: Simpler but less general heteroskedasticity test.
    """
    import numpy as np
    from statsmodels.stats.diagnostic import het_white

    from wraquant.core._coerce import coerce_array

    resid = coerce_array(residuals, name="residuals")
    X = np.asarray(exog, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    lm_stat, lm_pval, f_stat, f_pval = het_white(resid, X)

    return {
        "lm_stat": float(lm_stat),
        "p_value": float(lm_pval),
        "f_stat": float(f_stat),
        "f_p_value": float(f_pval),
        "is_heteroskedastic": bool(lm_pval < 0.05),
    }


# ---------------------------------------------------------------------------
# Chow structural break test
# ---------------------------------------------------------------------------


def chow_test(
    y: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
    break_point: int,
    add_constant: bool = True,
) -> dict:
    """Chow test for structural break at a known break point.

    The Chow test examines whether the regression coefficients differ
    between two sub-periods, i.e., whether a structural break occurred
    at the specified point.  This is fundamental in finance for detecting
    regime changes, policy shifts, or market structure changes.

    When to use:
        - To test whether a known event (e.g., a policy announcement,
          market crash, regulatory change) caused a structural change in
          the relationship between variables.
        - As a diagnostic for rolling regression: if the Chow test
          rejects stability, rolling or regime-switching models are
          warranted.
        - To validate that a backtested model's parameters are stable
          across in-sample and out-of-sample periods.

    Mathematical formulation:
        Fit the regression on the full sample, sub-sample 1 (before
        break), and sub-sample 2 (after break).  The F-statistic is:

        .. math::

            F = \\frac{(\\text{RSS}_{\\text{full}} - \\text{RSS}_1 - \\text{RSS}_2) / k}{(\\text{RSS}_1 + \\text{RSS}_2) / (n - 2k)}

        where *k* is the number of parameters and *n* is the total
        sample size.

    How to interpret:
        - Large F-stat (small p-value < 0.05): reject the null of stable
          coefficients.  A structural break is detected.
        - Small F-stat: no evidence of a break.  The relationship appears
          stable across the two sub-periods.

    Parameters:
        y: Dependent variable (1-D array or Series).
        X: Independent variables.
        break_point: Index (0-based row number) at which to split the
            sample.  Must be at least ``k + 1`` from either end.
        add_constant: Whether to add an intercept to X.

    Returns:
        Dictionary with:
        - ``f_stat``: Chow F-statistic.
        - ``p_value``: p-value from the F-distribution.
        - ``break_detected``: bool, True if p_value < 0.05.

    Raises:
        ValueError: If *break_point* is too close to the endpoints.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> X = rng.normal(0, 1, (200, 1))
        >>> y = np.concatenate([
        ...     X[:100] @ [1.0] + rng.normal(0, 0.5, 100),
        ...     X[100:] @ [3.0] + rng.normal(0, 0.5, 100),
        ... ])
        >>> result = chow_test(y, X, break_point=100)
        >>> result["break_detected"]
        True
    """
    import numpy as np
    import statsmodels.api as sm

    from wraquant.core._coerce import coerce_array

    y_arr = coerce_array(y, name="y")
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    if add_constant:
        X_arr = sm.add_constant(X_arr)

    n, k = X_arr.shape

    if break_point < k or break_point > n - k:
        msg = (
            f"break_point={break_point} is too close to the endpoints. "
            f"Need at least k={k} observations in each sub-sample."
        )
        raise ValueError(msg)

    # Full sample
    model_full = sm.OLS(y_arr, X_arr).fit()
    rss_full = float(np.sum(model_full.resid ** 2))

    # Sub-sample 1 (before break)
    model_1 = sm.OLS(y_arr[:break_point], X_arr[:break_point]).fit()
    rss_1 = float(np.sum(model_1.resid ** 2))

    # Sub-sample 2 (after break)
    model_2 = sm.OLS(y_arr[break_point:], X_arr[break_point:]).fit()
    rss_2 = float(np.sum(model_2.resid ** 2))

    # F-statistic
    f_num = (rss_full - rss_1 - rss_2) / k
    f_den = (rss_1 + rss_2) / (n - 2 * k)

    if f_den <= 0:
        return {
            "f_stat": float("inf"),
            "p_value": 0.0,
            "break_detected": True,
        }

    f_stat = f_num / f_den
    p_value = float(1.0 - sp_stats.f.cdf(f_stat, k, n - 2 * k))

    return {
        "f_stat": float(f_stat),
        "p_value": p_value,
        "break_detected": bool(p_value < 0.05),
    }


# ---------------------------------------------------------------------------
# Variance Inflation Factor
# ---------------------------------------------------------------------------


def variance_inflation_factor(
    X: pd.DataFrame,
) -> pd.Series:
    """Compute the Variance Inflation Factor (VIF) for each feature.

    VIF measures how much the variance of a regression coefficient is
    inflated due to multicollinearity with the other features.  High VIF
    indicates that a feature is nearly a linear combination of other
    features, making its coefficient estimate unstable.

    When to use:
        - Before running any multiple regression (OLS, factor model,
          Fama-MacBeth) to check for multicollinearity.
        - When regression coefficients have unexpected signs or large
          standard errors despite significant F-statistics.
        - As a feature selection diagnostic in ML pipelines.

    Mathematical formulation:
        For each feature ``X_j``, regress it on all other features and
        compute:

        .. math::

            \\text{VIF}_j = \\frac{1}{1 - R_j^2}

        where ``R_j^2`` is the R-squared from regressing ``X_j`` on the
        remaining features.

    How to interpret:
        - ``VIF = 1``: no collinearity.
        - ``VIF < 5``: low collinearity, generally acceptable.
        - ``5 <= VIF < 10``: moderate collinearity, warrants attention.
        - ``VIF >= 10``: severe collinearity.  The coefficient is poorly
          estimated.  Consider removing the feature, combining features,
          or using regularization (ridge regression).

    Parameters:
        X: DataFrame of independent variables (each column is a feature).
            Do **not** include an intercept/constant column.

    Returns:
        pd.Series of VIF values indexed by feature name.

    Example:
        >>> import pandas as pd, numpy as np
        >>> rng = np.random.default_rng(42)
        >>> x1 = rng.normal(0, 1, 200)
        >>> x2 = x1 + rng.normal(0, 0.1, 200)  # nearly collinear with x1
        >>> x3 = rng.normal(0, 1, 200)
        >>> X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
        >>> vif = variance_inflation_factor(X)
        >>> vif["x1"] > 10  # collinear pair
        True

    See Also:
        ols: OLS regression where VIF diagnostics are needed.
    """
    import numpy as np
    import statsmodels.api as sm

    from wraquant.core._coerce import coerce_dataframe

    X = coerce_dataframe(X, name="X")
    cols = X.columns.tolist()
    X_arr = X.values.astype(float)
    n_features = X_arr.shape[1]
    vif_values = np.zeros(n_features)

    for i in range(n_features):
        y_i = X_arr[:, i]
        X_other = np.delete(X_arr, i, axis=1)
        X_other = sm.add_constant(X_other)

        model = sm.OLS(y_i, X_other).fit()
        r_sq = model.rsquared

        if r_sq >= 1.0:
            vif_values[i] = float("inf")
        else:
            vif_values[i] = 1.0 / (1.0 - r_sq)

    return pd.Series(vif_values, index=cols, name="VIF")
