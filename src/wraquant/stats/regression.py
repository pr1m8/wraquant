"""Regression models for financial econometrics.

Provides OLS, WLS, rolling OLS, Fama-MacBeth cross-sectional
regression, and Newey-West HAC-robust regression -- the standard
toolkit for empirical asset pricing, factor modelling, and
return attribution.

How to choose:
    - **OLS** (``ols``): the starting point. Estimates the linear
      relationship between a dependent variable and regressors.
      Assumes homoskedastic, serially uncorrelated errors.
    - **WLS** (``wls``): use when observation reliability varies (e.g.,
      weight by inverse variance, or give more weight to recent data).
      Common for heteroskedastic financial returns.
    - **Newey-West OLS** (``newey_west_ols``): use when residuals are
      both heteroskedastic *and* autocorrelated. Standard errors are
      HAC-robust, so t-statistics and p-values are reliable even when
      the OLS error assumptions fail (which they usually do in finance).
    - **Rolling OLS** (``rolling_ols``): use to track time-varying
      coefficients (e.g., evolving beta, hedge ratio). Essential for
      detecting parameter instability.
    - **Fama-MacBeth** (``fama_macbeth``): the standard for estimating
      risk premia in cross-sectional asset pricing. Two-pass procedure
      that handles the errors-in-variables problem.

References:
    - Fama & MacBeth (1973), "Risk, Return, and Equilibrium: Empirical
      Tests"
    - Newey & West (1987), "A Simple, Positive Semi-Definite,
      Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from wraquant.core._coerce import coerce_array


def ols(
    y: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
    add_constant: bool = True,
) -> dict:
    """Ordinary least squares regression.

    OLS finds the linear coefficients that minimise the sum of squared
    residuals. It is the foundation of empirical finance -- used for
    CAPM beta estimation, factor model fitting, and return attribution.

    When to use:
        Use OLS as the default regression when you have a cross-section
        or time-series of returns to explain. Switch to WLS if errors
        are heteroskedastic, or to Newey-West if errors are also
        autocorrelated (common in overlapping return regressions).

    Mathematical formulation:
        y = X * beta + epsilon
        beta_hat = (X'X)^{-1} X'y

    How to interpret:
        - ``coefficients[0]`` is the intercept (alpha in CAPM).
        - ``coefficients[1:]`` are the factor loadings (betas).
        - ``t_stats`` and ``p_values``: test H0: beta_i = 0. Reject if
          |t| > 2 (roughly, p < 0.05).
        - ``r_squared``: fraction of variance explained. For CAPM
          on individual stocks, R^2 of 0.10-0.30 is typical.
        - ``residuals``: unexplained returns. Check for autocorrelation,
          heteroskedasticity, and normality.

    Parameters:
        y: Dependent variable (e.g., asset returns).
        X: Independent variables (e.g., factor returns). Can be 1-D
            (single factor) or 2-D (multiple factors).
        add_constant: Whether to add an intercept column to *X*.

    Returns:
        Dictionary with ``coefficients`` (array), ``t_stats`` (array),
        ``p_values`` (array), ``r_squared``, ``adj_r_squared``, and
        ``residuals`` (array).

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> market = np.random.normal(0.0005, 0.01, 252)
        >>> stock = 0.001 + 1.2 * market + np.random.normal(0, 0.005, 252)
        >>> result = ols(stock, market)
        >>> abs(result["coefficients"][1] - 1.2) < 0.3  # beta near 1.2
        True

    See Also:
        wls: Weighted least squares.
        newey_west_ols: OLS with HAC-robust standard errors.
        rolling_ols: Time-varying coefficient estimation.
    """
    y_arr = coerce_array(y, "y")
    X_arr = np.asarray(X, dtype=float)

    if add_constant:
        X_arr = sm.add_constant(X_arr)

    model = sm.OLS(y_arr, X_arr).fit()

    return {
        "coefficients": model.params,
        "t_stats": model.tvalues,
        "p_values": model.pvalues,
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "residuals": model.resid,
    }


def rolling_ols(
    y: pd.Series,
    X: pd.DataFrame | pd.Series,
    window: int = 60,
    add_constant: bool = True,
) -> dict:
    """Rolling window OLS regression.

    Fits OLS independently at each time step using only the most recent
    *window* observations, producing time-varying coefficient estimates.
    This is essential for detecting parameter instability -- a common
    feature of financial data where betas, hedge ratios, and factor
    loadings evolve over time.

    When to use:
        Use rolling OLS when:
        - You suspect the relationship between variables changes over
          time (e.g., a stock's beta increasing during crises).
        - You need time-varying hedge ratios for pairs trading.
        - You want to validate that a full-sample OLS result is stable.
        For a more adaptive approach, consider Kalman regression
        (``wraquant.regimes.kalman_regression``), which estimates
        time-varying coefficients with a state-space model rather than
        fixed windows.

    How to interpret:
        - ``coefficients``: DataFrame of rolling betas. Plot to see
          how each coefficient evolves. Large swings indicate parameter
          instability.
        - ``r_squared``: rolling R-squared. A declining R-squared
          suggests the model's explanatory power is deteriorating.

    Parameters:
        y: Dependent variable series.
        X: Independent variable(s). A Series is treated as a single
            regressor; a DataFrame may contain multiple regressors.
        window: Rolling window size (e.g., 60 for ~3 months of daily
            data). Shorter windows are more responsive but noisier.
        add_constant: Whether to add an intercept column.

    Returns:
        Dictionary with ``coefficients`` (DataFrame of rolling betas,
        NaN before the first full window) and ``r_squared`` (Series of
        rolling R-squared values).

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> market = pd.Series(np.random.normal(0, 0.01, 252))
        >>> stock = 1.2 * market + np.random.normal(0, 0.005, 252)
        >>> result = rolling_ols(stock, market, window=60)
        >>> result["coefficients"].dropna().iloc[-1, 1]  # beta estimate
        1.2...

    See Also:
        ols: Full-sample OLS regression.
        wraquant.regimes.kalman_regression: Kalman-filter-based
            time-varying regression.
    """
    if isinstance(X, pd.Series):
        X = X.to_frame()

    if add_constant:
        X_with_const = sm.add_constant(X)
    else:
        X_with_const = X.copy()

    n = len(y)
    n_params = X_with_const.shape[1]
    col_names = (
        list(X_with_const.columns)
        if hasattr(X_with_const, "columns")
        else [f"x{i}" for i in range(n_params)]
    )

    coef_data = np.full((n, n_params), np.nan)
    r2_data = np.full(n, np.nan)

    y_vals = np.asarray(y, dtype=float)
    X_vals = np.asarray(X_with_const, dtype=float)

    for i in range(window - 1, n):
        start = i - window + 1
        y_win = y_vals[start : i + 1]
        X_win = X_vals[start : i + 1]

        try:
            model = sm.OLS(y_win, X_win).fit()
            coef_data[i] = model.params
            r2_data[i] = model.rsquared
        except (np.linalg.LinAlgError, ValueError):
            continue

    index = y.index if isinstance(y, pd.Series) else range(n)
    coefficients = pd.DataFrame(coef_data, index=index, columns=col_names)
    r_squared = pd.Series(r2_data, index=index, name="r_squared")

    return {
        "coefficients": coefficients,
        "r_squared": r_squared,
    }


def wls(
    y: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
    weights: pd.Series | np.ndarray,
    add_constant: bool = True,
) -> dict:
    """Weighted least squares regression.

    WLS accounts for heteroskedasticity by assigning different weights
    to observations. Observations with higher weight have more influence
    on the coefficient estimates. This is the appropriate estimator when
    the variance of the error term differs across observations.

    When to use:
        Use WLS when:
        - You know or suspect heteroskedasticity (non-constant error
          variance). For example, high-cap stocks have less noisy
          returns than micro-caps.
        - You want to give more weight to recent observations
          (exponentially decaying weights for time-series regression).
        - You have grouped data where some groups are more precisely
          measured than others.
        If you do not know the weights, use OLS with Newey-West (HAC)
        standard errors (``newey_west_ols``) instead.

    Mathematical formulation:
        Minimises sum_i w_i * (y_i - X_i * beta)^2

        Equivalent to OLS on the transformed system:
        sqrt(w_i) * y_i = sqrt(w_i) * X_i * beta + epsilon_i

    How to interpret:
        Same output structure as ``ols``. The difference is that
        coefficients are efficient under heteroskedasticity (lower
        standard errors than OLS when the weights are correctly
        specified). If the weights are misspecified, WLS can be worse
        than OLS.

    Parameters:
        y: Dependent variable.
        X: Independent variables.
        weights: Observation weights (higher weight = more influence).
            Common choices: inverse variance, exponential decay, or
            sample size per group.
        add_constant: Whether to add an intercept column.

    Returns:
        Dictionary with ``coefficients``, ``t_stats``, ``p_values``,
        ``r_squared``, ``adj_r_squared``, and ``residuals``.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = np.random.randn(100, 2)
        >>> y = X @ [1.5, -0.5] + np.random.randn(100) * (1 + np.abs(X[:, 0]))
        >>> w = 1.0 / (1 + np.abs(X[:, 0]))  # inverse heteroskedasticity
        >>> result = wls(y, X, weights=w)
        >>> len(result["coefficients"])
        3

    See Also:
        ols: Unweighted OLS (assumes homoskedasticity).
        newey_west_ols: OLS with robust standard errors.
    """
    y_arr = coerce_array(y, "y")
    X_arr = np.asarray(X, dtype=float)
    w_arr = coerce_array(weights, "weights")

    if add_constant:
        X_arr = sm.add_constant(X_arr)

    model = sm.WLS(y_arr, X_arr, weights=w_arr).fit()

    return {
        "coefficients": model.params,
        "t_stats": model.tvalues,
        "p_values": model.pvalues,
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "residuals": model.resid,
    }


def fama_macbeth(
    panel_y: pd.DataFrame,
    panel_X: pd.DataFrame | dict[str, pd.DataFrame],
) -> dict:
    """Fama-MacBeth two-pass cross-sectional regression.

    The Fama-MacBeth procedure is the standard methodology for testing
    whether a factor commands a risk premium in cross-sectional asset
    pricing. It handles the errors-in-variables problem that arises when
    estimated betas are used as regressors.

    When to use:
        Use Fama-MacBeth when:
        - You want to estimate risk premia for factors (e.g., "does
          size, value, or momentum have a positive risk premium?").
        - You have panel data: many assets observed over many time
          periods.
        - You want *t*-statistics that properly account for cross-
          sectional correlation (unlike pooled OLS, which overstates
          significance).
        For time-series factor models (e.g., CAPM alpha of a single
        fund), use plain OLS instead.

    Mathematical formulation:
        Pass 1: For each time period t, run cross-sectional OLS:
            r_{i,t} = gamma_{0,t} + gamma_{1,t} * beta_{i,t} + e_{i,t}

        Pass 2: Average the cross-sectional slopes over time:
            gamma_k = mean(gamma_{k,t})
            t_k = gamma_k / (std(gamma_{k,t}) / sqrt(T))

    How to interpret:
        - ``risk_premia``: average slope coefficients. A positive,
          statistically significant risk premium means the factor is
          priced.
        - ``t_stats``: Fama-MacBeth *t*-statistics. |t| > 2 suggests
          the premium is statistically significant.
        - ``gamma_series``: period-by-period slopes. Plot to see
          time variation in risk premia.
        - ``r_squared``: average explanatory power of the factors in
          the cross-section.

    Parameters:
        panel_y: DataFrame of returns with shape ``(T, N)`` where *T*
            is the number of time periods and *N* is the number of
            assets.
        panel_X: Factor exposures. Either a single DataFrame with the
            same shape as *panel_y* (single factor), or a dictionary
            mapping factor names to DataFrames of exposures.

    Returns:
        Dictionary with ``risk_premia`` (array), ``t_stats`` (array),
        ``r_squared`` (float), and ``gamma_series`` (DataFrame of
        period-by-period slope coefficients).

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> T, N = 120, 30
        >>> betas = np.random.randn(T, N)
        >>> returns = pd.DataFrame(0.005 * betas + np.random.randn(T, N) * 0.01)
        >>> exposures = pd.DataFrame(betas)
        >>> result = fama_macbeth(returns, exposures)
        >>> len(result["risk_premia"])
        2

    See Also:
        ols: Single time-series OLS regression.
        newey_west_ols: OLS with HAC-robust standard errors.

    References:
        - Fama & MacBeth (1973), "Risk, Return, and Equilibrium:
          Empirical Tests"
    """
    # Normalise panel_X to a dict of DataFrames
    if isinstance(panel_X, pd.DataFrame):
        factor_dict = {"factor_1": panel_X}
    else:
        factor_dict = dict(panel_X)

    factor_names = list(factor_dict.keys())
    n_factors = len(factor_names)
    periods = panel_y.index

    gamma_data = np.full((len(periods), n_factors + 1), np.nan)  # +1 for intercept

    r2_list: list[float] = []

    for t_idx, t in enumerate(periods):
        y_cross = panel_y.loc[t].dropna()
        assets = y_cross.index

        # Build cross-sectional X matrix
        X_parts: list[np.ndarray] = []
        valid = True
        for fname in factor_names:
            fdata = factor_dict[fname]
            if t not in fdata.index:
                valid = False
                break
            x_row = fdata.loc[t].reindex(assets)
            if x_row.isna().all():
                valid = False
                break
            X_parts.append(x_row.values.astype(float))

        if not valid or len(y_cross) < n_factors + 2:
            continue

        X_cross = np.column_stack(X_parts)
        X_cross = sm.add_constant(X_cross)
        y_arr = y_cross.values.astype(float)

        # Drop rows with NaN
        mask = ~(np.isnan(y_arr) | np.isnan(X_cross).any(axis=1))
        if mask.sum() < n_factors + 2:
            continue

        try:
            model = sm.OLS(y_arr[mask], X_cross[mask]).fit()
            gamma_data[t_idx] = model.params
            r2_list.append(float(model.rsquared))
        except (np.linalg.LinAlgError, ValueError):
            continue

    col_names = ["intercept"] + factor_names
    gamma_df = pd.DataFrame(gamma_data, index=periods, columns=col_names)
    gamma_clean = gamma_df.dropna()

    if len(gamma_clean) == 0:
        return {
            "risk_premia": np.zeros(n_factors + 1),
            "t_stats": np.zeros(n_factors + 1),
            "r_squared": 0.0,
            "gamma_series": gamma_df,
        }

    means = gamma_clean.mean().values
    stds = gamma_clean.std(ddof=1).values
    n_periods = len(gamma_clean)
    t_stats = means / (stds / np.sqrt(n_periods))

    avg_r2 = float(np.mean(r2_list)) if r2_list else 0.0

    return {
        "risk_premia": means,
        "t_stats": t_stats,
        "r_squared": avg_r2,
        "gamma_series": gamma_df,
    }


def newey_west_ols(
    y: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
    max_lags: int | None = None,
    add_constant: bool = True,
) -> dict:
    """OLS regression with Newey-West HAC standard errors.

    Produces the same coefficient estimates as OLS, but computes
    standard errors that are robust to both heteroskedasticity *and*
    autocorrelation (HAC). This is the standard approach in financial
    econometrics where both issues are almost always present.

    When to use:
        Use Newey-West whenever:
        - The regression involves overlapping returns (e.g., monthly
          returns sampled from daily data) which introduce mechanical
          autocorrelation.
        - You suspect GARCH-type volatility clustering in the
          residuals (heteroskedasticity + autocorrelation).
        - You want valid inference without specifying the exact form
          of heteroskedasticity (unlike WLS which requires known
          weights).
        Newey-West is strictly better than plain OLS for inference
        in financial data. The point estimates are the same; only the
        standard errors, t-statistics, and p-values differ.

    Mathematical formulation:
        The HAC variance estimator is:
            V_HAC = (X'X)^{-1} S (X'X)^{-1}

        where S = sum_{j=-L}^{L} w(j) * Gamma_j is the kernel-
        weighted sum of autocovariance matrices of X * epsilon,
        using the Bartlett kernel w(j) = 1 - |j|/(L+1).

    How to interpret:
        Same output as ``ols``, plus ``hac_se`` (the HAC standard
        errors). Compare ``hac_se`` to the OLS standard errors: if
        HAC SE > OLS SE, the OLS was underestimating uncertainty
        (common in finance). Use the HAC-based ``t_stats`` and
        ``p_values`` for inference.

    Parameters:
        y: Dependent variable.
        X: Independent variables.
        max_lags: Maximum number of lags for the Newey-West kernel.
            When *None*, uses ``floor(4 * (T/100)^(2/9))`` (Andrews
            1991 rule of thumb).
        add_constant: Whether to add an intercept column.

    Returns:
        Dictionary with ``coefficients``, ``t_stats``, ``p_values``,
        ``r_squared``, ``adj_r_squared``, ``residuals``, and
        ``hac_se`` (HAC standard errors).

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = np.random.randn(200, 2)
        >>> y = X @ [1.0, 0.5] + np.random.randn(200) * 0.5
        >>> result = newey_west_ols(y, X)
        >>> len(result["hac_se"])
        3

    See Also:
        ols: OLS with classical (non-robust) standard errors.
        wls: Weighted least squares for known heteroskedasticity.

    References:
        - Newey & West (1987), "A Simple, Positive Semi-Definite,
          Heteroskedasticity and Autocorrelation Consistent Covariance
          Matrix"
    """
    y_arr = coerce_array(y, "y")
    X_arr = np.asarray(X, dtype=float)

    if add_constant:
        X_arr = sm.add_constant(X_arr)

    if max_lags is None:
        T = len(y_arr)
        max_lags = int(np.floor(4 * (T / 100) ** (2 / 9)))

    model = sm.OLS(y_arr, X_arr).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": max_lags},
    )

    return {
        "coefficients": model.params,
        "t_stats": model.tvalues,
        "p_values": model.pvalues,
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "residuals": model.resid,
        "hac_se": model.bse,
    }
