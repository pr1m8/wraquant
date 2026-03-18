"""Regression models for financial econometrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def ols(
    y: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
    add_constant: bool = True,
) -> dict:
    """Ordinary least squares regression.

    Parameters:
        y: Dependent variable (response).
        X: Independent variables (regressors).
        add_constant: Whether to add an intercept column to *X*.

    Returns:
        Dictionary with ``coefficients`` (array), ``t_stats`` (array),
        ``p_values`` (array), ``r_squared``, ``adj_r_squared``, and
        ``residuals`` (array).
    """
    y_arr = np.asarray(y, dtype=float)
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

    Parameters:
        y: Dependent variable series.
        X: Independent variable(s).  A Series is treated as a single
            regressor; a DataFrame may contain multiple regressors.
        window: Rolling window size.
        add_constant: Whether to add an intercept column.

    Returns:
        Dictionary with ``coefficients`` (DataFrame of rolling betas)
        and ``r_squared`` (Series of rolling R-squared values).
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

    Parameters:
        y: Dependent variable.
        X: Independent variables.
        weights: Observation weights (higher weight = more influence).
        add_constant: Whether to add an intercept column.

    Returns:
        Dictionary with ``coefficients``, ``t_stats``, ``p_values``,
        ``r_squared``, ``adj_r_squared``, and ``residuals``.
    """
    y_arr = np.asarray(y, dtype=float)
    X_arr = np.asarray(X, dtype=float)
    w_arr = np.asarray(weights, dtype=float)

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

    Pass 1 (time-series): For each time period, regress cross-sectional
    returns on factor exposures.
    Pass 2 (cross-section): Average the time-series of cross-sectional
    slope coefficients to estimate risk premia.

    Parameters:
        panel_y: DataFrame of returns with shape ``(T, N)`` where *T*
            is the number of time periods and *N* is the number of
            assets.
        panel_X: Factor exposures.  Either a single DataFrame with the
            same shape as *panel_y* (single factor), or a dictionary
            mapping factor names to DataFrames of exposures.

    Returns:
        Dictionary with ``risk_premia`` (array of average slope
        coefficients), ``t_stats`` (Fama-MacBeth *t*-statistics),
        ``r_squared`` (average cross-sectional R-squared), and
        ``gamma_series`` (DataFrame of period-by-period slope
        coefficients).
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

    Heteroskedasticity and autocorrelation consistent (HAC) standard
    errors are used, which are robust to both heteroskedasticity and
    serial correlation in the residuals.

    Parameters:
        y: Dependent variable.
        X: Independent variables.
        max_lags: Maximum number of lags for the Newey-West estimator.
            When *None*, uses ``floor(4 * (T/100)^(2/9))``.
        add_constant: Whether to add an intercept column.

    Returns:
        Dictionary with ``coefficients``, ``t_stats``, ``p_values``,
        ``r_squared``, ``adj_r_squared``, ``residuals``, and
        ``hac_se`` (HAC standard errors).
    """
    y_arr = np.asarray(y, dtype=float)
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
