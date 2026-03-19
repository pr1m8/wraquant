"""Time series econometrics.

Provides Vector Autoregression (VAR), Vector Error Correction Models (VECM),
Granger causality tests, impulse response functions, forecast error variance
decomposition, and structural break tests.  These are the core multivariate
time series tools used in macrofinance and empirical asset pricing
(Campbell-Lo-MacKinlay ch. 11; Hamilton ch. 11-19; Lutkepohl, 2005).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def var_model(
    data: pd.DataFrame | np.ndarray,
    max_lags: int | None = None,
    ic: str = "aic",
) -> dict[str, Any]:
    """Fit a Vector Autoregression (VAR) model.

    Selects lag order by information criterion and estimates the reduced-form
    VAR by equation-by-equation OLS.

    Parameters:
        data: Multivariate time series (T, k).  Columns are treated as
            endogenous variables.
        max_lags: Maximum lag order to consider.  Defaults to
            ``int(12 * (T / 100) ** (1/4))``.
        ic: Information criterion for lag selection -- ``"aic"``, ``"bic"``,
            ``"hqic"``, or ``"fpe"``.

    Returns:
        Dictionary with ``coefficients`` (k x k*p + 1 matrix where the last
        column is the intercept), ``lag_order``, ``residuals`` (T-p, k),
        ``sigma_u`` (innovation covariance), ``aic``, ``bic``, ``fittedvalues``,
        and a ``forecast`` callable ``(steps) -> np.ndarray``.
    """
    from statsmodels.tsa.api import VAR as _VAR

    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    model = _VAR(data)

    if max_lags is None:
        T = len(data)
        max_lags = int(12 * (T / 100) ** 0.25)
        max_lags = max(max_lags, 1)

    result = model.fit(maxlags=max_lags, ic=ic)

    # Build coefficient matrix [A1 | A2 | ... | Ap | c]
    p = result.k_ar
    k = result.neqs
    coef_matrices = []
    for lag in range(1, p + 1):
        coef_matrices.append(result.coefs[lag - 1])  # (k, k)
    intercept = result.intercept.reshape(-1, 1)  # (k, 1)
    coef_full = np.hstack([*coef_matrices, intercept])  # (k, k*p + 1)

    def _forecast(steps: int) -> np.ndarray:
        return result.forecast(result.endog[-p:], steps=steps)

    return {
        "coefficients": coef_full,
        "lag_order": p,
        "residuals": result.resid,
        "sigma_u": result.sigma_u,
        "aic": float(result.aic),
        "bic": float(result.bic),
        "fittedvalues": result.fittedvalues,
        "forecast": _forecast,
    }


def vecm_model(
    data: pd.DataFrame | np.ndarray,
    k_ar_diff: int = 1,
    det_order: int = 0,
) -> dict[str, Any]:
    """Fit a Vector Error Correction Model (VECM) for cointegrated systems.

    Estimates the Johansen cointegrating rank and then fits the VECM of the
    form:  Delta y_t = alpha * beta' * y_{t-1} + Gamma * Delta y_{t-1} + ...

    Parameters:
        data: Multivariate time series (T, k).
        k_ar_diff: Number of lagged difference terms.
        det_order: Deterministic term order (-1 = none, 0 = constant inside
            the cointegrating relation, 1 = linear trend).

    Returns:
        Dictionary with ``alpha`` (adjustment coefficients), ``beta``
        (cointegrating vectors), ``gamma`` (short-run coefficients),
        ``det_coef`` (deterministic coefficients), ``coint_rank``,
        ``residuals``, and ``sigma_u``.
    """
    from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank

    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    # Select cointegration rank via trace test
    rank_test = select_coint_rank(data, det_order=det_order, k_ar_diff=k_ar_diff)
    coint_rank = rank_test.rank

    # Ensure at least rank 1 for estimation (VECM requires rank >= 1)
    if coint_rank == 0:
        coint_rank = 1

    model = VECM(data, k_ar_diff=k_ar_diff, coint_rank=coint_rank, deterministic="ci")
    result = model.fit()

    return {
        "alpha": result.alpha,
        "beta": result.beta,
        "gamma": result.gamma,
        "det_coef": result.det_coef,
        "coint_rank": coint_rank,
        "residuals": result.resid,
        "sigma_u": result.sigma_u,
    }


def granger_causality(
    data: pd.DataFrame | np.ndarray,
    max_lag: int = 10,
) -> dict[str, Any]:
    """Pairwise Granger causality tests for all variable pairs.

    For each ordered pair (X, Y), tests whether lagged values of X help
    predict Y beyond Y's own lags.  Uses a VAR framework.

    Parameters:
        data: Multivariate time series (T, k).
        max_lag: Maximum lag order to test.

    Returns:
        Dictionary mapping ``"X -> Y"`` to a dict with ``f_statistic``,
        ``p_value``, ``lag_order``, and ``is_causal`` (at 5 % level) for
        each directed pair.
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    columns = list(data.columns)
    k = len(columns)
    results: dict[str, Any] = {}

    for i in range(k):
        for j in range(k):
            if i == j:
                continue

            # grangercausalitytests expects [y, x] column order
            pair_data = data[[columns[j], columns[i]]].dropna()

            if len(pair_data) < max_lag + 2:
                continue

            try:
                gc = grangercausalitytests(pair_data, maxlag=max_lag, verbose=False)
            except Exception:
                continue

            # Find the best lag by minimum p-value
            best_lag = 1
            best_p = 1.0
            best_f = 0.0
            for lag in range(1, max_lag + 1):
                if lag in gc:
                    f_test = gc[lag][0]["ssr_ftest"]
                    f_val, p_val = f_test[0], f_test[1]
                    if p_val < best_p:
                        best_p = p_val
                        best_f = f_val
                        best_lag = lag

            label = f"{columns[i]} -> {columns[j]}"
            results[label] = {
                "f_statistic": float(best_f),
                "p_value": float(best_p),
                "lag_order": best_lag,
                "is_causal": bool(best_p < 0.05),
            }

    return results


def impulse_response(
    var_coefficients: np.ndarray,
    n_periods: int = 20,
    shock_var: int = 0,
) -> np.ndarray:
    """Compute impulse response functions from VAR coefficient matrices.

    Applies a one-unit shock to *shock_var* at time 0 and traces out the
    dynamic response of all variables over *n_periods*.

    Parameters:
        var_coefficients: VAR coefficient matrix of shape (k, k*p) or
            (k, k*p + 1) where the last column is the intercept.  The
            k-by-k blocks are [A1 | A2 | ... | Ap].
        n_periods: Number of periods for the IRF (default 20).
        shock_var: Index of the variable receiving the unit shock.

    Returns:
        Array of shape (n_periods + 1, k) where row *h* is the response
        of all k variables at horizon h.
    """
    coef = np.asarray(var_coefficients, dtype=float)
    k = coef.shape[0]

    # Determine number of lags
    n_cols = coef.shape[1]
    # If the last column is an intercept (n_cols not divisible by k), strip it
    if n_cols % k != 0:
        coef = coef[:, : -(n_cols % k)]

    p = coef.shape[1] // k
    A_mats = [coef[:, i * k : (i + 1) * k] for i in range(p)]

    # Compute IRFs by recursion
    irf = np.zeros((n_periods + 1, k))
    # Initial shock: unit impulse to shock_var
    irf[0, shock_var] = 1.0

    for h in range(1, n_periods + 1):
        for lag, A in enumerate(A_mats):
            idx = h - lag - 1
            if 0 <= idx:
                irf[h] += A @ irf[idx]

    return irf


def variance_decomposition(
    var_coefficients: np.ndarray,
    n_periods: int = 20,
) -> np.ndarray:
    """Forecast error variance decomposition from VAR coefficients.

    Decomposes the forecast error variance of each variable into
    contributions from each structural shock (Cholesky identification).

    Parameters:
        var_coefficients: VAR coefficient matrix (k, k*p) or (k, k*p + 1).
        n_periods: Forecast horizon.

    Returns:
        Array of shape (n_periods + 1, k, k) where entry [h, i, j] is the
        fraction of the h-step forecast error variance of variable *i*
        attributable to shocks in variable *j*.
    """
    coef = np.asarray(var_coefficients, dtype=float)
    k = coef.shape[0]

    # Compute IRFs for each shock variable
    irfs = np.zeros((n_periods + 1, k, k))  # [horizon, response_var, shock_var]
    for shock in range(k):
        irfs[:, :, shock] = impulse_response(coef, n_periods, shock_var=shock)

    # Cumulative squared IRFs
    cum_sq = np.cumsum(irfs**2, axis=0)  # (n_periods + 1, k, k)

    # Total variance for each variable at each horizon
    total_var = cum_sq.sum(axis=2, keepdims=True)  # (n_periods + 1, k, 1)

    # Avoid division by zero at h=0
    total_var = np.where(total_var == 0, 1.0, total_var)

    fevd = cum_sq / total_var  # (n_periods + 1, k, k)

    return fevd


def structural_break_test(
    y: np.ndarray | pd.Series,
    X: np.ndarray | pd.DataFrame | None = None,
    method: str = "chow",
    break_point: int | None = None,
) -> dict[str, Any]:
    """Test for structural breaks in a regression relationship.

    Parameters:
        y: Dependent variable (T,).
        X: Design matrix (T, k).  If ``None``, a constant-only model is
            used (testing a break in the mean).
        method: ``"chow"`` for a known break point, or ``"sup_f"`` for the
            supremum-F test (Andrews, 1993) which searches over candidate
            break points.
        break_point: Observation index of the hypothesised break (required
            for ``method="chow"``).  For ``"sup_f"`` this is ignored.

    Returns:
        Dictionary with ``f_statistic``, ``p_value``, ``break_point``, and
        ``is_break`` (at 5 % level).

    Raises:
        ValueError: If ``method="chow"`` and *break_point* is ``None``.
    """
    import statsmodels.api as sm

    y_arr = np.asarray(y, dtype=float).ravel()
    T = len(y_arr)

    if X is None:
        X_arr = np.ones((T, 1))
    else:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

    k = X_arr.shape[1]

    if method == "chow":
        if break_point is None:
            msg = "break_point must be specified for method='chow'"
            raise ValueError(msg)

        # Delegate to stats/tests.chow_test (single source of truth)
        from wraquant.stats.tests import chow_test as _chow_test

        chow_result = _chow_test(y_arr, X_arr, break_point=break_point, add_constant=False)
        return {
            "f_statistic": chow_result["f_stat"],
            "p_value": chow_result["p_value"],
            "break_point": break_point,
            "is_break": chow_result["break_detected"],
        }

    elif method == "sup_f":
        # Andrews (1993) supremum F-test
        # Search over the central 70% of observations
        trim = max(int(0.15 * T), k + 1)
        candidates = range(trim, T - trim)

        best_f = -np.inf
        best_bp = trim
        best_p = 1.0

        # Full-sample SSR
        ols_full = sm.OLS(y_arr, X_arr).fit()
        ssr_full = np.sum(ols_full.resid**2)

        for bp in candidates:
            if bp < k + 1 or (T - bp) < k + 1:
                continue

            y1, X1 = y_arr[:bp], X_arr[:bp]
            y2, X2 = y_arr[bp:], X_arr[bp:]

            try:
                ols1 = sm.OLS(y1, X1).fit()
                ols2 = sm.OLS(y2, X2).fit()
            except Exception:
                continue

            ssr_sub = np.sum(ols1.resid**2) + np.sum(ols2.resid**2)
            df_num = k
            df_denom = T - 2 * k

            f_val = ((ssr_full - ssr_sub) / df_num) / (
                ssr_sub / max(df_denom, 1)
            )

            if f_val > best_f:
                best_f = f_val
                best_bp = bp

        # Approximate p-value using F distribution (conservative)
        df_num = k
        df_denom = T - 2 * k
        best_p = 1.0 - sp_stats.f.cdf(best_f, df_num, max(df_denom, 1))

        return {
            "f_statistic": float(best_f),
            "p_value": float(best_p),
            "break_point": best_bp,
            "is_break": bool(best_p < 0.05),
        }

    else:
        msg = f"Unknown method: {method!r}. Use 'chow' or 'sup_f'."
        raise ValueError(msg)
