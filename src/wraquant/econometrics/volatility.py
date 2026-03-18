"""Volatility econometrics.

Provides the ARCH-LM test and backward-compatible wrappers for GARCH-family
models.  All GARCH estimation is delegated to :mod:`wraquant.vol.models`,
which is the canonical home for volatility modeling.

The pure-numpy GARCH(1,1) fallback is retained as a private helper
(:func:`_garch_numpy_fallback`) for environments without the ``arch``
library.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize as sp_optimize
from scipy import stats as sp_stats

from wraquant.core.decorators import requires_extra


# ---------------------------------------------------------------------------
# ARCH-LM test (pure statsmodels, always available)
# ---------------------------------------------------------------------------


def arch_test(
    residuals: np.ndarray | pd.Series,
    nlags: int = 5,
) -> dict[str, Any]:
    """Engle's ARCH-LM test for conditional heteroskedasticity.

    Regresses squared residuals on their own lags.  A significant test
    statistic indicates the presence of ARCH effects, justifying the use
    of GARCH-type models.

    Parameters:
        residuals: Model residuals or return series.
        nlags: Number of lags to include in the auxiliary regression.

    Returns:
        Dictionary with ``statistic`` (LM statistic), ``p_value``,
        ``f_statistic``, ``f_p_value``, and ``is_arch`` (True at 5 %
        level).
    """
    from statsmodels.stats.diagnostic import het_arch

    resid = np.asarray(residuals, dtype=float).ravel()
    lm_stat, lm_p, f_stat, f_p = het_arch(resid, nlags=nlags)

    return {
        "statistic": float(lm_stat),
        "p_value": float(lm_p),
        "f_statistic": float(f_stat),
        "f_p_value": float(f_p),
        "is_arch": bool(lm_p < 0.05),
    }


# ---------------------------------------------------------------------------
# Pure numpy GARCH(1,1) fallback (private)
# ---------------------------------------------------------------------------


def _garch_loglik(
    params: np.ndarray,
    returns: np.ndarray,
) -> float:
    """Negative log-likelihood for GARCH(1,1) with normal innovations."""
    omega, alpha, beta = params
    T = len(returns)

    # Enforce parameter constraints
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10

    sigma2 = np.empty(T)
    sigma2[0] = np.var(returns)

    for t in range(1, T):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        if sigma2[t] <= 0:
            return 1e10

    ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + returns**2 / sigma2)
    return -ll


def _garch_numpy_fallback(
    returns: np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Fit a GARCH(1,1) model using pure numpy/scipy (private fallback).

    This is a fallback for when the ``arch`` library is not installed.
    Only supports GARCH(1,1) with normal innovations.

    Parameters:
        returns: Return series.

    Returns:
        Dictionary with ``params``, ``conditional_volatility``,
        ``standardized_residuals``, ``forecast``, and ``loglikelihood``.
    """
    ret = np.asarray(returns, dtype=float).ravel()
    ret = ret - ret.mean()
    var0 = np.var(ret)

    # Initial guesses: omega, alpha, beta
    x0 = np.array([var0 * 0.05, 0.05, 0.90])
    bounds = [(1e-8, None), (1e-8, 0.999), (1e-8, 0.999)]

    result = sp_optimize.minimize(
        _garch_loglik,
        x0,
        args=(ret,),
        method="L-BFGS-B",
        bounds=bounds,
    )

    omega, alpha, beta = result.x

    # Reconstruct conditional variance
    T = len(ret)
    sigma2 = np.empty(T)
    sigma2[0] = var0
    for t in range(1, T):
        sigma2[t] = omega + alpha * ret[t - 1] ** 2 + beta * sigma2[t - 1]

    cond_vol = np.sqrt(sigma2)
    std_resid = ret / cond_vol

    # One-step forecast
    forecast_var = omega + alpha * ret[-1] ** 2 + beta * sigma2[-1]

    return {
        "params": {
            "omega": float(omega),
            "alpha[1]": float(alpha),
            "beta[1]": float(beta),
        },
        "conditional_volatility": cond_vol,
        "standardized_residuals": std_resid,
        "forecast": np.sqrt(forecast_var),
        "loglikelihood": -result.fun,
    }


# ---------------------------------------------------------------------------
# GARCH family — thin wrappers delegating to vol.models
# ---------------------------------------------------------------------------


@requires_extra("timeseries")
def garch(
    returns: np.ndarray | pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
) -> dict[str, Any]:
    """Fit a GARCH(p,q) model.

    Delegates to :func:`wraquant.vol.models.garch_fit`.

    Parameters:
        returns: Return series (not percentage returns).
        p: GARCH lag order (conditional variance lags).
        q: ARCH lag order (squared innovation lags).
        dist: Error distribution -- ``"normal"``, ``"studentst"``, or
            ``"skewt"``.

    Returns:
        Dictionary with ``params``, ``conditional_volatility``,
        ``standardized_residuals``, ``forecast`` (one-step-ahead
        volatility), ``aic``, ``bic``, and ``loglikelihood``.
    """
    from wraquant.vol.models import garch_fit

    result = garch_fit(returns, p=p, q=q, dist=dist)

    # Map vol.models output keys to legacy econometrics interface
    return {
        "params": result["params"],
        "conditional_volatility": result["conditional_volatility"],
        "standardized_residuals": result["standardized_residuals"],
        "forecast": result.get("forecast", None),
        "aic": result.get("aic", None),
        "bic": result.get("bic", None),
        "loglikelihood": result.get("log_likelihood", None),
    }


def garch_numpy_fallback(
    returns: np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Fit a GARCH(1,1) model using pure numpy/scipy.

    This is a backward-compatible public wrapper around the private
    :func:`_garch_numpy_fallback` helper.

    Parameters:
        returns: Return series.

    Returns:
        Dictionary with ``params``, ``conditional_volatility``,
        ``standardized_residuals``, ``forecast``, and ``loglikelihood``.
    """
    return _garch_numpy_fallback(returns)


@requires_extra("timeseries")
def egarch(
    returns: np.ndarray | pd.Series,
    p: int = 1,
    q: int = 1,
) -> dict[str, Any]:
    """Fit an EGARCH(p,q) model (exponential GARCH).

    Delegates to :func:`wraquant.vol.models.egarch_fit`.

    Parameters:
        returns: Return series.
        p: GARCH lag order.
        q: ARCH lag order.

    Returns:
        Dictionary with ``params``, ``conditional_volatility``,
        ``standardized_residuals``, ``forecast``, ``aic``, ``bic``,
        and ``loglikelihood``.
    """
    from wraquant.vol.models import egarch_fit

    result = egarch_fit(returns, p=p, q=q)

    return {
        "params": result["params"],
        "conditional_volatility": result["conditional_volatility"],
        "standardized_residuals": result["standardized_residuals"],
        "forecast": result.get("forecast", None),
        "aic": result.get("aic", None),
        "bic": result.get("bic", None),
        "loglikelihood": result.get("log_likelihood", None),
    }


@requires_extra("timeseries")
def gjr_garch(
    returns: np.ndarray | pd.Series,
    p: int = 1,
    q: int = 1,
) -> dict[str, Any]:
    """Fit a GJR-GARCH(p,q) model (threshold GARCH).

    Delegates to :func:`wraquant.vol.models.gjr_garch_fit`.

    Parameters:
        returns: Return series.
        p: GARCH lag order.
        q: ARCH lag order.

    Returns:
        Dictionary with ``params``, ``conditional_volatility``,
        ``standardized_residuals``, ``forecast``, ``aic``, ``bic``,
        and ``loglikelihood``.
    """
    from wraquant.vol.models import gjr_garch_fit

    result = gjr_garch_fit(returns, p=p, q=q)

    return {
        "params": result["params"],
        "conditional_volatility": result["conditional_volatility"],
        "standardized_residuals": result["standardized_residuals"],
        "forecast": result.get("forecast", None),
        "aic": result.get("aic", None),
        "bic": result.get("bic", None),
        "loglikelihood": result.get("log_likelihood", None),
    }


@requires_extra("timeseries")
def dcc_garch(
    returns_df: pd.DataFrame,
    p: int = 1,
    q: int = 1,
) -> dict[str, Any]:
    """Fit a Dynamic Conditional Correlation (DCC) GARCH model.

    Delegates to :func:`wraquant.vol.models.dcc_fit`.

    Parameters:
        returns_df: DataFrame of return series (T, k), one column per asset.
        p: GARCH lag order for univariate models.
        q: ARCH lag order for univariate models.

    Returns:
        Dictionary with ``univariate_params`` (per-asset GARCH parameters),
        ``conditional_correlations`` (T, k, k), ``conditional_covariances``
        (T, k, k), and ``standardized_residuals`` (T, k).
    """
    from wraquant.vol.models import dcc_fit

    result = dcc_fit(returns_df, p=p, q=q)

    # Map vol.models output to legacy econometrics interface
    # Extract per-asset params from univariate_results
    univariate_params: dict[str, dict[str, float]] = {}
    columns = list(returns_df.columns) if isinstance(returns_df, pd.DataFrame) else []
    for i, uni_res in enumerate(result.get("univariate_results", [])):
        col_name = columns[i] if i < len(columns) else f"series_{i}"
        univariate_params[col_name] = uni_res.get("params", {})

    return {
        "univariate_params": univariate_params,
        "conditional_correlations": result["conditional_correlations"],
        "conditional_covariances": result["conditional_covariances"],
        "standardized_residuals": result["standardized_residuals"],
        "dcc_params": result["dcc_params"],
    }
