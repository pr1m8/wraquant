"""Volatility econometrics.

Provides GARCH-family models (GARCH, EGARCH, GJR-GARCH, DCC-GARCH,
Realized GARCH), the ARCH-LM test, and pure-numpy fallbacks for basic
GARCH estimation when the ``arch`` library is unavailable.
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
# Pure numpy GARCH(1,1) fallback
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


def _fit_garch_numpy(
    returns: np.ndarray,
) -> dict[str, Any]:
    """Fit GARCH(1,1) via MLE using pure numpy/scipy (fallback)."""
    ret = returns - returns.mean()
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
# GARCH family (arch library)
# ---------------------------------------------------------------------------


@requires_extra("timeseries")
def garch(
    returns: np.ndarray | pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
) -> dict[str, Any]:
    """Fit a GARCH(p,q) model.

    Uses the ``arch`` library for estimation.  If the library is not
    installed, a pure-numpy GARCH(1,1) fallback is available via
    :func:`garch_numpy_fallback`.

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
    from arch import arch_model

    ret = np.asarray(returns, dtype=float).ravel()

    am = arch_model(ret * 100, vol="GARCH", p=p, q=q, dist=dist, mean="Constant")
    result = am.fit(disp="off")

    forecasts = result.forecast(horizon=1)

    return {
        "params": dict(result.params),
        "conditional_volatility": result.conditional_volatility / 100,
        "standardized_residuals": result.std_resid,
        "forecast": float(np.sqrt(forecasts.variance.iloc[-1].values[0]) / 100),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "loglikelihood": float(result.loglikelihood),
    }


def garch_numpy_fallback(
    returns: np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Fit a GARCH(1,1) model using pure numpy/scipy.

    This is a fallback for when the ``arch`` library is not installed.
    Only supports GARCH(1,1) with normal innovations.

    Parameters:
        returns: Return series.

    Returns:
        Dictionary with ``params``, ``conditional_volatility``,
        ``standardized_residuals``, ``forecast``, and ``loglikelihood``.
    """
    ret = np.asarray(returns, dtype=float).ravel()
    return _fit_garch_numpy(ret)


@requires_extra("timeseries")
def egarch(
    returns: np.ndarray | pd.Series,
    p: int = 1,
    q: int = 1,
) -> dict[str, Any]:
    """Fit an EGARCH(p,q) model (exponential GARCH).

    The EGARCH specification of Nelson (1991) models the log of conditional
    variance, allowing for asymmetric effects of positive and negative
    shocks without requiring parameter constraints for positivity.

    Parameters:
        returns: Return series.
        p: GARCH lag order.
        q: ARCH lag order.

    Returns:
        Dictionary with ``params``, ``conditional_volatility``,
        ``standardized_residuals``, ``forecast``, ``aic``, ``bic``,
        and ``loglikelihood``.
    """
    from arch import arch_model

    ret = np.asarray(returns, dtype=float).ravel()

    am = arch_model(ret * 100, vol="EGARCH", p=p, q=q, mean="Constant")
    result = am.fit(disp="off")

    forecasts = result.forecast(horizon=1)

    return {
        "params": dict(result.params),
        "conditional_volatility": result.conditional_volatility / 100,
        "standardized_residuals": result.std_resid,
        "forecast": float(np.sqrt(forecasts.variance.iloc[-1].values[0]) / 100),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "loglikelihood": float(result.loglikelihood),
    }


@requires_extra("timeseries")
def gjr_garch(
    returns: np.ndarray | pd.Series,
    p: int = 1,
    q: int = 1,
) -> dict[str, Any]:
    """Fit a GJR-GARCH(p,q) model (threshold GARCH).

    The GJR-GARCH model of Glosten, Jagannathan, and Runkle (1993) includes
    an asymmetry term that captures the leverage effect -- negative shocks
    tend to increase volatility more than positive shocks of equal magnitude.

    Parameters:
        returns: Return series.
        p: GARCH lag order.
        q: ARCH lag order.

    Returns:
        Dictionary with ``params``, ``conditional_volatility``,
        ``standardized_residuals``, ``forecast``, ``aic``, ``bic``,
        and ``loglikelihood``.
    """
    from arch import arch_model

    ret = np.asarray(returns, dtype=float).ravel()

    am = arch_model(
        ret * 100, vol="GARCH", p=p, o=1, q=q, mean="Constant"
    )
    result = am.fit(disp="off")

    forecasts = result.forecast(horizon=1)

    return {
        "params": dict(result.params),
        "conditional_volatility": result.conditional_volatility / 100,
        "standardized_residuals": result.std_resid,
        "forecast": float(np.sqrt(forecasts.variance.iloc[-1].values[0]) / 100),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "loglikelihood": float(result.loglikelihood),
    }


@requires_extra("timeseries")
def dcc_garch(
    returns_df: pd.DataFrame,
    p: int = 1,
    q: int = 1,
) -> dict[str, Any]:
    """Fit a Dynamic Conditional Correlation (DCC) GARCH model.

    Estimates univariate GARCH(p,q) models for each series, then fits the
    DCC correlation dynamics on the standardised residuals.  This is the
    two-step approach of Engle (2002).

    Parameters:
        returns_df: DataFrame of return series (T, k), one column per asset.
        p: GARCH lag order for univariate models.
        q: ARCH lag order for univariate models.

    Returns:
        Dictionary with ``univariate_params`` (per-asset GARCH parameters),
        ``conditional_correlations`` (T, k, k), ``conditional_covariances``
        (T, k, k), and ``standardized_residuals`` (T, k).
    """
    from arch import arch_model

    k = returns_df.shape[1]
    T = returns_df.shape[0]
    columns = list(returns_df.columns)

    cond_vols = np.empty((T, k))
    std_resids = np.empty((T, k))
    uni_params: dict[str, dict[str, float]] = {}

    # Step 1: Univariate GARCH for each series
    for i, col in enumerate(columns):
        ret_i = returns_df[col].values * 100
        am = arch_model(ret_i, vol="GARCH", p=p, q=q, mean="Constant")
        result = am.fit(disp="off")

        cond_vols[:, i] = result.conditional_volatility / 100
        std_resids[:, i] = result.std_resid
        uni_params[col] = dict(result.params)

    # Step 2: DCC dynamics on standardised residuals
    # Q_bar = unconditional correlation of standardised residuals
    Q_bar = np.corrcoef(std_resids.T)

    # DCC parameters (simple grid search for a, b)
    best_ll = -np.inf
    best_a, best_b = 0.05, 0.90

    for a_cand in np.linspace(0.01, 0.15, 8):
        for b_cand in np.linspace(0.70, 0.98, 8):
            if a_cand + b_cand >= 1.0:
                continue

            Q_t = Q_bar.copy()
            ll = 0.0
            valid = True

            for t in range(1, T):
                eps = std_resids[t - 1].reshape(-1, 1)
                Q_t = (1 - a_cand - b_cand) * Q_bar + a_cand * (eps @ eps.T) + b_cand * Q_t

                # Standardise to correlation
                d = np.sqrt(np.diag(Q_t))
                if np.any(d <= 0):
                    valid = False
                    break
                R_t = Q_t / np.outer(d, d)

                try:
                    sign, logdet = np.linalg.slogdet(R_t)
                    if sign <= 0:
                        valid = False
                        break
                    R_inv = np.linalg.inv(R_t)
                    e = std_resids[t]
                    ll += -0.5 * (logdet + e @ R_inv @ e - e @ e)
                except np.linalg.LinAlgError:
                    valid = False
                    break

            if valid and ll > best_ll:
                best_ll = ll
                best_a, best_b = a_cand, b_cand

    # Reconstruct with best parameters
    Q_t = Q_bar.copy()
    cond_corr = np.empty((T, k, k))
    cond_cov = np.empty((T, k, k))
    cond_corr[0] = Q_bar
    D0 = np.diag(cond_vols[0])
    cond_cov[0] = D0 @ Q_bar @ D0

    for t in range(1, T):
        eps = std_resids[t - 1].reshape(-1, 1)
        Q_t = (1 - best_a - best_b) * Q_bar + best_a * (eps @ eps.T) + best_b * Q_t

        d = np.sqrt(np.diag(Q_t))
        d = np.where(d <= 0, 1e-10, d)
        R_t = Q_t / np.outer(d, d)
        cond_corr[t] = R_t

        D_t = np.diag(cond_vols[t])
        cond_cov[t] = D_t @ R_t @ D_t

    return {
        "univariate_params": uni_params,
        "conditional_correlations": cond_corr,
        "conditional_covariances": cond_cov,
        "standardized_residuals": std_resids,
        "dcc_params": {"a": float(best_a), "b": float(best_b)},
    }


@requires_extra("timeseries")
def realized_garch(
    returns: np.ndarray | pd.Series,
    realized_vol: np.ndarray | pd.Series,
    p: int = 1,
    q: int = 1,
) -> dict[str, Any]:
    """Fit a Realized GARCH model.

    The Realized GARCH model of Hansen, Huang, and Shek (2012) augments
    the standard GARCH with a measurement equation linking the conditional
    variance to a realised volatility measure, improving forecasting
    performance.

    This implementation includes lagged realized volatility in the mean
    equation as an exogenous regressor and fits a standard GARCH on the
    residuals.  This approximates the Realized GARCH by allowing the
    realized measure to inform the conditional mean.

    Parameters:
        returns: Return series.
        realized_vol: Corresponding realized volatility measure (same length
            as *returns*).
        p: GARCH lag order.
        q: ARCH lag order.

    Returns:
        Dictionary with ``params``, ``conditional_volatility``,
        ``standardized_residuals``, ``forecast``, ``aic``, ``bic``,
        and ``loglikelihood``.
    """
    from arch import arch_model

    ret = np.asarray(returns, dtype=float).ravel() * 100
    rv = np.asarray(realized_vol, dtype=float).ravel() * 100

    # Include lagged realized vol as exogenous variable in the mean equation
    # (arch_model's `x` parameter is for the mean equation)
    rv_lagged = np.empty_like(rv)
    rv_lagged[0] = rv[0]
    rv_lagged[1:] = rv[:-1]

    try:
        am = arch_model(
            ret, vol="GARCH", p=p, q=q, mean="ARX", lags=0,
            x=pd.DataFrame({"rv_lag": rv_lagged}),
        )
        result = am.fit(disp="off")
    except Exception:
        # Fallback: standard GARCH if ARX specification fails
        am = arch_model(ret, vol="GARCH", p=p, q=q, mean="Constant")
        result = am.fit(disp="off")

    forecasts = result.forecast(horizon=1)

    return {
        "params": dict(result.params),
        "conditional_volatility": result.conditional_volatility / 100,
        "standardized_residuals": result.std_resid,
        "forecast": float(np.sqrt(forecasts.variance.iloc[-1].values[0]) / 100),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "loglikelihood": float(result.loglikelihood),
    }
