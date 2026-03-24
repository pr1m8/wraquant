"""Dynamic Conditional Correlation (DCC-GARCH) models.

Provides univariate GARCH(1,1) fitting, DCC parameter estimation via MLE,
rolling DCC-based correlations, correlation forecasting, and time-varying
covariance matrix computation.  All implementations use pure numpy/scipy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize

# ---------------------------------------------------------------------------
# Univariate GARCH(1,1)
# ---------------------------------------------------------------------------


def _garch11_loglik(
    params: np.ndarray,
    returns: np.ndarray,
) -> float:
    """Negative log-likelihood for a GARCH(1,1) model.

    sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
    """
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e12

    T = len(returns)
    sigma2 = np.empty(T)
    sigma2[0] = np.var(returns)

    for t in range(1, T):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        if sigma2[t] <= 0:
            return 1e12

    # Gaussian log-likelihood (ignoring constant)
    ll = -0.5 * np.sum(np.log(sigma2) + returns**2 / sigma2)
    return -ll


def _fit_garch11(returns: np.ndarray) -> dict[str, Any]:
    """Fit univariate GARCH(1,1) via MLE.

    Returns:
        Dict with ``"omega"``, ``"alpha"``, ``"beta"``,
        ``"conditional_vol"`` (array of sigma_t).
    """
    var0 = np.var(returns)
    x0 = np.array([var0 * 0.05, 0.05, 0.90])

    res = optimize.minimize(
        _garch11_loglik,
        x0,
        args=(returns,),
        method="Nelder-Mead",
        options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8},
    )
    omega, alpha, beta = res.x
    omega = max(omega, 1e-10)
    alpha = max(alpha, 1e-6)
    beta = max(beta, 1e-6)
    # Re-normalise if needed
    if alpha + beta >= 1.0:
        s = alpha + beta
        alpha = alpha / s * 0.999
        beta = beta / s * 0.999

    T = len(returns)
    sigma2 = np.empty(T)
    sigma2[0] = np.var(returns)
    for t in range(1, T):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]

    return {
        "omega": float(omega),
        "alpha": float(alpha),
        "beta": float(beta),
        "conditional_vol": np.sqrt(sigma2),
    }


# ---------------------------------------------------------------------------
# DCC estimation
# ---------------------------------------------------------------------------


def _dcc_loglik(
    params: np.ndarray,
    std_residuals: np.ndarray,
    qbar: np.ndarray,
) -> float:
    """Negative log-likelihood for DCC parameters (a, b).

    Q_t = (1 - a - b) * Qbar + a * e_{t-1} e_{t-1}' + b * Q_{t-1}
    R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
    """
    a, b = params
    if a < 0 or b < 0 or a + b >= 1:
        return 1e12

    T, k = std_residuals.shape
    Qt = qbar.copy()
    ll = 0.0
    c = 1 - a - b

    for t in range(1, T):
        et_prev = std_residuals[t - 1].reshape(-1, 1)
        Qt = c * qbar + a * (et_prev @ et_prev.T) + b * Qt

        # Correlation matrix from Qt
        d = np.sqrt(np.diag(Qt))
        if np.any(d <= 0):
            return 1e12
        D_inv = np.diag(1.0 / d)
        Rt = D_inv @ Qt @ D_inv

        # Ensure positive definite
        det_Rt = np.linalg.det(Rt)
        if det_Rt <= 0:
            return 1e12

        et = std_residuals[t]
        quad_form = et @ np.linalg.solve(Rt, et) - et @ et
        ll += -0.5 * (np.log(det_Rt) + quad_form)

    return -ll


def dcc_garch(
    returns: np.ndarray,
    p: int = 1,
    q: int = 1,
) -> dict[str, Any]:
    """Fit a DCC-GARCH(p, q) model.

    Currently supports p=1, q=1 (DCC(1,1)).

    Procedure:

    1. Fit univariate GARCH(1,1) to each return series.
    2. Compute standardized residuals.
    3. Estimate DCC parameters (a, b) via MLE.

    Parameters:
        returns: Array of shape ``(T, k)`` with k asset return series.
        p: DCC lag order for innovation (currently must be 1).
        q: DCC lag order for conditional (currently must be 1).

    Returns:
        Dict with keys:

        * ``"a"`` -- DCC innovation parameter.
        * ``"b"`` -- DCC persistence parameter.
        * ``"garch_params"`` -- list of per-asset GARCH(1,1) parameter
          dicts.
        * ``"qbar"`` -- unconditional correlation matrix of standardized
          residuals.
        * ``"conditional_vols"`` -- array of per-asset conditional
          volatilities ``(T, k)``.
        * ``"std_residuals"`` -- standardized residuals ``(T, k)``.
    """
    from wraquant.core._coerce import coerce_array

    if p != 1 or q != 1:
        msg = "Only DCC(1,1) is currently supported"
        raise ValueError(msg)

    returns = np.asarray(returns, dtype=np.float64)
    T, k = returns.shape

    # Step 1: fit univariate GARCH to each series
    garch_results = []
    cond_vols = np.empty((T, k))
    std_resids = np.empty((T, k))

    for j in range(k):
        g = _fit_garch11(returns[:, j])
        garch_results.append(g)
        cond_vols[:, j] = g["conditional_vol"]
        std_resids[:, j] = returns[:, j] / g["conditional_vol"]

    # Step 2: unconditional correlation of standardized residuals
    qbar = np.corrcoef(std_resids, rowvar=False)

    # Step 3: estimate DCC parameters
    res = optimize.minimize(
        _dcc_loglik,
        x0=np.array([0.01, 0.95]),
        args=(std_resids, qbar),
        method="Nelder-Mead",
        options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8},
    )
    a_hat, b_hat = res.x
    a_hat = max(float(a_hat), 1e-6)
    b_hat = max(float(b_hat), 1e-6)
    if a_hat + b_hat >= 1.0:
        s = a_hat + b_hat
        a_hat = a_hat / s * 0.999
        b_hat = b_hat / s * 0.999

    return {
        "a": a_hat,
        "b": b_hat,
        "garch_params": [
            {"omega": g["omega"], "alpha": g["alpha"], "beta": g["beta"]}
            for g in garch_results
        ],
        "qbar": qbar,
        "conditional_vols": cond_vols,
        "std_residuals": std_resids,
    }


# ---------------------------------------------------------------------------
# Rolling correlation
# ---------------------------------------------------------------------------


def rolling_correlation_dcc(
    returns: np.ndarray,
    window: int | None = None,
) -> dict[str, Any]:
    """Compute DCC-based time-varying correlation matrices.

    Fits DCC-GARCH to the full sample, then extracts the time-varying
    correlation matrix at each time step.

    Parameters:
        returns: Array of shape ``(T, k)``.
        window: Ignored (present for API compatibility with simple
            rolling-window methods).  DCC uses the full sample.

    Returns:
        Dict with keys:

        * ``"correlations"`` -- array of shape ``(T, k, k)`` with the
          time-varying correlation matrices.
        * ``"dcc_model"`` -- the fitted DCC model dict.
    """
    from wraquant.core._coerce import coerce_array

    returns = np.asarray(returns, dtype=np.float64)
    model = dcc_garch(returns)
    corrs = _compute_dcc_correlations(model)

    return {
        "correlations": corrs,
        "dcc_model": model,
    }


def _compute_dcc_correlations(model: dict[str, Any]) -> np.ndarray:
    """Extract time-varying correlation matrices from a fitted DCC model."""
    a = model["a"]
    b = model["b"]
    qbar = model["qbar"]
    std_resids = model["std_residuals"]
    T, k = std_resids.shape
    c = 1 - a - b

    correlations = np.empty((T, k, k))
    Qt = qbar.copy()
    correlations[0] = qbar.copy()

    for t in range(1, T):
        et_prev = std_resids[t - 1].reshape(-1, 1)
        Qt = c * qbar + a * (et_prev @ et_prev.T) + b * Qt

        d = np.sqrt(np.diag(Qt))
        D_inv = np.diag(1.0 / np.maximum(d, 1e-10))
        Rt = D_inv @ Qt @ D_inv
        # Clip to valid correlation range
        np.clip(Rt, -1, 1, out=Rt)
        np.fill_diagonal(Rt, 1.0)
        correlations[t] = Rt

    return correlations


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------


def forecast_correlation(
    dcc_model: dict[str, Any],
    horizon: int = 1,
) -> dict[str, Any]:
    """Forecast future correlation matrices from a fitted DCC model.

    Uses the mean-reverting property of DCC:
    ``E[Q_{T+h}] -> Qbar`` as ``h -> inf``.  For finite horizons the
    forecasted Q is computed recursively assuming that future
    innovations have zero outer product (their expectation).

    Parameters:
        dcc_model: Output of :func:`dcc_garch`.
        horizon: Number of steps ahead to forecast.

    Returns:
        Dict with keys:

        * ``"forecasted_correlations"`` -- array of shape
          ``(horizon, k, k)``.
        * ``"forecasted_covariances"`` -- array of shape
          ``(horizon, k, k)``.
    """
    a = dcc_model["a"]
    b = dcc_model["b"]
    qbar = dcc_model["qbar"]
    std_resids = dcc_model["std_residuals"]
    cond_vols = dcc_model["conditional_vols"]
    garch_params = dcc_model["garch_params"]
    T, k = std_resids.shape
    c = 1 - a - b

    # Last Qt
    Qt = qbar.copy()
    for t in range(1, T):
        et_prev = std_resids[t - 1].reshape(-1, 1)
        Qt = c * qbar + a * (et_prev @ et_prev.T) + b * Qt

    # Forecast GARCH volatilities
    last_sigma2 = cond_vols[-1] ** 2
    last_r2 = (std_resids[-1] * cond_vols[-1]) ** 2  # original returns squared

    forecasted_corrs = np.empty((horizon, k, k))
    forecasted_covs = np.empty((horizon, k, k))

    sigma2_forecast = last_sigma2.copy()

    for h in range(horizon):
        # GARCH vol forecast: sigma2_{T+h} = omega + (alpha+beta)*sigma2_{T+h-1}
        for j in range(k):
            gp = garch_params[j]
            if h == 0:
                sigma2_forecast[j] = (
                    gp["omega"] + gp["alpha"] * last_r2[j] + gp["beta"] * last_sigma2[j]
                )
            else:
                sigma2_forecast[j] = (
                    gp["omega"] + (gp["alpha"] + gp["beta"]) * sigma2_forecast[j]
                )

        # DCC correlation forecast: Qt -> Qbar
        # E[e_{t} e_{t}'] = Qbar (under stationarity)
        Qt = c * qbar + a * qbar + b * Qt  # = (c+a)*qbar + b*Qt = (1-b)*qbar + b*Qt

        d = np.sqrt(np.diag(Qt))
        D_inv = np.diag(1.0 / np.maximum(d, 1e-10))
        Rt = D_inv @ Qt @ D_inv
        np.clip(Rt, -1, 1, out=Rt)
        np.fill_diagonal(Rt, 1.0)

        forecasted_corrs[h] = Rt

        # Covariance = D_sigma * R * D_sigma
        D_sigma = np.diag(np.sqrt(np.maximum(sigma2_forecast, 1e-15)))
        forecasted_covs[h] = D_sigma @ Rt @ D_sigma

    return {
        "forecasted_correlations": forecasted_corrs,
        "forecasted_covariances": forecasted_covs,
    }


# ---------------------------------------------------------------------------
# Conditional covariance
# ---------------------------------------------------------------------------


def conditional_covariance(
    returns: np.ndarray,
    dcc_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute time-varying covariance matrices from DCC-GARCH.

    If *dcc_params* is not supplied, the model is fitted to *returns*
    automatically.

    Parameters:
        returns: Array of shape ``(T, k)``.
        dcc_params: Pre-fitted DCC model dict (optional).

    Returns:
        Dict with keys:

        * ``"covariances"`` -- array of shape ``(T, k, k)`` with
          conditional covariance matrices.
        * ``"correlations"`` -- array of shape ``(T, k, k)`` with
          conditional correlation matrices.
        * ``"volatilities"`` -- array of shape ``(T, k)`` with
          conditional volatilities.
    """
    from wraquant.core._coerce import coerce_array

    returns = np.asarray(returns, dtype=np.float64)
    if dcc_params is None:
        dcc_params = dcc_garch(returns)

    cond_vols = dcc_params["conditional_vols"]
    corrs = _compute_dcc_correlations(dcc_params)
    T, k = returns.shape

    covs = np.empty((T, k, k))
    for t in range(T):
        D = np.diag(cond_vols[t])
        covs[t] = D @ corrs[t] @ D

    return {
        "covariances": covs,
        "correlations": corrs,
        "volatilities": cond_vols,
    }
