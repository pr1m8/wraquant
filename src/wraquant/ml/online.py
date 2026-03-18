"""Online (streaming) machine learning for quantitative finance.

Provides recursive and weighted regression algorithms that update
incrementally with each new observation, enabling real-time tracking of
time-varying relationships in financial data.

These algorithms require only numpy and pandas -- no optional dependencies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "online_linear_regression",
    "exponential_weighted_regression",
]


# ---------------------------------------------------------------------------
# Online Linear Regression (Recursive Least Squares)
# ---------------------------------------------------------------------------


def online_linear_regression(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    forgetting_factor: float = 1.0,
    initial_covariance: float = 100.0,
) -> dict[str, Any]:
    """Recursive Least Squares (RLS) online linear regression.

    Processes observations one at a time, updating regression coefficients
    with each new data point. This is the online analogue of ordinary least
    squares and is fundamental to adaptive signal processing in finance:
    tracking time-varying betas, hedge ratios, and factor loadings.

    When to use:
        Use online regression when you need to:
        - Track a hedge ratio that drifts over time (pairs trading).
        - Estimate time-varying factor exposures (rolling beta).
        - Build adaptive trading signals that respond to regime changes.
        - Process streaming tick data without re-estimating from scratch.

    Mathematical background:
        Recursive Least Squares maintains:
            P_t = (1/lambda) * (P_{t-1} - K_t x_t^T P_{t-1})
            K_t = P_{t-1} x_t / (lambda + x_t^T P_{t-1} x_t)
            w_t = w_{t-1} + K_t (y_t - x_t^T w_{t-1})

        where:
        - w_t is the coefficient vector at time t
        - P_t is the inverse covariance matrix (precision)
        - K_t is the Kalman gain
        - lambda is the forgetting factor (1 = no forgetting, <1 = down-weight old data)

        With lambda = 1 and infinite data, RLS converges to OLS. With
        lambda < 1, the effective window length is approximately
        1 / (1 - lambda) observations.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix of shape ``(T, p)`` where T is the number of
        observations and p is the number of features.
    y : pd.Series or np.ndarray
        Target vector of length T.
    forgetting_factor : float
        Forgetting factor lambda in (0, 1]. Values close to 1 give long
        memory; values like 0.99 give an effective window of ~100
        observations. Use 0.95-0.99 for fast-adapting signals.
    initial_covariance : float
        Scalar multiplier for the initial covariance matrix P_0 = c * I.
        Larger values make the filter more responsive early on.

    Returns
    -------
    dict
        ``coefficients``: np.ndarray of shape ``(T, p)`` -- the
        time-varying coefficient vector at each step,
        ``predictions``: np.ndarray of shape ``(T,)`` -- one-step-ahead
        predictions (each y_hat_t uses coefficients estimated from data
        up to t-1),
        ``residuals``: np.ndarray of shape ``(T,)`` -- prediction errors,
        ``final_coefficients``: np.ndarray of shape ``(p,)`` -- the
        coefficients at the last time step.

    Example
    -------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> T = 500
    >>> X = np.random.randn(T, 2)
    >>> # True coefficients shift halfway through
    >>> beta_true = np.where(np.arange(T)[:, None] < 250,
    ...     [1.0, 0.5], [0.5, 1.0])
    >>> y = np.sum(X * beta_true, axis=1) + np.random.randn(T) * 0.1
    >>> result = online_linear_regression(X, y, forgetting_factor=0.98)
    >>> result["coefficients"].shape
    (500, 2)
    >>> # After convergence, coefficients should track the true values
    >>> np.abs(result["final_coefficients"][0] - 0.5) < 0.3
    True

    Caveats
    -------
    - The forgetting factor is critical: too low causes noisy estimates,
      too high causes slow adaptation to regime changes.
    - RLS assumes the noise variance is constant; for heteroskedastic
      data, consider the exponential weighted variant or Kalman filters.
    - Initial predictions (before the filter converges) should be
      discarded in any evaluation.

    References
    ----------
    - Haykin (2002), "Adaptive Filter Theory", Ch. 13 (RLS)
    - Montana et al. (2009), "Flexible least squares for temporal data
      mining and statistical arbitrage"
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).ravel()

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    T, p = X_arr.shape
    lam = forgetting_factor

    # Initialise
    w = np.zeros(p, dtype=np.float64)
    P = np.eye(p, dtype=np.float64) * initial_covariance

    coefficients = np.zeros((T, p), dtype=np.float64)
    predictions = np.zeros(T, dtype=np.float64)
    residuals = np.zeros(T, dtype=np.float64)

    for t in range(T):
        x_t = X_arr[t]  # (p,)

        # One-step-ahead prediction using current coefficients
        y_hat = x_t @ w
        predictions[t] = y_hat
        residuals[t] = y_arr[t] - y_hat

        # Kalman gain
        Px = P @ x_t  # (p,)
        denom = lam + x_t @ Px
        K = Px / denom  # (p,)

        # Update coefficients
        w = w + K * residuals[t]

        # Update inverse covariance
        P = (P - np.outer(K, x_t @ P)) / lam

        coefficients[t] = w.copy()

    return {
        "coefficients": coefficients,
        "predictions": predictions,
        "residuals": residuals,
        "final_coefficients": coefficients[-1].copy(),
    }


# ---------------------------------------------------------------------------
# Exponential Weighted Regression
# ---------------------------------------------------------------------------


def exponential_weighted_regression(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    halflife: float = 63.0,
    min_periods: int = 30,
) -> dict[str, Any]:
    """Exponentially weighted linear regression favouring recent data.

    At each time step t, fits a weighted least squares regression where
    observation weights decay exponentially into the past. This produces
    smooth, adaptive coefficient estimates that naturally respond to
    regime changes without the abrupt sensitivity of rolling-window OLS.

    When to use:
        Use exponential weighted regression when:
        - You want smoother coefficient paths than RLS.
        - The halflife of predictive relationships is approximately known
          (e.g., 63 trading days ~ 3 months).
        - You need an interpretable "recency bias" in your factor model.

    Mathematical background:
        At time t, the weight for observation s (where s <= t) is:
            w_s = exp(-ln(2) * (t - s) / halflife)

        The weighted regression solves:
            beta_t = (X_t^T W_t X_t)^{-1} X_t^T W_t y_t

        where W_t = diag(w_0, w_1, ..., w_t). This is equivalent to
        EWMA smoothing of the sufficient statistics X^T X and X^T y.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix of shape ``(T, p)``.
    y : pd.Series or np.ndarray
        Target vector of length T.
    halflife : float
        Halflife in observations. After ``halflife`` observations, the
        weight of a past data point has decayed to 50%. Common financial
        values: 21 (1 month), 63 (1 quarter), 252 (1 year).
    min_periods : int
        Minimum number of observations before producing a coefficient
        estimate. Earlier entries are filled with NaN.

    Returns
    -------
    dict
        ``coefficients``: np.ndarray of shape ``(T, p)`` -- time-varying
        coefficients (NaN for the first ``min_periods - 1`` rows),
        ``predictions``: np.ndarray of shape ``(T,)`` -- fitted values
        using contemporaneous coefficients,
        ``residuals``: np.ndarray of shape ``(T,)`` -- prediction errors,
        ``final_coefficients``: np.ndarray of shape ``(p,)`` -- last
        estimated coefficients.

    Example
    -------
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> T = 300
    >>> X = np.random.randn(T, 2)
    >>> beta_true = np.column_stack([
    ...     np.linspace(1, 0, T),      # drifting coefficient
    ...     np.full(T, 0.5),            # constant coefficient
    ... ])
    >>> y = np.sum(X * beta_true, axis=1) + np.random.randn(T) * 0.1
    >>> result = exponential_weighted_regression(X, y, halflife=60)
    >>> result["coefficients"].shape
    (300, 2)

    Caveats
    -------
    - Halflife selection is subjective; cross-validate if possible.
    - For very short halflives (<10), the effective sample size is small
      and estimates become noisy.
    - Assumes homoskedastic errors; for heteroskedastic data, consider
      EWMA-weighted robust regression.
    - Numerically less stable than RLS for ill-conditioned problems.

    References
    ----------
    - Pozzi et al. (2012), "Exponentially weighted moving average charts
      for detecting concept drift"
    - de Prado (2018), "Advances in Financial Machine Learning", Ch. 17
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).ravel()

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    T, p = X_arr.shape
    decay = np.log(2.0) / halflife

    coefficients = np.full((T, p), np.nan, dtype=np.float64)
    predictions = np.full(T, np.nan, dtype=np.float64)
    residuals = np.full(T, np.nan, dtype=np.float64)

    for t in range(min_periods - 1, T):
        # Weights for observations 0..t
        ages = np.arange(t, -1, -1, dtype=np.float64)  # t, t-1, ..., 0
        weights = np.exp(-decay * ages)

        X_t = X_arr[: t + 1]
        y_t = y_arr[: t + 1]

        # Weighted normal equations: (X^T W X) beta = X^T W y
        W_sqrt = np.sqrt(weights)
        Xw = X_t * W_sqrt[:, None]
        yw = y_t * W_sqrt

        XtX = Xw.T @ Xw
        Xty = Xw.T @ yw

        try:
            beta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            # Singular matrix -- use pseudoinverse
            beta = np.linalg.lstsq(XtX, Xty, rcond=None)[0]

        coefficients[t] = beta
        predictions[t] = X_arr[t] @ beta
        residuals[t] = y_arr[t] - predictions[t]

    last_valid = coefficients[~np.isnan(coefficients[:, 0])]
    final = last_valid[-1] if len(last_valid) > 0 else np.full(p, np.nan)

    return {
        "coefficients": coefficients,
        "predictions": predictions,
        "residuals": residuals,
        "final_coefficients": final,
    }
