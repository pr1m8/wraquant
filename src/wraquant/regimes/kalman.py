"""Kalman filter, smoother, and time-varying regression.

This module provides pure-numpy implementations of the Kalman filter,
Rauch-Tung-Striebel smoother, time-varying coefficient regression
(dynamic linear model), and the Unscented Kalman Filter for nonlinear
state estimation.

**When to use each:**

- **kalman_filter**: Linear state-space models where you want filtered
  (causal) state estimates. Core building block.
- **kalman_smoother**: When you have all data and want the best possible
  state estimates (non-causal, uses future data).
- **kalman_regression**: Estimate time-varying betas (e.g., hedge ratios,
  factor exposures) that evolve as random walks.
- **unscented_kalman**: Nonlinear dynamics or observation models where
  the standard Kalman filter's linearity assumption breaks down.

References:
    Durbin, J. & Koopman, S. J. (2012). *Time Series Analysis by State
    Space Methods*. Oxford University Press.

    Haykin, S. (2001). *Kalman Filtering and Neural Networks*. Wiley.

    Julier, S. J. & Uhlmann, J. K. (2004). "Unscented Filtering and
    Nonlinear Estimation." *Proceedings of the IEEE*, 92(3).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from wraquant.core._coerce import coerce_array


# ---------------------------------------------------------------------------
# Kalman Filter
# ---------------------------------------------------------------------------


def kalman_filter(
    observations: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
) -> dict[str, Any]:
    """Run a linear Kalman filter over a sequence of observations.

    The Kalman filter is the optimal recursive estimator for a linear
    Gaussian state-space model:

    .. math::

        x_t = F \\, x_{t-1} + w_t, \\quad w_t \\sim N(0, Q)

        y_t = H \\, x_t + v_t, \\quad v_t \\sim N(0, R)

    At each time step, the filter produces:

    1. **Prediction**: :math:`\\hat{x}_{t|t-1}` and :math:`P_{t|t-1}`
    2. **Update**: :math:`\\hat{x}_{t|t}` and :math:`P_{t|t}` after
       incorporating observation :math:`y_t`

    **Financial applications:**

    - Estimating latent factors or unobserved states
    - Noise reduction (e.g., estimating "true" volatility from noisy proxies)
    - Preprocessing for regime detection

    Parameters:
        observations: Observation matrix of shape ``(T, m)`` where *T* is
            the number of time steps and *m* is the observation dimension.
            For univariate observations, shape ``(T, 1)``.
        F: State transition matrix ``(n, n)``.
        H: Observation matrix ``(m, n)``.
        Q: Process noise covariance ``(n, n)``.
        R: Observation noise covariance ``(m, m)``.
        x0: Initial state estimate ``(n,)`` or ``(n, 1)``.
        P0: Initial state covariance ``(n, n)``.

    Returns:
        Dictionary with:

        - ``filtered_states`` (np.ndarray): Filtered state estimates,
          shape ``(T, n)``.
        - ``filtered_covs`` (np.ndarray): Filtered covariance matrices,
          shape ``(T, n, n)``.
        - ``predicted_states`` (np.ndarray): One-step-ahead predicted
          states, shape ``(T, n)``.
        - ``predicted_covs`` (np.ndarray): One-step-ahead predicted
          covariances, shape ``(T, n, n)``.
        - ``innovations`` (np.ndarray): Innovation (residual) sequence,
          shape ``(T, m)``.
        - ``innovation_covs`` (np.ndarray): Innovation covariances,
          shape ``(T, m, m)``.
        - ``kalman_gains`` (np.ndarray): Kalman gain matrices,
          shape ``(T, n, m)``.
        - ``log_likelihood`` (float): Total log-likelihood computed from
          the innovation sequence.

    Example:
        >>> # Track a random walk with noisy observations
        >>> F = np.array([[1.0]])
        >>> H = np.array([[1.0]])
        >>> Q = np.array([[0.01]])   # Process noise
        >>> R = np.array([[0.1]])    # Observation noise
        >>> x0 = np.array([0.0])
        >>> P0 = np.array([[1.0]])
        >>> obs = np.cumsum(np.random.randn(100, 1) * 0.1) + \\
        ...       np.random.randn(100, 1) * 0.3
        >>> result = kalman_filter(obs, F, H, Q, R, x0, P0)
        >>> print(result['log_likelihood'])
        >>> print(result['filtered_states'][-5:])

    See Also:
        kalman_smoother: Backward pass for optimal smoothed estimates.
        kalman_regression: Time-varying coefficient regression.
        unscented_kalman: Nonlinear state estimation.
    """
    observations = np.asarray(observations, dtype=np.float64)
    if observations.ndim == 1:
        observations = observations.reshape(-1, 1)
    x0 = np.atleast_1d(x0).flatten()
    n = len(x0)
    T = len(observations)
    m = H.shape[0]

    filtered_states = np.zeros((T, n))
    filtered_covs = np.zeros((T, n, n))
    predicted_states = np.zeros((T, n))
    predicted_covs = np.zeros((T, n, n))
    innovations = np.zeros((T, m))
    innovation_covs = np.zeros((T, m, m))
    kalman_gains = np.zeros((T, n, m))

    x = x0.copy()
    P = P0.copy()
    log_likelihood = 0.0

    for t in range(T):
        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        predicted_states[t] = x_pred
        predicted_covs[t] = P_pred

        # Innovation
        y = observations[t] - H @ x_pred
        S = H @ P_pred @ H.T + R

        innovations[t] = y
        innovation_covs[t] = S

        # Log-likelihood contribution: -0.5 * (log|S| + y' S^-1 y + m*log(2pi))
        S_inv = np.linalg.inv(S)
        sign, logdet = np.linalg.slogdet(S)
        log_likelihood += -0.5 * (
            logdet + y @ S_inv @ y + m * np.log(2 * np.pi)
        )

        # Kalman gain
        K = P_pred @ H.T @ S_inv
        kalman_gains[t] = K

        # Update
        x = x_pred + K @ y
        P = (np.eye(n) - K @ H) @ P_pred

        filtered_states[t] = x
        filtered_covs[t] = P

    return {
        "filtered_states": filtered_states,
        "filtered_covs": filtered_covs,
        "predicted_states": predicted_states,
        "predicted_covs": predicted_covs,
        "innovations": innovations,
        "innovation_covs": innovation_covs,
        "kalman_gains": kalman_gains,
        "log_likelihood": float(log_likelihood),
    }


# ---------------------------------------------------------------------------
# Rauch-Tung-Striebel Smoother
# ---------------------------------------------------------------------------


def kalman_smoother(
    observations: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
) -> dict[str, Any]:
    """Run the Kalman filter followed by the RTS backward smoother.

    The Rauch-Tung-Striebel (RTS) smoother uses all observations
    (past and future) to produce the optimal state estimate at each
    time step. The smoothed estimates have lower variance than the
    filtered estimates.

    **When to use:** Offline analysis where you have the complete
    time series and want the best possible state estimates.

    Parameters:
        observations: Observations, shape ``(T, m)``.
        F: State transition matrix ``(n, n)``.
        H: Observation matrix ``(m, n)``.
        Q: Process noise covariance ``(n, n)``.
        R: Observation noise covariance ``(m, m)``.
        x0: Initial state ``(n,)``.
        P0: Initial covariance ``(n, n)``.

    Returns:
        Dictionary with everything from ``kalman_filter`` plus:

        - ``smoothed_states`` (np.ndarray): Smoothed state estimates,
          shape ``(T, n)``.
        - ``smoothed_covs`` (np.ndarray): Smoothed covariance matrices,
          shape ``(T, n, n)``.

    Example:
        >>> result = kalman_smoother(obs, F, H, Q, R, x0, P0)
        >>> # Smoothed states are more accurate than filtered
        >>> print(result['smoothed_states'][-5:])
        >>> print(result['filtered_states'][-5:])

    See Also:
        kalman_filter: Forward pass only.
        kalman_regression: Time-varying betas using DLM.
    """
    filt = kalman_filter(observations, F, H, Q, R, x0, P0)

    T = len(observations)
    n = F.shape[0]

    smoothed_states = np.zeros((T, n))
    smoothed_covs = np.zeros((T, n, n))

    # Initialise from last filtered state
    smoothed_states[-1] = filt["filtered_states"][-1]
    smoothed_covs[-1] = filt["filtered_covs"][-1]

    # Backward pass
    for t in range(T - 2, -1, -1):
        P_filt = filt["filtered_covs"][t]
        P_pred = filt["predicted_covs"][t + 1]

        # Smoother gain
        P_pred_inv = np.linalg.inv(P_pred)
        L = P_filt @ F.T @ P_pred_inv

        x_filt = filt["filtered_states"][t]
        x_pred = filt["predicted_states"][t + 1]

        smoothed_states[t] = x_filt + L @ (
            smoothed_states[t + 1] - x_pred
        )
        smoothed_covs[t] = P_filt + L @ (
            smoothed_covs[t + 1] - P_pred
        ) @ L.T

    result = dict(filt)
    result["smoothed_states"] = smoothed_states
    result["smoothed_covs"] = smoothed_covs
    return result


# ---------------------------------------------------------------------------
# Kalman Regression (Time-Varying Coefficients)
# ---------------------------------------------------------------------------


def kalman_regression(
    y: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
    state_noise: float = 1e-4,
    obs_noise: float | None = None,
    x0: np.ndarray | None = None,
    P0_scale: float = 1.0,
    smooth: bool = True,
) -> dict[str, Any]:
    """Estimate time-varying regression coefficients using a Kalman filter.

    Models the regression coefficients as a random walk:

    .. math::

        \\beta_t = \\beta_{t-1} + w_t, \\quad w_t \\sim N(0, Q)

        y_t = X_t' \\beta_t + v_t, \\quad v_t \\sim N(0, R)

    This is a Dynamic Linear Model (DLM) where the state vector is
    the coefficient vector. Common financial applications:

    - **Time-varying betas**: Estimate CAPM beta that changes over time
    - **Dynamic hedge ratios**: For pairs trading or hedging
    - **Factor exposure monitoring**: Track how factor loadings evolve

    Parameters:
        y: Dependent variable, shape ``(T,)``.
        X: Regressors, shape ``(T, p)``. Include a column of ones for
            an intercept.
        state_noise: Process noise variance for each coefficient.
            Controls how quickly coefficients can change. Larger values
            allow faster adaptation.
        obs_noise: Observation noise variance. If None, estimated as
            the variance of OLS residuals.
        x0: Initial coefficient estimate, shape ``(p,)``. If None,
            OLS estimates are used.
        P0_scale: Scale for initial covariance (``P0 = P0_scale * I``).
        smooth: If True, apply the RTS smoother for better estimates.

    Returns:
        Dictionary with:

        - ``coefficients`` (np.ndarray): Time-varying coefficients,
          shape ``(T, p)``. If ``smooth=True``, these are smoothed.
        - ``coefficient_covs`` (np.ndarray): Covariance of coefficients,
          shape ``(T, p, p)``.
        - ``residuals`` (np.ndarray): Observation residuals, shape ``(T,)``.
        - ``log_likelihood`` (float): Log-likelihood.
        - ``filtered_coefficients`` (np.ndarray): Filtered (causal)
          coefficient estimates, shape ``(T, p)``.

    Example:
        >>> import numpy as np, pandas as pd
        >>> # Time-varying beta estimation
        >>> market = np.random.randn(500) * 0.01
        >>> # True beta changes from 1.0 to 1.5 halfway
        >>> true_beta = np.where(np.arange(500) < 250, 1.0, 1.5)
        >>> stock = true_beta * market + np.random.randn(500) * 0.005
        >>> X = np.column_stack([np.ones(500), market])
        >>> result = kalman_regression(stock, X, state_noise=1e-5)
        >>> betas = result['coefficients'][:, 1]  # Time-varying beta
        >>> print(f"Beta at t=100: {betas[100]:.2f}")  # ~1.0
        >>> print(f"Beta at t=400: {betas[400]:.2f}")  # ~1.5

    Notes:
        The ``state_noise`` parameter is critical. Too large and the
        estimates are noisy; too small and they adapt too slowly.
        A good starting point is ``1e-4`` to ``1e-6`` for daily returns.

    See Also:
        kalman_filter: General-purpose Kalman filter.
        kalman_smoother: RTS backward smoother.
    """
    y_arr = coerce_array(y, "y")
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    T, p = X_arr.shape

    # State-space formulation
    F = np.eye(p)               # Coefficients follow random walk
    Q = np.eye(p) * state_noise  # Process noise

    # Estimate observation noise from OLS if not provided
    if obs_noise is None:
        try:
            beta_ols = np.linalg.lstsq(X_arr, y_arr, rcond=None)[0]
            resid = y_arr - X_arr @ beta_ols
            obs_noise = float(np.var(resid))
        except np.linalg.LinAlgError:
            obs_noise = float(np.var(y_arr))

    R = np.array([[obs_noise]])

    # Initial state
    if x0 is None:
        try:
            x0 = np.linalg.lstsq(X_arr[:min(50, T)], y_arr[:min(50, T)], rcond=None)[0]
        except np.linalg.LinAlgError:
            x0 = np.zeros(p)
    else:
        x0 = np.atleast_1d(x0).flatten()

    P0 = np.eye(p) * P0_scale

    # Run filter with time-varying H
    filtered_states = np.zeros((T, p))
    filtered_covs = np.zeros((T, p, p))
    predicted_states = np.zeros((T, p))
    predicted_covs = np.zeros((T, p, p))
    residuals = np.zeros(T)
    log_likelihood = 0.0

    x = x0.copy()
    P = P0.copy()

    for t in range(T):
        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        predicted_states[t] = x_pred
        predicted_covs[t] = P_pred

        # Time-varying observation matrix
        H_t = X_arr[t : t + 1, :]  # shape (1, p)

        # Innovation
        y_pred = float((H_t @ x_pred).item())
        innovation = y_arr[t] - y_pred
        S = H_t @ P_pred @ H_t.T + R
        S_inv = 1.0 / S[0, 0]

        residuals[t] = innovation

        # Log-likelihood
        log_likelihood += -0.5 * (
            np.log(S[0, 0]) + innovation**2 * S_inv + np.log(2 * np.pi)
        )

        # Kalman gain
        K = P_pred @ H_t.T * S_inv  # shape (p, 1)

        # Update
        x = x_pred + K.flatten() * innovation
        P = (np.eye(p) - K @ H_t) @ P_pred

        filtered_states[t] = x
        filtered_covs[t] = P

    result: dict[str, Any] = {
        "filtered_coefficients": filtered_states,
        "coefficients": filtered_states,
        "coefficient_covs": filtered_covs,
        "residuals": residuals,
        "log_likelihood": float(log_likelihood),
    }

    # Apply RTS smoother if requested
    if smooth and T > 1:
        smoothed_states = np.zeros((T, p))
        smoothed_covs = np.zeros((T, p, p))

        smoothed_states[-1] = filtered_states[-1]
        smoothed_covs[-1] = filtered_covs[-1]

        for t in range(T - 2, -1, -1):
            P_filt = filtered_covs[t]
            P_pred = predicted_covs[t + 1]

            P_pred_inv = np.linalg.inv(P_pred)
            L = P_filt @ F.T @ P_pred_inv

            smoothed_states[t] = filtered_states[t] + L @ (
                smoothed_states[t + 1] - predicted_states[t + 1]
            )
            smoothed_covs[t] = P_filt + L @ (
                smoothed_covs[t + 1] - P_pred
            ) @ L.T

        result["coefficients"] = smoothed_states
        result["coefficient_covs"] = smoothed_covs

    return result


# ---------------------------------------------------------------------------
# Unscented Kalman Filter
# ---------------------------------------------------------------------------


def unscented_kalman(
    observations: np.ndarray,
    state_dim: int,
    f_func: Callable[[np.ndarray], np.ndarray],
    h_func: Callable[[np.ndarray], np.ndarray],
    Q: np.ndarray,
    R: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> dict[str, Any]:
    """Unscented Kalman Filter for nonlinear state estimation.

    The UKF uses sigma points to propagate the state distribution
    through nonlinear dynamics and observation functions, avoiding
    the need for Jacobian computation (unlike the Extended Kalman
    Filter).

    The state-space model:

    .. math::

        x_t = f(x_{t-1}) + w_t, \\quad w_t \\sim N(0, Q)

        y_t = h(x_t) + v_t, \\quad v_t \\sim N(0, R)

    **When to use:** When ``f`` or ``h`` are nonlinear. Examples
    include stochastic volatility models, nonlinear factor models,
    or target tracking with angular observations.

    Parameters:
        observations: Observations, shape ``(T, m)``.
        state_dim: Dimension of the state vector (``n``).
        f_func: State transition function. Takes ``x`` of shape ``(n,)``
            and returns predicted state of shape ``(n,)``.
        h_func: Observation function. Takes ``x`` of shape ``(n,)`` and
            returns predicted observation of shape ``(m,)``.
        Q: Process noise covariance ``(n, n)``.
        R: Observation noise covariance ``(m, m)``.
        x0: Initial state estimate ``(n,)``.
        P0: Initial covariance ``(n, n)``.
        alpha: Spread of sigma points (typically ``1e-4`` to ``1``).
        beta: Prior knowledge of distribution (``2`` is optimal for
            Gaussian).
        kappa: Secondary scaling parameter (``0`` or ``3 - n``).

    Returns:
        Dictionary with:

        - ``filtered_states`` (np.ndarray): Filtered state estimates,
          shape ``(T, n)``.
        - ``filtered_covs`` (np.ndarray): Filtered covariances,
          shape ``(T, n, n)``.
        - ``innovations`` (np.ndarray): Innovation sequence,
          shape ``(T, m)``.
        - ``log_likelihood`` (float): Approximate log-likelihood.

    Example:
        >>> # Nonlinear observation model: observe angle to a target
        >>> def f(x):
        ...     return x  # Random walk dynamics
        >>> def h(x):
        ...     return np.array([np.arctan2(x[1], x[0])])  # Angle
        >>> result = unscented_kalman(
        ...     observations, state_dim=2, f_func=f, h_func=h,
        ...     Q=np.eye(2)*0.01, R=np.array([[0.1]]),
        ...     x0=np.zeros(2), P0=np.eye(2),
        ... )

    Notes:
        The UKF with the standard unscented transform uses
        ``2n + 1`` sigma points. Computational cost scales as
        ``O(n^2)`` per step (vs ``O(n^3)`` for EKF with Jacobian).

    See Also:
        kalman_filter: Linear Kalman filter.
        kalman_smoother: Linear Kalman smoother.
    """
    n = state_dim
    T = len(observations)
    obs = np.atleast_2d(observations)
    if obs.shape[0] == 1 and T > 1:
        obs = obs.T
    m = obs.shape[1]

    # Sigma point weights
    lam = alpha**2 * (n + kappa) - n
    gamma = np.sqrt(n + lam)

    # Weight vectors
    W_m = np.zeros(2 * n + 1)
    W_c = np.zeros(2 * n + 1)
    W_m[0] = lam / (n + lam)
    W_c[0] = lam / (n + lam) + (1 - alpha**2 + beta)
    for i in range(1, 2 * n + 1):
        W_m[i] = 1.0 / (2 * (n + lam))
        W_c[i] = 1.0 / (2 * (n + lam))

    filtered_states = np.zeros((T, n))
    filtered_covs = np.zeros((T, n, n))
    innovations_arr = np.zeros((T, m))
    log_likelihood = 0.0

    x = np.atleast_1d(x0).flatten().copy()
    P = P0.copy()

    for t in range(T):
        # Generate sigma points
        try:
            sqrt_P = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            # If P is not PD, add small diagonal
            P += np.eye(n) * 1e-8
            sqrt_P = np.linalg.cholesky(P)

        sigma_pts = np.zeros((2 * n + 1, n))
        sigma_pts[0] = x
        for i in range(n):
            sigma_pts[i + 1] = x + gamma * sqrt_P[:, i]
            sigma_pts[n + i + 1] = x - gamma * sqrt_P[:, i]

        # Propagate through state transition
        sigma_pts_pred = np.array([f_func(sp) for sp in sigma_pts])

        # Predicted state mean and covariance
        x_pred = W_m @ sigma_pts_pred
        P_pred = Q.copy()
        for i in range(2 * n + 1):
            diff = sigma_pts_pred[i] - x_pred
            P_pred += W_c[i] * np.outer(diff, diff)

        # Propagate through observation function
        obs_sigma = np.array([h_func(sp) for sp in sigma_pts_pred])

        # Predicted observation
        y_pred = W_m @ obs_sigma

        # Innovation covariance and cross-covariance
        S = R.copy()
        P_xy = np.zeros((n, m))
        for i in range(2 * n + 1):
            obs_diff = obs_sigma[i] - y_pred
            state_diff = sigma_pts_pred[i] - x_pred
            S += W_c[i] * np.outer(obs_diff, obs_diff)
            P_xy += W_c[i] * np.outer(state_diff, obs_diff)

        # Innovation
        innovation = obs[t] - y_pred
        innovations_arr[t] = innovation

        # Log-likelihood contribution
        S_inv = np.linalg.inv(S)
        sign, logdet = np.linalg.slogdet(S)
        log_likelihood += -0.5 * (
            logdet + innovation @ S_inv @ innovation + m * np.log(2 * np.pi)
        )

        # Kalman gain
        K = P_xy @ S_inv

        # Update
        x = x_pred + K @ innovation
        P = P_pred - K @ S @ K.T

        filtered_states[t] = x
        filtered_covs[t] = P

    return {
        "filtered_states": filtered_states,
        "filtered_covs": filtered_covs,
        "innovations": innovations_arr,
        "log_likelihood": float(log_likelihood),
    }
