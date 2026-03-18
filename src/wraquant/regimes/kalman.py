"""Basic Kalman filter implementation using numpy only."""

from __future__ import annotations

import numpy as np


def kalman_filter(
    observations: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a linear Kalman filter over a sequence of observations.

    Parameters:
        observations: Observation matrix of shape ``(T, m)`` where *T* is
            the number of time steps and *m* is the observation dimension.
        F: State transition matrix ``(n, n)``.
        H: Observation matrix ``(m, n)``.
        Q: Process noise covariance ``(n, n)``.
        R: Observation noise covariance ``(m, m)``.
        x0: Initial state estimate ``(n,)`` or ``(n, 1)``.
        P0: Initial state covariance ``(n, n)``.

    Returns:
        Tuple of ``(filtered_states, filtered_covs)`` where
        ``filtered_states`` has shape ``(T, n)`` and
        ``filtered_covs`` has shape ``(T, n, n)``.
    """
    x0 = np.atleast_1d(x0).flatten()
    n = len(x0)
    T = len(observations)

    filtered_states = np.zeros((T, n))
    filtered_covs = np.zeros((T, n, n))

    x = x0.copy()
    P = P0.copy()

    for t in range(T):
        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # Update
        y = observations[t] - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x = x_pred + K @ y
        P = (np.eye(n) - K @ H) @ P_pred

        filtered_states[t] = x
        filtered_covs[t] = P

    return filtered_states, filtered_covs
