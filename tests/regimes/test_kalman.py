"""Tests for Kalman filter."""

from __future__ import annotations

import numpy as np

from wraquant.regimes.kalman import kalman_filter


class TestKalmanFilter:
    def test_output_shapes(self) -> None:
        T = 50
        F = np.eye(2)
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.01
        R = np.eye(1) * 0.1
        x0 = np.zeros(2)
        P0 = np.eye(2)
        obs = np.random.default_rng(42).normal(size=(T, 1))

        states, covs = kalman_filter(obs, F, H, Q, R, x0, P0)
        assert states.shape == (T, 2)
        assert covs.shape == (T, 2, 2)

    def test_constant_observation(self) -> None:
        """Filter on constant observations should converge to the constant."""
        T = 100
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.001]])
        R = np.array([[0.1]])
        x0 = np.array([0.0])
        P0 = np.array([[1.0]])
        obs = np.full((T, 1), 5.0)

        states, _covs = kalman_filter(obs, F, H, Q, R, x0, P0)
        # After many steps, the filter should converge close to 5.0
        np.testing.assert_allclose(states[-1, 0], 5.0, atol=0.5)

    def test_covariance_positive_definite(self) -> None:
        T = 30
        F = np.eye(2)
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.01
        R = np.eye(1) * 0.1
        x0 = np.zeros(2)
        P0 = np.eye(2)
        obs = np.random.default_rng(42).normal(size=(T, 1))

        _states, covs = kalman_filter(obs, F, H, Q, R, x0, P0)
        for t in range(T):
            eigvals = np.linalg.eigvalsh(covs[t])
            assert (eigvals >= 0).all(), f"Non-PSD covariance at t={t}"

    def test_tracks_linear_trend(self) -> None:
        """Filter should track a linear trend observation."""
        T = 100
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.01]])
        R = np.array([[0.01]])
        x0 = np.array([0.0])
        P0 = np.array([[1.0]])

        rng = np.random.default_rng(42)
        trend = np.linspace(0, 10, T) + rng.normal(0, 0.1, T)
        obs = trend.reshape(-1, 1)

        states, _covs = kalman_filter(obs, F, H, Q, R, x0, P0)
        # Should track the trend fairly well at the end
        np.testing.assert_allclose(states[-1, 0], trend[-1], atol=1.0)
