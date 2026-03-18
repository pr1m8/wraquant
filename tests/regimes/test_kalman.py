"""Tests for Kalman filter, smoother, regression, and UKF.

Tests verify correct shapes, mathematical properties (e.g., positive
definite covariances, convergence to known values), and ability to
track known linear systems.
"""

from __future__ import annotations

import numpy as np
import pytest

from wraquant.regimes.kalman import (
    kalman_filter,
    kalman_regression,
    kalman_smoother,
    unscented_kalman,
)


# ---------------------------------------------------------------------------
# Tests: kalman_filter
# ---------------------------------------------------------------------------


class TestKalmanFilter:
    def test_output_keys(self) -> None:
        T = 50
        F = np.eye(2)
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.01
        R = np.eye(1) * 0.1
        x0 = np.zeros(2)
        P0 = np.eye(2)
        obs = np.random.default_rng(42).normal(size=(T, 1))

        result = kalman_filter(obs, F, H, Q, R, x0, P0)
        expected_keys = {
            "filtered_states", "filtered_covs",
            "predicted_states", "predicted_covs",
            "innovations", "innovation_covs",
            "kalman_gains", "log_likelihood",
        }
        assert expected_keys.issubset(result.keys())

    def test_output_shapes(self) -> None:
        T = 50
        n = 2
        m = 1
        F = np.eye(n)
        H = np.array([[1.0, 0.0]])
        Q = np.eye(n) * 0.01
        R = np.eye(m) * 0.1
        x0 = np.zeros(n)
        P0 = np.eye(n)
        obs = np.random.default_rng(42).normal(size=(T, m))

        result = kalman_filter(obs, F, H, Q, R, x0, P0)
        assert result["filtered_states"].shape == (T, n)
        assert result["filtered_covs"].shape == (T, n, n)
        assert result["predicted_states"].shape == (T, n)
        assert result["predicted_covs"].shape == (T, n, n)
        assert result["innovations"].shape == (T, m)
        assert result["innovation_covs"].shape == (T, m, m)
        assert result["kalman_gains"].shape == (T, n, m)

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

        result = kalman_filter(obs, F, H, Q, R, x0, P0)
        np.testing.assert_allclose(
            result["filtered_states"][-1, 0], 5.0, atol=0.5
        )

    def test_covariance_positive_definite(self) -> None:
        T = 30
        F = np.eye(2)
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.01
        R = np.eye(1) * 0.1
        x0 = np.zeros(2)
        P0 = np.eye(2)
        obs = np.random.default_rng(42).normal(size=(T, 1))

        result = kalman_filter(obs, F, H, Q, R, x0, P0)
        for t in range(T):
            eigvals = np.linalg.eigvalsh(result["filtered_covs"][t])
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

        result = kalman_filter(obs, F, H, Q, R, x0, P0)
        np.testing.assert_allclose(
            result["filtered_states"][-1, 0], trend[-1], atol=1.0
        )

    def test_log_likelihood_finite(self) -> None:
        T = 50
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.01]])
        R = np.array([[0.1]])
        x0 = np.array([0.0])
        P0 = np.array([[1.0]])
        obs = np.random.default_rng(42).normal(size=(T, 1))

        result = kalman_filter(obs, F, H, Q, R, x0, P0)
        assert np.isfinite(result["log_likelihood"])

    def test_known_linear_system(self) -> None:
        """Test against a known linear system where true state is known."""
        rng = np.random.default_rng(42)
        T = 200
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        q = 0.001
        r = 0.1
        Q = np.array([[q]])
        R = np.array([[r]])

        # Generate true states and observations
        true_state = np.zeros(T)
        observations = np.zeros((T, 1))
        true_state[0] = 0.0
        observations[0, 0] = true_state[0] + rng.normal(0, np.sqrt(r))
        for t in range(1, T):
            true_state[t] = true_state[t - 1] + rng.normal(0, np.sqrt(q))
            observations[t, 0] = true_state[t] + rng.normal(0, np.sqrt(r))

        result = kalman_filter(
            observations, F, H, Q, R,
            x0=np.array([0.0]),
            P0=np.array([[1.0]]),
        )

        # Filtered states should be close to true states (RMSE check)
        rmse = np.sqrt(
            np.mean((result["filtered_states"][:, 0] - true_state) ** 2)
        )
        # RMSE should be less than the observation noise std
        assert rmse < np.sqrt(r), f"RMSE {rmse} >= observation noise {np.sqrt(r)}"

    def test_innovations_zero_mean_asymptotically(self) -> None:
        """Innovations should be approximately zero-mean if model is correct."""
        rng = np.random.default_rng(42)
        T = 500
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.01]])
        R = np.array([[0.1]])

        true_x = np.cumsum(rng.normal(0, 0.1, T))
        obs = (true_x + rng.normal(0, np.sqrt(0.1), T)).reshape(-1, 1)

        result = kalman_filter(
            obs, F, H, Q, R,
            x0=np.array([0.0]),
            P0=np.array([[1.0]]),
        )

        # Skip first 50 for transient
        mean_innov = np.mean(result["innovations"][50:])
        assert abs(mean_innov) < 0.1, f"Mean innovation {mean_innov} not near zero"


# ---------------------------------------------------------------------------
# Tests: kalman_smoother
# ---------------------------------------------------------------------------


class TestKalmanSmoother:
    def test_output_keys(self) -> None:
        T = 50
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.01]])
        R = np.array([[0.1]])
        x0 = np.array([0.0])
        P0 = np.array([[1.0]])
        obs = np.random.default_rng(42).normal(size=(T, 1))

        result = kalman_smoother(obs, F, H, Q, R, x0, P0)
        assert "smoothed_states" in result
        assert "smoothed_covs" in result
        assert "filtered_states" in result

    def test_smoothed_shapes(self) -> None:
        T = 50
        n = 2
        F = np.eye(n)
        H = np.array([[1.0, 0.0]])
        Q = np.eye(n) * 0.01
        R = np.eye(1) * 0.1
        x0 = np.zeros(n)
        P0 = np.eye(n)
        obs = np.random.default_rng(42).normal(size=(T, 1))

        result = kalman_smoother(obs, F, H, Q, R, x0, P0)
        assert result["smoothed_states"].shape == (T, n)
        assert result["smoothed_covs"].shape == (T, n, n)

    def test_smoother_has_lower_variance(self) -> None:
        """Smoothed estimates should have lower or equal variance than filtered."""
        T = 100
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.01]])
        R = np.array([[0.5]])
        x0 = np.array([0.0])
        P0 = np.array([[1.0]])
        obs = np.random.default_rng(42).normal(size=(T, 1))

        result = kalman_smoother(obs, F, H, Q, R, x0, P0)

        # Compare trace of covariances (skip first and last for boundary)
        for t in range(5, T - 5):
            filt_var = np.trace(result["filtered_covs"][t])
            smooth_var = np.trace(result["smoothed_covs"][t])
            assert smooth_var <= filt_var + 1e-10, (
                f"Smoothed variance {smooth_var} > filtered {filt_var} at t={t}"
            )

    def test_last_step_matches_filter(self) -> None:
        """At the last time step, smoother = filter (no future data)."""
        T = 50
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.01]])
        R = np.array([[0.1]])
        x0 = np.array([0.0])
        P0 = np.array([[1.0]])
        obs = np.random.default_rng(42).normal(size=(T, 1))

        result = kalman_smoother(obs, F, H, Q, R, x0, P0)

        np.testing.assert_allclose(
            result["smoothed_states"][-1],
            result["filtered_states"][-1],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            result["smoothed_covs"][-1],
            result["filtered_covs"][-1],
            atol=1e-10,
        )

    def test_smoother_better_rmse(self) -> None:
        """Smoother should have lower RMSE than filter for known system."""
        rng = np.random.default_rng(42)
        T = 200
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.01]])
        R = np.array([[0.5]])

        true_state = np.cumsum(rng.normal(0, 0.1, T))
        obs = (true_state + rng.normal(0, np.sqrt(0.5), T)).reshape(-1, 1)

        result = kalman_smoother(
            obs, F, H, Q, R,
            x0=np.array([0.0]),
            P0=np.array([[1.0]]),
        )

        filt_rmse = np.sqrt(
            np.mean((result["filtered_states"][:, 0] - true_state) ** 2)
        )
        smooth_rmse = np.sqrt(
            np.mean((result["smoothed_states"][:, 0] - true_state) ** 2)
        )
        assert smooth_rmse <= filt_rmse + 0.01


# ---------------------------------------------------------------------------
# Tests: kalman_regression
# ---------------------------------------------------------------------------


class TestKalmanRegression:
    def test_output_keys(self) -> None:
        rng = np.random.default_rng(42)
        T = 100
        x = rng.normal(0, 1, T)
        y = 0.5 * x + rng.normal(0, 0.1, T)
        X = np.column_stack([np.ones(T), x])

        result = kalman_regression(y, X)
        expected_keys = {
            "coefficients", "coefficient_covs", "residuals",
            "log_likelihood", "filtered_coefficients",
        }
        assert expected_keys.issubset(result.keys())

    def test_output_shapes(self) -> None:
        rng = np.random.default_rng(42)
        T = 100
        p = 3
        X = rng.normal(0, 1, (T, p))
        y = X @ np.array([1, 2, 3]) + rng.normal(0, 0.1, T)

        result = kalman_regression(y, X)
        assert result["coefficients"].shape == (T, p)
        assert result["coefficient_covs"].shape == (T, p, p)
        assert result["residuals"].shape == (T,)

    def test_constant_coefficients(self) -> None:
        """With constant true betas, estimated betas should converge."""
        rng = np.random.default_rng(42)
        T = 300
        true_beta = np.array([1.0, 2.0])
        X = np.column_stack([np.ones(T), rng.normal(0, 1, T)])
        y = X @ true_beta + rng.normal(0, 0.1, T)

        result = kalman_regression(y, X, state_noise=1e-6)

        # At the end, coefficients should be close to true values
        final_coefs = result["coefficients"][-1]
        np.testing.assert_allclose(final_coefs, true_beta, atol=0.3)

    def test_time_varying_beta(self) -> None:
        """Should track a beta that changes from 1.0 to 2.0."""
        rng = np.random.default_rng(42)
        T = 400
        market = rng.normal(0, 0.01, T)

        # True beta changes at midpoint
        true_beta = np.where(np.arange(T) < 200, 1.0, 2.0)
        stock = true_beta * market + rng.normal(0, 0.005, T)

        X = np.column_stack([np.ones(T), market])
        result = kalman_regression(stock, X, state_noise=1e-4)

        betas = result["coefficients"][:, 1]
        # Beta should be near 1.0 in first half and 2.0 in second half
        # (with transition period)
        assert betas[50] < 1.5
        assert betas[350] > 1.5

    def test_log_likelihood_finite(self) -> None:
        rng = np.random.default_rng(42)
        T = 100
        X = np.column_stack([np.ones(T), rng.normal(0, 1, T)])
        y = X @ np.array([0.5, 1.0]) + rng.normal(0, 0.1, T)

        result = kalman_regression(y, X)
        assert np.isfinite(result["log_likelihood"])

    def test_no_smooth(self) -> None:
        """Without smoothing, coefficients should equal filtered."""
        rng = np.random.default_rng(42)
        T = 100
        X = np.column_stack([np.ones(T), rng.normal(0, 1, T)])
        y = X @ np.array([0.5, 1.0]) + rng.normal(0, 0.1, T)

        result = kalman_regression(y, X, smooth=False)
        np.testing.assert_array_equal(
            result["coefficients"],
            result["filtered_coefficients"],
        )


# ---------------------------------------------------------------------------
# Tests: unscented_kalman
# ---------------------------------------------------------------------------


class TestUnscentedKalman:
    def test_output_keys(self) -> None:
        T = 50
        obs = np.random.default_rng(42).normal(size=(T, 1))

        def f(x):
            return x

        def h(x):
            return x

        result = unscented_kalman(
            obs, state_dim=1, f_func=f, h_func=h,
            Q=np.array([[0.01]]), R=np.array([[0.1]]),
            x0=np.array([0.0]), P0=np.array([[1.0]]),
        )

        expected_keys = {
            "filtered_states", "filtered_covs",
            "innovations", "log_likelihood",
        }
        assert expected_keys.issubset(result.keys())

    def test_output_shapes(self) -> None:
        T = 50
        n = 2
        m = 1
        obs = np.random.default_rng(42).normal(size=(T, m))

        def f(x):
            return x

        def h(x):
            return np.array([x[0]])

        result = unscented_kalman(
            obs, state_dim=n, f_func=f, h_func=h,
            Q=np.eye(n) * 0.01, R=np.eye(m) * 0.1,
            x0=np.zeros(n), P0=np.eye(n),
        )

        assert result["filtered_states"].shape == (T, n)
        assert result["filtered_covs"].shape == (T, n, n)
        assert result["innovations"].shape == (T, m)

    def test_linear_case_matches_kalman(self) -> None:
        """For a linear system, UKF should give similar results to KF."""
        rng = np.random.default_rng(42)
        T = 100
        F_mat = np.array([[1.0]])
        H_mat = np.array([[1.0]])
        Q = np.array([[0.01]])
        R = np.array([[0.1]])
        x0 = np.array([0.0])
        P0 = np.array([[1.0]])
        obs = rng.normal(size=(T, 1))

        # Linear Kalman filter
        kf_result = kalman_filter(obs, F_mat, H_mat, Q, R, x0, P0)

        # UKF with same linear functions
        def f(x):
            return F_mat @ x

        def h(x):
            return H_mat @ x

        ukf_result = unscented_kalman(
            obs, state_dim=1, f_func=f, h_func=h,
            Q=Q, R=R, x0=x0, P0=P0,
        )

        # States should be very similar
        np.testing.assert_allclose(
            ukf_result["filtered_states"],
            kf_result["filtered_states"],
            atol=0.1,
        )

    def test_nonlinear_observation(self) -> None:
        """UKF should handle nonlinear observation model."""
        rng = np.random.default_rng(42)
        T = 100

        # Simple system: state is x, observe x^2 + noise
        true_x = np.cumsum(rng.normal(0, 0.1, T))
        obs = (true_x**2 + rng.normal(0, 0.5, T)).reshape(-1, 1)

        def f(x):
            return x  # Random walk

        def h(x):
            return np.array([x[0] ** 2])  # Nonlinear observation

        result = unscented_kalman(
            obs, state_dim=1, f_func=f, h_func=h,
            Q=np.array([[0.01]]), R=np.array([[0.5]]),
            x0=np.array([0.0]), P0=np.array([[1.0]]),
        )

        assert result["filtered_states"].shape == (T, 1)
        assert np.isfinite(result["log_likelihood"])

    def test_covariance_positive_definite(self) -> None:
        rng = np.random.default_rng(42)
        T = 50
        obs = rng.normal(size=(T, 1))

        def f(x):
            return x

        def h(x):
            return x

        result = unscented_kalman(
            obs, state_dim=1, f_func=f, h_func=h,
            Q=np.array([[0.01]]), R=np.array([[0.1]]),
            x0=np.array([0.0]), P0=np.array([[1.0]]),
        )

        for t in range(T):
            eigvals = np.linalg.eigvalsh(result["filtered_covs"][t])
            assert (eigvals >= -1e-10).all(), f"Non-PSD covariance at t={t}"
