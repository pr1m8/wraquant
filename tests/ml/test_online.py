"""Tests for wraquant.ml.online — Online/streaming ML models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.ml.online import (
    exponential_weighted_regression,
    online_linear_regression,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def stationary_regression_data() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic data with fixed true coefficients."""
    np.random.seed(42)
    T = 500
    X = np.random.randn(T, 3)
    beta_true = np.array([2.0, -1.0, 0.5])
    y = X @ beta_true + np.random.randn(T) * 0.1
    return X, y


@pytest.fixture()
def switching_regression_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic data with a coefficient regime switch at t=250."""
    np.random.seed(55)
    T = 500
    X = np.random.randn(T, 2)
    beta1 = np.array([1.0, 0.5])
    beta2 = np.array([-0.5, 1.5])
    beta_true = np.where(
        np.arange(T)[:, None] < 250,
        beta1,
        beta2,
    )
    y = np.sum(X * beta_true, axis=1) + np.random.randn(T) * 0.1
    return X, y, beta_true


# ---------------------------------------------------------------------------
# online_linear_regression
# ---------------------------------------------------------------------------


class TestOnlineLinearRegression:
    def test_output_keys(self, stationary_regression_data: tuple) -> None:
        X, y = stationary_regression_data
        result = online_linear_regression(X, y)
        expected = {"coefficients", "predictions", "residuals", "final_coefficients"}
        assert set(result.keys()) == expected

    def test_output_shapes(self, stationary_regression_data: tuple) -> None:
        X, y = stationary_regression_data
        result = online_linear_regression(X, y)
        assert result["coefficients"].shape == (500, 3)
        assert result["predictions"].shape == (500,)
        assert result["residuals"].shape == (500,)
        assert result["final_coefficients"].shape == (3,)

    def test_convergence_stationary(
        self, stationary_regression_data: tuple
    ) -> None:
        """With stationary coefficients and lambda=1, RLS should converge to OLS."""
        X, y = stationary_regression_data
        result = online_linear_regression(X, y, forgetting_factor=1.0)
        final = result["final_coefficients"]
        # True coefficients are [2.0, -1.0, 0.5]
        np.testing.assert_allclose(final, [2.0, -1.0, 0.5], atol=0.15)

    def test_tracks_regime_switch(self, switching_regression_data: tuple) -> None:
        """With forgetting, coefficients should adapt to the new regime."""
        X, y, beta_true = switching_regression_data
        result = online_linear_regression(X, y, forgetting_factor=0.98)
        coefs = result["coefficients"]
        # By t=400 (well after switch at 250), should be closer to beta2
        late_coefs = coefs[400]
        assert np.abs(late_coefs[0] - (-0.5)) < 0.5
        assert np.abs(late_coefs[1] - 1.5) < 0.5

    def test_accepts_pandas(self) -> None:
        np.random.seed(10)
        X = pd.DataFrame(np.random.randn(100, 2), columns=["a", "b"])
        y = pd.Series(X["a"] * 1.5 + X["b"] * (-0.5) + np.random.randn(100) * 0.1)
        result = online_linear_regression(X, y)
        assert result["coefficients"].shape == (100, 2)

    def test_residuals_shrink(self, stationary_regression_data: tuple) -> None:
        """Residuals should become smaller over time as filter converges."""
        X, y = stationary_regression_data
        result = online_linear_regression(X, y, forgetting_factor=1.0)
        residuals = result["residuals"]
        # Compare mean absolute residual in first 50 vs last 50
        early = np.mean(np.abs(residuals[:50]))
        late = np.mean(np.abs(residuals[-50:]))
        assert late < early


# ---------------------------------------------------------------------------
# exponential_weighted_regression
# ---------------------------------------------------------------------------


class TestExponentialWeightedRegression:
    def test_output_keys(self, stationary_regression_data: tuple) -> None:
        X, y = stationary_regression_data
        result = exponential_weighted_regression(X, y, halflife=63)
        expected = {"coefficients", "predictions", "residuals", "final_coefficients"}
        assert set(result.keys()) == expected

    def test_output_shapes(self, stationary_regression_data: tuple) -> None:
        X, y = stationary_regression_data
        result = exponential_weighted_regression(X, y, halflife=63, min_periods=30)
        assert result["coefficients"].shape == (500, 3)
        assert result["predictions"].shape == (500,)
        assert result["residuals"].shape == (500,)

    def test_nan_before_min_periods(
        self, stationary_regression_data: tuple
    ) -> None:
        X, y = stationary_regression_data
        result = exponential_weighted_regression(X, y, halflife=63, min_periods=50)
        # First 49 rows should be NaN
        assert np.all(np.isnan(result["coefficients"][:49]))
        # Row 49 (index 49, 50th row) should be valid
        assert not np.any(np.isnan(result["coefficients"][49]))

    def test_convergence(self, stationary_regression_data: tuple) -> None:
        X, y = stationary_regression_data
        result = exponential_weighted_regression(X, y, halflife=63, min_periods=30)
        final = result["final_coefficients"]
        np.testing.assert_allclose(final, [2.0, -1.0, 0.5], atol=0.2)

    def test_tracks_drift(self) -> None:
        """Coefficients should track drifting true parameters."""
        np.random.seed(33)
        T = 400
        X = np.random.randn(T, 1)
        # True coefficient drifts from 2 to 0
        beta_true = np.linspace(2.0, 0.0, T)
        y = X[:, 0] * beta_true + np.random.randn(T) * 0.1

        result = exponential_weighted_regression(
            X, y, halflife=30, min_periods=20
        )
        coefs = result["coefficients"]
        # Early: coefficient near 2
        assert np.abs(coefs[50, 0] - 2.0) < 0.6
        # Late: coefficient near 0
        assert np.abs(coefs[-1, 0] - 0.0) < 0.6

    def test_accepts_pandas(self) -> None:
        np.random.seed(10)
        X = pd.DataFrame(np.random.randn(150, 2))
        y = pd.Series(X[0] * 1.0 + X[1] * 0.5 + np.random.randn(150) * 0.1)
        result = exponential_weighted_regression(X, y, halflife=30, min_periods=20)
        assert result["final_coefficients"].shape == (2,)
