"""Tests for stochastic process forecasting."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from wraquant.ts.stochastic import (
    jump_diffusion_forecast,
    ornstein_uhlenbeck_forecast,
    var_forecast,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simulate_ou(
    n: int = 500,
    theta: float = 5.0,
    mu: float = 100.0,
    sigma: float = 2.0,
    dt: float = 1 / 252,
    seed: int = 42,
) -> pd.Series:
    """Simulate an OU process for testing."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    x[0] = mu
    for i in range(1, n):
        x[i] = (
            x[i - 1]
            + theta * (mu - x[i - 1]) * dt
            + sigma * np.sqrt(dt) * rng.normal()
        )
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(x, index=idx, name="ou")


def _simulate_gbm(
    n: int = 300,
    mu: float = 0.05,
    sigma: float = 0.2,
    s0: float = 100.0,
    seed: int = 42,
) -> pd.Series:
    """Simulate geometric Brownian motion prices."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.normal(
        size=n
    )
    prices = s0 * np.exp(np.cumsum(log_returns))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(prices, index=idx, name="gbm")


# ---------------------------------------------------------------------------
# ornstein_uhlenbeck_forecast
# ---------------------------------------------------------------------------


class TestOrnsteinUhlenbeckForecast:
    def test_output_keys(self) -> None:
        data = _simulate_ou()
        result = ornstein_uhlenbeck_forecast(data, h=20)
        expected = {"params", "half_life", "forecast", "confidence_intervals"}
        assert expected.issubset(set(result.keys()))

    def test_params_structure(self) -> None:
        data = _simulate_ou()
        result = ornstein_uhlenbeck_forecast(data, h=10)
        params = result["params"]
        assert "theta" in params
        assert "mu" in params
        assert "sigma" in params
        assert params["theta"] > 0
        assert params["sigma"] > 0

    def test_half_life_reasonable(self) -> None:
        """Half-life should be positive and finite for mean-reverting data."""
        data = _simulate_ou(n=1000, theta=5.0)
        result = ornstein_uhlenbeck_forecast(data, h=20)
        hl = result["half_life"]
        assert hl > 0
        assert np.isfinite(hl)
        # For theta=5, half_life ~ ln(2)/5 ~ 0.14 in continuous time
        # In observation units (dt = 1/252): ~0.14 * 252 ~ 35
        # Allow wide tolerance since estimation is noisy
        assert hl < 500

    def test_forecast_length(self) -> None:
        data = _simulate_ou()
        result = ornstein_uhlenbeck_forecast(data, h=30)
        assert len(result["forecast"]) == 30

    def test_forecast_reverts_to_mean(self) -> None:
        """Long-horizon forecast should approach mu."""
        data = _simulate_ou(n=1000, theta=10.0, mu=100.0)
        result = ornstein_uhlenbeck_forecast(data, h=100)
        mu_hat = result["params"]["mu"]
        # Last forecast value should be close to estimated mu
        last_fcast = result["forecast"].iloc[-1]
        assert abs(last_fcast - mu_hat) < abs(data.iloc[-1] - mu_hat) + 5

    def test_confidence_intervals(self) -> None:
        data = _simulate_ou()
        result = ornstein_uhlenbeck_forecast(data, h=20)
        ci = result["confidence_intervals"]
        assert "lower" in ci
        assert "upper" in ci
        assert len(ci["lower"]) == 20
        assert np.all(ci["upper"].values >= ci["lower"].values)

    def test_ci_widens_over_horizon(self) -> None:
        """Confidence intervals should widen as horizon increases."""
        data = _simulate_ou()
        result = ornstein_uhlenbeck_forecast(data, h=50)
        ci = result["confidence_intervals"]
        widths = ci["upper"].values - ci["lower"].values
        # Width should generally increase (may plateau for OU)
        assert widths[-1] >= widths[0]


# ---------------------------------------------------------------------------
# jump_diffusion_forecast
# ---------------------------------------------------------------------------


class TestJumpDiffusionForecast:
    def test_output_keys(self) -> None:
        data = _simulate_gbm()
        result = jump_diffusion_forecast(data, h=10, n_paths=100, seed=42)
        expected = {
            "params", "forecast_paths", "mean_forecast",
            "confidence_intervals",
        }
        assert expected.issubset(set(result.keys()))

    def test_forecast_paths_shape(self) -> None:
        data = _simulate_gbm()
        n_paths, h = 200, 15
        result = jump_diffusion_forecast(
            data, h=h, n_paths=n_paths, seed=42
        )
        assert result["forecast_paths"].shape == (n_paths, h)

    def test_mean_forecast_length(self) -> None:
        data = _simulate_gbm()
        result = jump_diffusion_forecast(data, h=20, n_paths=100, seed=42)
        assert len(result["mean_forecast"]) == 20

    def test_positive_prices(self) -> None:
        """Simulated price paths should remain positive (GBM property)."""
        data = _simulate_gbm()
        result = jump_diffusion_forecast(data, h=10, n_paths=500, seed=42)
        assert np.all(result["forecast_paths"] > 0)

    def test_params_structure(self) -> None:
        data = _simulate_gbm()
        result = jump_diffusion_forecast(data, h=10, n_paths=100, seed=42)
        params = result["params"]
        assert "mu" in params
        assert "sigma" in params
        assert "lambda_" in params
        assert "mu_j" in params
        assert "sigma_j" in params

    def test_confidence_intervals(self) -> None:
        data = _simulate_gbm()
        result = jump_diffusion_forecast(data, h=10, n_paths=500, seed=42)
        ci = result["confidence_intervals"]
        assert len(ci["lower"]) == 10
        assert np.all(ci["upper"].values >= ci["lower"].values)


# ---------------------------------------------------------------------------
# var_forecast
# ---------------------------------------------------------------------------


class TestVARForecast:
    def _make_var_data(self, n: int = 300, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        e1 = rng.normal(0, 1, n)
        e2 = rng.normal(0, 1, n)
        x1 = np.zeros(n)
        x2 = np.zeros(n)
        for i in range(1, n):
            x1[i] = 0.5 * x1[i - 1] + 0.1 * x2[i - 1] + e1[i]
            x2[i] = 0.2 * x1[i - 1] + 0.4 * x2[i - 1] + e2[i]
        idx = pd.bdate_range("2020-01-01", periods=n)
        return pd.DataFrame({"x1": x1, "x2": x2}, index=idx)

    def test_output_keys(self) -> None:
        data = self._make_var_data()
        result = var_forecast(data, h=10)
        expected = {"forecast", "irf", "fevd", "granger_causality", "lag_order"}
        assert expected.issubset(set(result.keys()))

    def test_forecast_shape(self) -> None:
        data = self._make_var_data()
        h = 15
        result = var_forecast(data, h=h)
        assert result["forecast"].shape == (h, 2)

    def test_forecast_columns(self) -> None:
        data = self._make_var_data()
        result = var_forecast(data, h=10)
        assert list(result["forecast"].columns) == ["x1", "x2"]

    def test_lag_order_positive(self) -> None:
        data = self._make_var_data()
        result = var_forecast(data, h=10)
        assert result["lag_order"] >= 1

    def test_irf_structure(self) -> None:
        data = self._make_var_data()
        result = var_forecast(data, h=10)
        assert "x1" in result["irf"]
        assert "x2" in result["irf"]
        assert isinstance(result["irf"]["x1"], pd.DataFrame)

    def test_granger_causality(self) -> None:
        data = self._make_var_data()
        result = var_forecast(data, h=10)
        gc = result["granger_causality"]
        assert len(gc) > 0
        for key, pval in gc.items():
            if np.isfinite(pval):
                assert 0 <= pval <= 1
