"""Tests for DCC-GARCH models."""

from __future__ import annotations

import numpy as np
import pytest

from wraquant.risk.dcc import (
    conditional_covariance,
    dcc_garch,
    forecast_correlation,
    rolling_correlation_dcc,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_garch_returns(
    n: int = 500, k: int = 2, seed: int = 42
) -> np.ndarray:
    """Generate k correlated return series with mild GARCH-like clustering."""
    rng = np.random.default_rng(seed)
    # Build correlated innovations
    rho = 0.5
    cov = np.eye(k) * 0.0004
    for i in range(k):
        for j in range(i + 1, k):
            cov[i, j] = cov[j, i] = rho * 0.0004
    eps = rng.multivariate_normal(np.zeros(k), cov, size=n)

    # Simple GARCH-like vol clustering
    sigma2 = np.full((n, k), 0.0004)
    returns = np.empty((n, k))
    returns[0] = eps[0]
    for t in range(1, n):
        for j in range(k):
            sigma2[t, j] = 0.00002 + 0.05 * returns[t - 1, j] ** 2 + 0.90 * sigma2[t - 1, j]
        returns[t] = eps[t] * np.sqrt(sigma2[t] / 0.0004)

    return returns


# ---------------------------------------------------------------------------
# dcc_garch
# ---------------------------------------------------------------------------

class TestDCCGarch:
    def test_returns_correct_keys(self) -> None:
        data = _make_garch_returns(n=300, k=2)
        model = dcc_garch(data)
        assert "a" in model
        assert "b" in model
        assert "garch_params" in model
        assert "qbar" in model
        assert "conditional_vols" in model
        assert "std_residuals" in model

    def test_a_b_positive(self) -> None:
        data = _make_garch_returns(n=300, k=2)
        model = dcc_garch(data)
        assert model["a"] > 0
        assert model["b"] > 0
        assert model["a"] + model["b"] < 1

    def test_garch_params_per_asset(self) -> None:
        data = _make_garch_returns(n=300, k=3)
        model = dcc_garch(data)
        assert len(model["garch_params"]) == 3
        for gp in model["garch_params"]:
            assert "omega" in gp
            assert "alpha" in gp
            assert "beta" in gp
            assert gp["omega"] > 0
            assert gp["alpha"] + gp["beta"] < 1

    def test_qbar_is_correlation(self) -> None:
        data = _make_garch_returns(n=300, k=2)
        model = dcc_garch(data)
        qbar = model["qbar"]
        np.testing.assert_allclose(np.diag(qbar), 1.0, atol=1e-10)
        np.testing.assert_allclose(qbar, qbar.T, atol=1e-10)

    def test_conditional_vols_positive(self) -> None:
        data = _make_garch_returns(n=300, k=2)
        model = dcc_garch(data)
        assert np.all(model["conditional_vols"] > 0)

    def test_only_dcc11_supported(self) -> None:
        data = _make_garch_returns(n=300, k=2)
        with pytest.raises(ValueError, match="Only DCC"):
            dcc_garch(data, p=2, q=1)


# ---------------------------------------------------------------------------
# rolling_correlation_dcc
# ---------------------------------------------------------------------------

class TestRollingCorrelationDCC:
    def test_output_shape(self) -> None:
        data = _make_garch_returns(n=300, k=2)
        result = rolling_correlation_dcc(data)
        corrs = result["correlations"]
        assert corrs.shape == (300, 2, 2)

    def test_diagonal_ones(self) -> None:
        data = _make_garch_returns(n=300, k=2)
        result = rolling_correlation_dcc(data)
        corrs = result["correlations"]
        for t in range(corrs.shape[0]):
            np.testing.assert_allclose(np.diag(corrs[t]), 1.0, atol=1e-8)

    def test_correlations_in_range(self) -> None:
        data = _make_garch_returns(n=300, k=2)
        result = rolling_correlation_dcc(data)
        corrs = result["correlations"]
        assert np.all(corrs >= -1 - 1e-8) and np.all(corrs <= 1 + 1e-8)


# ---------------------------------------------------------------------------
# forecast_correlation
# ---------------------------------------------------------------------------

class TestForecastCorrelation:
    def test_forecast_shape(self) -> None:
        data = _make_garch_returns(n=300, k=2)
        model = dcc_garch(data)
        result = forecast_correlation(model, horizon=5)
        assert result["forecasted_correlations"].shape == (5, 2, 2)
        assert result["forecasted_covariances"].shape == (5, 2, 2)

    def test_forecast_converges_to_qbar(self) -> None:
        data = _make_garch_returns(n=300, k=2)
        model = dcc_garch(data)
        result = forecast_correlation(model, horizon=500)
        # At long horizon, correlation should converge toward qbar
        last_corr = result["forecasted_correlations"][-1]
        qbar = model["qbar"]
        # Allow some tolerance since convergence may not be perfect in 500 steps
        np.testing.assert_allclose(last_corr, qbar, atol=0.15)


# ---------------------------------------------------------------------------
# conditional_covariance
# ---------------------------------------------------------------------------

class TestConditionalCovariance:
    def test_output_keys(self) -> None:
        data = _make_garch_returns(n=300, k=2)
        result = conditional_covariance(data)
        assert "covariances" in result
        assert "correlations" in result
        assert "volatilities" in result

    def test_covariance_shape(self) -> None:
        data = _make_garch_returns(n=300, k=3)
        result = conditional_covariance(data)
        assert result["covariances"].shape == (300, 3, 3)
        assert result["volatilities"].shape == (300, 3)

    def test_covariance_symmetric(self) -> None:
        data = _make_garch_returns(n=300, k=2)
        result = conditional_covariance(data)
        for t in range(300):
            cov_t = result["covariances"][t]
            np.testing.assert_allclose(cov_t, cov_t.T, atol=1e-12)

    def test_with_prefit_model(self) -> None:
        data = _make_garch_returns(n=300, k=2)
        model = dcc_garch(data)
        result = conditional_covariance(data, dcc_params=model)
        assert result["covariances"].shape == (300, 2, 2)
