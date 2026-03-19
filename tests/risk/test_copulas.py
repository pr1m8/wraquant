"""Tests for copula models."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sp_stats

from wraquant.risk.copulas import (
    copula_simulate,
    fit_clayton_copula,
    fit_frank_copula,
    fit_gaussian_copula,
    fit_gumbel_copula,
    fit_t_copula,
    rank_correlation,
    tail_dependence,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_correlated_returns(
    n: int = 1000, rho: float = 0.6, seed: int = 42
) -> np.ndarray:
    """Generate bivariate normal returns with given correlation."""
    rng = np.random.default_rng(seed)
    cov = np.array([[1, rho], [rho, 1]]) * 0.0004
    return rng.multivariate_normal([0, 0], cov, size=n)


def _make_multivariate_returns(n: int = 1000, k: int = 4, seed: int = 42) -> np.ndarray:
    """Generate k-variate normal returns."""
    rng = np.random.default_rng(seed)
    # Build a correlation matrix
    A = rng.normal(0, 1, (k, k))
    cov = A @ A.T / k * 0.0004
    mu = np.zeros(k)
    return rng.multivariate_normal(mu, cov, size=n)


def _make_uniform_pair(n: int = 1000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate positively dependent pseudo-observations."""
    rng = np.random.default_rng(seed)
    # Gaussian copula with known correlation
    z = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n)
    u = sp_stats.norm.cdf(z[:, 0])
    v = sp_stats.norm.cdf(z[:, 1])
    return u, v


# ---------------------------------------------------------------------------
# Gaussian copula
# ---------------------------------------------------------------------------


class TestFitGaussianCopula:
    def test_returns_correlation_matrix(self) -> None:
        data = _make_multivariate_returns(n=2000, k=3)
        result = fit_gaussian_copula(data)
        assert result["copula_type"] == "gaussian"
        corr = result["correlation"]
        assert corr.shape == (3, 3)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-10)

    def test_symmetric(self) -> None:
        data = _make_multivariate_returns(n=2000, k=3)
        corr = fit_gaussian_copula(data)["correlation"]
        np.testing.assert_allclose(corr, corr.T, atol=1e-10)

    def test_correlation_in_range(self) -> None:
        data = _make_multivariate_returns(n=2000, k=4)
        corr = fit_gaussian_copula(data)["correlation"]
        assert np.all(corr >= -1) and np.all(corr <= 1)


# ---------------------------------------------------------------------------
# Student-t copula
# ---------------------------------------------------------------------------


class TestFitTCopula:
    def test_returns_correct_df(self) -> None:
        data = _make_multivariate_returns(n=2000, k=3)
        result = fit_t_copula(data, df=4.0)
        assert result["copula_type"] == "student_t"
        assert result["df"] == 4.0

    def test_correlation_matrix_shape(self) -> None:
        data = _make_multivariate_returns(n=2000, k=3)
        result = fit_t_copula(data, df=5.0)
        assert result["correlation"].shape == (3, 3)

    def test_diagonal_ones(self) -> None:
        data = _make_multivariate_returns(n=2000, k=3)
        result = fit_t_copula(data, df=5.0)
        np.testing.assert_allclose(np.diag(result["correlation"]), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Clayton copula
# ---------------------------------------------------------------------------


class TestFitClaytonCopula:
    def test_positive_theta(self) -> None:
        u, v = _make_uniform_pair(n=2000)
        result = fit_clayton_copula(u, v)
        assert result["copula_type"] == "clayton"
        assert result["theta"] > 0

    def test_lower_tail_dependence(self) -> None:
        u, v = _make_uniform_pair(n=2000)
        result = fit_clayton_copula(u, v)
        ltd = result["lower_tail_dependence"]
        assert 0 < ltd < 1

    def test_raw_data_auto_converted(self) -> None:
        # Pass raw returns, should auto-convert to pseudo-obs
        data = _make_correlated_returns(n=2000, rho=0.5)
        result = fit_clayton_copula(data[:, 0], data[:, 1])
        assert result["theta"] > 0


# ---------------------------------------------------------------------------
# Gumbel copula
# ---------------------------------------------------------------------------


class TestFitGumbelCopula:
    def test_theta_ge_one(self) -> None:
        u, v = _make_uniform_pair(n=2000)
        result = fit_gumbel_copula(u, v)
        assert result["copula_type"] == "gumbel"
        assert result["theta"] >= 1.0

    def test_upper_tail_dependence(self) -> None:
        u, v = _make_uniform_pair(n=2000)
        result = fit_gumbel_copula(u, v)
        utd = result["upper_tail_dependence"]
        assert 0 <= utd < 1


# ---------------------------------------------------------------------------
# Frank copula
# ---------------------------------------------------------------------------


class TestFitFrankCopula:
    def test_nonzero_theta(self) -> None:
        u, v = _make_uniform_pair(n=2000)
        result = fit_frank_copula(u, v)
        assert result["copula_type"] == "frank"
        assert abs(result["theta"]) > 0.01

    def test_positive_dependence_positive_theta(self) -> None:
        u, v = _make_uniform_pair(n=2000)
        result = fit_frank_copula(u, v)
        # Positive concordance => positive theta
        assert result["theta"] > 0


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


class TestCopulaSimulate:
    def test_gaussian_output_shape(self) -> None:
        data = _make_multivariate_returns(n=1000, k=3)
        params = fit_gaussian_copula(data)
        sims = copula_simulate(params, n_sims=500, seed=42)
        assert sims.shape == (500, 3)

    def test_gaussian_in_unit_interval(self) -> None:
        data = _make_multivariate_returns(n=1000, k=3)
        params = fit_gaussian_copula(data)
        sims = copula_simulate(params, n_sims=1000, seed=42)
        assert np.all(sims >= 0) and np.all(sims <= 1)

    def test_t_copula_simulation(self) -> None:
        data = _make_multivariate_returns(n=1000, k=2)
        params = fit_t_copula(data, df=5.0)
        sims = copula_simulate(params, n_sims=500, seed=42)
        assert sims.shape == (500, 2)
        assert np.all(sims >= 0) and np.all(sims <= 1)

    def test_clayton_simulation(self) -> None:
        u, v = _make_uniform_pair(n=1000)
        params = fit_clayton_copula(u, v)
        sims = copula_simulate(params, n_sims=500, seed=42)
        assert sims.shape == (500, 2)

    def test_frank_simulation(self) -> None:
        u, v = _make_uniform_pair(n=1000)
        params = fit_frank_copula(u, v)
        sims = copula_simulate(params, n_sims=500, seed=42)
        assert sims.shape == (500, 2)

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown copula type"):
            copula_simulate({"copula_type": "invalid"}, n_sims=10)


# ---------------------------------------------------------------------------
# Tail dependence
# ---------------------------------------------------------------------------


class TestTailDependence:
    def test_returns_lower_and_upper(self) -> None:
        u, v = _make_uniform_pair(n=5000)
        result = tail_dependence(u, v)
        assert "lower" in result
        assert "upper" in result

    def test_values_in_range(self) -> None:
        u, v = _make_uniform_pair(n=5000)
        result = tail_dependence(u, v, threshold=0.10)
        assert 0 <= result["lower"] <= 1
        assert 0 <= result["upper"] <= 1

    def test_unknown_method_raises(self) -> None:
        u, v = _make_uniform_pair()
        with pytest.raises(ValueError, match="Unknown tail dependence"):
            tail_dependence(u, v, method="invalid")


# ---------------------------------------------------------------------------
# Rank correlation
# ---------------------------------------------------------------------------


class TestRankCorrelation:
    def test_both_returned(self) -> None:
        data = _make_multivariate_returns(n=500, k=3)
        result = rank_correlation(data, method="both")
        assert "kendall_tau" in result
        assert "spearman_rho" in result

    def test_kendall_only(self) -> None:
        data = _make_multivariate_returns(n=500, k=2)
        result = rank_correlation(data, method="kendall")
        assert "kendall_tau" in result
        assert "spearman_rho" not in result

    def test_spearman_only(self) -> None:
        data = _make_multivariate_returns(n=500, k=2)
        result = rank_correlation(data, method="spearman")
        assert "spearman_rho" in result
        assert "kendall_tau" not in result

    def test_diagonal_ones(self) -> None:
        data = _make_multivariate_returns(n=500, k=3)
        result = rank_correlation(data)
        np.testing.assert_allclose(np.diag(result["kendall_tau"]), 1.0, atol=1e-10)
        np.testing.assert_allclose(np.diag(result["spearman_rho"]), 1.0, atol=1e-10)

    def test_unknown_method_raises(self) -> None:
        data = _make_multivariate_returns(n=100, k=2)
        with pytest.raises(ValueError, match="Unknown rank correlation"):
            rank_correlation(data, method="invalid")
