"""Tests for statistical factor analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.stats.factor_analysis import (
    common_factors,
    factor_correlation,
    factor_loadings,
    factor_mimicking_portfolios,
    pca_factors,
    risk_factor_decomposition,
    scree_plot_data,
    varimax_rotation,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_returns(
    n_obs: int = 200,
    n_assets: int = 10,
    n_true_factors: int = 3,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Create synthetic returns driven by a known factor model.

    Returns
    -------
    returns : pd.DataFrame (T, N)
    true_factors : ndarray (T, n_true_factors)
    true_loadings : ndarray (N, n_true_factors)
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_obs)
    assets = [f"asset_{i}" for i in range(n_assets)]

    factors = rng.normal(0, 0.01, size=(n_obs, n_true_factors))
    loadings = rng.normal(0, 1, size=(n_assets, n_true_factors))
    noise = rng.normal(0, 0.003, size=(n_obs, n_assets))

    ret = factors @ loadings.T + noise
    df = pd.DataFrame(ret, index=dates, columns=assets)
    return df, factors, loadings


# ---------------------------------------------------------------------------
# PCA factors
# ---------------------------------------------------------------------------


class TestPcaFactors:
    def test_output_keys(self) -> None:
        ret, _, _ = _make_returns()
        result = pca_factors(ret, n_components=3)
        assert "factors" in result
        assert "loadings" in result
        assert "explained_variance" in result
        assert "explained_variance_ratio" in result

    def test_factors_shape(self) -> None:
        ret, _, _ = _make_returns(n_obs=100, n_assets=8)
        result = pca_factors(ret, n_components=3)
        assert result["factors"].shape == (100, 3)
        assert result["loadings"].shape == (8, 3)

    def test_explained_variance_ratio_sums_less_than_one(self) -> None:
        ret, _, _ = _make_returns()
        result = pca_factors(ret, n_components=3)
        assert result["explained_variance_ratio"].sum() <= 1.0 + 1e-10

    def test_explained_variance_descending(self) -> None:
        ret, _, _ = _make_returns()
        result = pca_factors(ret, n_components=5)
        ev = result["explained_variance"]
        assert all(ev[i] >= ev[i + 1] - 1e-10 for i in range(len(ev) - 1))

    def test_eig_method_agrees_with_svd(self) -> None:
        ret, _, _ = _make_returns(n_obs=100, n_assets=5)
        svd_result = pca_factors(ret, n_components=2, method="svd")
        eig_result = pca_factors(ret, n_components=2, method="eig")
        # Explained variance ratios should be close
        np.testing.assert_allclose(
            svd_result["explained_variance_ratio"],
            eig_result["explained_variance_ratio"],
            atol=1e-6,
        )

    def test_dominant_factors_capture_most_variance(self) -> None:
        # With 3 true factors and low noise, top 3 PCs should explain > 60%
        ret, _, _ = _make_returns(n_obs=300, n_true_factors=3)
        result = pca_factors(ret, n_components=3)
        assert result["explained_variance_ratio"].sum() > 0.6


# ---------------------------------------------------------------------------
# Factor loadings
# ---------------------------------------------------------------------------


class TestFactorLoadings:
    def test_loadings_shape(self) -> None:
        ret, true_factors, _ = _make_returns(n_obs=200, n_assets=8)
        result = factor_loadings(ret, true_factors)
        assert result["loadings"].shape == (8, 3)

    def test_alphas_shape(self) -> None:
        ret, true_factors, _ = _make_returns()
        result = factor_loadings(ret, true_factors)
        assert result["alphas"].shape == (ret.shape[1],)

    def test_r_squared_valid(self) -> None:
        ret, true_factors, _ = _make_returns()
        result = factor_loadings(ret, true_factors)
        assert np.all(result["r_squared"] >= 0)
        assert np.all(result["r_squared"] <= 1.0 + 1e-10)

    def test_residuals_shape(self) -> None:
        ret, true_factors, _ = _make_returns()
        result = factor_loadings(ret, true_factors)
        assert result["residuals"].shape == ret.shape

    def test_high_r_squared_with_true_factors(self) -> None:
        # When we pass the true factors, R-squared should be high
        ret, true_factors, _ = _make_returns(n_obs=500)
        result = factor_loadings(ret, true_factors)
        assert np.mean(result["r_squared"]) > 0.5


# ---------------------------------------------------------------------------
# Scree plot data
# ---------------------------------------------------------------------------


class TestScreePlotData:
    def test_eigenvalues_descending(self) -> None:
        ret, _, _ = _make_returns()
        result = scree_plot_data(ret)
        ev = result["eigenvalues"]
        assert all(ev[i] >= ev[i + 1] - 1e-12 for i in range(len(ev) - 1))

    def test_cumulative_ends_at_one(self) -> None:
        ret, _, _ = _make_returns()
        result = scree_plot_data(ret)
        np.testing.assert_allclose(result["cumulative_variance_ratio"][-1], 1.0, atol=1e-10)

    def test_n_eigenvalues_equals_n_assets(self) -> None:
        ret, _, _ = _make_returns(n_assets=7)
        result = scree_plot_data(ret)
        assert len(result["eigenvalues"]) == 7


# ---------------------------------------------------------------------------
# Varimax rotation
# ---------------------------------------------------------------------------


class TestVarimaxRotation:
    def test_rotation_matrix_is_orthogonal(self) -> None:
        rng = np.random.default_rng(42)
        loadings = rng.normal(0, 1, size=(10, 3))
        result = varimax_rotation(loadings)
        R = result["rotation_matrix"]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_rotated_shape_matches_input(self) -> None:
        rng = np.random.default_rng(42)
        loadings = rng.normal(0, 1, size=(8, 4))
        result = varimax_rotation(loadings)
        assert result["rotated_loadings"].shape == (8, 4)

    def test_single_factor_returns_unchanged(self) -> None:
        rng = np.random.default_rng(42)
        loadings = rng.normal(0, 1, size=(5, 1))
        result = varimax_rotation(loadings)
        np.testing.assert_allclose(result["rotated_loadings"], loadings)


# ---------------------------------------------------------------------------
# Factor-mimicking portfolios
# ---------------------------------------------------------------------------


class TestFactorMimickingPortfolios:
    def test_static_characteristic(self) -> None:
        rng = np.random.default_rng(42)
        n_obs, n_assets = 100, 20
        dates = pd.bdate_range("2020-01-01", periods=n_obs)
        assets = [f"a{i}" for i in range(n_assets)]
        ret = pd.DataFrame(
            rng.normal(0, 0.01, (n_obs, n_assets)),
            index=dates,
            columns=assets,
        )
        # Static characteristic: market cap
        chars = pd.DataFrame(
            {"market_cap": np.arange(1, n_assets + 1, dtype=float)},
            index=assets,
        )
        result = factor_mimicking_portfolios(ret, chars)
        assert result.shape[0] == n_obs
        assert not np.all(np.isnan(result.values))

    def test_time_varying_characteristic(self) -> None:
        rng = np.random.default_rng(42)
        n_obs, n_assets = 100, 20
        dates = pd.bdate_range("2020-01-01", periods=n_obs)
        assets = [f"a{i}" for i in range(n_assets)]
        ret = pd.DataFrame(
            rng.normal(0, 0.01, (n_obs, n_assets)),
            index=dates,
            columns=assets,
        )
        chars = pd.DataFrame(
            rng.normal(0, 1, (n_obs, n_assets)),
            index=dates,
            columns=assets,
        )
        result = factor_mimicking_portfolios(ret, chars)
        assert result.shape[0] == n_obs


# ---------------------------------------------------------------------------
# Risk factor decomposition
# ---------------------------------------------------------------------------


class TestRiskFactorDecomposition:
    def test_variance_decomposition_sums(self) -> None:
        ret, true_factors, _ = _make_returns()
        port_ret = ret.iloc[:, 0].values
        result = risk_factor_decomposition(port_ret, true_factors)
        total = result["total_variance"]
        parts = result["factor_variance"] + result["idiosyncratic_variance"]
        np.testing.assert_allclose(total, parts, atol=1e-10)

    def test_shares_sum_to_one(self) -> None:
        ret, true_factors, _ = _make_returns()
        port_ret = ret.iloc[:, 0].values
        result = risk_factor_decomposition(port_ret, true_factors)
        share_sum = result["factor_risk_share"] + result["idiosyncratic_risk_share"]
        np.testing.assert_allclose(share_sum, 1.0, atol=1e-10)

    def test_betas_shape(self) -> None:
        ret, true_factors, _ = _make_returns()
        port_ret = ret.iloc[:, 0].values
        result = risk_factor_decomposition(port_ret, true_factors)
        assert len(result["betas"]) == true_factors.shape[1]


# ---------------------------------------------------------------------------
# Factor correlation
# ---------------------------------------------------------------------------


class TestFactorCorrelation:
    def test_diagonal_is_one(self) -> None:
        rng = np.random.default_rng(42)
        factors = rng.normal(0, 1, (100, 4))
        result = factor_correlation(factors)
        np.testing.assert_allclose(np.diag(result["correlation"]), 1.0, atol=1e-10)

    def test_p_values_shape(self) -> None:
        rng = np.random.default_rng(42)
        factors = rng.normal(0, 1, (100, 4))
        result = factor_correlation(factors)
        assert result["p_values"].shape == (4, 4)

    def test_diagonal_p_values_zero(self) -> None:
        rng = np.random.default_rng(42)
        factors = rng.normal(0, 1, (100, 3))
        result = factor_correlation(factors)
        np.testing.assert_allclose(np.diag(result["p_values"]), 0.0)


# ---------------------------------------------------------------------------
# Common factors
# ---------------------------------------------------------------------------


class TestCommonFactors:
    def test_output_keys(self) -> None:
        rng = np.random.default_rng(42)
        ret1 = rng.normal(0, 0.01, (100, 5))
        ret2 = rng.normal(0, 0.01, (100, 8))
        result = common_factors([ret1, ret2], n_components=2)
        assert "individual_factors" in result
        assert "common_factor_scores" in result
        assert "cross_correlations" in result

    def test_common_scores_shape(self) -> None:
        rng = np.random.default_rng(42)
        ret1 = rng.normal(0, 0.01, (150, 5))
        ret2 = rng.normal(0, 0.01, (150, 8))
        result = common_factors([ret1, ret2], n_components=3)
        assert result["common_factor_scores"].shape == (150, 3)

    def test_individual_factors_count(self) -> None:
        rng = np.random.default_rng(42)
        ret1 = rng.normal(0, 0.01, (100, 5))
        ret2 = rng.normal(0, 0.01, (100, 8))
        ret3 = rng.normal(0, 0.01, (100, 4))
        result = common_factors([ret1, ret2, ret3], n_components=2)
        assert len(result["individual_factors"]) == 3
