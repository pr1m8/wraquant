"""Tests for wraquant.risk.factor module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.risk.factor import (
    factor_contribution,
    factor_risk_model,
    fama_french_regression,
    statistical_factor_model,
)


@pytest.fixture()
def factor_data():
    """Generate synthetic factor and return data."""
    np.random.seed(42)
    n = 500
    mkt = np.random.normal(0.0005, 0.01, n)
    smb = np.random.normal(0, 0.005, n)
    hml = np.random.normal(0, 0.005, n)
    stock = 1.1 * mkt + 0.3 * smb - 0.2 * hml + np.random.normal(0, 0.004, n)
    factors = pd.DataFrame({"MKT": mkt, "SMB": smb, "HML": hml})
    returns = pd.Series(stock)
    return returns, factors


class TestFactorRiskModel:
    """Tests for factor_risk_model."""

    def test_returns_dict(self, factor_data):
        returns, factors = factor_data
        result = factor_risk_model(returns, factors)
        assert "betas" in result
        assert "alpha" in result
        assert "factor_risk" in result
        assert "specific_risk" in result
        assert "r_squared" in result
        assert "residual_vol" in result
        assert "contributions" in result

    def test_betas_close_to_true(self, factor_data):
        returns, factors = factor_data
        result = factor_risk_model(returns, factors)
        assert abs(result["betas"]["MKT"] - 1.1) < 0.15
        assert abs(result["betas"]["SMB"] - 0.3) < 0.15
        assert abs(result["betas"]["HML"] - (-0.2)) < 0.15

    def test_risk_decomposition_sums_to_one(self, factor_data):
        returns, factors = factor_data
        result = factor_risk_model(returns, factors)
        assert result["factor_risk"] + result["specific_risk"] == pytest.approx(
            1.0, abs=0.01
        )

    def test_r_squared_range(self, factor_data):
        returns, factors = factor_data
        result = factor_risk_model(returns, factors)
        assert 0.0 <= result["r_squared"] <= 1.0

    def test_high_factor_risk(self, factor_data):
        """True model has high factor loading, so factor risk should dominate."""
        returns, factors = factor_data
        result = factor_risk_model(returns, factors)
        assert result["factor_risk"] > 0.5

    def test_contributions_keys(self, factor_data):
        returns, factors = factor_data
        result = factor_risk_model(returns, factors)
        assert set(result["contributions"].keys()) == {"MKT", "SMB", "HML"}

    def test_dataframe_input(self, factor_data):
        returns, factors = factor_data
        returns_df = pd.DataFrame({"stock": returns})
        result = factor_risk_model(returns_df, factors)
        assert "betas" in result


class TestStatisticalFactorModel:
    """Tests for statistical_factor_model."""

    def test_returns_dict(self):
        np.random.seed(42)
        market = np.random.normal(0, 0.01, 252)
        returns = pd.DataFrame(
            {
                f"asset_{i}": market * (0.5 + i * 0.2) + np.random.normal(0, 0.005, 252)
                for i in range(5)
            }
        )
        result = statistical_factor_model(returns, n_factors=2)
        assert "factors" in result
        assert "loadings" in result
        assert "explained_variance" in result
        assert "explained_variance_ratio" in result
        assert "factor_risk" in result
        assert "specific_risk" in result

    def test_factors_shape(self):
        np.random.seed(42)
        n, k = 252, 5
        returns = pd.DataFrame(
            np.random.normal(0, 0.01, (n, k)),
            columns=[f"a{i}" for i in range(k)],
        )
        result = statistical_factor_model(returns, n_factors=3)
        assert result["factors"].shape == (n, 3)
        assert result["loadings"].shape == (k, 3)

    def test_variance_ratio_sums(self):
        np.random.seed(42)
        returns = pd.DataFrame(np.random.normal(0, 0.01, (252, 5)))
        result = statistical_factor_model(returns, n_factors=5)
        assert result["factor_risk"] == pytest.approx(1.0, abs=0.01)

    def test_cumulative_monotonic(self):
        np.random.seed(42)
        returns = pd.DataFrame(np.random.normal(0, 0.01, (252, 5)))
        result = statistical_factor_model(returns, n_factors=3)
        cum = result["cumulative_variance_ratio"]
        assert all(cum[i] <= cum[i + 1] for i in range(len(cum) - 1))


class TestFamaFrenchRegression:
    """Tests for fama_french_regression."""

    def test_returns_dict(self, factor_data):
        returns, factors = factor_data
        result = fama_french_regression(returns, factors)
        assert "alpha" in result
        assert "betas" in result
        assert "t_stats" in result
        assert "p_values" in result
        assert "r_squared" in result
        assert "adj_r_squared" in result
        assert "residual_vol" in result

    def test_betas_match_true(self, factor_data):
        returns, factors = factor_data
        result = fama_french_regression(returns, factors)
        assert abs(result["betas"]["MKT"] - 1.1) < 0.15
        assert abs(result["betas"]["SMB"] - 0.3) < 0.15

    def test_t_stats_significant(self, factor_data):
        returns, factors = factor_data
        result = fama_french_regression(returns, factors)
        # MKT should be significant
        assert abs(result["t_stats"]["MKT"]) > 2.0

    def test_p_values_range(self, factor_data):
        returns, factors = factor_data
        result = fama_french_regression(returns, factors)
        for _key, pval in result["p_values"].items():
            assert 0.0 <= pval <= 1.0

    def test_adj_r_squared_less_than_r_squared(self, factor_data):
        returns, factors = factor_data
        result = fama_french_regression(returns, factors)
        assert result["adj_r_squared"] <= result["r_squared"]


class TestFactorContribution:
    """Tests for factor_contribution."""

    def test_returns_dict(self):
        weights = np.array([0.3, 0.3, 0.4])
        betas = np.array([[1.0, 0.5], [1.2, -0.3], [0.8, 0.1]])
        factor_cov = np.array([[0.0004, 0.00005], [0.00005, 0.0001]])
        result = factor_contribution(weights, betas, factor_cov)
        assert "total_factor_var" in result
        assert "total_factor_vol" in result
        assert "factor_contributions" in result
        assert "factor_pct_contributions" in result

    def test_pct_contributions_sum_to_one(self):
        weights = np.array([0.3, 0.3, 0.4])
        betas = np.array([[1.0, 0.5], [1.2, -0.3], [0.8, 0.1]])
        factor_cov = np.array([[0.0004, 0.00005], [0.00005, 0.0001]])
        result = factor_contribution(weights, betas, factor_cov)
        assert np.sum(result["factor_pct_contributions"]) == pytest.approx(
            1.0, abs=0.01
        )

    def test_total_var_positive(self):
        weights = np.array([0.5, 0.5])
        betas = np.array([[1.0, 0.3], [0.8, -0.1]])
        factor_cov = np.array([[0.0004, 0.0001], [0.0001, 0.0002]])
        result = factor_contribution(weights, betas, factor_cov)
        assert result["total_factor_var"] > 0
        assert result["total_factor_vol"] > 0

    def test_vol_is_sqrt_var(self):
        weights = np.array([0.5, 0.5])
        betas = np.array([[1.0, 0.3], [0.8, -0.1]])
        factor_cov = np.array([[0.0004, 0.0001], [0.0001, 0.0002]])
        result = factor_contribution(weights, betas, factor_cov)
        assert result["total_factor_vol"] == pytest.approx(
            np.sqrt(result["total_factor_var"]), rel=1e-10
        )
