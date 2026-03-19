"""Tests for wraquant.risk.tail module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.risk.tail import (
    conditional_drawdown_at_risk,
    cornish_fisher_var,
    drawdown_at_risk,
    expected_shortfall_decomposition,
    tail_ratio_analysis,
)


@pytest.fixture()
def return_data():
    """Generate synthetic return data."""
    np.random.seed(42)
    return pd.Series(np.random.normal(0.0005, 0.01, 1000))


@pytest.fixture()
def multi_asset_data():
    """Generate synthetic multi-asset return data."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame(
        {
            "A": np.random.normal(0.0005, 0.01, n),
            "B": np.random.normal(0.0003, 0.015, n),
        }
    )


class TestCornishFisherVar:
    """Tests for cornish_fisher_var."""

    def test_returns_dict(self, return_data):
        result = cornish_fisher_var(return_data, alpha=0.05)
        assert "cf_var" in result
        assert "normal_var" in result
        assert "z_cf" in result
        assert "z_normal" in result
        assert "skewness" in result
        assert "excess_kurtosis" in result
        assert "adjustment_factor" in result

    def test_var_positive(self, return_data):
        result = cornish_fisher_var(return_data)
        assert result["cf_var"] > 0
        assert result["normal_var"] > 0

    def test_near_normal_close_to_parametric(self):
        """For near-normal data, CF-VaR should be close to normal VaR."""
        np.random.seed(42)
        normal_data = pd.Series(np.random.normal(0, 0.01, 10000))
        result = cornish_fisher_var(normal_data)
        assert abs(result["cf_var"] - result["normal_var"]) / result["normal_var"] < 0.1

    def test_fat_tailed_kurtosis_detected(self):
        """For fat-tailed data, excess kurtosis should be positive."""
        np.random.seed(42)
        # Student-t with df=3 has excess kurtosis
        fat_data = pd.Series(np.random.standard_t(3, size=5000) * 0.01)
        result = cornish_fisher_var(fat_data)
        assert result["excess_kurtosis"] > 1.0
        # CF quantile should differ from normal quantile
        assert result["z_cf"] != pytest.approx(result["z_normal"], abs=0.01)

    def test_different_alpha(self, return_data):
        var_95 = cornish_fisher_var(return_data, alpha=0.05)
        var_99 = cornish_fisher_var(return_data, alpha=0.01)
        # 99% VaR should be larger than 95% VaR
        assert var_99["cf_var"] > var_95["cf_var"]


class TestExpectedShortfallDecomposition:
    """Tests for expected_shortfall_decomposition."""

    def test_returns_series(self, multi_asset_data):
        weights = np.array([0.6, 0.4])
        result = expected_shortfall_decomposition(weights, multi_asset_data)
        assert isinstance(result, pd.Series)
        assert len(result) == 2

    def test_asset_names(self, multi_asset_data):
        weights = np.array([0.6, 0.4])
        result = expected_shortfall_decomposition(weights, multi_asset_data)
        assert list(result.index) == ["A", "B"]

    def test_positive_total(self, multi_asset_data):
        weights = np.array([0.6, 0.4])
        result = expected_shortfall_decomposition(weights, multi_asset_data)
        assert result.sum() > 0

    def test_higher_vol_higher_contribution(self):
        """Asset B (higher vol) should contribute more to ES."""
        np.random.seed(42)
        returns = pd.DataFrame(
            {
                "safe": np.random.normal(0.001, 0.005, 500),
                "risky": np.random.normal(0.001, 0.03, 500),
            }
        )
        weights = np.array([0.5, 0.5])
        result = expected_shortfall_decomposition(weights, returns)
        assert result["risky"] > result["safe"]


class TestConditionalDrawdownAtRisk:
    """Tests for conditional_drawdown_at_risk."""

    def test_positive(self, return_data):
        cdar = conditional_drawdown_at_risk(return_data, alpha=0.05)
        assert cdar >= 0

    def test_higher_alpha_lower_cdar(self, return_data):
        cdar_5 = conditional_drawdown_at_risk(return_data, alpha=0.05)
        cdar_20 = conditional_drawdown_at_risk(return_data, alpha=0.20)
        # Averaging over more (less extreme) drawdowns should give lower CDaR
        assert cdar_20 <= cdar_5

    def test_no_drawdown(self):
        """Monotonically increasing prices should have zero CDaR."""
        returns = pd.Series(np.ones(100) * 0.01)
        cdar = conditional_drawdown_at_risk(returns, alpha=0.05)
        assert cdar == pytest.approx(0.0, abs=1e-10)


class TestTailRatioAnalysis:
    """Tests for tail_ratio_analysis."""

    def test_returns_dict(self, return_data):
        result = tail_ratio_analysis(return_data)
        assert "tail_ratio" in result
        assert "right_tail" in result
        assert "left_tail" in result
        assert "tail_ratio_99" in result
        assert "skewness" in result
        assert "excess_kurtosis" in result
        assert "interpretation" in result

    def test_symmetric_near_one(self):
        """For symmetric data, tail ratio should be near 1.0."""
        np.random.seed(42)
        symmetric = pd.Series(np.random.normal(0, 0.01, 10000))
        result = tail_ratio_analysis(symmetric)
        assert abs(result["tail_ratio"] - 1.0) < 0.3

    def test_right_skewed_above_one(self):
        """Right-skewed data should have tail ratio > 1."""
        np.random.seed(42)
        right_skewed = pd.Series(np.random.lognormal(0, 0.5, 10000) - 1)
        result = tail_ratio_analysis(right_skewed)
        assert result["tail_ratio"] > 1.0

    def test_interpretation_string(self, return_data):
        result = tail_ratio_analysis(return_data)
        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 10


class TestDrawdownAtRisk:
    """Tests for drawdown_at_risk."""

    def test_positive(self, return_data):
        dar = drawdown_at_risk(return_data, alpha=0.05)
        assert dar >= 0

    def test_higher_alpha_lower_dar(self, return_data):
        dar_1 = drawdown_at_risk(return_data, alpha=0.01)
        dar_10 = drawdown_at_risk(return_data, alpha=0.10)
        # More extreme quantile should give higher DaR
        assert dar_1 >= dar_10

    def test_no_drawdown(self):
        """Monotonically increasing should have zero DaR."""
        returns = pd.Series(np.ones(100) * 0.01)
        dar = drawdown_at_risk(returns, alpha=0.05)
        assert dar == pytest.approx(0.0, abs=1e-10)

    def test_dar_less_than_max_drawdown(self, return_data):
        """DaR at 5% should be less than or equal to max drawdown."""
        from wraquant.risk.metrics import max_drawdown

        cum = (1 + return_data).cumprod()
        mdd = abs(max_drawdown(cum))
        dar = drawdown_at_risk(return_data, alpha=0.01)
        assert dar <= mdd + 1e-10
