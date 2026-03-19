"""Tests for wraquant.risk.portfolio_analytics module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.risk.portfolio_analytics import (
    active_share,
    component_var,
    concentration_ratio,
    diversification_ratio,
    incremental_var,
    marginal_var,
    risk_budgeting,
    tracking_error,
)


@pytest.fixture()
def portfolio_data():
    """Generate synthetic multi-asset return data."""
    np.random.seed(42)
    n = 500
    returns = pd.DataFrame(
        {
            "A": np.random.normal(0.0005, 0.01, n),
            "B": np.random.normal(0.0003, 0.015, n),
            "C": np.random.normal(0.0004, 0.008, n),
        }
    )
    weights = np.array([0.4, 0.35, 0.25])
    return returns, weights


class TestComponentVar:
    """Tests for component_var."""

    def test_returns_series(self, portfolio_data):
        returns, weights = portfolio_data
        result = component_var(weights, returns, alpha=0.05)
        assert isinstance(result, pd.Series)
        assert len(result) == 3

    def test_sum_equals_total_var(self, portfolio_data):
        returns, weights = portfolio_data
        cov = returns.cov().values
        from scipy import stats as sp_stats

        z = sp_stats.norm.ppf(0.05)
        port_vol = np.sqrt(weights @ cov @ weights)
        total_var = -z * port_vol

        result = component_var(weights, returns, alpha=0.05)
        assert result.sum() == pytest.approx(total_var, rel=0.01)

    def test_asset_names(self, portfolio_data):
        returns, weights = portfolio_data
        result = component_var(weights, returns)
        assert list(result.index) == ["A", "B", "C"]

    def test_positive_values(self, portfolio_data):
        returns, weights = portfolio_data
        result = component_var(weights, returns)
        # All components should be positive for positive weights
        assert (result > 0).all()


class TestMarginalVar:
    """Tests for marginal_var."""

    def test_returns_array(self, portfolio_data):
        returns, weights = portfolio_data
        cov = returns.cov().values
        result = marginal_var(weights, cov, alpha=0.05)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_positive_values(self, portfolio_data):
        returns, weights = portfolio_data
        cov = returns.cov().values
        result = marginal_var(weights, cov)
        assert (result > 0).all()

    def test_higher_vol_higher_marginal(self):
        """Asset with higher volatility should have higher marginal VaR."""
        cov = np.array(
            [
                [0.01, 0.0, 0.0],
                [0.0, 0.04, 0.0],
                [0.0, 0.0, 0.0025],
            ]
        )
        weights = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        result = marginal_var(weights, cov)
        # B (highest vol) should have highest marginal VaR
        assert result[1] > result[0]
        assert result[1] > result[2]


class TestIncrementalVar:
    """Tests for incremental_var."""

    def test_returns_array(self, portfolio_data):
        returns, weights = portfolio_data
        result = incremental_var(weights, returns, alpha=0.05)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_values_finite(self, portfolio_data):
        """All incremental VaR values should be finite."""
        returns, weights = portfolio_data
        result = incremental_var(weights, returns)
        assert np.all(np.isfinite(result))


class TestRiskBudgeting:
    """Tests for risk_budgeting."""

    def test_equal_risk_contribution(self):
        cov = np.array([[0.04, 0.006], [0.006, 0.01]])
        result = risk_budgeting(cov)
        assert "weights" in result
        assert "risk_contributions" in result
        assert "portfolio_vol" in result
        assert "converged" in result

    def test_weights_sum_to_one(self):
        cov = np.array([[0.04, 0.006], [0.006, 0.01]])
        result = risk_budgeting(cov)
        assert np.sum(result["weights"]) == pytest.approx(1.0, abs=1e-6)

    def test_equal_contributions(self):
        cov = np.array([[0.04, 0.006], [0.006, 0.01]])
        result = risk_budgeting(cov)
        rc = result["risk_contributions"]
        assert np.allclose(rc, 0.5, atol=0.05)

    def test_custom_budget(self):
        cov = np.array([[0.04, 0.006], [0.006, 0.01]])
        target = np.array([0.7, 0.3])
        result = risk_budgeting(cov, target_risk=target)
        rc = result["risk_contributions"]
        assert abs(rc[0] - 0.7) < 0.1

    def test_three_asset(self):
        np.random.seed(42)
        data = np.random.normal(0, 0.01, (252, 3))
        cov = np.cov(data, rowvar=False)
        result = risk_budgeting(cov)
        assert len(result["weights"]) == 3
        assert np.allclose(result["risk_contributions"], 1.0 / 3, atol=0.05)


class TestDiversificationRatio:
    """Tests for diversification_ratio."""

    def test_uncorrelated(self):
        cov = np.diag([0.04, 0.01])
        weights = np.array([0.5, 0.5])
        dr = diversification_ratio(weights, cov)
        assert dr > 1.0

    def test_perfectly_correlated(self):
        """Perfect correlation = DR of 1.0."""
        vols = np.array([0.2, 0.1])
        corr = np.array([[1.0, 1.0], [1.0, 1.0]])
        cov = np.outer(vols, vols) * corr
        weights = np.array([0.5, 0.5])
        dr = diversification_ratio(weights, cov)
        assert dr == pytest.approx(1.0, abs=0.01)

    def test_single_asset(self):
        cov = np.array([[0.04]])
        weights = np.array([1.0])
        dr = diversification_ratio(weights, cov)
        assert dr == pytest.approx(1.0)


class TestConcentrationRatio:
    """Tests for concentration_ratio."""

    def test_equal_risk(self):
        """Equal vol + equal weight + zero corr -> CR = 1/n."""
        cov = np.diag([0.04, 0.04])
        weights = np.array([0.5, 0.5])
        cr = concentration_ratio(weights, cov)
        assert cr == pytest.approx(0.5, abs=0.01)

    def test_single_asset(self):
        cov = np.array([[0.04, 0.0], [0.0, 0.01]])
        weights = np.array([1.0, 0.0])
        cr = concentration_ratio(weights, cov)
        assert cr == pytest.approx(1.0, abs=0.01)

    def test_range(self, portfolio_data):
        returns, weights = portfolio_data
        cov = returns.cov().values
        cr = concentration_ratio(weights, cov)
        assert 1.0 / 3 <= cr <= 1.0


class TestTrackingError:
    """Tests for tracking_error."""

    def test_returns_dict(self):
        np.random.seed(42)
        portfolio = pd.Series(np.random.normal(0.0005, 0.01, 252))
        benchmark = pd.Series(np.random.normal(0.0004, 0.009, 252))
        result = tracking_error(portfolio, benchmark)
        assert "tracking_error" in result
        assert "information_ratio" in result
        assert "active_return" in result
        assert "max_active_drawdown" in result

    def test_te_positive(self):
        np.random.seed(42)
        portfolio = pd.Series(np.random.normal(0.0005, 0.01, 252))
        benchmark = pd.Series(np.random.normal(0.0004, 0.009, 252))
        result = tracking_error(portfolio, benchmark)
        assert result["tracking_error"] > 0

    def test_identical_returns_zero_te(self):
        returns = pd.Series(np.random.normal(0, 0.01, 100))
        result = tracking_error(returns, returns)
        assert result["tracking_error"] == pytest.approx(0.0, abs=1e-10)


class TestActiveShare:
    """Tests for active_share."""

    def test_identical_weights(self):
        w = np.array([0.25, 0.25, 0.25, 0.25])
        assert active_share(w, w) == pytest.approx(0.0)

    def test_maximum_difference(self):
        w1 = np.array([1.0, 0.0])
        w2 = np.array([0.0, 1.0])
        assert active_share(w1, w2) == pytest.approx(1.0)

    def test_range(self):
        np.random.seed(42)
        w1 = np.random.dirichlet([1, 1, 1, 1])
        w2 = np.random.dirichlet([1, 1, 1, 1])
        result = active_share(w1, w2)
        assert 0.0 <= result <= 1.0

    def test_symmetric(self):
        w1 = np.array([0.4, 0.3, 0.2, 0.1])
        w2 = np.array([0.25, 0.25, 0.25, 0.25])
        assert active_share(w1, w2) == active_share(w2, w1)
