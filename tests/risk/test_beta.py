"""Tests for wraquant.risk.beta module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.risk.beta import (
    blume_adjusted_beta,
    conditional_beta,
    dimson_beta,
    ewma_beta,
    rolling_beta,
    vasicek_adjusted_beta,
)


@pytest.fixture()
def market_data():
    """Generate synthetic market and stock return data."""
    np.random.seed(42)
    n = 500
    market = pd.Series(np.random.normal(0.0005, 0.01, n), name="market")
    stock = 1.2 * market + np.random.normal(0, 0.005, n)
    stock = pd.Series(stock, name="stock")
    return stock, market


class TestRollingBeta:
    """Tests for rolling_beta."""

    def test_returns_series(self, market_data):
        stock, market = market_data
        result = rolling_beta(stock, market, window=60)
        assert isinstance(result, pd.Series)
        assert len(result) == len(stock)

    def test_initial_nans(self, market_data):
        stock, market = market_data
        result = rolling_beta(stock, market, window=60)
        assert result.iloc[:59].isna().all()

    def test_beta_close_to_true(self, market_data):
        stock, market = market_data
        result = rolling_beta(stock, market, window=120)
        # After warm-up, beta should be close to 1.2
        valid = result.dropna()
        median_beta = valid.median()
        assert abs(median_beta - 1.2) < 0.3

    def test_window_effect(self, market_data):
        stock, market = market_data
        short = rolling_beta(stock, market, window=20)
        long_ = rolling_beta(stock, market, window=120)
        # Short window should be noisier (higher std)
        assert short.dropna().std() > long_.dropna().std()


class TestBlumeAdjustedBeta:
    """Tests for blume_adjusted_beta."""

    def test_formula(self):
        assert blume_adjusted_beta(1.0) == pytest.approx(1.0)
        assert blume_adjusted_beta(1.5) == pytest.approx(1.335)
        assert blume_adjusted_beta(0.5) == pytest.approx(0.665)

    def test_shrinks_toward_one(self):
        # High beta shrinks down
        assert blume_adjusted_beta(2.0) < 2.0
        # Low beta shrinks up
        assert blume_adjusted_beta(0.3) > 0.3

    def test_zero_beta(self):
        assert blume_adjusted_beta(0.0) == pytest.approx(0.33)

    def test_negative_beta(self):
        result = blume_adjusted_beta(-0.5)
        assert result == pytest.approx(0.33 + 0.67 * (-0.5))


class TestVasicekAdjustedBeta:
    """Tests for vasicek_adjusted_beta."""

    def test_formula(self):
        result = vasicek_adjusted_beta(1.5, 1.0, 0.2, 0.3)
        # w_raw = 0.09 / 0.13 = 0.6923
        # w_prior = 0.04 / 0.13 = 0.3077
        expected = (0.09 / 0.13) * 1.5 + (0.04 / 0.13) * 1.0
        assert result == pytest.approx(expected, rel=1e-6)

    def test_high_se_more_shrinkage(self):
        low_se = vasicek_adjusted_beta(1.5, 1.0, raw_se=0.1, prior_se=0.3)
        high_se = vasicek_adjusted_beta(1.5, 1.0, raw_se=0.5, prior_se=0.3)
        # High SE should shrink more toward 1.0
        assert abs(high_se - 1.0) < abs(low_se - 1.0)

    def test_returns_prior_when_se_infinite(self):
        result = vasicek_adjusted_beta(1.5, 1.0, raw_se=1000.0, prior_se=0.3)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_returns_raw_when_prior_se_infinite(self):
        result = vasicek_adjusted_beta(1.5, 1.0, raw_se=0.2, prior_se=1000.0)
        assert result == pytest.approx(1.5, abs=0.01)


class TestDimsonBeta:
    """Tests for dimson_beta."""

    def test_returns_dict(self, market_data):
        stock, market = market_data
        result = dimson_beta(stock, market, lags=1)
        assert "total_beta" in result
        assert "lag_betas" in result
        assert "alpha" in result
        assert "r_squared" in result

    def test_total_is_sum_of_lags(self, market_data):
        stock, market = market_data
        result = dimson_beta(stock, market, lags=2)
        assert result["total_beta"] == pytest.approx(
            sum(result["lag_betas"]), rel=1e-10
        )

    def test_lag_count(self, market_data):
        stock, market = market_data
        result = dimson_beta(stock, market, lags=3)
        assert len(result["lag_betas"]) == 4  # contemporaneous + 3 lags

    def test_illiquid_asset(self):
        """Dimson beta should be higher for lagged reactions."""
        np.random.seed(42)
        n = 500
        market = pd.Series(np.random.normal(0.0005, 0.01, n))
        # Asset with delayed reaction
        stock = (
            0.5 * market
            + 0.4 * market.shift(1).fillna(0)
            + np.random.normal(0, 0.003, n)
        )
        result = dimson_beta(stock, market, lags=1)
        assert result["total_beta"] > result["lag_betas"][0]

    def test_r_squared_range(self, market_data):
        stock, market = market_data
        result = dimson_beta(stock, market, lags=1)
        assert 0.0 <= result["r_squared"] <= 1.0


class TestConditionalBeta:
    """Tests for conditional_beta."""

    def test_returns_dict(self, market_data):
        stock, market = market_data
        result = conditional_beta(stock, market)
        assert "upside_beta" in result
        assert "downside_beta" in result
        assert "beta_asymmetry" in result
        assert "n_up" in result
        assert "n_down" in result

    def test_asymmetry_calculation(self, market_data):
        stock, market = market_data
        result = conditional_beta(stock, market)
        assert result["beta_asymmetry"] == pytest.approx(
            result["downside_beta"] - result["upside_beta"], rel=1e-10
        )

    def test_sample_counts(self, market_data):
        stock, market = market_data
        result = conditional_beta(stock, market)
        # Should use all observations
        assert result["n_up"] + result["n_down"] == len(
            pd.concat([stock, market], axis=1).dropna()
        )

    def test_symmetric_returns(self):
        """For symmetric returns, up and down betas should be similar."""
        np.random.seed(42)
        n = 2000
        market = pd.Series(np.random.normal(0, 0.01, n))
        stock = 1.0 * market + np.random.normal(0, 0.003, n)
        result = conditional_beta(pd.Series(stock), market)
        assert abs(result["upside_beta"] - result["downside_beta"]) < 0.4


class TestEwmaBeta:
    """Tests for ewma_beta."""

    def test_returns_series(self, market_data):
        stock, market = market_data
        result = ewma_beta(stock, market, halflife=60)
        assert isinstance(result, pd.Series)
        assert len(result) == len(pd.concat([stock, market], axis=1).dropna())

    def test_beta_close_to_true(self, market_data):
        stock, market = market_data
        result = ewma_beta(stock, market, halflife=60)
        valid = result.dropna()
        median_beta = valid.median()
        assert abs(median_beta - 1.2) < 0.4

    def test_different_halflife(self, market_data):
        stock, market = market_data
        short = ewma_beta(stock, market, halflife=20)
        long_ = ewma_beta(stock, market, halflife=120)
        # Both should exist
        assert short.dropna().shape[0] > 0
        assert long_.dropna().shape[0] > 0
