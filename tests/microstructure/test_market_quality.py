"""Tests for market quality metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.microstructure.market_quality import (
    depth,
    gonzalo_granger_component,
    hasbrouck_information_share,
    intraday_volatility_pattern,
    market_efficiency_ratio,
    price_impact_regression,
    quoted_spread,
    relative_spread,
    resiliency,
    variance_ratio,
)


class TestQuotedSpread:
    def test_basic(self) -> None:
        bid = np.array([99.0, 100.0])
        ask = np.array([101.0, 102.0])
        result = quoted_spread(bid, ask)
        np.testing.assert_allclose(result, [2.0, 2.0])

    def test_series_input(self) -> None:
        bid = pd.Series([99.0, 100.0])
        ask = pd.Series([101.0, 102.0])
        result = quoted_spread(bid, ask)
        np.testing.assert_allclose(result, [2.0, 2.0])


class TestRelativeSpread:
    def test_basic(self) -> None:
        bid = np.array([99.0, 100.0])
        ask = np.array([101.0, 102.0])
        result = relative_spread(bid, ask)
        expected = np.array([2.0 / 100.0, 2.0 / 101.0])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_series_output(self) -> None:
        bid = pd.Series([99.0, 100.0])
        ask = pd.Series([101.0, 102.0])
        result = relative_spread(bid, ask)
        assert isinstance(result, pd.Series)

    def test_narrow_spread(self) -> None:
        bid = np.array([100.00])
        ask = np.array([100.01])
        result = relative_spread(bid, ask)
        assert result[0] < 0.001


class TestDepth:
    def test_1d_input(self) -> None:
        bid_vol = np.array([100, 200, 300, 400, 500], dtype=np.float64)
        ask_vol = np.array([150, 250, 350, 450, 550], dtype=np.float64)
        result = depth(bid_vol, ask_vol, levels=3)
        expected = (100 + 200 + 300) + (150 + 250 + 350)
        assert result == expected

    def test_2d_input(self) -> None:
        bid_vol = np.array([[100, 200, 300], [400, 500, 600]], dtype=np.float64)
        ask_vol = np.array([[150, 250, 350], [450, 550, 650]], dtype=np.float64)
        result = depth(bid_vol, ask_vol, levels=2)
        expected = np.array([100 + 200 + 150 + 250, 400 + 500 + 450 + 550], dtype=np.float64)
        np.testing.assert_allclose(result, expected)

    def test_dataframe_input(self) -> None:
        bid_vol = pd.DataFrame({"L1": [100, 200], "L2": [300, 400]})
        ask_vol = pd.DataFrame({"L1": [150, 250], "L2": [350, 450]})
        result = depth(bid_vol, ask_vol, levels=2)
        assert isinstance(result, pd.Series)


class TestResiliency:
    def test_output_type(self) -> None:
        rng = np.random.default_rng(42)
        spreads = pd.Series(rng.uniform(0.01, 0.05, 100))
        result = resiliency(spreads, window=20)
        assert isinstance(result, pd.Series)
        assert len(result) == 100


class TestVarianceRatio:
    def test_random_walk(self) -> None:
        rng = np.random.default_rng(42)
        # Geometric random walk: always positive
        prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 5000))))
        result = variance_ratio(prices, short_period=2, long_period=10)
        assert 0.8 < result["vr"] < 1.2  # Should be near 1

    def test_returns_dict(self) -> None:
        prices = pd.Series(np.linspace(100, 200, 500))
        result = variance_ratio(prices, short_period=2, long_period=10)
        assert "vr" in result
        assert "z_stat" in result
        assert "p_value" in result


# ---------------------------------------------------------------------------
# Tests for enhanced market quality analytics
# ---------------------------------------------------------------------------


class TestHasbrouckInformationShare:
    def test_sums_to_approximately_one(self) -> None:
        """Information shares should sum to ~1 across venues."""
        rng = np.random.default_rng(42)
        idx = pd.bdate_range("2020-01-01", periods=200)
        # Common efficient price + venue-specific noise
        efficient = np.cumsum(rng.normal(0, 0.1, 200))
        p1 = pd.Series(100 + efficient + rng.normal(0, 0.01, 200), index=idx)
        p2 = pd.Series(100 + efficient + rng.normal(0, 0.05, 200), index=idx)
        p3 = pd.Series(100 + efficient + rng.normal(0, 0.03, 200), index=idx)

        result = hasbrouck_information_share([p1, p2, p3])
        assert "midpoint" in result
        np.testing.assert_allclose(np.sum(result["midpoint"]), 1.0, atol=0.01)

    def test_upper_geq_lower(self) -> None:
        """Per-venue upper bound should be >= per-venue lower bound."""
        rng = np.random.default_rng(42)
        idx = pd.bdate_range("2020-01-01", periods=100)
        p1 = pd.Series(100 + np.cumsum(rng.normal(0, 0.1, 100)), index=idx)
        p2 = pd.Series(100 + np.cumsum(rng.normal(0, 0.1, 100)), index=idx)
        result = hasbrouck_information_share([p1, p2])
        # Upper is max across orderings, lower is min -- per venue
        assert np.all(result["upper"] >= result["lower"] - 1e-10)

    def test_two_venues_sum(self) -> None:
        rng = np.random.default_rng(42)
        idx = pd.bdate_range("2020-01-01", periods=100)
        p1 = pd.Series(100 + np.cumsum(rng.normal(0, 0.1, 100)), index=idx)
        p2 = pd.Series(100 + np.cumsum(rng.normal(0, 0.1, 100)), index=idx)
        result = hasbrouck_information_share([p1, p2])
        np.testing.assert_allclose(np.sum(result["midpoint"]), 1.0, atol=0.05)


class TestGonzaloGrangerComponent:
    def test_weights_sum_to_one(self) -> None:
        rng = np.random.default_rng(42)
        idx = pd.bdate_range("2020-01-01", periods=200)
        efficient = np.cumsum(rng.normal(0, 0.1, 200))
        p1 = pd.Series(100 + efficient + rng.normal(0, 0.01, 200), index=idx)
        p2 = pd.Series(100 + efficient + rng.normal(0, 0.05, 200), index=idx)
        result = gonzalo_granger_component([p1, p2])
        assert "gg_weights" in result
        np.testing.assert_allclose(np.sum(result["gg_weights"]), 1.0, atol=1e-10)

    def test_single_venue(self) -> None:
        idx = pd.bdate_range("2020-01-01", periods=50)
        p = pd.Series(np.linspace(100, 110, 50), index=idx)
        result = gonzalo_granger_component([p])
        np.testing.assert_allclose(result["gg_weights"], [1.0])


class TestMarketEfficiencyRatio:
    def test_random_walk_efficient(self) -> None:
        rng = np.random.default_rng(42)
        prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 5000))))
        result = market_efficiency_ratio(prices)
        assert "efficiency_score" in result
        assert "variance_ratios" in result
        # Random walk should be relatively efficient
        assert result["efficiency_score"] < 0.3

    def test_variance_ratios_returned(self) -> None:
        rng = np.random.default_rng(42)
        prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500))))
        result = market_efficiency_ratio(prices, lags=[2, 5, 10])
        assert 2 in result["variance_ratios"]
        assert 5 in result["variance_ratios"]
        assert 10 in result["variance_ratios"]


class TestPriceImpactRegression:
    def test_returns_correct_keys(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        signed_vol = pd.Series(rng.normal(0, 100, n))
        dp = pd.Series(0.001 * signed_vol + rng.normal(0, 0.1, n))
        result = price_impact_regression(dp, signed_vol, lags=3)
        assert "permanent_impact" in result
        assert "temporary_impact" in result
        assert "beta_0" in result
        assert "r_squared" in result

    def test_positive_permanent_impact(self) -> None:
        """When prices respond to order flow, permanent impact should be positive."""
        rng = np.random.default_rng(42)
        n = 500
        signed_vol = pd.Series(rng.normal(0, 100, n))
        # Price changes positively correlated with signed volume
        dp = pd.Series(0.01 * signed_vol + rng.normal(0, 0.1, n))
        result = price_impact_regression(dp, signed_vol, lags=3)
        assert result["permanent_impact"] > 0

    def test_r_squared_bounded(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        signed_vol = pd.Series(rng.normal(0, 100, n))
        dp = pd.Series(rng.normal(0, 1, n))
        result = price_impact_regression(dp, signed_vol, lags=3)
        assert 0 <= result["r_squared"] <= 1


class TestIntradayVolatilityPattern:
    def test_returns_series(self) -> None:
        # Create intraday data
        idx = pd.date_range("2020-01-02 09:30", "2020-01-02 16:00", freq="5min")
        rng = np.random.default_rng(42)
        prices = pd.Series(100.0 + np.cumsum(rng.normal(0, 0.01, len(idx))), index=idx)
        result = intraday_volatility_pattern(prices, freq="h")
        assert isinstance(result, pd.Series)
        assert len(result) > 0

    def test_positive_values(self) -> None:
        idx = pd.date_range("2020-01-02 09:30", "2020-01-03 16:00", freq="5min")
        rng = np.random.default_rng(42)
        prices = pd.Series(100.0 + np.cumsum(rng.normal(0, 0.01, len(idx))), index=idx)
        result = intraday_volatility_pattern(prices, freq="h")
        assert (result >= 0).all()
