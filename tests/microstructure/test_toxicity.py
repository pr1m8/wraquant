"""Tests for order flow toxicity metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.microstructure.toxicity import (
    adjusted_pin,
    bulk_volume_classification,
    information_share,
    informed_trading_intensity,
    order_flow_imbalance,
    pin_model,
    toxicity_index,
    trade_classification,
    vpin,
)


class TestVPIN:
    def test_output_length(self) -> None:
        n = 1000
        rng = np.random.default_rng(42)
        volume = rng.uniform(100, 200, n)
        buy_vol = volume * rng.uniform(0.3, 0.7, n)
        result = vpin(volume, buy_vol, n_buckets=20)
        assert len(result) == 20

    def test_values_bounded(self) -> None:
        n = 1000
        rng = np.random.default_rng(42)
        volume = rng.uniform(100, 200, n)
        buy_vol = volume * rng.uniform(0.3, 0.7, n)
        result = vpin(volume, buy_vol, n_buckets=20)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_balanced_flow_low_vpin(self) -> None:
        n = 1000
        volume = np.ones(n) * 100.0
        buy_vol = np.ones(n) * 50.0  # perfectly balanced
        result = vpin(volume, buy_vol, n_buckets=10)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_imbalanced_flow_high_vpin(self) -> None:
        n = 1000
        volume = np.ones(n) * 100.0
        buy_vol = np.ones(n) * 100.0  # all buys
        result = vpin(volume, buy_vol, n_buckets=10)
        np.testing.assert_allclose(result, 1.0, atol=1e-10)


class TestPINModel:
    def test_returns_dict(self) -> None:
        rng = np.random.default_rng(42)
        buy = rng.poisson(50, 60)
        sell = rng.poisson(50, 60)
        result = pin_model(buy, sell)
        assert isinstance(result, dict)
        assert "pin" in result
        assert "alpha" in result
        assert "mu" in result

    def test_pin_bounded(self) -> None:
        rng = np.random.default_rng(42)
        buy = rng.poisson(50, 60)
        sell = rng.poisson(50, 60)
        result = pin_model(buy, sell)
        assert 0 <= result["pin"] <= 1

    def test_symmetric_flow_low_pin(self) -> None:
        # When buy and sell flows are similar, PIN should be relatively low
        rng = np.random.default_rng(42)
        buy = rng.poisson(100, 100)
        sell = rng.poisson(100, 100)
        result = pin_model(buy, sell)
        assert result["pin"] < 0.8  # Should not be extremely high


class TestOrderFlowImbalance:
    def test_output_type(self) -> None:
        buy = pd.Series([100, 200, 150, 300, 250], dtype=np.float64)
        sell = pd.Series([150, 100, 200, 100, 300], dtype=np.float64)
        result = order_flow_imbalance(buy, sell, window=3)
        assert isinstance(result, pd.Series)
        assert len(result) == 5

    def test_bounded(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        buy = pd.Series(rng.uniform(50, 150, n))
        sell = pd.Series(rng.uniform(50, 150, n))
        result = order_flow_imbalance(buy, sell, window=10)
        valid = result.dropna()
        assert (valid >= -1).all()
        assert (valid <= 1).all()

    def test_all_buys_positive(self) -> None:
        buy = pd.Series([100, 100, 100, 100, 100], dtype=np.float64)
        sell = pd.Series([0, 0, 0, 0, 0], dtype=np.float64)
        result = order_flow_imbalance(buy, sell, window=3)
        valid = result.dropna()
        assert (valid > 0).all()


class TestTradeClassification:
    def test_above_mid_classified_buy(self) -> None:
        trades = pd.Series([100.05, 100.10])
        bid = pd.Series([100.0, 100.0])
        ask = pd.Series([100.10, 100.10])
        result = trade_classification(trades, bid, ask)
        # Both above midpoint -> buy
        assert (result == 1).all()

    def test_below_mid_classified_sell(self) -> None:
        trades = pd.Series([99.98, 99.95])
        bid = pd.Series([100.0, 100.0])
        ask = pd.Series([100.10, 100.10])
        result = trade_classification(trades, bid, ask)
        assert (result == -1).all()

    def test_at_mid_uses_tick_test(self) -> None:
        trades = pd.Series([100.05, 100.05])
        bid = pd.Series([100.0, 100.0])
        ask = pd.Series([100.10, 100.10])
        # midpoint = 100.05, so exactly at mid
        result = trade_classification(trades, bid, ask)
        assert len(result) == 2
        assert result.isin([1, -1]).all()


class TestInformationShare:
    def test_sums_to_one(self) -> None:
        rng = np.random.default_rng(42)
        idx = pd.bdate_range("2020-01-01", periods=100)
        p1 = pd.Series(100 + np.cumsum(rng.normal(0, 0.1, 100)), index=idx)
        p2 = pd.Series(100 + np.cumsum(rng.normal(0, 0.1, 100)), index=idx)
        result = information_share([p1, p2])
        np.testing.assert_allclose(np.sum(result), 1.0)

    def test_single_venue(self) -> None:
        idx = pd.bdate_range("2020-01-01", periods=50)
        p = pd.Series(np.linspace(100, 110, 50), index=idx)
        result = information_share([p])
        np.testing.assert_allclose(result, [1.0])


# ---------------------------------------------------------------------------
# Tests for enhanced toxicity analytics
# ---------------------------------------------------------------------------


class TestBulkVolumeClassification:
    def test_output_columns(self) -> None:
        close = pd.Series([100.0, 101.0, 99.0])
        high = pd.Series([102.0, 103.0, 101.0])
        low = pd.Series([98.0, 99.0, 97.0])
        volume = pd.Series([1000.0, 2000.0, 1500.0])
        result = bulk_volume_classification(close, high, low, volume)
        assert isinstance(result, pd.DataFrame)
        assert "buy_volume" in result.columns
        assert "sell_volume" in result.columns
        assert "buy_fraction" in result.columns

    def test_close_at_high_all_buys(self) -> None:
        """When close == high, all volume classified as buys."""
        close = pd.Series([100.0])
        high = pd.Series([100.0])
        low = pd.Series([95.0])
        volume = pd.Series([1000.0])
        result = bulk_volume_classification(close, high, low, volume)
        np.testing.assert_allclose(result["buy_fraction"].values, [1.0])
        np.testing.assert_allclose(result["buy_volume"].values, [1000.0])

    def test_close_at_low_all_sells(self) -> None:
        """When close == low, all volume classified as sells."""
        close = pd.Series([95.0])
        high = pd.Series([100.0])
        low = pd.Series([95.0])
        volume = pd.Series([1000.0])
        result = bulk_volume_classification(close, high, low, volume)
        np.testing.assert_allclose(result["buy_fraction"].values, [0.0])
        np.testing.assert_allclose(result["sell_volume"].values, [1000.0])

    def test_matches_lee_ready_simple_case(self) -> None:
        """BVC classification direction should match Lee-Ready for
        simple cases where close is unambiguously near high or low."""
        # Close near high -> buy, close near low -> sell
        close = pd.Series([99.8, 90.2])
        high = pd.Series([100.0, 100.0])
        low = pd.Series([90.0, 90.0])
        volume = pd.Series([1000.0, 1000.0])
        result = bulk_volume_classification(close, high, low, volume)
        # First bar: close near high -> buy_fraction > 0.9
        assert result["buy_fraction"].iloc[0] > 0.9
        # Second bar: close near low -> buy_fraction < 0.1
        assert result["buy_fraction"].iloc[1] < 0.1

    def test_volume_conservation(self) -> None:
        """Buy + sell volume should equal total volume."""
        rng = np.random.default_rng(42)
        n = 50
        low = pd.Series(90.0 + rng.uniform(0, 5, n))
        high = low + rng.uniform(1, 5, n)
        close = low + rng.uniform(0, 1, n) * (high - low)
        volume = pd.Series(rng.uniform(100, 1000, n))
        result = bulk_volume_classification(close, high, low, volume)
        np.testing.assert_allclose(
            result["buy_volume"].values + result["sell_volume"].values,
            volume.values,
            rtol=1e-10,
        )


class TestAdjustedPIN:
    def test_returns_dict_with_adj_pin(self) -> None:
        rng = np.random.default_rng(42)
        buy = rng.poisson(50, 60)
        sell = rng.poisson(50, 60)
        result = adjusted_pin(buy, sell)
        assert isinstance(result, dict)
        assert "adj_pin" in result
        assert "pin_unadjusted" in result
        assert "theta" in result

    def test_adj_pin_bounded(self) -> None:
        rng = np.random.default_rng(42)
        buy = rng.poisson(50, 60)
        sell = rng.poisson(50, 60)
        result = adjusted_pin(buy, sell)
        assert 0 <= result["adj_pin"] <= 1

    def test_adj_pin_leq_unadjusted(self) -> None:
        """Adjusted PIN should generally be <= unadjusted PIN."""
        rng = np.random.default_rng(42)
        buy = rng.poisson(80, 80)
        sell = rng.poisson(80, 80)
        result = adjusted_pin(buy, sell)
        # AdjPIN accounts for symmetric shocks so should be <= standard PIN
        assert result["adj_pin"] <= result["pin_unadjusted"] + 0.05  # small tolerance


class TestToxicityIndex:
    def test_bounded_0_100(self) -> None:
        rng = np.random.default_rng(42)
        n = 50
        v = rng.uniform(0, 1, n)
        o = rng.uniform(-1, 1, n)
        s = rng.uniform(0.01, 0.05, n)
        result = toxicity_index(v, o, s)
        assert np.all(result >= 0)
        assert np.all(result <= 100)

    def test_max_at_end_gives_100(self) -> None:
        """The maximum point across all inputs should score 100."""
        v = np.array([0.0, 0.5, 1.0])
        o = np.array([0.0, 0.5, 1.0])
        s = np.array([0.0, 0.5, 1.0])
        result = toxicity_index(v, o, s)
        np.testing.assert_allclose(result[-1], 100.0, atol=1e-10)

    def test_min_at_start_gives_0(self) -> None:
        """The minimum point across all inputs should score 0."""
        v = np.array([0.0, 0.5, 1.0])
        o = np.array([0.0, 0.5, 1.0])
        s = np.array([0.0, 0.5, 1.0])
        result = toxicity_index(v, o, s)
        np.testing.assert_allclose(result[0], 0.0, atol=1e-10)

    def test_custom_weights(self) -> None:
        """With weight only on VPIN and varying VPIN, score should reflect VPIN."""
        v = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        o = np.zeros(5)
        s = np.zeros(5)
        # Weight all on VPIN
        result = toxicity_index(v, o, s, weights=(1.0, 0.0, 0.0))
        # Last element (VPIN=1.0) should be 100
        np.testing.assert_allclose(result[-1], 100.0, atol=1e-10)
        # First element (VPIN=0.0) should be 0
        np.testing.assert_allclose(result[0], 0.0, atol=1e-10)


class TestInformedTradingIntensity:
    def test_bounded(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        buy = pd.Series(rng.uniform(50, 150, n))
        sell = pd.Series(rng.uniform(50, 150, n))
        result = informed_trading_intensity(buy, sell, window=10)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_all_buys_high_intensity(self) -> None:
        n = 50
        buy = pd.Series(np.ones(n) * 100.0)
        sell = pd.Series(np.zeros(n))
        result = informed_trading_intensity(buy, sell, window=10)
        valid = result.dropna()
        # All volume on one side -> intensity = 1.0
        np.testing.assert_allclose(valid.values, 1.0, atol=1e-10)
