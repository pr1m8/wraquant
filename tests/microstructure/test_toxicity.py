"""Tests for order flow toxicity metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.microstructure.toxicity import (
    information_share,
    order_flow_imbalance,
    pin_model,
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
