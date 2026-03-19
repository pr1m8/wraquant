"""Tests for transaction cost analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.execution.cost import (
    commission_cost,
    expected_cost_model,
    market_impact_model,
    slippage,
    total_cost,
    transaction_cost_analysis,
)


class TestSlippage:
    def test_buy_positive(self) -> None:
        result = slippage(101.0, 100.0, side="buy")
        assert result == pytest.approx(1.0)

    def test_buy_negative(self) -> None:
        result = slippage(99.0, 100.0, side="buy")
        assert result == pytest.approx(-1.0)

    def test_sell_positive(self) -> None:
        result = slippage(99.0, 100.0, side="sell")
        assert result == pytest.approx(1.0)

    def test_sell_negative(self) -> None:
        result = slippage(101.0, 100.0, side="sell")
        assert result == pytest.approx(-1.0)

    def test_array_input(self) -> None:
        ep = np.array([101.0, 99.0, 100.5])
        bp = np.array([100.0, 100.0, 100.0])
        result = slippage(ep, bp, side="buy")
        np.testing.assert_allclose(result, [1.0, -1.0, 0.5])

    def test_invalid_side(self) -> None:
        with pytest.raises(ValueError):
            slippage(100.0, 100.0, side="hold")


class TestCommissionCost:
    def test_basic(self) -> None:
        result = commission_cost(100, 50.0, rate=0.001)
        assert result == pytest.approx(5.0)

    def test_array(self) -> None:
        qty = np.array([100, 200])
        price = np.array([50.0, 25.0])
        result = commission_cost(qty, price, rate=0.001)
        np.testing.assert_allclose(result, [5.0, 5.0])


class TestTotalCost:
    def test_basic_breakdown(self) -> None:
        trades = pd.DataFrame({
            "execution_price": [101.0, 102.0],
            "benchmark_price": [100.0, 100.0],
            "qty": [100, 200],
            "side": ["buy", "buy"],
        })
        result = total_cost(trades, commission_rate=0.001)
        assert result["n_trades"] == 2
        assert result["total_slippage"] > 0
        assert result["total_commission"] > 0
        assert result["total_cost"] > 0
        assert result["cost_bps"] > 0

    def test_missing_columns(self) -> None:
        trades = pd.DataFrame({"price": [100.0]})
        with pytest.raises(ValueError):
            total_cost(trades)


class TestMarketImpactModel:
    def test_sqrt_model(self) -> None:
        result = market_impact_model(10_000, 1_000_000, 0.02, model="sqrt")
        assert result > 0

    def test_linear_model(self) -> None:
        result = market_impact_model(10_000, 1_000_000, 0.02, model="linear")
        assert result > 0

    def test_sqrt_greater_for_small_orders(self) -> None:
        # For participation < 1, sqrt(p) > p, so sqrt impact > linear impact
        sqrt_impact = market_impact_model(10_000, 1_000_000, 0.02, model="sqrt")
        linear_impact = market_impact_model(10_000, 1_000_000, 0.02, model="linear")
        assert sqrt_impact > linear_impact

    def test_invalid_model(self) -> None:
        with pytest.raises(ValueError):
            market_impact_model(10_000, 1_000_000, 0.02, model="cubic")

    def test_impact_scales_with_size(self) -> None:
        small = market_impact_model(1_000, 1_000_000, 0.02, model="sqrt")
        large = market_impact_model(100_000, 1_000_000, 0.02, model="sqrt")
        assert large > small


class TestExpectedCostModel:
    def test_spread_cost(self) -> None:
        result = expected_cost_model(5000, 100.0, 1_000_000, 0.02, 0.02)
        assert result["spread_cost"] == pytest.approx(50.0)

    def test_total_greater_than_spread(self) -> None:
        result = expected_cost_model(5000, 100.0, 1_000_000, 0.02, 0.02)
        assert result["total_cost"] > result["spread_cost"]

    def test_all_components_positive(self) -> None:
        result = expected_cost_model(10_000, 50.0, 500_000, 0.015, 0.01)
        assert result["spread_cost"] > 0
        assert result["impact_cost"] > 0
        assert result["timing_risk"] >= 0
        assert result["cost_bps"] > 0

    def test_returns_dict_keys(self) -> None:
        result = expected_cost_model(1000, 100.0, 1_000_000, 0.02, 0.02)
        expected_keys = {"spread_cost", "impact_cost", "timing_risk", "total_cost", "cost_bps"}
        assert expected_keys == set(result.keys())

    def test_larger_order_costs_more(self) -> None:
        small = expected_cost_model(1000, 100.0, 1_000_000, 0.02, 0.02)
        large = expected_cost_model(50_000, 100.0, 1_000_000, 0.02, 0.02)
        assert large["total_cost"] > small["total_cost"]


class TestTransactionCostAnalysis:
    def test_basic_tca(self) -> None:
        trades = pd.DataFrame({
            "execution_price": [100.05, 100.10, 100.08],
            "qty": [1000, 2000, 1500],
            "side": ["buy", "buy", "buy"],
        })
        market = pd.DataFrame({
            "arrival_price": [100.00],
            "vwap": [100.06],
            "close": [100.12],
        })
        tca = transaction_cost_analysis(trades, market)
        assert "arrival_cost_bps" in tca.columns
        assert "vwap_cost_bps" in tca.columns
        assert "close_cost_bps" in tca.columns

    def test_buy_arrival_cost_positive(self) -> None:
        trades = pd.DataFrame({
            "execution_price": [100.10],
            "qty": [1000],
            "side": ["buy"],
        })
        market = pd.DataFrame({
            "arrival_price": [100.00],
            "vwap": [100.05],
            "close": [100.15],
        })
        tca = transaction_cost_analysis(trades, market)
        assert tca["arrival_cost"].iloc[0] > 0  # paid more than arrival

    def test_sell_cost_sign(self) -> None:
        trades = pd.DataFrame({
            "execution_price": [99.90],
            "qty": [1000],
            "side": ["sell"],
        })
        market = pd.DataFrame({
            "arrival_price": [100.00],
            "vwap": [100.00],
            "close": [100.00],
        })
        tca = transaction_cost_analysis(trades, market)
        # Sold below arrival -> positive cost for seller
        assert tca["arrival_cost"].iloc[0] > 0

    def test_per_trade_market_data(self) -> None:
        trades = pd.DataFrame({
            "execution_price": [100.05, 100.10],
            "qty": [1000, 2000],
            "side": ["buy", "buy"],
        })
        market = pd.DataFrame({
            "arrival_price": [100.00, 100.05],
            "vwap": [100.02, 100.07],
            "close": [100.10, 100.12],
        })
        tca = transaction_cost_analysis(trades, market)
        assert len(tca) == 2
        assert tca["arrival_cost"].iloc[0] == pytest.approx(0.05)

    def test_output_preserves_original_columns(self) -> None:
        trades = pd.DataFrame({
            "execution_price": [100.05],
            "qty": [1000],
            "side": ["buy"],
        })
        market = pd.DataFrame({
            "arrival_price": [100.00],
            "vwap": [100.02],
            "close": [100.10],
        })
        tca = transaction_cost_analysis(trades, market)
        assert "execution_price" in tca.columns
        assert "qty" in tca.columns
