"""Tests for forex analysis functions."""

from __future__ import annotations

import pytest

from wraquant.forex.analysis import (
    lot_size,
    margin_call_price,
    pip_distance,
    pip_value,
    pips,
    position_value,
    risk_reward_ratio,
)


class TestPips:
    def test_non_jpy(self) -> None:
        assert pips(0.0050) == 50.0

    def test_jpy(self) -> None:
        assert pips(0.50, is_jpy=True) == 50.0


class TestPipValue:
    def test_standard_lot(self) -> None:
        pv = pip_value(lot_size_units=100_000)
        assert pv == 10.0

    def test_mini_lot(self) -> None:
        pv = pip_value(lot_size_units=10_000)
        assert pv == 1.0


class TestLotSize:
    def test_risk_calculation(self) -> None:
        # $10,000 account, 1% risk, 50 pip stop loss
        lots = lot_size(10_000, risk_percent=1.0, stop_loss_pips=50)
        assert abs(lots - 0.2) < 0.01

    def test_zero_stop_loss(self) -> None:
        lots = lot_size(10_000, risk_percent=1.0, stop_loss_pips=0)
        assert lots == 0.0


class TestPipDistance:
    def test_positive_distance(self) -> None:
        dist = pip_distance(1.1000, 1.1050)
        assert dist == pytest.approx(50.0, abs=0.1)

    def test_negative_distance(self) -> None:
        dist = pip_distance(1.1050, 1.1000)
        assert dist == pytest.approx(-50.0, abs=0.1)

    def test_jpy_pair_flag(self) -> None:
        dist = pip_distance(110.00, 110.50, is_jpy=True)
        assert dist == pytest.approx(50.0, abs=0.1)

    def test_jpy_pair_string(self) -> None:
        dist = pip_distance(110.00, 110.50, pair="USDJPY")
        assert dist == pytest.approx(50.0, abs=0.1)

    def test_zero_distance(self) -> None:
        dist = pip_distance(1.1000, 1.1000)
        assert dist == pytest.approx(0.0)


class TestPositionValue:
    def test_profit(self) -> None:
        pnl = position_value(lots=1.0, pip_val=10.0, pips_moved=50)
        assert pnl == pytest.approx(500.0)

    def test_loss(self) -> None:
        pnl = position_value(lots=0.5, pip_val=10.0, pips_moved=-30)
        assert pnl == pytest.approx(-150.0)

    def test_zero_pips(self) -> None:
        pnl = position_value(lots=1.0, pip_val=10.0, pips_moved=0)
        assert pnl == pytest.approx(0.0)

    def test_fractional_lot(self) -> None:
        pnl = position_value(lots=0.1, pip_val=10.0, pips_moved=100)
        assert pnl == pytest.approx(100.0)


class TestRiskRewardRatio:
    def test_basic_rr(self) -> None:
        result = risk_reward_ratio(1.1000, 1.0950, 1.1100)
        assert result["ratio"] == pytest.approx(2.0)
        assert result["risk_pips"] == pytest.approx(50.0, abs=0.1)
        assert result["reward_pips"] == pytest.approx(100.0, abs=0.1)

    def test_short_trade(self) -> None:
        # Short entry at 1.1100, stop at 1.1150, target at 1.1000
        result = risk_reward_ratio(1.1100, 1.1150, 1.1000)
        assert result["ratio"] == pytest.approx(2.0)
        assert result["risk_pips"] == pytest.approx(50.0, abs=0.1)

    def test_jpy_pair(self) -> None:
        result = risk_reward_ratio(110.00, 109.50, 111.00, is_jpy=True)
        assert result["ratio"] == pytest.approx(2.0)
        assert result["risk_pips"] == pytest.approx(50.0, abs=0.1)
        assert result["reward_pips"] == pytest.approx(100.0, abs=0.1)

    def test_zero_risk(self) -> None:
        result = risk_reward_ratio(1.1000, 1.1000, 1.1050)
        assert result["ratio"] == float("inf")

    def test_returns_dict_keys(self) -> None:
        result = risk_reward_ratio(1.1000, 1.0950, 1.1100)
        assert "ratio" in result
        assert "risk_pips" in result
        assert "reward_pips" in result


class TestMarginCallPrice:
    def test_long_below_entry(self) -> None:
        mc = margin_call_price(1.1000, 10_000, 2_000, 50.0)
        assert mc < 1.1000

    def test_short_above_entry(self) -> None:
        mc = margin_call_price(1.1000, 10_000, 2_000, 50.0, side="short")
        assert mc > 1.1000

    def test_zero_entry(self) -> None:
        mc = margin_call_price(0.0, 10_000, 2_000, 50.0)
        assert mc == 0.0

    def test_long_non_negative(self) -> None:
        # Even with very large balance, margin call price >= 0
        mc = margin_call_price(1.1000, 1_000_000, 2_000, 50.0)
        assert mc >= 0.0

    def test_invalid_side(self) -> None:
        with pytest.raises(ValueError, match="side must be"):
            margin_call_price(1.1000, 10_000, 2_000, 50.0, side="hold")
