"""Tests for forex analysis functions."""

from __future__ import annotations

from wraquant.forex.analysis import lot_size, pip_value, pips


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
