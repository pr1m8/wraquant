"""Tests for wraquant.core.types."""

from __future__ import annotations

from wraquant.core.types import (
    AssetClass,
    Currency,
    Frequency,
    OptionType,
    OrderSide,
    RegimeState,
    ReturnType,
    RiskMeasure,
    VolModel,
)


class TestEnums:
    def test_frequency_values(self) -> None:
        assert Frequency.DAILY == "1d"
        assert Frequency.HOURLY == "1h"
        assert Frequency.TICK == "tick"

    def test_asset_class_values(self) -> None:
        assert AssetClass.EQUITY == "equity"
        assert AssetClass.FX == "fx"
        assert AssetClass.CRYPTO == "crypto"

    def test_currency_major_pairs(self) -> None:
        assert Currency.USD == "USD"
        assert Currency.EUR == "EUR"
        assert Currency.JPY == "JPY"
        assert Currency.GBP == "GBP"

    def test_return_type(self) -> None:
        assert ReturnType.SIMPLE == "simple"
        assert ReturnType.LOG == "log"

    def test_option_type(self) -> None:
        assert OptionType.CALL == "call"
        assert OptionType.PUT == "put"

    def test_order_side(self) -> None:
        assert OrderSide.BUY == "buy"
        assert OrderSide.SELL == "sell"

    def test_regime_state(self) -> None:
        assert RegimeState.BULL == "bull"
        assert RegimeState.CRISIS == "crisis"

    def test_risk_measure(self) -> None:
        assert RiskMeasure.VAR == "var"
        assert RiskMeasure.CVAR == "cvar"

    def test_vol_model(self) -> None:
        assert VolModel.GARCH == "garch"
        assert VolModel.HESTON == "heston"
        assert VolModel.SABR == "sabr"
