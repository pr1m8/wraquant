"""Tests for forex carry trade functions."""

from __future__ import annotations

import pandas as pd
import pytest

from wraquant.forex.carry import (
    carry_attractiveness,
    carry_portfolio,
    carry_return,
    forward_premium,
    interest_rate_differential,
    uncovered_interest_parity,
)


class TestInterestRateDifferential:
    def test_positive_differential(self) -> None:
        diff = interest_rate_differential(0.05, 0.01)
        assert diff == pytest.approx(0.04)

    def test_negative_differential(self) -> None:
        diff = interest_rate_differential(0.01, 0.05)
        assert diff == pytest.approx(-0.04)


class TestCarryReturn:
    def test_carry_adds_return(self) -> None:
        spot_returns = pd.Series([0.001, -0.002, 0.0005, 0.001])
        total = carry_return(spot_returns, base_rate=0.05, quote_rate=0.01)
        assert total.iloc[0] > spot_returns.iloc[0]

    def test_zero_carry(self) -> None:
        spot_returns = pd.Series([0.001, -0.002])
        total = carry_return(spot_returns, base_rate=0.03, quote_rate=0.03)
        pd.testing.assert_series_equal(total, spot_returns)


class TestForwardPremium:
    def test_discount_when_base_higher(self) -> None:
        fwd = forward_premium(1.1000, base_rate=0.04, quote_rate=0.02)
        assert fwd < 1.1000  # base currency at forward discount


class TestCarryAttractiveness:
    def test_sorted_by_differential(self) -> None:
        rates = {"USD": 0.05, "JPY": 0.001, "EUR": 0.04}
        df = carry_attractiveness(rates)
        assert df.iloc[0]["pair"] == "USDJPY"  # highest carry

    def test_custom_pairs(self) -> None:
        rates = {"USD": 0.05, "JPY": 0.001, "EUR": 0.04}
        df = carry_attractiveness(rates, pairs=[("USD", "JPY")])
        assert len(df) == 1


class TestCarryPortfolio:
    def test_basic_portfolio(self) -> None:
        rates = {
            "USD": 0.05,
            "JPY": 0.001,
            "AUD": 0.04,
            "EUR": 0.03,
            "CHF": 0.015,
            "NZD": 0.045,
        }
        result = carry_portfolio(rates, n_long=2, n_short=2)
        assert result["expected_carry"] > 0
        assert len(result["long_currencies"]) == 2
        assert len(result["short_currencies"]) == 2

    def test_weights_dollar_neutral(self) -> None:
        rates = {"USD": 0.05, "JPY": 0.001, "AUD": 0.04, "EUR": 0.03}
        result = carry_portfolio(rates, n_long=2, n_short=2)
        total_weight = sum(result["weights"].values())
        assert total_weight == pytest.approx(0.0, abs=1e-10)

    def test_long_currencies_have_positive_weights(self) -> None:
        rates = {"USD": 0.05, "JPY": 0.001, "AUD": 0.04, "EUR": 0.03}
        result = carry_portfolio(rates, n_long=2, n_short=2)
        for ccy in result["long_currencies"]:
            assert result["weights"][ccy] > 0

    def test_short_currencies_have_negative_weights(self) -> None:
        rates = {"USD": 0.05, "JPY": 0.001, "AUD": 0.04, "EUR": 0.03}
        result = carry_portfolio(rates, n_long=2, n_short=2)
        for ccy in result["short_currencies"]:
            assert result["weights"][ccy] < 0

    def test_custom_weights(self) -> None:
        rates = {"USD": 0.05, "JPY": 0.001, "AUD": 0.04}
        custom = {"USD": 0.6, "AUD": 0.4, "JPY": -1.0}
        result = carry_portfolio(rates, weights=custom, n_long=2, n_short=1)
        assert result["weights"]["USD"] == pytest.approx(0.6)


class TestUncoveredInterestParity:
    def test_higher_domestic_rate_implies_depreciation(self) -> None:
        result = uncovered_interest_parity(0.05, 0.01, 1.1000)
        assert result["forward_rate"] > 1.1000
        assert result["forward_premium"] > 0

    def test_equal_rates(self) -> None:
        result = uncovered_interest_parity(0.03, 0.03, 1.1000)
        assert result["forward_rate"] == pytest.approx(1.1000)
        assert result["forward_premium"] == pytest.approx(0.0, abs=1e-10)

    def test_shorter_maturity(self) -> None:
        result_1y = uncovered_interest_parity(0.05, 0.01, 1.1000, maturity=1.0)
        result_6m = uncovered_interest_parity(0.05, 0.01, 1.1000, maturity=0.5)
        # Shorter maturity -> smaller premium
        assert abs(result_6m["forward_premium"]) < abs(result_1y["forward_premium"])

    def test_returns_dict_keys(self) -> None:
        result = uncovered_interest_parity(0.05, 0.01, 1.1000)
        assert "forward_rate" in result
        assert "forward_premium" in result
