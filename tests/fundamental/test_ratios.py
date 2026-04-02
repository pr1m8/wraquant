"""Tests for wraquant.fundamental.ratios and valuation modules."""

from __future__ import annotations

import pandas as pd
import pytest

from wraquant.fundamental.ratios import (
    current_ratio,
    debt_to_equity,
    operating_margin,
    pb_ratio,
    pe_ratio,
    roe,
)
from wraquant.fundamental.valuation import (
    dcf_valuation,
    piotroski_f_score,
    quality_screen,
)

# ---------------------------------------------------------------------------
# Financial ratios
# ---------------------------------------------------------------------------


class TestPERatio:
    def test_basic(self):
        assert pe_ratio(150.0, 7.5) == 20.0

    def test_zero_earnings(self):
        assert pe_ratio(100.0, 0.0) == 0.0

    def test_negative_earnings(self):
        result = pe_ratio(100.0, -5.0)
        assert result == -20.0

    def test_returns_float(self):
        assert isinstance(pe_ratio(100, 5), float)


class TestPBRatio:
    def test_basic(self):
        assert pb_ratio(100.0, 50.0) == 2.0

    def test_zero_book(self):
        assert pb_ratio(100.0, 0.0) == 0.0

    def test_returns_float(self):
        assert isinstance(pb_ratio(100, 25), float)


class TestROE:
    def test_basic(self):
        result = roe(1_000_000, 5_000_000)
        assert abs(result - 0.2) < 1e-10

    def test_zero_equity(self):
        assert roe(100.0, 0.0) == 0.0

    def test_returns_float(self):
        assert isinstance(roe(100, 500), float)


class TestDebtToEquity:
    def test_basic(self):
        result = debt_to_equity(2_000_000, 5_000_000)
        assert abs(result - 0.4) < 1e-10

    def test_zero_equity(self):
        assert debt_to_equity(100.0, 0.0) == 0.0

    def test_returns_float(self):
        assert isinstance(debt_to_equity(100, 200), float)


class TestCurrentRatio:
    def test_basic(self):
        result = current_ratio(500_000, 300_000)
        assert abs(result - 5 / 3) < 1e-10

    def test_zero_liabilities(self):
        assert current_ratio(100.0, 0.0) == 0.0

    def test_returns_float(self):
        assert isinstance(current_ratio(100, 50), float)


class TestOperatingMargin:
    def test_basic(self):
        result = operating_margin(200_000, 1_000_000)
        assert abs(result - 0.2) < 1e-10

    def test_zero_revenue(self):
        assert operating_margin(100.0, 0.0) == 0.0

    def test_returns_float(self):
        assert isinstance(operating_margin(20, 100), float)


# ---------------------------------------------------------------------------
# Piotroski F-Score
# ---------------------------------------------------------------------------


class TestPiotroskiFScore:
    def test_perfect_score(self):
        financials = {
            "net_income": 1e6,
            "prev_net_income": 8e5,
            "operating_cash_flow": 1.2e6,
            "total_assets": 5e6,
            "prev_total_assets": 4.8e6,
            "long_term_debt": 1e6,
            "prev_long_term_debt": 1.1e6,
            "current_ratio": 1.5,
            "prev_current_ratio": 1.3,
            "shares_outstanding": 1e6,
            "prev_shares_outstanding": 1e6,
            "gross_margin": 0.4,
            "prev_gross_margin": 0.38,
            "asset_turnover": 0.8,
            "prev_asset_turnover": 0.75,
        }
        assert piotroski_f_score(financials) == 9

    def test_zero_score(self):
        financials = {
            "net_income": -1e6,
            "prev_net_income": 1e6,
            "operating_cash_flow": -5e5,
            "total_assets": 5e6,
            "prev_total_assets": 4e6,
            "long_term_debt": 2e6,
            "prev_long_term_debt": 1e6,
            "current_ratio": 0.8,
            "prev_current_ratio": 1.2,
            "shares_outstanding": 2e6,
            "prev_shares_outstanding": 1e6,
            "gross_margin": 0.3,
            "prev_gross_margin": 0.4,
            "asset_turnover": 0.5,
            "prev_asset_turnover": 0.7,
        }
        assert piotroski_f_score(financials) == 0

    def test_returns_int(self):
        result = piotroski_f_score({"net_income": 100, "total_assets": 1000})
        assert isinstance(result, int)
        assert 0 <= result <= 9

    def test_empty_dict(self):
        result = piotroski_f_score({})
        assert isinstance(result, int)
        assert 0 <= result <= 9


# ---------------------------------------------------------------------------
# DCF valuation
# ---------------------------------------------------------------------------


class TestDCFValuation:
    def test_basic(self):
        result = dcf_valuation(
            [100, 110, 121], discount_rate=0.10, terminal_growth=0.02
        )
        assert isinstance(result["present_value"], float)
        assert result["present_value"] > 0
        assert result["pv_cash_flows"] > 0
        assert result["pv_terminal"] > 0
        assert result["terminal_value"] > 0

    def test_terminal_value_dominates(self):
        result = dcf_valuation([100], discount_rate=0.10, terminal_growth=0.02)
        assert result["pv_terminal"] > result["pv_cash_flows"]

    def test_invalid_rates(self):
        with pytest.raises(ValueError, match="discount_rate"):
            dcf_valuation([100], discount_rate=0.02, terminal_growth=0.05)

    def test_empty_cash_flows(self):
        result = dcf_valuation([], discount_rate=0.10)
        assert result["present_value"] == 0.0

    def test_single_cash_flow(self):
        result = dcf_valuation([100], discount_rate=0.10, terminal_growth=0.02)
        expected_pv_cf = 100 / 1.10
        expected_tv = 100 * 1.02 / 0.08
        expected_pv_tv = expected_tv / 1.10
        assert abs(result["pv_cash_flows"] - expected_pv_cf) < 1e-6
        assert abs(result["pv_terminal"] - expected_pv_tv) < 1e-4


# ---------------------------------------------------------------------------
# Quality screen
# ---------------------------------------------------------------------------


class TestQualityScreen:
    def test_basic(self):
        stocks = pd.DataFrame(
            {
                "roe": [0.25, 0.30, 0.20],
                "operating_margin": [0.30, 0.35, 0.25],
                "current_ratio": [1.5, 2.0, 3.0],
            },
            index=["AAPL", "MSFT", "GOOG"],
        )
        result = quality_screen(stocks)
        assert "quality_score" in result.columns
        assert "quality_rank" in result.columns
        assert result["quality_rank"].iloc[0] == 1
        assert len(result) == 3

    def test_custom_metrics(self):
        stocks = pd.DataFrame(
            {
                "roe": [0.25, 0.30, 0.20],
                "foo": [10, 20, 30],
            },
            index=["A", "B", "C"],
        )
        result = quality_screen(stocks, metrics=["roe"])
        assert "quality_score" in result.columns

    def test_no_matching_metrics(self):
        stocks = pd.DataFrame({"price": [100, 200, 300]}, index=["A", "B", "C"])
        result = quality_screen(stocks, metrics=["roe"])
        assert "quality_score" in result.columns
        assert (result["quality_score"] == 0.5).all()

    def test_returns_dataframe(self):
        stocks = pd.DataFrame({"roe": [0.1, 0.2]}, index=["A", "B"])
        result = quality_screen(stocks)
        assert isinstance(result, pd.DataFrame)
