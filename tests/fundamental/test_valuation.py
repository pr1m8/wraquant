"""Tests for wraquant.fundamental.valuation — FMP-backed valuation models.

All FMP API calls are mocked.  No network requests are made.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Common mock data
# ---------------------------------------------------------------------------

_CASH_FLOW = [
    {
        "date": "2024-12-31",
        "freeCashFlow": 80_000,
        "operatingCashFlow": 110_000,
        "capitalExpenditure": -30_000,
    },
    {
        "date": "2023-12-31",
        "freeCashFlow": 70_000,
        "operatingCashFlow": 95_000,
        "capitalExpenditure": -25_000,
    },
    {
        "date": "2022-12-31",
        "freeCashFlow": 60_000,
        "operatingCashFlow": 85_000,
        "capitalExpenditure": -25_000,
    },
    {
        "date": "2021-12-31",
        "freeCashFlow": 50_000,
        "operatingCashFlow": 75_000,
        "capitalExpenditure": -25_000,
    },
]

_PROFILE = [
    {
        "companyName": "Test Corp",
        "price": 150.0,
        "mktCap": 3_000_000,
        "lastDiv": 3.0,
        "sector": "Technology",
        "industry": "Software",
    }
]

_BALANCE = [
    {
        "totalDebt": 100_000,
        "cashAndCashEquivalents": 50_000,
        "totalStockholdersEquity": 250_000,
        "totalAssets": 500_000,
        "commonStock": 20_000,
    },
]

_INCOME = [
    {
        "date": "2024-12-31",
        "revenue": 400_000,
        "netIncome": 60_000,
        "operatingIncome": 100_000,
    },
]

_KEY_METRICS = [
    {
        "netIncomePerShare": 3.0,
        "bookValuePerShare": 12.5,
        "dividendYield": 0.02,
        "evToOperatingCashFlow": 18.0,
    }
]

_RATIOS_TTM = {
    "peRatioTTM": 25.0,
    "priceToBookRatioTTM": 4.0,
    "priceToSalesRatioTTM": 6.0,
    "pegRatioTTM": 1.5,
    "dividendYieldTTM": 0.02,
    "priceToFreeCashFlowsRatioTTM": 20.0,
}

_FMP_DCF = [{"dcf": 180.0}]

_GROWTH = [
    {"epsgrowth": 0.15, "dividendsperShareGrowth": 0.05, "revenueGrowth": 0.14},
]


@pytest.fixture()
def mock_fmp():
    client = MagicMock()
    client.cash_flow.return_value = _CASH_FLOW
    client.company_profile.return_value = _PROFILE
    client.balance_sheet.return_value = _BALANCE
    client.income_statement.return_value = _INCOME
    client.key_metrics.return_value = _KEY_METRICS
    client.ratios_ttm.return_value = _RATIOS_TTM
    client.dcf.return_value = _FMP_DCF
    client.financial_growth.return_value = _GROWTH
    client.enterprise_value.return_value = [{"evToEBITDA": 15.0}]
    client.ratios.return_value = [{"returnOnEquity": 0.24, "returnOnAssets": 0.12}]
    return client


_PATCH_CHECK = patch("wraquant._lazy.check_extra", return_value=True)


# ---------------------------------------------------------------------------
# DCF valuation
# ---------------------------------------------------------------------------


class TestDCFValuation:
    @_PATCH_CHECK
    def test_keys(self, _ce, mock_fmp):
        from wraquant.fundamental.valuation import dcf_valuation

        result = dcf_valuation(
            "AAPL", growth_rate=0.10, discount_rate=0.12, fmp_client=mock_fmp
        )
        expected_keys = {
            "intrinsic_value",
            "equity_value",
            "intrinsic_value_per_share",
            "current_price",
            "margin_of_safety",
            "upside_potential",
            "pv_cash_flows",
            "pv_terminal",
            "terminal_value",
            "terminal_pct",
            "projected_fcf",
            "assumptions",
            "fmp_dcf",
        }
        assert expected_keys == set(result.keys())

    @_PATCH_CHECK
    def test_positive_intrinsic_value(self, _ce, mock_fmp):
        from wraquant.fundamental.valuation import dcf_valuation

        result = dcf_valuation(
            "AAPL", growth_rate=0.10, discount_rate=0.12, fmp_client=mock_fmp
        )
        assert result["intrinsic_value"] > 0
        assert result["pv_cash_flows"] > 0
        assert result["pv_terminal"] > 0

    @_PATCH_CHECK
    def test_projected_fcf_length(self, _ce, mock_fmp):
        from wraquant.fundamental.valuation import dcf_valuation

        result = dcf_valuation(
            "AAPL",
            growth_rate=0.10,
            discount_rate=0.12,
            projection_years=7,
            fmp_client=mock_fmp,
        )
        assert len(result["projected_fcf"]) == 7

    @_PATCH_CHECK
    def test_discount_lte_terminal_raises(self, _ce, mock_fmp):
        from wraquant.fundamental.valuation import dcf_valuation

        with pytest.raises(ValueError, match="discount_rate"):
            dcf_valuation(
                "AAPL",
                growth_rate=0.10,
                discount_rate=0.02,
                terminal_growth=0.05,
                fmp_client=mock_fmp,
            )

    @_PATCH_CHECK
    def test_assumptions_stored(self, _ce, mock_fmp):
        from wraquant.fundamental.valuation import dcf_valuation

        result = dcf_valuation(
            "AAPL",
            growth_rate=0.08,
            discount_rate=0.11,
            terminal_growth=0.03,
            fmp_client=mock_fmp,
        )
        assert result["assumptions"]["growth_rate"] == 0.08
        assert result["assumptions"]["discount_rate"] == 0.11
        assert result["assumptions"]["terminal_growth"] == 0.03

    @_PATCH_CHECK
    def test_fmp_dcf_passthrough(self, _ce, mock_fmp):
        from wraquant.fundamental.valuation import dcf_valuation

        result = dcf_valuation(
            "AAPL", growth_rate=0.10, discount_rate=0.12, fmp_client=mock_fmp
        )
        assert result["fmp_dcf"] == 180.0

    @_PATCH_CHECK
    def test_auto_growth_rate(self, _ce, mock_fmp):
        """When growth_rate is None, it should be estimated from FCF history."""
        from wraquant.fundamental.valuation import dcf_valuation

        result = dcf_valuation("AAPL", discount_rate=0.12, fmp_client=mock_fmp)
        assert result["assumptions"]["growth_rate"] != 0.0
        assert result["intrinsic_value"] > 0


# ---------------------------------------------------------------------------
# Relative valuation
# ---------------------------------------------------------------------------


class TestRelativeValuation:
    @_PATCH_CHECK
    def test_keys(self, _ce, mock_fmp):
        from wraquant.fundamental.valuation import relative_valuation

        result = relative_valuation("AAPL", peers=["MSFT", "GOOG"], fmp_client=mock_fmp)
        expected_keys = {
            "symbol",
            "multiples",
            "peer_medians",
            "peer_means",
            "premium_discount",
            "peers_data",
            "verdict",
        }
        assert expected_keys == set(result.keys())

    @_PATCH_CHECK
    def test_symbol_passthrough(self, _ce, mock_fmp):
        from wraquant.fundamental.valuation import relative_valuation

        result = relative_valuation("AAPL", peers=["MSFT"], fmp_client=mock_fmp)
        assert result["symbol"] == "AAPL"

    @_PATCH_CHECK
    def test_verdict_values(self, _ce, mock_fmp):
        from wraquant.fundamental.valuation import relative_valuation

        result = relative_valuation("AAPL", peers=["MSFT"], fmp_client=mock_fmp)
        assert result["verdict"] in {"undervalued", "fairly valued", "overvalued"}

    @_PATCH_CHECK
    def test_empty_peers(self, _ce, mock_fmp):
        from wraquant.fundamental.valuation import relative_valuation

        result = relative_valuation("AAPL", peers=[], fmp_client=mock_fmp)
        assert result["peers_data"] == []
        assert result["verdict"] in {"undervalued", "fairly valued", "overvalued"}


# ---------------------------------------------------------------------------
# Graham Number
# ---------------------------------------------------------------------------


class TestGrahamNumber:
    @_PATCH_CHECK
    def test_keys(self, _ce, mock_fmp):
        from wraquant.fundamental.valuation import graham_number

        result = graham_number("JNJ", fmp_client=mock_fmp)
        expected_keys = {
            "graham_number",
            "current_price",
            "margin_of_safety",
            "eps",
            "bvps",
        }
        assert expected_keys == set(result.keys())

    @_PATCH_CHECK
    def test_formula(self, _ce, mock_fmp):
        from wraquant.fundamental.valuation import graham_number

        result = graham_number("JNJ", fmp_client=mock_fmp)
        expected = math.sqrt(22.5 * 3.0 * 12.5)
        assert abs(result["graham_number"] - expected) < 1e-6

    @_PATCH_CHECK
    def test_positive_mos_when_cheap(self, _ce, mock_fmp):
        from wraquant.fundamental.valuation import graham_number

        result = graham_number("JNJ", fmp_client=mock_fmp)
        gn = result["graham_number"]
        price = result["current_price"]
        # Our mock has price=150, gn should be ~29.05 so MoS is negative
        if gn < price:
            assert result["margin_of_safety"] < 0

    @_PATCH_CHECK
    def test_zero_eps(self, _ce):
        from wraquant.fundamental.valuation import graham_number

        client = MagicMock()
        client.key_metrics.return_value = [
            {"netIncomePerShare": 0, "bookValuePerShare": 10}
        ]
        client.company_profile.return_value = [{"price": 50.0}]
        result = graham_number("LOSS", fmp_client=client)
        assert result["graham_number"] == 0.0


# ---------------------------------------------------------------------------
# Peter Lynch Value
# ---------------------------------------------------------------------------


class TestPeterLynchValue:
    @_PATCH_CHECK
    def test_keys(self, _ce, mock_fmp):
        from wraquant.fundamental.valuation import peter_lynch_value

        result = peter_lynch_value("NVDA", fmp_client=mock_fmp)
        expected_keys = {
            "fair_value",
            "current_price",
            "peg_ratio",
            "pe_ratio",
            "eps_growth_rate",
            "margin_of_safety",
            "lynch_category",
        }
        assert expected_keys == set(result.keys())

    @_PATCH_CHECK
    def test_category_values(self, _ce, mock_fmp):
        from wraquant.fundamental.valuation import peter_lynch_value

        result = peter_lynch_value("NVDA", fmp_client=mock_fmp)
        assert result["lynch_category"] in {
            "undervalued",
            "fairly valued",
            "overvalued",
        }

    @_PATCH_CHECK
    def test_peg_computation(self, _ce, mock_fmp):
        from wraquant.fundamental.valuation import peter_lynch_value

        result = peter_lynch_value("NVDA", fmp_client=mock_fmp)
        # PE=25, eps_growth=0.15 => eps_growth_pct=15 => PEG=25/15=1.667
        assert abs(result["peg_ratio"] - 25.0 / 15.0) < 1e-6


# ---------------------------------------------------------------------------
# Dividend Discount Model
# ---------------------------------------------------------------------------


class TestDividendDiscountModel:
    @_PATCH_CHECK
    def test_keys(self, _ce, mock_fmp):
        from wraquant.fundamental.valuation import dividend_discount_model

        result = dividend_discount_model("KO", fmp_client=mock_fmp)
        expected_keys = {
            "intrinsic_value",
            "current_price",
            "margin_of_safety",
            "dividend_per_share",
            "dividend_growth_rate",
            "dividend_yield",
            "implied_return",
            "model_applicable",
        }
        assert expected_keys == set(result.keys())

    @_PATCH_CHECK
    def test_gordon_growth_formula(self, _ce, mock_fmp):
        from wraquant.fundamental.valuation import dividend_discount_model

        result = dividend_discount_model(
            "KO", required_return=0.10, fmp_client=mock_fmp
        )
        # D0=3.0, g=0.05, r=0.10 => D1=3.15 => V=3.15/(0.10-0.05)=63
        assert result["model_applicable"] == "yes"
        expected = 3.0 * 1.05 / (0.10 - 0.05)
        assert abs(result["intrinsic_value"] - expected) < 1e-4

    @_PATCH_CHECK
    def test_no_dividend(self, _ce):
        from wraquant.fundamental.valuation import dividend_discount_model

        client = MagicMock()
        client.company_profile.return_value = [{"price": 100.0, "lastDiv": 0}]
        client.ratios_ttm.return_value = {"dividendYieldTTM": 0}
        client.financial_growth.return_value = [{"dividendsperShareGrowth": 0}]
        client.key_metrics.return_value = [{"dividendYield": 0}]
        result = dividend_discount_model("AMZN", fmp_client=client)
        assert result["model_applicable"] == "no"
        assert result["intrinsic_value"] == 0.0

    @_PATCH_CHECK
    def test_growth_exceeds_required_return(self, _ce):
        from wraquant.fundamental.valuation import dividend_discount_model

        client = MagicMock()
        client.company_profile.return_value = [{"price": 100.0, "lastDiv": 2.0}]
        client.ratios_ttm.return_value = {"dividendYieldTTM": 0.02}
        client.financial_growth.return_value = [{"dividendsperShareGrowth": 0.15}]
        client.key_metrics.return_value = [{"dividendYield": 0.02}]
        result = dividend_discount_model(
            "FAST", required_return=0.10, fmp_client=client
        )
        assert result["model_applicable"] == "no"


# ---------------------------------------------------------------------------
# Residual Income Model
# ---------------------------------------------------------------------------


class TestResidualIncomeModel:
    @_PATCH_CHECK
    def test_keys(self, _ce, mock_fmp):
        from wraquant.fundamental.valuation import residual_income_model

        result = residual_income_model("JPM", fmp_client=mock_fmp)
        expected_keys = {
            "intrinsic_value",
            "current_price",
            "margin_of_safety",
            "book_value_per_share",
            "current_roe",
            "residual_income",
            "pv_residual_income",
            "pv_terminal",
            "excess_return_spread",
        }
        assert expected_keys == set(result.keys())

    @_PATCH_CHECK
    def test_positive_excess_spread(self, _ce, mock_fmp):
        from wraquant.fundamental.valuation import residual_income_model

        result = residual_income_model("JPM", cost_of_equity=0.10, fmp_client=mock_fmp)
        # ROE = 60000/250000 = 0.24 > cost_of_equity=0.10
        assert result["excess_return_spread"] > 0
        assert result["current_roe"] > 0.10

    @_PATCH_CHECK
    def test_intrinsic_above_book(self, _ce, mock_fmp):
        """When ROE > CoE, intrinsic > book value."""
        from wraquant.fundamental.valuation import residual_income_model

        result = residual_income_model("JPM", cost_of_equity=0.10, fmp_client=mock_fmp)
        assert result["intrinsic_value"] > result["book_value_per_share"]


# ---------------------------------------------------------------------------
# Piotroski F-Score (standalone function)
# ---------------------------------------------------------------------------


class TestPiotroskiFScore:
    def test_perfect_score(self):
        from wraquant.fundamental.valuation import piotroski_f_score

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
        from wraquant.fundamental.valuation import piotroski_f_score

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

    def test_range(self):
        from wraquant.fundamental.valuation import piotroski_f_score

        result = piotroski_f_score({})
        assert 0 <= result <= 9

    def test_returns_int(self):
        from wraquant.fundamental.valuation import piotroski_f_score

        assert isinstance(
            piotroski_f_score({"net_income": 100, "total_assets": 1000}), int
        )


# ---------------------------------------------------------------------------
# Quality Screen (standalone function — no FMP mock needed)
# ---------------------------------------------------------------------------


class TestQualityScreen:
    def test_basic_ranking(self):
        from wraquant.fundamental.valuation import quality_screen

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

    def test_no_matching_metrics(self):
        from wraquant.fundamental.valuation import quality_screen

        stocks = pd.DataFrame({"price": [100, 200, 300]}, index=["A", "B", "C"])
        result = quality_screen(stocks, metrics=["roe"])
        assert (result["quality_score"] == 0.5).all()
