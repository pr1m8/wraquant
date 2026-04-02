"""Tests for wraquant.fundamental.financials — FMP-backed financial analysis.

All FMP API calls are mocked.  No network requests are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Common mock data
# ---------------------------------------------------------------------------

_INCOME_5Y = [
    {
        "date": "2024-12-31",
        "revenue": 400_000,
        "grossProfit": 160_000,
        "operatingIncome": 100_000,
        "netIncome": 60_000,
        "depreciationAndAmortization": 15_000,
        "eps": 3.0,
        "costOfRevenue": 240_000,
        "incomeTaxExpense": 20_000,
        "interestExpense": 5_000,
        "researchAndDevelopmentExpenses": 30_000,
        "sellingGeneralAndAdministrativeExpenses": 25_000,
        "incomeBeforeTax": 80_000,
    },
    {
        "date": "2023-12-31",
        "revenue": 350_000,
        "grossProfit": 140_000,
        "operatingIncome": 84_000,
        "netIncome": 49_000,
        "depreciationAndAmortization": 14_000,
        "eps": 2.45,
        "costOfRevenue": 210_000,
        "incomeTaxExpense": 21_000,
        "interestExpense": 6_000,
        "researchAndDevelopmentExpenses": 28_000,
        "sellingGeneralAndAdministrativeExpenses": 22_000,
        "incomeBeforeTax": 70_000,
    },
    {
        "date": "2022-12-31",
        "revenue": 300_000,
        "grossProfit": 120_000,
        "operatingIncome": 72_000,
        "netIncome": 42_000,
        "depreciationAndAmortization": 12_000,
        "eps": 2.10,
        "costOfRevenue": 180_000,
        "incomeTaxExpense": 18_000,
        "interestExpense": 7_000,
        "researchAndDevelopmentExpenses": 25_000,
        "sellingGeneralAndAdministrativeExpenses": 20_000,
        "incomeBeforeTax": 60_000,
    },
    {
        "date": "2021-12-31",
        "revenue": 270_000,
        "grossProfit": 108_000,
        "operatingIncome": 64_800,
        "netIncome": 37_800,
        "depreciationAndAmortization": 11_000,
        "eps": 1.89,
        "costOfRevenue": 162_000,
        "incomeTaxExpense": 16_200,
        "interestExpense": 8_000,
        "researchAndDevelopmentExpenses": 22_000,
        "sellingGeneralAndAdministrativeExpenses": 18_000,
        "incomeBeforeTax": 54_000,
    },
    {
        "date": "2020-12-31",
        "revenue": 250_000,
        "grossProfit": 100_000,
        "operatingIncome": 60_000,
        "netIncome": 35_000,
        "depreciationAndAmortization": 10_000,
        "eps": 1.75,
        "costOfRevenue": 150_000,
        "incomeTaxExpense": 15_000,
        "interestExpense": 9_000,
        "researchAndDevelopmentExpenses": 20_000,
        "sellingGeneralAndAdministrativeExpenses": 16_000,
        "incomeBeforeTax": 50_000,
    },
]

_BALANCE_5Y = [
    {
        "date": "2024-12-31",
        "totalAssets": 500_000,
        "totalStockholdersEquity": 250_000,
        "totalDebt": 100_000,
        "cashAndCashEquivalents": 50_000,
        "totalCurrentAssets": 200_000,
        "totalCurrentLiabilities": 120_000,
        "inventory": 30_000,
        "goodwill": 20_000,
        "intangibleAssets": 15_000,
        "commonStock": 20_000,
        "netReceivables": 40_000,
        "propertyPlantEquipmentNet": 80_000,
        "retainedEarnings": 180_000,
        "longTermDebt": 75_000,
    },
    {
        "date": "2023-12-31",
        "totalAssets": 460_000,
        "totalStockholdersEquity": 230_000,
        "totalDebt": 95_000,
        "cashAndCashEquivalents": 45_000,
        "totalCurrentAssets": 185_000,
        "totalCurrentLiabilities": 115_000,
        "inventory": 28_000,
        "goodwill": 20_000,
        "intangibleAssets": 14_000,
        "commonStock": 20_000,
        "netReceivables": 38_000,
        "propertyPlantEquipmentNet": 75_000,
        "retainedEarnings": 160_000,
        "longTermDebt": 72_000,
    },
    {
        "date": "2022-12-31",
        "totalAssets": 420_000,
        "totalStockholdersEquity": 210_000,
        "totalDebt": 90_000,
        "cashAndCashEquivalents": 40_000,
        "totalCurrentAssets": 170_000,
        "totalCurrentLiabilities": 110_000,
        "inventory": 25_000,
        "goodwill": 18_000,
        "intangibleAssets": 12_000,
        "commonStock": 20_000,
        "netReceivables": 35_000,
        "propertyPlantEquipmentNet": 70_000,
        "retainedEarnings": 140_000,
        "longTermDebt": 68_000,
    },
    {
        "date": "2021-12-31",
        "totalAssets": 380_000,
        "totalStockholdersEquity": 190_000,
        "totalDebt": 85_000,
        "cashAndCashEquivalents": 35_000,
        "totalCurrentAssets": 155_000,
        "totalCurrentLiabilities": 100_000,
        "inventory": 22_000,
        "goodwill": 16_000,
        "intangibleAssets": 10_000,
        "commonStock": 20_000,
        "netReceivables": 32_000,
        "propertyPlantEquipmentNet": 65_000,
        "retainedEarnings": 120_000,
        "longTermDebt": 65_000,
    },
    {
        "date": "2020-12-31",
        "totalAssets": 350_000,
        "totalStockholdersEquity": 175_000,
        "totalDebt": 80_000,
        "cashAndCashEquivalents": 30_000,
        "totalCurrentAssets": 140_000,
        "totalCurrentLiabilities": 95_000,
        "inventory": 20_000,
        "goodwill": 15_000,
        "intangibleAssets": 9_000,
        "commonStock": 20_000,
        "netReceivables": 30_000,
        "propertyPlantEquipmentNet": 60_000,
        "retainedEarnings": 100_000,
        "longTermDebt": 60_000,
    },
]

_CASH_FLOW_5Y = [
    {
        "date": "2024-12-31",
        "operatingCashFlow": 110_000,
        "capitalExpenditure": -30_000,
        "freeCashFlow": 80_000,
        "dividendsPaid": -10_000,
        "commonStockRepurchased": -20_000,
    },
    {
        "date": "2023-12-31",
        "operatingCashFlow": 95_000,
        "capitalExpenditure": -25_000,
        "freeCashFlow": 70_000,
        "dividendsPaid": -9_000,
        "commonStockRepurchased": -18_000,
    },
    {
        "date": "2022-12-31",
        "operatingCashFlow": 85_000,
        "capitalExpenditure": -25_000,
        "freeCashFlow": 60_000,
        "dividendsPaid": -8_000,
        "commonStockRepurchased": -15_000,
    },
    {
        "date": "2021-12-31",
        "operatingCashFlow": 75_000,
        "capitalExpenditure": -25_000,
        "freeCashFlow": 50_000,
        "dividendsPaid": -7_000,
        "commonStockRepurchased": -12_000,
    },
    {
        "date": "2020-12-31",
        "operatingCashFlow": 65_000,
        "capitalExpenditure": -20_000,
        "freeCashFlow": 45_000,
        "dividendsPaid": -6_000,
        "commonStockRepurchased": -10_000,
    },
]

_PROFILE = [{"companyName": "Test Corp", "price": 150.0, "mktCap": 3_000_000}]

_SCORE_DATA = {"altmanZScore": 3.5, "piotroskiScore": 7}


@pytest.fixture()
def mock_fmp():
    client = MagicMock()
    client.income_statement.return_value = _INCOME_5Y
    client.balance_sheet.return_value = _BALANCE_5Y
    client.cash_flow.return_value = _CASH_FLOW_5Y
    client.company_profile.return_value = _PROFILE
    client.ratios.return_value = [{}]
    client.score.return_value = _SCORE_DATA
    return client


_PATCH_CHECK = patch("wraquant._lazy.check_extra", return_value=True)


# ---------------------------------------------------------------------------
# Income Analysis
# ---------------------------------------------------------------------------


class TestIncomeAnalysis:
    @_PATCH_CHECK
    def test_keys(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import income_analysis

        result = income_analysis("AAPL", fmp_client=mock_fmp)
        expected_keys = {
            "revenue",
            "revenue_growth",
            "gross_margin",
            "operating_margin",
            "net_margin",
            "ebitda_margin",
            "margin_trend",
            "revenue_cagr_3y",
            "revenue_cagr_5y",
            "eps",
            "dates",
            "periods_analysed",
        }
        assert expected_keys == set(result.keys())

    @_PATCH_CHECK
    def test_revenue_values(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import income_analysis

        result = income_analysis("AAPL", fmp_client=mock_fmp)
        assert result["revenue"][0] == 400_000
        assert result["periods_analysed"] == 5

    @_PATCH_CHECK
    def test_margin_trend(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import income_analysis

        result = income_analysis("AAPL", fmp_client=mock_fmp)
        assert result["margin_trend"] in {"expanding", "contracting", "stable"}

    @_PATCH_CHECK
    def test_revenue_growth_length(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import income_analysis

        result = income_analysis("AAPL", fmp_client=mock_fmp)
        # 5 periods => 4 growth rates
        assert len(result["revenue_growth"]) == 4

    @_PATCH_CHECK
    def test_cagr_3y(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import income_analysis

        result = income_analysis("AAPL", fmp_client=mock_fmp)
        expected = (400_000 / 270_000) ** (1.0 / 3.0) - 1.0
        assert abs(result["revenue_cagr_3y"] - expected) < 1e-6

    @_PATCH_CHECK
    def test_empty_data(self, _ce):
        from wraquant.fundamental.financials import income_analysis

        client = MagicMock()
        client.income_statement.return_value = []
        result = income_analysis("EMPTY", fmp_client=client)
        assert result["periods_analysed"] == 0
        assert result["margin_trend"] == "unknown"
        assert result["revenue"] == []


# ---------------------------------------------------------------------------
# Balance Sheet Analysis
# ---------------------------------------------------------------------------


class TestBalanceSheetAnalysis:
    @_PATCH_CHECK
    def test_keys(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import balance_sheet_analysis

        result = balance_sheet_analysis("AAPL", fmp_client=mock_fmp)
        expected_keys = {
            "total_assets",
            "total_equity",
            "total_debt",
            "cash",
            "net_debt",
            "debt_to_equity",
            "debt_to_assets",
            "current_ratio",
            "equity_pct",
            "intangible_pct",
            "leverage_trend",
            "book_value_per_share",
            "tangible_bvps",
            "dates",
        }
        assert expected_keys == set(result.keys())

    @_PATCH_CHECK
    def test_list_lengths_match(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import balance_sheet_analysis

        result = balance_sheet_analysis("AAPL", fmp_client=mock_fmp)
        n = len(result["dates"])
        assert len(result["total_assets"]) == n
        assert len(result["debt_to_equity"]) == n
        assert len(result["current_ratio"]) == n

    @_PATCH_CHECK
    def test_net_debt_formula(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import balance_sheet_analysis

        result = balance_sheet_analysis("AAPL", fmp_client=mock_fmp)
        # net_debt = debt - cash for each period
        assert result["net_debt"][0] == 100_000 - 50_000

    @_PATCH_CHECK
    def test_leverage_trend(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import balance_sheet_analysis

        result = balance_sheet_analysis("AAPL", fmp_client=mock_fmp)
        assert result["leverage_trend"] in {
            "increasing",
            "decreasing",
            "stable",
            "unknown",
        }

    @_PATCH_CHECK
    def test_empty_data(self, _ce):
        from wraquant.fundamental.financials import balance_sheet_analysis

        client = MagicMock()
        client.balance_sheet.return_value = []
        result = balance_sheet_analysis("EMPTY", fmp_client=client)
        assert result["total_assets"] == []
        assert result["leverage_trend"] == "unknown"


# ---------------------------------------------------------------------------
# Cash Flow Analysis
# ---------------------------------------------------------------------------


class TestCashFlowAnalysis:
    @_PATCH_CHECK
    def test_keys(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import cash_flow_analysis

        result = cash_flow_analysis("AAPL", fmp_client=mock_fmp)
        expected_keys = {
            "operating_cash_flow",
            "capital_expenditures",
            "free_cash_flow",
            "fcf_margin",
            "fcf_growth",
            "fcf_yield",
            "cash_conversion",
            "capex_to_revenue",
            "capex_to_ocf",
            "dividends_paid",
            "buybacks",
            "total_shareholder_return",
            "fcf_payout_ratio",
            "dates",
        }
        assert expected_keys == set(result.keys())

    @_PATCH_CHECK
    def test_fcf_values(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import cash_flow_analysis

        result = cash_flow_analysis("AAPL", fmp_client=mock_fmp)
        assert result["free_cash_flow"][0] == 80_000

    @_PATCH_CHECK
    def test_fcf_growth_length(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import cash_flow_analysis

        result = cash_flow_analysis("AAPL", fmp_client=mock_fmp)
        # 5 periods => 4 growth rates
        assert len(result["fcf_growth"]) == 4

    @_PATCH_CHECK
    def test_fcf_margin(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import cash_flow_analysis

        result = cash_flow_analysis("AAPL", fmp_client=mock_fmp)
        # FCF / Revenue = 80000 / 400000
        assert abs(result["fcf_margin"][0] - 80_000 / 400_000) < 1e-9

    @_PATCH_CHECK
    def test_cash_conversion(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import cash_flow_analysis

        result = cash_flow_analysis("AAPL", fmp_client=mock_fmp)
        # OCF / NI = 110000 / 60000
        assert abs(result["cash_conversion"][0] - 110_000 / 60_000) < 1e-6

    @_PATCH_CHECK
    def test_empty_data(self, _ce):
        from wraquant.fundamental.financials import cash_flow_analysis

        client = MagicMock()
        client.cash_flow.return_value = []
        client.income_statement.return_value = []
        client.company_profile.return_value = [{"mktCap": 0}]
        result = cash_flow_analysis("EMPTY", fmp_client=client)
        assert result["free_cash_flow"] == []
        assert result["fcf_yield"] == 0.0


# ---------------------------------------------------------------------------
# Financial Health Score
# ---------------------------------------------------------------------------


class TestFinancialHealthScore:
    @_PATCH_CHECK
    def test_keys(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import financial_health_score

        result = financial_health_score("AAPL", fmp_client=mock_fmp)
        expected_keys = {
            "total_score",
            "category",
            "grade",
            "profitability_score",
            "liquidity_score",
            "leverage_score",
            "efficiency_score",
            "cash_flow_score",
            "strengths",
            "weaknesses",
            "piotroski_f_score",
            "symbol",
        }
        assert expected_keys == set(result.keys())

    @_PATCH_CHECK
    def test_score_range(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import financial_health_score

        result = financial_health_score("AAPL", fmp_client=mock_fmp)
        assert 0.0 <= result["total_score"] <= 100.0

    @_PATCH_CHECK
    def test_category_values(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import financial_health_score

        result = financial_health_score("AAPL", fmp_client=mock_fmp)
        assert result["category"] in {"excellent", "good", "fair", "weak", "critical"}

    @_PATCH_CHECK
    def test_sub_scores_bounded(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import financial_health_score

        result = financial_health_score("AAPL", fmp_client=mock_fmp)
        assert 0.0 <= result["profitability_score"] <= 30.0
        assert 0.0 <= result["liquidity_score"] <= 15.0
        assert 0.0 <= result["leverage_score"] <= 20.0
        assert 0.0 <= result["efficiency_score"] <= 15.0
        assert 0.0 <= result["cash_flow_score"] <= 20.0

    @_PATCH_CHECK
    def test_piotroski_in_range(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import financial_health_score

        result = financial_health_score("AAPL", fmp_client=mock_fmp)
        assert 0 <= result["piotroski_f_score"] <= 9

    @_PATCH_CHECK
    def test_strengths_weaknesses_are_lists(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import financial_health_score

        result = financial_health_score("AAPL", fmp_client=mock_fmp)
        assert isinstance(result["strengths"], list)
        assert isinstance(result["weaknesses"], list)


# ---------------------------------------------------------------------------
# Earnings Quality
# ---------------------------------------------------------------------------


class TestEarningsQuality:
    @_PATCH_CHECK
    def test_keys(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import earnings_quality

        result = earnings_quality("AAPL", fmp_client=mock_fmp)
        expected_keys = {
            "accruals_ratio",
            "cash_conversion_ratio",
            "earnings_persistence",
            "fcf_to_net_income",
            "quality_grade",
            "accruals_trend",
            "red_flags",
            "periods_analysed",
            "symbol",
        }
        assert expected_keys == set(result.keys())

    @_PATCH_CHECK
    def test_quality_grade_values(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import earnings_quality

        result = earnings_quality("AAPL", fmp_client=mock_fmp)
        assert result["quality_grade"] in {"A", "B", "C", "D", "F", "N/A"}

    @_PATCH_CHECK
    def test_cash_conversion(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import earnings_quality

        result = earnings_quality("AAPL", fmp_client=mock_fmp)
        # OCF[0]=110_000, NI[0]=60_000 => 1.833
        expected = 110_000 / 60_000
        assert abs(result["cash_conversion_ratio"] - expected) < 1e-6

    @_PATCH_CHECK
    def test_accruals_trend_length(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import earnings_quality

        result = earnings_quality("AAPL", fmp_client=mock_fmp)
        assert len(result["accruals_trend"]) > 0

    @_PATCH_CHECK
    def test_red_flags_is_list(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import earnings_quality

        result = earnings_quality("AAPL", fmp_client=mock_fmp)
        assert isinstance(result["red_flags"], list)

    @_PATCH_CHECK
    def test_empty_data(self, _ce):
        from wraquant.fundamental.financials import earnings_quality

        client = MagicMock()
        client.income_statement.return_value = []
        client.balance_sheet.return_value = []
        client.cash_flow.return_value = []
        result = earnings_quality("EMPTY", fmp_client=client)
        assert result["quality_grade"] == "N/A"
        assert result["periods_analysed"] == 0


# ---------------------------------------------------------------------------
# Common Size Analysis
# ---------------------------------------------------------------------------


class TestCommonSizeAnalysis:
    @_PATCH_CHECK
    def test_columns(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import common_size_analysis

        result = common_size_analysis("AAPL", fmp_client=mock_fmp)
        assert isinstance(result, pd.DataFrame)
        for col in [
            "date",
            "cost_of_revenue_pct",
            "gross_profit_pct",
            "equity_pct",
        ]:
            assert col in result.columns

    @_PATCH_CHECK
    def test_income_pcts_sum(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import common_size_analysis

        result = common_size_analysis("AAPL", fmp_client=mock_fmp)
        latest = result.iloc[0]
        # COGS% + Gross% should equal ~1.0
        assert (
            abs(latest["cost_of_revenue_pct"] + latest["gross_profit_pct"] - 1.0) < 1e-9
        )

    @_PATCH_CHECK
    def test_gross_profit_pct(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import common_size_analysis

        result = common_size_analysis("AAPL", fmp_client=mock_fmp)
        latest = result.iloc[0]
        assert abs(latest["gross_profit_pct"] - 160_000 / 400_000) < 1e-9

    @_PATCH_CHECK
    def test_balance_sheet_equity_pct(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import common_size_analysis

        result = common_size_analysis("AAPL", fmp_client=mock_fmp)
        latest = result.iloc[0]
        assert abs(latest["equity_pct"] - 250_000 / 500_000) < 1e-9

    @_PATCH_CHECK
    def test_periods_analysed(self, _ce, mock_fmp):
        from wraquant.fundamental.financials import common_size_analysis

        result = common_size_analysis("AAPL", fmp_client=mock_fmp)
        assert len(result) == 5
