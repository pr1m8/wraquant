"""Tests for wraquant.fundamental.ratios — FMP-backed ratio functions.

All FMP API calls are mocked.  No network requests are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_INCOME_STMT = [
    {
        "date": "2024-12-31",
        "revenue": 400_000,
        "grossProfit": 160_000,
        "operatingIncome": 100_000,
        "netIncome": 60_000,
        "incomeBeforeTax": 80_000,
        "incomeTaxExpense": 20_000,
        "costOfRevenue": 240_000,
        "interestExpense": 5_000,
        "depreciationAndAmortization": 15_000,
        "eps": 3.0,
    },
    {
        "date": "2023-12-31",
        "revenue": 350_000,
        "grossProfit": 140_000,
        "operatingIncome": 84_000,
        "netIncome": 49_000,
        "incomeBeforeTax": 70_000,
        "incomeTaxExpense": 21_000,
        "costOfRevenue": 210_000,
        "interestExpense": 6_000,
        "depreciationAndAmortization": 14_000,
        "eps": 2.45,
    },
    {
        "date": "2022-12-31",
        "revenue": 300_000,
        "grossProfit": 120_000,
        "operatingIncome": 72_000,
        "netIncome": 42_000,
        "incomeBeforeTax": 60_000,
        "incomeTaxExpense": 18_000,
        "costOfRevenue": 180_000,
        "interestExpense": 7_000,
        "depreciationAndAmortization": 12_000,
        "eps": 2.10,
    },
    {
        "date": "2021-12-31",
        "revenue": 270_000,
        "grossProfit": 108_000,
        "operatingIncome": 64_800,
        "netIncome": 37_800,
        "incomeBeforeTax": 54_000,
        "incomeTaxExpense": 16_200,
        "costOfRevenue": 162_000,
        "interestExpense": 8_000,
        "depreciationAndAmortization": 11_000,
        "eps": 1.89,
    },
]

_BALANCE_SHEET = [
    {
        "date": "2024-12-31",
        "totalAssets": 500_000,
        "totalStockholdersEquity": 250_000,
        "totalDebt": 100_000,
        "cashAndCashEquivalents": 50_000,
        "totalCurrentAssets": 200_000,
        "totalCurrentLiabilities": 120_000,
        "inventory": 30_000,
        "netReceivables": 40_000,
        "accountPayables": 25_000,
        "goodwill": 20_000,
        "intangibleAssets": 15_000,
        "commonStock": 20_000,
    },
]

_RATIOS_DATA = [
    {
        "returnOnEquity": 0.24,
        "returnOnAssets": 0.12,
    },
]

_RATIOS_TTM = {
    "peRatioTTM": 25.0,
    "priceToBookRatioTTM": 4.0,
    "priceToSalesRatioTTM": 6.0,
    "pegRatioTTM": 1.5,
    "dividendYieldTTM": 0.015,
    "priceToFreeCashFlowsRatioTTM": 20.0,
}

_KEY_METRICS = [{"evToOperatingCashFlow": 18.0}]

_EV_DATA = [{"evToEBITDA": 15.0}]

_GROWTH_DATA = [
    {
        "revenueGrowth": 0.143,
        "epsgrowth": 0.224,
        "dividendsperShareGrowth": 0.05,
        "ebitdagrowth": 0.19,
        "freeCashFlowGrowth": 0.25,
    },
    {
        "revenueGrowth": 0.167,
        "epsgrowth": 0.167,
        "dividendsperShareGrowth": 0.04,
        "ebitdagrowth": 0.16,
        "freeCashFlowGrowth": 0.18,
    },
]


@pytest.fixture()
def mock_fmp():
    """Create a fully-mocked FMP client."""
    client = MagicMock()
    client.income_statement.return_value = _INCOME_STMT
    client.balance_sheet.return_value = _BALANCE_SHEET
    client.ratios.return_value = _RATIOS_DATA
    client.ratios_ttm.return_value = _RATIOS_TTM
    client.key_metrics.return_value = _KEY_METRICS
    client.enterprise_value.return_value = _EV_DATA
    client.financial_growth.return_value = _GROWTH_DATA
    return client


# Patch check_extra so @requires_extra("market-data") never blocks us.
_PATCH_CHECK = patch(
    "wraquant._lazy.check_extra",
    return_value=True,
)


# ---------------------------------------------------------------------------
# Profitability ratios
# ---------------------------------------------------------------------------


class TestProfitabilityRatios:
    @_PATCH_CHECK
    def test_keys(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import profitability_ratios

        result = profitability_ratios("AAPL", fmp_client=mock_fmp)
        expected_keys = {
            "roe",
            "roa",
            "roic",
            "gross_margin",
            "operating_margin",
            "net_margin",
            "period",
        }
        assert expected_keys == set(result.keys())

    @_PATCH_CHECK
    def test_margin_values(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import profitability_ratios

        result = profitability_ratios("AAPL", fmp_client=mock_fmp)
        assert abs(result["gross_margin"] - 160_000 / 400_000) < 1e-9
        assert abs(result["operating_margin"] - 100_000 / 400_000) < 1e-9
        assert abs(result["net_margin"] - 60_000 / 400_000) < 1e-9

    @_PATCH_CHECK
    def test_roe_from_fmp(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import profitability_ratios

        result = profitability_ratios("AAPL", fmp_client=mock_fmp)
        assert result["roe"] == 0.24  # from mock ratios data

    @_PATCH_CHECK
    def test_roic_computed(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import profitability_ratios

        result = profitability_ratios("AAPL", fmp_client=mock_fmp)
        # ROIC = EBIT*(1-tax_rate) / (debt + equity - cash)
        tax_rate = 20_000 / 80_000  # 0.25
        nopat = 100_000 * (1.0 - tax_rate)
        invested = 100_000 + 250_000 - 50_000
        expected_roic = nopat / invested
        assert abs(result["roic"] - expected_roic) < 1e-9

    @_PATCH_CHECK
    def test_period_passthrough(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import profitability_ratios

        result = profitability_ratios("AAPL", period="quarter", fmp_client=mock_fmp)
        assert result["period"] == "quarter"

    @_PATCH_CHECK
    def test_zero_revenue(self, _ce):
        """Edge case: zero revenue should not crash."""
        from wraquant.fundamental.ratios import profitability_ratios

        client = MagicMock()
        client.income_statement.return_value = [
            {
                "revenue": 0,
                "grossProfit": 0,
                "operatingIncome": 0,
                "netIncome": 0,
                "incomeBeforeTax": 0,
                "incomeTaxExpense": 0,
            }
        ]
        client.balance_sheet.return_value = [
            {
                "totalAssets": 100,
                "totalStockholdersEquity": 50,
                "totalDebt": 30,
                "cashAndCashEquivalents": 10,
            }
        ]
        client.ratios.return_value = [{}]
        result = profitability_ratios("ZERO", fmp_client=client)
        assert result["gross_margin"] == 0.0
        assert result["operating_margin"] == 0.0
        assert result["net_margin"] == 0.0


# ---------------------------------------------------------------------------
# Liquidity ratios
# ---------------------------------------------------------------------------


class TestLiquidityRatios:
    @_PATCH_CHECK
    def test_keys(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import liquidity_ratios

        result = liquidity_ratios("AAPL", fmp_client=mock_fmp)
        expected_keys = {
            "current_ratio",
            "quick_ratio",
            "cash_ratio",
            "working_capital",
            "period",
        }
        assert expected_keys == set(result.keys())

    @_PATCH_CHECK
    def test_values(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import liquidity_ratios

        result = liquidity_ratios("AAPL", fmp_client=mock_fmp)
        assert abs(result["current_ratio"] - 200_000 / 120_000) < 1e-9
        assert abs(result["quick_ratio"] - (200_000 - 30_000) / 120_000) < 1e-9
        assert abs(result["cash_ratio"] - 50_000 / 120_000) < 1e-9
        assert abs(result["working_capital"] - (200_000 - 120_000)) < 1e-9

    @_PATCH_CHECK
    def test_zero_liabilities(self, _ce):
        from wraquant.fundamental.ratios import liquidity_ratios

        client = MagicMock()
        client.balance_sheet.return_value = [
            {
                "totalCurrentAssets": 100,
                "totalCurrentLiabilities": 0,
                "inventory": 10,
                "cashAndCashEquivalents": 50,
            }
        ]
        result = liquidity_ratios("ZERO", fmp_client=client)
        assert result["current_ratio"] == 0.0
        assert result["quick_ratio"] == 0.0


# ---------------------------------------------------------------------------
# Leverage ratios
# ---------------------------------------------------------------------------


class TestLeverageRatios:
    @_PATCH_CHECK
    def test_keys(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import leverage_ratios

        result = leverage_ratios("AAPL", fmp_client=mock_fmp)
        expected_keys = {
            "debt_to_equity",
            "debt_ratio",
            "interest_coverage",
            "equity_multiplier",
            "debt_to_ebitda",
            "period",
        }
        assert expected_keys == set(result.keys())

    @_PATCH_CHECK
    def test_de_ratio(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import leverage_ratios

        result = leverage_ratios("AAPL", fmp_client=mock_fmp)
        assert abs(result["debt_to_equity"] - 100_000 / 250_000) < 1e-9

    @_PATCH_CHECK
    def test_interest_coverage(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import leverage_ratios

        result = leverage_ratios("AAPL", fmp_client=mock_fmp)
        # EBIT / |interest_expense| = 100000 / 5000 = 20
        assert abs(result["interest_coverage"] - 20.0) < 1e-9

    @_PATCH_CHECK
    def test_debt_to_ebitda(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import leverage_ratios

        result = leverage_ratios("AAPL", fmp_client=mock_fmp)
        ebitda = 100_000 + 15_000  # EBIT + D&A
        assert abs(result["debt_to_ebitda"] - 100_000 / ebitda) < 1e-9


# ---------------------------------------------------------------------------
# Efficiency ratios
# ---------------------------------------------------------------------------


class TestEfficiencyRatios:
    @_PATCH_CHECK
    def test_keys(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import efficiency_ratios

        result = efficiency_ratios("AAPL", fmp_client=mock_fmp)
        expected_keys = {
            "asset_turnover",
            "inventory_turnover",
            "receivable_turnover",
            "payable_turnover",
            "days_sales_outstanding",
            "days_inventory_outstanding",
            "days_payable_outstanding",
            "cash_conversion_cycle",
            "period",
        }
        assert expected_keys == set(result.keys())

    @_PATCH_CHECK
    def test_asset_turnover(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import efficiency_ratios

        result = efficiency_ratios("AAPL", fmp_client=mock_fmp)
        assert abs(result["asset_turnover"] - 400_000 / 500_000) < 1e-9

    @_PATCH_CHECK
    def test_cash_conversion_cycle(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import efficiency_ratios

        result = efficiency_ratios("AAPL", fmp_client=mock_fmp)
        # DSO + DIO - DPO
        inv_turnover = 240_000 / 30_000
        rec_turnover = 400_000 / 40_000
        pay_turnover = 240_000 / 25_000
        dso = 365.0 / rec_turnover
        dio = 365.0 / inv_turnover
        dpo = 365.0 / pay_turnover
        assert abs(result["cash_conversion_cycle"] - (dso + dio - dpo)) < 1e-6


# ---------------------------------------------------------------------------
# Valuation ratios
# ---------------------------------------------------------------------------


class TestValuationRatios:
    @_PATCH_CHECK
    def test_keys(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import valuation_ratios

        result = valuation_ratios("AAPL", fmp_client=mock_fmp)
        expected_keys = {
            "pe_ratio",
            "pb_ratio",
            "ps_ratio",
            "ev_to_ebitda",
            "peg_ratio",
            "dividend_yield",
            "earnings_yield",
            "price_to_fcf",
        }
        assert expected_keys == set(result.keys())

    @_PATCH_CHECK
    def test_pe(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import valuation_ratios

        result = valuation_ratios("AAPL", fmp_client=mock_fmp)
        assert result["pe_ratio"] == 25.0

    @_PATCH_CHECK
    def test_earnings_yield_inverse_pe(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import valuation_ratios

        result = valuation_ratios("AAPL", fmp_client=mock_fmp)
        assert abs(result["earnings_yield"] - 1.0 / 25.0) < 1e-9

    @_PATCH_CHECK
    def test_ev_ebitda_prefers_direct(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import valuation_ratios

        result = valuation_ratios("AAPL", fmp_client=mock_fmp)
        # EV data has evToEBITDA=15, key_metrics has evToOperatingCashFlow=18
        assert result["ev_to_ebitda"] == 15.0


# ---------------------------------------------------------------------------
# Growth ratios
# ---------------------------------------------------------------------------


class TestGrowthRatios:
    @_PATCH_CHECK
    def test_keys(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import growth_ratios

        result = growth_ratios("AAPL", fmp_client=mock_fmp)
        expected_keys = {
            "revenue_growth",
            "eps_growth",
            "dividend_growth",
            "ebitda_growth",
            "fcf_growth",
            "revenue_growth_3y",
            "revenue_growth_5y",
            "revenue_growth_history",
            "period",
        }
        assert expected_keys == set(result.keys())

    @_PATCH_CHECK
    def test_latest_growth(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import growth_ratios

        result = growth_ratios("AAPL", fmp_client=mock_fmp)
        assert abs(result["revenue_growth"] - 0.143) < 1e-9
        assert abs(result["eps_growth"] - 0.224) < 1e-9

    @_PATCH_CHECK
    def test_revenue_growth_history_is_list(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import growth_ratios

        result = growth_ratios("AAPL", fmp_client=mock_fmp)
        assert isinstance(result["revenue_growth_history"], list)

    @_PATCH_CHECK
    def test_cagr_3y(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import growth_ratios

        result = growth_ratios("AAPL", fmp_client=mock_fmp)
        # 4 revenue values => values[0]=400k, values[3]=270k
        expected_cagr = (400_000 / 270_000) ** (1.0 / 3.0) - 1.0
        assert abs(result["revenue_growth_3y"] - expected_cagr) < 1e-6


# ---------------------------------------------------------------------------
# DuPont decomposition
# ---------------------------------------------------------------------------


class TestDuPontDecomposition:
    @_PATCH_CHECK
    def test_keys(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import dupont_decomposition

        result = dupont_decomposition("AAPL", fmp_client=mock_fmp)
        expected_keys = {
            "net_margin",
            "asset_turnover",
            "equity_multiplier",
            "roe_3way",
            "tax_burden",
            "interest_burden",
            "operating_margin",
            "roe_5way",
            "period",
        }
        assert expected_keys == set(result.keys())

    @_PATCH_CHECK
    def test_3way_identity(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import dupont_decomposition

        result = dupont_decomposition("AAPL", fmp_client=mock_fmp)
        product = (
            result["net_margin"]
            * result["asset_turnover"]
            * result["equity_multiplier"]
        )
        assert abs(result["roe_3way"] - product) < 1e-9

    @_PATCH_CHECK
    def test_5way_identity(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import dupont_decomposition

        result = dupont_decomposition("AAPL", fmp_client=mock_fmp)
        product = (
            result["tax_burden"]
            * result["interest_burden"]
            * result["operating_margin"]
            * result["asset_turnover"]
            * result["equity_multiplier"]
        )
        assert abs(result["roe_5way"] - product) < 1e-9

    @_PATCH_CHECK
    def test_3way_matches_manual(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import dupont_decomposition

        result = dupont_decomposition("AAPL", fmp_client=mock_fmp)
        # NI/Rev * Rev/Assets * Assets/Equity
        expected = (60_000 / 400_000) * (400_000 / 500_000) * (500_000 / 250_000)
        assert abs(result["roe_3way"] - expected) < 1e-9


# ---------------------------------------------------------------------------
# Comprehensive ratios
# ---------------------------------------------------------------------------


class TestComprehensiveRatios:
    @_PATCH_CHECK
    def test_top_level_keys(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import comprehensive_ratios

        result = comprehensive_ratios("AAPL", fmp_client=mock_fmp)
        expected_keys = {
            "symbol",
            "period",
            "profitability",
            "liquidity",
            "leverage",
            "efficiency",
            "valuation",
            "growth",
            "dupont",
        }
        assert expected_keys == set(result.keys())

    @_PATCH_CHECK
    def test_symbol_passthrough(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import comprehensive_ratios

        result = comprehensive_ratios("MSFT", fmp_client=mock_fmp)
        assert result["symbol"] == "MSFT"

    @_PATCH_CHECK
    def test_nested_dicts(self, _ce, mock_fmp):
        from wraquant.fundamental.ratios import comprehensive_ratios

        result = comprehensive_ratios("AAPL", fmp_client=mock_fmp)
        assert isinstance(result["profitability"], dict)
        assert "roe" in result["profitability"]
        assert isinstance(result["leverage"], dict)
        assert "debt_to_equity" in result["leverage"]


# ---------------------------------------------------------------------------
# Edge case: FMP returns a single dict instead of a list
# ---------------------------------------------------------------------------


class TestDictReturn:
    @_PATCH_CHECK
    def test_liquidity_with_dict_return(self, _ce):
        """FMP sometimes returns a dict instead of a list."""
        from wraquant.fundamental.ratios import liquidity_ratios

        client = MagicMock()
        # Return a plain dict rather than a list
        client.balance_sheet.return_value = {
            "totalCurrentAssets": 200,
            "totalCurrentLiabilities": 100,
            "inventory": 20,
            "cashAndCashEquivalents": 80,
        }
        result = liquidity_ratios("X", fmp_client=client)
        assert abs(result["current_ratio"] - 2.0) < 1e-9

    @_PATCH_CHECK
    def test_empty_data(self, _ce):
        """Empty list response should not crash."""
        from wraquant.fundamental.ratios import liquidity_ratios

        client = MagicMock()
        client.balance_sheet.return_value = []
        result = liquidity_ratios("EMPTY", fmp_client=client)
        assert result["current_ratio"] == 0.0
        assert result["working_capital"] == 0.0
