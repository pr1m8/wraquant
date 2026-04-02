"""Tests for wraquant.fundamental.screening — FMP-backed stock screeners.

All FMP API calls are mocked.  No network requests are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Common mock data
# ---------------------------------------------------------------------------

_SCREENER_DF = pd.DataFrame(
    [
        {
            "symbol": "AAPL",
            "companyName": "Apple",
            "marketCap": 3e12,
            "sector": "Technology",
            "beta": 1.2,
        },
        {
            "symbol": "JNJ",
            "companyName": "Johnson & Johnson",
            "marketCap": 4e11,
            "sector": "Healthcare",
            "beta": 0.6,
        },
        {
            "symbol": "KO",
            "companyName": "Coca-Cola",
            "marketCap": 2.5e11,
            "sector": "Consumer Staples",
            "beta": 0.5,
        },
        {
            "symbol": "XOM",
            "companyName": "Exxon Mobil",
            "marketCap": 5e11,
            "sector": "Energy",
            "beta": 0.9,
        },
        {
            "symbol": "MSFT",
            "companyName": "Microsoft",
            "marketCap": 3.2e12,
            "sector": "Technology",
            "beta": 0.9,
        },
    ]
)


@pytest.fixture()
def mock_fmp():
    client = MagicMock()
    client.stock_screener.return_value = _SCREENER_DF.copy()
    client.score.return_value = {"piotroskiScore": 8, "altmanZScore": 3.5}
    client.ratios_ttm.return_value = {
        "peRatioTTM": 20.0,
        "returnOnCapitalEmployedTTM": 0.25,
        "roicTTM": 0.22,
    }
    return client


_PATCH_CHECK = patch("wraquant._lazy.check_extra", return_value=True)


# ---------------------------------------------------------------------------
# Value Screen
# ---------------------------------------------------------------------------


class TestValueScreen:
    @_PATCH_CHECK
    def test_returns_dataframe(self, _ce, mock_fmp):
        from wraquant.fundamental.screening import value_screen

        result = value_screen(fmp_client=mock_fmp)
        assert isinstance(result, pd.DataFrame)

    @_PATCH_CHECK
    def test_passes_params_to_screener(self, _ce, mock_fmp):
        from wraquant.fundamental.screening import value_screen

        value_screen(
            max_pe=15,
            min_dividend_yield=0.03,
            max_debt_equity=1.0,
            min_market_cap=2_000_000_000,
            country="US",
            limit=30,
            fmp_client=mock_fmp,
        )
        mock_fmp.stock_screener.assert_called_once()
        call_kwargs = mock_fmp.stock_screener.call_args
        assert call_kwargs.kwargs.get("market_cap_gt") == 2_000_000_000
        assert call_kwargs.kwargs.get("dividend_gt") == 0.03
        assert call_kwargs.kwargs.get("limit") == 30

    @_PATCH_CHECK
    def test_non_empty_result(self, _ce, mock_fmp):
        from wraquant.fundamental.screening import value_screen

        result = value_screen(fmp_client=mock_fmp)
        assert len(result) > 0

    @_PATCH_CHECK
    def test_empty_screener_result(self, _ce):
        from wraquant.fundamental.screening import value_screen

        client = MagicMock()
        client.stock_screener.return_value = pd.DataFrame()
        result = value_screen(fmp_client=client)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Growth Screen
# ---------------------------------------------------------------------------


class TestGrowthScreen:
    @_PATCH_CHECK
    def test_returns_dataframe(self, _ce, mock_fmp):
        from wraquant.fundamental.screening import growth_screen

        result = growth_screen(fmp_client=mock_fmp)
        assert isinstance(result, pd.DataFrame)

    @_PATCH_CHECK
    def test_passes_market_cap(self, _ce, mock_fmp):
        from wraquant.fundamental.screening import growth_screen

        growth_screen(min_market_cap=1_000_000_000, fmp_client=mock_fmp)
        call_kwargs = mock_fmp.stock_screener.call_args
        assert call_kwargs.kwargs.get("market_cap_gt") == 1_000_000_000

    @_PATCH_CHECK
    def test_default_params(self, _ce, mock_fmp):
        from wraquant.fundamental.screening import growth_screen

        result = growth_screen(fmp_client=mock_fmp)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# Quality Screen (screening module)
# ---------------------------------------------------------------------------


class TestQualityScreenScreening:
    @_PATCH_CHECK
    def test_returns_dataframe(self, _ce, mock_fmp):
        from wraquant.fundamental.screening import quality_screen

        result = quality_screen(fmp_client=mock_fmp)
        assert isinstance(result, pd.DataFrame)

    @_PATCH_CHECK
    def test_passes_beta_lt(self, _ce, mock_fmp):
        from wraquant.fundamental.screening import quality_screen

        quality_screen(fmp_client=mock_fmp)
        call_kwargs = mock_fmp.stock_screener.call_args
        assert call_kwargs.kwargs.get("beta_lt") == 1.5


# ---------------------------------------------------------------------------
# Piotroski Screen
# ---------------------------------------------------------------------------


class TestPiotroskiScreen:
    @_PATCH_CHECK
    def test_returns_dataframe(self, _ce, mock_fmp):
        from wraquant.fundamental.screening import piotroski_screen

        result = piotroski_screen(min_score=7, fmp_client=mock_fmp)
        assert isinstance(result, pd.DataFrame)

    @_PATCH_CHECK
    def test_filters_by_score(self, _ce, mock_fmp):
        from wraquant.fundamental.screening import piotroski_screen

        result = piotroski_screen(min_score=7, fmp_client=mock_fmp)
        # All mock stocks should pass since score=8 >= 7
        assert len(result) == 5
        if not result.empty:
            assert (result["piotroski_score"] >= 7).all()

    @_PATCH_CHECK
    def test_filters_out_low_score(self, _ce):
        from wraquant.fundamental.screening import piotroski_screen

        client = MagicMock()
        client.stock_screener.return_value = _SCREENER_DF.copy()
        # Return low F-score for all
        client.score.return_value = {"piotroskiScore": 3, "altmanZScore": 1.5}
        result = piotroski_screen(min_score=7, fmp_client=client)
        assert len(result) == 0

    @_PATCH_CHECK
    def test_altman_z_included(self, _ce, mock_fmp):
        from wraquant.fundamental.screening import piotroski_screen

        result = piotroski_screen(min_score=7, fmp_client=mock_fmp)
        if not result.empty:
            assert "altman_z" in result.columns

    @_PATCH_CHECK
    def test_score_exception_handled(self, _ce):
        """If client.score() raises for a symbol, that symbol is skipped."""
        from wraquant.fundamental.screening import piotroski_screen

        client = MagicMock()
        client.stock_screener.return_value = _SCREENER_DF.head(2).copy()
        client.score.side_effect = [
            {"piotroskiScore": 8, "altmanZScore": 3.0},
            Exception("API error"),
        ]
        result = piotroski_screen(min_score=7, fmp_client=client)
        # Only the first symbol should survive
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Custom Screen
# ---------------------------------------------------------------------------


class TestCustomScreen:
    @_PATCH_CHECK
    def test_returns_dataframe(self, _ce, mock_fmp):
        from wraquant.fundamental.screening import custom_screen

        result = custom_screen({"min_market_cap": 1e9}, fmp_client=mock_fmp)
        assert isinstance(result, pd.DataFrame)

    @_PATCH_CHECK
    def test_passes_all_criteria(self, _ce, mock_fmp):
        from wraquant.fundamental.screening import custom_screen

        criteria = {
            "min_market_cap": 1e9,
            "max_market_cap": 1e13,
            "sector": "Technology",
            "country": "US",
            "min_dividend_yield": 0.01,
            "min_beta": 0.5,
            "max_beta": 1.5,
            "min_price": 10,
            "max_price": 500,
            "min_volume": 1_000_000,
            "limit": 25,
        }
        custom_screen(criteria, fmp_client=mock_fmp)
        call_kwargs = mock_fmp.stock_screener.call_args.kwargs
        assert call_kwargs["market_cap_gt"] == 1e9
        assert call_kwargs["market_cap_lt"] == 1e13
        assert call_kwargs["sector"] == "Technology"
        assert call_kwargs["country"] == "US"
        assert call_kwargs["dividend_gt"] == 0.01
        assert call_kwargs["beta_gt"] == 0.5
        assert call_kwargs["beta_lt"] == 1.5
        assert call_kwargs["price_gt"] == 10
        assert call_kwargs["price_lt"] == 500
        assert call_kwargs["volume_gt"] == 1_000_000
        assert call_kwargs["limit"] == 25

    @_PATCH_CHECK
    def test_default_country(self, _ce, mock_fmp):
        from wraquant.fundamental.screening import custom_screen

        custom_screen({}, fmp_client=mock_fmp)
        call_kwargs = mock_fmp.stock_screener.call_args.kwargs
        assert call_kwargs["country"] == "US"

    @_PATCH_CHECK
    def test_empty_criteria(self, _ce, mock_fmp):
        from wraquant.fundamental.screening import custom_screen

        result = custom_screen({}, fmp_client=mock_fmp)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Magic Formula Screen
# ---------------------------------------------------------------------------


class TestMagicFormulaScreen:
    @_PATCH_CHECK
    def test_returns_dataframe(self, _ce, mock_fmp):
        from wraquant.fundamental.screening import magic_formula_screen

        result = magic_formula_screen(top_n=3, fmp_client=mock_fmp)
        assert isinstance(result, pd.DataFrame)

    @_PATCH_CHECK
    def test_ranking_columns(self, _ce, mock_fmp):
        from wraquant.fundamental.screening import magic_formula_screen

        result = magic_formula_screen(top_n=5, fmp_client=mock_fmp)
        if not result.empty:
            assert "roic_rank" in result.columns
            assert "ey_rank" in result.columns
            assert "magic_rank" in result.columns

    @_PATCH_CHECK
    def test_top_n_limit(self, _ce, mock_fmp):
        from wraquant.fundamental.screening import magic_formula_screen

        result = magic_formula_screen(top_n=2, fmp_client=mock_fmp)
        assert len(result) <= 2

    @_PATCH_CHECK
    def test_empty_screener(self, _ce):
        from wraquant.fundamental.screening import magic_formula_screen

        client = MagicMock()
        client.stock_screener.return_value = pd.DataFrame()
        result = magic_formula_screen(fmp_client=client)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
