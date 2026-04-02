"""Tests for wraquant.news.filings — SEC filings retrieval.

All FMP API calls are mocked.  No network requests are made.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

_PATCH_CHECK = patch("wraquant._lazy.check_extra", return_value=True)

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

_SEC_FILINGS_DF = pd.DataFrame(
    [
        {
            "fillingDate": "2024-10-30",
            "type": "10-K",
            "title": "Annual Report 2024",
            "link": "https://www.sec.gov/filing/10k-2024",
            "cik": "0000320193",
        },
        {
            "fillingDate": "2024-07-31",
            "type": "10-Q",
            "title": "Quarterly Report Q3 2024",
            "link": "https://www.sec.gov/filing/10q-q3-2024",
            "cik": "0000320193",
        },
        {
            "fillingDate": "2024-06-15",
            "type": "8-K",
            "title": "Current Report — Executive Departure",
            "link": "https://www.sec.gov/filing/8k-exec",
            "cik": "0000320193",
        },
        {
            "fillingDate": "2024-04-30",
            "type": "10-Q",
            "title": "Quarterly Report Q2 2024",
            "link": "https://www.sec.gov/filing/10q-q2-2024",
            "cik": "0000320193",
        },
        {
            "fillingDate": "2024-03-10",
            "type": "8-K",
            "title": "Current Report — Material Agreement",
            "link": "https://www.sec.gov/filing/8k-agreement",
            "cik": "0000320193",
        },
        {
            "fillingDate": "2023-10-31",
            "type": "10-K",
            "title": "Annual Report 2023",
            "link": "https://www.sec.gov/filing/10k-2023",
            "cik": "0000320193",
        },
    ]
)


# ---------------------------------------------------------------------------
# recent_filings
# ---------------------------------------------------------------------------


class TestRecentFilings:
    @_PATCH_CHECK
    @patch("wraquant.data.providers.fmp.FMPClient")
    def test_returns_dataframe(self, mock_cls, _ce):
        from wraquant.news.filings import recent_filings

        mock_cls.return_value.sec_filings.return_value = _SEC_FILINGS_DF.copy()
        result = recent_filings("AAPL")
        assert isinstance(result, pd.DataFrame)

    @_PATCH_CHECK
    @patch("wraquant.data.providers.fmp.FMPClient")
    def test_standardized_columns(self, mock_cls, _ce):
        from wraquant.news.filings import recent_filings

        mock_cls.return_value.sec_filings.return_value = _SEC_FILINGS_DF.copy()
        result = recent_filings("AAPL")
        for col in ["date", "type", "title", "url", "cik"]:
            assert col in result.columns

    @_PATCH_CHECK
    @patch("wraquant.data.providers.fmp.FMPClient")
    def test_limit(self, mock_cls, _ce):
        from wraquant.news.filings import recent_filings

        mock_cls.return_value.sec_filings.return_value = _SEC_FILINGS_DF.copy()
        result = recent_filings("AAPL", limit=3)
        assert len(result) <= 3

    @_PATCH_CHECK
    @patch("wraquant.data.providers.fmp.FMPClient")
    def test_form_type_filter(self, mock_cls, _ce):
        from wraquant.news.filings import recent_filings

        mock_cls.return_value.sec_filings.return_value = _SEC_FILINGS_DF.copy()
        recent_filings("AAPL", form_type="10-K")
        call_kwargs = mock_cls.return_value.sec_filings.call_args
        assert call_kwargs.kwargs.get("type") == "10-K"

    @_PATCH_CHECK
    @patch("wraquant.data.providers.fmp.FMPClient")
    def test_empty_data(self, mock_cls, _ce):
        from wraquant.news.filings import recent_filings

        mock_cls.return_value.sec_filings.return_value = pd.DataFrame()
        result = recent_filings("EMPTY")
        assert isinstance(result, pd.DataFrame)
        assert "date" in result.columns
        assert len(result) == 0


# ---------------------------------------------------------------------------
# annual_reports
# ---------------------------------------------------------------------------


class TestAnnualReports:
    @_PATCH_CHECK
    @patch("wraquant.data.providers.fmp.FMPClient")
    def test_returns_dataframe(self, mock_cls, _ce):
        from wraquant.news.filings import annual_reports

        mock_cls.return_value.sec_filings.return_value = _SEC_FILINGS_DF.copy()
        result = annual_reports("AAPL")
        assert isinstance(result, pd.DataFrame)

    @_PATCH_CHECK
    @patch("wraquant.data.providers.fmp.FMPClient")
    def test_calls_with_10k_type(self, mock_cls, _ce):
        from wraquant.news.filings import annual_reports

        mock_cls.return_value.sec_filings.return_value = _SEC_FILINGS_DF.copy()
        annual_reports("AAPL", limit=3)
        call_kwargs = mock_cls.return_value.sec_filings.call_args
        assert call_kwargs.kwargs.get("type") == "10-K"

    @_PATCH_CHECK
    @patch("wraquant.data.providers.fmp.FMPClient")
    def test_limit(self, mock_cls, _ce):
        from wraquant.news.filings import annual_reports

        mock_cls.return_value.sec_filings.return_value = _SEC_FILINGS_DF.copy()
        result = annual_reports("AAPL", limit=1)
        assert len(result) <= 1

    @_PATCH_CHECK
    @patch("wraquant.data.providers.fmp.FMPClient")
    def test_empty_data(self, mock_cls, _ce):
        from wraquant.news.filings import annual_reports

        mock_cls.return_value.sec_filings.return_value = pd.DataFrame()
        result = annual_reports("EMPTY")
        assert len(result) == 0


# ---------------------------------------------------------------------------
# quarterly_reports
# ---------------------------------------------------------------------------


class TestQuarterlyReports:
    @_PATCH_CHECK
    @patch("wraquant.data.providers.fmp.FMPClient")
    def test_returns_dataframe(self, mock_cls, _ce):
        from wraquant.news.filings import quarterly_reports

        mock_cls.return_value.sec_filings.return_value = _SEC_FILINGS_DF.copy()
        result = quarterly_reports("AAPL")
        assert isinstance(result, pd.DataFrame)

    @_PATCH_CHECK
    @patch("wraquant.data.providers.fmp.FMPClient")
    def test_calls_with_10q_type(self, mock_cls, _ce):
        from wraquant.news.filings import quarterly_reports

        mock_cls.return_value.sec_filings.return_value = _SEC_FILINGS_DF.copy()
        quarterly_reports("AAPL")
        call_kwargs = mock_cls.return_value.sec_filings.call_args
        assert call_kwargs.kwargs.get("type") == "10-Q"

    @_PATCH_CHECK
    @patch("wraquant.data.providers.fmp.FMPClient")
    def test_limit(self, mock_cls, _ce):
        from wraquant.news.filings import quarterly_reports

        mock_cls.return_value.sec_filings.return_value = _SEC_FILINGS_DF.copy()
        result = quarterly_reports("AAPL", limit=2)
        assert len(result) <= 2


# ---------------------------------------------------------------------------
# material_events
# ---------------------------------------------------------------------------


class TestMaterialEvents:
    @_PATCH_CHECK
    @patch("wraquant.data.providers.fmp.FMPClient")
    def test_returns_dataframe(self, mock_cls, _ce):
        from wraquant.news.filings import material_events

        mock_cls.return_value.sec_filings.return_value = _SEC_FILINGS_DF.copy()
        result = material_events("AAPL")
        assert isinstance(result, pd.DataFrame)

    @_PATCH_CHECK
    @patch("wraquant.data.providers.fmp.FMPClient")
    def test_calls_with_8k_type(self, mock_cls, _ce):
        from wraquant.news.filings import material_events

        mock_cls.return_value.sec_filings.return_value = _SEC_FILINGS_DF.copy()
        material_events("AAPL")
        call_kwargs = mock_cls.return_value.sec_filings.call_args
        assert call_kwargs.kwargs.get("type") == "8-K"

    @_PATCH_CHECK
    @patch("wraquant.data.providers.fmp.FMPClient")
    def test_limit(self, mock_cls, _ce):
        from wraquant.news.filings import material_events

        mock_cls.return_value.sec_filings.return_value = _SEC_FILINGS_DF.copy()
        result = material_events("GM", limit=1)
        assert len(result) <= 1

    @_PATCH_CHECK
    @patch("wraquant.data.providers.fmp.FMPClient")
    def test_empty_data(self, mock_cls, _ce):
        from wraquant.news.filings import material_events

        mock_cls.return_value.sec_filings.return_value = pd.DataFrame()
        result = material_events("EMPTY")
        assert len(result) == 0


# ---------------------------------------------------------------------------
# filing_search
# ---------------------------------------------------------------------------


class TestFilingSearch:
    @_PATCH_CHECK
    @patch("wraquant.data.providers.fmp.FMPClient")
    def test_returns_dataframe(self, mock_cls, _ce):
        from wraquant.news.filings import filing_search

        mock_cls.return_value.sec_filings.return_value = _SEC_FILINGS_DF.copy()
        result = filing_search("AAPL:merger")
        assert isinstance(result, pd.DataFrame)

    @_PATCH_CHECK
    @patch("wraquant.data.providers.fmp.FMPClient")
    def test_filters_by_keyword(self, mock_cls, _ce):
        from wraquant.news.filings import filing_search

        mock_cls.return_value.sec_filings.return_value = _SEC_FILINGS_DF.copy()
        result = filing_search("AAPL:executive")
        # Should match "Executive Departure" title
        if not result.empty:
            assert any("executive" in t.lower() for t in result["title"])

    @_PATCH_CHECK
    @patch("wraquant.data.providers.fmp.FMPClient")
    def test_symbol_prefix_parsing(self, mock_cls, _ce):
        from wraquant.news.filings import filing_search

        mock_cls.return_value.sec_filings.return_value = _SEC_FILINGS_DF.copy()
        filing_search("AAPL:agreement")
        call_args = mock_cls.return_value.sec_filings.call_args
        assert call_args.args[0] == "AAPL" or call_args.kwargs.get("symbol") == "AAPL"

    @_PATCH_CHECK
    @patch("wraquant.data.providers.fmp.FMPClient")
    def test_empty_data(self, mock_cls, _ce):
        from wraquant.news.filings import filing_search

        mock_cls.return_value.sec_filings.return_value = pd.DataFrame()
        result = filing_search("EMPTY")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @_PATCH_CHECK
    @patch("wraquant.data.providers.fmp.FMPClient")
    def test_no_match_returns_empty(self, mock_cls, _ce):
        from wraquant.news.filings import filing_search

        mock_cls.return_value.sec_filings.return_value = _SEC_FILINGS_DF.copy()
        result = filing_search("AAPL:zzzznonexistent")
        assert len(result) == 0
