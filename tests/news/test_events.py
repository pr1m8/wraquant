"""Tests for wraquant.news.events — FMP-backed event-driven analysis.

All FMP API calls are mocked.  No network requests are made.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

_PATCH_CHECK = patch("wraquant._lazy.check_extra", return_value=True)

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

_NOW = pd.Timestamp.now()

_EARNINGS_CAL_DF = pd.DataFrame(
    [
        {
            "symbol": "AAPL",
            "date": str((_NOW + pd.Timedelta(days=5)).date()),
            "epsEstimated": 1.50,
            "epsActual": None,
            "revenueEstimated": 90_000_000_000,
            "revenueActual": None,
            "time": "amc",
        },
        {
            "symbol": "MSFT",
            "date": str((_NOW + pd.Timedelta(days=10)).date()),
            "epsEstimated": 2.80,
            "epsActual": None,
            "revenueEstimated": 55_000_000_000,
            "revenueActual": None,
            "time": "bmo",
        },
    ]
)

_EARNINGS_SURPRISES_DF = pd.DataFrame(
    [
        {"date": "2024-10-31", "actualEarningResult": 1.65, "estimatedEarning": 1.50},
        {"date": "2024-07-31", "actualEarningResult": 1.40, "estimatedEarning": 1.35},
        {"date": "2024-04-30", "actualEarningResult": 1.55, "estimatedEarning": 1.60},
        {"date": "2024-01-31", "actualEarningResult": 2.20, "estimatedEarning": 2.10},
    ]
)

_EARNINGS_FOR_UPCOMING_DF = pd.DataFrame(
    [
        {
            "date": str((_NOW + pd.Timedelta(days=15)).date()),
            "epsEstimated": 3.00,
            "revenueEstimated": 100_000_000_000,
            "time": "amc",
        },
        {
            "date": str((_NOW - pd.Timedelta(days=60)).date()),
            "epsEstimated": 2.80,
            "revenueEstimated": 95_000_000_000,
            "time": "bmo",
        },
    ]
)

_DIVIDEND_EARNINGS_DF = pd.DataFrame(
    [
        {"date": "2024-11-15", "dividend": 0.25},
        {"date": "2024-08-15", "dividend": 0.24},
        {"date": "2024-05-15", "dividend": 0.23},
        {"date": "2024-02-15", "dividend": 0.22},
    ]
)

_SEC_FILINGS_INSIDER = pd.DataFrame(
    [
        {
            "fillingDate": "2024-11-01",
            "reportingName": "Tim Cook",
            "transactionType": "Sale",
            "securitiesTransacted": 50_000,
            "price": 200.0,
        },
        {
            "fillingDate": "2024-10-15",
            "reportingName": "Luca Maestri",
            "transactionType": "Purchase",
            "securitiesTransacted": 10_000,
            "price": 190.0,
        },
        {
            "fillingDate": "2024-09-20",
            "reportingName": "Tim Cook",
            "transactionType": "Sale",
            "securitiesTransacted": 30_000,
            "price": 195.0,
        },
    ]
)


# ---------------------------------------------------------------------------
# Earnings Calendar
# ---------------------------------------------------------------------------


class TestEarningsCalendar:
    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_returns_dataframe(self, mock_cls, _ce):
        from wraquant.news.events import earnings_calendar

        mock_cls.return_value.earnings.return_value = _EARNINGS_CAL_DF.copy()
        result = earnings_calendar("2024-10-01", "2024-10-31")
        assert isinstance(result, pd.DataFrame)

    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_column_rename(self, mock_cls, _ce):
        from wraquant.news.events import earnings_calendar

        mock_cls.return_value.earnings.return_value = _EARNINGS_CAL_DF.copy()
        result = earnings_calendar()
        if not result.empty:
            assert "eps_estimated" in result.columns
            assert "revenue_estimated" in result.columns

    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_empty_data(self, mock_cls, _ce):
        from wraquant.news.events import earnings_calendar

        mock_cls.return_value.earnings.return_value = pd.DataFrame()
        result = earnings_calendar()
        assert isinstance(result, pd.DataFrame)
        assert "symbol" in result.columns

    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_defaults_to_7_days(self, mock_cls, _ce):
        from wraquant.news.events import earnings_calendar

        mock_cls.return_value.earnings.return_value = _EARNINGS_CAL_DF.copy()
        # Should not raise when only from_date is given
        result = earnings_calendar("2024-10-01")
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Earnings Surprises
# ---------------------------------------------------------------------------


class TestEarningsSurprises:
    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_returns_dataframe(self, mock_cls, _ce):
        from wraquant.news.events import earnings_surprises

        mock_cls.return_value.earnings_surprises.return_value = (
            _EARNINGS_SURPRISES_DF.copy()
        )
        result = earnings_surprises("AAPL")
        assert isinstance(result, pd.DataFrame)

    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_surprise_columns(self, mock_cls, _ce):
        from wraquant.news.events import earnings_surprises

        mock_cls.return_value.earnings_surprises.return_value = (
            _EARNINGS_SURPRISES_DF.copy()
        )
        result = earnings_surprises("AAPL")
        assert "surprise" in result.columns
        assert "surprise_pct" in result.columns
        assert "beat" in result.columns

    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_beat_detection(self, mock_cls, _ce):
        from wraquant.news.events import earnings_surprises

        mock_cls.return_value.earnings_surprises.return_value = (
            _EARNINGS_SURPRISES_DF.copy()
        )
        result = earnings_surprises("AAPL")
        # Row 0: 1.65 > 1.50 => beat, Row 2: 1.55 < 1.60 => miss
        assert result["beat"].iloc[0] is True or result["beat"].iloc[0]  # noqa: E712
        assert (
            result["beat"].iloc[2] is False or not result["beat"].iloc[2]
        )  # noqa: E712

    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_surprise_sign(self, mock_cls, _ce):
        from wraquant.news.events import earnings_surprises

        mock_cls.return_value.earnings_surprises.return_value = (
            _EARNINGS_SURPRISES_DF.copy()
        )
        result = earnings_surprises("AAPL")
        assert result["surprise"].iloc[0] > 0  # 1.65 > 1.50
        assert result["surprise"].iloc[2] < 0  # 1.55 < 1.60

    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_limit(self, mock_cls, _ce):
        from wraquant.news.events import earnings_surprises

        mock_cls.return_value.earnings_surprises.return_value = (
            _EARNINGS_SURPRISES_DF.copy()
        )
        result = earnings_surprises("AAPL", limit=2)
        assert len(result) <= 2

    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_empty_data(self, mock_cls, _ce):
        from wraquant.news.events import earnings_surprises

        mock_cls.return_value.earnings_surprises.return_value = pd.DataFrame()
        result = earnings_surprises("EMPTY")
        assert isinstance(result, pd.DataFrame)
        assert "surprise" in result.columns


# ---------------------------------------------------------------------------
# Upcoming Earnings
# ---------------------------------------------------------------------------


class TestUpcomingEarnings:
    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_keys(self, mock_cls, _ce):
        from wraquant.news.events import upcoming_earnings

        mock_cls.return_value.earnings.return_value = _EARNINGS_FOR_UPCOMING_DF.copy()
        result = upcoming_earnings("AAPL")
        expected_keys = {
            "symbol",
            "next_date",
            "eps_estimate",
            "revenue_estimate",
            "days_until",
            "time",
        }
        assert expected_keys == set(result.keys())

    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_symbol_passthrough(self, mock_cls, _ce):
        from wraquant.news.events import upcoming_earnings

        mock_cls.return_value.earnings.return_value = _EARNINGS_FOR_UPCOMING_DF.copy()
        result = upcoming_earnings("GOOG")
        assert result["symbol"] == "GOOG"

    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_finds_future_date(self, mock_cls, _ce):
        from wraquant.news.events import upcoming_earnings

        mock_cls.return_value.earnings.return_value = _EARNINGS_FOR_UPCOMING_DF.copy()
        result = upcoming_earnings("AAPL")
        assert result["next_date"] is not None
        assert result["days_until"] is not None
        assert result["days_until"] >= 0

    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_empty_data(self, mock_cls, _ce):
        from wraquant.news.events import upcoming_earnings

        mock_cls.return_value.earnings.return_value = pd.DataFrame()
        result = upcoming_earnings("EMPTY")
        assert result["next_date"] is None
        assert result["days_until"] is None

    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_all_past_dates(self, mock_cls, _ce):
        from wraquant.news.events import upcoming_earnings

        past_df = pd.DataFrame(
            [{"date": str((_NOW - pd.Timedelta(days=30)).date()), "epsEstimated": 1.0}]
        )
        mock_cls.return_value.earnings.return_value = past_df
        result = upcoming_earnings("OLD")
        assert result["next_date"] is None


# ---------------------------------------------------------------------------
# Earnings History
# ---------------------------------------------------------------------------


class TestEarningsHistory:
    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_keys(self, mock_cls, _ce):
        from wraquant.news.events import earnings_history

        mock_cls.return_value.earnings_surprises.return_value = (
            _EARNINGS_SURPRISES_DF.copy()
        )
        result = earnings_history("AAPL")
        expected_keys = {
            "symbol",
            "quarters_analyzed",
            "surprises",
            "beat_rate",
            "miss_rate",
            "avg_surprise",
            "avg_beat_magnitude",
            "avg_miss_magnitude",
            "surprise_std",
            "streak",
            "pead_signal",
        }
        assert expected_keys == set(result.keys())

    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_beat_rate_range(self, mock_cls, _ce):
        from wraquant.news.events import earnings_history

        mock_cls.return_value.earnings_surprises.return_value = (
            _EARNINGS_SURPRISES_DF.copy()
        )
        result = earnings_history("AAPL")
        assert 0.0 <= result["beat_rate"] <= 1.0
        assert 0.0 <= result["miss_rate"] <= 1.0
        assert abs(result["beat_rate"] + result["miss_rate"] - 1.0) < 1e-9

    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_streak_structure(self, mock_cls, _ce):
        from wraquant.news.events import earnings_history

        mock_cls.return_value.earnings_surprises.return_value = (
            _EARNINGS_SURPRISES_DF.copy()
        )
        result = earnings_history("AAPL")
        assert "type" in result["streak"]
        assert "length" in result["streak"]
        assert result["streak"]["type"] in {"beat", "miss", "none"}
        assert result["streak"]["length"] >= 0

    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_pead_signal_values(self, mock_cls, _ce):
        from wraquant.news.events import earnings_history

        mock_cls.return_value.earnings_surprises.return_value = (
            _EARNINGS_SURPRISES_DF.copy()
        )
        result = earnings_history("AAPL")
        assert result["pead_signal"] in {
            "strong_beat",
            "moderate_beat",
            "neutral",
            "moderate_miss",
            "strong_miss",
        }

    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_empty_data(self, mock_cls, _ce):
        from wraquant.news.events import earnings_history

        mock_cls.return_value.earnings_surprises.return_value = pd.DataFrame()
        result = earnings_history("EMPTY")
        assert result["quarters_analyzed"] == 0
        assert result["beat_rate"] == 0.0
        assert result["pead_signal"] == "neutral"


# ---------------------------------------------------------------------------
# Dividend History
# ---------------------------------------------------------------------------


class TestDividendHistory:
    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_keys(self, mock_cls, _ce):
        from wraquant.news.events import dividend_history

        mock_cls.return_value.earnings.return_value = _DIVIDEND_EARNINGS_DF.copy()
        result = dividend_history("KO")
        expected_keys = {
            "symbol",
            "dividends",
            "total_dividends",
            "current_annual_dividend",
            "dividend_growth_rate",
            "consecutive_payments",
            "is_grower",
        }
        assert expected_keys == set(result.keys())

    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_annual_dividend(self, mock_cls, _ce):
        from wraquant.news.events import dividend_history

        mock_cls.return_value.earnings.return_value = _DIVIDEND_EARNINGS_DF.copy()
        result = dividend_history("KO")
        # Most recent * 4 (quarterly assumption)
        assert result["current_annual_dividend"] == 0.25 * 4

    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_is_grower(self, mock_cls, _ce):
        from wraquant.news.events import dividend_history

        mock_cls.return_value.earnings.return_value = _DIVIDEND_EARNINGS_DF.copy()
        result = dividend_history("KO")
        # 0.25 >= 0.24 >= 0.23 => True
        assert result["is_grower"] is True

    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_empty_data(self, mock_cls, _ce):
        from wraquant.news.events import dividend_history

        mock_cls.return_value.earnings.return_value = pd.DataFrame()
        result = dividend_history("EMPTY")
        assert result["total_dividends"] == 0
        assert result["current_annual_dividend"] == 0.0

    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_growth_rate_positive(self, mock_cls, _ce):
        from wraquant.news.events import dividend_history

        mock_cls.return_value.earnings.return_value = _DIVIDEND_EARNINGS_DF.copy()
        result = dividend_history("KO")
        # Dividends are increasing (0.22 -> 0.25), growth rate > 0
        assert result["dividend_growth_rate"] > 0


# ---------------------------------------------------------------------------
# Insider Activity
# ---------------------------------------------------------------------------


class TestInsiderActivity:
    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_keys(self, mock_cls, _ce):
        from wraquant.news.events import insider_activity

        mock_cls.return_value.sec_filings.return_value = _SEC_FILINGS_INSIDER.copy()
        result = insider_activity("AAPL")
        expected_keys = {
            "symbol",
            "transactions",
            "total_transactions",
            "buy_count",
            "sell_count",
            "buy_sell_ratio",
            "net_shares",
            "net_value",
            "notable_trades",
            "signal",
        }
        assert expected_keys == set(result.keys())

    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_buy_sell_counts(self, mock_cls, _ce):
        from wraquant.news.events import insider_activity

        mock_cls.return_value.sec_filings.return_value = _SEC_FILINGS_INSIDER.copy()
        result = insider_activity("AAPL")
        # 2 sales, 1 purchase
        assert result["sell_count"] == 2
        assert result["buy_count"] == 1

    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_total_transactions(self, mock_cls, _ce):
        from wraquant.news.events import insider_activity

        mock_cls.return_value.sec_filings.return_value = _SEC_FILINGS_INSIDER.copy()
        result = insider_activity("AAPL")
        assert result["total_transactions"] == 3

    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_signal_values(self, mock_cls, _ce):
        from wraquant.news.events import insider_activity

        mock_cls.return_value.sec_filings.return_value = _SEC_FILINGS_INSIDER.copy()
        result = insider_activity("AAPL")
        assert result["signal"] in {"bullish", "bearish", "neutral"}

    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_empty_filings(self, mock_cls, _ce):
        from wraquant.news.events import insider_activity

        mock_cls.return_value.sec_filings.return_value = pd.DataFrame()
        result = insider_activity("EMPTY")
        assert result["total_transactions"] == 0
        assert result["signal"] == "neutral"

    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_exception_returns_default(self, mock_cls, _ce):
        from wraquant.news.events import insider_activity

        mock_cls.return_value.sec_filings.side_effect = Exception("API error")
        result = insider_activity("ERR")
        assert result["total_transactions"] == 0
        assert result["signal"] == "neutral"

    @_PATCH_CHECK
    @patch("wraquant.news.events.FMPClient")
    def test_notable_trades(self, mock_cls, _ce):
        from wraquant.news.events import insider_activity

        # Create a trade > $1M
        big_trade_df = pd.DataFrame(
            [
                {
                    "fillingDate": "2024-11-01",
                    "reportingName": "CEO",
                    "transactionType": "Purchase",
                    "securitiesTransacted": 100_000,
                    "price": 200.0,
                },
            ]
        )
        mock_cls.return_value.sec_filings.return_value = big_trade_df
        result = insider_activity("BIG")
        # $20M trade should appear in notable_trades
        assert len(result["notable_trades"]) > 0
