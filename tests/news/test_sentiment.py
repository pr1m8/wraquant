"""Tests for wraquant.news.sentiment — FMP-backed sentiment analysis.

Mocks FMPClient for news_sentiment, sentiment_timeseries, and
sentiment_signal.  Tests the keyword scorer directly (no mock needed).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from wraquant.news.sentiment import (
    _keyword_score,
    _recency_weights,
    _resolve_col,
    earnings_surprise,
    news_signal,
    sentiment_aggregate,
    sentiment_score,
)

# ---------------------------------------------------------------------------
# Mock news data
# ---------------------------------------------------------------------------

_NOW = pd.Timestamp.now(tz="UTC")

_NEWS_DF = pd.DataFrame(
    {
        "title": [
            "Apple beats earnings estimates, shares surge",
            "Apple faces antitrust investigation in Europe",
            "Apple launches new iPhone with strong demand",
            "Market volatility rises amid uncertainty",
            "Apple stock rallied on strong quarterly revenue growth",
            "Tech stocks decline on tariff fears",
        ],
        "publishedDate": [
            str(_NOW - pd.Timedelta(days=0)),
            str(_NOW - pd.Timedelta(days=1)),
            str(_NOW - pd.Timedelta(days=2)),
            str(_NOW - pd.Timedelta(days=3)),
            str(_NOW - pd.Timedelta(days=5)),
            str(_NOW - pd.Timedelta(days=7)),
        ],
        "site": ["Reuters", "Bloomberg", "CNBC", "WSJ", "Reuters", "CNBC"],
        "url": [
            "https://example.com/1",
            "https://example.com/2",
            "https://example.com/3",
            "https://example.com/4",
            "https://example.com/5",
            "https://example.com/6",
        ],
    }
)

_PATCH_CHECK = patch("wraquant._lazy.check_extra", return_value=True)


# ---------------------------------------------------------------------------
# Keyword scorer (no mock needed)
# ---------------------------------------------------------------------------


class TestKeywordScorer:
    def test_positive_text(self):
        score = _keyword_score("Company beats earnings and shares surge")
        assert score > 0

    def test_negative_text(self):
        score = _keyword_score("Stock crashes amid bankruptcy fears")
        assert score < 0

    def test_neutral_text(self):
        score = _keyword_score("Company released its annual report today")
        assert score == 0.0

    def test_empty_text(self):
        assert _keyword_score("") == 0.0

    def test_negation_flips_sentiment(self):
        positive = _keyword_score("Company gains")
        negated = _keyword_score("Company doesn't gain")
        # Negation should flip or reduce the positive score
        assert negated < positive

    def test_intensifier_amplifies(self):
        normal = _keyword_score("strong revenue")
        intensified = _keyword_score("extremely strong revenue")
        assert abs(intensified) >= abs(normal)

    def test_range_bounded(self):
        """Score should always be in [-1, 1]."""
        texts = [
            "surge surge surge surge beat rally gain profit",
            "crash crash crash crash loss decline drop fall",
            "the a an for with about",
        ]
        for text in texts:
            score = _keyword_score(text)
            assert -1.0 <= score <= 1.0

    def test_mixed_sentiment(self):
        score = _keyword_score("Stock rallied despite recession fears")
        # Mixed signals — should be non-zero but modest
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# Recency weights
# ---------------------------------------------------------------------------


class TestRecencyWeights:
    def test_most_recent_gets_weight_1(self):
        dates = pd.Series([_NOW, _NOW - pd.Timedelta(days=7)])
        weights = _recency_weights(dates, half_life_days=7.0)
        assert abs(weights[0] - 1.0) < 1e-6

    def test_half_life_decay(self):
        dates = pd.Series([_NOW, _NOW - pd.Timedelta(days=7)])
        weights = _recency_weights(dates, half_life_days=7.0)
        # 7 days ago should have weight ~ 0.5
        assert abs(weights[1] - 0.5) < 0.05

    def test_empty_series(self):
        dates = pd.Series([], dtype="datetime64[ns, UTC]")
        weights = _recency_weights(dates)
        assert len(weights) == 0


# ---------------------------------------------------------------------------
# sentiment_score (low-level scoring, no FMP)
# ---------------------------------------------------------------------------


class TestSentimentScore:
    def test_single_text(self):
        result = sentiment_score("Stock rallied on strong earnings", engine="keyword")
        assert len(result["scores"]) == 1
        assert result["scores"][0] > 0  # positive sentiment keywords
        assert result["engine"] == "keyword"

    def test_multiple_texts(self):
        texts = [
            "Shares surge on approval",
            "Stock crashes amid fears",
            "Annual report released",
        ]
        result = sentiment_score(texts, engine="keyword")
        assert len(result["scores"]) == 3
        assert result["scores"][0] > 0  # surge, approval
        assert result["scores"][1] < 0  # crashes, fears
        assert isinstance(result["mean_score"], float)

    def test_empty_list(self):
        result = sentiment_score([], engine="keyword")
        assert result["scores"] == []
        assert result["mean_score"] == 0.0

    def test_engine_keyword_explicit(self):
        result = sentiment_score("test", engine="keyword")
        assert result["engine"] == "keyword"


# ---------------------------------------------------------------------------
# earnings_surprise (standalone, no FMP)
# ---------------------------------------------------------------------------


class TestEarningsSurprise:
    def test_positive_surprise(self):
        result = earnings_surprise(actual=2.50, estimate=2.30)
        assert result > 0
        assert abs(result - (2.50 - 2.30) / 2.30) < 1e-10

    def test_negative_surprise(self):
        result = earnings_surprise(actual=1.80, estimate=2.00)
        assert result < 0
        assert abs(result - (-0.1)) < 1e-10

    def test_zero_estimate(self):
        assert earnings_surprise(actual=1.0, estimate=0.0) == 0.0

    def test_exact_match(self):
        assert earnings_surprise(actual=2.0, estimate=2.0) == 0.0


# ---------------------------------------------------------------------------
# sentiment_aggregate (standalone, no FMP)
# ---------------------------------------------------------------------------


class TestSentimentAggregate:
    def test_mean(self):
        result = sentiment_aggregate([0.5, 0.3, -0.1, 0.7])
        assert abs(result - 0.35) < 1e-10

    def test_median(self):
        result = sentiment_aggregate([0.5, 0.3, -0.1, 0.7], method="median")
        assert abs(result - 0.4) < 1e-10

    def test_empty(self):
        assert sentiment_aggregate([]) == 0.0

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            sentiment_aggregate([0.5], method="max")


# ---------------------------------------------------------------------------
# news_signal (standalone, no FMP)
# ---------------------------------------------------------------------------


class TestNewsSignal:
    def test_threshold_classification(self):
        sent = pd.Series([0.8, 0.3, -0.6, 0.1, -0.9])
        signals = news_signal(sent, threshold=0.5)
        expected = [1, 0, -1, 0, -1]
        np.testing.assert_array_equal(signals.values, expected)

    def test_list_input(self):
        signals = news_signal([0.8, -0.8, 0.0], threshold=0.5)
        assert isinstance(signals, pd.Series)
        np.testing.assert_array_equal(signals.values, [1, -1, 0])

    def test_all_neutral(self):
        signals = news_signal([0.1, -0.1, 0.0, 0.2], threshold=0.5)
        np.testing.assert_array_equal(signals.values, [0, 0, 0, 0])

    def test_preserves_index(self):
        idx = pd.date_range("2024-01-01", periods=3)
        sent = pd.Series([0.8, 0.0, -0.8], index=idx)
        signals = news_signal(sent, threshold=0.5)
        pd.testing.assert_index_equal(signals.index, idx)


# ---------------------------------------------------------------------------
# news_sentiment (FMP mocked)
# ---------------------------------------------------------------------------


class TestNewsSentiment:
    @_PATCH_CHECK
    @patch("wraquant.news.sentiment.FMPClient")
    def test_keys(self, mock_cls, _ce):
        from wraquant.news.sentiment import news_sentiment

        mock_cls.return_value.stock_news.return_value = _NEWS_DF.copy()
        mock_cls.return_value.press_releases.return_value = pd.DataFrame()
        result = news_sentiment("AAPL", engine="keyword")
        expected_keys = {
            "symbol",
            "engine",
            "article_count",
            "articles",
            "aggregate",
            "trend",
            "trend_delta",
            "news_volume",
        }
        assert expected_keys == set(result.keys())

    @_PATCH_CHECK
    @patch("wraquant.news.sentiment.FMPClient")
    def test_article_count(self, mock_cls, _ce):
        from wraquant.news.sentiment import news_sentiment

        mock_cls.return_value.stock_news.return_value = _NEWS_DF.copy()
        mock_cls.return_value.press_releases.return_value = pd.DataFrame()
        result = news_sentiment("AAPL", engine="keyword")
        assert result["article_count"] == len(_NEWS_DF)

    @_PATCH_CHECK
    @patch("wraquant.news.sentiment.FMPClient")
    def test_aggregate_keys(self, mock_cls, _ce):
        from wraquant.news.sentiment import news_sentiment

        mock_cls.return_value.stock_news.return_value = _NEWS_DF.copy()
        mock_cls.return_value.press_releases.return_value = pd.DataFrame()
        result = news_sentiment("AAPL", engine="keyword")
        agg = result["aggregate"]
        expected_agg_keys = {
            "mean",
            "weighted_mean",
            "median",
            "std",
            "bullish_pct",
            "bearish_pct",
            "neutral_pct",
        }
        assert expected_agg_keys == set(agg.keys())

    @_PATCH_CHECK
    @patch("wraquant.news.sentiment.FMPClient")
    def test_trend_values(self, mock_cls, _ce):
        from wraquant.news.sentiment import news_sentiment

        mock_cls.return_value.stock_news.return_value = _NEWS_DF.copy()
        mock_cls.return_value.press_releases.return_value = pd.DataFrame()
        result = news_sentiment("AAPL", engine="keyword")
        assert result["trend"] in {"improving", "deteriorating", "stable"}

    @_PATCH_CHECK
    @patch("wraquant.news.sentiment.FMPClient")
    def test_empty_news(self, mock_cls, _ce):
        from wraquant.news.sentiment import news_sentiment

        mock_cls.return_value.stock_news.return_value = pd.DataFrame()
        mock_cls.return_value.press_releases.return_value = pd.DataFrame()
        result = news_sentiment("UNKNOWN", engine="keyword")
        assert result["article_count"] == 0
        assert result["trend"] == "stable"
        assert result["aggregate"]["mean"] == 0.0

    @_PATCH_CHECK
    @patch("wraquant.news.sentiment.FMPClient")
    def test_pct_sum_to_one(self, mock_cls, _ce):
        from wraquant.news.sentiment import news_sentiment

        mock_cls.return_value.stock_news.return_value = _NEWS_DF.copy()
        mock_cls.return_value.press_releases.return_value = pd.DataFrame()
        result = news_sentiment("AAPL", engine="keyword")
        agg = result["aggregate"]
        total = agg["bullish_pct"] + agg["bearish_pct"] + agg["neutral_pct"]
        assert abs(total - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# sentiment_timeseries (FMP mocked)
# ---------------------------------------------------------------------------


class TestSentimentTimeseries:
    @_PATCH_CHECK
    @patch("wraquant.news.sentiment.FMPClient")
    def test_returns_series(self, mock_cls, _ce):
        from wraquant.news.sentiment import sentiment_timeseries

        mock_cls.return_value.stock_news.return_value = _NEWS_DF.copy()
        result = sentiment_timeseries("AAPL", days=30, engine="keyword")
        assert isinstance(result, pd.Series)
        assert result.name == "sentiment"

    @_PATCH_CHECK
    @patch("wraquant.news.sentiment.FMPClient")
    def test_empty_news_returns_zeros(self, mock_cls, _ce):
        from wraquant.news.sentiment import sentiment_timeseries

        mock_cls.return_value.stock_news.return_value = pd.DataFrame()
        result = sentiment_timeseries("EMPTY", days=10, engine="keyword")
        assert isinstance(result, pd.Series)
        assert (result == 0.0).all()


# ---------------------------------------------------------------------------
# sentiment_signal (FMP mocked — delegates to news_sentiment)
# ---------------------------------------------------------------------------


class TestSentimentSignalFMP:
    @_PATCH_CHECK
    @patch("wraquant.news.sentiment.FMPClient")
    def test_returns_string(self, mock_cls, _ce):
        from wraquant.news.sentiment import sentiment_signal

        mock_cls.return_value.stock_news.return_value = _NEWS_DF.copy()
        mock_cls.return_value.press_releases.return_value = pd.DataFrame()
        result = sentiment_signal("AAPL", engine="keyword")
        assert result in {"bullish", "bearish", "neutral"}

    @_PATCH_CHECK
    @patch("wraquant.news.sentiment.FMPClient")
    def test_neutral_when_empty(self, mock_cls, _ce):
        from wraquant.news.sentiment import sentiment_signal

        mock_cls.return_value.stock_news.return_value = pd.DataFrame()
        mock_cls.return_value.press_releases.return_value = pd.DataFrame()
        result = sentiment_signal("EMPTY", engine="keyword")
        assert result == "neutral"


# ---------------------------------------------------------------------------
# _resolve_col helper
# ---------------------------------------------------------------------------


class TestResolveCol:
    def test_finds_first_match(self):
        df = pd.DataFrame({"title": [1], "text": [2]})
        assert _resolve_col(df, ["title", "text"]) == "title"

    def test_fallback(self):
        df = pd.DataFrame({"text": [1]})
        assert _resolve_col(df, ["title", "text"]) == "text"

    def test_none_when_missing(self):
        df = pd.DataFrame({"foo": [1]})
        assert _resolve_col(df, ["title", "text"]) is None
