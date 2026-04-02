"""Tests for wraquant.news.sentiment module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.news.sentiment import (
    earnings_surprise,
    news_signal,
    sentiment_aggregate,
    sentiment_score,
)

# ---------------------------------------------------------------------------
# sentiment_score
# ---------------------------------------------------------------------------


class TestSentimentScore:
    def test_single_text_returns_neutral(self):
        result = sentiment_score("Stock rallied on strong earnings")
        assert result["scores"] == [0.0]
        assert result["mean_score"] == 0.0

    def test_multiple_texts(self):
        texts = ["Good news", "Bad news", "Neutral report"]
        result = sentiment_score(texts)
        assert len(result["scores"]) == 3
        assert all(isinstance(s, float) for s in result["scores"])
        assert isinstance(result["mean_score"], float)

    def test_empty_list(self):
        result = sentiment_score([])
        assert result["scores"] == []
        assert result["mean_score"] == 0.0


# ---------------------------------------------------------------------------
# earnings_surprise
# ---------------------------------------------------------------------------


class TestEarningsSurprise:
    def test_positive_surprise(self):
        result = earnings_surprise(actual=2.50, estimate=2.30)
        assert isinstance(result, float)
        assert result > 0
        assert abs(result - 0.08695652173913043) < 1e-10

    def test_negative_surprise(self):
        result = earnings_surprise(actual=1.80, estimate=2.00)
        assert result < 0
        assert abs(result - (-0.1)) < 1e-10

    def test_zero_surprise(self):
        result = earnings_surprise(actual=2.0, estimate=2.0)
        assert result == 0.0

    def test_zero_estimate(self):
        result = earnings_surprise(actual=1.0, estimate=0.0)
        assert result == 0.0

    def test_negative_estimate(self):
        result = earnings_surprise(actual=-0.5, estimate=-1.0)
        assert isinstance(result, float)
        assert abs(result - 0.5) < 1e-10


# ---------------------------------------------------------------------------
# sentiment_aggregate
# ---------------------------------------------------------------------------


class TestSentimentAggregate:
    def test_mean(self):
        result = sentiment_aggregate([0.5, 0.3, -0.1, 0.7])
        assert isinstance(result, float)
        assert abs(result - 0.35) < 1e-10

    def test_median(self):
        result = sentiment_aggregate([0.5, 0.3, -0.1, 0.7], method="median")
        assert isinstance(result, float)
        assert abs(result - 0.4) < 1e-10

    def test_empty_scores(self):
        result = sentiment_aggregate([])
        assert result == 0.0

    def test_single_score(self):
        result = sentiment_aggregate([0.8])
        assert result == 0.8

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            sentiment_aggregate([0.5], method="max")


# ---------------------------------------------------------------------------
# news_signal
# ---------------------------------------------------------------------------


class TestNewsSignal:
    def test_threshold_classification(self):
        sent = pd.Series([0.8, 0.3, -0.6, 0.1, -0.9])
        signals = news_signal(sent, threshold=0.5)
        assert isinstance(signals, pd.Series)
        expected = [1, 0, -1, 0, -1]
        np.testing.assert_array_equal(signals.values, expected)

    def test_default_threshold(self):
        sent = pd.Series([0.6, -0.6, 0.0])
        signals = news_signal(sent, threshold=0.5)
        expected = [1, -1, 0]
        np.testing.assert_array_equal(signals.values, expected)

    def test_list_input(self):
        signals = news_signal([0.8, -0.8, 0.0], threshold=0.5)
        assert isinstance(signals, pd.Series)
        expected = [1, -1, 0]
        np.testing.assert_array_equal(signals.values, expected)

    def test_all_neutral(self):
        sent = pd.Series([0.1, -0.1, 0.0, 0.2])
        signals = news_signal(sent, threshold=0.5)
        np.testing.assert_array_equal(signals.values, [0, 0, 0, 0])

    def test_preserves_index(self):
        idx = pd.date_range("2023-01-01", periods=3)
        sent = pd.Series([0.8, 0.0, -0.8], index=idx)
        signals = news_signal(sent, threshold=0.5)
        pd.testing.assert_index_equal(signals.index, idx)
