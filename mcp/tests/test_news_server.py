"""Tests for news/sentiment MCP server tools.

Tests sentiment_score, earnings_surprise, sentiment_aggregate via
underlying wraquant.news.sentiment functions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Add mcp source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wraquant_mcp.context import _sanitize_for_json


class TestNewsServer:
    """Test news/sentiment MCP tool functions via underlying wraquant.news."""

    def test_sentiment_score(self):
        """sentiment_score returns scores for a list of texts."""
        from wraquant.news.sentiment import sentiment_score as _score

        texts = [
            "Stock rallied on strong earnings",
            "Revenue missed expectations",
            "Company announced new product launch",
        ]

        result = _score(texts)

        output = _sanitize_for_json({
            "tool": "sentiment_score",
            "n_texts": len(texts),
            "scores": result["scores"],
            "mean_score": result["mean_score"],
        })

        assert output["tool"] == "sentiment_score"
        assert output["n_texts"] == 3
        assert isinstance(output["scores"], list)
        assert len(output["scores"]) == 3
        assert isinstance(output["mean_score"], float)
        # Each score should be in [-1, 1]
        for score in output["scores"]:
            assert isinstance(score, float)
            assert -1.0 <= score <= 1.0

    def test_earnings_surprise(self):
        """earnings_surprise computes the standardized surprise percentage."""
        from wraquant.news.sentiment import earnings_surprise as _surprise

        actual = 2.50
        estimate = 2.30

        surprise = _surprise(actual, estimate)

        # Determine signal
        if surprise > 0.05:
            signal = "strong_beat"
        elif surprise > 0:
            signal = "beat"
        elif surprise > -0.05:
            signal = "miss"
        else:
            signal = "strong_miss"

        output = _sanitize_for_json({
            "tool": "earnings_surprise",
            "actual": actual,
            "estimate": estimate,
            "surprise": surprise,
            "signal": signal,
        })

        assert output["tool"] == "earnings_surprise"
        assert isinstance(output["surprise"], float)
        # Surprise = (2.50 - 2.30) / |2.30| = 0.2/2.3 ~= 0.0870
        assert output["surprise"] == pytest.approx(0.0870, rel=0.01)
        assert output["surprise"] > 0  # positive = beat
        assert output["signal"] == "strong_beat"  # > 0.05
        assert output["actual"] == 2.50
        assert output["estimate"] == 2.30

        # Test a miss scenario
        surprise_miss = _surprise(actual=1.80, estimate=2.00)
        assert surprise_miss < 0  # negative = miss
        assert surprise_miss == pytest.approx(-0.10)

    def test_sentiment_aggregate(self):
        """sentiment_aggregate averages multiple scores."""
        from wraquant.news.sentiment import sentiment_aggregate as _agg

        scores = [0.5, 0.3, -0.1, 0.7]

        result = _agg(scores, method="mean")

        arr = np.array(scores)

        output = _sanitize_for_json({
            "tool": "sentiment_aggregate",
            "n_scores": len(scores),
            "method": "mean",
            "aggregate_score": result,
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        })

        assert output["tool"] == "sentiment_aggregate"
        assert output["n_scores"] == 4
        assert output["method"] == "mean"
        assert isinstance(output["aggregate_score"], float)
        assert output["aggregate_score"] == pytest.approx(0.35)  # (0.5+0.3-0.1+0.7)/4
        assert isinstance(output["std"], float)
        assert output["std"] > 0
        assert isinstance(output["min"], float)
        assert output["min"] == pytest.approx(-0.1)
        assert isinstance(output["max"], float)
        assert output["max"] == pytest.approx(0.7)

        # Also test median method
        result_median = _agg(scores, method="median")
        assert isinstance(result_median, float)
        assert result_median == pytest.approx(0.4)  # median of [0.5, 0.3, -0.1, 0.7]
