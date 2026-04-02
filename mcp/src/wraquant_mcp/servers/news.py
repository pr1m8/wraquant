"""News and sentiment analysis MCP tools.

Tools: sentiment_score, news_impact, earnings_surprise,
sentiment_aggregate, news_signal.
"""

from __future__ import annotations

import json
from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_news_tools(mcp, ctx: AnalysisContext) -> None:
    """Register news and sentiment tools on the MCP server."""

    @mcp.tool()
    def sentiment_score(
        texts_json: str,
    ) -> dict[str, Any]:
        """Score a list of text passages on a numeric sentiment scale.

        Returns a sentiment score in [-1, 1] for each input text.
        Negative = bearish, positive = bullish, zero = neutral.

        Parameters:
            texts_json: JSON array of text strings to score
                (e.g., '["Stock rallied on strong earnings", "Revenue missed expectations"]').
        """
        try:
            from wraquant.news.sentiment import sentiment_score as _score

            texts = json.loads(texts_json)
            result = _score(texts)

            return _sanitize_for_json(
                {
                    "tool": "sentiment_score",
                    "n_texts": len(texts),
                    "scores": result["scores"],
                    "mean_score": result["mean_score"],
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "sentiment_score"}

    @mcp.tool()
    def news_impact(
        dataset: str,
        column: str,
        event_dates_json: str,
        window: int = 5,
    ) -> dict[str, Any]:
        """Measure the impact of news events on a return series.

        Uses an event-study framework to compute cumulative abnormal
        returns (CARs) around each event date.

        Parameters:
            dataset: Dataset containing the return series.
            column: Column name with returns.
            event_dates_json: JSON array of event date strings
                (e.g., '["2024-01-15", "2024-03-20"]').
            window: Number of periods before and after each event
                to include in the analysis window.
        """
        try:
            import pandas as pd

            from wraquant.news.sentiment import news_impact as _impact

            df = ctx.get_dataset(dataset)
            returns = df[column].dropna()

            event_dates = json.loads(event_dates_json)
            event_dates = pd.to_datetime(event_dates)

            result = _impact(returns, event_dates, window=window)

            return _sanitize_for_json(
                {
                    "tool": "news_impact",
                    "dataset": dataset,
                    "column": column,
                    "n_events": len(event_dates),
                    "window": window,
                    "car": result["car"],
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "news_impact"}

    @mcp.tool()
    def earnings_surprise(
        actual: float,
        estimate: float,
    ) -> dict[str, Any]:
        """Compute the standardized earnings surprise.

        Measures how much actual earnings deviated from the consensus
        estimate, normalized by the estimate magnitude.  Positive
        values indicate a beat; negative values indicate a miss.

        Parameters:
            actual: Actual reported earnings per share.
            estimate: Consensus analyst estimate of earnings per share.
        """
        try:
            from wraquant.news.sentiment import earnings_surprise as _surprise

            surprise = _surprise(actual, estimate)

            if surprise > 0.05:
                signal = "strong_beat"
            elif surprise > 0:
                signal = "beat"
            elif surprise > -0.05:
                signal = "miss"
            else:
                signal = "strong_miss"

            return _sanitize_for_json(
                {
                    "tool": "earnings_surprise",
                    "actual": actual,
                    "estimate": estimate,
                    "surprise": surprise,
                    "signal": signal,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "earnings_surprise"}

    @mcp.tool()
    def sentiment_aggregate(
        scores_json: str,
        method: str = "mean",
    ) -> dict[str, Any]:
        """Aggregate multiple sentiment scores into a single composite.

        Combines scores from multiple sources (articles, analyst reports,
        social media) into a consensus sentiment value.

        Parameters:
            scores_json: JSON array of sentiment scores
                (e.g., '[0.5, 0.3, -0.1, 0.7]').
            method: Aggregation method ('mean' or 'median').
        """
        try:
            import numpy as np

            from wraquant.news.sentiment import sentiment_aggregate as _agg

            scores = json.loads(scores_json)
            result = _agg(scores, method=method)

            arr = np.array(scores)

            return _sanitize_for_json(
                {
                    "tool": "sentiment_aggregate",
                    "n_scores": len(scores),
                    "method": method,
                    "aggregate_score": result,
                    "std": float(np.std(arr)) if len(arr) > 0 else 0.0,
                    "min": float(np.min(arr)) if len(arr) > 0 else 0.0,
                    "max": float(np.max(arr)) if len(arr) > 0 else 0.0,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "sentiment_aggregate"}

    @mcp.tool()
    def news_signal(
        dataset: str,
        sentiment_col: str,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Convert continuous sentiment scores into discrete trading signals.

        Applies threshold-based classification: scores above +threshold
        become +1 (bullish), below -threshold become -1 (bearish),
        and values in between become 0 (neutral).

        Parameters:
            dataset: Dataset containing sentiment scores.
            sentiment_col: Column with sentiment values.
            threshold: Absolute threshold for signal generation.
        """
        try:
            import pandas as pd

            from wraquant.news.sentiment import news_signal as _signal

            df = ctx.get_dataset(dataset)
            sentiment = df[sentiment_col].dropna()

            signals = _signal(sentiment, threshold=threshold)

            signal_df = pd.DataFrame({"signal": signals})
            stored = ctx.store_dataset(
                f"signals_{dataset}",
                signal_df,
                source_op="news_signal",
                parent=dataset,
            )

            n_bullish = int((signals == 1).sum())
            n_bearish = int((signals == -1).sum())
            n_neutral = int((signals == 0).sum())

            return _sanitize_for_json(
                {
                    "tool": "news_signal",
                    "dataset": dataset,
                    "sentiment_col": sentiment_col,
                    "threshold": threshold,
                    "n_bullish": n_bullish,
                    "n_bearish": n_bearish,
                    "n_neutral": n_neutral,
                    "total_observations": n_bullish + n_bearish + n_neutral,
                    "latest_signal": (
                        int(signals.iloc[-1]) if len(signals) > 0 else None
                    ),
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "news_signal"}
