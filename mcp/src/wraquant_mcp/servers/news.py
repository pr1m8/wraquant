"""News and sentiment analysis MCP tools.

Tools: sentiment_score, news_signal.
"""

from __future__ import annotations

from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_news_tools(mcp, ctx: AnalysisContext) -> None:
    """Register news and sentiment tools on the MCP server."""

    @mcp.tool()
    def sentiment_score(
        texts: list[str],
    ) -> dict[str, Any]:
        """Score text passages on a numeric sentiment scale.

        Returns a sentiment score in [-1, 1] for each input text.
        Currently uses a placeholder model; production use should
        replace with FinBERT or a custom transformer.

        Parameters:
            texts: List of text strings to score.
        """
        from wraquant.news.sentiment import sentiment_score as _score

        result = _score(texts)

        return _sanitize_for_json({
            "tool": "sentiment_score",
            "n_texts": len(texts),
            "scores": result["scores"],
            "mean_score": result["mean_score"],
        })

    @mcp.tool()
    def news_signal(
        dataset: str,
        sentiment_column: str = "sentiment",
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Convert sentiment scores into discrete trading signals.

        Applies threshold-based classification: scores above +threshold
        become +1 (bullish), below -threshold become -1 (bearish),
        and values in between become 0 (neutral).

        Parameters:
            dataset: Dataset containing sentiment scores.
            sentiment_column: Column with sentiment values.
            threshold: Absolute threshold for signal generation.
        """
        import pandas as pd

        from wraquant.news.sentiment import news_signal as _signal

        df = ctx.get_dataset(dataset)
        sentiment = df[sentiment_column].dropna()

        signals = _signal(sentiment, threshold=threshold)

        signal_df = pd.DataFrame({"signal": signals})
        stored = ctx.store_dataset(
            f"signals_{dataset}", signal_df,
            source_op="news_signal", parent=dataset,
        )

        n_bullish = int((signals == 1).sum())
        n_bearish = int((signals == -1).sum())
        n_neutral = int((signals == 0).sum())

        return _sanitize_for_json({
            "tool": "news_signal",
            "dataset": dataset,
            "threshold": threshold,
            "n_bullish": n_bullish,
            "n_bearish": n_bearish,
            "n_neutral": n_neutral,
            "latest_signal": int(signals.iloc[-1]) if len(signals) > 0 else None,
            **stored,
        })
