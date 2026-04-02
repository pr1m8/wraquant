"""News, sentiment, and event-driven analysis MCP tools.

Tools: stock_news, news_sentiment, earnings_data, earnings_surprises,
insider_activity, sec_filings, dividend_history, sentiment_signal.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_news_tools(mcp, ctx: AnalysisContext) -> None:
    """Register news and event-driven tools on the MCP server."""

    @mcp.tool()
    def stock_news(
        symbol: str,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Fetch recent news articles for a stock from FMP.

        Returns headlines, dates, sources, and URLs. Use with
        news_sentiment for sentiment analysis.

        Parameters:
            symbol: Stock ticker.
            limit: Number of articles to return (default 20).
        """
        try:
            from wraquant.data.providers.fmp import FMPClient

            client = FMPClient()
            news_df = client.stock_news(symbol, limit=limit)

            stored = ctx.store_dataset(
                f"news_{symbol.lower()}",
                news_df,
                source_op="stock_news",
            )

            headlines = []
            for _, row in news_df.head(10).iterrows():
                headlines.append(
                    {
                        "title": str(row.get("title", "")),
                        "date": str(row.get("publishedDate", row.get("date", ""))),
                        "source": str(row.get("site", row.get("source", ""))),
                    }
                )

            return _sanitize_for_json(
                {
                    "tool": "stock_news",
                    "symbol": symbol,
                    "total_articles": len(news_df),
                    "recent_headlines": headlines,
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "stock_news"}

    @mcp.tool()
    def news_sentiment(
        symbol: str,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Analyze sentiment of recent news for a stock.

        Computes aggregate sentiment score (-1 to +1), sentiment trend,
        and categorizes as bullish/bearish/neutral.

        Parameters:
            symbol: Stock ticker.
            limit: Number of articles to analyze.
        """
        try:
            from wraquant.news.sentiment import news_sentiment as _sentiment

            result = _sentiment(symbol, limit=limit)

            return _sanitize_for_json(
                {
                    "tool": "news_sentiment",
                    "symbol": symbol,
                    **result,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "news_sentiment"}

    @mcp.tool()
    def earnings_data(
        symbol: str,
    ) -> dict[str, Any]:
        """Get earnings history and upcoming earnings for a stock.

        Returns historical EPS (actual vs estimate), beat/miss history,
        and next earnings date.

        Parameters:
            symbol: Stock ticker.
        """
        try:
            from wraquant.news.events import earnings_history, upcoming_earnings

            history = earnings_history(symbol)
            upcoming = upcoming_earnings(symbol)

            return _sanitize_for_json(
                {
                    "tool": "earnings_data",
                    "symbol": symbol,
                    "upcoming": upcoming,
                    "history": history,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "earnings_data"}

    @mcp.tool()
    def earnings_surprises(
        symbol: str,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Get earnings surprise data: actual vs estimate EPS.

        Shows how often the company beats/misses estimates and by how much.

        Parameters:
            symbol: Stock ticker.
            limit: Number of quarters.
        """
        try:
            from wraquant.data.providers.fmp import FMPClient

            client = FMPClient()
            surprises_df = client.earnings_surprises(symbol)

            if len(surprises_df) > limit:
                surprises_df = surprises_df.head(limit)

            beats = 0
            misses = 0
            for _, row in surprises_df.iterrows():
                actual = row.get("actualEarningResult", row.get("actual", 0))
                estimate = row.get("estimatedEarning", row.get("estimate", 0))
                if actual and estimate:
                    if float(actual) > float(estimate):
                        beats += 1
                    else:
                        misses += 1

            return _sanitize_for_json(
                {
                    "tool": "earnings_surprises",
                    "symbol": symbol,
                    "total_quarters": len(surprises_df),
                    "beats": beats,
                    "misses": misses,
                    "beat_rate": beats / max(beats + misses, 1),
                    "recent": surprises_df.head(5).to_dict(orient="records"),
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "earnings_surprises"}

    @mcp.tool()
    def insider_activity(
        symbol: str,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Analyze insider trading activity for a stock.

        Returns recent insider buys/sells, buy/sell ratio, and
        notable large transactions.

        Parameters:
            symbol: Stock ticker.
            limit: Number of transactions to analyze.
        """
        try:
            from wraquant.news.events import insider_activity as _insider

            result = _insider(symbol, limit=limit)

            return _sanitize_for_json(
                {
                    "tool": "insider_activity",
                    "symbol": symbol,
                    **result,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "insider_activity"}

    @mcp.tool()
    def sec_filings(
        symbol: str,
        form_type: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Get recent SEC filings for a stock.

        Returns 10-K, 10-Q, 8-K and other filings with dates and links.

        Parameters:
            symbol: Stock ticker.
            form_type: Filter by type ('10-K', '10-Q', '8-K'). None = all.
            limit: Number of filings.
        """
        try:
            from wraquant.news.filings import recent_filings

            filings_df = recent_filings(symbol, form_type=form_type, limit=limit)

            stored = ctx.store_dataset(
                f"filings_{symbol.lower()}",
                filings_df,
                source_op="sec_filings",
            )

            return _sanitize_for_json(
                {
                    "tool": "sec_filings",
                    "symbol": symbol,
                    "form_type": form_type,
                    "total_filings": len(filings_df),
                    "recent": filings_df.head(10).to_dict(orient="records"),
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "sec_filings"}

    @mcp.tool()
    def dividend_history(
        symbol: str,
    ) -> dict[str, Any]:
        """Get dividend history: yield, growth, payout ratio over time.

        Parameters:
            symbol: Stock ticker.
        """
        try:
            from wraquant.news.events import dividend_history as _div

            result = _div(symbol)

            return _sanitize_for_json(
                {
                    "tool": "dividend_history",
                    "symbol": symbol,
                    **(
                        result
                        if isinstance(result, dict)
                        else {
                            "data": (
                                result.to_dict(orient="records")
                                if isinstance(result, pd.DataFrame)
                                else result
                            )
                        }
                    ),
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "dividend_history"}

    @mcp.tool()
    def sentiment_signal(
        symbol: str,
        threshold: float = 0.3,
    ) -> dict[str, Any]:
        """Generate a trading signal from news sentiment.

        Returns 'bullish', 'bearish', or 'neutral' based on aggregate
        sentiment exceeding the threshold.

        Parameters:
            symbol: Stock ticker.
            threshold: Sentiment threshold for signal (default 0.3).
        """
        try:
            from wraquant.news.sentiment import sentiment_signal as _signal

            signal = _signal(symbol, threshold=threshold)

            return _sanitize_for_json(
                {
                    "tool": "sentiment_signal",
                    "symbol": symbol,
                    "signal": signal,
                    "threshold": threshold,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "sentiment_signal"}
