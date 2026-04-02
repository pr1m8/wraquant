"""News, sentiment, and event-driven analysis.

FMP-backed module for news sentiment scoring, earnings analysis,
insider trading activity, dividend history, SEC filings, and
institutional ownership tracking.

Modules:
    sentiment: News sentiment scoring, timeseries, and signal generation
    events: Earnings, dividends, insider trades, institutional holders
    filings: SEC filings (10-K, 10-Q, 8-K) search and retrieval

Example:
    >>> from wraquant.news import news_sentiment, earnings_history
    >>> sentiment = news_sentiment("AAPL")
    >>> print(f"Sentiment: {sentiment['aggregate_sentiment']:.2f}")
    >>> earnings = earnings_history("AAPL")
"""

from __future__ import annotations

from wraquant.news.events import (
    dividend_history,
    earnings_calendar,
    earnings_history,
    earnings_surprises,
    insider_activity,
    institutional_ownership,
    upcoming_earnings,
)
from wraquant.news.filings import (
    annual_reports,
    filing_search,
    material_events,
    quarterly_reports,
    recent_filings,
)
from wraquant.news.sentiment import (
    news_impact,
    news_sentiment,
    sentiment_score,
    sentiment_signal,
    sentiment_timeseries,
)

__all__ = [
    # Sentiment
    "news_sentiment",
    "sentiment_timeseries",
    "sentiment_signal",
    "sentiment_score",
    "news_impact",
    # Events
    "earnings_calendar",
    "earnings_surprises",
    "upcoming_earnings",
    "earnings_history",
    "dividend_history",
    "insider_activity",
    "institutional_ownership",
    # Filings
    "recent_filings",
    "annual_reports",
    "quarterly_reports",
    "material_events",
    "filing_search",
]
