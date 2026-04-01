"""News and sentiment analysis for quantitative finance.

Provides tools for incorporating news sentiment and event-driven signals
into quantitative strategies.  In modern quant finance, alternative data
sources -- especially news sentiment -- can provide alpha when combined
with traditional price/volume data.

This module covers five core capabilities:

1. **Sentiment scoring** (``sentiment_score``) -- Score text passages on
   a numeric sentiment scale.  Placeholder implementation returns neutral
   scores; plug in your preferred NLP model (FinBERT, VADER, etc.) for
   production use.

2. **News impact analysis** (``news_impact``) -- Measure the effect of
   news events on returns using an event-study framework (delegates to
   ``wraquant.causal.treatment.event_study``).

3. **Earnings surprise** (``earnings_surprise``) -- Compute standardized
   earnings surprises: ``(actual - estimate) / |estimate|``.  A core
   signal in fundamental-driven quant strategies.

4. **Sentiment aggregation** (``sentiment_aggregate``) -- Aggregate
   multiple sentiment scores into a single composite via mean, median,
   or weighted average.

5. **News signal generation** (``news_signal``) -- Convert a continuous
   sentiment series into a discrete trading signal: +1 (bullish),
   -1 (bearish), or 0 (neutral) based on threshold crossings.

Example:
    >>> from wraquant.news import earnings_surprise, news_signal
    >>> surprise = earnings_surprise(actual=2.50, estimate=2.30)
    >>> print(f"Earnings surprise: {surprise:.2%}")
    >>> import pandas as pd
    >>> sent = pd.Series([0.6, 0.3, -0.2, 0.8, 0.1])
    >>> signals = news_signal(sent, threshold=0.4)

References:
    - Tetlock (2007), "Giving Content to Investor Sentiment"
    - Loughran & McDonald (2011), "When Is a Liability Not a Liability?"
"""

from wraquant.news.sentiment import (
    earnings_surprise,
    news_impact,
    news_signal,
    sentiment_aggregate,
    sentiment_score,
)

__all__ = [
    "sentiment_score",
    "news_impact",
    "earnings_surprise",
    "sentiment_aggregate",
    "news_signal",
]
