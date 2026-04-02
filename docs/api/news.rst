News & Sentiment (``wraquant.news``)
=====================================

The news module provides 17 functions for news sentiment analysis,
corporate event monitoring, and SEC filings retrieval -- the data
layer for event-driven and sentiment-based quant strategies.

**Three areas of coverage:**

1. **Sentiment analysis** (``sentiment`` submodule) -- Score news headlines
   using a built-in Loughran-McDonald-inspired keyword lexicon, or
   optionally VADER/TextBlob. Build sentiment time series with recency
   weighting, and generate discrete trading signals from aggregate
   sentiment.

2. **Corporate events** (``events`` submodule) -- Earnings calendar and
   surprise analysis (PEAD), dividend history and payout trends, insider
   transaction activity (net buy/sell ratios), and institutional ownership
   tracking with quarterly changes.

3. **SEC filings** (``filings`` submodule) -- Search and retrieve 10-K,
   10-Q, 8-K, and other filing types. Convenience functions for annual
   reports, quarterly reports, and material events.

All functions use the FMP (Financial Modeling Prep) API for data. Requires
the ``market-data`` extra and an ``FMP_API_KEY`` environment variable.


Quick Example
-------------

.. code-block:: python

   from wraquant.news import news_sentiment, earnings_history, recent_filings

   # Aggregate sentiment for a ticker
   sentiment = news_sentiment("AAPL")
   print(f"Sentiment: {sentiment['aggregate_sentiment']:.2f}")
   print(f"Articles analyzed: {sentiment['n_articles']}")

   # Earnings history with surprise analysis
   earnings = earnings_history("AAPL")
   print(earnings[["date", "eps_actual", "eps_estimate", "surprise_pct"]].head())

   # Recent SEC filings
   filings = recent_filings("AAPL", limit=5)
   print(filings[["date", "type", "title"]])

Sentiment Pipeline
^^^^^^^^^^^^^^^^^^^

The sentiment module implements a four-step pipeline:

1. **Fetch** -- Pull headlines/articles via ``FMPClient``
2. **Score** -- Assign each headline a score in [-1, +1]
3. **Aggregate** -- Combine scores with recency weighting
4. **Signal** -- Convert the aggregate into a discrete trading signal

.. code-block:: python

   from wraquant.news import (
       sentiment_score,
       sentiment_timeseries,
       sentiment_signal,
       news_impact,
   )

   # Score individual headlines
   score = sentiment_score("Apple beats earnings expectations, raises guidance")
   print(f"Score: {score:.2f}")  # Positive

   # Sentiment time series (daily aggregate scores)
   ts = sentiment_timeseries("AAPL", days=90)
   print(ts.tail())  # pd.Series with date index

   # Generate a trading signal from sentiment
   signal = sentiment_signal("AAPL", threshold=0.3)
   print(f"Signal: {signal['signal']}")  # 1 (bullish), 0, or -1 (bearish)

   # Measure news impact on price
   impact = news_impact("AAPL", event_date="2025-01-30")
   print(f"1-day return: {impact['return_1d']:.2%}")

Event-Driven Analysis
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from wraquant.news import (
       earnings_calendar,
       earnings_surprises,
       upcoming_earnings,
       dividend_history,
       insider_activity,
       institutional_ownership,
   )

   # Earnings calendar for a date range
   calendar = earnings_calendar(from_date="2025-04-01", to_date="2025-04-30")
   print(f"Earnings reports this month: {len(calendar)}")

   # Earnings surprises (beat/miss history)
   surprises = earnings_surprises("AAPL")
   avg_surprise = surprises["surprise_pct"].mean()
   print(f"Average surprise: {avg_surprise:.1%}")

   # Upcoming earnings for a ticker
   upcoming = upcoming_earnings("TSLA")

   # Dividend history
   dividends = dividend_history("JNJ")
   print(f"Current yield: {dividends['current_yield']:.2%}")

   # Insider activity
   insiders = insider_activity("AAPL")
   print(f"Net insider sentiment: {insiders['net_sentiment']}")

   # Institutional ownership
   holders = institutional_ownership("AAPL")
   print(holders[["holder", "shares", "pct_held"]].head(10))

SEC Filings
^^^^^^^^^^^^

.. code-block:: python

   from wraquant.news import (
       recent_filings,
       annual_reports,
       quarterly_reports,
       material_events,
       filing_search,
   )

   # Recent filings of any type
   filings = recent_filings("AAPL", limit=10)

   # Filter by filing type
   annuals = annual_reports("MSFT", limit=3)      # 10-K filings
   quarterlies = quarterly_reports("MSFT", limit=4)  # 10-Q filings
   events = material_events("MSFT", limit=5)      # 8-K filings

   # Search filings with keywords
   results = filing_search("AAPL", query="risk factors", filing_type="10-K")

.. seealso::

   - :doc:`fundamental` -- Financial ratios and valuation models
   - :doc:`/tutorials/risk_analysis` -- Using news signals in risk analysis
   - :doc:`ml` -- News sentiment as ML features
   - :doc:`data` -- Data fetching and cleaning


API Reference
-------------

.. automodule:: wraquant.news
   :members:
   :undoc-members:
   :show-inheritance:

Sentiment
^^^^^^^^^

.. automodule:: wraquant.news.sentiment
   :members:

Events
^^^^^^

.. automodule:: wraquant.news.events
   :members:

Filings
^^^^^^^

.. automodule:: wraquant.news.filings
   :members:
