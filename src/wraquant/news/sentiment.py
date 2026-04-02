"""Sentiment analysis and news-based signal generation.

Provides functions for scoring news sentiment, building sentiment time series,
and generating discrete trading signals from textual data.  Uses FMP as the
primary news data source and includes a built-in keyword-based sentiment
scorer that requires no NLP dependencies.  When ``textblob`` or ``vaderSentiment``
are installed, those engines can be used for higher-quality scoring.

The sentiment pipeline is:

1. **Fetch** -- Pull headlines/articles via ``FMPClient``.
2. **Score** -- Assign each headline a score in [-1, +1].
3. **Aggregate** -- Combine scores with recency weighting.
4. **Signal** -- Convert the aggregate into a discrete trading signal.

References:
    - Tetlock (2007), "Giving Content to Investor Sentiment"
    - Loughran & McDonald (2011), "When Is a Liability Not a Liability?"
    - Hutto & Gilbert (2014), "VADER: A Parsimonious Rule-based Model for
      Sentiment Analysis of Social Media Text"
"""

from __future__ import annotations

import math
import re
from typing import Any, Sequence

import numpy as np
import pandas as pd

from wraquant._lazy import is_available
from wraquant.core.decorators import requires_extra

# ---------------------------------------------------------------------------
# Built-in keyword lexicon (Loughran-McDonald inspired)
# ---------------------------------------------------------------------------

_POSITIVE_WORDS: frozenset[str] = frozenset(
    {
        "beat",
        "beats",
        "exceeds",
        "exceeded",
        "surge",
        "surges",
        "surged",
        "rally",
        "rallies",
        "rallied",
        "gain",
        "gains",
        "gained",
        "profit",
        "profitable",
        "profitability",
        "upgrade",
        "upgrades",
        "upgraded",
        "outperform",
        "outperforms",
        "outperformed",
        "bullish",
        "optimistic",
        "strong",
        "stronger",
        "strongest",
        "record",
        "high",
        "higher",
        "highest",
        "growth",
        "growing",
        "grew",
        "expand",
        "expands",
        "expanded",
        "expansion",
        "positive",
        "upside",
        "breakout",
        "breakthrough",
        "boom",
        "booming",
        "recover",
        "recovers",
        "recovered",
        "recovery",
        "rebound",
        "rebounds",
        "robust",
        "solid",
        "impressive",
        "innovation",
        "innovative",
        "opportunity",
        "opportunities",
        "favorable",
        "success",
        "successful",
        "dividend",
        "buyback",
        "repurchase",
        "win",
        "wins",
        "won",
        "approval",
        "approved",
        "approves",
        "launch",
        "launches",
        "launched",
        "partnership",
        "collaboration",
        "acquisition",
        "momentum",
        "accelerate",
        "accelerated",
        "accelerating",
        "soar",
        "soars",
        "soared",
        "boost",
        "boosts",
        "boosted",
        "exceed",
        "top",
        "tops",
        "topped",
    }
)

_NEGATIVE_WORDS: frozenset[str] = frozenset(
    {
        "miss",
        "misses",
        "missed",
        "decline",
        "declines",
        "declined",
        "drop",
        "drops",
        "dropped",
        "fall",
        "falls",
        "fell",
        "loss",
        "losses",
        "losing",
        "downgrade",
        "downgrades",
        "downgraded",
        "underperform",
        "underperforms",
        "underperformed",
        "bearish",
        "pessimistic",
        "weak",
        "weaker",
        "weakest",
        "low",
        "lower",
        "lowest",
        "risk",
        "risks",
        "risky",
        "negative",
        "downside",
        "crash",
        "crashes",
        "crashed",
        "selloff",
        "sell-off",
        "recession",
        "recessionary",
        "contraction",
        "shrink",
        "shrinks",
        "shrunk",
        "bankruptcy",
        "bankrupt",
        "default",
        "defaults",
        "defaulted",
        "fraud",
        "fraudulent",
        "scandal",
        "investigation",
        "lawsuit",
        "litigation",
        "fine",
        "fined",
        "penalty",
        "warning",
        "warns",
        "warned",
        "cut",
        "cuts",
        "layoff",
        "layoffs",
        "restructuring",
        "impairment",
        "writedown",
        "write-down",
        "volatility",
        "volatile",
        "uncertainty",
        "uncertain",
        "concern",
        "concerns",
        "worried",
        "worry",
        "fear",
        "fears",
        "plunge",
        "plunges",
        "plunged",
        "slump",
        "slumps",
        "slumped",
        "tumble",
        "tumbles",
        "tumbled",
        "deficit",
        "debt",
        "overvalued",
        "bubble",
        "inflation",
        "inflationary",
        "tariff",
        "tariffs",
        "sanctions",
        "shutdown",
        "delay",
        "delays",
        "delayed",
        "disappointing",
        "disappointed",
        "disappoint",
    }
)

_NEGATION_WORDS: frozenset[str] = frozenset(
    {
        "not",
        "no",
        "never",
        "neither",
        "nor",
        "hardly",
        "barely",
        "scarcely",
        "doesn't",
        "don't",
        "didn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "won't",
        "wouldn't",
        "couldn't",
        "shouldn't",
    }
)

_INTENSIFIER_WORDS: frozenset[str] = frozenset(
    {
        "very",
        "extremely",
        "significantly",
        "substantially",
        "dramatically",
        "sharply",
        "strongly",
        "massively",
        "hugely",
        "remarkably",
    }
)

_WORD_RE = re.compile(r"[a-z'\-]+")


def _keyword_score(text: str) -> float:
    """Score a text using the built-in keyword lexicon.

    Uses Loughran-McDonald-inspired word lists with negation handling
    and intensity modifiers to produce a score in [-1, +1].

    Parameters:
        text: Raw text to score.

    Returns:
        Sentiment score in [-1.0, +1.0].
    """
    words = _WORD_RE.findall(text.lower())
    if not words:
        return 0.0

    score = 0.0
    negated = False
    intensified = False

    for word in words:
        if word in _NEGATION_WORDS:
            negated = True
            continue
        if word in _INTENSIFIER_WORDS:
            intensified = True
            continue

        base = 0.0
        if word in _POSITIVE_WORDS:
            base = 1.0
        elif word in _NEGATIVE_WORDS:
            base = -1.0

        if base != 0.0:
            if intensified:
                base *= 1.5
            if negated:
                base *= -0.75  # Negation partially flips, not full reversal
            score += base

        # Reset modifiers after consuming a sentiment word
        negated = False
        intensified = False

    # Normalize: divide by sqrt(word count) so longer texts don't dominate
    # but still benefit from repeated sentiment words
    normalized = score / math.sqrt(len(words))
    # Clamp to [-1, 1]
    return float(max(-1.0, min(1.0, normalized)))


def _vader_score(text: str) -> float:
    """Score text using VADER sentiment analyzer.

    Parameters:
        text: Raw text to score.

    Returns:
        VADER compound score in [-1.0, +1.0].
    """
    from vaderSentiment.vaderSentiment import (
        SentimentIntensityAnalyzer,  # type: ignore[import-untyped]
    )

    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return float(scores["compound"])


def _textblob_score(text: str) -> float:
    """Score text using TextBlob sentiment analysis.

    Parameters:
        text: Raw text to score.

    Returns:
        TextBlob polarity score in [-1.0, +1.0].
    """
    from textblob import TextBlob  # type: ignore[import-untyped]

    blob = TextBlob(text)
    return float(blob.sentiment.polarity)


def _get_scorer(engine: str = "auto") -> tuple[str, Any]:
    """Resolve the sentiment scoring engine.

    Parameters:
        engine: One of ``"auto"``, ``"keyword"``, ``"vader"``,
            ``"textblob"``.  ``"auto"`` tries VADER first, then TextBlob,
            then falls back to the built-in keyword scorer.

    Returns:
        Tuple of (engine_name, scorer_function).

    Raises:
        ValueError: If the requested engine is not recognized or not
            installed.
    """
    if engine == "keyword":
        return "keyword", _keyword_score
    if engine == "vader":
        if not is_available("vaderSentiment"):
            msg = (
                "vaderSentiment is not installed.  "
                "Install it with: pip install vaderSentiment"
            )
            raise ValueError(msg)
        return "vader", _vader_score
    if engine == "textblob":
        if not is_available("textblob"):
            msg = "textblob is not installed.  " "Install it with: pip install textblob"
            raise ValueError(msg)
        return "textblob", _textblob_score
    if engine == "auto":
        if is_available("vaderSentiment"):
            return "vader", _vader_score
        if is_available("textblob"):
            return "textblob", _textblob_score
        return "keyword", _keyword_score

    msg = f"Unknown sentiment engine: {engine!r}.  Use 'auto', 'keyword', 'vader', or 'textblob'."
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Recency weighting
# ---------------------------------------------------------------------------


def _recency_weights(dates: pd.Series, half_life_days: float = 7.0) -> np.ndarray:
    """Compute exponential decay weights based on recency.

    Parameters:
        dates: Series of datetime values.
        half_life_days: Half-life for the exponential decay in days.

    Returns:
        Array of weights in (0, 1], most recent = 1.0.
    """
    if dates.empty:
        return np.array([])

    dates_dt = pd.to_datetime(dates, utc=True)
    most_recent = dates_dt.max()
    days_ago = (most_recent - dates_dt).dt.total_seconds() / 86400.0
    decay = np.log(2) / half_life_days
    weights = np.exp(-decay * days_ago.values)
    return weights.astype(float)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def news_sentiment(
    symbol: str,
    limit: int = 50,
    *,
    engine: str = "auto",
    half_life_days: float = 7.0,
    include_press_releases: bool = True,
) -> dict[str, Any]:
    """Analyze sentiment of recent news for a stock.

    Fetches recent news headlines (and optionally press releases) from FMP,
    scores each headline using the specified sentiment engine, and computes
    aggregate statistics including recency-weighted sentiment and trend
    direction.

    The recency weighting uses exponential decay so that recent articles
    contribute more to the aggregate than older ones.  The trend is
    determined by comparing first-half vs. second-half sentiment to detect
    whether coverage is improving or deteriorating.

    Parameters:
        symbol: Ticker symbol (e.g., ``"AAPL"``).
        limit: Maximum number of news articles to fetch.  Higher values
            give a more robust sentiment estimate but include older news.
        engine: Sentiment scoring engine.  ``"auto"`` tries VADER, then
            TextBlob, then falls back to the built-in keyword scorer.
            Options: ``"auto"``, ``"keyword"``, ``"vader"``, ``"textblob"``.
        half_life_days: Half-life for recency weighting in days.  Default
            of 7 means a one-week-old article gets half the weight of
            today's article.
        include_press_releases: If True, also fetch press releases and
            include them in the analysis.

    Returns:
        Dictionary containing:
        - **symbol** (*str*) -- Ticker symbol.
        - **engine** (*str*) -- Sentiment engine used.
        - **article_count** (*int*) -- Total number of articles scored.
        - **articles** (*list[dict]*) -- List of dicts, each with keys
          ``title``, ``date``, ``source``, ``sentiment``, ``url``.
        - **aggregate** (*dict*) -- Aggregate statistics:
          - **mean** (*float*) -- Simple mean sentiment.
          - **weighted_mean** (*float*) -- Recency-weighted mean.
          - **median** (*float*) -- Median sentiment.
          - **std** (*float*) -- Standard deviation of scores.
          - **bullish_pct** (*float*) -- Fraction of positive articles.
          - **bearish_pct** (*float*) -- Fraction of negative articles.
          - **neutral_pct** (*float*) -- Fraction of neutral articles.
        - **trend** (*str*) -- ``"improving"``, ``"deteriorating"``, or
          ``"stable"`` based on first-half vs. second-half comparison.
        - **trend_delta** (*float*) -- Second-half mean minus first-half mean.
        - **news_volume** (*str*) -- ``"high"``, ``"medium"``, or ``"low"``
          based on article count relative to limit.

    Example:
        >>> from wraquant.news.sentiment import news_sentiment
        >>> result = news_sentiment("AAPL", limit=30)
        >>> print(f"Weighted sentiment: {result['aggregate']['weighted_mean']:.3f}")
        >>> print(f"Trend: {result['trend']}")

    Notes:
        Reference: Tetlock (2007). "Giving Content to Investor Sentiment."
        *The Journal of Finance*, 62(3), 1139-1168.

    See Also:
        sentiment_timeseries: Build a daily time series of sentiment.
        sentiment_signal: Convert sentiment to a trading signal.
    """
    from wraquant.data.providers.fmp import FMPClient

    client = FMPClient()
    engine_name, scorer = _get_scorer(engine)

    # Fetch news data
    news_df = client.stock_news(symbol, limit=limit)
    frames = [news_df]

    if include_press_releases:
        try:
            pr_df = client.press_releases(symbol, limit=max(10, limit // 3))
            frames.append(pr_df)
        except Exception:  # noqa: BLE001
            pass  # Press releases may not be available for all symbols

    combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else news_df

    if combined.empty:
        return {
            "symbol": symbol,
            "engine": engine_name,
            "article_count": 0,
            "articles": [],
            "aggregate": {
                "mean": 0.0,
                "weighted_mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "bullish_pct": 0.0,
                "bearish_pct": 0.0,
                "neutral_pct": 0.0,
            },
            "trend": "stable",
            "trend_delta": 0.0,
            "news_volume": "low",
        }

    # Identify column names (FMP may vary)
    title_col = _resolve_col(combined, ["title", "headline", "text"])
    date_col = _resolve_col(combined, ["publishedDate", "date", "published_date"])
    source_col = _resolve_col(combined, ["site", "source", "publisher"])
    url_col = _resolve_col(combined, ["url", "link"])

    # Score each article
    titles = combined[title_col].fillna("").astype(str)
    scores = np.array([scorer(t) for t in titles], dtype=float)

    # Build articles list
    articles: list[dict[str, Any]] = []
    for i in range(len(combined)):
        article: dict[str, Any] = {
            "title": titles.iloc[i],
            "sentiment": float(scores[i]),
        }
        if date_col:
            article["date"] = str(combined[date_col].iloc[i])
        if source_col:
            article["source"] = str(combined[source_col].iloc[i])
        if url_col:
            article["url"] = str(combined[url_col].iloc[i])
        articles.append(article)

    # Aggregate statistics
    mean_score = float(np.mean(scores))
    median_score = float(np.median(scores))
    std_score = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0

    # Recency-weighted mean
    if date_col:
        weights = _recency_weights(combined[date_col], half_life_days)
        if len(weights) > 0 and weights.sum() > 0:
            weighted_mean = float(np.average(scores, weights=weights))
        else:
            weighted_mean = mean_score
    else:
        weighted_mean = mean_score

    bullish_pct = float(np.mean(scores > 0.05))
    bearish_pct = float(np.mean(scores < -0.05))
    neutral_pct = 1.0 - bullish_pct - bearish_pct

    # Trend detection: split into halves chronologically
    n = len(scores)
    mid = n // 2
    if mid > 0 and n - mid > 0:
        first_half_mean = float(np.mean(scores[:mid]))
        second_half_mean = float(np.mean(scores[mid:]))
        trend_delta = second_half_mean - first_half_mean
        if trend_delta > 0.1:
            trend = "improving"
        elif trend_delta < -0.1:
            trend = "deteriorating"
        else:
            trend = "stable"
    else:
        trend = "stable"
        trend_delta = 0.0

    # News volume assessment
    if len(combined) >= limit * 0.8:
        news_volume = "high"
    elif len(combined) >= limit * 0.3:
        news_volume = "medium"
    else:
        news_volume = "low"

    return {
        "symbol": symbol,
        "engine": engine_name,
        "article_count": len(combined),
        "articles": articles,
        "aggregate": {
            "mean": mean_score,
            "weighted_mean": weighted_mean,
            "median": median_score,
            "std": std_score,
            "bullish_pct": bullish_pct,
            "bearish_pct": bearish_pct,
            "neutral_pct": neutral_pct,
        },
        "trend": trend,
        "trend_delta": trend_delta,
        "news_volume": news_volume,
    }


@requires_extra("market-data")
def sentiment_timeseries(
    symbol: str,
    days: int = 90,
    *,
    engine: str = "auto",
    resample: str = "D",
) -> pd.Series:
    """Build a daily (or custom frequency) sentiment time series.

    Fetches up to ``days`` worth of news for a symbol, scores each article,
    and resamples into a regular time series by averaging sentiment within
    each period.  Missing days are forward-filled so the series can be
    used directly alongside price data.

    Parameters:
        symbol: Ticker symbol (e.g., ``"MSFT"``).
        days: Number of calendar days of history to request.  FMP may
            return fewer articles than this span covers.
        engine: Sentiment scoring engine (see ``news_sentiment``).
        resample: Pandas resample frequency string.  ``"D"`` for daily,
            ``"W"`` for weekly, ``"B"`` for business days.

    Returns:
        pd.Series with a DatetimeIndex and sentiment scores averaged
        per period.  Index name is ``"date"``, series name is
        ``"sentiment"``.

    Example:
        >>> from wraquant.news.sentiment import sentiment_timeseries
        >>> ts = sentiment_timeseries("TSLA", days=30)
        >>> print(ts.tail())

    See Also:
        news_sentiment: Detailed sentiment analysis for a single snapshot.
        sentiment_signal: Convert the time series to a signal.
    """
    from wraquant.data.providers.fmp import FMPClient

    client = FMPClient()
    _, scorer = _get_scorer(engine)

    # Estimate limit: assume ~2 articles/day on average
    estimated_limit = max(50, days * 3)
    news_df = client.stock_news(symbol, limit=estimated_limit)

    if news_df.empty:
        idx = pd.date_range(
            end=pd.Timestamp.now(tz="UTC").normalize(),
            periods=days,
            freq="D",
        )
        return pd.Series(0.0, index=idx, name="sentiment")

    date_col = _resolve_col(news_df, ["publishedDate", "date", "published_date"])
    title_col = _resolve_col(news_df, ["title", "headline", "text"])

    if not date_col or not title_col:
        msg = "News DataFrame missing required date or title columns."
        raise ValueError(msg)

    df = news_df[[date_col, title_col]].copy()
    df["date"] = pd.to_datetime(df[date_col], utc=True)
    df["sentiment"] = df[title_col].fillna("").astype(str).apply(scorer)

    # Filter to requested date range
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    df = df.loc[df["date"] >= cutoff]

    if df.empty:
        idx = pd.date_range(
            end=pd.Timestamp.now(tz="UTC").normalize(),
            periods=days,
            freq="D",
        )
        return pd.Series(0.0, index=idx, name="sentiment")

    # Resample and fill
    df = df.set_index("date")
    resampled = df["sentiment"].resample(resample).mean()

    # Forward-fill gaps, then back-fill any leading NaNs
    resampled = resampled.ffill().bfill()
    resampled.index.name = "date"
    resampled.name = "sentiment"

    return resampled


@requires_extra("market-data")
def sentiment_signal(
    symbol: str,
    threshold: float = 0.3,
    *,
    engine: str = "auto",
    half_life_days: float = 7.0,
) -> str:
    """Generate a discrete sentiment-based trading signal for a stock.

    Fetches recent news, computes the recency-weighted aggregate sentiment,
    and classifies it as bullish, bearish, or neutral based on the
    threshold.

    The signal logic is:
    - ``weighted_mean > threshold``  =>  ``"bullish"``
    - ``weighted_mean < -threshold`` =>  ``"bearish"``
    - Otherwise                      =>  ``"neutral"``

    Parameters:
        symbol: Ticker symbol (e.g., ``"GOOG"``).
        threshold: Absolute threshold for signal classification.
            Lower values produce more signals (more sensitive);
            higher values filter out weak sentiment.  Default of 0.3
            is moderately conservative.
        engine: Sentiment scoring engine (see ``news_sentiment``).
        half_life_days: Half-life for recency weighting (see
            ``news_sentiment``).

    Returns:
        One of ``"bullish"``, ``"bearish"``, or ``"neutral"``.

    Example:
        >>> from wraquant.news.sentiment import sentiment_signal
        >>> signal = sentiment_signal("NVDA", threshold=0.2)
        >>> print(f"Sentiment signal: {signal}")

    See Also:
        news_sentiment: Full sentiment analysis with article-level detail.
        sentiment_timeseries: Historical sentiment time series.
    """
    result = news_sentiment(
        symbol,
        limit=50,
        engine=engine,
        half_life_days=half_life_days,
    )

    weighted_mean = result["aggregate"]["weighted_mean"]

    if weighted_mean > threshold:
        return "bullish"
    if weighted_mean < -threshold:
        return "bearish"
    return "neutral"


# ---------------------------------------------------------------------------
# Legacy API (kept for backward compatibility)
# ---------------------------------------------------------------------------


def sentiment_score(
    texts: str | Sequence[str],
    *,
    engine: str = "auto",
) -> dict[str, Any]:
    """Score text passages on a numeric sentiment scale.

    Scores arbitrary text using the specified sentiment engine.  This is
    the low-level scoring function; for news-specific analysis with
    data fetching and aggregation, use ``news_sentiment`` instead.

    Parameters:
        texts: A single text string or a sequence of text strings to
            score.
        engine: Sentiment engine.  ``"auto"`` tries VADER, then TextBlob,
            then the built-in keyword scorer.  Options: ``"auto"``,
            ``"keyword"``, ``"vader"``, ``"textblob"``.

    Returns:
        Dictionary containing:
        - **scores** (*list[float]*) -- Sentiment score for each text,
          in the range [-1.0, 1.0].
        - **mean_score** (*float*) -- Mean sentiment across all texts.
        - **engine** (*str*) -- Name of the engine used.

    Example:
        >>> result = sentiment_score("Stock rallied on strong earnings")
        >>> print(f"Score: {result['scores'][0]:.3f}")
        >>> print(f"Engine: {result['engine']}")

    See Also:
        news_sentiment: Full news sentiment pipeline with data fetching.
        sentiment_signal: Discrete signal from sentiment.
    """
    if isinstance(texts, str):
        texts = [texts]

    engine_name, scorer = _get_scorer(engine)
    scores = [scorer(t) for t in texts]
    mean_score = float(np.mean(scores)) if scores else 0.0

    return {
        "scores": scores,
        "mean_score": mean_score,
        "engine": engine_name,
    }


def news_impact(
    returns: pd.Series,
    event_dates: list | pd.DatetimeIndex,
    window: int = 5,
) -> dict[str, Any]:
    """Measure the impact of news events on returns using event study.

    Delegates to ``wraquant.causal.treatment.event_study`` to compute
    cumulative abnormal returns (CARs) around each event date.

    When to use:
        Use news impact analysis to quantify whether specific news
        events (earnings releases, FDA approvals, geopolitical shocks)
        have a statistically significant effect on returns.

    Parameters:
        returns: Return series with a DatetimeIndex.
        event_dates: List of event dates to study.
        window: Number of periods before and after each event to
            include in the analysis window.

    Returns:
        Dictionary containing:
        - **car** (*float*) -- Mean cumulative abnormal return across
          all events.
        - **event_results** -- Detailed event study output from
          ``wraquant.causal.treatment.event_study``.

    Example:
        >>> import pandas as pd, numpy as np
        >>> rng = np.random.default_rng(42)
        >>> dates = pd.bdate_range("2023-01-01", periods=252)
        >>> rets = pd.Series(rng.normal(0.0005, 0.01, 252), index=dates)
        >>> events = [dates[50], dates[150]]
        >>> result = news_impact(rets, events, window=5)

    See Also:
        wraquant.causal.treatment.event_study: Underlying event study.
        earnings_surprise: Earnings-specific impact metric.
    """
    from wraquant.causal.treatment import event_study

    result = event_study(returns, event_dates, window=window)

    if hasattr(result, "effect") and result.effect is not None:
        car = float(result.effect)
    else:
        car = 0.0

    return {
        "car": car,
        "event_results": result,
    }


def earnings_surprise(
    actual: float,
    estimate: float,
) -> float:
    """Compute the standardized earnings surprise.

    Earnings surprise is one of the most widely used signals in
    fundamental-driven quant strategies.  A positive surprise (actual
    exceeds estimate) typically triggers positive abnormal returns in
    the short term (post-earnings announcement drift, or PEAD).

    Mathematical formulation:
        surprise = (actual - estimate) / |estimate|

    When to use:
        Use earnings surprise as an input to event-driven strategies.
        Combine with ``news_impact`` to quantify the return effect.

    Parameters:
        actual: Actual reported earnings per share.
        estimate: Consensus analyst estimate of earnings per share.

    Returns:
        Standardized earnings surprise as a float.  Positive values
        indicate a beat; negative values indicate a miss.

    Example:
        >>> earnings_surprise(actual=2.50, estimate=2.30)
        0.08695652173913043
        >>> earnings_surprise(actual=1.80, estimate=2.00)
        -0.1

    See Also:
        news_impact: Measure the return impact of events.
        sentiment_score: Score textual sentiment.
    """
    if abs(estimate) < 1e-12:
        return 0.0
    return float((actual - estimate) / abs(estimate))


def sentiment_aggregate(
    scores: Sequence[float],
    method: str = "mean",
) -> float:
    """Aggregate multiple sentiment scores into a single composite.

    When to use:
        Use after collecting sentiment scores from multiple sources
        (multiple news articles, analyst reports, social media posts)
        to produce a single consensus sentiment for a given asset or
        time period.

    Parameters:
        scores: Sequence of sentiment scores (each in [-1, 1]).
        method: Aggregation method.  ``"mean"`` (default) computes the
            arithmetic mean.  ``"median"`` computes the median.

    Returns:
        Aggregated sentiment score as a float.

    Raises:
        ValueError: If *method* is not ``"mean"`` or ``"median"``.

    Example:
        >>> sentiment_aggregate([0.5, 0.3, -0.1, 0.7])
        0.35
        >>> sentiment_aggregate([0.5, 0.3, -0.1, 0.7], method="median")
        0.4

    See Also:
        sentiment_score: Generate individual scores.
        news_sentiment: Full sentiment pipeline.
    """
    arr = np.asarray(scores, dtype=float)
    if len(arr) == 0:
        return 0.0

    if method == "mean":
        return float(np.mean(arr))
    if method == "median":
        return float(np.median(arr))

    msg = f"Unknown aggregation method: {method!r}. Use 'mean' or 'median'."
    raise ValueError(msg)


def news_signal(
    sentiment_series: pd.Series | Sequence[float],
    threshold: float = 0.5,
) -> pd.Series:
    """Convert a continuous sentiment series into discrete trading signals.

    Applies threshold-based classification to convert continuous
    sentiment scores into actionable trading signals: +1 (bullish),
    -1 (bearish), or 0 (neutral).

    When to use:
        Use as the final step in a sentiment pipeline, after scoring
        and aggregation, to generate position signals for a trading
        strategy.

    Parameters:
        sentiment_series: Series or sequence of sentiment scores.
        threshold: Absolute threshold for signal generation.  Scores
            above ``+threshold`` produce +1; below ``-threshold``
            produce -1; values in between produce 0.

    Returns:
        pd.Series of integer signals (-1, 0, or +1).

    Example:
        >>> import pandas as pd
        >>> sent = pd.Series([0.8, 0.3, -0.6, 0.1, -0.9])
        >>> news_signal(sent, threshold=0.5)
        0    1
        1    0
        2   -1
        3    0
        4   -1
        dtype: int64

    See Also:
        sentiment_score: Generate sentiment scores.
        sentiment_aggregate: Combine multiple scores.
    """
    if isinstance(sentiment_series, pd.Series):
        arr = sentiment_series.values.astype(float)
        index = sentiment_series.index
    else:
        arr = np.asarray(sentiment_series, dtype=float)
        index = range(len(arr))

    signals = np.where(arr > threshold, 1, np.where(arr < -threshold, -1, 0))

    return pd.Series(signals.astype(int), index=index)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_col(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """Find the first matching column name from a list of candidates.

    Parameters:
        df: DataFrame to search.
        candidates: Ordered list of possible column names.

    Returns:
        The first matching column name, or None if none match.
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None
