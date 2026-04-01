"""Sentiment analysis and news-based signal generation.

Provides functions for scoring text sentiment, measuring news impact on
returns, computing earnings surprises, aggregating sentiment scores, and
converting sentiment into discrete trading signals.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


def sentiment_score(
    texts: str | Sequence[str],
) -> dict[str, Any]:
    """Score text passages on a numeric sentiment scale.

    This is a placeholder implementation that returns neutral scores
    (0.0) for all inputs.  In production, replace the scoring logic
    with a fine-tuned NLP model such as FinBERT, VADER, or a custom
    transformer trained on financial text.

    When to use:
        Use sentiment scoring as the first step in a news-driven alpha
        pipeline.  Feed the output into ``sentiment_aggregate`` to
        combine scores across multiple sources, then into
        ``news_signal`` to generate discrete trading signals.

    Parameters:
        texts: A single text string or a sequence of text strings to
            score.

    Returns:
        Dictionary containing:
        - **scores** (*list[float]*) -- Sentiment score for each text,
          in the range [-1.0, 1.0].  Negative = bearish, positive =
          bullish, zero = neutral.
        - **mean_score** (*float*) -- Mean sentiment across all texts.

    Example:
        >>> result = sentiment_score(["Stock rallied on strong earnings"])
        >>> result["scores"]
        [0.0]
        >>> result["mean_score"]
        0.0

    See Also:
        sentiment_aggregate: Combine multiple scores.
        news_signal: Convert scores to trading signals.
    """
    if isinstance(texts, str):
        texts = [texts]

    # Placeholder: return neutral for all texts
    scores = [0.0 for _ in texts]
    mean_score = float(np.mean(scores)) if scores else 0.0

    return {
        "scores": scores,
        "mean_score": mean_score,
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

    # Compute mean CAR from the event study result
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
        news_signal: Convert aggregate score to trading signal.
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
