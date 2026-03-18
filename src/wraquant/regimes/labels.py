"""Regime labeling, classification, and duration analysis.

Provides rule-based and statistical approaches to labeling market
regimes without requiring a fitted model.  These functions are useful
for backtesting, for creating training labels for supervised regime
classifiers, and for generating interpretable regime descriptions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def label_regimes(states: pd.Series, returns: pd.Series) -> pd.Series:
    """Assign descriptive labels to numeric regime states.

    States are sorted by mean return: the state with the highest mean
    return is labeled ``"bull"``, the lowest ``"bear"``, and any
    intermediate states ``"neutral_1"``, ``"neutral_2"``, etc.

    Parameters:
        states: Integer regime state series.
        returns: Corresponding return series (same index).

    Returns:
        Series of string regime labels.
    """
    aligned_returns, aligned_states = returns.align(states, join="inner")
    unique_states = sorted(aligned_states.unique())

    if len(unique_states) <= 1:
        return pd.Series("neutral", index=aligned_states.index, name="regime_label")

    # Rank states by mean return
    mean_by_state = {
        s: float(aligned_returns[aligned_states == s].mean()) for s in unique_states
    }
    ranked = sorted(mean_by_state, key=lambda s: mean_by_state[s])

    label_map: dict[int, str] = {}
    label_map[ranked[0]] = "bear"
    label_map[ranked[-1]] = "bull"
    for i, s in enumerate(ranked[1:-1], start=1):
        label_map[s] = f"neutral_{i}"

    return aligned_states.map(label_map).rename("regime_label")


def regime_statistics(
    returns: pd.Series,
    states: pd.Series,
) -> pd.DataFrame:
    """Compute descriptive statistics for each regime.

    Parameters:
        returns: Return series.
        states: Integer regime state series (same index).

    Returns:
        DataFrame indexed by regime state with columns for mean, std,
        skew, count, and fraction of total observations.
    """
    aligned_returns, aligned_states = returns.align(states, join="inner")
    total = len(aligned_returns)

    records = []
    for state in sorted(aligned_states.unique()):
        mask = aligned_states == state
        regime_rets = aligned_returns[mask]
        records.append(
            {
                "state": state,
                "mean": float(regime_rets.mean()),
                "std": float(regime_rets.std()),
                "skew": float(regime_rets.skew()),
                "count": int(mask.sum()),
                "fraction": float(mask.sum() / total) if total > 0 else 0.0,
            }
        )

    return pd.DataFrame(records).set_index("state")


# ---------------------------------------------------------------------------
# Volatility regime labels
# ---------------------------------------------------------------------------


def volatility_regime_labels(
    returns: pd.Series | np.ndarray,
    *,
    window: int = 21,
    n_levels: int = 3,
    quantiles: list[float] | None = None,
) -> pd.Series:
    """Label regimes based on realised volatility quantiles.

    A simple, model-free approach that classifies each period by
    where its rolling volatility falls within the historical
    distribution.  No fitting, no hidden states -- just raw
    vol percentiles.

    **Interpretation guidance:**

    - ``"low_vol"`` periods typically correspond to trending or
      complacent markets.  Strategy-wise, favour momentum and
      carry.
    - ``"high_vol"`` periods correspond to stressed or mean-reverting
      markets.  Favour defensive positioning or mean-reversion.
    - ``"medium_vol"`` is the transition zone.

    Parameters:
        returns: Return series.
        window: Rolling window for realised volatility estimation.
            Default 21 (roughly one trading month).
        n_levels: Number of volatility levels.  Default 3 produces
            ``low_vol`` / ``medium_vol`` / ``high_vol``.  Use 2 for
            a binary split or 4+ for finer granularity.
        quantiles: Explicit quantile boundaries.  If provided,
            overrides ``n_levels``.  Must have ``n_levels - 1``
            elements, each in (0, 1).

    Returns:
        pd.Series of string labels (e.g., ``"low_vol"``,
        ``"medium_vol"``, ``"high_vol"``).  NaN-filled for the
        warm-up period where rolling volatility is unavailable.

    Example:
        >>> import pandas as pd, numpy as np
        >>> rng = np.random.default_rng(0)
        >>> returns = pd.Series(rng.normal(0, 0.01, 500))
        >>> labels = volatility_regime_labels(returns, n_levels=3)
        >>> print(labels.value_counts())

    See Also:
        trend_regime_labels: Label by trend direction.
        composite_regime_labels: Combine vol + trend labels.
    """
    r = pd.Series(np.asarray(returns, dtype=np.float64).flatten())
    rolling_vol = r.rolling(window=window, min_periods=max(window // 2, 2)).std()

    # Determine quantile boundaries
    if quantiles is None:
        quantiles = [i / n_levels for i in range(1, n_levels)]

    thresholds = rolling_vol.quantile(quantiles).values

    # Assign labels
    level_names = _vol_level_names(n_levels)
    labels = pd.Series(np.nan, index=r.index, name="vol_regime", dtype=object)

    valid = rolling_vol.notna()
    vol_vals = rolling_vol[valid].values

    label_arr = np.full(len(vol_vals), level_names[-1], dtype=object)
    for i, thresh in enumerate(thresholds):
        label_arr[vol_vals <= thresh] = level_names[min(i, len(level_names) - 1)]

    # Fix: assign from highest threshold down so that each observation
    # gets the correct bucket
    label_arr = np.full(len(vol_vals), level_names[0], dtype=object)
    for i in range(len(thresholds)):
        label_arr[vol_vals > thresholds[i]] = level_names[i + 1]

    labels.loc[valid] = label_arr

    # Propagate the original index if returns was a Series
    if isinstance(returns, pd.Series):
        labels.index = returns.index

    return labels


# ---------------------------------------------------------------------------
# Trend regime labels
# ---------------------------------------------------------------------------


def trend_regime_labels(
    returns: pd.Series | np.ndarray,
    *,
    fast_window: int = 10,
    slow_window: int = 50,
    hysteresis: float = 0.0005,
) -> pd.Series:
    """Label regimes based on moving average slope with hysteresis.

    Uses a dual moving-average crossover system with a hysteresis
    band to avoid whipsaw signals.  The result is a clean,
    three-state classification: **uptrend**, **downtrend**, or
    **sideways**.

    **Interpretation guidance:**

    - ``"uptrend"``: Fast MA is above slow MA by more than the
      hysteresis threshold.  Bullish bias.
    - ``"downtrend"``: Fast MA is below slow MA by more than
      the hysteresis threshold.  Bearish bias.
    - ``"sideways"``: The two MAs are within the hysteresis band.
      No directional conviction -- favour range-bound strategies.

    Parameters:
        returns: Return series.
        fast_window: Fast moving average window (periods).
        slow_window: Slow moving average window (periods).
        hysteresis: Minimum difference between fast and slow MA
            (in return units) required to declare a trend.  Larger
            values suppress whipsaws but delay signals.

    Returns:
        pd.Series of string labels (``"uptrend"``, ``"downtrend"``,
        ``"sideways"``).  NaN-filled during warm-up.

    Example:
        >>> import pandas as pd, numpy as np
        >>> rng = np.random.default_rng(0)
        >>> returns = pd.Series(rng.normal(0.001, 0.01, 500))
        >>> labels = trend_regime_labels(returns)
        >>> print(labels.value_counts())

    See Also:
        volatility_regime_labels: Label by vol level.
        composite_regime_labels: Combine vol + trend labels.
    """
    r = pd.Series(np.asarray(returns, dtype=np.float64).flatten())

    # Cumulative returns (price proxy)
    cum_price = (1 + r).cumprod()

    fast_ma = cum_price.rolling(window=fast_window, min_periods=fast_window).mean()
    slow_ma = cum_price.rolling(window=slow_window, min_periods=slow_window).mean()

    diff = fast_ma - slow_ma

    labels = pd.Series(np.nan, index=r.index, name="trend_regime", dtype=object)
    valid = diff.notna()

    # Apply hysteresis
    diff_vals = diff[valid].values
    label_arr = np.where(
        diff_vals > hysteresis, "uptrend",
        np.where(diff_vals < -hysteresis, "downtrend", "sideways"),
    )

    labels.loc[valid] = label_arr

    if isinstance(returns, pd.Series):
        labels.index = returns.index

    return labels


# ---------------------------------------------------------------------------
# Composite regime labels
# ---------------------------------------------------------------------------


def composite_regime_labels(
    returns: pd.Series | np.ndarray,
    *,
    vol_window: int = 21,
    fast_window: int = 10,
    slow_window: int = 50,
    hysteresis: float = 0.0005,
    n_vol_levels: int = 2,
) -> pd.Series:
    """Combine volatility and trend regimes into composite states.

    Creates 4-6 composite labels by crossing trend direction
    (uptrend / downtrend / sideways) with volatility level
    (low / high or low / medium / high).  Common composite states:

    - **bull_calm**: Uptrend + low vol.  The best environment for
      passive equity holding.
    - **bull_volatile**: Uptrend + high vol.  Often late-cycle or
      recovery rallies.
    - **bear_calm**: Downtrend + low vol.  Grinding bear markets.
    - **bear_volatile**: Downtrend + high vol.  Crisis periods
      (2008, March 2020).
    - **sideways_calm**: Range-bound, quiet.
    - **sideways_volatile**: Choppy, difficult to trade.

    **Interpretation guidance:**

    The composite label captures both *direction* and *turbulence*,
    which together determine the optimal strategy.  For instance,
    momentum strategies work in ``bull_calm`` but fail in
    ``bear_volatile``.

    Parameters:
        returns: Return series.
        vol_window: Window for rolling volatility.
        fast_window: Fast MA window for trend.
        slow_window: Slow MA window for trend.
        hysteresis: Trend hysteresis threshold.
        n_vol_levels: 2 or 3 volatility levels.

    Returns:
        pd.Series of string composite labels.  NaN-filled during
        warm-up.

    Example:
        >>> import pandas as pd, numpy as np
        >>> rng = np.random.default_rng(0)
        >>> returns = pd.Series(rng.normal(0.001, 0.01, 500))
        >>> labels = composite_regime_labels(returns)
        >>> print(labels.value_counts())

    See Also:
        volatility_regime_labels: Volatility-only labeling.
        trend_regime_labels: Trend-only labeling.
        regime_duration_analysis: Analyse how long each composite
            state typically lasts.
    """
    vol_labels = volatility_regime_labels(
        returns, window=vol_window, n_levels=n_vol_levels,
    )
    trend_labels = trend_regime_labels(
        returns,
        fast_window=fast_window,
        slow_window=slow_window,
        hysteresis=hysteresis,
    )

    # Map trend labels to short names
    trend_map = {
        "uptrend": "bull",
        "downtrend": "bear",
        "sideways": "sideways",
    }
    # Map vol labels to short names
    vol_map = {
        "low_vol": "calm",
        "medium_vol": "moderate",
        "high_vol": "volatile",
    }

    composite = pd.Series(
        np.nan, index=vol_labels.index, name="composite_regime", dtype=object,
    )
    both_valid = vol_labels.notna() & trend_labels.notna()

    trend_short = trend_labels[both_valid].map(trend_map)
    vol_short = vol_labels[both_valid].map(vol_map)

    composite.loc[both_valid] = trend_short.astype(str) + "_" + vol_short.astype(str)

    return composite


# ---------------------------------------------------------------------------
# Regime duration analysis
# ---------------------------------------------------------------------------


def regime_duration_analysis(
    states: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Analyse how long each regime typically lasts.

    Computes the survival function, hazard rate, and expected
    remaining duration for each regime.  This helps answer questions
    like "we've been in a bull regime for 60 days -- how much longer
    can we expect it to last?"

    **Interpretation guidance:**

    - **survival_curve[k]**: Probability that a regime-*k* spell
      lasts at least *d* periods.  A slowly-decaying curve means
      the regime tends to persist.
    - **hazard_rate[k]**: Instantaneous probability of exiting
      regime *k* after having been in it for *d* periods.  If the
      hazard rate is approximately constant, regime duration is
      memoryless (geometric distribution, consistent with Markov).
      An *increasing* hazard rate means longer spells are more
      likely to end soon.
    - **expected_remaining[k]**: Given that we are currently in
      regime *k* and have been for *d* periods, how many more
      periods should we expect?  Computed from the empirical
      survival function.

    Parameters:
        states: Integer regime labels, shape ``(T,)``.

    Returns:
        Dictionary with:

        - **durations** (dict[int, list[int]]): List of spell
          durations for each regime.
        - **survival_curve** (dict[int, pd.Series]): Kaplan-Meier
          style survival curve for each regime, indexed by duration.
        - **hazard_rate** (dict[int, pd.Series]): Empirical hazard
          rate for each regime, indexed by duration.
        - **expected_remaining** (dict[int, pd.Series]): Expected
          remaining duration conditional on having survived *d*
          periods, indexed by duration.
        - **summary** (pd.DataFrame): Per-regime summary with
          ``mean_duration``, ``median_duration``, ``max_duration``,
          ``n_spells``.

    Example:
        >>> states = np.array([0]*50 + [1]*30 + [0]*80 + [1]*40)
        >>> result = regime_duration_analysis(states)
        >>> print(result["summary"])
        >>> # Survival curve for regime 0
        >>> print(result["survival_curve"][0])

    See Also:
        regime_stability_score: Composite stability metric.
        composite_regime_labels: Generate regime labels to analyse.
    """
    s = np.asarray(states, dtype=int).flatten()
    T = len(s)
    unique_states = sorted(np.unique(s))

    # Extract spell durations
    durations: dict[int, list[int]] = {int(k): [] for k in unique_states}
    current_state = int(s[0])
    current_len = 1
    for t in range(1, T):
        if int(s[t]) == current_state:
            current_len += 1
        else:
            durations[current_state].append(current_len)
            current_state = int(s[t])
            current_len = 1
    durations[current_state].append(current_len)

    # Survival curves, hazard rates, expected remaining duration
    survival_curves: dict[int, pd.Series] = {}
    hazard_rates: dict[int, pd.Series] = {}
    expected_remaining: dict[int, pd.Series] = {}
    summary_records = []

    for k in unique_states:
        k = int(k)
        durs = durations[k]
        if not durs:
            survival_curves[k] = pd.Series(dtype=float)
            hazard_rates[k] = pd.Series(dtype=float)
            expected_remaining[k] = pd.Series(dtype=float)
            summary_records.append({
                "regime": k,
                "mean_duration": 0.0,
                "median_duration": 0.0,
                "max_duration": 0,
                "n_spells": 0,
            })
            continue

        max_dur = max(durs)
        n_spells = len(durs)

        # Kaplan-Meier style survival: S(d) = P(duration >= d)
        surv = np.zeros(max_dur + 1)
        for d in range(max_dur + 1):
            surv[d] = sum(1 for dur in durs if dur >= d) / n_spells

        surv_series = pd.Series(
            surv, index=range(max_dur + 1), name=f"survival_{k}",
        )
        survival_curves[k] = surv_series

        # Hazard rate: h(d) = P(exit at d | survived to d)
        # h(d) = (S(d) - S(d+1)) / S(d)
        hazard = np.zeros(max_dur)
        for d in range(max_dur):
            if surv[d] > 0:
                hazard[d] = (surv[d] - surv[d + 1]) / surv[d]
            else:
                hazard[d] = 0.0

        hazard_rates[k] = pd.Series(
            hazard, index=range(max_dur), name=f"hazard_{k}",
        )

        # Expected remaining duration given survival to d:
        # E[remaining | survived d] = sum_{j=d}^{max} S(j) / S(d) - 1
        # (using discrete version)
        exp_rem = np.zeros(max_dur + 1)
        for d in range(max_dur + 1):
            if surv[d] > 0:
                exp_rem[d] = sum(surv[j] for j in range(d, max_dur + 1)) / surv[d]
            else:
                exp_rem[d] = 0.0

        expected_remaining[k] = pd.Series(
            exp_rem, index=range(max_dur + 1), name=f"expected_remaining_{k}",
        )

        summary_records.append({
            "regime": k,
            "mean_duration": float(np.mean(durs)),
            "median_duration": float(np.median(durs)),
            "max_duration": int(max_dur),
            "n_spells": n_spells,
        })

    summary = pd.DataFrame(summary_records).set_index("regime")

    return {
        "durations": durations,
        "survival_curve": survival_curves,
        "hazard_rate": hazard_rates,
        "expected_remaining": expected_remaining,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _vol_level_names(n: int) -> list[str]:
    """Generate volatility level names for *n* levels."""
    if n == 2:
        return ["low_vol", "high_vol"]
    elif n == 3:
        return ["low_vol", "medium_vol", "high_vol"]
    else:
        return [f"vol_level_{i}" for i in range(n)]
