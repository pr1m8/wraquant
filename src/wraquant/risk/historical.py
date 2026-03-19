"""Historical crisis analysis and drawdown attribution.

This module provides tools for analysing portfolio behaviour during
historical crises, measuring event impacts, quantifying contagion,
and attributing drawdowns to individual assets.

These functions complement the stress testing module (``risk.stress``)
by focusing on *what actually happened* rather than hypothetical
scenarios. Use them for:

- Post-mortem analysis: understand what drove past losses.
- Regime-aware portfolio construction: identify assets that provide
  protection in crises.
- Contagion monitoring: detect when correlations spike during stress.
- Investor reporting: show drawdown history with recovery timelines.

References:
    - Forbes & Rigobon (2002), "No Contagion, Only Interdependence"
    - Bacon (2008), "Practical Portfolio Performance Measurement and
      Attribution"
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wraquant.risk.metrics import max_drawdown as _max_drawdown


def crisis_drawdowns(
    returns: pd.Series,
    top_n: int = 5,
) -> pd.DataFrame:
    """Identify the top N drawdowns with full lifecycle metrics.

    Scans the return series for the largest peak-to-trough drawdowns and
    reports start date, trough date, recovery date, duration, and
    magnitude for each.

    When to use:
        Use crisis drawdowns for:
        - Investor reporting: show the worst historical losses and
          recovery times.
        - Strategy evaluation: compare drawdown profiles across strategies.
        - Risk limit calibration: set max drawdown limits based on
          historical experience.

    Parameters:
        returns: Simple return series with a DatetimeIndex (or integer
            index).
        top_n: Number of largest drawdowns to return.

    Returns:
        pd.DataFrame with columns:
        - **start** -- Date the drawdown began (peak).
        - **trough** -- Date of maximum drawdown.
        - **end** -- Date the drawdown recovered (or last date if
          still in drawdown).
        - **drawdown** -- Magnitude of drawdown (negative number).
        - **days_to_trough** -- Trading days from start to trough.
        - **days_to_recovery** -- Trading days from trough to recovery
          (NaN if not recovered).
        - **total_days** -- Total drawdown duration.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> idx = pd.bdate_range("2020-01-01", periods=500)
        >>> returns = pd.Series(np.random.normal(0.0003, 0.01, 500), index=idx)
        >>> dd = crisis_drawdowns(returns, top_n=3)
        >>> len(dd) <= 3
        True

    See Also:
        drawdown_attribution: Which assets caused the drawdowns.
        wraquant.risk.metrics.max_drawdown: Single worst drawdown.
    """
    clean = returns.dropna()
    cum = (1 + clean).cumprod()
    running_max = cum.cummax()
    drawdowns = (cum - running_max) / running_max

    # Find drawdown periods
    in_drawdown = drawdowns < 0
    records: list[dict[str, Any]] = []

    i = 0
    n = len(drawdowns)
    while i < n:
        if in_drawdown.iloc[i]:
            start_idx = i - 1 if i > 0 else 0
            # Find trough
            j = i
            while j < n and in_drawdown.iloc[j]:
                j += 1
            # j is now the first index where drawdown == 0 (recovery) or end
            trough_pos = int(drawdowns.iloc[start_idx:j].argmin()) + start_idx
            dd_val = float(drawdowns.iloc[trough_pos])

            start_date = drawdowns.index[start_idx]
            trough_date = drawdowns.index[trough_pos]

            if j < n:
                end_date = drawdowns.index[j]
                days_to_recovery = j - trough_pos
                total_days = j - start_idx
            else:
                end_date = drawdowns.index[-1]
                days_to_recovery = float("nan")
                total_days = n - 1 - start_idx

            records.append(
                {
                    "start": start_date,
                    "trough": trough_date,
                    "end": end_date,
                    "drawdown": dd_val,
                    "days_to_trough": trough_pos - start_idx,
                    "days_to_recovery": days_to_recovery,
                    "total_days": total_days,
                }
            )
            i = j
        else:
            i += 1

    if not records:
        return pd.DataFrame(
            columns=[
                "start",
                "trough",
                "end",
                "drawdown",
                "days_to_trough",
                "days_to_recovery",
                "total_days",
            ]
        )

    df = pd.DataFrame(records)
    df = df.sort_values("drawdown", ascending=True).head(top_n).reset_index(drop=True)
    return df


def event_impact(
    returns: pd.Series,
    event_dates: list[str],
    window: int = 10,
) -> dict[str, Any]:
    """Measure portfolio returns around specific events.

    For each event date, extracts the returns in a window before and
    after the event and computes cumulative return, max drawdown, and
    volatility within each window.

    When to use:
        Use event impact analysis for:
        - Post-mortem: "how did the portfolio react to the Fed rate hike?"
        - Event studies: systematic analysis of recurring events
          (earnings, FOMC, NFP).
        - Scenario planning: calibrate stress scenarios based on actual
          event impacts.

    Parameters:
        returns: Return series with a DatetimeIndex.
        event_dates: List of event date strings (ISO format, e.g.,
            "2020-03-16"). Dates not in the index are matched to the
            nearest available date.
        window: Number of trading days before and after the event to
            analyse.

    Returns:
        Dictionary mapping each event date string to a dict with:
        - **pre_cumulative** (*float*) -- Cumulative return in the window
          before the event.
        - **post_cumulative** (*float*) -- Cumulative return in the window
          after the event.
        - **event_day_return** (*float*) -- Return on the event day itself.
        - **pre_vol** (*float*) -- Volatility in the pre-event window.
        - **post_vol** (*float*) -- Volatility in the post-event window.
        - **total_impact** (*float*) -- Cumulative return over the full
          window (pre + event + post).

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> idx = pd.bdate_range("2020-01-01", periods=252)
        >>> returns = pd.Series(np.random.normal(0.0005, 0.01, 252), index=idx)
        >>> result = event_impact(returns, ["2020-03-16", "2020-06-15"], window=5)
        >>> len(result) >= 1
        True

    See Also:
        wraquant.risk.stress.historical_stress_test: Replay known crises.
        crisis_drawdowns: Top drawdown periods.
    """
    results: dict[str, dict[str, float]] = {}
    ret = returns.dropna()

    for date_str in event_dates:
        target = pd.Timestamp(date_str)

        # Find nearest date in index
        if hasattr(ret.index, "get_indexer"):
            idx = ret.index.get_indexer([target], method="nearest")[0]
        else:
            idx = int(np.argmin(np.abs(ret.index - target)))

        if idx < 0 or idx >= len(ret):
            continue

        # Pre-event window
        pre_start = max(0, idx - window)
        pre_slice = ret.iloc[pre_start:idx]

        # Post-event window
        post_end = min(len(ret), idx + window + 1)
        post_slice = ret.iloc[idx + 1 : post_end]

        # Event day
        event_return = float(ret.iloc[idx])

        pre_cum = (
            float(np.prod(1 + pre_slice.values) - 1) if len(pre_slice) > 0 else 0.0
        )
        post_cum = (
            float(np.prod(1 + post_slice.values) - 1) if len(post_slice) > 0 else 0.0
        )

        pre_vol = float(pre_slice.std()) if len(pre_slice) > 1 else 0.0
        post_vol = float(post_slice.std()) if len(post_slice) > 1 else 0.0

        # Total impact over full window
        full_slice = ret.iloc[pre_start:post_end]
        total = (
            float(np.prod(1 + full_slice.values) - 1) if len(full_slice) > 0 else 0.0
        )

        results[date_str] = {
            "pre_cumulative": pre_cum,
            "post_cumulative": post_cum,
            "event_day_return": event_return,
            "pre_vol": pre_vol,
            "post_vol": post_vol,
            "total_impact": total,
        }

    return results


def contagion_analysis(
    returns_df: pd.DataFrame,
    crisis_dates: tuple[str, str],
) -> dict[str, Any]:
    """Compare normal vs. crisis-period correlations to detect contagion.

    Contagion occurs when correlations increase during stress periods
    beyond what would be expected from higher volatility alone. This
    function computes the correlation matrix in normal and crisis periods
    and tests for statistically significant increases.

    When to use:
        Use contagion analysis for:
        - Evaluating diversification reliability: do correlations spike
          when you need diversification most?
        - Stress testing: adjust portfolio correlations based on
          empirically observed crisis behaviour.
        - Regime-aware portfolio construction: allocate less to assets
          that become highly correlated during crises.

    Parameters:
        returns_df: Multi-asset return DataFrame with DatetimeIndex.
        crisis_dates: Tuple of (start_date, end_date) strings defining
            the crisis period.

    Returns:
        Dictionary containing:
        - **normal_corr** (*pd.DataFrame*) -- Correlation matrix during
          non-crisis period.
        - **crisis_corr** (*pd.DataFrame*) -- Correlation matrix during
          the crisis period.
        - **corr_change** (*pd.DataFrame*) -- Change in correlation
          (crisis - normal).
        - **avg_normal_corr** (*float*) -- Average off-diagonal correlation
          in normal period.
        - **avg_crisis_corr** (*float*) -- Average off-diagonal correlation
          in crisis period.
        - **contagion_detected** (*bool*) -- True if average crisis
          correlation significantly exceeds normal.
        - **n_normal** (*int*) -- Number of normal-period observations.
        - **n_crisis** (*int*) -- Number of crisis-period observations.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> idx = pd.bdate_range("2019-01-01", periods=500)
        >>> returns = pd.DataFrame({
        ...     "A": np.random.normal(0.0005, 0.01, 500),
        ...     "B": np.random.normal(0.0003, 0.012, 500),
        ... }, index=idx)
        >>> result = contagion_analysis(returns, ("2020-02-01", "2020-06-01"))
        >>> "contagion_detected" in result
        True

    See Also:
        wraquant.risk.stress.joint_stress_test: Apply correlation shocks.

    References:
        - Forbes & Rigobon (2002), "No Contagion, Only Interdependence:
          Measuring Stock Market Comovements"
    """
    start = pd.Timestamp(crisis_dates[0])
    end = pd.Timestamp(crisis_dates[1])

    crisis_mask = (returns_df.index >= start) & (returns_df.index <= end)
    normal_mask = ~crisis_mask

    crisis_returns = returns_df[crisis_mask].dropna()
    normal_returns = returns_df[normal_mask].dropna()

    crisis_corr = crisis_returns.corr()
    normal_corr = normal_returns.corr()
    corr_change = crisis_corr - normal_corr

    n = len(returns_df.columns)

    def _avg_offdiag(corr_df: pd.DataFrame) -> float:
        """Mean of off-diagonal elements."""
        vals = corr_df.values
        mask = ~np.eye(n, dtype=bool)
        if mask.sum() == 0:
            return 0.0
        return float(np.mean(vals[mask]))

    avg_normal = _avg_offdiag(normal_corr)
    avg_crisis = _avg_offdiag(crisis_corr)

    # Simple heuristic: contagion detected if crisis correlation is
    # substantially higher (> 0.1 absolute increase)
    contagion = avg_crisis - avg_normal > 0.1

    return {
        "normal_corr": normal_corr,
        "crisis_corr": crisis_corr,
        "corr_change": corr_change,
        "avg_normal_corr": avg_normal,
        "avg_crisis_corr": avg_crisis,
        "contagion_detected": contagion,
        "n_normal": int(normal_mask.sum()),
        "n_crisis": int(crisis_mask.sum()),
    }


def drawdown_attribution(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
) -> pd.DataFrame:
    """Attribute portfolio drawdowns to individual asset contributions.

    For each point in the drawdown, decomposes the portfolio's loss from
    peak into per-asset contributions. This shows which assets are
    responsible for the drawdown at each point in time.

    When to use:
        Use drawdown attribution for:
        - Post-mortem analysis: "which position caused the 2020 drawdown?"
        - Risk monitoring: track per-asset drawdown contributions in
          real time.
        - Portfolio construction: identify assets that consistently
          contribute to drawdowns and consider hedging or removing them.

    Parameters:
        returns_df: Multi-asset return DataFrame (columns = assets).
        weights: Portfolio weight vector aligned with columns.

    Returns:
        pd.DataFrame with:
        - **portfolio_dd** -- Total portfolio drawdown at each point.
        - One column per asset showing that asset's contribution to
          the drawdown.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> idx = pd.bdate_range("2020-01-01", periods=252)
        >>> returns = pd.DataFrame({
        ...     "A": np.random.normal(0.0005, 0.01, 252),
        ...     "B": np.random.normal(0.0003, 0.015, 252),
        ... }, index=idx)
        >>> weights = np.array([0.6, 0.4])
        >>> attr = drawdown_attribution(returns, weights)
        >>> "portfolio_dd" in attr.columns
        True

    See Also:
        crisis_drawdowns: Identify top drawdown periods.
        wraquant.risk.stress.marginal_stress_contribution: Stress-based
            attribution.
    """
    clean = returns_df.dropna()
    assets = clean.columns.tolist()

    # Portfolio returns
    port_returns = clean.values @ weights

    # Portfolio cumulative and drawdowns
    port_cum = np.cumprod(1 + port_returns)
    port_running_max = np.maximum.accumulate(port_cum)
    port_dd = (port_cum - port_running_max) / port_running_max

    # Per-asset cumulative weighted returns
    weighted_returns = clean.values * weights[np.newaxis, :]
    asset_cum = np.cumsum(weighted_returns, axis=0)

    # Attribution: per-asset contribution to drawdown
    # We track cumulative weighted return since the peak
    n = len(clean)
    peak_idx = 0
    contributions = np.zeros((n, len(assets)))

    cum_port = np.cumsum(port_returns)
    running_max_cum = np.maximum.accumulate(cum_port)

    for t in range(n):
        # Find the most recent peak
        if t > 0 and cum_port[t - 1] >= running_max_cum[t - 1] - 1e-15:
            peak_idx = t
        # Asset contribution = cumulative weighted return since peak
        if t > peak_idx:
            contributions[t] = asset_cum[t] - asset_cum[peak_idx]
        elif t == peak_idx and t > 0:
            contributions[t] = 0.0

    result = pd.DataFrame(
        contributions, index=clean.index, columns=[f"{a}_contribution" for a in assets]
    )
    result.insert(0, "portfolio_dd", port_dd)

    return result
