"""Regime labeling and statistics."""

from __future__ import annotations

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
