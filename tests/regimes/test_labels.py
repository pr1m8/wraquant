"""Tests for regime labeling and statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wraquant.regimes.labels import label_regimes, regime_statistics


def _make_regime_data(
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    n = 200
    dates = pd.bdate_range("2020-01-01", periods=n)

    # State 0: bear, State 1: bull
    states_arr = np.concatenate([np.zeros(100, dtype=int), np.ones(100, dtype=int)])
    returns_arr = np.concatenate(
        [rng.normal(-0.005, 0.02, 100), rng.normal(0.005, 0.02, 100)]
    )

    states = pd.Series(states_arr, index=dates, name="state")
    returns = pd.Series(returns_arr, index=dates, name="returns")
    return states, returns


class TestLabelRegimes:
    def test_bull_bear_labels(self) -> None:
        states, returns = _make_regime_data()
        labels = label_regimes(states, returns)
        unique = set(labels.unique())
        assert "bull" in unique
        assert "bear" in unique

    def test_correct_assignment(self) -> None:
        states, returns = _make_regime_data()
        labels = label_regimes(states, returns)
        # State 0 has negative mean -> bear
        assert labels.iloc[0] == "bear"
        # State 1 has positive mean -> bull
        assert labels.iloc[-1] == "bull"

    def test_preserves_index(self) -> None:
        states, returns = _make_regime_data()
        labels = label_regimes(states, returns)
        assert labels.index.equals(states.index)


class TestRegimeStatistics:
    def test_columns(self) -> None:
        states, returns = _make_regime_data()
        stats = regime_statistics(returns, states)
        expected_cols = {"mean", "std", "skew", "count", "fraction"}
        assert expected_cols == set(stats.columns)

    def test_fractions_sum_to_one(self) -> None:
        states, returns = _make_regime_data()
        stats = regime_statistics(returns, states)
        np.testing.assert_allclose(stats["fraction"].sum(), 1.0, atol=1e-10)

    def test_count_totals(self) -> None:
        states, returns = _make_regime_data()
        stats = regime_statistics(returns, states)
        assert stats["count"].sum() == len(returns)

    def test_bear_has_negative_mean(self) -> None:
        states, returns = _make_regime_data()
        stats = regime_statistics(returns, states)
        # State 0 is bear
        assert stats.loc[0, "mean"] < 0
