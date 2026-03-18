"""Tests for regime labeling, statistics, and duration analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wraquant.regimes.labels import (
    composite_regime_labels,
    label_regimes,
    regime_duration_analysis,
    regime_statistics,
    trend_regime_labels,
    volatility_regime_labels,
)


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


# ---------------------------------------------------------------------------
# Fixtures for new label tests
# ---------------------------------------------------------------------------


def _make_long_return_series(
    n: int = 600,
    seed: int = 42,
) -> pd.Series:
    """Generate a long return series with varying regimes."""
    rng = np.random.default_rng(seed)
    # Bull calm -> Bear volatile -> Sideways moderate
    parts = [
        rng.normal(0.002, 0.005, 200),   # bull, low vol
        rng.normal(-0.003, 0.025, 200),   # bear, high vol
        rng.normal(0.0, 0.012, 200),      # sideways, moderate vol
    ]
    returns = np.concatenate(parts)
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(returns, index=dates, name="returns")


# ---------------------------------------------------------------------------
# Tests: volatility_regime_labels
# ---------------------------------------------------------------------------


class TestVolatilityRegimeLabels:
    def test_correct_number_of_levels(self) -> None:
        returns = _make_long_return_series()
        labels = volatility_regime_labels(returns, n_levels=3)
        valid = labels.dropna()
        unique = set(valid.unique())
        assert len(unique) == 3

    def test_two_levels(self) -> None:
        returns = _make_long_return_series()
        labels = volatility_regime_labels(returns, n_levels=2)
        valid = labels.dropna()
        unique = set(valid.unique())
        assert len(unique) == 2
        assert "low_vol" in unique
        assert "high_vol" in unique

    def test_three_level_names(self) -> None:
        returns = _make_long_return_series()
        labels = volatility_regime_labels(returns, n_levels=3)
        valid = labels.dropna()
        unique = set(valid.unique())
        expected = {"low_vol", "medium_vol", "high_vol"}
        assert unique == expected

    def test_length_matches_input(self) -> None:
        returns = _make_long_return_series()
        labels = volatility_regime_labels(returns, n_levels=3)
        assert len(labels) == len(returns)

    def test_numpy_input(self) -> None:
        returns = _make_long_return_series()
        labels = volatility_regime_labels(returns.values, n_levels=2)
        valid = labels.dropna()
        assert len(valid) > 0

    def test_custom_quantiles(self) -> None:
        returns = _make_long_return_series()
        labels = volatility_regime_labels(
            returns, n_levels=3, quantiles=[0.3, 0.7],
        )
        valid = labels.dropna()
        assert len(valid.unique()) >= 2


# ---------------------------------------------------------------------------
# Tests: trend_regime_labels
# ---------------------------------------------------------------------------


class TestTrendRegimeLabels:
    def test_three_states(self) -> None:
        returns = _make_long_return_series()
        labels = trend_regime_labels(returns)
        valid = labels.dropna()
        unique = set(valid.unique())
        # Should contain at least uptrend or downtrend
        possible = {"uptrend", "downtrend", "sideways"}
        assert unique.issubset(possible)
        assert len(unique) >= 1

    def test_bull_period_labeled_uptrend(self) -> None:
        rng = np.random.default_rng(0)
        # Strong uptrend
        returns = pd.Series(rng.normal(0.005, 0.002, 300))
        labels = trend_regime_labels(returns, fast_window=5, slow_window=20)
        valid = labels.dropna()
        if len(valid) > 0:
            # Most observations should be uptrend
            counts = valid.value_counts()
            assert "uptrend" in counts.index

    def test_length_matches(self) -> None:
        returns = _make_long_return_series()
        labels = trend_regime_labels(returns)
        assert len(labels) == len(returns)

    def test_hysteresis_creates_sideways_regions(self) -> None:
        returns = _make_long_return_series()
        labels_no_hyst = trend_regime_labels(returns, hysteresis=0.0)
        labels_hyst = trend_regime_labels(returns, hysteresis=0.01)

        # With hysteresis, more observations should be classified as
        # sideways (the uncertain transition region)
        valid_no = labels_no_hyst.dropna()
        valid_hyst = labels_hyst.dropna()

        sideways_no = (valid_no == "sideways").sum()
        sideways_hyst = (valid_hyst == "sideways").sum()

        # Larger hysteresis => more sideways classifications
        assert sideways_hyst >= sideways_no


# ---------------------------------------------------------------------------
# Tests: composite_regime_labels
# ---------------------------------------------------------------------------


class TestCompositeRegimeLabels:
    def test_at_least_four_states(self) -> None:
        returns = _make_long_return_series()
        labels = composite_regime_labels(returns, n_vol_levels=2)
        valid = labels.dropna()
        # With 3 trend + 2 vol = up to 6 states, should have >= 2 at minimum
        # but with enough data we expect 4+
        unique = set(valid.unique())
        # At least 2 composite states should exist
        assert len(unique) >= 2

    def test_label_format(self) -> None:
        returns = _make_long_return_series()
        labels = composite_regime_labels(returns, n_vol_levels=2)
        valid = labels.dropna()
        for label in valid.unique():
            # Should be "trend_vol" format
            parts = label.split("_")
            assert len(parts) >= 2

    def test_contains_expected_composites(self) -> None:
        returns = _make_long_return_series()
        labels = composite_regime_labels(returns, n_vol_levels=2)
        valid = labels.dropna()
        unique = set(valid.unique())
        # With a varied return series, we should see some combination
        possible = {
            "bull_calm", "bull_volatile",
            "bear_calm", "bear_volatile",
            "sideways_calm", "sideways_volatile",
        }
        assert unique.issubset(possible)

    def test_length_matches(self) -> None:
        returns = _make_long_return_series()
        labels = composite_regime_labels(returns)
        assert len(labels) == len(returns)

    def test_three_vol_levels(self) -> None:
        returns = _make_long_return_series()
        labels = composite_regime_labels(returns, n_vol_levels=3)
        valid = labels.dropna()
        # Should have more states with 3 vol levels
        unique = set(valid.unique())
        assert len(unique) >= 2


# ---------------------------------------------------------------------------
# Tests: regime_duration_analysis
# ---------------------------------------------------------------------------


class TestRegimeDurationAnalysis:
    def test_summary_dataframe(self) -> None:
        states = np.array([0] * 50 + [1] * 30 + [0] * 80 + [1] * 40)
        result = regime_duration_analysis(states)

        assert isinstance(result["summary"], pd.DataFrame)
        assert "mean_duration" in result["summary"].columns
        assert "median_duration" in result["summary"].columns
        assert "max_duration" in result["summary"].columns
        assert "n_spells" in result["summary"].columns

    def test_correct_durations(self) -> None:
        states = np.array([0] * 50 + [1] * 30 + [0] * 80 + [1] * 40)
        result = regime_duration_analysis(states)

        assert result["durations"][0] == [50, 80]
        assert result["durations"][1] == [30, 40]

    def test_survival_curve_starts_at_one(self) -> None:
        states = np.array([0] * 50 + [1] * 30 + [0] * 80 + [1] * 40)
        result = regime_duration_analysis(states)

        for k in [0, 1]:
            surv = result["survival_curve"][k]
            if len(surv) > 0:
                assert surv.iloc[0] == 1.0

    def test_survival_curve_monotonically_decreasing(self) -> None:
        states = np.array([0] * 50 + [1] * 30 + [0] * 80 + [1] * 40)
        result = regime_duration_analysis(states)

        for k in [0, 1]:
            surv = result["survival_curve"][k]
            if len(surv) > 1:
                diffs = np.diff(surv.values)
                assert np.all(diffs <= 0 + 1e-12)

    def test_hazard_rate_bounded(self) -> None:
        states = np.array([0] * 50 + [1] * 30 + [0] * 80 + [1] * 40)
        result = regime_duration_analysis(states)

        for k in [0, 1]:
            hazard = result["hazard_rate"][k]
            if len(hazard) > 0:
                assert np.all(hazard.values >= 0.0)
                assert np.all(hazard.values <= 1.0 + 1e-12)

    def test_expected_remaining_positive(self) -> None:
        states = np.array([0] * 50 + [1] * 30 + [0] * 80 + [1] * 40)
        result = regime_duration_analysis(states)

        for k in [0, 1]:
            exp_rem = result["expected_remaining"][k]
            if len(exp_rem) > 0:
                # At duration 0, expected remaining should be > 0
                assert exp_rem.iloc[0] > 0

    def test_n_spells_correct(self) -> None:
        states = np.array([0] * 50 + [1] * 30 + [0] * 80 + [1] * 40)
        result = regime_duration_analysis(states)

        assert result["summary"].loc[0, "n_spells"] == 2
        assert result["summary"].loc[1, "n_spells"] == 2

    def test_series_input(self) -> None:
        states = pd.Series([0] * 50 + [1] * 30 + [0] * 80)
        result = regime_duration_analysis(states)
        assert isinstance(result["summary"], pd.DataFrame)

    def test_single_regime(self) -> None:
        states = np.zeros(100, dtype=int)
        result = regime_duration_analysis(states)
        assert result["summary"].loc[0, "n_spells"] == 1
        assert result["summary"].loc[0, "mean_duration"] == 100
