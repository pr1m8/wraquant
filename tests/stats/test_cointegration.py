"""Tests for cointegration and pairs trading utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wraquant.stats.cointegration import (
    engle_granger,
    find_cointegrated_pairs,
    half_life,
    hedge_ratio,
    pairs_backtest_signals,
    spread,
    zscore_signal,
)


def _make_cointegrated_pair(
    n: int = 500, seed: int = 42
) -> tuple[pd.Series, pd.Series]:
    """Create two cointegrated series: y2 = 2*y1 + noise."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    # Random walk for y1
    y1 = pd.Series(np.cumsum(rng.normal(0, 1, size=n)) + 100, index=dates, name="y1")
    # y2 is cointegrated with y1 (shared stochastic trend + small noise)
    noise = rng.normal(0, 0.5, size=n)
    y2 = pd.Series(2 * y1.values + noise + 50, index=dates, name="y2")
    return y1, y2


def _make_unrelated_pair(n: int = 500, seed: int = 42) -> tuple[pd.Series, pd.Series]:
    """Create two independent random walk series."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    y1 = pd.Series(np.cumsum(rng.normal(0, 1, size=n)) + 100, index=dates, name="y1")
    y2 = pd.Series(np.cumsum(rng.normal(0, 1, size=n)) + 100, index=dates, name="y2")
    return y1, y2


class TestEngleGranger:
    def test_cointegrated_pair_detected(self) -> None:
        y1, y2 = _make_cointegrated_pair()
        result = engle_granger(y1, y2)
        assert result["is_cointegrated"] is True
        assert result["p_value"] < 0.05
        assert "statistic" in result
        assert "hedge_ratio" in result
        assert "residuals" in result

    def test_unrelated_pair_not_cointegrated(self) -> None:
        # Use a seed that reliably produces non-cointegrated walks
        rng = np.random.default_rng(123)
        dates = pd.bdate_range("2020-01-01", periods=1000)
        y1 = pd.Series(np.cumsum(rng.normal(0, 1, size=1000)), index=dates)
        y2 = pd.Series(np.cumsum(rng.normal(0, 1, size=1000)), index=dates)
        result = engle_granger(y1, y2)
        # Unrelated random walks should typically not be cointegrated
        assert result["p_value"] > 0.01

    def test_hedge_ratio_close_to_true_value(self) -> None:
        y1, y2 = _make_cointegrated_pair()
        # y2 = 2*y1 + noise => regressing y2 on y1 gives hedge_ratio ~ 2
        result = engle_granger(y2, y1)
        assert abs(result["hedge_ratio"] - 2.0) < 0.5

    def test_residuals_length(self) -> None:
        y1, y2 = _make_cointegrated_pair()
        result = engle_granger(y1, y2)
        assert len(result["residuals"]) == len(y1)


class TestHalfLife:
    def test_positive_for_mean_reverting_spread(self) -> None:
        y1, y2 = _make_cointegrated_pair()
        s = spread(y1, y2)
        hl = half_life(s)
        assert hl > 0
        assert np.isfinite(hl)

    def test_large_for_random_walk(self) -> None:
        rng = np.random.default_rng(42)
        rw = pd.Series(np.cumsum(rng.normal(0, 1, size=500)))
        hl = half_life(rw)
        # A random walk is not strongly mean-reverting, so half-life should be large
        assert hl > 10 or hl == float("inf")


class TestSpread:
    def test_spread_with_known_ratio(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=100)
        y1 = pd.Series(np.arange(100, dtype=float) + 100, index=dates)
        y2 = pd.Series(np.arange(100, dtype=float) * 2 + 50, index=dates)
        s = spread(y1, y2, hedge_ratio=0.5)
        # spread = y1 - 0.5 * y2 = (100+i) - 0.5*(50+2i) = 100+i - 25 - i = 75
        expected = 75.0
        np.testing.assert_allclose(s.values, expected, atol=1e-10)

    def test_spread_auto_hedge_ratio(self) -> None:
        y1, y2 = _make_cointegrated_pair()
        s = spread(y1, y2)
        assert len(s) == len(y1)
        assert s.name == "spread"


class TestZscoreSignal:
    def test_zscore_approximately_normalized(self) -> None:
        y1, y2 = _make_cointegrated_pair(n=1000)
        s = spread(y1, y2)
        z = zscore_signal(s, window=20)
        # After warmup period, z-score should be approximately mean 0, std 1
        z_clean = z.dropna()
        assert abs(z_clean.mean()) < 0.5
        assert abs(z_clean.std() - 1.0) < 0.5

    def test_zscore_has_nan_warmup(self) -> None:
        y1, y2 = _make_cointegrated_pair()
        s = spread(y1, y2)
        z = zscore_signal(s, window=20)
        # First 19 values should be NaN (window - 1)
        assert z.iloc[:19].isna().all()


class TestHedgeRatio:
    def test_ols_recovers_known_ratio(self) -> None:
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-01", periods=500)
        y1 = pd.Series(np.cumsum(rng.normal(0, 1, size=500)) + 100, index=dates)
        y2 = pd.Series(3.0 * y1.values + rng.normal(0, 0.5, size=500), index=dates)
        hr = hedge_ratio(y2, y1, method="ols")
        assert abs(hr - 3.0) < 0.5

    def test_tls_method(self) -> None:
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-01", periods=500)
        y1 = pd.Series(np.cumsum(rng.normal(0, 1, size=500)) + 100, index=dates)
        y2 = pd.Series(2.0 * y1.values + rng.normal(0, 0.5, size=500), index=dates)
        hr = hedge_ratio(y2, y1, method="tls")
        assert abs(hr - 2.0) < 0.5

    def test_invalid_method_raises(self) -> None:
        import pytest

        y1, y2 = _make_cointegrated_pair()
        with pytest.raises(ValueError, match="Unknown hedge ratio method"):
            hedge_ratio(y1, y2, method="invalid")


class TestPairsBacktestSignals:
    def test_output_values_in_valid_set(self) -> None:
        y1, y2 = _make_cointegrated_pair(n=500)
        s = spread(y1, y2)
        signals = pairs_backtest_signals(s, entry_z=2.0, exit_z=0.5)
        unique_vals = set(signals.unique())
        assert unique_vals.issubset({-1, 0, 1})

    def test_output_length_matches_input(self) -> None:
        y1, y2 = _make_cointegrated_pair()
        s = spread(y1, y2)
        signals = pairs_backtest_signals(s)
        assert len(signals) == len(s)

    def test_signals_start_flat(self) -> None:
        y1, y2 = _make_cointegrated_pair()
        s = spread(y1, y2)
        signals = pairs_backtest_signals(s)
        # During warmup (NaN z-scores), signals should be 0
        assert signals.iloc[0] == 0


class TestFindCointegratedPairs:
    def test_finds_known_pair(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        dates = pd.bdate_range("2020-01-01", periods=n)
        # Asset A: random walk
        a = np.cumsum(rng.normal(0, 1, size=n)) + 100
        # Asset B: cointegrated with A
        b = 1.5 * a + rng.normal(0, 0.5, size=n)
        # Asset C: independent random walk
        c = np.cumsum(rng.normal(0, 1, size=n)) + 100

        prices_df = pd.DataFrame({"A": a, "B": b, "C": c}, index=dates)
        pairs = find_cointegrated_pairs(prices_df, significance=0.05)

        # Should find (A, B) or (B, A) as cointegrated
        pair_assets = [(p[0], p[1]) for p in pairs]
        assert ("A", "B") in pair_assets or ("B", "A") in pair_assets

    def test_returns_list_of_tuples(self) -> None:
        y1, y2 = _make_cointegrated_pair()
        prices_df = pd.DataFrame({"X": y1, "Y": y2})
        pairs = find_cointegrated_pairs(prices_df)
        assert isinstance(pairs, list)
        for pair in pairs:
            assert isinstance(pair, tuple)
            assert len(pair) == 4  # (asset1, asset2, p_value, hedge_ratio)
