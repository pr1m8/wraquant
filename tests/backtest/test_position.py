"""Tests for position sizing and weight management utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.backtest.position import (
    PositionSizer,
    clip_weights,
    invert_signal,
    rebalance_threshold,
)


# ------------------------------------------------------------------
# PositionSizer.fixed_fraction
# ------------------------------------------------------------------

class TestFixedFraction:
    def test_basic(self) -> None:
        assert PositionSizer.fixed_fraction(100_000, 0.02) == 2000.0

    def test_zero_risk(self) -> None:
        assert PositionSizer.fixed_fraction(100_000, 0.0) == 0.0

    def test_full_risk(self) -> None:
        assert PositionSizer.fixed_fraction(50_000, 1.0) == 50_000.0

    def test_negative_equity_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            PositionSizer.fixed_fraction(-100, 0.01)

    def test_invalid_risk_pct_raises(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            PositionSizer.fixed_fraction(100_000, 1.5)


# ------------------------------------------------------------------
# PositionSizer.kelly_criterion
# ------------------------------------------------------------------

class TestKellyCriterion:
    def test_favorable_odds(self) -> None:
        f = PositionSizer.kelly_criterion(win_rate=0.6, avg_win=2.0, avg_loss=1.0)
        # Kelly = (0.6*2 - 0.4) / 2 = 0.4
        assert abs(f - 0.4) < 1e-10

    def test_breakeven(self) -> None:
        # 50 % win rate, even odds -> Kelly = 0
        f = PositionSizer.kelly_criterion(win_rate=0.5, avg_win=1.0, avg_loss=1.0)
        assert f == 0.0

    def test_clamped_above_one(self) -> None:
        # Extreme edge: 100 % win rate, huge avg_win -> clamped to 1.0
        f = PositionSizer.kelly_criterion(win_rate=1.0, avg_win=10.0, avg_loss=1.0)
        assert f == 1.0

    def test_negative_kelly_clamped_to_zero(self) -> None:
        # Very unfavorable: 20 % win rate, even odds
        f = PositionSizer.kelly_criterion(win_rate=0.2, avg_win=1.0, avg_loss=1.0)
        assert f == 0.0

    def test_invalid_win_rate(self) -> None:
        with pytest.raises(ValueError, match="win_rate"):
            PositionSizer.kelly_criterion(win_rate=1.5, avg_win=1.0, avg_loss=1.0)

    def test_invalid_avg_loss(self) -> None:
        with pytest.raises(ValueError, match="avg_loss must be positive"):
            PositionSizer.kelly_criterion(win_rate=0.5, avg_win=1.0, avg_loss=0.0)


# ------------------------------------------------------------------
# PositionSizer.volatility_targeting
# ------------------------------------------------------------------

class TestVolatilityTargeting:
    def test_scales_up(self) -> None:
        rng = np.random.default_rng(0)
        rets = pd.Series(rng.normal(0, 0.005, size=100))  # low vol
        scalar = PositionSizer.volatility_targeting(rets, target_vol=0.20, lookback=20)
        # Realised vol ~ 0.005 * sqrt(252) ~ 0.08 -> scalar ~ 2.5
        assert scalar > 1.0

    def test_scales_down(self) -> None:
        rng = np.random.default_rng(0)
        rets = pd.Series(rng.normal(0, 0.05, size=100))  # high vol
        scalar = PositionSizer.volatility_targeting(rets, target_vol=0.05, lookback=20)
        assert scalar < 1.0

    def test_short_series_returns_one(self) -> None:
        rets = pd.Series([0.01, -0.01])
        assert PositionSizer.volatility_targeting(rets, target_vol=0.10, lookback=20) == 1.0


# ------------------------------------------------------------------
# PositionSizer.risk_parity_weights
# ------------------------------------------------------------------

class TestRiskParityWeights:
    def test_sums_to_one(self) -> None:
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        w = PositionSizer.risk_parity_weights(cov)
        assert abs(w.sum() - 1.0) < 1e-10

    def test_lower_vol_gets_more(self) -> None:
        cov = np.array([[0.01, 0.0], [0.0, 0.09]])
        w = PositionSizer.risk_parity_weights(cov)
        assert w[0] > w[1]  # lower vol asset gets higher weight

    def test_accepts_dataframe(self) -> None:
        df = pd.DataFrame(
            [[0.04, 0.01], [0.01, 0.09]], columns=["A", "B"], index=["A", "B"]
        )
        w = PositionSizer.risk_parity_weights(df)
        assert abs(w.sum() - 1.0) < 1e-10

    def test_nonsquare_raises(self) -> None:
        with pytest.raises(ValueError, match="square"):
            PositionSizer.risk_parity_weights(np.ones((2, 3)))


# ------------------------------------------------------------------
# PositionSizer.equal_risk_contribution
# ------------------------------------------------------------------

class TestEqualRiskContribution:
    def test_sums_to_one(self) -> None:
        cov = np.array([[0.04, 0.005], [0.005, 0.09]])
        w = PositionSizer.equal_risk_contribution(cov)
        assert abs(w.sum() - 1.0) < 1e-8

    def test_equal_vols_equal_weights(self) -> None:
        # Diagonal with equal variances -> equal weights
        cov = np.eye(3) * 0.04
        w = PositionSizer.equal_risk_contribution(cov)
        np.testing.assert_allclose(w, np.ones(3) / 3, atol=1e-4)

    def test_unequal_vols_lower_vol_higher_weight(self) -> None:
        cov = np.diag([0.01, 0.04, 0.09])
        w = PositionSizer.equal_risk_contribution(cov)
        # Lower vol assets should get more weight
        assert w[0] > w[1] > w[2]


# ------------------------------------------------------------------
# invert_signal
# ------------------------------------------------------------------

class TestInvertSignal:
    def test_series(self) -> None:
        sig = pd.Series([1.0, -1.0, 0.0, 0.5])
        inv = invert_signal(sig)
        pd.testing.assert_series_equal(inv, pd.Series([-1.0, 1.0, 0.0, -0.5]))

    def test_dataframe(self) -> None:
        df = pd.DataFrame({"a": [1, -1], "b": [0.5, -0.5]})
        inv = invert_signal(df)
        expected = pd.DataFrame({"a": [-1, 1], "b": [-0.5, 0.5]})
        pd.testing.assert_frame_equal(inv, expected)

    def test_ndarray(self) -> None:
        arr = np.array([1.0, -1.0, 0.0])
        np.testing.assert_array_equal(invert_signal(arr), np.array([-1.0, 1.0, 0.0]))


# ------------------------------------------------------------------
# clip_weights
# ------------------------------------------------------------------

class TestClipWeights:
    def test_clips_and_normalises(self) -> None:
        w = np.array([0.7, 0.2, 0.1])
        clipped = clip_weights(w, min_w=0.0, max_w=0.4)
        assert float(np.max(clipped)) <= 0.4 + 1e-6
        assert abs(clipped.sum() - 1.0) < 1e-6

    def test_series_input(self) -> None:
        w = pd.Series([0.8, 0.1, 0.1])
        clipped = clip_weights(w, min_w=0.0, max_w=0.5)
        assert clipped.sum() == pytest.approx(1.0, abs=1e-10)

    def test_no_clip_needed(self) -> None:
        w = np.array([0.5, 0.3, 0.2])
        clipped = clip_weights(w, min_w=0.0, max_w=1.0)
        assert abs(clipped.sum() - 1.0) < 1e-10


# ------------------------------------------------------------------
# rebalance_threshold
# ------------------------------------------------------------------

class TestRebalanceThreshold:
    def test_no_rebalance_needed(self) -> None:
        current = np.array([0.50, 0.30, 0.20])
        target = np.array([0.51, 0.29, 0.20])
        assert rebalance_threshold(current, target, threshold=0.05) is False

    def test_rebalance_needed(self) -> None:
        current = np.array([0.60, 0.25, 0.15])
        target = np.array([0.33, 0.34, 0.33])
        assert rebalance_threshold(current, target, threshold=0.05) is True

    def test_exact_threshold(self) -> None:
        current = np.array([0.55, 0.45])
        target = np.array([0.50, 0.50])
        # Max drift = 0.05, threshold = 0.05 -> not exceeded (<=)
        assert rebalance_threshold(current, target, threshold=0.05) is False

    def test_series_input(self) -> None:
        current = pd.Series([0.7, 0.3])
        target = pd.Series([0.5, 0.5])
        assert rebalance_threshold(current, target, threshold=0.10) is True


# ------------------------------------------------------------------
# risk_parity_position
# ------------------------------------------------------------------

from wraquant.backtest.position import (
    regime_conditional_sizing,
    regime_signal_filter,
    risk_parity_position,
)


class TestRiskParityPosition:
    def test_weights_sum_to_one(self) -> None:
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        w = risk_parity_position(cov)
        assert abs(w.sum() - 1.0) < 1e-4

    def test_lower_vol_gets_more_weight(self) -> None:
        cov = np.diag([0.01, 0.04, 0.09])
        w = risk_parity_position(cov)
        assert w[0] > w[1] > w[2]

    def test_with_target_vol(self) -> None:
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        w = risk_parity_position(cov, target_vol=0.10)
        # With target_vol, weights may not sum to 1 (scaled by leverage)
        assert isinstance(w, np.ndarray)
        assert len(w) == 2
        # Weights should be different from the unscaled version
        w_unscaled = risk_parity_position(cov)
        assert not np.allclose(w, w_unscaled, atol=1e-6)

    def test_three_assets(self) -> None:
        cov = np.array([
            [0.04, 0.005, 0.002],
            [0.005, 0.09, 0.01],
            [0.002, 0.01, 0.01],
        ])
        w = risk_parity_position(cov)
        assert abs(w.sum() - 1.0) < 1e-4
        assert len(w) == 3


# ------------------------------------------------------------------
# regime_conditional_sizing
# ------------------------------------------------------------------

class TestRegimeConditionalSizing:
    def test_scaling_down(self) -> None:
        base = np.array([0.5, 0.3, 0.2])
        probs = {"normal": 0.2, "high_vol": 0.8}
        mults = {"normal": 1.0, "high_vol": 0.5}
        adj = regime_conditional_sizing(base, probs, mults)
        # Effective mult = 0.2*1.0 + 0.8*0.5 = 0.6
        np.testing.assert_allclose(adj, base * 0.6, atol=1e-10)

    def test_scaling_up(self) -> None:
        base = np.array([0.5, 0.5])
        probs = {"low_vol": 1.0}
        mults = {"low_vol": 1.5}
        adj = regime_conditional_sizing(base, probs, mults)
        np.testing.assert_allclose(adj, base * 1.5, atol=1e-10)

    def test_neutral_regime(self) -> None:
        base = np.array([0.4, 0.3, 0.3])
        probs = {"normal": 1.0}
        mults = {"normal": 1.0}
        adj = regime_conditional_sizing(base, probs, mults)
        np.testing.assert_allclose(adj, base, atol=1e-10)

    def test_missing_multiplier_defaults_to_one(self) -> None:
        base = np.array([0.5, 0.5])
        probs = {"unknown_regime": 1.0}
        mults = {"normal": 0.5}  # no "unknown_regime" key
        adj = regime_conditional_sizing(base, probs, mults)
        # Default multiplier = 1.0
        np.testing.assert_allclose(adj, base, atol=1e-10)

    def test_mixed_regimes(self) -> None:
        base = np.array([0.6, 0.4])
        probs = {"bull": 0.6, "bear": 0.4}
        mults = {"bull": 1.2, "bear": 0.5}
        # Effective mult = 0.6*1.2 + 0.4*0.5 = 0.72 + 0.20 = 0.92
        adj = regime_conditional_sizing(base, probs, mults)
        np.testing.assert_allclose(adj, base * 0.92, atol=1e-10)

    def test_pandas_series_input(self) -> None:
        base = pd.Series([0.5, 0.3, 0.2])
        probs = {"normal": 1.0}
        mults = {"normal": 0.8}
        adj = regime_conditional_sizing(base, probs, mults)
        expected = np.array([0.4, 0.24, 0.16])
        np.testing.assert_allclose(adj, expected, atol=1e-10)


# ------------------------------------------------------------------
# regime_signal_filter
# ------------------------------------------------------------------

class TestRegimeSignalFilter:
    def test_filters_low_prob_signals(self) -> None:
        signals = pd.Series([1, -1, 1, -1, 1], dtype=float)
        # Regime 0 has high prob for first 3, low for last 2
        regime_probs = np.array([
            [0.8, 0.2],
            [0.7, 0.3],
            [0.9, 0.1],
            [0.3, 0.7],
            [0.2, 0.8],
        ])
        filtered = regime_signal_filter(signals, regime_probs, active_regime=0, min_prob=0.6)
        # First 3 signals should be preserved, last 2 zeroed
        assert filtered.iloc[0] == 1.0
        assert filtered.iloc[1] == -1.0
        assert filtered.iloc[2] == 1.0
        assert filtered.iloc[3] == 0.0
        assert filtered.iloc[4] == 0.0

    def test_all_active(self) -> None:
        signals = np.array([1.0, -1.0, 1.0])
        regime_probs = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
        ])
        filtered = regime_signal_filter(signals, regime_probs, active_regime=0, min_prob=0.5)
        np.testing.assert_array_equal(filtered.values, signals)

    def test_all_filtered(self) -> None:
        signals = np.array([1.0, -1.0, 1.0])
        regime_probs = np.array([
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
        ])
        filtered = regime_signal_filter(signals, regime_probs, active_regime=0, min_prob=0.6)
        np.testing.assert_array_equal(filtered.values, np.zeros(3))

    def test_different_active_regime(self) -> None:
        signals = pd.Series([1.0, 1.0, 1.0])
        regime_probs = np.array([
            [0.1, 0.9],
            [0.8, 0.2],
            [0.3, 0.7],
        ])
        # Use regime 1 as active
        filtered = regime_signal_filter(signals, regime_probs, active_regime=1, min_prob=0.6)
        assert filtered.iloc[0] == 1.0  # regime 1 prob=0.9 >= 0.6
        assert filtered.iloc[1] == 0.0  # regime 1 prob=0.2 < 0.6
        assert filtered.iloc[2] == 1.0  # regime 1 prob=0.7 >= 0.6

    def test_invalid_probs_shape_raises(self) -> None:
        signals = np.array([1.0, -1.0])
        regime_probs = np.array([0.8, 0.2])  # 1-D, invalid
        with pytest.raises(ValueError, match="2-D"):
            regime_signal_filter(signals, regime_probs)

    def test_preserves_series_index(self) -> None:
        idx = pd.date_range("2020-01-01", periods=3, freq="D")
        signals = pd.Series([1.0, -1.0, 1.0], index=idx, name="sig")
        regime_probs = np.array([
            [0.8, 0.2],
            [0.3, 0.7],
            [0.9, 0.1],
        ])
        filtered = regime_signal_filter(signals, regime_probs, min_prob=0.6)
        pd.testing.assert_index_equal(filtered.index, idx)
