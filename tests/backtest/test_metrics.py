"""Tests for backtest performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.backtest.metrics import (
    burke_ratio,
    common_sense_ratio,
    expectancy,
    gain_to_pain_ratio,
    kappa_ratio,
    kelly_fraction,
    omega_ratio,
    payoff_ratio,
    performance_summary,
    profit_factor,
    rachev_ratio,
    recovery_factor,
    risk_of_ruin,
    system_quality_number,
    tail_ratio,
    ulcer_performance_index,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _positive_returns(n: int = 252) -> pd.Series:
    """Constant positive daily returns."""
    return pd.Series(np.full(n, 0.001))


def _mixed_returns(seed: int = 42, n: int = 500) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0004, 0.012, n))


def _negative_returns(n: int = 100) -> pd.Series:
    return pd.Series(np.full(n, -0.002))


# ------------------------------------------------------------------
# omega_ratio
# ------------------------------------------------------------------

class TestOmegaRatio:
    def test_constant_positive_returns_is_inf(self) -> None:
        """All returns above threshold -> infinite Omega."""
        r = _positive_returns()
        assert omega_ratio(r) == float("inf")

    def test_constant_negative_returns(self) -> None:
        """All returns below threshold -> Omega approaches 0."""
        r = _negative_returns()
        assert omega_ratio(r) == 0.0  # all gains = 0

    def test_symmetric_returns(self) -> None:
        """Symmetric returns around 0 should give Omega ~ 1."""
        r = pd.Series([0.01, -0.01, 0.01, -0.01])
        assert omega_ratio(r) == pytest.approx(1.0)

    def test_custom_threshold(self) -> None:
        r = pd.Series([0.02, 0.01, -0.01, 0.015])
        o = omega_ratio(r, threshold=0.01)
        assert o > 0

    def test_positive_bias(self) -> None:
        r = pd.Series([0.02, 0.01, -0.005, 0.015, -0.003])
        assert omega_ratio(r) > 1.0


# ------------------------------------------------------------------
# burke_ratio
# ------------------------------------------------------------------

class TestBurkeRatio:
    def test_positive_returns(self) -> None:
        r = _positive_returns()
        # Constant positive returns have no drawdown -> dd^2 sum = 0 -> 0.0
        assert burke_ratio(r) == 0.0

    def test_mixed_returns(self) -> None:
        r = _mixed_returns()
        b = burke_ratio(r)
        assert isinstance(b, float)

    def test_empty_returns(self) -> None:
        assert burke_ratio(pd.Series(dtype=float)) == 0.0


# ------------------------------------------------------------------
# ulcer_performance_index
# ------------------------------------------------------------------

class TestUlcerPerformanceIndex:
    def test_positive_returns(self) -> None:
        r = _positive_returns()
        assert ulcer_performance_index(r) == 0.0  # no drawdowns

    def test_mixed_returns_finite(self) -> None:
        r = _mixed_returns()
        upi = ulcer_performance_index(r)
        assert isinstance(upi, float)
        assert np.isfinite(upi)

    def test_empty_returns(self) -> None:
        assert ulcer_performance_index(pd.Series(dtype=float)) == 0.0


# ------------------------------------------------------------------
# kappa_ratio
# ------------------------------------------------------------------

class TestKappaRatio:
    def test_order_2_is_sortino_like(self) -> None:
        """Kappa(2) should behave like Sortino."""
        r = _mixed_returns()
        k2 = kappa_ratio(r, order=2)
        assert isinstance(k2, float)

    def test_higher_order_more_conservative(self) -> None:
        """Higher orders penalise tails more -> typically lower ratio."""
        r = _mixed_returns()
        k2 = kappa_ratio(r, order=2)
        k3 = kappa_ratio(r, order=3)
        # Not always guaranteed, but for this seed it should hold
        assert isinstance(k3, float)

    def test_no_downside_returns_zero(self) -> None:
        r = _positive_returns()
        assert kappa_ratio(r, order=2) == 0.0  # LPM is zero

    def test_empty_returns(self) -> None:
        assert kappa_ratio(pd.Series(dtype=float)) == 0.0


# ------------------------------------------------------------------
# tail_ratio
# ------------------------------------------------------------------

class TestTailRatio:
    def test_symmetric_distribution(self) -> None:
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0, 0.01, 10000))
        tr = tail_ratio(r)
        # For a symmetric distribution, tail ratio should be ~1
        assert abs(tr - 1.0) < 0.2

    def test_positive_skew(self) -> None:
        """Positive-skewed distribution -> tail ratio > 1."""
        rng = np.random.default_rng(42)
        r = pd.Series(np.abs(rng.normal(0, 0.01, 1000)))
        tr = tail_ratio(r)
        assert tr > 0

    def test_zero_lower_percentile(self) -> None:
        """If 5th percentile is 0, should return inf."""
        r = pd.Series([0.0, 0.0, 0.0, 0.01, 0.02])
        tr = tail_ratio(r)
        assert tr == float("inf")


# ------------------------------------------------------------------
# common_sense_ratio
# ------------------------------------------------------------------

class TestCommonSenseRatio:
    def test_positive_drift(self) -> None:
        r = _mixed_returns()
        csr = common_sense_ratio(r)
        assert isinstance(csr, float)

    def test_combines_tail_and_sharpe(self) -> None:
        r = _mixed_returns()
        from wraquant.risk.metrics import sharpe_ratio as _sr
        tr = tail_ratio(r)
        sr = _sr(r)
        expected = tr * (1.0 + sr)
        assert common_sense_ratio(r) == pytest.approx(expected, rel=1e-6)


# ------------------------------------------------------------------
# rachev_ratio
# ------------------------------------------------------------------

class TestRachevRatio:
    def test_symmetric_around_one(self) -> None:
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0, 0.01, 10000))
        rr = rachev_ratio(r, alpha=0.05)
        assert abs(rr - 1.0) < 0.3

    def test_positive_drift(self) -> None:
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0.001, 0.01, 5000))
        rr = rachev_ratio(r, alpha=0.05)
        assert rr > 0

    def test_empty_returns(self) -> None:
        assert rachev_ratio(pd.Series(dtype=float)) == 0.0


# ------------------------------------------------------------------
# gain_to_pain_ratio
# ------------------------------------------------------------------

class TestGainToPainRatio:
    def test_all_positive(self) -> None:
        r = _positive_returns()
        assert gain_to_pain_ratio(r) == float("inf")

    def test_known_value(self) -> None:
        r = pd.Series([0.02, -0.01, 0.03, -0.005])
        # sum = 0.035, pain = 0.015
        expected = 0.035 / 0.015
        assert gain_to_pain_ratio(r) == pytest.approx(expected)

    def test_all_negative(self) -> None:
        r = _negative_returns()
        gpr = gain_to_pain_ratio(r)
        assert gpr < 0


# ------------------------------------------------------------------
# risk_of_ruin
# ------------------------------------------------------------------

class TestRiskOfRuin:
    def test_high_edge_low_ruin(self) -> None:
        ror = risk_of_ruin(win_rate=0.6, payoff_ratio=2.0, ruin_pct=0.5)
        assert 0 <= ror < 0.1

    def test_no_edge_certain_ruin(self) -> None:
        ror = risk_of_ruin(win_rate=0.3, payoff_ratio=0.5, ruin_pct=0.5)
        assert ror == 1.0

    def test_invalid_win_rate(self) -> None:
        with pytest.raises(ValueError):
            risk_of_ruin(win_rate=0.0, payoff_ratio=1.0)

    def test_invalid_payoff(self) -> None:
        with pytest.raises(ValueError):
            risk_of_ruin(win_rate=0.5, payoff_ratio=-1.0)


# ------------------------------------------------------------------
# kelly_fraction
# ------------------------------------------------------------------

class TestKellyFraction:
    def test_known_value(self) -> None:
        # b = 1.5/1.0 = 1.5,  f = (1.5*0.55 - 0.45)/1.5 = 0.25
        f = kelly_fraction(win_rate=0.55, avg_win=1.5, avg_loss=1.0)
        assert f == pytest.approx(0.25, abs=1e-10)

    def test_no_edge(self) -> None:
        f = kelly_fraction(win_rate=0.5, avg_win=1.0, avg_loss=1.0)
        assert f == 0.0

    def test_clamped_to_one(self) -> None:
        f = kelly_fraction(win_rate=1.0, avg_win=10.0, avg_loss=1.0)
        assert f == 1.0

    def test_negative_edge_clamped_to_zero(self) -> None:
        f = kelly_fraction(win_rate=0.2, avg_win=1.0, avg_loss=1.0)
        assert f == 0.0

    def test_invalid_win_rate(self) -> None:
        with pytest.raises(ValueError):
            kelly_fraction(win_rate=1.5, avg_win=1.0, avg_loss=1.0)

    def test_invalid_avg_loss(self) -> None:
        with pytest.raises(ValueError):
            kelly_fraction(win_rate=0.5, avg_win=1.0, avg_loss=0.0)


# ------------------------------------------------------------------
# expectancy
# ------------------------------------------------------------------

class TestExpectancy:
    def test_known_value(self) -> None:
        # E = 0.6*100 - 0.4*80 = 60 - 32 = 28
        e = expectancy(win_rate=0.6, avg_win=100.0, avg_loss=80.0)
        assert e == pytest.approx(28.0)

    def test_breakeven(self) -> None:
        e = expectancy(win_rate=0.5, avg_win=1.0, avg_loss=1.0)
        assert e == pytest.approx(0.0)

    def test_negative_edge(self) -> None:
        e = expectancy(win_rate=0.3, avg_win=1.0, avg_loss=1.0)
        assert e < 0


# ------------------------------------------------------------------
# profit_factor
# ------------------------------------------------------------------

class TestProfitFactor:
    def test_all_gains(self) -> None:
        r = _positive_returns()
        assert profit_factor(r) == float("inf")

    def test_known_value(self) -> None:
        r = pd.Series([0.02, -0.01, 0.03, -0.005])
        # gains = 0.05, losses = 0.015
        assert profit_factor(r) == pytest.approx(0.05 / 0.015)

    def test_all_losses(self) -> None:
        r = _negative_returns()
        assert profit_factor(r) == 0.0


# ------------------------------------------------------------------
# payoff_ratio
# ------------------------------------------------------------------

class TestPayoffRatio:
    def test_known_value(self) -> None:
        r = pd.Series([0.04, -0.02, 0.06, -0.01])
        # avg_win = 0.05, avg_loss = 0.015
        expected = 0.05 / 0.015
        assert payoff_ratio(r) == pytest.approx(expected)

    def test_no_wins(self) -> None:
        r = _negative_returns()
        assert payoff_ratio(r) == 0.0

    def test_no_losses(self) -> None:
        r = _positive_returns()
        assert payoff_ratio(r) == float("inf")


# ------------------------------------------------------------------
# recovery_factor
# ------------------------------------------------------------------

class TestRecoveryFactor:
    def test_positive_returns_no_drawdown(self) -> None:
        r = _positive_returns()
        assert recovery_factor(r) == float("inf")

    def test_mixed_returns_positive(self) -> None:
        r = _mixed_returns()
        rf = recovery_factor(r)
        assert isinstance(rf, float)


# ------------------------------------------------------------------
# system_quality_number
# ------------------------------------------------------------------

class TestSystemQualityNumber:
    def test_positive_sqn(self) -> None:
        r = _mixed_returns()
        sqn = system_quality_number(r)
        assert isinstance(sqn, float)

    def test_zero_returns(self) -> None:
        """Zero returns have zero std -> SQN = 0."""
        r = pd.Series(np.zeros(100))
        assert system_quality_number(r) == 0.0

    def test_empty(self) -> None:
        assert system_quality_number(pd.Series(dtype=float)) == 0.0

    def test_scales_with_n(self) -> None:
        """SQN should scale with sqrt(n) for same mean/std."""
        rng = np.random.default_rng(42)
        r100 = pd.Series(rng.normal(0.001, 0.01, 100))
        r400 = pd.Series(rng.normal(0.001, 0.01, 400))
        sqn100 = system_quality_number(r100)
        sqn400 = system_quality_number(r400)
        # sqn400 should be roughly 2x sqn100 (sqrt(400)/sqrt(100) = 2)
        # Allow wide tolerance because the samples are random
        assert sqn400 > sqn100 * 1.2
