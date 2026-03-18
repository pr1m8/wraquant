"""Tests for wraquant.math.ergodicity."""

from __future__ import annotations

import numpy as np
import pytest

from wraquant.math.ergodicity import (
    ensemble_average,
    ergodicity_gap,
    ergodicity_ratio,
    growth_optimal_leverage,
    kelly_fraction,
    time_average,
)


class TestEnsembleAverage:
    """Tests for ensemble_average."""

    def test_equals_np_mean(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.01, 0.05, 1000)
        assert ensemble_average(returns) == pytest.approx(np.mean(returns))

    def test_simple_case(self) -> None:
        assert ensemble_average([0.1, -0.1, 0.2]) == pytest.approx(
            (0.1 - 0.1 + 0.2) / 3
        )


class TestTimeAverage:
    """Tests for time_average."""

    def test_known_geometric_series(self) -> None:
        """Constant 10% return each period: geometric mean should be 0.10."""
        returns = np.full(100, 0.10)
        ta = time_average(returns)
        assert ta == pytest.approx(0.10, abs=1e-10)

    def test_symmetric_returns_below_arithmetic(self) -> None:
        """Geometric mean should be below arithmetic mean for volatile returns."""
        returns = np.array([0.20, -0.20, 0.20, -0.20])
        ta = time_average(returns)
        ea = ensemble_average(returns)
        assert ta < ea


class TestErgodicityGap:
    """Tests for ergodicity_gap."""

    def test_positive_for_volatile_returns(self) -> None:
        """Volatile returns should have a positive ergodicity gap."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.05, 0.20, 10_000)
        gap = ergodicity_gap(returns)
        assert gap > 0.0

    def test_zero_for_constant_returns(self) -> None:
        returns = np.full(100, 0.05)
        gap = ergodicity_gap(returns)
        assert gap == pytest.approx(0.0, abs=1e-10)


class TestKellyFraction:
    """Tests for kelly_fraction."""

    def test_reasonable_range(self) -> None:
        """Kelly fraction for modest positive-EV returns should be reasonable."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.01, 0.05, 5000)
        kf = kelly_fraction(returns)
        assert 0.0 <= kf <= 5.0

    def test_positive_for_positive_ev(self) -> None:
        """Positive expected returns should yield positive Kelly fraction."""
        returns = np.array([0.10, 0.10, 0.10, -0.05])
        kf = kelly_fraction(returns)
        assert kf > 0.0


class TestGrowthOptimalLeverage:
    """Tests for growth_optimal_leverage."""

    def test_positive_leverage(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.02, 0.05, 5000)
        lev = growth_optimal_leverage(returns)
        assert lev > 0.0

    def test_with_risk_free(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.02, 0.05, 5000)
        lev = growth_optimal_leverage(returns, risk_free=0.005)
        assert lev > 0.0


class TestErgodicityRatio:
    """Tests for ergodicity_ratio."""

    def test_le_one_for_volatile(self) -> None:
        """For volatile returns the ratio should be <= 1."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.05, 0.20, 10_000)
        ratio = ergodicity_ratio(returns)
        assert ratio <= 1.0 + 1e-10

    def test_one_for_constant(self) -> None:
        """Constant returns are ergodic: ratio should be 1.0."""
        returns = np.full(100, 0.05)
        ratio = ergodicity_ratio(returns)
        assert ratio == pytest.approx(1.0, abs=1e-10)
