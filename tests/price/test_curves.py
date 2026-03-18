"""Tests for yield curve construction and utilities."""

from __future__ import annotations

import numpy as np
import pytest

from wraquant.price.curves import (
    bootstrap_zero_curve,
    discount_factor,
    forward_rate,
    interpolate_curve,
)


class TestBootstrap:
    """Tests for zero curve bootstrapping."""

    def test_positive_rates(self) -> None:
        """Bootstrap produces positive rates for positive par rates."""
        mats = [0.5, 1.0, 1.5, 2.0]
        pars = [0.04, 0.042, 0.044, 0.046]
        zeros = bootstrap_zero_curve(mats, pars, freq=2)
        assert np.all(zeros > 0)

    def test_correct_length(self) -> None:
        """Output length matches input length."""
        mats = [0.5, 1.0, 1.5, 2.0]
        pars = [0.04, 0.042, 0.044, 0.046]
        zeros = bootstrap_zero_curve(mats, pars, freq=2)
        assert len(zeros) == len(mats)

    def test_flat_curve(self) -> None:
        """When par rates are all the same, zero rates are close to par."""
        mats = [0.5, 1.0, 1.5, 2.0]
        pars = [0.05, 0.05, 0.05, 0.05]
        zeros = bootstrap_zero_curve(mats, pars, freq=2)
        # For a flat par curve, zero rates should be very close
        for z in zeros:
            assert z == pytest.approx(0.05, abs=0.005)

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched maturities and rates raises ValueError."""
        with pytest.raises(ValueError):
            bootstrap_zero_curve([0.5, 1.0], [0.04], freq=2)


class TestInterpolation:
    """Tests for yield curve interpolation."""

    def test_linear_at_knots(self) -> None:
        """Linear interpolation matches exactly at knot points."""
        mats = [0.5, 1.0, 2.0, 5.0]
        rates = [0.03, 0.035, 0.04, 0.045]
        result = interpolate_curve(mats, rates, mats, method="linear")
        np.testing.assert_allclose(result, rates, atol=1e-10)

    def test_cubic_at_knots(self) -> None:
        """Cubic interpolation matches exactly at knot points."""
        mats = [0.5, 1.0, 2.0, 5.0]
        rates = [0.03, 0.035, 0.04, 0.045]
        result = interpolate_curve(mats, rates, mats, method="cubic")
        np.testing.assert_allclose(result, rates, atol=1e-10)

    def test_interpolation_between_knots(self) -> None:
        """Interpolated values are between neighboring knot values (for monotone data)."""
        mats = [0.5, 1.0, 2.0, 5.0]
        rates = [0.03, 0.035, 0.04, 0.045]
        targets = [0.75, 1.5, 3.0]
        result = interpolate_curve(mats, rates, targets, method="linear")
        assert 0.03 < result[0] < 0.035  # Between 0.5y and 1y rates
        assert 0.035 < result[1] < 0.04  # Between 1y and 2y rates
        assert 0.04 < result[2] < 0.045  # Between 2y and 5y rates

    def test_flat_forward_at_knots(self) -> None:
        """Flat forward interpolation matches at knot points."""
        mats = [0.5, 1.0, 2.0, 5.0]
        rates = [0.03, 0.035, 0.04, 0.045]
        result = interpolate_curve(mats, rates, mats, method="flat_forward")
        np.testing.assert_allclose(result, rates, atol=1e-10)

    def test_unknown_method_raises(self) -> None:
        """Unknown interpolation method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown"):
            interpolate_curve([1.0], [0.05], [1.5], method="spline42")


class TestDiscountFactor:
    """Tests for discount factor calculation."""

    def test_df_at_zero_maturity(self) -> None:
        """Discount factor at maturity 0 is 1."""
        df = discount_factor(0.05, 0.0)
        assert df == pytest.approx(1.0, abs=1e-15)

    def test_df_decreasing_with_maturity(self) -> None:
        """Discount factor decreases with maturity (for positive rates)."""
        df1 = discount_factor(0.05, 1.0)
        df5 = discount_factor(0.05, 5.0)
        assert df5 < df1 < 1.0

    def test_df_known_value(self) -> None:
        """Discount factor matches exp(-r*t)."""
        df = discount_factor(0.05, 1.0)
        assert df == pytest.approx(np.exp(-0.05), abs=1e-15)

    def test_df_negative_rate(self) -> None:
        """Discount factor > 1 for negative rates."""
        df = discount_factor(-0.02, 1.0)
        assert df > 1.0


class TestForwardRate:
    """Tests for forward rate calculation."""

    def test_forward_rate_consistency(self) -> None:
        """Forward rate is consistent with zero rates.

        Investing at r1 for t1 then f for (t2-t1) equals investing at r2 for t2.
        exp(r1*t1) * exp(f*(t2-t1)) = exp(r2*t2)
        """
        mats = [1.0, 2.0, 3.0, 5.0]
        rates = [0.04, 0.045, 0.048, 0.05]
        t1, t2 = 1.0, 2.0
        fwd = forward_rate(rates, mats, t1, t2)

        # Verify consistency: r1*t1 + f*(t2-t1) = r2*t2
        r1 = rates[0]
        r2 = rates[1]
        assert r1 * t1 + fwd * (t2 - t1) == pytest.approx(r2 * t2, abs=1e-10)

    def test_forward_rate_positive(self) -> None:
        """Forward rate is positive for upward-sloping curve."""
        mats = [1.0, 2.0, 3.0]
        rates = [0.03, 0.04, 0.05]
        fwd = forward_rate(rates, mats, 1.0, 2.0)
        assert fwd > 0.0

    def test_t2_le_t1_raises(self) -> None:
        """t2 <= t1 raises ValueError."""
        with pytest.raises(ValueError, match="t2 must be greater"):
            forward_rate([0.04, 0.05], [1.0, 2.0], 2.0, 1.0)
