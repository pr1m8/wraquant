"""Tests for fixed income pricing functions."""

from __future__ import annotations

import numpy as np
import pytest

from wraquant.price.fixed_income import (
    bond_price,
    bond_yield,
    convexity,
    duration,
    modified_duration,
    zero_rate,
)


class TestBondPrice:
    """Tests for bond pricing."""

    def test_par_bond(self) -> None:
        """When coupon rate = YTM, price = face value."""
        price = bond_price(1000, 0.05, 0.05, 10, 2)
        assert price == pytest.approx(1000.0, abs=1e-8)

    def test_premium_bond(self) -> None:
        """When coupon rate > YTM, bond trades at premium."""
        price = bond_price(1000, 0.06, 0.04, 10, 2)
        assert price > 1000.0

    def test_discount_bond(self) -> None:
        """When coupon rate < YTM, bond trades at discount."""
        price = bond_price(1000, 0.04, 0.06, 10, 2)
        assert price < 1000.0

    def test_zero_coupon_bond(self) -> None:
        """Zero coupon bond price equals PV of face value."""
        face = 1000.0
        ytm = 0.05
        periods = 10
        freq = 2
        price = bond_price(face, 0.0, ytm, periods, freq)
        expected = face / (1.0 + ytm / freq) ** periods
        assert price == pytest.approx(expected, abs=1e-8)

    def test_positive_price(self) -> None:
        """Bond price is always positive for positive inputs."""
        price = bond_price(1000, 0.05, 0.1, 20, 2)
        assert price > 0.0


class TestBondYield:
    """Tests for yield-to-maturity calculation."""

    def test_par_bond_yield(self) -> None:
        """YTM of a par bond equals the coupon rate."""
        ytm = bond_yield(1000, 1000, 0.05, 10, 2)
        assert ytm == pytest.approx(0.05, abs=1e-6)

    def test_discount_bond_yield(self) -> None:
        """YTM of a discount bond is higher than coupon rate."""
        ytm = bond_yield(950, 1000, 0.05, 10, 2)
        assert ytm > 0.05

    def test_premium_bond_yield(self) -> None:
        """YTM of a premium bond is lower than coupon rate."""
        ytm = bond_yield(1050, 1000, 0.05, 10, 2)
        assert ytm < 0.05

    def test_round_trip(self) -> None:
        """Price -> yield -> price round trip."""
        original_price = 950.0
        ytm = bond_yield(original_price, 1000, 0.05, 10, 2)
        recovered_price = bond_price(1000, 0.05, ytm, 10, 2)
        assert recovered_price == pytest.approx(original_price, abs=1e-6)


class TestDuration:
    """Tests for Macaulay and modified duration."""

    def test_duration_positive(self) -> None:
        """Duration is positive for coupon bonds."""
        d = duration(1000, 0.05, 0.05, 10, 2)
        assert d > 0.0

    def test_duration_less_than_maturity(self) -> None:
        """Macaulay duration is less than maturity for coupon bonds."""
        periods = 20
        freq = 2
        maturity = periods / freq  # 10 years
        d = duration(1000, 0.05, 0.05, periods, freq)
        assert d < maturity

    def test_zero_coupon_duration_equals_maturity(self) -> None:
        """For zero coupon bond, Macaulay duration = maturity."""
        periods = 10
        freq = 2
        maturity = periods / freq
        d = duration(1000, 0.0, 0.05, periods, freq)
        assert d == pytest.approx(maturity, abs=1e-8)

    def test_modified_duration_less_than_macaulay(self) -> None:
        """Modified duration < Macaulay duration (for positive yields)."""
        mac = duration(1000, 0.05, 0.05, 10, 2)
        mod = modified_duration(1000, 0.05, 0.05, 10, 2)
        assert mod < mac

    def test_modified_duration_formula(self) -> None:
        """Modified duration = Macaulay / (1 + y/freq)."""
        mac = duration(1000, 0.05, 0.06, 10, 2)
        mod = modified_duration(1000, 0.05, 0.06, 10, 2)
        expected_mod = mac / (1.0 + 0.06 / 2)
        assert mod == pytest.approx(expected_mod, abs=1e-10)

    def test_higher_coupon_lower_duration(self) -> None:
        """Higher coupon rate leads to lower duration."""
        d_low = duration(1000, 0.03, 0.05, 20, 2)
        d_high = duration(1000, 0.08, 0.05, 20, 2)
        assert d_high < d_low


class TestConvexity:
    """Tests for bond convexity."""

    def test_convexity_positive(self) -> None:
        """Convexity is positive for standard bonds."""
        c = convexity(1000, 0.05, 0.05, 10, 2)
        assert c > 0.0

    def test_longer_maturity_higher_convexity(self) -> None:
        """Longer maturity bonds have higher convexity."""
        c_short = convexity(1000, 0.05, 0.05, 4, 2)
        c_long = convexity(1000, 0.05, 0.05, 20, 2)
        assert c_long > c_short


class TestZeroRate:
    """Tests for zero coupon rate."""

    def test_positive_rate(self) -> None:
        """Zero rate is positive when price < face value."""
        r = zero_rate(950, 1000, 1.0)
        assert r > 0.0

    def test_negative_rate_premium(self) -> None:
        """Zero rate is negative when price > face value."""
        r = zero_rate(1050, 1000, 1.0)
        assert r < 0.0

    def test_round_trip(self) -> None:
        """rate -> price -> rate round trip."""
        r_original = 0.05
        price = 1000 * np.exp(-r_original * 2.0)
        r_recovered = zero_rate(price, 1000, 2.0)
        assert r_recovered == pytest.approx(r_original, abs=1e-10)

    def test_invalid_price_raises(self) -> None:
        """Negative price raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            zero_rate(-100, 1000, 1.0)

    def test_invalid_periods_raises(self) -> None:
        """Non-positive periods raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            zero_rate(950, 1000, 0.0)
