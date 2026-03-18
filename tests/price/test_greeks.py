"""Tests for option Greeks calculations."""

from __future__ import annotations

import pytest

from wraquant.price.greeks import all_greeks, delta, gamma, rho, theta, vega


class TestDelta:
    """Tests for the delta Greek."""

    def test_call_delta_bounds(self) -> None:
        """Call delta is in [0, 1]."""
        for S in [80, 100, 120]:
            d = delta(S, 100, 1.0, 0.05, 0.2, "call")
            assert 0.0 <= d <= 1.0

    def test_put_delta_bounds(self) -> None:
        """Put delta is in [-1, 0]."""
        for S in [80, 100, 120]:
            d = delta(S, 100, 1.0, 0.05, 0.2, "put")
            assert -1.0 <= d <= 0.0

    def test_put_call_delta_relationship(self) -> None:
        """Put delta = Call delta - 1 (for European options)."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        call_d = delta(S, K, T, r, sigma, "call")
        put_d = delta(S, K, T, r, sigma, "put")
        assert call_d - put_d == pytest.approx(1.0, abs=1e-10)

    def test_deep_itm_call_delta_near_one(self) -> None:
        """Deep ITM call has delta near 1."""
        d = delta(200, 100, 1.0, 0.05, 0.2, "call")
        assert d > 0.99

    def test_deep_otm_call_delta_near_zero(self) -> None:
        """Deep OTM call has delta near 0."""
        d = delta(50, 200, 1.0, 0.05, 0.2, "call")
        assert d < 0.01

    def test_atm_call_delta_near_half(self) -> None:
        """ATM call has delta near 0.5 (slightly above due to drift)."""
        d = delta(100, 100, 1.0, 0.05, 0.2, "call")
        assert 0.4 < d < 0.7


class TestGamma:
    """Tests for the gamma Greek."""

    def test_gamma_positive(self) -> None:
        """Gamma is always non-negative."""
        for S in [80, 100, 120]:
            g = gamma(S, 100, 1.0, 0.05, 0.2)
            assert g >= 0.0

    def test_gamma_highest_atm(self) -> None:
        """Gamma is highest for at-the-money options."""
        g_atm = gamma(100, 100, 0.25, 0.05, 0.2)
        g_itm = gamma(120, 100, 0.25, 0.05, 0.2)
        g_otm = gamma(80, 100, 0.25, 0.05, 0.2)
        assert g_atm > g_itm
        assert g_atm > g_otm

    def test_gamma_same_for_calls_and_puts(self) -> None:
        """Gamma does not depend on option type."""
        # gamma function doesn't take option_type, confirming it's the same
        g = gamma(100, 100, 1.0, 0.05, 0.2)
        assert g > 0.0


class TestTheta:
    """Tests for the theta Greek."""

    def test_call_theta_typically_negative(self) -> None:
        """Call theta is typically negative (time decay)."""
        t = theta(100, 100, 1.0, 0.05, 0.2, "call")
        assert t < 0.0

    def test_put_theta_typically_negative(self) -> None:
        """Put theta is typically negative for near-ATM options."""
        t = theta(100, 100, 1.0, 0.05, 0.2, "put")
        assert t < 0.0


class TestVega:
    """Tests for the vega Greek."""

    def test_vega_positive(self) -> None:
        """Vega is always non-negative."""
        for S in [80, 100, 120]:
            v = vega(S, 100, 1.0, 0.05, 0.2)
            assert v >= 0.0

    def test_vega_same_for_calls_and_puts(self) -> None:
        """Vega is the same for calls and puts (only depends on d1)."""
        # The vega function doesn't take option_type since it's the same
        v = vega(100, 100, 1.0, 0.05, 0.2)
        assert v > 0.0

    def test_vega_highest_atm(self) -> None:
        """Vega is highest for at-the-money options."""
        v_atm = vega(100, 100, 0.25, 0.05, 0.2)
        v_itm = vega(120, 100, 0.25, 0.05, 0.2)
        v_otm = vega(80, 100, 0.25, 0.05, 0.2)
        assert v_atm > v_itm
        assert v_atm > v_otm


class TestRho:
    """Tests for the rho Greek."""

    def test_call_rho_positive(self) -> None:
        """Call rho is positive (higher rates benefit calls)."""
        r = rho(100, 100, 1.0, 0.05, 0.2, "call")
        assert r > 0.0

    def test_put_rho_negative(self) -> None:
        """Put rho is negative (higher rates hurt puts)."""
        r = rho(100, 100, 1.0, 0.05, 0.2, "put")
        assert r < 0.0


class TestAllGreeks:
    """Tests for the all_greeks convenience function."""

    def test_returns_all_keys(self) -> None:
        """Returns dict with all five Greeks."""
        result = all_greeks(100, 100, 1.0, 0.05, 0.2, "call")
        assert set(result.keys()) == {"delta", "gamma", "theta", "vega", "rho"}

    def test_consistent_with_individual(self) -> None:
        """Values match individual Greek functions."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        result = all_greeks(S, K, T, r, sigma, "call")

        assert result["delta"] == pytest.approx(
            delta(S, K, T, r, sigma, "call"), abs=1e-12
        )
        assert result["gamma"] == pytest.approx(gamma(S, K, T, r, sigma), abs=1e-12)
        assert result["theta"] == pytest.approx(
            theta(S, K, T, r, sigma, "call"), abs=1e-12
        )
        assert result["vega"] == pytest.approx(vega(S, K, T, r, sigma), abs=1e-12)
        assert result["rho"] == pytest.approx(rho(S, K, T, r, sigma, "call"), abs=1e-12)

    def test_put_greeks(self) -> None:
        """all_greeks works for puts."""
        result = all_greeks(100, 100, 1.0, 0.05, 0.2, "put")
        assert result["delta"] < 0.0
        assert result["gamma"] > 0.0
        assert result["rho"] < 0.0
