"""Tests for wraquant.price.characteristic — characteristic function methods."""

from __future__ import annotations

import numpy as np
import pytest

from wraquant.price.characteristic import (
    cgmy_characteristic,
    characteristic_function_price,
    heston_characteristic,
    nig_characteristic,
    vg_characteristic,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bs_call(S: float, K: float, r: float, T: float, sigma: float) -> float:
    """Black-Scholes call price for comparison."""
    from scipy.stats import norm

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def _bs_char_fn(
    spot: float, r: float, T: float, sigma: float
):
    """Return the BS characteristic function of log(S_T)."""
    log_S = np.log(spot)
    drift = (r - 0.5 * sigma ** 2) * T

    def char_fn(u):
        return np.exp(1j * u * (log_S + drift) - 0.5 * sigma ** 2 * T * u ** 2)

    return char_fn


# ---------------------------------------------------------------------------
# characteristic_function_price
# ---------------------------------------------------------------------------

class TestCharacteristicFunctionPrice:
    """Tests for the unified pricing interface."""

    def test_bs_via_fft(self) -> None:
        """FFT method with BS char fn matches Black-Scholes."""
        S, K, r, T, sigma = 100.0, 100.0, 0.05, 1.0, 0.2
        bs_price = _bs_call(S, K, r, T, sigma)
        char_fn = _bs_char_fn(S, r, T, sigma)

        price = characteristic_function_price(char_fn, S, K, r, T, method="fft")
        assert price == pytest.approx(bs_price, abs=0.3)

    def test_bs_via_cos(self) -> None:
        """COS method with BS char fn matches Black-Scholes."""
        S, K, r, T, sigma = 100.0, 100.0, 0.05, 1.0, 0.2
        bs_price = _bs_call(S, K, r, T, sigma)
        char_fn = _bs_char_fn(S, r, T, sigma)

        price = characteristic_function_price(
            char_fn, S, K, r, T, method="cos", n_terms=256,
        )
        assert price == pytest.approx(bs_price, abs=0.5)

    def test_invalid_method(self) -> None:
        """Invalid method should raise ValueError."""
        char_fn = _bs_char_fn(100, 0.05, 1.0, 0.2)
        with pytest.raises(ValueError, match="Unknown method"):
            characteristic_function_price(char_fn, 100, 100, 0.05, 1.0, method="xyz")

    def test_nonnegative_price(self) -> None:
        """Price should be non-negative."""
        char_fn = _bs_char_fn(100, 0.05, 1.0, 0.2)
        price = characteristic_function_price(char_fn, 100, 110, 0.05, 1.0)
        assert price >= 0.0


# ---------------------------------------------------------------------------
# Heston characteristic function
# ---------------------------------------------------------------------------

class TestHestonCharacteristic:
    """Tests for heston_characteristic."""

    def test_phi_zero_is_one(self) -> None:
        """Characteristic function at u=0 should be 1."""
        char_fn = heston_characteristic(
            v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3,
            rho=-0.7, rf=0.05, T=1.0, spot=100.0,
        )
        val = char_fn(np.array([0.0]))
        assert np.abs(val[0]) == pytest.approx(1.0, abs=1e-10)

    def test_heston_price_positive(self) -> None:
        """Heston char fn pricing should give positive call price."""
        char_fn = heston_characteristic(
            v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3,
            rho=-0.7, rf=0.05, T=1.0, spot=100.0,
        )
        price = characteristic_function_price(char_fn, 100, 100, 0.05, 1.0)
        assert price > 0.0

    def test_heston_price_finite(self) -> None:
        """Heston price should be finite."""
        char_fn = heston_characteristic(
            v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3,
            rho=-0.7, rf=0.05, T=1.0, spot=100.0,
        )
        price = characteristic_function_price(char_fn, 100, 100, 0.05, 1.0)
        assert np.isfinite(price)

    def test_heston_price_reasonable(self) -> None:
        """Heston ATM call should be in a reasonable range."""
        # With v0 = theta = 0.04, effective vol ~ 0.2
        char_fn = heston_characteristic(
            v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3,
            rho=-0.7, rf=0.05, T=1.0, spot=100.0,
        )
        price = characteristic_function_price(char_fn, 100, 100, 0.05, 1.0)
        # BS price with sigma=0.2 is ~10.45
        assert 5.0 < price < 20.0

    def test_heston_itm_higher_than_otm(self) -> None:
        """ITM call should be more expensive than OTM."""
        char_fn = heston_characteristic(
            v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3,
            rho=-0.7, rf=0.05, T=1.0, spot=100.0,
        )
        price_itm = characteristic_function_price(char_fn, 100, 80, 0.05, 1.0)
        price_otm = characteristic_function_price(char_fn, 100, 120, 0.05, 1.0)
        assert price_itm > price_otm

    def test_heston_vol_smile(self) -> None:
        """Heston with negative rho should produce vol smile/skew."""
        char_fn = heston_characteristic(
            v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.5,
            rho=-0.8, rf=0.05, T=1.0, spot=100.0,
        )
        # Price at multiple strikes
        strikes = [90, 100, 110]
        prices = [
            characteristic_function_price(char_fn, 100, K, 0.05, 1.0)
            for K in strikes
        ]
        # All should be positive and finite
        for p in prices:
            assert p > 0.0
            assert np.isfinite(p)


# ---------------------------------------------------------------------------
# Variance Gamma characteristic function
# ---------------------------------------------------------------------------

class TestVGCharacteristic:
    """Tests for vg_characteristic."""

    def test_phi_zero_is_one(self) -> None:
        """phi(0) should be 1."""
        char_fn = vg_characteristic(0.2, 0.5, -0.1, 0.05, 1.0, spot=100)
        val = char_fn(np.array([0.0]))
        assert np.abs(val[0]) == pytest.approx(1.0, abs=1e-10)

    def test_vg_price_positive(self) -> None:
        """VG char fn should produce positive call price."""
        char_fn = vg_characteristic(0.2, 0.5, -0.1, 0.05, 1.0, spot=100)
        price = characteristic_function_price(char_fn, 100, 100, 0.05, 1.0)
        assert price > 0.0

    def test_vg_price_finite(self) -> None:
        """VG price should be finite."""
        char_fn = vg_characteristic(0.2, 0.5, -0.1, 0.05, 1.0, spot=100)
        price = characteristic_function_price(char_fn, 100, 100, 0.05, 1.0)
        assert np.isfinite(price)


# ---------------------------------------------------------------------------
# NIG characteristic function
# ---------------------------------------------------------------------------

class TestNIGCharacteristic:
    """Tests for nig_characteristic."""

    def test_phi_zero_is_one(self) -> None:
        """phi(0) should be 1."""
        char_fn = nig_characteristic(15.0, -3.0, 0.0, 0.5, 0.05, 1.0, spot=100)
        val = char_fn(np.array([0.0]))
        assert np.abs(val[0]) == pytest.approx(1.0, abs=1e-10)

    def test_nig_price_positive(self) -> None:
        """NIG char fn should produce positive call price."""
        char_fn = nig_characteristic(15.0, -3.0, 0.0, 0.5, 0.05, 1.0, spot=100)
        price = characteristic_function_price(char_fn, 100, 100, 0.05, 1.0)
        assert price > 0.0

    def test_nig_price_finite(self) -> None:
        """NIG price should be finite."""
        char_fn = nig_characteristic(15.0, -3.0, 0.0, 0.5, 0.05, 1.0, spot=100)
        price = characteristic_function_price(char_fn, 100, 100, 0.05, 1.0)
        assert np.isfinite(price)


# ---------------------------------------------------------------------------
# CGMY characteristic function
# ---------------------------------------------------------------------------

class TestCGMYCharacteristic:
    """Tests for cgmy_characteristic."""

    def test_phi_zero_is_one(self) -> None:
        """phi(0) should be 1."""
        char_fn = cgmy_characteristic(1.0, 5.0, 10.0, 0.5, 0.05, 1.0, spot=100)
        val = char_fn(np.array([0.0]))
        assert np.abs(val[0]) == pytest.approx(1.0, abs=1e-10)

    def test_cgmy_price_positive(self) -> None:
        """CGMY char fn should produce positive call price."""
        char_fn = cgmy_characteristic(1.0, 5.0, 10.0, 0.5, 0.05, 1.0, spot=100)
        price = characteristic_function_price(char_fn, 100, 100, 0.05, 1.0)
        assert price > 0.0

    def test_cgmy_price_finite(self) -> None:
        """CGMY price should be finite."""
        char_fn = cgmy_characteristic(1.0, 5.0, 10.0, 0.5, 0.05, 1.0, spot=100)
        price = characteristic_function_price(char_fn, 100, 100, 0.05, 1.0)
        assert np.isfinite(price)

    def test_cgmy_itm_higher_than_otm(self) -> None:
        """ITM call more expensive than OTM."""
        char_fn = cgmy_characteristic(1.0, 5.0, 10.0, 0.5, 0.05, 1.0, spot=100)
        p_itm = characteristic_function_price(char_fn, 100, 80, 0.05, 1.0)
        p_otm = characteristic_function_price(char_fn, 100, 120, 0.05, 1.0)
        assert p_itm > p_otm
