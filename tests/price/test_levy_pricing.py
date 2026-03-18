"""Tests for wraquant.price.levy_pricing."""

from __future__ import annotations

import numpy as np
import pytest

from wraquant.price.levy_pricing import (
    cos_method,
    fft_option_price,
    nig_european_fft,
    vg_european_fft,
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
    u: np.ndarray,
    log_spot: float,
    r: float,
    T: float,
    sigma: float,
) -> np.ndarray:
    """Characteristic function of log(S_T) under GBM (for BS)."""
    drift = (r - 0.5 * sigma ** 2) * T
    return np.exp(
        1j * u * (log_spot + drift) - 0.5 * sigma ** 2 * T * u ** 2
    )


# ---------------------------------------------------------------------------
# FFT option pricing (generic)
# ---------------------------------------------------------------------------

class TestFFTOptionPrice:
    """Tests for fft_option_price."""

    def test_bs_call_via_fft(self) -> None:
        """FFT with BS char fn should match BS formula."""
        S, K, r, T, sigma = 100.0, 100.0, 0.05, 1.0, 0.2
        bs_price = _bs_call(S, K, r, T, sigma)

        log_spot = np.log(S)

        def char_fn(u: np.ndarray) -> np.ndarray:
            return _bs_char_fn(u, log_spot, r, T, sigma)

        fft_price = fft_option_price(char_fn, S, K, r, T, n_fft=4096)
        assert fft_price == pytest.approx(bs_price, abs=0.3)

    def test_otm_call_near_zero(self) -> None:
        """Deep OTM call should be near zero."""
        S, K, r, T, sigma = 100.0, 200.0, 0.05, 0.1, 0.2
        log_spot = np.log(S)

        def char_fn(u: np.ndarray) -> np.ndarray:
            return _bs_char_fn(u, log_spot, r, T, sigma)

        price = fft_option_price(char_fn, S, K, r, T)
        assert price < 1.0

    def test_call_nonnegative(self) -> None:
        S, K, r, T, sigma = 100.0, 110.0, 0.05, 1.0, 0.2
        log_spot = np.log(S)

        def char_fn(u: np.ndarray) -> np.ndarray:
            return _bs_char_fn(u, log_spot, r, T, sigma)

        price = fft_option_price(char_fn, S, K, r, T)
        assert price >= 0.0

    def test_multiple_strikes(self) -> None:
        """Should handle array of strikes."""
        S, r, T, sigma = 100.0, 0.05, 1.0, 0.2
        strikes = np.array([90.0, 100.0, 110.0])
        log_spot = np.log(S)

        def char_fn(u: np.ndarray) -> np.ndarray:
            return _bs_char_fn(u, log_spot, r, T, sigma)

        prices = fft_option_price(char_fn, S, strikes, r, T)
        assert len(prices) == 3
        # Prices should decrease with strike
        assert prices[0] > prices[1] > prices[2]


# ---------------------------------------------------------------------------
# VG European FFT
# ---------------------------------------------------------------------------

class TestVGEuropeanFFT:
    """Tests for vg_european_fft."""

    def test_price_positive(self) -> None:
        price = vg_european_fft(
            spot=100, strike=100, rf_rate=0.05, T=1.0,
            sigma=0.2, nu=0.5, theta=-0.1,
        )
        assert price > 0.0

    def test_price_finite(self) -> None:
        price = vg_european_fft(
            spot=100, strike=100, rf_rate=0.05, T=1.0,
            sigma=0.2, nu=0.5, theta=-0.1,
        )
        assert np.isfinite(price)

    def test_itm_higher_than_otm(self) -> None:
        """ITM call (low strike) should cost more than OTM (high strike)."""
        price_itm = vg_european_fft(
            spot=100, strike=80, rf_rate=0.05, T=1.0,
            sigma=0.2, nu=0.5, theta=-0.1,
        )
        price_otm = vg_european_fft(
            spot=100, strike=120, rf_rate=0.05, T=1.0,
            sigma=0.2, nu=0.5, theta=-0.1,
        )
        assert price_itm > price_otm


# ---------------------------------------------------------------------------
# NIG European FFT
# ---------------------------------------------------------------------------

class TestNIGEuropeanFFT:
    """Tests for nig_european_fft."""

    def test_price_positive(self) -> None:
        price = nig_european_fft(
            spot=100, strike=100, rf_rate=0.05, T=1.0,
            alpha=15.0, beta=-3.0, mu=0.0, delta=0.5,
        )
        assert price > 0.0

    def test_price_finite(self) -> None:
        price = nig_european_fft(
            spot=100, strike=100, rf_rate=0.05, T=1.0,
            alpha=15.0, beta=-3.0, mu=0.0, delta=0.5,
        )
        assert np.isfinite(price)


# ---------------------------------------------------------------------------
# COS method
# ---------------------------------------------------------------------------

class TestCOSMethod:
    """Tests for cos_method."""

    def test_bs_call_via_cos(self) -> None:
        """COS method with BS char fn should match BS formula."""
        S, K, r, T, sigma = 100.0, 100.0, 0.05, 1.0, 0.2
        bs_price = _bs_call(S, K, r, T, sigma)
        log_spot = np.log(S)

        def char_fn(u: np.ndarray) -> np.ndarray:
            return _bs_char_fn(u, log_spot, r, T, sigma)

        cos_price = cos_method(char_fn, S, K, r, T, n_terms=256)
        assert cos_price == pytest.approx(bs_price, abs=0.5)

    def test_price_nonnegative(self) -> None:
        S, K, r, T, sigma = 100.0, 110.0, 0.05, 1.0, 0.2
        log_spot = np.log(S)

        def char_fn(u: np.ndarray) -> np.ndarray:
            return _bs_char_fn(u, log_spot, r, T, sigma)

        price = cos_method(char_fn, S, K, r, T)
        assert price >= 0.0

    def test_itm_higher_than_otm(self) -> None:
        S, r, T, sigma = 100.0, 0.05, 1.0, 0.2
        log_spot = np.log(S)

        def char_fn(u: np.ndarray) -> np.ndarray:
            return _bs_char_fn(u, log_spot, r, T, sigma)

        price_itm = cos_method(char_fn, S, 80, r, T)
        price_otm = cos_method(char_fn, S, 120, r, T)
        assert price_itm > price_otm
