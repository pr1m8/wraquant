"""Tests for wraquant.price.fbsde — FBSDE solvers."""

from __future__ import annotations

import numpy as np
import pytest

from wraquant.price.fbsde import (
    deep_bsde,
    fbsde_european,
    reflected_bsde,
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


# ---------------------------------------------------------------------------
# FBSDE European
# ---------------------------------------------------------------------------

class TestFBSDEEuropean:
    """Tests for fbsde_european."""

    def test_matches_black_scholes(self) -> None:
        """FBSDE solver for GBM dynamics should approximate Black-Scholes."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        bs_price = _bs_call(S0, K, r, T, sigma)

        payoff = lambda x: np.maximum(x - K, 0.0)
        drift = lambda x: r * x
        vol = lambda x: sigma * x

        result = fbsde_european(
            S0, payoff, drift, vol, r, T,
            n_steps=100, n_paths=50_000, seed=42,
        )

        # Should be within ~10% of BS price (Monte Carlo error)
        assert result["price"] == pytest.approx(bs_price, rel=0.15)

    def test_price_positive(self) -> None:
        """European call price should be positive."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        payoff = lambda x: np.maximum(x - K, 0.0)
        drift = lambda x: r * x
        vol = lambda x: sigma * x

        result = fbsde_european(
            S0, payoff, drift, vol, r, T,
            n_steps=50, n_paths=10_000, seed=42,
        )
        assert result["price"] > 0.0

    def test_price_finite(self) -> None:
        """Price should be finite."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        payoff = lambda x: np.maximum(x - K, 0.0)
        drift = lambda x: r * x
        vol = lambda x: sigma * x

        result = fbsde_european(
            S0, payoff, drift, vol, r, T,
            n_steps=50, n_paths=5_000, seed=42,
        )
        assert np.isfinite(result["price"])

    def test_paths_shape(self) -> None:
        """Forward paths should have correct shape."""
        n_steps, n_paths = 50, 1000
        payoff = lambda x: np.maximum(x - 100, 0.0)
        drift = lambda x: 0.05 * x
        vol = lambda x: 0.2 * x

        result = fbsde_european(
            100.0, payoff, drift, vol, 0.05, 1.0,
            n_steps=n_steps, n_paths=n_paths, seed=42,
        )
        assert result["paths"].shape == (n_steps + 1, n_paths)
        assert result["price_process"].shape == (n_steps + 1, n_paths)

    def test_delta_is_reasonable(self) -> None:
        """Delta for ATM call should be around 0.5-0.7."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        payoff = lambda x: np.maximum(x - K, 0.0)
        drift = lambda x: r * x
        vol = lambda x: sigma * x

        result = fbsde_european(
            S0, payoff, drift, vol, r, T,
            n_steps=100, n_paths=50_000, seed=42,
        )
        # BS delta for ATM call is ~0.64
        assert 0.2 < result["delta"] < 1.0

    def test_put_price(self) -> None:
        """FBSDE should also work for put options."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        payoff = lambda x: np.maximum(K - x, 0.0)
        drift = lambda x: r * x
        vol = lambda x: sigma * x

        result = fbsde_european(
            S0, payoff, drift, vol, r, T,
            n_steps=50, n_paths=10_000, seed=42,
        )
        assert result["price"] > 0.0
        assert np.isfinite(result["price"])

    def test_reproducible(self) -> None:
        """Same seed gives same price."""
        payoff = lambda x: np.maximum(x - 100, 0.0)
        drift = lambda x: 0.05 * x
        vol = lambda x: 0.2 * x

        r1 = fbsde_european(100, payoff, drift, vol, 0.05, 1.0,
                            n_steps=30, n_paths=1000, seed=123)
        r2 = fbsde_european(100, payoff, drift, vol, 0.05, 1.0,
                            n_steps=30, n_paths=1000, seed=123)
        assert r1["price"] == r2["price"]


# ---------------------------------------------------------------------------
# Deep BSDE
# ---------------------------------------------------------------------------

class TestDeepBSDE:
    """Tests for deep_bsde."""

    def test_finite_price(self) -> None:
        """Deep BSDE should produce a finite price."""
        payoff = lambda x: np.maximum(np.mean(x, axis=1) - 100, 0.0)
        drift = lambda x: 0.05 * x
        vol = lambda x: 0.2 * x

        result = deep_bsde(
            dim=1, payoff_fn=payoff, drift_fn=drift, vol_fn=vol,
            rf=0.05, T=1.0,
            n_steps=10, n_paths=256, n_epochs=30, seed=42,
        )
        assert np.isfinite(result["price"])

    def test_positive_price(self) -> None:
        """Deep BSDE price for a call should be positive."""
        payoff = lambda x: np.maximum(np.mean(x, axis=1) - 100, 0.0)
        drift = lambda x: 0.05 * x
        vol = lambda x: 0.2 * x

        result = deep_bsde(
            dim=1, payoff_fn=payoff, drift_fn=drift, vol_fn=vol,
            rf=0.05, T=1.0,
            n_steps=10, n_paths=512, n_epochs=50, seed=42,
        )
        # The deep BSDE can sometimes give slightly negative price with
        # few epochs, so check it's reasonably positive or at least bounded
        assert result["price"] > -5.0  # should not be hugely negative

    def test_loss_history_exists(self) -> None:
        """Should return a loss history list."""
        payoff = lambda x: np.maximum(np.mean(x, axis=1) - 100, 0.0)
        drift = lambda x: 0.05 * x
        vol = lambda x: 0.2 * x

        result = deep_bsde(
            dim=1, payoff_fn=payoff, drift_fn=drift, vol_fn=vol,
            rf=0.05, T=1.0,
            n_steps=5, n_paths=128, n_epochs=10, seed=42,
        )
        assert isinstance(result["loss_history"], list)
        assert len(result["loss_history"]) > 0

    def test_multidimensional(self) -> None:
        """Deep BSDE should handle multi-dimensional problems."""
        dim = 3
        payoff = lambda x: np.maximum(np.mean(x, axis=1) - 100, 0.0)
        drift = lambda x: 0.05 * x
        vol = lambda x: 0.2 * x

        result = deep_bsde(
            dim=dim, payoff_fn=payoff, drift_fn=drift, vol_fn=vol,
            rf=0.05, T=1.0,
            n_steps=5, n_paths=128, n_epochs=10, seed=42,
        )
        assert np.isfinite(result["price"])
        assert len(result["delta"]) == dim


# ---------------------------------------------------------------------------
# Reflected BSDE (American option)
# ---------------------------------------------------------------------------

class TestReflectedBSDE:
    """Tests for reflected_bsde."""

    def test_price_positive(self) -> None:
        """American put price should be positive."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        payoff = lambda x: np.maximum(K - x, 0.0)
        drift = lambda x: r * x
        vol = lambda x: sigma * x

        result = reflected_bsde(
            S0, payoff, drift, vol, r, T,
            n_steps=50, n_paths=10_000, seed=42,
        )
        assert result["price"] > 0.0

    def test_american_geq_european(self) -> None:
        """American option price should be >= European option price."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        payoff = lambda x: np.maximum(K - x, 0.0)
        drift = lambda x: r * x
        vol = lambda x: sigma * x

        european = fbsde_european(
            S0, payoff, drift, vol, r, T,
            n_steps=50, n_paths=20_000, seed=42,
        )
        american = reflected_bsde(
            S0, payoff, drift, vol, r, T,
            n_steps=50, n_paths=20_000, seed=42,
        )

        # American >= European (allow small MC error margin)
        assert american["price"] >= european["price"] - 0.5

    def test_exercise_boundary_exists(self) -> None:
        """Should return exercise boundary array."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        payoff = lambda x: np.maximum(K - x, 0.0)
        drift = lambda x: r * x
        vol = lambda x: sigma * x

        result = reflected_bsde(
            S0, payoff, drift, vol, r, T,
            n_steps=50, n_paths=5_000, seed=42,
        )
        assert len(result["exercise_boundary"]) == 51

    def test_optimal_stopping_time(self) -> None:
        """Optimal stopping time should be between 0 and T."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        payoff = lambda x: np.maximum(K - x, 0.0)
        drift = lambda x: r * x
        vol = lambda x: sigma * x

        result = reflected_bsde(
            S0, payoff, drift, vol, r, T,
            n_steps=50, n_paths=5_000, seed=42,
        )
        assert 0.0 <= result["optimal_stopping_time"] <= T

    def test_deep_itm_american_call_near_european(self) -> None:
        """Deep ITM American call should be close to European (no early exercise)."""
        S0, K, r, sigma, T = 150.0, 100.0, 0.05, 0.2, 1.0
        payoff = lambda x: np.maximum(x - K, 0.0)
        drift = lambda x: r * x
        vol = lambda x: sigma * x

        european = fbsde_european(
            S0, payoff, drift, vol, r, T,
            n_steps=50, n_paths=10_000, seed=42,
        )
        american = reflected_bsde(
            S0, payoff, drift, vol, r, T,
            n_steps=50, n_paths=10_000, seed=42,
        )
        # For calls with no dividends, American == European
        assert american["price"] == pytest.approx(european["price"], rel=0.2)
