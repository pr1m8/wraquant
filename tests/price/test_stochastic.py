"""Tests for stochastic process simulations."""

from __future__ import annotations

import numpy as np
import pytest

from wraquant.price.stochastic import (
    cir_process,
    geometric_brownian_motion,
    heston,
    jump_diffusion,
    ornstein_uhlenbeck,
)


class TestGBM:
    """Tests for geometric Brownian motion."""

    def test_output_shape(self) -> None:
        """Output has shape (n_paths, n_steps + 1)."""
        paths = geometric_brownian_motion(100, 0.05, 0.2, 1.0, 252, 1000, seed=42)
        assert paths.shape == (1000, 253)

    def test_initial_value(self) -> None:
        """First column is the initial price."""
        paths = geometric_brownian_motion(100, 0.05, 0.2, 1.0, 252, 500, seed=42)
        np.testing.assert_array_equal(paths[:, 0], 100.0)

    def test_mean_converges(self) -> None:
        """Mean of terminal values converges to S0 * exp(mu * T)."""
        S0, mu, T = 100.0, 0.05, 1.0
        paths = geometric_brownian_motion(S0, mu, 0.2, T, 252, 50_000, seed=42)
        terminal_mean = np.mean(paths[:, -1])
        expected = S0 * np.exp(mu * T)
        assert terminal_mean == pytest.approx(expected, rel=0.02)

    def test_positive_prices(self) -> None:
        """GBM prices are always positive."""
        paths = geometric_brownian_motion(100, 0.05, 0.2, 1.0, 252, 1000, seed=42)
        assert np.all(paths > 0)

    def test_reproducible(self) -> None:
        """Same seed gives same paths."""
        p1 = geometric_brownian_motion(100, 0.05, 0.2, 1.0, 100, 10, seed=99)
        p2 = geometric_brownian_motion(100, 0.05, 0.2, 1.0, 100, 10, seed=99)
        np.testing.assert_array_equal(p1, p2)


class TestHeston:
    """Tests for the Heston stochastic volatility model."""

    def test_output_shapes(self) -> None:
        """Returns two arrays each with shape (n_paths, n_steps + 1)."""
        prices, vols = heston(
            100, 0.04, 0.05, 2.0, 0.04, 0.3, -0.7, 1.0, 252, 500, seed=42
        )
        assert prices.shape == (500, 253)
        assert vols.shape == (500, 253)

    def test_initial_values(self) -> None:
        """First column matches initial values."""
        prices, vols = heston(
            100, 0.04, 0.05, 2.0, 0.04, 0.3, -0.7, 1.0, 100, 100, seed=42
        )
        np.testing.assert_array_equal(prices[:, 0], 100.0)
        np.testing.assert_array_equal(vols[:, 0], 0.04)

    def test_prices_positive(self) -> None:
        """Heston price paths are positive."""
        prices, _ = heston(
            100, 0.04, 0.05, 2.0, 0.04, 0.3, -0.7, 1.0, 252, 500, seed=42
        )
        assert np.all(prices > 0)


class TestJumpDiffusion:
    """Tests for Merton jump-diffusion."""

    def test_output_shape(self) -> None:
        """Output has correct shape."""
        paths = jump_diffusion(100, 0.05, 0.2, 1.0, -0.1, 0.15, 1.0, 252, 500, seed=42)
        assert paths.shape == (500, 253)

    def test_initial_value(self) -> None:
        """First column is the initial price."""
        paths = jump_diffusion(100, 0.05, 0.2, 1.0, -0.1, 0.15, 1.0, 100, 100, seed=42)
        np.testing.assert_array_equal(paths[:, 0], 100.0)

    def test_positive_prices(self) -> None:
        """Jump-diffusion prices stay positive."""
        paths = jump_diffusion(100, 0.05, 0.2, 1.0, -0.1, 0.15, 1.0, 252, 500, seed=42)
        assert np.all(paths > 0)


class TestOrnsteinUhlenbeck:
    """Tests for Ornstein-Uhlenbeck process."""

    def test_output_shape(self) -> None:
        """Output has correct shape."""
        paths = ornstein_uhlenbeck(0.05, 5.0, 0.03, 0.01, 1.0, 252, 500, seed=42)
        assert paths.shape == (500, 253)

    def test_initial_value(self) -> None:
        """First column is the initial value."""
        paths = ornstein_uhlenbeck(0.05, 5.0, 0.03, 0.01, 1.0, 100, 100, seed=42)
        np.testing.assert_array_equal(paths[:, 0], 0.05)

    def test_mean_reversion(self) -> None:
        """Terminal mean should be near the long-run mean mu."""
        mu = 0.03
        # High theta for fast reversion, long T
        paths = ornstein_uhlenbeck(0.10, 10.0, mu, 0.005, 5.0, 1000, 10_000, seed=42)
        terminal_mean = np.mean(paths[:, -1])
        assert terminal_mean == pytest.approx(mu, abs=0.002)

    def test_starting_above_reverts_down(self) -> None:
        """When x0 > mu, paths tend downward."""
        mu = 0.03
        paths = ornstein_uhlenbeck(0.10, 5.0, mu, 0.01, 2.0, 500, 5000, seed=42)
        mean_terminal = np.mean(paths[:, -1])
        assert mean_terminal < 0.10  # Reverted toward mu


class TestCIR:
    """Tests for Cox-Ingersoll-Ross process."""

    def test_output_shape(self) -> None:
        """Output has correct shape."""
        paths = cir_process(0.04, 2.0, 0.04, 0.1, 1.0, 252, 500, seed=42)
        assert paths.shape == (500, 253)

    def test_initial_value(self) -> None:
        """First column is the initial value."""
        paths = cir_process(0.04, 2.0, 0.04, 0.1, 1.0, 100, 100, seed=42)
        np.testing.assert_array_equal(paths[:, 0], 0.04)

    def test_stays_positive_feller(self) -> None:
        """CIR stays positive when Feller condition is satisfied (2*kappa*theta >= sigma^2)."""
        kappa, theta_val, sigma = 2.0, 0.04, 0.1
        # Feller: 2*2*0.04 = 0.16 >= 0.01 = 0.1^2 -> satisfied
        assert 2 * kappa * theta_val >= sigma**2
        paths = cir_process(0.04, kappa, theta_val, sigma, 1.0, 1000, 1000, seed=42)
        assert np.all(paths >= 0.0)

    def test_mean_reversion(self) -> None:
        """Terminal mean reverts toward theta."""
        theta_val = 0.06
        paths = cir_process(0.02, 5.0, theta_val, 0.05, 5.0, 1000, 10_000, seed=42)
        terminal_mean = np.mean(paths[:, -1])
        assert terminal_mean == pytest.approx(theta_val, abs=0.005)
