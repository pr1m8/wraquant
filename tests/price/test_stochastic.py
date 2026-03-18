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
    simulate_3_2_model,
    simulate_cir,
    simulate_rough_bergomi,
    simulate_sabr,
    simulate_vasicek,
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


# ---------------------------------------------------------------------------
# SABR model
# ---------------------------------------------------------------------------

class TestSimulateSABR:
    """Tests for simulate_sabr."""

    def test_output_shapes(self) -> None:
        """Returns dict with forwards and vols of correct shape."""
        result = simulate_sabr(0.05, 0.3, 0.4, 0.5, -0.3, 1.0, 252, 500, seed=42)
        assert result["forwards"].shape == (253, 500)
        assert result["vols"].shape == (253, 500)

    def test_initial_values(self) -> None:
        """Initial values match parameters."""
        result = simulate_sabr(0.05, 0.3, 0.4, 0.5, -0.3, 1.0, 100, 100, seed=42)
        np.testing.assert_array_equal(result["forwards"][0], 0.05)
        np.testing.assert_array_equal(result["vols"][0], 0.3)

    def test_forwards_nonnegative(self) -> None:
        """Forward rates should remain non-negative."""
        result = simulate_sabr(0.05, 0.3, 0.4, 0.5, -0.3, 1.0, 252, 1000, seed=42)
        assert np.all(result["forwards"] >= 0.0)

    def test_vols_positive(self) -> None:
        """Stochastic vol should stay positive."""
        result = simulate_sabr(0.05, 0.3, 0.4, 0.5, -0.3, 1.0, 252, 1000, seed=42)
        assert np.all(result["vols"] > 0.0)

    def test_beta_zero_normal(self) -> None:
        """Beta=0 gives normal SABR — forwards can go negative without absorption."""
        result = simulate_sabr(0.05, 0.3, 0.4, 0.0, 0.0, 1.0, 100, 500, seed=42)
        assert result["forwards"].shape == (101, 500)

    def test_beta_one_lognormal(self) -> None:
        """Beta=1 gives lognormal SABR."""
        result = simulate_sabr(0.05, 0.3, 0.4, 1.0, -0.3, 1.0, 100, 500, seed=42)
        assert result["forwards"].shape == (101, 500)

    def test_reproducible(self) -> None:
        """Same seed gives same results."""
        r1 = simulate_sabr(0.05, 0.3, 0.4, 0.5, -0.3, 1.0, 50, 100, seed=99)
        r2 = simulate_sabr(0.05, 0.3, 0.4, 0.5, -0.3, 1.0, 50, 100, seed=99)
        np.testing.assert_array_equal(r1["forwards"], r2["forwards"])
        np.testing.assert_array_equal(r1["vols"], r2["vols"])


# ---------------------------------------------------------------------------
# Rough Bergomi model
# ---------------------------------------------------------------------------

class TestSimulateRoughBergomi:
    """Tests for simulate_rough_bergomi."""

    def test_output_shapes(self) -> None:
        """Returns prices and variances of correct shape."""
        result = simulate_rough_bergomi(
            100.0, 0.04, 1.9, 0.1, -0.7, 1.0, 50, 200, seed=42,
        )
        assert result["prices"].shape == (51, 200)
        assert result["variances"].shape == (51, 200)

    def test_initial_price(self) -> None:
        """Initial price matches spot."""
        result = simulate_rough_bergomi(
            100.0, 0.04, 1.9, 0.1, -0.7, 1.0, 50, 100, seed=42,
        )
        np.testing.assert_array_equal(result["prices"][0], 100.0)

    def test_initial_variance(self) -> None:
        """Initial variance matches xi."""
        result = simulate_rough_bergomi(
            100.0, 0.04, 1.9, 0.1, -0.7, 1.0, 50, 100, seed=42,
        )
        np.testing.assert_array_equal(result["variances"][0], 0.04)

    def test_prices_positive(self) -> None:
        """Simulated prices should be positive."""
        result = simulate_rough_bergomi(
            100.0, 0.04, 1.9, 0.1, -0.7, 1.0, 50, 500, seed=42,
        )
        assert np.all(result["prices"] > 0.0)

    def test_variances_positive(self) -> None:
        """Simulated variances should be positive."""
        result = simulate_rough_bergomi(
            100.0, 0.04, 1.9, 0.1, -0.7, 1.0, 50, 500, seed=42,
        )
        assert np.all(result["variances"] > 0.0)

    def test_rough_hurst(self) -> None:
        """Hurst parameter < 0.5 should work (rough regime)."""
        for H in [0.05, 0.1, 0.3, 0.45]:
            result = simulate_rough_bergomi(
                100.0, 0.04, 1.9, H, -0.7, 1.0, 30, 50, seed=42,
            )
            assert np.all(np.isfinite(result["prices"]))


# ---------------------------------------------------------------------------
# 3/2 stochastic volatility model
# ---------------------------------------------------------------------------

class TestSimulate32Model:
    """Tests for simulate_3_2_model."""

    def test_output_shapes(self) -> None:
        """Returns prices and variances of correct shape."""
        result = simulate_3_2_model(
            100.0, 0.04, 2.0, 0.04, 0.5, -0.7, 1.0, 252, 500, seed=42,
        )
        assert result["prices"].shape == (253, 500)
        assert result["variances"].shape == (253, 500)

    def test_initial_values(self) -> None:
        """Initial values match parameters."""
        result = simulate_3_2_model(
            100.0, 0.04, 2.0, 0.04, 0.5, -0.7, 1.0, 100, 100, seed=42,
        )
        np.testing.assert_array_equal(result["prices"][0], 100.0)
        np.testing.assert_array_equal(result["variances"][0], 0.04)

    def test_prices_positive(self) -> None:
        """Prices should be positive (log-Euler scheme)."""
        result = simulate_3_2_model(
            100.0, 0.04, 2.0, 0.04, 0.5, -0.7, 1.0, 252, 1000, seed=42,
        )
        assert np.all(result["prices"] > 0.0)

    def test_variances_positive(self) -> None:
        """Variances should stay positive."""
        result = simulate_3_2_model(
            100.0, 0.04, 2.0, 0.04, 0.5, -0.7, 1.0, 252, 1000, seed=42,
        )
        assert np.all(result["variances"] > 0.0)

    def test_reproducible(self) -> None:
        """Same seed gives same results."""
        r1 = simulate_3_2_model(100.0, 0.04, 2.0, 0.04, 0.5, -0.7, 1.0, 50, 100, seed=42)
        r2 = simulate_3_2_model(100.0, 0.04, 2.0, 0.04, 0.5, -0.7, 1.0, 50, 100, seed=42)
        np.testing.assert_array_equal(r1["prices"], r2["prices"])


# ---------------------------------------------------------------------------
# CIR model (enhanced)
# ---------------------------------------------------------------------------

class TestSimulateCIR:
    """Tests for simulate_cir (enhanced version with dict output)."""

    def test_output_shape(self) -> None:
        """Paths have correct shape."""
        result = simulate_cir(0.05, 0.5, 0.04, 0.1, 10.0, 252, 500, seed=42)
        assert result["paths"].shape == (253, 500)

    def test_initial_value(self) -> None:
        """First row is initial rate."""
        result = simulate_cir(0.05, 0.5, 0.04, 0.1, 10.0, 100, 100, seed=42)
        np.testing.assert_array_equal(result["paths"][0], 0.05)

    def test_stays_positive_under_feller(self) -> None:
        """CIR stays non-negative when Feller condition satisfied."""
        kappa, theta_val, sigma = 2.0, 0.04, 0.1
        result = simulate_cir(0.04, kappa, theta_val, sigma, 1.0, 1000, 1000, seed=42)
        assert result["params"]["feller_satisfied"]
        assert np.all(result["paths"] >= 0.0)

    def test_feller_condition_check(self) -> None:
        """Feller condition diagnostic is correct."""
        # Satisfied: 2*2*0.04 = 0.16 >= 0.01
        result = simulate_cir(0.04, 2.0, 0.04, 0.1, 1.0, 10, 10, seed=42)
        assert result["params"]["feller_satisfied"] is True
        assert result["params"]["feller_ratio"] == pytest.approx(16.0)

        # Not satisfied: 2*0.1*0.04 = 0.008 < 0.09
        result2 = simulate_cir(0.04, 0.1, 0.04, 0.3, 1.0, 10, 10, seed=42)
        assert result2["params"]["feller_satisfied"] is False

    def test_mean_reversion(self) -> None:
        """Terminal mean reverts toward theta."""
        theta_val = 0.06
        result = simulate_cir(0.02, 5.0, theta_val, 0.05, 5.0, 1000, 10_000, seed=42)
        terminal_mean = np.mean(result["paths"][-1])
        assert terminal_mean == pytest.approx(theta_val, abs=0.005)


# ---------------------------------------------------------------------------
# Vasicek model
# ---------------------------------------------------------------------------

class TestSimulateVasicek:
    """Tests for simulate_vasicek."""

    def test_output_shapes(self) -> None:
        """Returns paths, bond_prices, and yield_curve."""
        result = simulate_vasicek(0.05, 0.5, 0.04, 0.01, 10.0, 100, 500, seed=42)
        assert result["paths"].shape == (101, 500)
        assert len(result["bond_prices"]) == 101
        assert len(result["yield_curve"]) == 100

    def test_initial_rate(self) -> None:
        """First row matches initial rate."""
        result = simulate_vasicek(0.05, 0.5, 0.04, 0.01, 10.0, 100, 100, seed=42)
        np.testing.assert_array_equal(result["paths"][0], 0.05)

    def test_bond_prices_decreasing(self) -> None:
        """Bond prices should generally decrease with maturity (positive rates)."""
        result = simulate_vasicek(0.05, 0.5, 0.04, 0.01, 10.0, 100, 100, seed=42)
        bp = np.array(result["bond_prices"])
        # P(0,0) = 1, and P(0,T) < 1 for positive rates
        assert bp[0] == 1.0
        assert bp[-1] < 1.0

    def test_yield_curve_shape(self) -> None:
        """Yield curve should be positive for positive long-run mean."""
        result = simulate_vasicek(0.05, 0.5, 0.04, 0.01, 10.0, 100, 100, seed=42)
        yc = np.array(result["yield_curve"])
        assert np.all(yc > 0.0)
        assert np.all(np.isfinite(yc))

    def test_mean_reversion(self) -> None:
        """Terminal mean reverts toward theta."""
        theta_val = 0.04
        result = simulate_vasicek(0.08, 2.0, theta_val, 0.005, 10.0, 500, 10_000, seed=42)
        terminal_mean = np.mean(result["paths"][-1])
        assert terminal_mean == pytest.approx(theta_val, abs=0.005)

    def test_can_go_negative(self) -> None:
        """Vasicek rates can go negative (Gaussian model)."""
        # Start near zero with high vol -> should occasionally go negative
        result = simulate_vasicek(0.001, 0.1, 0.001, 0.05, 5.0, 500, 5_000, seed=42)
        assert np.any(result["paths"] < 0.0)

    def test_reproducible(self) -> None:
        """Same seed gives same results."""
        r1 = simulate_vasicek(0.05, 0.5, 0.04, 0.01, 1.0, 50, 100, seed=99)
        r2 = simulate_vasicek(0.05, 0.5, 0.04, 0.01, 1.0, 50, 100, seed=99)
        np.testing.assert_array_equal(r1["paths"], r2["paths"])
