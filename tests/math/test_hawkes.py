"""Tests for wraquant.math.hawkes."""

from __future__ import annotations

import numpy as np
import pytest

from wraquant.math.hawkes import (
    fit_hawkes,
    hawkes_branching_ratio,
    hawkes_intensity,
    simulate_hawkes,
)


class TestSimulateHawkes:
    """Tests for simulate_hawkes."""

    def test_events_in_range(self) -> None:
        """All simulated event times should be in [0, T]."""
        T = 100.0
        events = simulate_hawkes(mu=1.0, alpha=0.5, beta=1.5, T=T, seed=42)
        assert len(events) > 0
        assert events.min() >= 0.0
        assert events.max() <= T

    def test_deterministic_with_seed(self) -> None:
        """Same seed should produce identical results."""
        e1 = simulate_hawkes(mu=1.0, alpha=0.5, beta=1.5, T=50.0, seed=99)
        e2 = simulate_hawkes(mu=1.0, alpha=0.5, beta=1.5, T=50.0, seed=99)
        np.testing.assert_array_equal(e1, e2)

    def test_non_stationary_raises(self) -> None:
        """Branching ratio >= 1 should raise ValueError."""
        with pytest.raises(ValueError, match="non-stationary"):
            simulate_hawkes(mu=1.0, alpha=2.0, beta=1.0, T=10.0)


class TestHawkesBranchingRatio:
    """Tests for hawkes_branching_ratio."""

    def test_simple_calculation(self) -> None:
        assert hawkes_branching_ratio(0.5, 1.0) == pytest.approx(0.5)
        assert hawkes_branching_ratio(3.0, 4.0) == pytest.approx(0.75)

    def test_stationarity_condition(self) -> None:
        """Stationary processes have branching ratio < 1."""
        ratio = hawkes_branching_ratio(0.3, 1.0)
        assert ratio < 1.0


class TestHawkesIntensity:
    """Tests for hawkes_intensity."""

    def test_intensity_at_least_mu(self) -> None:
        """Intensity should always be >= mu (the base rate)."""
        mu, alpha, beta = 0.5, 0.8, 2.0
        events = simulate_hawkes(mu=mu, alpha=alpha, beta=beta, T=50.0, seed=7)
        if len(events) > 1:
            intensity = hawkes_intensity(events, mu=mu, alpha=alpha, beta=beta)
            assert np.all(intensity >= mu - 1e-10)

    def test_single_event(self) -> None:
        """With one event the intensity is exactly mu."""
        intensity = hawkes_intensity(np.array([1.0]), mu=2.0, alpha=0.5, beta=1.0)
        assert intensity[0] == pytest.approx(2.0)


class TestFitHawkes:
    """Tests for fit_hawkes."""

    def test_recovers_approximate_parameters(self) -> None:
        """MLE should recover parameters roughly from simulated data."""
        mu_true, alpha_true, beta_true = 1.0, 0.5, 1.5
        events = simulate_hawkes(
            mu=mu_true,
            alpha=alpha_true,
            beta=beta_true,
            T=500.0,
            seed=12345,
        )
        result = fit_hawkes(events, T=500.0)

        # Allow generous tolerances — MLE on a single realisation
        assert result["mu"] > 0
        assert result["alpha"] > 0
        assert result["beta"] > 0
        assert result["branching_ratio"] < 1.0
        # Check that the fitted branching ratio is in the right ballpark
        true_ratio = alpha_true / beta_true
        assert abs(result["branching_ratio"] - true_ratio) < 0.3

    def test_returns_expected_keys(self) -> None:
        events = simulate_hawkes(mu=1.0, alpha=0.4, beta=1.0, T=100.0, seed=0)
        result = fit_hawkes(events)
        assert "mu" in result
        assert "alpha" in result
        assert "beta" in result
        assert "log_likelihood" in result
        assert "branching_ratio" in result
