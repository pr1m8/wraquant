"""Tests for wraquant.math.levy."""

from __future__ import annotations

import numpy as np
import pytest

from wraquant.math.levy import (
    cgmy_simulate,
    characteristic_function_vg,
    fit_nig,
    fit_variance_gamma,
    levy_stable_simulate,
    nig_pdf,
    nig_simulate,
    variance_gamma_pdf,
    variance_gamma_simulate,
)


# ---------------------------------------------------------------------------
# Variance Gamma
# ---------------------------------------------------------------------------

class TestVarianceGammaPDF:
    """Tests for variance_gamma_pdf."""

    def test_pdf_nonnegative(self) -> None:
        x = np.linspace(-1, 1, 200)
        pdf = variance_gamma_pdf(x, sigma=0.2, nu=0.5, theta=0.0)
        assert np.all(pdf >= 0.0)

    def test_pdf_integrates_near_one(self) -> None:
        """Numerical integration of PDF should be close to 1."""
        x = np.linspace(-5, 5, 5000)
        dx = x[1] - x[0]
        pdf = variance_gamma_pdf(x, sigma=0.2, nu=0.5, theta=0.0)
        integral = np.trapezoid(pdf, x)
        assert integral == pytest.approx(1.0, abs=0.05)

    def test_pdf_symmetric_when_theta_zero(self) -> None:
        """With theta=0, VG is symmetric around 0."""
        x = np.linspace(0.01, 3, 100)
        pdf_pos = variance_gamma_pdf(x, sigma=0.2, nu=0.5, theta=0.0)
        pdf_neg = variance_gamma_pdf(-x, sigma=0.2, nu=0.5, theta=0.0)
        np.testing.assert_allclose(pdf_pos, pdf_neg, rtol=0.05)


class TestVarianceGammaSimulate:
    """Tests for variance_gamma_simulate."""

    def test_starts_at_zero(self) -> None:
        path = variance_gamma_simulate(0.2, 0.5, 0.0, 1000, seed=42)
        assert path[0] == 0.0

    def test_correct_length(self) -> None:
        path = variance_gamma_simulate(0.2, 0.5, 0.0, 500, seed=42)
        assert len(path) == 501

    def test_reproducible(self) -> None:
        p1 = variance_gamma_simulate(0.2, 0.5, 0.0, 100, seed=42)
        p2 = variance_gamma_simulate(0.2, 0.5, 0.0, 100, seed=42)
        np.testing.assert_array_equal(p1, p2)


class TestCharacteristicFunctionVG:
    """Tests for characteristic_function_vg."""

    def test_cf_at_zero_is_one(self) -> None:
        cf = characteristic_function_vg(np.array([0.0]), 0.2, 0.5, 0.1)
        np.testing.assert_allclose(np.abs(cf), 1.0, atol=1e-12)

    def test_cf_values_finite(self) -> None:
        u = np.linspace(-10, 10, 100)
        cf = characteristic_function_vg(u, 0.2, 0.5, 0.0)
        assert np.all(np.isfinite(cf))


# ---------------------------------------------------------------------------
# NIG
# ---------------------------------------------------------------------------

class TestNIGPDF:
    """Tests for nig_pdf."""

    def test_pdf_nonnegative(self) -> None:
        x = np.linspace(-3, 3, 200)
        pdf = nig_pdf(x, alpha=1.5, beta=0.0, mu=0.0, delta=1.0)
        assert np.all(pdf >= 0.0)

    def test_pdf_integrates_near_one(self) -> None:
        x = np.linspace(-10, 10, 5000)
        pdf = nig_pdf(x, alpha=1.5, beta=0.0, mu=0.0, delta=1.0)
        integral = np.trapezoid(pdf, x)
        assert integral == pytest.approx(1.0, abs=0.05)


class TestNIGSimulate:
    """Tests for nig_simulate."""

    def test_starts_at_zero(self) -> None:
        path = nig_simulate(1.5, 0.0, 0.0, 1.0, 500, seed=42)
        assert path[0] == 0.0

    def test_correct_length(self) -> None:
        path = nig_simulate(1.5, 0.0, 0.0, 1.0, 500, seed=42)
        assert len(path) == 501


# ---------------------------------------------------------------------------
# CGMY
# ---------------------------------------------------------------------------

class TestCGMYSimulate:
    """Tests for cgmy_simulate."""

    def test_starts_at_zero_finite_activity(self) -> None:
        path = cgmy_simulate(C=1.0, G=5.0, M=5.0, Y=-0.5, n_steps=200, seed=42)
        assert path[0] == 0.0

    def test_starts_at_zero_infinite_activity(self) -> None:
        path = cgmy_simulate(C=1.0, G=5.0, M=5.0, Y=0.5, n_steps=200, seed=42)
        assert path[0] == 0.0

    def test_correct_length(self) -> None:
        path = cgmy_simulate(C=1.0, G=5.0, M=5.0, Y=0.5, n_steps=300, seed=42)
        assert len(path) == 301


# ---------------------------------------------------------------------------
# Stable Lévy
# ---------------------------------------------------------------------------

class TestLevyStableSimulate:
    """Tests for levy_stable_simulate."""

    def test_starts_at_zero(self) -> None:
        path = levy_stable_simulate(1.5, 0.0, 500, seed=42)
        assert path[0] == 0.0

    def test_correct_length(self) -> None:
        path = levy_stable_simulate(1.5, 0.0, 300, seed=42)
        assert len(path) == 301

    def test_gaussian_when_alpha_2(self) -> None:
        """alpha=2 gives Gaussian (finite variance)."""
        path = levy_stable_simulate(2.0, 0.0, 10000, seed=42)
        increments = np.diff(path)
        # Should have finite variance and be approximately normal
        assert np.isfinite(np.var(increments))


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

class TestFitVarianceGamma:
    """Tests for fit_variance_gamma."""

    def test_returns_correct_keys(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500) * 0.02
        result = fit_variance_gamma(data)
        assert "sigma" in result
        assert "nu" in result
        assert "theta" in result
        assert "log_likelihood" in result

    def test_sigma_positive(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500) * 0.02
        result = fit_variance_gamma(data)
        assert result["sigma"] > 0


class TestFitNIG:
    """Tests for fit_nig."""

    def test_returns_correct_keys(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500) * 0.02
        result = fit_nig(data)
        assert "alpha" in result
        assert "beta" in result
        assert "mu" in result
        assert "delta" in result
        assert "log_likelihood" in result

    def test_alpha_positive(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500) * 0.02
        result = fit_nig(data)
        assert result["alpha"] > 0
        assert result["delta"] > 0
