"""Tests for wraquant.math.numerical."""

from __future__ import annotations

import numpy as np
import pytest

from wraquant.math.numerical import (
    bisection,
    finite_difference_gradient,
    finite_difference_hessian,
    monte_carlo_integration,
    newton_raphson,
    trapezoidal_integration,
)


class TestNewtonRaphson:
    """Tests for newton_raphson."""

    def test_simple_polynomial_root(self) -> None:
        """x^2 - 4 = 0 should give root at x = 2 (starting from x0 = 3)."""
        root = newton_raphson(lambda x: x**2 - 4, x0=3.0)
        assert root == pytest.approx(2.0, abs=1e-7)

    def test_with_analytical_derivative(self) -> None:
        root = newton_raphson(
            lambda x: x**3 - 1,
            x0=2.0,
            dfn=lambda x: 3 * x**2,
        )
        assert root == pytest.approx(1.0, abs=1e-7)

    def test_cos_root(self) -> None:
        """Find root of cos(x) near pi/2."""
        root = newton_raphson(np.cos, x0=1.0)
        assert root == pytest.approx(np.pi / 2, abs=1e-6)


class TestBisection:
    """Tests for bisection."""

    def test_finds_root(self) -> None:
        """x^2 - 2 on [0, 2] should find sqrt(2)."""
        root = bisection(lambda x: x**2 - 2, a=0.0, b=2.0)
        assert root == pytest.approx(np.sqrt(2), abs=1e-7)

    def test_linear_root(self) -> None:
        root = bisection(lambda x: 2 * x - 6, a=0.0, b=10.0)
        assert root == pytest.approx(3.0, abs=1e-7)

    def test_raises_on_same_sign(self) -> None:
        with pytest.raises(ValueError, match="opposite signs"):
            bisection(lambda x: x**2 + 1, a=0.0, b=2.0)


class TestFiniteDifferenceGradient:
    """Tests for finite_difference_gradient."""

    def test_matches_analytical(self) -> None:
        """Gradient of f(x,y) = x^2 + 3*y at (2, 1) is (4, 3)."""

        def f(x: np.ndarray) -> float:
            return float(x[0] ** 2 + 3 * x[1])

        grad = finite_difference_gradient(f, np.array([2.0, 1.0]))
        np.testing.assert_allclose(grad, [4.0, 3.0], atol=1e-5)

    def test_1d(self) -> None:
        """Gradient of x^3 at x=2 is 12."""

        def f(x: np.ndarray) -> float:
            return float(x[0] ** 3)

        grad = finite_difference_gradient(f, np.array([2.0]))
        assert grad[0] == pytest.approx(12.0, abs=1e-4)


class TestFiniteDifferenceHessian:
    """Tests for finite_difference_hessian."""

    def test_quadratic(self) -> None:
        """Hessian of x^2 + 2*y^2 + x*y should be [[2, 1], [1, 4]]."""

        def f(x: np.ndarray) -> float:
            return float(x[0] ** 2 + 2 * x[1] ** 2 + x[0] * x[1])

        hess = finite_difference_hessian(f, np.array([1.0, 1.0]))
        np.testing.assert_allclose(hess, [[2.0, 1.0], [1.0, 4.0]], atol=1e-4)


class TestTrapezoidalIntegration:
    """Tests for trapezoidal_integration."""

    def test_x_squared(self) -> None:
        """Integral of x^2 from 0 to 1 should be 1/3."""
        result = trapezoidal_integration(lambda x: x**2, 0.0, 1.0, n=10_000)
        assert result == pytest.approx(1.0 / 3.0, abs=1e-6)

    def test_constant(self) -> None:
        """Integral of 5 from 0 to 3 is 15."""
        result = trapezoidal_integration(lambda x: 5.0, 0.0, 3.0)
        assert result == pytest.approx(15.0, abs=1e-10)


class TestMonteCarloIntegration:
    """Tests for monte_carlo_integration."""

    def test_1d_integral(self) -> None:
        """Integral of x^2 from 0 to 1."""
        result = monte_carlo_integration(
            lambda x: x[0] ** 2,
            bounds=[(0.0, 1.0)],
            n_samples=200_000,
            seed=42,
        )
        assert result["estimate"] == pytest.approx(1.0 / 3.0, abs=0.01)
        assert result["std_error"] > 0

    def test_2d_integral(self) -> None:
        """Integral of 1 over the unit square = 1."""
        result = monte_carlo_integration(
            lambda x: 1.0,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            n_samples=10_000,
            seed=0,
        )
        assert result["estimate"] == pytest.approx(1.0, abs=0.05)
