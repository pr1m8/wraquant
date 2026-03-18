"""Tests for nonlinear optimization wrappers."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from wraquant.opt.nonlinear import global_minimize, minimize, root_find


def _rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function: min at (1, 1)."""
    return float(100.0 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)


class TestMinimize:
    def test_rosenbrock(self) -> None:
        x0 = np.array([0.0, 0.0])
        result = minimize(_rosenbrock, x0, method="L-BFGS-B")
        assert result["success"]
        assert_allclose(result["x"], [1.0, 1.0], atol=1e-4)
        assert result["fun"] < 1e-8

    def test_constrained(self) -> None:
        """min x^2 + y^2  s.t. x + y == 1."""

        def obj(x: np.ndarray) -> float:
            return float(x[0] ** 2 + x[1] ** 2)

        constraints = [{"type": "eq", "fun": lambda x: x[0] + x[1] - 1.0}]
        x0 = np.array([0.0, 0.0])
        result = minimize(obj, x0, constraints=constraints)
        assert result["success"]
        assert_allclose(result["x"], [0.5, 0.5], atol=1e-6)

    def test_bounded(self) -> None:
        """min -x  s.t. 0 <= x <= 3."""
        result = minimize(
            lambda x: -x[0],
            np.array([0.0]),
            method="L-BFGS-B",
            bounds=[(0.0, 3.0)],
        )
        assert result["success"]
        assert_allclose(result["x"], [3.0], atol=1e-6)


class TestGlobalMinimize:
    def test_differential_evolution(self) -> None:
        result = global_minimize(
            _rosenbrock,
            bounds=[(-5.0, 5.0), (-5.0, 5.0)],
            method="differential_evolution",
            seed=42,
        )
        assert result["success"]
        assert_allclose(result["x"], [1.0, 1.0], atol=1e-4)

    def test_dual_annealing(self) -> None:
        result = global_minimize(
            _rosenbrock,
            bounds=[(-5.0, 5.0), (-5.0, 5.0)],
            method="dual_annealing",
            seed=42,
        )
        assert_allclose(result["x"], [1.0, 1.0], atol=1e-3)

    def test_invalid_method_raises(self) -> None:
        try:
            global_minimize(
                _rosenbrock,
                bounds=[(-5.0, 5.0), (-5.0, 5.0)],
                method="nonexistent",
            )
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass


class TestRootFind:
    def test_polynomial_root(self) -> None:
        """Find root of x^2 - 4 = 0 near x=3 => x=2."""

        def f(x: np.ndarray) -> np.ndarray:
            return x**2 - 4.0

        result = root_find(f, np.array([3.0]))
        assert result["success"]
        assert_allclose(result["x"], [2.0], atol=1e-8)

    def test_system_of_equations(self) -> None:
        """x + y = 3, x - y = 1 => (2, 1)."""

        def system(x: np.ndarray) -> np.ndarray:
            return np.array([x[0] + x[1] - 3.0, x[0] - x[1] - 1.0])

        result = root_find(system, np.array([0.0, 0.0]))
        assert result["success"]
        assert_allclose(result["x"], [2.0, 1.0], atol=1e-8)
