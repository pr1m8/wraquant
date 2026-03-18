"""Tests for convex optimization wrappers."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from wraquant.opt.convex import minimize_quadratic, solve_qp


class TestMinimizeQuadratic:
    def test_simple_quadratic(self) -> None:
        """min 0.5*x'Ix + [-1,-1]'x  =>  x* = [1,1]."""
        Q = np.eye(2)
        c = np.array([-1.0, -1.0])
        result = minimize_quadratic(Q, c)
        assert result["success"]
        assert_allclose(result["x"], [1.0, 1.0], atol=1e-6)
        assert_allclose(result["objective"], -1.0, atol=1e-6)

    def test_with_bounds(self) -> None:
        """Bounds should clamp the solution."""
        Q = np.eye(2)
        c = np.array([-10.0, -10.0])
        bounds = [(0.0, 2.0), (0.0, 2.0)]
        result = minimize_quadratic(Q, c, bounds=bounds)
        assert result["success"]
        assert_allclose(result["x"], [2.0, 2.0], atol=1e-6)

    def test_with_equality_constraint(self) -> None:
        """x1 + x2 == 1 should shift the optimal point."""
        Q = np.eye(2)
        c = np.array([0.0, 0.0])
        A_eq = np.array([[1.0, 1.0]])
        b_eq = np.array([1.0])
        result = minimize_quadratic(Q, c, A_eq=A_eq, b_eq=b_eq)
        assert result["success"]
        assert_allclose(result["x"].sum(), 1.0, atol=1e-6)
        assert_allclose(result["x"], [0.5, 0.5], atol=1e-6)


class TestSolveQP:
    def test_scipy_with_bounds(self) -> None:
        Q = 2.0 * np.eye(3)
        c = np.array([-2.0, -4.0, -6.0])
        bounds = [(0.0, 5.0)] * 3
        result = solve_qp(Q, c, bounds=bounds, solver="scipy")
        assert result["success"]
        assert_allclose(result["x"], [1.0, 2.0, 3.0], atol=1e-3)

    def test_scipy_with_equality(self) -> None:
        """Weights sum to 1."""
        Q = np.eye(3)
        c = np.zeros(3)
        A_eq = np.ones((1, 3))
        b_eq = np.ones(1)
        result = solve_qp(Q, c, A_eq=A_eq, b_eq=b_eq, solver="scipy")
        assert result["success"]
        assert_allclose(result["x"].sum(), 1.0, atol=1e-6)

    def test_invalid_solver_raises(self) -> None:
        Q = np.eye(2)
        c = np.zeros(2)
        try:
            solve_qp(Q, c, solver="nonexistent")
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass
