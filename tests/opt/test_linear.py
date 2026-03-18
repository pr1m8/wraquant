"""Tests for linear programming solvers."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from wraquant.opt.linear import solve_lp, solve_milp, transportation_problem


class TestSolveLP:
    def test_simple_lp(self) -> None:
        """min -x1 - 2*x2  s.t. x1+x2<=4, x1<=3, x2<=3, x>=0."""
        c = np.array([-1.0, -2.0])
        A_ub = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        b_ub = np.array([4.0, 3.0, 3.0])
        bounds = [(0.0, None), (0.0, None)]
        result = solve_lp(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        assert result["success"]
        assert_allclose(result["x"], [1.0, 3.0], atol=1e-6)
        assert_allclose(result["objective"], -7.0, atol=1e-6)

    def test_equality_constraint(self) -> None:
        """min x1  s.t. x1+x2==2, x>=0."""
        c = np.array([1.0, 0.0])
        A_eq = np.array([[1.0, 1.0]])
        b_eq = np.array([2.0])
        bounds = [(0.0, None), (0.0, None)]
        result = solve_lp(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        assert result["success"]
        assert_allclose(result["objective"], 0.0, atol=1e-6)


class TestSolveMILP:
    def test_integer_constraint(self) -> None:
        """min -x1 - x2  s.t. 2x1+x2<=6, x1,x2 integer, x>=0."""
        c = np.array([-1.0, -1.0])
        A_ub = np.array([[2.0, 1.0]])
        b_ub = np.array([6.0])
        bounds = [(0.0, None), (0.0, None)]
        integrality = [1, 1]
        result = solve_milp(
            c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, integrality=integrality
        )
        assert result["success"]
        # Solution should be integer
        assert_allclose(result["x"], np.round(result["x"]), atol=1e-6)

    def test_continuous_fallback(self) -> None:
        """Without integrality it behaves like an LP."""
        c = np.array([-1.0, -2.0])
        bounds = [(0.0, 3.0), (0.0, 3.0)]
        result = solve_milp(c, bounds=bounds)
        assert result["success"]
        assert_allclose(result["x"], [3.0, 3.0], atol=1e-6)


class TestTransportationProblem:
    def test_balanced_problem(self) -> None:
        costs = np.array([[2.0, 3.0], [5.0, 1.0]])
        supply = np.array([10.0, 15.0])
        demand = np.array([12.0, 13.0])
        result = transportation_problem(costs, supply, demand)
        assert result["success"]
        shipment = result["x"]
        # Supply constraints
        assert_allclose(shipment.sum(axis=1), supply, atol=1e-6)
        # Demand constraints
        assert_allclose(shipment.sum(axis=0), demand, atol=1e-6)
        # All shipments non-negative
        assert np.all(shipment >= -1e-10)

    def test_unbalanced_raises(self) -> None:
        costs = np.array([[1.0, 2.0]])
        supply = np.array([10.0])
        demand = np.array([5.0, 3.0])
        try:
            transportation_problem(costs, supply, demand)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass
