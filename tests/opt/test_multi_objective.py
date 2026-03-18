"""Tests for multi-objective optimization."""

from __future__ import annotations

import numpy as np

from wraquant.opt.multi_objective import epsilon_constraint, pareto_front


class TestParetoFront:
    def test_returns_multiple_points(self) -> None:
        """Two competing quadratics should produce a non-trivial front."""

        def f1(x: np.ndarray) -> float:
            return float((x[0] - 0.0) ** 2 + (x[1] - 0.0) ** 2)

        def f2(x: np.ndarray) -> float:
            return float((x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2)

        result = pareto_front(
            objectives=[f1, f2],
            n_points=20,
            bounds=[(-1.0, 2.0), (-1.0, 2.0)],
        )
        assert result["n_points"] > 1
        assert result["points"].shape[0] == result["n_points"]
        assert result["objectives"].shape[1] == 2

    def test_too_few_objectives_raises(self) -> None:
        try:
            pareto_front(
                objectives=[lambda x: x[0] ** 2],
                bounds=[(-1.0, 1.0)],
            )
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_points_are_non_dominated(self) -> None:
        """No returned point should be dominated by another."""

        def f1(x: np.ndarray) -> float:
            return float(x[0] ** 2)

        def f2(x: np.ndarray) -> float:
            return float((x[0] - 2.0) ** 2)

        result = pareto_front(
            objectives=[f1, f2],
            n_points=30,
            bounds=[(-1.0, 3.0)],
        )
        objs = result["objectives"]
        n = objs.shape[0]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # j should not dominate i
                assert not (
                    np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i])
                ), f"Point {j} dominates point {i}"


class TestEpsilonConstraint:
    def test_output_structure(self) -> None:
        def primary(x: np.ndarray) -> float:
            return float(x[0] ** 2)

        def secondary(x: np.ndarray) -> float:
            return float((x[0] - 2.0) ** 2)

        eps_vals = [np.array([0.5, 1.0, 2.0, 4.0])]
        result = epsilon_constraint(
            primary_obj=primary,
            secondary_objs=[secondary],
            epsilon_values=eps_vals,
            bounds=[(-3.0, 3.0)],
        )
        assert "points" in result
        assert "primary_values" in result
        assert "secondary_values" in result
        assert "n_points" in result
        assert result["n_points"] > 0
        assert result["points"].shape[0] == result["n_points"]

    def test_mismatched_lengths_raises(self) -> None:
        try:
            epsilon_constraint(
                primary_obj=lambda x: x[0],
                secondary_objs=[lambda x: x[0]],
                epsilon_values=[np.array([1.0]), np.array([2.0])],
                bounds=[(-1.0, 1.0)],
            )
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass
