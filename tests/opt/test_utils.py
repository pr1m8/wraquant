"""Tests for optimization utility helpers."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from wraquant.opt.utils import (
    cardinality_constraint,
    sector_constraints,
    sum_to_one_constraint,
    turnover_constraint,
    weight_constraint,
)


class TestWeightConstraint:
    def test_default_bounds(self) -> None:
        result = weight_constraint(5)
        assert len(result["bounds"]) == 5
        for lb, ub in result["bounds"]:
            assert lb == 0.0
            assert ub == 1.0

    def test_custom_bounds(self) -> None:
        result = weight_constraint(3, lb=-0.5, ub=0.5)
        assert len(result["bounds"]) == 3
        for lb, ub in result["bounds"]:
            assert lb == -0.5
            assert ub == 0.5


class TestSumToOneConstraint:
    def test_satisfied(self) -> None:
        con = sum_to_one_constraint(4)
        assert con["type"] == "eq"
        w = np.array([0.25, 0.25, 0.25, 0.25])
        assert_allclose(con["fun"](w), 0.0, atol=1e-12)

    def test_violated(self) -> None:
        con = sum_to_one_constraint(3)
        w = np.array([0.5, 0.5, 0.5])
        assert con["fun"](w) > 0  # sum > 1


class TestSectorConstraints:
    def test_generates_two_per_sector(self) -> None:
        cons = sector_constraints(
            5,
            sectors={"tech": [0, 1], "energy": [2, 3, 4]},
            sector_limits={"tech": (0.1, 0.5), "energy": (0.2, 0.6)},
        )
        assert len(cons) == 4  # 2 sectors * 2 bounds each

    def test_lower_bound_satisfied(self) -> None:
        cons = sector_constraints(
            4,
            sectors={"A": [0, 1]},
            sector_limits={"A": (0.3, 0.7)},
        )
        w = np.array([0.2, 0.2, 0.3, 0.3])
        # sector A weight = 0.4, which is >= 0.3
        lower = [c for c in cons if c.get("bound") == "lower"][0]
        assert lower["fun"](w) >= 0

    def test_upper_bound_violated(self) -> None:
        cons = sector_constraints(
            4,
            sectors={"A": [0, 1]},
            sector_limits={"A": (0.0, 0.3)},
        )
        w = np.array([0.4, 0.4, 0.1, 0.1])
        # sector A weight = 0.8, which exceeds 0.3
        upper = [c for c in cons if c.get("bound") == "upper"][0]
        assert upper["fun"](w) < 0

    def test_missing_sector_limit_skipped(self) -> None:
        cons = sector_constraints(
            4,
            sectors={"A": [0, 1], "B": [2, 3]},
            sector_limits={"A": (0.1, 0.5)},
        )
        # Only sector A has limits
        assert len(cons) == 2


class TestTurnoverConstraint:
    def test_within_limit(self) -> None:
        current = np.array([0.25, 0.25, 0.25, 0.25])
        con = turnover_constraint(current, max_turnover=0.20)
        new_w = np.array([0.30, 0.20, 0.25, 0.25])
        # turnover = 0.5 * (0.05 + 0.05 + 0 + 0) = 0.05 <= 0.20
        assert con["fun"](new_w) >= 0

    def test_exceeds_limit(self) -> None:
        current = np.array([0.25, 0.25, 0.25, 0.25])
        con = turnover_constraint(current, max_turnover=0.05)
        new_w = np.array([0.50, 0.00, 0.25, 0.25])
        # turnover = 0.5 * (0.25 + 0.25 + 0 + 0) = 0.25 > 0.05
        assert con["fun"](new_w) < 0


class TestCardinalityConstraint:
    def test_structure(self) -> None:
        result = cardinality_constraint(10, max_holdings=5)
        assert result["n_assets"] == 10
        assert result["max_holdings"] == 5
        assert "description" in result
