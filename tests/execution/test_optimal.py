"""Tests for optimal execution models."""

from __future__ import annotations

import numpy as np
import pytest

from wraquant.execution.optimal import (
    almgren_chriss,
    execution_frontier,
    optimal_execution_cost,
)


class TestAlmgrenChriss:
    def test_boundary_conditions(self) -> None:
        traj = almgren_chriss(1000.0, 0.02, 0.01, 0.001, 1.0, 10)
        assert traj[0] == pytest.approx(1000.0)
        assert traj[-1] == pytest.approx(0.0)

    def test_length(self) -> None:
        traj = almgren_chriss(1000.0, 0.02, 0.01, 0.001, 1.0, 20)
        assert len(traj) == 21

    def test_monotonically_decreasing(self) -> None:
        traj = almgren_chriss(1000.0, 0.02, 0.01, 0.001, 1.0, 10)
        assert np.all(np.diff(traj) <= 1e-10)

    def test_risk_neutral_is_linear(self) -> None:
        traj = almgren_chriss(1000.0, 0.02, 0.01, 0.001, 0.0, 10)
        expected = np.linspace(1000.0, 0.0, 11)
        np.testing.assert_allclose(traj, expected, atol=1e-10)

    def test_high_risk_aversion_front_loads(self) -> None:
        traj_low = almgren_chriss(1000.0, 0.02, 0.01, 0.001, 0.01, 10)
        traj_high = almgren_chriss(1000.0, 0.02, 0.01, 0.001, 100.0, 10)
        # High risk aversion: trade more aggressively early
        assert traj_high[1] < traj_low[1]

    def test_invalid_periods(self) -> None:
        with pytest.raises(ValueError):
            almgren_chriss(1000.0, 0.02, 0.01, 0.001, 1.0, 0)


class TestOptimalExecutionCost:
    def test_returns_dict(self) -> None:
        traj = almgren_chriss(1000.0, 0.02, 0.01, 0.001, 1.0, 10)
        result = optimal_execution_cost(traj, 0.02, 0.01, 0.001)
        assert "expected_cost" in result
        assert "variance" in result
        assert "std_dev" in result

    def test_positive_cost(self) -> None:
        traj = almgren_chriss(1000.0, 0.02, 0.01, 0.001, 1.0, 10)
        result = optimal_execution_cost(traj, 0.02, 0.01, 0.001)
        assert result["expected_cost"] > 0

    def test_variance_non_negative(self) -> None:
        traj = almgren_chriss(1000.0, 0.02, 0.01, 0.001, 1.0, 10)
        result = optimal_execution_cost(traj, 0.02, 0.01, 0.001)
        assert result["variance"] >= 0
        assert result["std_dev"] >= 0


class TestExecutionFrontier:
    def test_output_shape(self) -> None:
        result = execution_frontier(1000.0, 0.02, 0.01, 0.001, n_points=15)
        assert len(result["lambda_values"]) == 15
        assert len(result["expected_cost"]) == 15
        assert len(result["std_dev"]) == 15

    def test_cost_risk_tradeoff(self) -> None:
        result = execution_frontier(1000.0, 0.02, 0.01, 0.001, n_points=20)
        # Higher lambda -> lower risk (std_dev) generally
        # The last point (highest lambda) should have lower std_dev
        assert result["std_dev"][-1] < result["std_dev"][0]
