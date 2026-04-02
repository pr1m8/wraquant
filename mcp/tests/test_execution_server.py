"""Tests for execution algorithms MCP tools.

Tests: optimal_schedule, execution_cost, almgren_chriss.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
import shutil

from wraquant_mcp.context import AnalysisContext


# ------------------------------------------------------------------
# Mock MCP
# ------------------------------------------------------------------


class MockMCP:
    """Capture tool functions registered via @mcp.tool()."""

    def __init__(self):
        self.tools: dict[str, callable] = {}

    def tool(self):
        def decorator(func):
            self.tools[func.__name__] = func
            return func
        return decorator


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def ctx(tmp_path):
    ws = tmp_path / "test_ws"
    context = AnalysisContext(str(ws))
    rng = np.random.default_rng(42)
    prices = pd.DataFrame({
        "close": 100 + rng.normal(0, 1, 252).cumsum(),
        "volume": rng.integers(1000, 10000, 252).astype(float),
    })
    context.store_dataset("prices", prices)
    returns = prices["close"].pct_change().dropna()
    context.store_dataset("returns", returns.to_frame("returns"), parent="prices")
    yield context
    context.close()


@pytest.fixture
def exec_tools(ctx):
    """Register execution tools and return them."""
    from wraquant_mcp.servers.execution import register_execution_tools

    mock = MockMCP()
    register_execution_tools(mock, ctx)
    return mock.tools


# ------------------------------------------------------------------
# Optimal schedule
# ------------------------------------------------------------------


class TestOptimalSchedule:
    """Test optimal_schedule tool."""

    def test_vwap_schedule_returns_valid_schedule(self, exec_tools):
        """VWAP schedule allocates total quantity proportionally to volume."""
        total_qty = 10000.0

        result = exec_tools["optimal_schedule"](
            total_quantity=total_qty,
            dataset="prices",
            volume_col="volume",
            method="vwap",
        )

        assert result["tool"] == "optimal_schedule"
        assert result["method"] == "vwap"
        assert result["total_quantity"] == total_qty
        assert isinstance(result["n_intervals"], int)
        assert result["n_intervals"] == 252

        assert isinstance(result["max_slice"], float)
        assert result["max_slice"] > 0
        assert np.isfinite(result["max_slice"])

        assert isinstance(result["min_slice"], float)
        assert result["min_slice"] > 0
        assert np.isfinite(result["min_slice"])

        assert isinstance(result["first_5_slices"], list)
        assert len(result["first_5_slices"]) == 5
        for s in result["first_5_slices"]:
            assert isinstance(s, float)
            assert np.isfinite(s)
            assert s > 0

        # Should store a dataset
        assert "dataset_id" in result

        # Total schedule should approximately sum to total_quantity
        # (stored dataset should have correct sum)


# ------------------------------------------------------------------
# Execution cost
# ------------------------------------------------------------------


class TestExecutionCost:
    """Test execution_cost tool."""

    def test_execution_cost_returns_total_cost(self, exec_tools):
        """Execution cost returns breakdown with total cost."""
        result = exec_tools["execution_cost"](
            quantity=5000.0,
            price=100.0,
            spread=0.02,
            adv=1_000_000.0,
            volatility=0.02,
        )

        assert result["tool"] == "execution_cost"
        assert result["quantity"] == 5000.0
        assert result["price"] == 100.0

        # Should have cost breakdown
        assert "spread_cost" in result
        assert isinstance(result["spread_cost"], float)
        assert result["spread_cost"] > 0
        assert np.isfinite(result["spread_cost"])

        assert "impact_cost" in result
        assert isinstance(result["impact_cost"], float)
        assert result["impact_cost"] >= 0
        assert np.isfinite(result["impact_cost"])

        assert "timing_risk" in result
        assert isinstance(result["timing_risk"], float)
        assert result["timing_risk"] >= 0
        assert np.isfinite(result["timing_risk"])

        assert "total_cost" in result
        assert isinstance(result["total_cost"], float)
        assert result["total_cost"] > 0
        assert np.isfinite(result["total_cost"])

        # Total should be sum of components
        expected_total = (
            result["spread_cost"]
            + result["impact_cost"]
            + result["timing_risk"]
        )
        assert abs(result["total_cost"] - expected_total) < 1e-6

        assert "cost_bps" in result
        assert isinstance(result["cost_bps"], float)
        assert result["cost_bps"] > 0


# ------------------------------------------------------------------
# Almgren-Chriss
# ------------------------------------------------------------------


class TestAlmgrenChriss:
    """Test almgren_chriss tool."""

    def test_almgren_chriss_returns_trajectory(self, exec_tools):
        """Almgren-Chriss returns optimal execution trajectory."""
        total_shares = 10000.0
        n_periods = 20

        result = exec_tools["almgren_chriss"](
            total_shares=total_shares,
            n_periods=n_periods,
            dataset="prices",
            risk_aversion=0.001,
        )

        assert result["tool"] == "almgren_chriss"
        assert result["total_shares"] == total_shares
        assert result["n_periods"] == n_periods
        assert result["risk_aversion"] == 0.001

        # Estimated parameters should be positive and finite
        assert isinstance(result["estimated_sigma"], float)
        assert result["estimated_sigma"] > 0
        assert np.isfinite(result["estimated_sigma"])

        assert isinstance(result["estimated_eta"], float)
        assert result["estimated_eta"] > 0
        assert np.isfinite(result["estimated_eta"])

        assert isinstance(result["estimated_gamma"], float)
        assert result["estimated_gamma"] > 0
        assert np.isfinite(result["estimated_gamma"])

        # Trajectory summary
        summary = result["trajectory_summary"]
        assert isinstance(summary, dict)
        assert isinstance(summary["first_period"], float)
        assert np.isfinite(summary["first_period"])
        assert isinstance(summary["last_period"], float)
        assert np.isfinite(summary["last_period"])
        assert isinstance(summary["max_rate"], float)
        assert summary["max_rate"] > 0

        # Front-loaded percentage
        if result["front_loaded_pct"] is not None:
            assert isinstance(result["front_loaded_pct"], float)
            assert 0 <= result["front_loaded_pct"] <= 1

        # Should store a dataset
        assert "dataset_id" in result
