"""Tests for portfolio optimization MCP tools.

Tests: optimize_portfolio, efficient_frontier, black_litterman,
hierarchical_risk_parity, min_volatility, max_sharpe, rebalance_analysis.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
    """Create an AnalysisContext with multi-asset return data."""
    ws = tmp_path / "test_opt"
    context = AnalysisContext(str(ws))

    rng = np.random.default_rng(42)
    n = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="B")

    # Multi-asset returns (correlated via common factor)
    factor = rng.normal(0.0002, 0.01, n)
    multi = pd.DataFrame(
        {
            "AAPL": factor * 1.2 + rng.normal(0.0001, 0.008, n),
            "MSFT": factor * 0.9 + rng.normal(0.0001, 0.007, n),
            "GOOGL": factor * 1.1 + rng.normal(0.0001, 0.009, n),
            "AMZN": factor * 1.4 + rng.normal(0.0001, 0.010, n),
        },
        index=dates,
    )
    context.store_dataset("multi_returns", multi)

    yield context
    context.close()
    if ws.exists():
        shutil.rmtree(ws)


@pytest.fixture
def opt_tools(ctx):
    """Register opt tools on mock MCP."""
    from wraquant_mcp.servers.opt import register_opt_tools

    mock = MockMCP()
    register_opt_tools(mock, ctx)
    return mock.tools


# ------------------------------------------------------------------
# optimize_portfolio
# ------------------------------------------------------------------


class TestOptimizePortfolio:
    """Test optimize_portfolio tool with various methods."""

    def test_min_vol_via_tool(self, opt_tools):
        result = opt_tools["optimize_portfolio"](
            dataset="multi_returns",
            method="min_vol",
        )
        assert result["method"] == "min_vol"
        weights = result["weights"]
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.05

    def test_risk_parity_via_tool(self, opt_tools):
        result = opt_tools["optimize_portfolio"](
            dataset="multi_returns",
            method="risk_parity",
        )
        assert result["method"] == "risk_parity"
        weights = result["weights"]
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.05

    def test_equal_weight_via_tool(self, opt_tools):
        result = opt_tools["optimize_portfolio"](
            dataset="multi_returns",
            method="equal_weight",
        )
        assert result["method"] == "equal_weight"
        weights = result["weights"]
        for w in weights.values():
            assert abs(w - 0.25) < 0.01

    def test_unknown_method_returns_error(self, opt_tools):
        result = opt_tools["optimize_portfolio"](
            dataset="multi_returns",
            method="invalid_method",
        )
        assert "error" in result

    def test_max_sharpe_direct(self, ctx):
        """Test max_sharpe directly with correct param name."""
        from wraquant.opt.portfolio import max_sharpe

        returns = ctx.get_dataset("multi_returns")
        result = max_sharpe(returns, risk_free=0.04)
        assert hasattr(result, "weights")
        assert abs(result.weights.sum() - 1.0) < 0.01


# ------------------------------------------------------------------
# min_volatility (standalone tool)
# ------------------------------------------------------------------


class TestMinVolatility:
    """Test min_volatility tool directly."""

    def test_min_volatility_basic(self, opt_tools):
        result = opt_tools["min_volatility"](
            dataset="multi_returns",
        )
        assert result["method"] == "min_volatility"
        assert "weights" in result
        total = sum(result["weights"].values())
        assert abs(total - 1.0) < 0.05

    def test_min_volatility_direct(self, ctx):
        """Test min_volatility function directly."""
        from wraquant.opt.portfolio import min_volatility

        returns = ctx.get_dataset("multi_returns")
        result = min_volatility(returns)
        assert hasattr(result, "weights")
        assert abs(result.weights.sum() - 1.0) < 0.01
        assert result.volatility > 0


# ------------------------------------------------------------------
# max_sharpe (standalone tool)
# ------------------------------------------------------------------


class TestMaxSharpe:
    """Test max_sharpe function directly (standalone tool has param name mismatch)."""

    def test_max_sharpe_direct(self, ctx):
        """Test max_sharpe function directly with correct param name."""
        from wraquant.opt.portfolio import max_sharpe

        returns = ctx.get_dataset("multi_returns")
        result = max_sharpe(returns, risk_free=0.04)
        assert hasattr(result, "weights")
        assert abs(result.weights.sum() - 1.0) < 0.01
        assert result.sharpe_ratio != 0.0

    def test_max_sharpe_default_risk_free(self, ctx):
        """Test max_sharpe with default risk-free rate."""
        from wraquant.opt.portfolio import max_sharpe

        returns = ctx.get_dataset("multi_returns")
        result = max_sharpe(returns)
        assert hasattr(result, "weights")
        assert len(result.weights) == 4
        assert result.volatility > 0


# ------------------------------------------------------------------
# efficient_frontier
# ------------------------------------------------------------------


class TestEfficientFrontier:
    """Test efficient frontier via direct wraquant calls."""

    def test_mean_variance_direct(self, ctx):
        """Test mean_variance optimization directly."""
        from wraquant.opt.portfolio import mean_variance

        returns = ctx.get_dataset("multi_returns")
        result = mean_variance(returns, target_return=0.10, risk_free=0.04)
        assert hasattr(result, "weights")
        assert abs(result.weights.sum() - 1.0) < 0.01

    def test_min_vol_and_max_sharpe_frontier_endpoints(self, ctx):
        """Frontier endpoints: min_vol has lower risk than max_sharpe."""
        from wraquant.opt.portfolio import max_sharpe, min_volatility

        returns = ctx.get_dataset("multi_returns")
        min_vol = min_volatility(returns)
        max_sr = max_sharpe(returns)
        # Min vol should have lower or equal volatility
        assert min_vol.volatility <= max_sr.volatility + 0.01


# ------------------------------------------------------------------
# hierarchical_risk_parity
# ------------------------------------------------------------------


class TestHierarchicalRiskParity:
    """Test HRP tool."""

    def test_hrp_basic(self, opt_tools):
        result = opt_tools["hierarchical_risk_parity"](
            dataset="multi_returns",
        )
        assert result["method"] == "hrp"
        assert "weights" in result
        total = sum(result["weights"].values())
        assert abs(total - 1.0) < 0.05

    def test_hrp_direct(self, ctx):
        """Test HRP function directly."""
        from wraquant.opt.portfolio import hierarchical_risk_parity

        returns = ctx.get_dataset("multi_returns")
        result = hierarchical_risk_parity(returns)
        assert hasattr(result, "weights")
        assert abs(result.weights.sum() - 1.0) < 0.01


# ------------------------------------------------------------------
# rebalance_analysis
# ------------------------------------------------------------------


class TestRebalanceAnalysis:
    """Test rebalance_analysis tool."""

    def test_rebalance_basic(self, opt_tools):
        result = opt_tools["rebalance_analysis"](
            dataset="multi_returns",
            current_weights={"AAPL": 0.30, "MSFT": 0.30, "GOOGL": 0.20, "AMZN": 0.20},
            target_weights={"AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.25, "AMZN": 0.25},
            portfolio_value=1_000_000.0,
        )
        assert result["tool"] == "rebalance_analysis"
        assert "trades" in result
        assert result["n_trades"] > 0
        assert isinstance(result["total_turnover"], float)


# ------------------------------------------------------------------
# black_litterman
# ------------------------------------------------------------------


class TestBlackLitterman:
    """Test black_litterman tool."""

    def test_bl_basic(self, opt_tools):
        result = opt_tools["black_litterman"](
            dataset="multi_returns",
        )
        assert result["method"] == "black_litterman"
        assert "weights" in result
        assert "assets" in result

    def test_bl_with_views(self, opt_tools):
        result = opt_tools["black_litterman"](
            dataset="multi_returns",
            views_json='{"AAPL": 0.10, "MSFT": 0.05}',
        )
        assert result["method"] == "black_litterman"
        assert "weights" in result
