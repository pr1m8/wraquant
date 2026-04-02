"""Tests for supervisor MCP tool — workflow recommendation and module guide.

Tests recommend_workflow and module_guide by calling the functions
directly from the supervisor module.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add mcp source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def supervisor_tools():
    """Register supervisor tools on a mock MCP server and return them.

    The supervisor functions are closures registered via @mcp.tool().
    We use a mock MCP that captures the registered functions.
    """
    from wraquant_mcp.supervisor import register_supervisor_tools

    tools = {}

    class MockMCP:
        def tool(self):
            def decorator(fn):
                tools[fn.__name__] = fn
                return fn
            return decorator

    mock_mcp = MockMCP()
    mock_ctx = MagicMock()

    register_supervisor_tools(mock_mcp, mock_ctx)

    return tools


class TestSupervisor:
    """Test supervisor MCP tool functions."""

    def test_recommend_workflow_risk(self, supervisor_tools):
        """recommend_workflow('risk') returns the risk workflow."""
        recommend_workflow = supervisor_tools["recommend_workflow"]

        result = recommend_workflow("Analyze AAPL risk")

        assert isinstance(result, dict)
        assert result["workflow"] == "Risk Analysis"
        assert isinstance(result["steps"], list)
        assert len(result["steps"]) > 0
        assert any("risk_metrics" in step for step in result["steps"])
        assert any("var_analysis" in step.lower() or "VaR" in step for step in result["steps"])
        assert isinstance(result["modules"], list)
        assert "risk" in result["modules"]

    def test_recommend_workflow_pairs(self, supervisor_tools):
        """recommend_workflow('pairs') returns the pairs trading workflow."""
        recommend_workflow = supervisor_tools["recommend_workflow"]

        result = recommend_workflow("Build a pairs trading strategy")

        assert isinstance(result, dict)
        assert result["workflow"] == "Pairs Trading"
        assert isinstance(result["steps"], list)
        assert len(result["steps"]) > 0
        assert any("cointegration" in step.lower() for step in result["steps"])
        assert any("backtest" in step.lower() for step in result["steps"])
        assert isinstance(result["modules"], list)
        assert "stats" in result["modules"]
        assert "backtest" in result["modules"]

    def test_module_guide_risk(self, supervisor_tools):
        """module_guide('risk') returns a guide with key_tools."""
        module_guide = supervisor_tools["module_guide"]

        result = module_guide("risk")

        assert isinstance(result, dict)
        assert result["module"] == "risk"
        assert isinstance(result["description"], str)
        assert len(result["description"]) > 0
        assert isinstance(result["key_tools"], list)
        assert len(result["key_tools"]) > 0
        assert "risk_metrics" in result["key_tools"]
        assert "var_analysis" in result["key_tools"]
        assert isinstance(result["when_to_use"], str)
        assert isinstance(result["feeds_into"], list)
        assert isinstance(result["feeds_from"], list)
        assert isinstance(result["functions"], int)
        assert result["functions"] > 0
