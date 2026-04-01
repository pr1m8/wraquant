"""Module-specific MCP tool registrations for wraquant-mcp.

Each sub-module registers tools on a shared FastMCP instance,
keeping server.py focused on tier-1/tier-2 tools while these
provide deep module-specific capabilities.
"""

from __future__ import annotations

from wraquant_mcp.servers.backtest import register_backtest_tools
from wraquant_mcp.servers.ml import register_ml_tools
from wraquant_mcp.servers.opt import register_opt_tools
from wraquant_mcp.servers.price import register_price_tools
from wraquant_mcp.servers.risk import register_risk_tools
from wraquant_mcp.servers.stats import register_stats_tools
from wraquant_mcp.servers.ta import register_ta_tools
from wraquant_mcp.servers.ts import register_ts_tools
from wraquant_mcp.servers.viz import register_viz_tools
from wraquant_mcp.servers.vol import register_vol_tools


def register_all(mcp, ctx):
    """Register all module-specific tools on the MCP server."""
    register_risk_tools(mcp, ctx)
    register_vol_tools(mcp, ctx)
    register_stats_tools(mcp, ctx)
    register_ts_tools(mcp, ctx)
    register_opt_tools(mcp, ctx)
    register_backtest_tools(mcp, ctx)
    register_ml_tools(mcp, ctx)
    register_ta_tools(mcp, ctx)
    register_price_tools(mcp, ctx)
    register_viz_tools(mcp, ctx)
