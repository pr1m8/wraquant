"""Module-specific MCP tool registrations for wraquant-mcp.

Each sub-module registers tools on a shared FastMCP instance,
keeping server.py focused on tier-1/tier-2 tools while these
provide deep module-specific capabilities.
"""

from __future__ import annotations

from wraquant_mcp.servers.backtest import register_backtest_tools
from wraquant_mcp.servers.bayes import register_bayes_tools
from wraquant_mcp.servers.causal import register_causal_tools
from wraquant_mcp.servers.data import register_data_tools
from wraquant_mcp.servers.econometrics import register_econometrics_tools
from wraquant_mcp.servers.execution import register_execution_tools
from wraquant_mcp.servers.experiment import register_experiment_tools
from wraquant_mcp.servers.forex import register_forex_tools
from wraquant_mcp.servers.fundamental import register_fundamental_tools
from wraquant_mcp.servers.math import register_math_tools
from wraquant_mcp.servers.microstructure import register_microstructure_tools
from wraquant_mcp.servers.ml import register_ml_tools
from wraquant_mcp.servers.news import register_news_tools
from wraquant_mcp.servers.opt import register_opt_tools
from wraquant_mcp.servers.price import register_price_tools
from wraquant_mcp.servers.regimes import register_regimes_tools
from wraquant_mcp.servers.risk import register_risk_tools
from wraquant_mcp.servers.stats import register_stats_tools
from wraquant_mcp.servers.ta import register_ta_tools
from wraquant_mcp.servers.ts import register_ts_tools
from wraquant_mcp.servers.viz import register_viz_tools
from wraquant_mcp.servers.vol import register_vol_tools


def register_all(mcp, ctx):
    """Register all module-specific tools on the MCP server."""
    # Core analytics
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

    # High priority — deep analytics
    register_regimes_tools(mcp, ctx)
    register_causal_tools(mcp, ctx)
    register_bayes_tools(mcp, ctx)
    register_econometrics_tools(mcp, ctx)
    register_microstructure_tools(mcp, ctx)

    # Medium priority
    register_execution_tools(mcp, ctx)
    register_forex_tools(mcp, ctx)
    register_math_tools(mcp, ctx)
    register_experiment_tools(mcp, ctx)
    register_data_tools(mcp, ctx)

    # Low priority — stubs
    register_fundamental_tools(mcp, ctx)
    register_news_tools(mcp, ctx)
