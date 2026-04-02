"""MCP prompt templates for guided quant analysis workflows.

150+ prompts across 16 categories that guide AI agents through
multi-step quantitative finance workflows using wraquant tools.
"""

from __future__ import annotations

from typing import Any

__all__ = ["register_all_prompts"]


def register_all_prompts(mcp: Any) -> None:
    """Register all prompt templates on the MCP server."""
    from wraquant_mcp.prompts.analysis import register_analysis_prompts
    from wraquant_mcp.prompts.bayes import register_bayes_prompts
    from wraquant_mcp.prompts.data import register_data_prompts
    from wraquant_mcp.prompts.econometrics import register_econometrics_prompts
    from wraquant_mcp.prompts.execution import register_execution_prompts
    from wraquant_mcp.prompts.forex import register_forex_prompts
    from wraquant_mcp.prompts.math import register_math_prompts
    from wraquant_mcp.prompts.ml import register_ml_prompts
    from wraquant_mcp.prompts.portfolio import register_portfolio_prompts
    from wraquant_mcp.prompts.pricing import register_pricing_prompts
    from wraquant_mcp.prompts.regime import register_regime_prompts
    from wraquant_mcp.prompts.reporting import register_reporting_prompts
    from wraquant_mcp.prompts.risk import register_risk_prompts
    from wraquant_mcp.prompts.strategy import register_strategy_prompts
    from wraquant_mcp.prompts.system import register_system_prompts
    from wraquant_mcp.prompts.tools_guide import register_tool_guide_prompts

    register_system_prompts(mcp)
    register_analysis_prompts(mcp)
    register_risk_prompts(mcp)
    register_regime_prompts(mcp)
    register_portfolio_prompts(mcp)
    register_strategy_prompts(mcp)
    register_ml_prompts(mcp)
    register_pricing_prompts(mcp)
    register_reporting_prompts(mcp)
    register_tool_guide_prompts(mcp)
    register_execution_prompts(mcp)
    register_econometrics_prompts(mcp)
    register_forex_prompts(mcp)
    register_data_prompts(mcp)
    register_math_prompts(mcp)
    register_bayes_prompts(mcp)
