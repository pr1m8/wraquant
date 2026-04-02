"""Auto-register ALL wraquant functions as MCP tools.

Instead of manually writing tool definitions for 1097 functions,
this module introspects wraquant's __all__ exports and auto-generates
MCP tools using the ToolAdaptor pattern.

The auto-registered tools are Tier 3 — available on demand when
an agent calls list_tools(module). The hand-written tools in
servers/ are Tier 2 — curated, optimized, always loaded.

Both tiers coexist: Tier 2 tools override Tier 3 for the same function.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from typing import Any, Callable

from wraquant_mcp.adaptor import ToolAdaptor, _detect_data_params
from wraquant_mcp.context import AnalysisContext, _sanitize_for_json

logger = logging.getLogger("wraquant_mcp.auto_register")

# All wraquant modules to auto-register
WRAQUANT_MODULES = [
    "risk", "vol", "regimes", "ta", "stats", "ts", "opt",
    "backtest", "price", "ml", "causal", "bayes", "forex",
    "econometrics", "microstructure", "execution", "math",
    "viz", "experiment", "flow", "scale", "data", "io",
    "fundamental", "news",
]

# Functions that should NOT be auto-registered (internal helpers, classes)
SKIP_NAMES = {
    "Pipeline", "DAG", "Lab", "Experiment", "ExperimentResults",
    "ExperimentRunner", "ExperimentStore", "Backtest", "VectorizedBacktest",
    "PositionSizer", "EventTracker", "Workflow", "WorkflowResult",
    "RegimeDetector", "RegimeResult", "GARCHResult", "BacktestResult",
    "ForecastResult", "PriceSeries", "ReturnSeries", "OHLCVFrame",
    "ReturnFrame", "WQConfig", "get_config", "reset_config",
}

# Tools already defined in tier-2 servers (don't duplicate)
TIER2_TOOLS: set[str] = set()


def auto_register_all(mcp: Any, ctx: AnalysisContext) -> dict[str, int]:
    """Auto-register all wraquant functions as MCP tools.

    Returns dict of module_name → number of tools registered.
    """
    adaptor = ToolAdaptor(ctx)
    counts: dict[str, int] = {}

    for module_name in WRAQUANT_MODULES:
        try:
            mod = importlib.import_module(f"wraquant.{module_name}")
        except ImportError:
            logger.debug("Module wraquant.%s not importable, skipping", module_name)
            continue

        all_names = getattr(mod, "__all__", [])
        registered = 0

        for func_name in all_names:
            # Skip classes and already-registered tools
            if func_name in SKIP_NAMES:
                continue

            tool_id = f"{module_name}_{func_name}"
            if tool_id in TIER2_TOOLS:
                continue

            func = getattr(mod, func_name, None)
            if func is None or not callable(func):
                continue

            # Skip classes
            if isinstance(func, type):
                continue

            try:
                # Auto-wrap with adaptor
                wrapped = adaptor.wrap(func, module_name)

                # Get first line of docstring for description
                doc = func.__doc__ or f"{module_name}.{func_name}"
                description = doc.split("\n")[0].strip()

                mcp.add_tool(
                    wrapped,
                    name=tool_id,
                    description=description,
                )
                registered += 1
            except Exception as e:
                logger.debug(
                    "Failed to auto-register %s.%s: %s",
                    module_name, func_name, e,
                )

        counts[module_name] = registered
        if registered > 0:
            logger.info(
                "Auto-registered %d tools from wraquant.%s",
                registered, module_name,
            )

    return counts


def register_auto_tools(mcp: Any, ctx: AnalysisContext) -> None:
    """Register auto-generated tools plus a discovery tool.

    Also registers a meta-tool that reports auto-registration stats.
    """
    counts = auto_register_all(mcp, ctx)

    @mcp.tool()
    def auto_tool_stats() -> dict[str, Any]:
        """Show auto-registered tool counts per module.

        These are Tier 3 tools — every wraquant function available
        as an MCP tool via the auto-adaptor pattern.
        """
        total = sum(counts.values())
        return {
            "total_auto_tools": total,
            "per_module": counts,
            "note": "These supplement the hand-crafted Tier 2 tools in servers/",
        }
