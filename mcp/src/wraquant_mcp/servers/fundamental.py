"""Fundamental analysis MCP tools.

Tools: fundamental_screen, valuation.
"""

from __future__ import annotations

from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_fundamental_tools(mcp, ctx: AnalysisContext) -> None:
    """Register fundamental analysis tools on the MCP server."""

    @mcp.tool()
    def fundamental_screen(
        dataset: str,
    ) -> dict[str, Any]:
        """Screen stocks using Piotroski F-Score and financial ratios.

        Requires a dataset with financial statement columns:
        net_income, total_assets, operating_cash_flow, etc.

        Parameters:
            dataset: Dataset with fundamental data.
        """
        from wraquant.fundamental.ratios import (
            current_ratio,
            debt_to_equity,
            pe_ratio,
            pb_ratio,
            roe,
        )

        df = ctx.get_dataset(dataset)

        results = []
        for _, row in df.iterrows():
            entry: dict[str, Any] = {}
            rd = row.to_dict()

            if "price" in rd and "earnings" in rd:
                entry["pe_ratio"] = pe_ratio(rd["price"], rd["earnings"])
            if "price" in rd and "book_value" in rd:
                entry["pb_ratio"] = pb_ratio(rd["price"], rd["book_value"])
            if "net_income" in rd and "equity" in rd:
                entry["roe"] = roe(rd["net_income"], rd["equity"])
            if "total_debt" in rd and "equity" in rd:
                entry["debt_to_equity"] = debt_to_equity(rd["total_debt"], rd["equity"])
            if "current_assets" in rd and "current_liabilities" in rd:
                entry["current_ratio"] = current_ratio(
                    rd["current_assets"], rd["current_liabilities"],
                )

            if "name" in rd:
                entry["name"] = rd["name"]
            elif "ticker" in rd:
                entry["name"] = rd["ticker"]

            results.append(entry)

        return _sanitize_for_json({
            "tool": "fundamental_screen",
            "dataset": dataset,
            "n_stocks": len(results),
            "results": results,
        })

    @mcp.tool()
    def valuation(
        cash_flows: list[float],
        discount_rate: float = 0.10,
        terminal_growth: float = 0.02,
        model: str = "dcf",
    ) -> dict[str, Any]:
        """Estimate intrinsic value using discounted cash flow.

        Projects future cash flows and discounts them to present
        value using the Gordon growth model for terminal value.

        Parameters:
            cash_flows: Projected free cash flows for each future period.
            discount_rate: WACC or required return (e.g., 0.10 = 10%).
            terminal_growth: Perpetual growth rate (e.g., 0.02 = 2%).
            model: Valuation model ('dcf'). Other models planned.
        """
        if model != "dcf":
            return {"error": f"Model '{model}' not yet implemented. Use 'dcf'."}

        from wraquant.fundamental.valuation import dcf_valuation

        result = dcf_valuation(
            cash_flows=cash_flows,
            discount_rate=discount_rate,
            terminal_growth=terminal_growth,
        )

        return _sanitize_for_json({
            "tool": "valuation",
            "model": model,
            "discount_rate": discount_rate,
            "terminal_growth": terminal_growth,
            "n_periods": len(cash_flows),
            **result,
        })
