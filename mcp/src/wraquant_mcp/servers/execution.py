"""Execution algorithm MCP tools.

Tools: optimal_schedule, execution_cost, almgren_chriss.
"""

from __future__ import annotations

from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_execution_tools(mcp, ctx: AnalysisContext) -> None:
    """Register execution algorithm tools on the MCP server."""

    @mcp.tool()
    def optimal_schedule(
        quantity: float,
        dataset: str | None = None,
        method: str = "vwap",
        n_intervals: int = 78,
    ) -> dict[str, Any]:
        """Generate an optimal execution schedule.

        Produces a trade schedule that minimises deviation from
        a benchmark (TWAP or VWAP).

        Parameters:
            quantity: Total shares to execute.
            dataset: Dataset with volume profile (required for VWAP).
            method: Scheduling method ('twap' or 'vwap').
            n_intervals: Number of execution intervals (for TWAP).
        """
        from wraquant.execution.algorithms import twap_schedule, vwap_schedule

        if method == "vwap":
            if dataset is None:
                return {"error": "VWAP requires a dataset with volume profile"}

            df = ctx.get_dataset(dataset)
            volume_col = "volume"
            for col in df.columns:
                if "volume" in col.lower():
                    volume_col = col
                    break

            volume_profile = df[volume_col].dropna().values
            schedule = vwap_schedule(quantity, volume_profile)
        else:
            schedule = twap_schedule(quantity, n_intervals=n_intervals)

        import pandas as pd

        sched_df = pd.DataFrame({"quantity": schedule})
        stored = ctx.store_dataset(
            f"schedule_{method}", sched_df,
            source_op="optimal_schedule",
        )

        return _sanitize_for_json({
            "tool": "optimal_schedule",
            "method": method,
            "total_quantity": float(quantity),
            "n_intervals": len(schedule),
            "max_interval_qty": float(schedule.max()),
            "min_interval_qty": float(schedule.min()),
            **stored,
        })

    @mcp.tool()
    def execution_cost(
        quantity: float,
        price: float,
        spread: float,
        volume: float,
        commission_rate: float = 0.001,
    ) -> dict[str, Any]:
        """Estimate total execution cost for a trade.

        Combines spread cost, market impact, and commission into
        a total cost estimate.

        Parameters:
            quantity: Number of shares.
            price: Current price per share.
            spread: Bid-ask spread.
            volume: Average daily volume.
            commission_rate: Commission as fraction of trade value.
        """
        from wraquant.execution.cost import commission_cost, market_impact_model, slippage

        half_spread = spread / 2.0
        spread_cost = half_spread * abs(quantity)

        impact = market_impact_model(
            qty=abs(quantity),
            adv=volume,
            price=price,
            spread=spread,
        )

        comm = commission_cost(abs(quantity), price, rate=commission_rate)

        total = spread_cost + impact.get("total_impact_cost", 0.0) + comm

        return _sanitize_for_json({
            "tool": "execution_cost",
            "quantity": quantity,
            "price": price,
            "spread_cost": float(spread_cost),
            "market_impact": impact,
            "commission": float(comm),
            "total_estimated_cost": float(total),
            "cost_bps": float(total / (abs(quantity) * price) * 10_000)
            if quantity != 0 and price != 0 else 0.0,
        })

    @mcp.tool()
    def almgren_chriss(
        quantity: float,
        sigma: float,
        eta: float,
        gamma: float,
        risk_aversion: float = 1e-6,
        n_periods: int = 20,
    ) -> dict[str, Any]:
        """Compute the Almgren-Chriss optimal execution trajectory.

        Minimises a mean-variance cost objective balancing market
        impact (trading quickly) against price risk (trading slowly).

        Parameters:
            quantity: Total shares to execute.
            sigma: Price volatility per period.
            eta: Temporary impact coefficient.
            gamma: Permanent impact coefficient.
            risk_aversion: Risk aversion parameter (higher = faster).
            n_periods: Number of execution periods.
        """
        from wraquant.execution.optimal import almgren_chriss as _ac

        trajectory = _ac(
            total_qty=quantity,
            sigma=sigma,
            eta=eta,
            gamma=gamma,
            lambda_risk=risk_aversion,
            n_periods=n_periods,
        )

        import pandas as pd

        traj_df = pd.DataFrame({"remaining_position": trajectory})
        stored = ctx.store_dataset(
            "almgren_chriss_trajectory", traj_df,
            source_op="almgren_chriss",
        )

        return _sanitize_for_json({
            "tool": "almgren_chriss",
            "quantity": quantity,
            "n_periods": n_periods,
            "risk_aversion": risk_aversion,
            "trajectory": trajectory.tolist(),
            **stored,
        })
