"""Execution algorithms MCP tools.

Tools: optimal_schedule, execution_cost, almgren_chriss,
transaction_cost_analysis, close_auction.
"""

from __future__ import annotations

from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_execution_tools(mcp, ctx: AnalysisContext) -> None:
    """Register execution-specific tools on the MCP server."""

    @mcp.tool()
    def optimal_schedule(
        total_quantity: float,
        dataset: str,
        volume_col: str = "volume",
        method: str = "vwap",
    ) -> dict[str, Any]:
        """Generate an optimal execution schedule (TWAP, VWAP, or IS).

        Parameters:
            total_quantity: Total shares/units to execute.
            dataset: Dataset containing historical volume profile.
            volume_col: Volume column name.
            method: Schedule type — 'twap', 'vwap', or 'is'.
        """
        import pandas as pd

        from wraquant.execution.algorithms import (
            is_schedule,
            twap_schedule,
            vwap_schedule,
        )

        df = ctx.get_dataset(dataset)
        volume = df[volume_col].dropna()
        n_intervals = len(volume)

        if method == "twap":
            schedule = twap_schedule(total_quantity, n_intervals)
        elif method == "vwap":
            schedule = vwap_schedule(total_quantity, volume.values)
        elif method == "is":
            schedule = is_schedule(total_quantity, volume.values)
        else:
            msg = f"Unknown method '{method}'. Use 'twap', 'vwap', or 'is'."
            raise ValueError(msg)

        schedule_df = pd.DataFrame({"quantity": schedule})
        stored = ctx.store_dataset(
            f"schedule_{method}_{dataset}", schedule_df,
            source_op="optimal_schedule", parent=dataset,
        )

        return _sanitize_for_json({
            "tool": "optimal_schedule",
            "method": method,
            "total_quantity": total_quantity,
            "n_intervals": n_intervals,
            "max_slice": float(schedule.max()),
            "min_slice": float(schedule.min()),
            "first_5_slices": schedule[:5].tolist(),
            **stored,
        })

    @mcp.tool()
    def execution_cost(
        quantity: float,
        price: float,
        spread: float,
        adv: float,
        volatility: float,
    ) -> dict[str, Any]:
        """Pre-trade execution cost estimate.

        Estimates total expected cost broken down into spread crossing,
        market impact, and timing risk components.

        Parameters:
            quantity: Number of shares to trade.
            price: Current market price.
            spread: Bid-ask spread in price units.
            adv: Average daily volume (shares).
            volatility: Daily return volatility (e.g. 0.02 for 2%).
        """
        from wraquant.execution.cost import expected_cost_model

        result = expected_cost_model(
            quantity=quantity,
            price=price,
            adv=adv,
            volatility=volatility,
            spread=spread,
        )

        return _sanitize_for_json({
            "tool": "execution_cost",
            "quantity": quantity,
            "price": price,
            **result,
        })

    @mcp.tool()
    def almgren_chriss(
        total_shares: float,
        n_periods: int,
        dataset: str,
        risk_aversion: float = 0.001,
    ) -> dict[str, Any]:
        """Almgren-Chriss optimal execution trajectory.

        Computes the mean-variance optimal trading rate that balances
        market impact cost against price uncertainty risk.

        Parameters:
            total_shares: Total position to liquidate.
            n_periods: Number of trading periods.
            dataset: Dataset containing price data for volatility estimation.
            risk_aversion: Risk aversion parameter (lambda).
        """
        import pandas as pd

        from wraquant.execution.optimal import almgren_chriss as _ac

        df = ctx.get_dataset(dataset)
        prices = df.iloc[:, 0].dropna()
        returns = prices.pct_change().dropna()

        sigma = float(returns.std())
        # Reasonable defaults for market impact parameters
        eta = sigma * 0.1    # temporary impact coefficient
        gamma = sigma * 0.01  # permanent impact coefficient

        trajectory = _ac(
            total_qty=total_shares,
            sigma=sigma,
            eta=eta,
            gamma=gamma,
            lambda_risk=risk_aversion,
            n_periods=n_periods,
        )

        traj_df = pd.DataFrame({"optimal_trajectory": trajectory})
        stored = ctx.store_dataset(
            f"ac_trajectory_{dataset}", traj_df,
            source_op="almgren_chriss", parent=dataset,
        )

        import numpy as np

        # Trade rates: shares traded per period (positive = selling)
        trades = -np.diff(trajectory)

        return _sanitize_for_json({
            "tool": "almgren_chriss",
            "total_shares": total_shares,
            "n_periods": n_periods,
            "risk_aversion": risk_aversion,
            "estimated_sigma": sigma,
            "estimated_eta": eta,
            "estimated_gamma": gamma,
            "front_loaded_pct": float(trades[:n_periods // 4].sum() / total_shares)
            if n_periods >= 4 else None,
            "trajectory_summary": {
                "first_period": float(trades[0]),
                "last_period": float(trades[-1]),
                "max_rate": float(trades.max()),
            },
            **stored,
        })

    @mcp.tool()
    def transaction_cost_analysis(
        trades_json: str,
        market_data_dataset: str,
    ) -> dict[str, Any]:
        """Post-trade Transaction Cost Analysis (TCA).

        Compares each trade's execution price against arrival price,
        VWAP, and close benchmarks.

        Parameters:
            trades_json: JSON array of trade records. Each record must
                have keys: 'price', 'quantity', 'timestamp'.
            market_data_dataset: Dataset with market data (open, high,
                low, close, volume columns).
        """
        import json

        import pandas as pd

        from wraquant.execution.cost import (
            transaction_cost_analysis as _tca,
        )

        trades = json.loads(trades_json)
        trades_df = pd.DataFrame(trades)

        market_df = ctx.get_dataset(market_data_dataset)

        tca_result = _tca(trades_df, market_df)

        stored = ctx.store_dataset(
            f"tca_{market_data_dataset}", tca_result,
            source_op="transaction_cost_analysis", parent=market_data_dataset,
        )

        summary = {}
        for col in tca_result.columns:
            if tca_result[col].dtype in ("float64", "float32"):
                summary[f"mean_{col}"] = float(tca_result[col].mean())

        return _sanitize_for_json({
            "tool": "transaction_cost_analysis",
            "n_trades": len(trades_df),
            "market_data_dataset": market_data_dataset,
            "summary": summary,
            **stored,
        })

    @mcp.tool()
    def close_auction(
        total_quantity: float,
        close_volume_pct: float = 0.2,
    ) -> dict[str, Any]:
        """Allocate order quantity between continuous market and closing auction.

        Splits the order into a continuous-market portion and a
        closing-auction (MOC) portion based on historical close volume.

        Parameters:
            total_quantity: Total shares to execute.
            close_volume_pct: Fraction of daily volume at the close
                (default 0.2 = 20%).
        """
        from wraquant.execution.algorithms import close_auction_allocation

        result = close_auction_allocation(
            total_quantity=total_quantity,
            historical_close_volume_pct=close_volume_pct,
        )

        return _sanitize_for_json({
            "tool": "close_auction",
            "total_quantity": total_quantity,
            "close_volume_pct": close_volume_pct,
            **result,
        })
