"""Execution algorithms MCP tools.

Tools: optimal_schedule, execution_cost, almgren_chriss,
transaction_cost_analysis, close_auction,
is_schedule, pov_schedule, expected_cost_model,
bertsimas_lo, slippage_estimate.
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
        try:
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
                f"schedule_{method}_{dataset}",
                schedule_df,
                source_op="optimal_schedule",
                parent=dataset,
            )

            return _sanitize_for_json(
                {
                    "tool": "optimal_schedule",
                    "method": method,
                    "total_quantity": total_quantity,
                    "n_intervals": n_intervals,
                    "max_slice": float(schedule.max()),
                    "min_slice": float(schedule.min()),
                    "first_5_slices": schedule[:5].tolist(),
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "optimal_schedule"}

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
        try:
            from wraquant.execution.cost import expected_cost_model

            result = expected_cost_model(
                quantity=quantity,
                price=price,
                adv=adv,
                volatility=volatility,
                spread=spread,
            )

            return _sanitize_for_json(
                {
                    "tool": "execution_cost",
                    "quantity": quantity,
                    "price": price,
                    **result,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "execution_cost"}

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
        try:
            import pandas as pd

            from wraquant.execution.optimal import almgren_chriss as _ac

            df = ctx.get_dataset(dataset)
            prices = df.iloc[:, 0].dropna()
            returns = prices.pct_change().dropna()

            sigma = float(returns.std())
            # Reasonable defaults for market impact parameters
            eta = sigma * 0.1  # temporary impact coefficient
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
                f"ac_trajectory_{dataset}",
                traj_df,
                source_op="almgren_chriss",
                parent=dataset,
            )

            import numpy as np

            # Trade rates: shares traded per period (positive = selling)
            trades = -np.diff(trajectory)

            return _sanitize_for_json(
                {
                    "tool": "almgren_chriss",
                    "total_shares": total_shares,
                    "n_periods": n_periods,
                    "risk_aversion": risk_aversion,
                    "estimated_sigma": sigma,
                    "estimated_eta": eta,
                    "estimated_gamma": gamma,
                    "front_loaded_pct": (
                        float(trades[: n_periods // 4].sum() / total_shares)
                        if n_periods >= 4
                        else None
                    ),
                    "trajectory_summary": {
                        "first_period": float(trades[0]),
                        "last_period": float(trades[-1]),
                        "max_rate": float(trades.max()),
                    },
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "almgren_chriss"}

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
        try:
            import json

            import pandas as pd

            from wraquant.execution.cost import transaction_cost_analysis as _tca

            trades = json.loads(trades_json)
            trades_df = pd.DataFrame(trades)

            market_df = ctx.get_dataset(market_data_dataset)

            tca_result = _tca(trades_df, market_df)

            stored = ctx.store_dataset(
                f"tca_{market_data_dataset}",
                tca_result,
                source_op="transaction_cost_analysis",
                parent=market_data_dataset,
            )

            summary = {}
            for col in tca_result.columns:
                if tca_result[col].dtype in ("float64", "float32"):
                    summary[f"mean_{col}"] = float(tca_result[col].mean())

            return _sanitize_for_json(
                {
                    "tool": "transaction_cost_analysis",
                    "n_trades": len(trades_df),
                    "market_data_dataset": market_data_dataset,
                    "summary": summary,
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "transaction_cost_analysis"}

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
        try:
            from wraquant.execution.algorithms import close_auction_allocation

            result = close_auction_allocation(
                total_quantity=total_quantity,
                historical_close_volume_pct=close_volume_pct,
            )

            return _sanitize_for_json(
                {
                    "tool": "close_auction",
                    "total_quantity": total_quantity,
                    "close_volume_pct": close_volume_pct,
                    **result,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "close_auction"}

    # ------------------------------------------------------------------
    # Enhanced execution tools
    # ------------------------------------------------------------------

    @mcp.tool()
    def is_schedule_tool(
        total_quantity: float,
        dataset: str,
        volume_col: str = "volume",
        alpha: float = 0.5,
    ) -> dict[str, Any]:
        """Generate an Implementation Shortfall (IS) optimal schedule.

        Balances execution urgency against market impact via a blend of
        TWAP (uniform) and VWAP (volume-proportional) components. The
        alpha parameter controls the trade-off: higher alpha front-loads
        execution.

        Parameters:
            total_quantity: Total shares/units to execute.
            dataset: Dataset containing historical volume profile.
            volume_col: Volume column name.
            alpha: Urgency parameter in [0, 1]. 0.0 = pure VWAP,
                0.5 = balanced (default), 1.0 = pure TWAP.
        """
        try:
            import pandas as pd

            from wraquant.execution.algorithms import is_schedule as _is_schedule

            df = ctx.get_dataset(dataset)
            volume = df[volume_col].dropna()

            schedule = _is_schedule(total_quantity, volume.values, alpha=alpha)

            schedule_df = pd.DataFrame({"quantity": schedule})
            stored = ctx.store_dataset(
                f"is_schedule_{dataset}",
                schedule_df,
                source_op="is_schedule",
                parent=dataset,
            )

            return _sanitize_for_json(
                {
                    "tool": "is_schedule",
                    "total_quantity": total_quantity,
                    "alpha": alpha,
                    "n_intervals": len(schedule),
                    "max_slice": float(schedule.max()),
                    "min_slice": float(schedule.min()),
                    "front_loaded_pct": (
                        float(schedule[: len(schedule) // 4].sum() / total_quantity)
                        if len(schedule) >= 4
                        else None
                    ),
                    "first_5_slices": schedule[:5].tolist(),
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "is_schedule_tool"}

    @mcp.tool()
    def pov_schedule_tool(
        total_quantity: float,
        dataset: str,
        volume_col: str = "volume",
        pov_rate: float = 0.1,
    ) -> dict[str, Any]:
        """Generate a Percentage of Volume (POV) execution schedule.

        Targets a fixed fraction of market volume in each interval.
        Use when you need to limit your market footprint (e.g., never
        exceed 10% of volume per interval).

        Parameters:
            total_quantity: Total shares/units to execute.
            dataset: Dataset containing historical volume profile.
            volume_col: Volume column name.
            pov_rate: Target participation rate in (0, 1]. Common:
                0.05 (passive), 0.10 (standard), 0.20 (aggressive).
        """
        try:
            import pandas as pd

            from wraquant.execution.algorithms import pov_schedule as _pov_schedule

            df = ctx.get_dataset(dataset)
            volume = df[volume_col].dropna()

            schedule = _pov_schedule(total_quantity, volume.values, pov_rate=pov_rate)

            schedule_df = pd.DataFrame({"quantity": schedule})
            stored = ctx.store_dataset(
                f"pov_schedule_{dataset}",
                schedule_df,
                source_op="pov_schedule",
                parent=dataset,
            )

            executed = float(schedule.sum())

            return _sanitize_for_json(
                {
                    "tool": "pov_schedule",
                    "total_quantity": total_quantity,
                    "pov_rate": pov_rate,
                    "n_intervals": len(schedule),
                    "executed_quantity": executed,
                    "fill_pct": (
                        round(executed / total_quantity * 100, 1)
                        if total_quantity > 0
                        else 0
                    ),
                    "max_slice": float(schedule.max()),
                    "first_5_slices": schedule[:5].tolist(),
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "pov_schedule_tool"}

    @mcp.tool()
    def expected_cost_model_tool(
        quantity: float,
        price: float,
        adv: float,
        volatility: float,
        spread: float,
    ) -> dict[str, Any]:
        """Full pre-trade execution cost model with spread, impact, and timing risk.

        Estimates total expected cost broken down into:
        1. Spread cost (half-spread per share)
        2. Market impact (square-root model)
        3. Timing risk (opportunity cost of slow execution)

        Parameters:
            quantity: Number of shares to trade.
            price: Current market price.
            adv: Average daily volume (shares).
            volatility: Daily return volatility (e.g., 0.02 for 2%).
            spread: Bid-ask spread in price units.
        """
        try:
            from wraquant.execution.cost import expected_cost_model as _expected_cost

            result = _expected_cost(
                quantity=quantity,
                price=price,
                adv=adv,
                volatility=volatility,
                spread=spread,
            )

            return _sanitize_for_json(
                {
                    "tool": "expected_cost_model",
                    "quantity": quantity,
                    "price": price,
                    "adv": adv,
                    "volatility": volatility,
                    "spread": spread,
                    "participation_rate": abs(quantity) / adv if adv > 0 else None,
                    **result,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "expected_cost_model_tool"}

    @mcp.tool()
    def bertsimas_lo_tool(
        total_shares: float,
        n_periods: int,
        dataset: str,
        risk_aversion: float = 0.0,
    ) -> dict[str, Any]:
        """Bertsimas-Lo (1998) optimal execution with discrete trading.

        Computes an optimal execution trajectory using dynamic programming
        with a linear temporary impact model. Unlike Almgren-Chriss which
        models continuous trading, Bertsimas-Lo works in discrete periods.

        Parameters:
            total_shares: Total position to liquidate.
            n_periods: Number of discrete trading periods.
            dataset: Dataset containing price data for volatility estimation.
            risk_aversion: Risk aversion parameter (0.0 = risk-neutral linear
                liquidation; higher values front-load execution).
        """
        try:
            import pandas as pd

            from wraquant.execution.optimal import bertsimas_lo as _bl

            df = ctx.get_dataset(dataset)
            prices = df.iloc[:, 0].dropna()
            returns = prices.pct_change().dropna()

            sigma = float(returns.std())
            # Impact coefficient: estimate from volatility
            impact_coeff = sigma * 0.01

            result = _bl(
                total_shares=total_shares,
                n_periods=n_periods,
                volatility=sigma,
                impact_coeff=impact_coeff,
                risk_aversion=risk_aversion,
            )

            traj_df = pd.DataFrame(
                {
                    "trajectory": result["trajectory"],
                }
            )
            stored = ctx.store_dataset(
                f"bl_trajectory_{dataset}",
                traj_df,
                source_op="bertsimas_lo",
                parent=dataset,
            )

            trades = result["trades"]

            return _sanitize_for_json(
                {
                    "tool": "bertsimas_lo",
                    "total_shares": total_shares,
                    "n_periods": n_periods,
                    "risk_aversion": risk_aversion,
                    "estimated_volatility": sigma,
                    "estimated_impact_coeff": impact_coeff,
                    "expected_cost": result["expected_cost"],
                    "cost_variance": result["cost_variance"],
                    "front_loaded_pct": (
                        float(trades[: n_periods // 4].sum() / total_shares)
                        if n_periods >= 4
                        else None
                    ),
                    "trajectory_summary": {
                        "first_trade": float(trades[0]),
                        "last_trade": float(trades[-1]),
                        "max_trade": float(trades.max()),
                    },
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "bertsimas_lo_tool"}

    @mcp.tool()
    def slippage_estimate(
        quantity: float,
        price: float,
        volume: float,
        volatility: float,
    ) -> dict[str, Any]:
        """Quick slippage estimate using the square-root market impact model.

        Provides a fast pre-trade slippage estimate based on order size
        relative to average daily volume and current volatility.

        Parameters:
            quantity: Number of shares to trade.
            price: Current market price.
            volume: Average daily volume (shares).
            volatility: Daily return volatility (e.g., 0.02 for 2%).
        """
        try:

            from wraquant.execution.cost import market_impact_model

            participation = abs(quantity) / volume if volume > 0 else 1.0

            # Square-root impact model
            impact_frac = market_impact_model(
                qty=abs(quantity),
                avg_daily_volume=volume,
                volatility=volatility,
                model="sqrt",
            )

            slippage_dollars = impact_frac * price
            slippage_bps = impact_frac * 10_000
            total_slippage_cost = slippage_dollars * abs(quantity)

            return _sanitize_for_json(
                {
                    "tool": "slippage_estimate",
                    "quantity": quantity,
                    "price": price,
                    "volume": volume,
                    "volatility": volatility,
                    "participation_rate": participation,
                    "slippage_per_share": slippage_dollars,
                    "slippage_bps": slippage_bps,
                    "total_slippage_cost": total_slippage_cost,
                    "slippage_pct_of_notional": impact_frac * 100,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "slippage_estimate"}
