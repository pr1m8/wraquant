"""Market microstructure MCP tools.

Tools: liquidity_metrics, toxicity_analysis, market_quality,
spread_decomposition.
"""

from __future__ import annotations

from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_microstructure_tools(mcp, ctx: AnalysisContext) -> None:
    """Register microstructure tools on the MCP server."""

    @mcp.tool()
    def liquidity_metrics(
        dataset: str,
        returns_col: str = "returns",
        volume_col: str = "volume",
        price_col: str = "close",
        window: int | None = None,
    ) -> dict[str, Any]:
        """Compute multiple liquidity measures.

        Returns Amihud illiquidity ratio, Kyle's lambda (price impact),
        Roll spread estimate, and turnover ratio.

        Parameters:
            dataset: Dataset with returns, volume, and price columns.
            returns_col: Returns column name.
            volume_col: Volume column name.
            price_col: Price column name.
            window: Rolling window (None = full-sample scalar).
        """
        from wraquant.microstructure.liquidity import (
            amihud_illiquidity,
            kyle_lambda,
            roll_spread,
        )

        df = ctx.get_dataset(dataset)
        returns = df[returns_col].dropna()
        volume = df[volume_col].dropna()
        price = df[price_col].dropna()

        n = min(len(returns), len(volume), len(price))
        returns = returns.iloc[-n:]
        volume = volume.iloc[-n:]
        price = price.iloc[-n:]

        dollar_volume = price * volume

        amihud = amihud_illiquidity(returns, dollar_volume, window=window)
        kyle = kyle_lambda(returns, dollar_volume)
        roll = roll_spread(price)

        return _sanitize_for_json({
            "tool": "liquidity_metrics",
            "dataset": dataset,
            "amihud_illiquidity": float(amihud) if not hasattr(amihud, "__len__") else float(amihud.iloc[-1]),
            "kyle_lambda": float(kyle),
            "roll_spread": float(roll),
            "observations": n,
        })

    @mcp.tool()
    def toxicity_analysis(
        dataset: str,
        volume_col: str = "volume",
        buy_volume_col: str = "buy_volume",
        n_buckets: int = 50,
    ) -> dict[str, Any]:
        """Analyze informed trading using VPIN and order flow metrics.

        Computes Volume-Synchronized Probability of Informed Trading
        (VPIN) and order flow imbalance.

        Parameters:
            dataset: Dataset with volume and buy volume columns.
            volume_col: Total volume column.
            buy_volume_col: Buy-initiated volume column.
            n_buckets: Number of volume buckets for VPIN.
        """
        from wraquant.microstructure.toxicity import order_flow_imbalance, vpin

        df = ctx.get_dataset(dataset)
        volume = df[volume_col].dropna().values
        buy_volume = df[buy_volume_col].dropna().values

        n = min(len(volume), len(buy_volume))
        volume = volume[-n:]
        buy_volume = buy_volume[-n:]

        vpin_vals = vpin(volume, buy_volume, n_buckets=n_buckets)

        import pandas as pd

        sell_volume = volume - buy_volume
        ofi = order_flow_imbalance(
            pd.Series(buy_volume), pd.Series(sell_volume),
        )

        return _sanitize_for_json({
            "tool": "toxicity_analysis",
            "dataset": dataset,
            "vpin_latest": float(vpin_vals[-1]) if len(vpin_vals) > 0 else None,
            "vpin_mean": float(vpin_vals.mean()),
            "vpin_max": float(vpin_vals.max()),
            "ofi_latest": float(ofi.iloc[-1]) if len(ofi) > 0 else None,
            "ofi_mean": float(ofi.mean()),
            "n_buckets": n_buckets,
        })

    @mcp.tool()
    def market_quality(
        dataset: str,
        bid_col: str = "bid",
        ask_col: str = "ask",
        returns_col: str = "returns",
    ) -> dict[str, Any]:
        """Compute market quality and efficiency metrics.

        Returns quoted spread, relative spread, variance ratio
        (efficiency), and market efficiency ratio.

        Parameters:
            dataset: Dataset with bid/ask/returns columns.
            bid_col: Bid price column.
            ask_col: Ask price column.
            returns_col: Returns column (for efficiency metrics).
        """
        import numpy as np

        from wraquant.microstructure.market_quality import (
            market_efficiency_ratio,
            quoted_spread,
            relative_spread,
            variance_ratio,
        )

        df = ctx.get_dataset(dataset)

        results: dict[str, Any] = {"tool": "market_quality", "dataset": dataset}

        if bid_col in df.columns and ask_col in df.columns:
            bid = df[bid_col].dropna()
            ask = df[ask_col].dropna()
            n = min(len(bid), len(ask))
            bid = bid.iloc[-n:]
            ask = ask.iloc[-n:]

            qs = quoted_spread(bid.values, ask.values)
            rs = relative_spread(bid.values, ask.values)

            results["mean_quoted_spread"] = float(np.mean(qs))
            results["mean_relative_spread"] = float(np.mean(rs))

        if returns_col in df.columns:
            returns = df[returns_col].dropna()
            vr = variance_ratio(returns)
            mer = market_efficiency_ratio(returns)

            results["variance_ratio"] = vr
            results["market_efficiency_ratio"] = mer

        return _sanitize_for_json(results)

    @mcp.tool()
    def spread_decomposition(
        dataset: str,
        trade_col: str = "trade_price",
        bid_col: str = "bid",
        ask_col: str = "ask",
        direction_col: str = "direction",
        delay: int = 5,
    ) -> dict[str, Any]:
        """Huang-Stoll three-way spread decomposition.

        Decomposes the effective spread into adverse selection,
        order processing, and inventory holding components.

        Parameters:
            dataset: Dataset with trade-level data.
            trade_col: Trade price column.
            bid_col: Bid price column.
            ask_col: Ask price column.
            direction_col: Trade direction column (+1 buy, -1 sell).
            delay: Number of periods for adverse selection measurement.
        """
        from wraquant.microstructure.liquidity import spread_decomposition as _sd

        df = ctx.get_dataset(dataset)
        trade_prices = df[trade_col].dropna()
        bid = df[bid_col].dropna()
        ask = df[ask_col].dropna()
        direction = df[direction_col].dropna()

        n = min(len(trade_prices), len(bid), len(ask), len(direction))
        trade_prices = trade_prices.iloc[-n:]
        bid = bid.iloc[-n:]
        ask = ask.iloc[-n:]
        direction = direction.iloc[-n:]

        result = _sd(trade_prices, bid, ask, direction, delay=delay)

        return _sanitize_for_json({
            "tool": "spread_decomposition",
            "dataset": dataset,
            "adverse_selection": result.get("adverse_selection"),
            "order_processing": result.get("order_processing"),
            "inventory_holding": result.get("inventory_holding"),
            "effective_spread": result.get("effective_spread"),
        })
