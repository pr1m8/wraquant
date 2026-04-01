"""Market microstructure MCP tools.

Tools: liquidity_metrics, toxicity_analysis, market_quality,
spread_decomposition, price_impact, depth_analysis.
"""

from __future__ import annotations

from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_microstructure_tools(mcp, ctx: AnalysisContext) -> None:
    """Register microstructure-specific tools on the MCP server."""

    @mcp.tool()
    def liquidity_metrics(
        dataset: str,
        price_col: str = "close",
        volume_col: str = "volume",
        window: int = 20,
    ) -> dict[str, Any]:
        """Compute liquidity measures: Amihud illiquidity, Kyle lambda, Roll spread, effective spread.

        Parameters:
            dataset: Dataset containing price and volume data.
            price_col: Price column name.
            volume_col: Volume column name.
            window: Rolling window for Kyle lambda and Amihud.
        """
        import pandas as pd

        from wraquant.microstructure.liquidity import (
            amihud_illiquidity,
            kyle_lambda,
            roll_spread,
        )

        df = ctx.get_dataset(dataset)
        prices = df[price_col].dropna()
        volume = df[volume_col].dropna()

        n = min(len(prices), len(volume))
        prices = prices.iloc[-n:]
        volume = volume.iloc[-n:]

        returns = prices.pct_change().dropna()
        vol_aligned = volume.iloc[-len(returns):]

        amihud = amihud_illiquidity(returns, vol_aligned, window=window)
        kyle = kyle_lambda(prices, volume, window=window)
        roll = roll_spread(prices)

        metrics_df = pd.DataFrame({
            "amihud_illiquidity": amihud if isinstance(amihud, pd.Series) else pd.Series([amihud]),
            "kyle_lambda": kyle,
        })
        stored = ctx.store_dataset(
            f"liquidity_{dataset}", metrics_df,
            source_op="liquidity_metrics", parent=dataset,
        )

        return _sanitize_for_json({
            "tool": "liquidity_metrics",
            "dataset": dataset,
            "window": window,
            "amihud_illiquidity": float(amihud) if not isinstance(amihud, pd.Series) else float(amihud.iloc[-1]),
            "kyle_lambda_latest": float(kyle.iloc[-1]) if len(kyle) > 0 else None,
            "roll_spread": float(roll),
            "observations": n,
            **stored,
        })

    @mcp.tool()
    def toxicity_analysis(
        dataset: str,
        price_col: str = "close",
        volume_col: str = "volume",
        n_buckets: int = 50,
    ) -> dict[str, Any]:
        """Analyze order flow toxicity: VPIN, order flow imbalance, toxicity index.

        Estimates buy/sell volume via bulk volume classification when
        explicit buy/sell columns are not available.

        Parameters:
            dataset: Dataset containing price and volume data.
            price_col: Price column name.
            volume_col: Volume column name.
            n_buckets: Number of volume buckets for VPIN.
        """
        import pandas as pd

        from wraquant.microstructure.toxicity import (
            bulk_volume_classification,
            order_flow_imbalance,
            vpin,
        )

        df = ctx.get_dataset(dataset)
        prices = df[price_col].dropna()
        volume = df[volume_col].dropna()

        n = min(len(prices), len(volume))
        prices = prices.iloc[-n:]
        volume = volume.iloc[-n:]

        # Classify volume into buy/sell via BVC
        bvc = bulk_volume_classification(prices, volume)
        buy_vol = bvc * volume.values[-len(bvc):]
        sell_vol = (1 - bvc) * volume.values[-len(bvc):]

        buy_series = pd.Series(buy_vol, index=volume.index[-len(bvc):])
        sell_series = pd.Series(sell_vol, index=volume.index[-len(bvc):])

        vpin_values = vpin(volume.values[-len(bvc):], buy_vol, n_buckets=n_buckets)
        ofi = order_flow_imbalance(buy_series, sell_series, window=20)

        toxicity_df = pd.DataFrame({
            "vpin": pd.Series(vpin_values),
            "ofi": ofi.reset_index(drop=True),
        })
        stored = ctx.store_dataset(
            f"toxicity_{dataset}", toxicity_df,
            source_op="toxicity_analysis", parent=dataset,
        )

        return _sanitize_for_json({
            "tool": "toxicity_analysis",
            "dataset": dataset,
            "n_buckets": n_buckets,
            "vpin_latest": float(vpin_values[-1]) if len(vpin_values) > 0 else None,
            "vpin_mean": float(vpin_values.mean()) if len(vpin_values) > 0 else None,
            "ofi_latest": float(ofi.iloc[-1]) if len(ofi) > 0 else None,
            "observations": n,
            **stored,
        })

    @mcp.tool()
    def market_quality(
        dataset: str,
        price_col: str = "close",
    ) -> dict[str, Any]:
        """Assess market quality: variance ratio, efficiency ratio, quoted spread proxy.

        Parameters:
            dataset: Dataset containing price data.
            price_col: Price column name.
        """
        from wraquant.microstructure.market_quality import (
            market_efficiency_ratio,
            variance_ratio,
        )

        df = ctx.get_dataset(dataset)
        prices = df[price_col].dropna()

        vr = variance_ratio(prices)
        mer = market_efficiency_ratio(prices)

        return _sanitize_for_json({
            "tool": "market_quality",
            "dataset": dataset,
            "variance_ratio": vr,
            "market_efficiency_ratio": mer,
            "observations": len(prices),
        })

    @mcp.tool()
    def spread_decomposition(
        dataset: str,
        bid_col: str = "bid",
        ask_col: str = "ask",
        price_col: str = "close",
        volume_col: str = "volume",
    ) -> dict[str, Any]:
        """Huang-Stoll three-way spread decomposition.

        Decomposes the spread into adverse selection, inventory holding,
        and order processing components.

        Parameters:
            dataset: Dataset with bid, ask, price, and volume columns.
            bid_col: Bid price column.
            ask_col: Ask price column.
            price_col: Trade price column.
            volume_col: Volume column.
        """
        import numpy as np
        import pandas as pd

        from wraquant.microstructure.liquidity import (
            spread_decomposition as _spread_decomp,
        )

        df = ctx.get_dataset(dataset)
        bid = df[bid_col].dropna()
        ask = df[ask_col].dropna()
        trade_prices = df[price_col].dropna()
        volume = df[volume_col].dropna()

        n = min(len(bid), len(ask), len(trade_prices), len(volume))
        bid = bid.iloc[-n:]
        ask = ask.iloc[-n:]
        trade_prices = trade_prices.iloc[-n:]
        volume = volume.iloc[-n:]

        # Compute trade direction from price relative to midpoint
        mid = (bid + ask) / 2
        direction = np.sign(trade_prices.values - mid.values).astype(float)
        direction_series = pd.Series(direction, index=bid.index)

        result = _spread_decomp(
            trade_prices, bid, ask, direction_series, delay=5,
        )

        return _sanitize_for_json({
            "tool": "spread_decomposition",
            "dataset": dataset,
            "huang_stoll": result,
            "observations": n,
        })

    @mcp.tool()
    def price_impact(
        dataset: str,
        price_col: str = "close",
        volume_col: str = "volume",
    ) -> dict[str, Any]:
        """Measure permanent vs temporary price impact.

        Parameters:
            dataset: Dataset containing trade prices and volume.
            price_col: Trade price column.
            volume_col: Volume column.
        """
        import numpy as np
        import pandas as pd

        from wraquant.microstructure.liquidity import (
            price_impact as _price_impact,
        )

        df = ctx.get_dataset(dataset)
        prices = df[price_col].dropna()
        volume = df[volume_col].dropna()

        n = min(len(prices), len(volume))
        prices = prices.iloc[-n:]
        volume = volume.iloc[-n:]

        # Infer direction from price changes
        direction = np.sign(prices.diff().fillna(0).values)
        direction_series = pd.Series(direction, index=prices.index)

        impact = _price_impact(prices, volume, direction_series)

        impact_df = pd.DataFrame({"price_impact": impact})
        stored = ctx.store_dataset(
            f"price_impact_{dataset}", impact_df,
            source_op="price_impact", parent=dataset,
        )

        return _sanitize_for_json({
            "tool": "price_impact",
            "dataset": dataset,
            "mean_impact": float(impact.mean()) if len(impact) > 0 else None,
            "median_impact": float(impact.median()) if len(impact) > 0 else None,
            "observations": n,
            **stored,
        })

    @mcp.tool()
    def depth_analysis(
        dataset: str,
        bid_depth_col: str = "bid_depth",
        ask_depth_col: str = "ask_depth",
    ) -> dict[str, Any]:
        """Analyze order book depth imbalance.

        Computes the depth imbalance ratio (bid_depth - ask_depth) /
        (bid_depth + ask_depth), indicating directional pressure.

        Parameters:
            dataset: Dataset with bid and ask depth columns.
            bid_depth_col: Bid depth column name.
            ask_depth_col: Ask depth column name.
        """
        import pandas as pd

        from wraquant.microstructure.liquidity import depth_imbalance

        df = ctx.get_dataset(dataset)
        bid_depth = df[bid_depth_col].dropna()
        ask_depth = df[ask_depth_col].dropna()

        n = min(len(bid_depth), len(ask_depth))
        bid_depth = bid_depth.iloc[-n:]
        ask_depth = ask_depth.iloc[-n:]

        imbalance = depth_imbalance(bid_depth, ask_depth)

        imbalance_df = pd.DataFrame({"depth_imbalance": imbalance})
        stored = ctx.store_dataset(
            f"depth_{dataset}", imbalance_df,
            source_op="depth_analysis", parent=dataset,
        )

        return _sanitize_for_json({
            "tool": "depth_analysis",
            "dataset": dataset,
            "mean_imbalance": float(imbalance.mean()) if len(imbalance) > 0 else None,
            "latest_imbalance": float(imbalance.iloc[-1]) if len(imbalance) > 0 else None,
            "buy_pressure_pct": float((imbalance > 0).mean()) if len(imbalance) > 0 else None,
            "observations": n,
            **stored,
        })
