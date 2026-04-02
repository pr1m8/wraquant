"""Market microstructure MCP tools.

Tools: liquidity_metrics, toxicity_analysis, market_quality,
spread_decomposition, price_impact, depth_analysis,
kyle_lambda_rolling, amihud_rolling, corwin_schultz_spread,
roll_spread, effective_spread, order_flow_imbalance,
trade_classification, intraday_volatility_pattern,
information_share, liquidity_commonality.
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
        try:
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
            vol_aligned = volume.iloc[-len(returns) :]

            amihud = amihud_illiquidity(returns, vol_aligned, window=window)
            kyle = kyle_lambda(prices, volume, window=window)
            roll = roll_spread(prices)

            metrics_df = pd.DataFrame(
                {
                    "amihud_illiquidity": (
                        amihud if isinstance(amihud, pd.Series) else pd.Series([amihud])
                    ),
                    "kyle_lambda": kyle,
                }
            )
            stored = ctx.store_dataset(
                f"liquidity_{dataset}",
                metrics_df,
                source_op="liquidity_metrics",
                parent=dataset,
            )

            return _sanitize_for_json(
                {
                    "tool": "liquidity_metrics",
                    "dataset": dataset,
                    "window": window,
                    "amihud_illiquidity": (
                        float(amihud)
                        if not isinstance(amihud, pd.Series)
                        else float(amihud.iloc[-1])
                    ),
                    "kyle_lambda_latest": (
                        float(kyle.iloc[-1]) if len(kyle) > 0 else None
                    ),
                    "roll_spread": float(roll),
                    "observations": n,
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "liquidity_metrics"}

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
        try:
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
            # BVC requires close, high, low, volume columns
            if "high" in df.columns and "low" in df.columns:
                high = df["high"].dropna().iloc[-n:]
                low = df["low"].dropna().iloc[-n:]
            else:
                # Approximate high/low from close if not available
                high = prices + prices.abs() * 0.005
                low = prices - prices.abs() * 0.005

            bvc = bulk_volume_classification(prices, high, low, volume)
            buy_vol = bvc["buy_volume"].values
            sell_vol = bvc["sell_volume"].values

            buy_series = pd.Series(buy_vol, index=volume.index[-len(buy_vol) :])
            sell_series = pd.Series(sell_vol, index=volume.index[-len(sell_vol) :])

            vpin_values = vpin(
                volume.values[-len(buy_vol) :], buy_vol, n_buckets=n_buckets
            )
            ofi = order_flow_imbalance(buy_series, sell_series, window=20)

            toxicity_df = pd.DataFrame(
                {
                    "vpin": pd.Series(vpin_values),
                    "ofi": ofi.reset_index(drop=True),
                }
            )
            stored = ctx.store_dataset(
                f"toxicity_{dataset}",
                toxicity_df,
                source_op="toxicity_analysis",
                parent=dataset,
            )

            return _sanitize_for_json(
                {
                    "tool": "toxicity_analysis",
                    "dataset": dataset,
                    "n_buckets": n_buckets,
                    "vpin_latest": (
                        float(vpin_values[-1]) if len(vpin_values) > 0 else None
                    ),
                    "vpin_mean": (
                        float(vpin_values.mean()) if len(vpin_values) > 0 else None
                    ),
                    "ofi_latest": float(ofi.iloc[-1]) if len(ofi) > 0 else None,
                    "observations": n,
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "toxicity_analysis"}

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
        try:
            from wraquant.microstructure.market_quality import (
                market_efficiency_ratio,
                variance_ratio,
            )

            df = ctx.get_dataset(dataset)
            prices = df[price_col].dropna()

            vr = variance_ratio(prices)
            mer = market_efficiency_ratio(prices)

            return _sanitize_for_json(
                {
                    "tool": "market_quality",
                    "dataset": dataset,
                    "variance_ratio": vr,
                    "market_efficiency_ratio": mer,
                    "observations": len(prices),
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "market_quality"}

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
        try:
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
                trade_prices,
                bid,
                ask,
                direction_series,
                delay=5,
            )

            return _sanitize_for_json(
                {
                    "tool": "spread_decomposition",
                    "dataset": dataset,
                    "huang_stoll": result,
                    "observations": n,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "spread_decomposition"}

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
        try:
            import numpy as np
            import pandas as pd

            from wraquant.microstructure.liquidity import price_impact as _price_impact

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
                f"price_impact_{dataset}",
                impact_df,
                source_op="price_impact",
                parent=dataset,
            )

            return _sanitize_for_json(
                {
                    "tool": "price_impact",
                    "dataset": dataset,
                    "mean_impact": float(impact.mean()) if len(impact) > 0 else None,
                    "median_impact": (
                        float(impact.median()) if len(impact) > 0 else None
                    ),
                    "observations": n,
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "price_impact"}

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
        try:
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
                f"depth_{dataset}",
                imbalance_df,
                source_op="depth_analysis",
                parent=dataset,
            )

            return _sanitize_for_json(
                {
                    "tool": "depth_analysis",
                    "dataset": dataset,
                    "mean_imbalance": (
                        float(imbalance.mean()) if len(imbalance) > 0 else None
                    ),
                    "latest_imbalance": (
                        float(imbalance.iloc[-1]) if len(imbalance) > 0 else None
                    ),
                    "buy_pressure_pct": (
                        float((imbalance > 0).mean()) if len(imbalance) > 0 else None
                    ),
                    "observations": n,
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "depth_analysis"}

    # ------------------------------------------------------------------
    # Enhanced microstructure tools
    # ------------------------------------------------------------------

    @mcp.tool()
    def kyle_lambda_rolling(
        dataset: str,
        price_col: str = "close",
        volume_col: str = "volume",
        window: int = 60,
    ) -> dict[str, Any]:
        """Compute rolling Kyle's lambda with 95% confidence intervals.

        Estimates the permanent price impact coefficient via rolling OLS
        regression, yielding point estimates and confidence bounds.
        Useful for detecting regime changes in market liquidity.

        Parameters:
            dataset: Dataset containing price and volume data.
            price_col: Price column name.
            volume_col: Volume column name.
            window: Rolling regression window (must be >= 5).
        """
        try:
            from wraquant.microstructure.liquidity import (
                lambda_kyle_rolling as _kyle_rolling,
            )

            df = ctx.get_dataset(dataset)
            prices = df[price_col].dropna()
            volume = df[volume_col].dropna()

            n = min(len(prices), len(volume))
            prices = prices.iloc[-n:]
            volume = volume.iloc[-n:]

            result_df = _kyle_rolling(prices, volume, window=window)

            stored = ctx.store_dataset(
                f"kyle_lambda_rolling_{dataset}",
                result_df,
                source_op="kyle_lambda_rolling",
                parent=dataset,
            )

            latest = result_df.dropna()

            return _sanitize_for_json(
                {
                    "tool": "kyle_lambda_rolling",
                    "dataset": dataset,
                    "window": window,
                    "latest_lambda": (
                        float(latest["lambda"].iloc[-1]) if len(latest) > 0 else None
                    ),
                    "latest_ci_lower": (
                        float(latest["ci_lower"].iloc[-1]) if len(latest) > 0 else None
                    ),
                    "latest_ci_upper": (
                        float(latest["ci_upper"].iloc[-1]) if len(latest) > 0 else None
                    ),
                    "mean_lambda": (
                        float(latest["lambda"].mean()) if len(latest) > 0 else None
                    ),
                    "significant_pct": (
                        float((latest["ci_lower"] > 0).mean())
                        if len(latest) > 0
                        else None
                    ),
                    "observations": n,
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "kyle_lambda_rolling"}

    @mcp.tool()
    def amihud_rolling(
        dataset: str,
        price_col: str = "close",
        volume_col: str = "volume",
        window: int = 20,
    ) -> dict[str, Any]:
        """Compute rolling Amihud (2002) illiquidity ratio with normalization.

        Tracks how an asset's liquidity evolves over time. Higher values
        indicate less liquidity (more price impact per unit of volume).

        Parameters:
            dataset: Dataset containing price and volume data.
            price_col: Price column name (used to compute returns).
            volume_col: Volume column name (dollar volume).
            window: Rolling window size (default 20 for ~1 month).
        """
        try:
            import pandas as pd

            from wraquant.microstructure.liquidity import (
                amihud_rolling as _amihud_rolling,
            )

            df = ctx.get_dataset(dataset)
            prices = df[price_col].dropna()
            volume = df[volume_col].dropna()

            n = min(len(prices), len(volume))
            prices = prices.iloc[-n:]
            volume = volume.iloc[-n:]

            returns = prices.pct_change().dropna()
            vol_aligned = volume.iloc[-len(returns) :]

            rolling_amihud = _amihud_rolling(returns, vol_aligned, window=window)

            result_df = pd.DataFrame({"amihud_rolling": rolling_amihud})
            stored = ctx.store_dataset(
                f"amihud_rolling_{dataset}",
                result_df,
                source_op="amihud_rolling",
                parent=dataset,
            )

            clean = rolling_amihud.dropna()

            return _sanitize_for_json(
                {
                    "tool": "amihud_rolling",
                    "dataset": dataset,
                    "window": window,
                    "latest_amihud": float(clean.iloc[-1]) if len(clean) > 0 else None,
                    "mean_amihud": float(clean.mean()) if len(clean) > 0 else None,
                    "max_amihud": float(clean.max()) if len(clean) > 0 else None,
                    "observations": n,
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "amihud_rolling"}

    @mcp.tool()
    def corwin_schultz_spread(
        dataset: str,
        high_col: str = "high",
        low_col: str = "low",
    ) -> dict[str, Any]:
        """Estimate bid-ask spread from high-low prices (Corwin & Schultz 2012).

        Uses daily high and low prices to estimate the effective spread.
        More robust than Roll's estimator when only OHLC data is available.

        Parameters:
            dataset: Dataset containing high and low price columns.
            high_col: High price column name.
            low_col: Low price column name.
        """
        try:
            import pandas as pd

            from wraquant.microstructure.liquidity import (
                corwin_schultz_spread as _cs_spread,
            )

            df = ctx.get_dataset(dataset)
            high = df[high_col].dropna()
            low = df[low_col].dropna()

            n = min(len(high), len(low))
            high = high.iloc[-n:]
            low = low.iloc[-n:]

            spread = _cs_spread(high, low)

            result_df = pd.DataFrame({"corwin_schultz_spread": spread})
            stored = ctx.store_dataset(
                f"cs_spread_{dataset}",
                result_df,
                source_op="corwin_schultz_spread",
                parent=dataset,
            )

            clean = spread.dropna()

            return _sanitize_for_json(
                {
                    "tool": "corwin_schultz_spread",
                    "dataset": dataset,
                    "latest_spread": float(clean.iloc[-1]) if len(clean) > 0 else None,
                    "mean_spread": float(clean.mean()) if len(clean) > 0 else None,
                    "median_spread": float(clean.median()) if len(clean) > 0 else None,
                    "observations": n,
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "corwin_schultz_spread"}

    @mcp.tool()
    def roll_spread_tool(
        dataset: str,
        price_col: str = "close",
    ) -> dict[str, Any]:
        """Compute Roll's (1984) implied spread from serial price autocovariance.

        Estimates the effective bid-ask spread from trade prices alone,
        without needing quote data. Returns NaN if serial covariance is
        non-negative (model assumption violated).

        Parameters:
            dataset: Dataset containing a price column.
            price_col: Price column name.
        """
        try:
            from wraquant.microstructure.liquidity import roll_spread as _roll_spread

            df = ctx.get_dataset(dataset)
            prices = df[price_col].dropna()

            spread = _roll_spread(prices)

            ctx._log("roll_spread", dataset, price_col=price_col)

            return _sanitize_for_json(
                {
                    "tool": "roll_spread",
                    "dataset": dataset,
                    "roll_spread": float(spread),
                    "spread_is_valid": not (spread != spread),  # not NaN
                    "observations": len(prices),
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "roll_spread_tool"}

    @mcp.tool()
    def effective_spread_tool(
        dataset: str,
        trade_price_col: str = "trade_price",
        midpoint_col: str = "midpoint",
    ) -> dict[str, Any]:
        """Compute effective spread: 2 * |trade_price - midpoint|.

        The standard measure of execution cost. Requires trade prices
        and contemporaneous bid-ask midpoints.

        Parameters:
            dataset: Dataset with trade price and midpoint columns.
            trade_price_col: Trade price column name.
            midpoint_col: Bid-ask midpoint column name.
        """
        try:
            import pandas as pd

            from wraquant.microstructure.liquidity import (
                effective_spread as _effective_spread,
            )

            df = ctx.get_dataset(dataset)
            trades = df[trade_price_col].dropna()
            mids = df[midpoint_col].dropna()

            n = min(len(trades), len(mids))
            trades = trades.iloc[-n:]
            mids = mids.iloc[-n:]

            spreads = _effective_spread(trades, mids)

            result_df = pd.DataFrame({"effective_spread": spreads})
            stored = ctx.store_dataset(
                f"eff_spread_{dataset}",
                result_df,
                source_op="effective_spread",
                parent=dataset,
            )

            return _sanitize_for_json(
                {
                    "tool": "effective_spread",
                    "dataset": dataset,
                    "mean_effective_spread": (
                        float(spreads.mean()) if len(spreads) > 0 else None
                    ),
                    "median_effective_spread": (
                        float(spreads.median()) if len(spreads) > 0 else None
                    ),
                    "max_effective_spread": (
                        float(spreads.max()) if len(spreads) > 0 else None
                    ),
                    "observations": n,
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "effective_spread_tool"}

    @mcp.tool()
    def order_flow_imbalance(
        dataset: str,
        buy_vol_col: str = "buy_volume",
        sell_vol_col: str = "sell_volume",
    ) -> dict[str, Any]:
        """Compute rolling order flow imbalance (OFI).

        Measures directional pressure as (buy - sell) / (buy + sell)
        averaged over a rolling window. Values near +1 indicate strong
        buying; near -1 indicates selling pressure.

        Parameters:
            dataset: Dataset with buy and sell volume columns.
            buy_vol_col: Buy-initiated volume column name.
            sell_vol_col: Sell-initiated volume column name.
        """
        try:
            import pandas as pd

            from wraquant.microstructure.toxicity import order_flow_imbalance as _ofi

            df = ctx.get_dataset(dataset)
            buy_vol = df[buy_vol_col].dropna()
            sell_vol = df[sell_vol_col].dropna()

            n = min(len(buy_vol), len(sell_vol))
            buy_vol = buy_vol.iloc[-n:]
            sell_vol = sell_vol.iloc[-n:]

            ofi = _ofi(buy_vol, sell_vol, window=20)

            result_df = pd.DataFrame({"order_flow_imbalance": ofi})
            stored = ctx.store_dataset(
                f"ofi_{dataset}",
                result_df,
                source_op="order_flow_imbalance",
                parent=dataset,
            )

            clean = ofi.dropna()

            return _sanitize_for_json(
                {
                    "tool": "order_flow_imbalance",
                    "dataset": dataset,
                    "latest_ofi": float(clean.iloc[-1]) if len(clean) > 0 else None,
                    "mean_ofi": float(clean.mean()) if len(clean) > 0 else None,
                    "buy_dominant_pct": (
                        float((clean > 0).mean()) if len(clean) > 0 else None
                    ),
                    "observations": n,
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "order_flow_imbalance"}

    @mcp.tool()
    def trade_classification_tool(
        dataset: str,
        price_col: str = "trade_price",
        bid_col: str = "bid",
        ask_col: str = "ask",
    ) -> dict[str, Any]:
        """Classify trades as buyer- or seller-initiated via Lee-Ready algorithm.

        Uses the quote test (price vs midpoint) with tick test fallback
        to classify each trade as +1 (buy) or -1 (sell).

        Parameters:
            dataset: Dataset with trade price, bid, and ask columns.
            price_col: Trade price column name.
            bid_col: Best bid price column name.
            ask_col: Best ask price column name.
        """
        try:
            import pandas as pd

            from wraquant.microstructure.toxicity import (
                trade_classification as _trade_class,
            )

            df = ctx.get_dataset(dataset)
            trades = df[price_col].dropna()
            bid = df[bid_col].dropna()
            ask = df[ask_col].dropna()

            n = min(len(trades), len(bid), len(ask))
            trades = trades.iloc[-n:]
            bid = bid.iloc[-n:]
            ask = ask.iloc[-n:]

            direction = _trade_class(trades, bid, ask)

            result_df = pd.DataFrame({"trade_direction": direction})
            stored = ctx.store_dataset(
                f"trade_class_{dataset}",
                result_df,
                source_op="trade_classification",
                parent=dataset,
            )

            buy_pct = float((direction == 1).mean()) if len(direction) > 0 else None
            sell_pct = float((direction == -1).mean()) if len(direction) > 0 else None

            return _sanitize_for_json(
                {
                    "tool": "trade_classification",
                    "dataset": dataset,
                    "buy_pct": buy_pct,
                    "sell_pct": sell_pct,
                    "total_trades": n,
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "trade_classification_tool"}

    @mcp.tool()
    def intraday_volatility_pattern(
        dataset: str,
        price_col: str = "close",
        time_col: str | None = None,
    ) -> dict[str, Any]:
        """Estimate the intraday volatility pattern (U-shape or J-shape).

        Computes average absolute return at each hourly bucket, revealing
        the diurnal volatility cycle. Requires intraday price data with
        a DatetimeIndex.

        Parameters:
            dataset: Dataset containing intraday prices.
            price_col: Price column name.
            time_col: Unused (time is extracted from the DatetimeIndex).
                Kept for API consistency.
        """
        try:
            import pandas as pd

            from wraquant.microstructure.market_quality import (
                intraday_volatility_pattern as _intraday_vol,
            )

            df = ctx.get_dataset(dataset)
            prices = df[price_col].dropna()

            if not isinstance(prices.index, pd.DatetimeIndex):
                return {
                    "error": "Dataset must have a DatetimeIndex with intraday timestamps."
                }

            pattern = _intraday_vol(prices, freq="h")

            result_df = pd.DataFrame({"intraday_volatility": pattern})
            stored = ctx.store_dataset(
                f"intraday_vol_{dataset}",
                result_df,
                source_op="intraday_volatility_pattern",
                parent=dataset,
            )

            return _sanitize_for_json(
                {
                    "tool": "intraday_volatility_pattern",
                    "dataset": dataset,
                    "peak_hour": int(pattern.idxmax()) if len(pattern) > 0 else None,
                    "trough_hour": int(pattern.idxmin()) if len(pattern) > 0 else None,
                    "peak_vol": float(pattern.max()) if len(pattern) > 0 else None,
                    "trough_vol": float(pattern.min()) if len(pattern) > 0 else None,
                    "n_hours": len(pattern),
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "intraday_volatility_pattern"}

    @mcp.tool()
    def information_share(
        dataset_a: str,
        dataset_b: str,
        column: str = "close",
    ) -> dict[str, Any]:
        """Compute Hasbrouck information share between two venue price series.

        Measures each venue's contribution to price discovery via
        Cholesky-based variance decomposition. The venue with the higher
        share contributes more to the efficient price.

        Parameters:
            dataset_a: First venue dataset name.
            dataset_b: Second venue dataset name.
            column: Price column name (same in both datasets).
        """
        try:
            from wraquant.microstructure.market_quality import (
                hasbrouck_information_share,
            )

            df_a = ctx.get_dataset(dataset_a)
            df_b = ctx.get_dataset(dataset_b)

            prices_a = df_a[column].dropna()
            prices_b = df_b[column].dropna()

            # Align to common index
            common_idx = prices_a.index.intersection(prices_b.index)
            if len(common_idx) < 20:
                return {
                    "error": f"Insufficient common observations ({len(common_idx)}). Need at least 20."
                }

            prices_a = prices_a.loc[common_idx]
            prices_b = prices_b.loc[common_idx]

            result = hasbrouck_information_share([prices_a, prices_b])

            ctx._log(
                "information_share",
                f"{dataset_a}_vs_{dataset_b}",
                column=column,
            )

            return _sanitize_for_json(
                {
                    "tool": "information_share",
                    "dataset_a": dataset_a,
                    "dataset_b": dataset_b,
                    "column": column,
                    "info_share_a": float(result["midpoint"][0]),
                    "info_share_b": float(result["midpoint"][1]),
                    "upper_bound_a": float(result["upper"][0]),
                    "upper_bound_b": float(result["upper"][1]),
                    "lower_bound_a": float(result["lower"][0]),
                    "lower_bound_b": float(result["lower"][1]),
                    "dominant_venue": (
                        dataset_a
                        if result["midpoint"][0] > result["midpoint"][1]
                        else dataset_b
                    ),
                    "common_observations": len(common_idx),
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "information_share"}

    @mcp.tool()
    def liquidity_commonality(
        dataset: str,
        market_dataset: str,
        column: str = "close",
        market_column: str = "close",
    ) -> dict[str, Any]:
        """Measure systematic liquidity risk via commonality regression.

        Regresses changes in the asset's illiquidity on changes in
        market-wide illiquidity to measure co-movement. High commonality
        means the asset becomes illiquid exactly when the market does.

        Parameters:
            dataset: Individual asset dataset name.
            market_dataset: Market-wide dataset name (e.g., SPY, index).
            column: Price column in the asset dataset.
            market_column: Price column in the market dataset.
        """
        try:
            import pandas as pd

            from wraquant.microstructure.liquidity import (
                amihud_rolling as _amihud_rolling,
            )
            from wraquant.microstructure.liquidity import (
                liquidity_commonality as _liq_common,
            )

            df_asset = ctx.get_dataset(dataset)
            df_market = ctx.get_dataset(market_dataset)

            prices_asset = df_asset[column].dropna()
            prices_market = df_market[market_column].dropna()

            # Align
            common_idx = prices_asset.index.intersection(prices_market.index)
            if len(common_idx) < 80:
                return {
                    "error": f"Insufficient common observations ({len(common_idx)}). Need at least 80."
                }

            prices_asset = prices_asset.loc[common_idx]
            prices_market = prices_market.loc[common_idx]

            # Compute Amihud illiquidity for both
            returns_asset = prices_asset.pct_change().dropna()
            returns_market = prices_market.pct_change().dropna()

            # Use price * volume proxy if volume exists, else use returns magnitude
            vol_col_asset = "volume" if "volume" in df_asset.columns else None
            vol_col_market = "volume" if "volume" in df_market.columns else None

            if vol_col_asset and vol_col_market:
                vol_asset = df_asset[vol_col_asset].loc[common_idx].dropna()
                vol_market = df_market[vol_col_market].loc[common_idx].dropna()
                # Align returns and volume
                common_rv = returns_asset.index.intersection(
                    vol_asset.index
                ).intersection(vol_market.index)
                returns_asset = returns_asset.loc[common_rv]
                returns_market = returns_market.loc[common_rv]
                vol_asset = vol_asset.loc[common_rv]
                vol_market = vol_market.loc[common_rv]
            else:
                # Use absolute returns as proxy for illiquidity
                vol_asset = pd.Series(1.0, index=returns_asset.index)
                vol_market = pd.Series(1.0, index=returns_market.index)
                common_rv = returns_asset.index.intersection(returns_market.index)
                returns_asset = returns_asset.loc[common_rv]
                returns_market = returns_market.loc[common_rv]
                vol_asset = vol_asset.loc[common_rv]
                vol_market = vol_market.loc[common_rv]

            illiq_asset = _amihud_rolling(returns_asset, vol_asset, window=21)
            illiq_market = _amihud_rolling(returns_market, vol_market, window=21)

            r_squared = _liq_common(illiq_asset, illiq_market, window=60)

            result_df = pd.DataFrame({"liquidity_commonality": r_squared})
            stored = ctx.store_dataset(
                f"liq_common_{dataset}",
                result_df,
                source_op="liquidity_commonality",
                parent=dataset,
            )

            clean = r_squared.dropna()

            return _sanitize_for_json(
                {
                    "tool": "liquidity_commonality",
                    "dataset": dataset,
                    "market_dataset": market_dataset,
                    "latest_r_squared": (
                        float(clean.iloc[-1]) if len(clean) > 0 else None
                    ),
                    "mean_r_squared": float(clean.mean()) if len(clean) > 0 else None,
                    "max_r_squared": float(clean.max()) if len(clean) > 0 else None,
                    "high_commonality_pct": (
                        float((clean > 0.3).mean()) if len(clean) > 0 else None
                    ),
                    "observations": len(common_rv),
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "liquidity_commonality"}
