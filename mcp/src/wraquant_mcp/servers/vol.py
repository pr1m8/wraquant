"""Volatility modeling MCP tools.

Tools: forecast_volatility, news_impact_curve, model_selection,
realized_volatility, ewma_volatility.
"""

from __future__ import annotations

from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_vol_tools(mcp, ctx: AnalysisContext) -> None:
    """Register volatility-specific tools on the MCP server."""

    @mcp.tool()
    def forecast_volatility(
        model_name: str,
        horizon: int = 10,
    ) -> dict[str, Any]:
        """Forecast volatility from a previously fitted GARCH model.

        Parameters:
            model_name: Name of stored GARCH model (from fit_garch).
            horizon: Forecast horizon in periods.
        """
        from wraquant.vol.models import garch_forecast

        model = ctx.get_model(model_name)
        forecast = garch_forecast(model, horizon=horizon)

        import pandas as pd

        fc_df = pd.DataFrame({
            "forecast_vol": forecast["forecast"],
        })
        stored = ctx.store_dataset(
            f"{model_name}_forecast", fc_df,
            source_op="forecast_volatility",
        )

        return _sanitize_for_json({
            "tool": "forecast_volatility",
            "model": model_name,
            "horizon": horizon,
            "forecast": forecast["forecast"].tolist()
            if hasattr(forecast["forecast"], "tolist")
            else list(forecast["forecast"]),
            **stored,
        })

    @mcp.tool()
    def news_impact_curve(
        model_name: str,
        n_points: int = 100,
    ) -> dict[str, Any]:
        """Compute the news impact curve for a fitted GARCH model.

        Shows how positive and negative return shocks affect
        conditional variance -- reveals the leverage effect.

        Parameters:
            model_name: Name of stored GARCH model.
            n_points: Number of shock points to evaluate.
        """
        from wraquant.vol.models import news_impact_curve as _nic

        model = ctx.get_model(model_name)
        result = _nic(model, n_points=n_points)

        import pandas as pd

        nic_df = pd.DataFrame({
            "shock": result["shocks"],
            "variance": result["variances"],
        })
        stored = ctx.store_dataset(
            f"{model_name}_nic", nic_df,
            source_op="news_impact_curve",
        )

        return _sanitize_for_json({
            "tool": "news_impact_curve",
            "model": model_name,
            "asymmetry": result.get("asymmetry"),
            **stored,
        })

    @mcp.tool()
    def model_selection(
        dataset: str,
        column: str = "returns",
        models: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare GARCH-family models by AIC/BIC.

        Fits multiple models and ranks them by information criteria.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
            models: List of model types to compare.
                Defaults to ['GARCH', 'EGARCH', 'GJR'].
        """
        from wraquant.vol.models import garch_model_selection

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna().values

        if models is None:
            models = ["GARCH", "EGARCH", "GJR"]

        result = garch_model_selection(returns, models=models)

        return _sanitize_for_json({
            "tool": "model_selection",
            "dataset": dataset,
            "ranking": result,
        })

    @mcp.tool()
    def realized_volatility(
        dataset: str,
        method: str = "yang_zhang",
        window: int = 21,
    ) -> dict[str, Any]:
        """Compute realized volatility from OHLC data.

        Parameters:
            dataset: Dataset with OHLC columns (open, high, low, close).
            method: Estimator method. Options: 'close_to_close',
                'parkinson', 'garman_klass', 'rogers_satchell',
                'yang_zhang'.
            window: Rolling window in periods.
        """
        from wraquant.vol import realized as rv

        df = ctx.get_dataset(dataset)

        estimators = {
            "close_to_close": rv.realized_volatility,
            "parkinson": rv.parkinson,
            "garman_klass": rv.garman_klass,
            "rogers_satchell": rv.rogers_satchell,
            "yang_zhang": rv.yang_zhang,
        }

        func = estimators.get(method)
        if func is None:
            return {"error": f"Unknown method '{method}'. Options: {list(estimators)}"}

        if method == "close_to_close":
            result = func(df["close"], window=window)
        else:
            result = func(
                df["open"], df["high"], df["low"], df["close"],
                window=window,
            )

        import pandas as pd

        vol_df = pd.DataFrame({"realized_vol": result})
        stored = ctx.store_dataset(
            f"rvol_{dataset}_{method}", vol_df,
            source_op="realized_volatility", parent=dataset,
        )

        return _sanitize_for_json({
            "tool": "realized_volatility",
            "method": method,
            "window": window,
            "current_vol": float(result.iloc[-1]) if len(result) > 0 else None,
            "mean_vol": float(result.mean()),
            **stored,
        })

    @mcp.tool()
    def ewma_volatility(
        dataset: str,
        column: str = "returns",
        span: int = 30,
    ) -> dict[str, Any]:
        """Compute EWMA (RiskMetrics-style) volatility.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
            span: EWMA span (decay factor = 2/(span+1)).
        """
        from wraquant.vol.models import ewma_volatility as _ewma

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()

        result = _ewma(returns, span=span)

        import pandas as pd

        vol_df = pd.DataFrame({"ewma_vol": result})
        stored = ctx.store_dataset(
            f"ewma_{dataset}", vol_df,
            source_op="ewma_volatility", parent=dataset,
        )

        return _sanitize_for_json({
            "tool": "ewma_volatility",
            "span": span,
            "current_vol": float(result.iloc[-1]) if len(result) > 0 else None,
            "mean_vol": float(result.mean()),
            **stored,
        })
