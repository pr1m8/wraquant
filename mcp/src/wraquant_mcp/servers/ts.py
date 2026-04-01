"""Time series analysis MCP tools.

Tools: forecast, decompose, changepoint_detect, anomaly_detect,
seasonality_analysis.
"""

from __future__ import annotations

from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_ts_tools(mcp, ctx: AnalysisContext) -> None:
    """Register time-series-specific tools on the MCP server."""

    @mcp.tool()
    def forecast(
        dataset: str,
        column: str = "returns",
        method: str = "auto",
        horizon: int = 20,
    ) -> dict[str, Any]:
        """Forecast a time series.

        Parameters:
            dataset: Dataset containing the series.
            column: Column to forecast.
            method: Forecasting method. Options:
                'auto' (automatic selection), 'arima', 'ets'
                (exponential smoothing), 'theta', 'ensemble'.
            horizon: Number of periods to forecast.
        """
        from wraquant.ts.forecasting import (
            auto_arima,
            auto_forecast,
            ensemble_forecast,
            exponential_smoothing,
            theta_forecast,
        )

        df = ctx.get_dataset(dataset)
        data = df[column].dropna()

        methods = {
            "auto": lambda: auto_forecast(data, h=horizon),
            "arima": lambda: auto_arima(data, h=horizon),
            "ets": lambda: exponential_smoothing(data, h=horizon),
            "theta": lambda: theta_forecast(data, h=horizon),
            "ensemble": lambda: ensemble_forecast(data, h=horizon),
        }

        func = methods.get(method)
        if func is None:
            return {"error": f"Unknown method '{method}'. Options: {list(methods)}"}

        result = func()

        import pandas as pd

        if isinstance(result, dict) and "forecast" in result:
            fc_values = result["forecast"]
        else:
            fc_values = result

        fc_df = pd.DataFrame({"forecast": fc_values})
        stored = ctx.store_dataset(
            f"forecast_{dataset}_{method}", fc_df,
            source_op="forecast", parent=dataset,
        )

        return _sanitize_for_json({
            "tool": "forecast",
            "method": method,
            "horizon": horizon,
            "result": result if isinstance(result, dict) else {"forecast": result},
            **stored,
        })

    @mcp.tool()
    def decompose(
        dataset: str,
        column: str = "close",
        method: str = "stl",
        period: int = 252,
    ) -> dict[str, Any]:
        """Decompose a time series into trend, seasonal, and residual.

        Parameters:
            dataset: Dataset containing the series.
            column: Column to decompose.
            method: Decomposition method. Options:
                'stl', 'seasonal', 'ssa', 'emd'.
            period: Seasonal period (252 for daily financial data).
        """
        from wraquant.ts.decomposition import (
            seasonal_decompose,
            ssa_decompose,
            stl_decompose,
        )

        df = ctx.get_dataset(dataset)
        data = df[column].dropna()

        methods = {
            "stl": lambda: stl_decompose(data, period=period),
            "seasonal": lambda: seasonal_decompose(data, period=period),
            "ssa": lambda: ssa_decompose(data),
        }

        func = methods.get(method)
        if func is None:
            return {"error": f"Unknown method '{method}'. Options: {list(methods)}"}

        result = func()

        import pandas as pd

        if isinstance(result, dict):
            comp_df = pd.DataFrame({
                k: v for k, v in result.items()
                if hasattr(v, "__len__") and not isinstance(v, str)
            })
        else:
            comp_df = pd.DataFrame({"result": [str(result)]})

        stored = ctx.store_dataset(
            f"decomp_{dataset}_{method}", comp_df,
            source_op="decompose", parent=dataset,
        )

        return _sanitize_for_json({
            "tool": "decompose",
            "method": method,
            "period": period,
            **stored,
        })

    @mcp.tool()
    def changepoint_detect(
        dataset: str,
        column: str = "returns",
        method: str = "pelt",
        max_changepoints: int = 5,
    ) -> dict[str, Any]:
        """Detect structural change points in a time series.

        Parameters:
            dataset: Dataset containing the series.
            column: Column to analyze.
            method: Detection method ('pelt', 'bayesian', 'cusum').
            max_changepoints: Maximum number of changepoints to detect.
        """
        df = ctx.get_dataset(dataset)
        data = df[column].dropna()

        if method == "cusum":
            from wraquant.ts.changepoint import cusum

            result = cusum(data)
        else:
            from wraquant.ts.changepoint import detect_changepoints

            result = detect_changepoints(
                data, method=method, n_bkps=max_changepoints,
            )

        return _sanitize_for_json({
            "tool": "changepoint_detect",
            "dataset": dataset,
            "column": column,
            "method": method,
            "result": result,
        })

    @mcp.tool()
    def anomaly_detect(
        dataset: str,
        column: str = "returns",
        method: str = "isolation_forest",
        contamination: float = 0.05,
    ) -> dict[str, Any]:
        """Detect anomalies/outliers in a time series.

        Parameters:
            dataset: Dataset containing the series.
            column: Column to analyze.
            method: Detection method. Options:
                'isolation_forest', 'grubbs'.
            contamination: Expected proportion of anomalies (0-1).
        """
        from wraquant.ts.anomaly import grubbs_test_ts, isolation_forest_ts

        df = ctx.get_dataset(dataset)
        data = df[column].dropna()

        if method == "grubbs":
            result = grubbs_test_ts(data)
        else:
            result = isolation_forest_ts(data, contamination=contamination)

        import pandas as pd

        if isinstance(result, dict) and "anomalies" in result:
            anom_df = pd.DataFrame({"anomaly": result["anomalies"]})
            stored = ctx.store_dataset(
                f"anomalies_{dataset}", anom_df,
                source_op="anomaly_detect", parent=dataset,
            )
        else:
            stored = {}

        return _sanitize_for_json({
            "tool": "anomaly_detect",
            "method": method,
            "contamination": contamination,
            "result": result,
            **stored,
        })

    @mcp.tool()
    def seasonality_analysis(
        dataset: str,
        column: str = "close",
    ) -> dict[str, Any]:
        """Detect and analyze seasonal patterns in a time series.

        Automatically detects the dominant seasonal period and
        computes seasonal strength.

        Parameters:
            dataset: Dataset containing the series.
            column: Column to analyze.
        """
        from wraquant.ts.seasonality import detect_seasonality, seasonal_strength

        df = ctx.get_dataset(dataset)
        data = df[column].dropna()

        period = detect_seasonality(data)
        strength = seasonal_strength(data, period=period if isinstance(period, int) else 252)

        return _sanitize_for_json({
            "tool": "seasonality_analysis",
            "dataset": dataset,
            "column": column,
            "detected_period": period,
            "seasonal_strength": strength,
            "observations": len(data),
        })
