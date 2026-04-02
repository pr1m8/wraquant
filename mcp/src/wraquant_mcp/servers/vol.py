"""Volatility modeling MCP tools.

Tools: forecast_volatility, news_impact_curve, model_selection,
realized_volatility, ewma_volatility, hawkes_fit, stochastic_vol,
variance_risk_premium, bipower_variation, jump_detection, garch_rolling.
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

    @mcp.tool()
    def hawkes_fit(
        dataset: str,
        column: str = "returns",
    ) -> dict[str, Any]:
        """Fit a Hawkes self-exciting process for volatility clustering.

        Models how large return shocks temporarily increase the
        intensity of future shocks -- directly captures volatility
        clustering without a GARCH specification.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
        """
        import numpy as np

        from wraquant.vol.models import hawkes_process

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna().values

        # Use absolute returns as event magnitudes
        events = np.abs(returns)

        result = hawkes_process(events)

        model_name = f"hawkes_{dataset}"
        stored = ctx.store_model(
            model_name, result,
            model_type="hawkes",
            source_dataset=dataset,
        )

        return _sanitize_for_json({
            "tool": "hawkes_fit",
            "dataset": dataset,
            **stored,
            "result": {k: v for k, v in result.items()
                       if not hasattr(v, "__len__") or isinstance(v, str)}
            if isinstance(result, dict) else str(result),
        })

    @mcp.tool()
    def stochastic_vol(
        dataset: str,
        column: str = "returns",
    ) -> dict[str, Any]:
        """Fit a stochastic volatility model via particle filter.

        The SV model treats log-volatility as a latent AR(1) process,
        providing a richer description of vol dynamics than GARCH.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
        """
        from wraquant.vol.models import stochastic_vol_sv

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()

        result = stochastic_vol_sv(returns)

        import pandas as pd

        if isinstance(result, dict) and "filtered_vol" in result:
            sv_df = pd.DataFrame({"stochastic_vol": result["filtered_vol"]})
            stored = ctx.store_dataset(
                f"sv_{dataset}", sv_df,
                source_op="stochastic_vol", parent=dataset,
            )
        else:
            stored = {}

        model_name = f"sv_model_{dataset}"
        model_stored = ctx.store_model(
            model_name, result,
            model_type="stochastic_vol",
            source_dataset=dataset,
        )

        return _sanitize_for_json({
            "tool": "stochastic_vol",
            "dataset": dataset,
            **model_stored,
            **stored,
            "params": {k: v for k, v in result.items()
                       if not hasattr(v, "__len__") or isinstance(v, str)}
            if isinstance(result, dict) else str(result),
        })

    @mcp.tool()
    def variance_risk_premium(
        dataset: str,
        realized_col: str = "realized_vol",
        implied_col: str = "implied_vol",
    ) -> dict[str, Any]:
        """Compute the variance risk premium (VRP).

        VRP = implied variance - realized variance. A positive VRP
        indicates investors pay a premium for volatility protection.

        Parameters:
            dataset: Dataset containing both realized and implied vol.
            realized_col: Realized volatility column.
            implied_col: Implied volatility column.
        """
        from wraquant.vol.models import variance_risk_premium as _vrp

        df = ctx.get_dataset(dataset)
        realized = df[realized_col].dropna()
        implied = df[implied_col].dropna()

        n = min(len(realized), len(implied))
        realized = realized.iloc[-n:]
        implied = implied.iloc[-n:]

        result = _vrp(implied, realized)

        import pandas as pd

        if isinstance(result, dict) and "vrp" in result:
            vrp_df = pd.DataFrame({"vrp": result["vrp"]})
            stored = ctx.store_dataset(
                f"vrp_{dataset}", vrp_df,
                source_op="variance_risk_premium", parent=dataset,
            )
        else:
            stored = {}

        return _sanitize_for_json({
            "tool": "variance_risk_premium",
            "dataset": dataset,
            **stored,
            "result": result,
        })

    @mcp.tool()
    def bipower_variation(
        dataset: str,
        column: str = "returns",
        window: int = 20,
    ) -> dict[str, Any]:
        """Compute bipower variation -- jump-robust volatility estimator.

        Uses products of adjacent absolute returns to estimate the
        continuous (diffusive) component of volatility, filtering
        out the effect of jumps.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
            window: Rolling window in periods.
        """
        from wraquant.vol.realized import bipower_variation as _bpv

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()

        result = _bpv(returns, window=window)

        import pandas as pd

        bpv_df = pd.DataFrame({"bipower_vol": result})
        stored = ctx.store_dataset(
            f"bpv_{dataset}", bpv_df,
            source_op="bipower_variation", parent=dataset,
        )

        return _sanitize_for_json({
            "tool": "bipower_variation",
            "dataset": dataset,
            "window": window,
            "current_bpv": float(result.iloc[-1]) if len(result) > 0 else None,
            "mean_bpv": float(result.mean()),
            **stored,
        })

    @mcp.tool()
    def jump_detection(
        dataset: str,
        column: str = "returns",
    ) -> dict[str, Any]:
        """Run the Barndorff-Nielsen & Shephard jump detection test.

        Compares realized variance to bipower variation to detect
        whether jumps are present. Significant positive difference
        indicates jump activity.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
        """
        from wraquant.vol.realized import jump_test_bns

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()

        result = jump_test_bns(returns)

        return _sanitize_for_json({
            "tool": "jump_detection",
            "dataset": dataset,
            "column": column,
            "observations": len(returns),
            **(result if isinstance(result, dict) else {"result": str(result)}),
        })

    @mcp.tool()
    def garch_rolling(
        dataset: str,
        column: str = "returns",
        window: int = 500,
        horizon: int = 1,
    ) -> dict[str, Any]:
        """Rolling GARCH volatility forecast.

        Refits a GARCH model on a rolling window and produces
        one-step-ahead (or multi-step) forecasts at each point.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
            window: Rolling estimation window.
            horizon: Forecast horizon in periods.
        """
        from wraquant.vol.models import garch_rolling_forecast

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()

        result = garch_rolling_forecast(returns, window=window, horizon=horizon)

        import pandas as pd

        if isinstance(result, dict) and "forecasts" in result:
            fc_df = pd.DataFrame({"garch_forecast": result["forecasts"]})
            stored = ctx.store_dataset(
                f"garch_rolling_{dataset}", fc_df,
                source_op="garch_rolling", parent=dataset,
            )
        else:
            stored = {}

        return _sanitize_for_json({
            "tool": "garch_rolling",
            "dataset": dataset,
            "window": window,
            "horizon": horizon,
            **stored,
            "result": {k: v for k, v in result.items()
                       if not hasattr(v, "__len__") or isinstance(v, str)}
            if isinstance(result, dict) else str(result),
        })
