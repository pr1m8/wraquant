"""Econometric analysis MCP tools.

Tools: var_model, panel_regression, event_study_econometric,
structural_break, cointegration_johansen, impulse_response.
"""

from __future__ import annotations

import json
from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_econometrics_tools(mcp, ctx: AnalysisContext) -> None:
    """Register econometrics tools on the MCP server."""

    @mcp.tool()
    def var_model(
        dataset: str,
        columns_json: str = "[]",
        lags: int = 2,
        horizon: int = 10,
    ) -> dict[str, Any]:
        """Fit a Vector Autoregression (VAR) model and forecast.

        Selects lag order by information criterion and estimates a
        reduced-form VAR. Returns coefficients, residual covariance,
        selected lag order, model fit statistics, and out-of-sample
        forecasts.

        Parameters:
            dataset: Dataset with multivariate time series.
            columns_json: JSON array of column names to include.
                If empty, uses all numeric columns.
            lags: Maximum lag order to test.
            horizon: Forecast horizon (number of periods ahead).
        """
        import numpy as np
        import pandas as pd

        from wraquant.econometrics.timeseries import var_model as _var

        df = ctx.get_dataset(dataset)

        columns = json.loads(columns_json)
        if columns:
            data = df[columns].dropna()
        else:
            data = df.select_dtypes(include=[np.number]).dropna()

        result = _var(data, max_lags=lags, ic="aic")

        # Generate forecast
        forecast_fn = result.get("forecast")
        forecast = None
        if forecast_fn is not None:
            try:
                forecast_arr = forecast_fn(horizon)
                forecast_df = pd.DataFrame(
                    forecast_arr, columns=list(data.columns),
                )
                ctx.store_dataset(
                    f"var_forecast_{dataset}", forecast_df,
                    source_op="var_model", parent=dataset,
                )
                forecast = forecast_arr.tolist()
            except Exception:
                forecast = None

        model_name = f"var_{dataset}"
        stored = ctx.store_model(
            model_name, result,
            model_type="var",
            source_dataset=dataset,
        )

        return _sanitize_for_json({
            "tool": "var_model",
            "dataset": dataset,
            "columns": list(data.columns),
            "lag_order": result.get("lag_order"),
            "aic": result.get("aic"),
            "bic": result.get("bic"),
            "horizon": horizon,
            "forecast": forecast,
            **stored,
        })

    @mcp.tool()
    def panel_regression(
        dataset: str,
        y_col: str,
        x_cols_json: str = "[]",
        entity_col: str = "entity",
        time_col: str = "time",
        method: str = "fe",
    ) -> dict[str, Any]:
        """Run panel data regression.

        Supports pooled OLS, fixed effects (within estimator), and
        random effects estimators.

        Parameters:
            dataset: Panel dataset with entity and time identifiers.
            y_col: Dependent variable column.
            x_cols_json: JSON array of independent variable column names.
            entity_col: Column identifying the cross-sectional unit.
            time_col: Column identifying the time period.
            method: Estimation method:
                - 'fe' or 'fixed_effects': entity fixed effects.
                - 're' or 'random_effects': random effects GLS.
                - 'pooled' or 'pooled_ols': pooled OLS.
        """
        from wraquant.econometrics.panel import fixed_effects, pooled_ols, random_effects

        df = ctx.get_dataset(dataset)
        x_cols = json.loads(x_cols_json)
        if not x_cols:
            return {"tool": "panel_regression", "error": "No x_cols provided."}

        y = df[y_col]

        # Normalise method name
        method_map = {
            "fe": "fixed_effects",
            "fixed_effects": "fixed_effects",
            "re": "random_effects",
            "random_effects": "random_effects",
            "pooled": "pooled_ols",
            "pooled_ols": "pooled_ols",
        }
        method_key = method_map.get(method, method)

        methods = {
            "pooled_ols": lambda: pooled_ols(y, df[x_cols + [entity_col]]),
            "fixed_effects": lambda: fixed_effects(
                y, df[x_cols + [entity_col] + ([time_col] if time_col else [])],
                entity_col=entity_col, time_col=time_col,
            ),
            "random_effects": lambda: random_effects(
                y, df[x_cols + [entity_col]],
                entity_col=entity_col,
            ),
        }

        func = methods.get(method_key)
        if func is None:
            return {"tool": "panel_regression", "error": f"Unknown method '{method}'."}

        result = func()

        model_name = f"panel_{dataset}_{method_key}"
        stored = ctx.store_model(
            model_name, result,
            model_type=f"panel_{method_key}",
            source_dataset=dataset,
        )

        return _sanitize_for_json({
            "tool": "panel_regression",
            "dataset": dataset,
            "method": method_key,
            "y_col": y_col,
            "x_cols": x_cols,
            "coefficients": result.get("coefficients"),
            "std_errors": result.get("std_errors"),
            "r_squared": result.get("r_squared"),
            "nobs": result.get("nobs"),
            **stored,
        })

    @mcp.tool()
    def event_study_econometric(
        dataset: str,
        column: str = "returns",
        event_dates_json: str = "[]",
        market_dataset: str | None = None,
        market_column: str = "returns",
        estimation_window: int = 250,
        event_window: int = 5,
    ) -> dict[str, Any]:
        """Classic event study with market-model abnormal returns.

        Estimates a market model over the estimation window and computes
        abnormal returns (AR) and cumulative abnormal returns (CAR)
        over the event window.

        Parameters:
            dataset: Dataset containing security returns.
            column: Returns column name.
            event_dates_json: JSON array of event date strings
                (e.g. '["2024-01-15", "2024-06-01"]').
            market_dataset: Optional market returns dataset name.
            market_column: Market returns column name.
            estimation_window: Length of the estimation window in
                trading days (mapped to [-estimation_window, -10]).
            event_window: Symmetric event window size in trading days
                (mapped to [-event_window, +event_window]).
        """
        import pandas as pd

        from wraquant.econometrics.event_study import event_study as _es

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)

        event_dates = json.loads(event_dates_json)
        if not event_dates:
            return {"tool": "event_study_econometric", "error": "No event dates provided."}

        dates = pd.to_datetime(event_dates)

        est_win = (-estimation_window, -10)
        evt_win = (-event_window, event_window)

        market = None
        if market_dataset is not None:
            mdf = ctx.get_dataset(market_dataset)
            market = mdf[market_column].dropna()
            if not isinstance(market.index, pd.DatetimeIndex):
                market.index = pd.to_datetime(market.index)

        result = _es(
            returns,
            event_dates=dates,
            estimation_window=est_win,
            event_window=evt_win,
            market_returns=market,
        )

        return _sanitize_for_json({
            "tool": "event_study_econometric",
            "dataset": dataset,
            "n_events": result.get("n_events", len(event_dates)),
            "mean_car": result.get("mean_car"),
            "t_stat": result.get("t_stat"),
            "p_value": result.get("p_value"),
            "event_dates": [str(d) for d in dates],
        })

    @mcp.tool()
    def structural_break(
        dataset: str,
        column: str = "returns",
        method: str = "chow",
        break_point: int | None = None,
    ) -> dict[str, Any]:
        """Test for structural breaks in a time series.

        Uses the Chow test (known break point) or Andrews supremum-F
        test (unknown break point) to detect breaks in the mean or
        regression relationship.

        Parameters:
            dataset: Dataset containing the series.
            column: Column to test.
            method: 'chow' (known break point) or 'sup_f' (unknown).
            break_point: Required for method='chow' -- observation index
                of the hypothesized break.
        """
        from wraquant.econometrics.timeseries import structural_break_test

        df = ctx.get_dataset(dataset)
        data = df[column].dropna()

        result = structural_break_test(
            data.values,
            method=method,
            break_point=break_point,
        )

        return _sanitize_for_json({
            "tool": "structural_break",
            "dataset": dataset,
            "column": column,
            "method": method,
            "f_statistic": result.get("f_statistic"),
            "p_value": result.get("p_value"),
            "break_point": result.get("break_point"),
            "is_break": result.get("is_break"),
        })

    @mcp.tool()
    def cointegration_johansen(
        dataset: str,
        columns_json: str = "[]",
        det_order: int = 0,
    ) -> dict[str, Any]:
        """Johansen cointegration test for multiple time series.

        Tests for cointegrating relationships among a set of
        non-stationary time series using the Johansen trace and
        maximum eigenvalue tests. Essential for pairs/basket trading
        and error-correction models.

        Parameters:
            dataset: Dataset with price series (columns = assets).
            columns_json: JSON array of column names to include.
                If empty, uses all numeric columns.
            det_order: Deterministic term order. -1 for none,
                0 for constant, 1 for linear trend.
        """
        import numpy as np
        import pandas as pd

        from wraquant.stats.cointegration import johansen

        df = ctx.get_dataset(dataset)

        columns = json.loads(columns_json)
        if columns:
            data = df[columns].dropna()
        else:
            data = df.select_dtypes(include=[np.number]).dropna()

        result = johansen(data, det_order=det_order)

        return _sanitize_for_json({
            "tool": "cointegration_johansen",
            "dataset": dataset,
            "columns": list(data.columns),
            "det_order": det_order,
            "coint_rank": result.get("coint_rank"),
            "trace_stats": result.get("trace_stats"),
            "trace_crit": result.get("trace_crit"),
            "max_eig_stats": result.get("max_eig_stats"),
            "max_eig_crit": result.get("max_eig_crit"),
            "eigenvectors": result.get("eigenvectors"),
            "eigenvalues": result.get("eigenvalues"),
        })

    @mcp.tool()
    def impulse_response(
        var_model_id: str,
        periods: int = 20,
    ) -> dict[str, Any]:
        """Compute impulse response functions from a fitted VAR model.

        Applies a one-unit shock to each variable and traces out the
        dynamic response of all variables over the specified number of
        periods.

        Parameters:
            var_model_id: ID of a fitted VAR model stored in the
                context (from a prior var_model call).
            periods: Number of periods for the impulse response.
        """
        import pandas as pd

        from wraquant.econometrics.timeseries import impulse_response as _irf

        model = ctx.get_model(var_model_id)

        # Extract coefficient matrix from the stored VAR result
        coefficients = model.get("coefficients")
        if coefficients is None:
            return {"tool": "impulse_response", "error": "No coefficients in model."}

        k = coefficients.shape[0]

        # Compute IRF for each variable as the shock source
        all_irfs = {}
        for shock_var in range(k):
            irf = _irf(coefficients, n_periods=periods, shock_var=shock_var)
            all_irfs[f"shock_{shock_var}"] = irf.tolist()

        # Store as a dataset for downstream use
        irf_first = _irf(coefficients, n_periods=periods, shock_var=0)
        irf_df = pd.DataFrame(
            irf_first,
            columns=[f"var_{i}" for i in range(k)],
        )
        stored = ctx.store_dataset(
            f"irf_{var_model_id}", irf_df,
            source_op="impulse_response",
        )

        return _sanitize_for_json({
            "tool": "impulse_response",
            "var_model_id": var_model_id,
            "periods": periods,
            "n_variables": k,
            "irfs": all_irfs,
            **stored,
        })
