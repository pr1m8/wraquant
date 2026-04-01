"""Econometric analysis MCP tools.

Tools: var_model, panel_regression, event_study_econometric,
structural_break.
"""

from __future__ import annotations

from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_econometrics_tools(mcp, ctx: AnalysisContext) -> None:
    """Register econometrics tools on the MCP server."""

    @mcp.tool()
    def var_model(
        dataset: str,
        columns: list[str] | None = None,
        lags: int | None = 2,
        ic: str = "aic",
    ) -> dict[str, Any]:
        """Fit a Vector Autoregression (VAR) model.

        Selects lag order by information criterion and estimates a
        reduced-form VAR. Returns coefficients, residual covariance,
        selected lag order, and model fit statistics.

        Parameters:
            dataset: Dataset with multivariate time series.
            columns: Columns to include. If None, uses all numeric columns.
            lags: Maximum lag order (None = auto-select).
            ic: Information criterion ('aic', 'bic', 'hqic', 'fpe').
        """
        import numpy as np

        from wraquant.econometrics.timeseries import var_model as _var

        df = ctx.get_dataset(dataset)
        if columns is None:
            data = df.select_dtypes(include=[np.number]).dropna()
        else:
            data = df[columns].dropna()

        result = _var(data, max_lags=lags, ic=ic)

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
            **stored,
        })

    @mcp.tool()
    def panel_regression(
        dataset: str,
        y_col: str,
        x_cols: list[str],
        entity_col: str,
        time_col: str | None = None,
        method: str = "fixed_effects",
    ) -> dict[str, Any]:
        """Run panel data regression.

        Supports pooled OLS, fixed effects (within estimator), and
        random effects.

        Parameters:
            dataset: Panel dataset with entity and time identifiers.
            y_col: Dependent variable column.
            x_cols: Independent variable columns.
            entity_col: Column identifying the cross-sectional unit.
            time_col: Optional column identifying the time period.
            method: 'pooled_ols', 'fixed_effects', or 'random_effects'.
        """
        from wraquant.econometrics.panel import fixed_effects, pooled_ols, random_effects

        df = ctx.get_dataset(dataset)
        y = df[y_col]

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

        func = methods.get(method)
        if func is None:
            return {"error": f"Unknown method '{method}'. Options: {list(methods)}"}

        result = func()

        model_name = f"panel_{dataset}_{method}"
        stored = ctx.store_model(
            model_name, result,
            model_type=f"panel_{method}",
            source_dataset=dataset,
        )

        return _sanitize_for_json({
            "tool": "panel_regression",
            "dataset": dataset,
            "method": method,
            "coefficients": result.get("coefficients"),
            "std_errors": result.get("std_errors"),
            "r_squared": result.get("r_squared"),
            "nobs": result.get("nobs"),
            **stored,
        })

    @mcp.tool()
    def event_study_econometric(
        dataset: str,
        event_dates: list[str],
        column: str = "returns",
        market_dataset: str | None = None,
        market_column: str = "returns",
        estimation_window: list[int] | None = None,
        event_window: list[int] | None = None,
    ) -> dict[str, Any]:
        """Classic event study with market-model abnormal returns.

        Estimates a market model over the estimation window and computes
        abnormal returns (AR) and cumulative abnormal returns (CAR).

        Parameters:
            dataset: Dataset containing security returns.
            event_dates: List of event dates ('YYYY-MM-DD').
            column: Returns column name.
            market_dataset: Optional market returns dataset.
            market_column: Market returns column name.
            estimation_window: [start, end] offsets for estimation (default [-250, -10]).
            event_window: [start, end] offsets for event (default [-5, 5]).
        """
        import pandas as pd

        from wraquant.econometrics.event_study import event_study as _es

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)

        dates = pd.to_datetime(event_dates)

        est_win = tuple(estimation_window) if estimation_window else (-250, -10)
        evt_win = tuple(event_window) if event_window else (-5, 5)

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
        })

    @mcp.tool()
    def structural_break(
        dataset: str,
        column: str = "returns",
        method: str = "sup_f",
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
            break_point: Required for method='chow' — observation index
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
