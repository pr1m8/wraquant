"""Causal inference MCP tools.

Tools: granger_causality, event_study, diff_in_diff,
synthetic_control, cointegration_test.
"""

from __future__ import annotations

from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_causal_tools(mcp, ctx: AnalysisContext) -> None:
    """Register causal-inference tools on the MCP server."""

    @mcp.tool()
    def granger_causality(
        dataset_a: str,
        dataset_b: str,
        column_a: str = "returns",
        column_b: str = "returns",
        max_lag: int = 5,
    ) -> dict[str, Any]:
        """Test whether series A Granger-causes series B.

        Granger causality tests predictive causality — whether past
        values of A improve forecasts of B beyond B's own history.

        Parameters:
            dataset_a: Dataset containing the potential causal series.
            dataset_b: Dataset containing the series to predict.
            column_a: Column name in dataset_a.
            column_b: Column name in dataset_b.
            max_lag: Maximum lag order to test.
        """
        import numpy as np

        from wraquant.causal.treatment import granger_causality as _granger

        df_a = ctx.get_dataset(dataset_a)
        df_b = ctx.get_dataset(dataset_b)
        x = df_a[column_a].dropna().values
        y = df_b[column_b].dropna().values

        n = min(len(x), len(y))
        x = x[-n:]
        y = y[-n:]

        result = _granger(x, y, max_lag=max_lag)

        return _sanitize_for_json({
            "tool": "granger_causality",
            "dataset_a": dataset_a,
            "dataset_b": dataset_b,
            "max_lag": max_lag,
            "optimal_lag": result.optimal_lag,
            "f_statistic": result.f_statistic,
            "p_value": result.p_value,
            "reject": result.reject,
            "all_lags": result.all_lags if hasattr(result, "all_lags") else None,
        })

    @mcp.tool()
    def event_study(
        dataset: str,
        event_dates: list[str],
        column: str = "returns",
        window_pre: int = 5,
        window_post: int = 5,
        market_dataset: str | None = None,
        market_column: str = "returns",
    ) -> dict[str, Any]:
        """Conduct an event study around specified dates.

        Computes abnormal returns and cumulative abnormal returns (CAR)
        using a market model estimated before each event.

        Parameters:
            dataset: Dataset containing asset returns.
            event_dates: List of event date strings ('YYYY-MM-DD').
            column: Returns column name.
            window_pre: Pre-event window in trading days.
            window_post: Post-event window in trading days.
            market_dataset: Optional dataset for market returns.
            market_column: Market returns column name.
        """
        import pandas as pd

        from wraquant.econometrics.event_study import event_study as _es

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)

        dates = pd.to_datetime(event_dates)

        market = None
        if market_dataset is not None:
            mdf = ctx.get_dataset(market_dataset)
            market = mdf[market_column].dropna()
            if not isinstance(market.index, pd.DatetimeIndex):
                market.index = pd.to_datetime(market.index)

        result = _es(
            returns,
            event_dates=dates,
            event_window=(-window_pre, window_post),
            market_returns=market,
        )

        return _sanitize_for_json({
            "tool": "event_study",
            "dataset": dataset,
            "n_events": result.get("n_events", len(event_dates)),
            "mean_car": result.get("mean_car"),
            "t_stat": result.get("t_stat"),
            "p_value": result.get("p_value"),
            "event_dates": [str(d) for d in dates],
        })

    @mcp.tool()
    def diff_in_diff(
        dataset: str,
        treatment_col: str,
        post_col: str,
        outcome_col: str,
        entity_col: str | None = None,
    ) -> dict[str, Any]:
        """Estimate treatment effect using difference-in-differences.

        Compares the before-after change in the treatment group to
        the before-after change in the control group.

        Parameters:
            dataset: Dataset with treatment, post, and outcome columns.
            treatment_col: Binary column (1 = treatment group).
            post_col: Binary column (1 = post-treatment period).
            outcome_col: Outcome variable column.
            entity_col: Optional entity identifier for panel DID.
        """
        import numpy as np

        from wraquant.causal.treatment import diff_in_diff as _did

        df = ctx.get_dataset(dataset)
        outcome = df[outcome_col].values
        treatment = df[treatment_col].values
        post = df[post_col].values
        entity = df[entity_col].values if entity_col else None

        result = _did(outcome, treatment, post, entity=entity)

        return _sanitize_for_json({
            "tool": "diff_in_diff",
            "dataset": dataset,
            "ate": result.ate,
            "se": result.se,
            "t_stat": result.t_stat,
            "p_value": result.p_value,
            "pre_treatment_mean": result.pre_treatment_mean,
            "post_treatment_mean": result.post_treatment_mean,
            "pre_control_mean": result.pre_control_mean,
            "post_control_mean": result.post_control_mean,
        })

    @mcp.tool()
    def synthetic_control(
        treated_dataset: str,
        donor_dataset: str,
        outcome_col: str,
        pre_period: int,
    ) -> dict[str, Any]:
        """Estimate treatment effect using synthetic control.

        Constructs a weighted combination of donor units that matches
        the treated unit pre-treatment, then measures the gap post-treatment.

        Parameters:
            treated_dataset: Dataset with treated unit outcomes (single column).
            donor_dataset: Dataset with donor unit outcomes (multi-column).
            outcome_col: Outcome column name in treated dataset.
            pre_period: Number of pre-treatment time periods.
        """
        import numpy as np

        from wraquant.causal.treatment import synthetic_control as _sc

        df_treated = ctx.get_dataset(treated_dataset)
        df_donors = ctx.get_dataset(donor_dataset)

        treated = df_treated[outcome_col].dropna().values
        donors = df_donors.select_dtypes(include=[np.number]).dropna().values

        result = _sc(treated, donors, pre_period=pre_period)

        return _sanitize_for_json({
            "tool": "synthetic_control",
            "treated_dataset": treated_dataset,
            "donor_dataset": donor_dataset,
            "pre_period": pre_period,
            "ate": result.ate,
            "weights": result.weights,
            "pre_rmse": result.pre_rmse,
        })

    @mcp.tool()
    def cointegration_test(
        dataset_a: str,
        dataset_b: str,
        column_a: str = "close",
        column_b: str = "close",
        method: str = "engle_granger",
    ) -> dict[str, Any]:
        """Test for cointegration between two price series.

        Determines whether two non-stationary series share a long-run
        equilibrium (useful for pairs trading).

        Parameters:
            dataset_a: First dataset.
            dataset_b: Second dataset.
            column_a: Column name in first dataset.
            column_b: Column name in second dataset.
            method: 'engle_granger' or 'johansen'.
        """
        from wraquant.stats.cointegration import engle_granger, half_life, hedge_ratio

        df_a = ctx.get_dataset(dataset_a)
        df_b = ctx.get_dataset(dataset_b)
        a = df_a[column_a].dropna()
        b = df_b[column_b].dropna()

        n = min(len(a), len(b))
        a = a.iloc[-n:]
        b = b.iloc[-n:]

        if method == "johansen":
            from wraquant.stats.cointegration import johansen

            result = johansen(a, b)
        else:
            result = engle_granger(a, b)

        hr = hedge_ratio(a, b)
        hl = half_life(a - hr * b)

        return _sanitize_for_json({
            "tool": "cointegration_test",
            "dataset_a": dataset_a,
            "dataset_b": dataset_b,
            "method": method,
            "result": result,
            "hedge_ratio": float(hr),
            "half_life": float(hl),
        })
