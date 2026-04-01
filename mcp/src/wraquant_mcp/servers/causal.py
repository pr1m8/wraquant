"""Causal inference MCP tools.

Tools: granger_causality, event_study, diff_in_diff,
synthetic_control, instrumental_variable, regression_discontinuity,
mediation_analysis.
"""

from __future__ import annotations

import json
from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_causal_tools(mcp, ctx: AnalysisContext) -> None:
    """Register causal-inference tools on the MCP server."""

    @mcp.tool()
    def granger_causality(
        dataset_a: str,
        dataset_b: str,
        col_a: str = "close",
        col_b: str = "close",
        max_lag: int = 5,
    ) -> dict[str, Any]:
        """Test whether series A Granger-causes series B.

        Granger causality tests predictive causality -- whether past
        values of A improve forecasts of B beyond B's own history.

        Parameters:
            dataset_a: Dataset containing the potential causal series.
            dataset_b: Dataset containing the series to predict.
            col_a: Column name in dataset_a.
            col_b: Column name in dataset_b.
            max_lag: Maximum lag order to test.
        """
        from wraquant.causal.treatment import granger_causality as _granger

        df_a = ctx.get_dataset(dataset_a)
        df_b = ctx.get_dataset(dataset_b)
        x = df_a[col_a].dropna().values
        y = df_b[col_b].dropna().values

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
            "direction": result.direction,
            "all_lags": result.all_lags,
        })

    @mcp.tool()
    def event_study(
        dataset: str,
        column: str = "returns",
        event_dates_json: str = "[]",
        estimation_window: int = 60,
        event_window: int = 10,
    ) -> dict[str, Any]:
        """Conduct a market-model event study around specified dates.

        Computes abnormal returns and cumulative abnormal returns (CAR)
        using a market model estimated before each event.

        Parameters:
            dataset: Dataset containing asset returns.
            column: Returns column name.
            event_dates_json: JSON array of event date strings
                (e.g. '["2024-01-15", "2024-06-01"]').
            estimation_window: Length of the estimation window in
                trading days before the event.
            event_window: Number of trading days around the event
                (symmetric pre/post window).
        """
        import numpy as np
        import pandas as pd

        from wraquant.causal.treatment import event_study as _es

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()

        event_dates = json.loads(event_dates_json)
        if not event_dates:
            return {"tool": "event_study", "error": "No event dates provided."}

        # Convert returns to numpy array
        returns_arr = returns.values

        # Build a simple market-returns proxy (mean-return model)
        # using the returns themselves as market if no separate benchmark
        market_arr = returns_arr.copy()

        # Map event date strings to integer indices
        if isinstance(returns.index, pd.DatetimeIndex):
            event_indices = []
            for d in event_dates:
                dt = pd.Timestamp(d)
                idx_positions = returns.index.get_indexer([dt], method="nearest")
                event_indices.append(int(idx_positions[0]))
        else:
            # Assume sequential indices
            event_indices = list(range(0, len(returns), max(1, len(returns) // len(event_dates))))[:len(event_dates)]

        result = _es(
            returns_arr,
            market_arr,
            event_indices=event_indices,
            estimation_window=estimation_window,
            event_window_pre=event_window,
            event_window_post=event_window,
        )

        car_df = pd.DataFrame({
            "abnormal_returns": result.abnormal_returns,
            "cumulative_ar": result.cumulative_ar,
        })
        stored = ctx.store_dataset(
            f"event_study_{dataset}", car_df,
            source_op="event_study", parent=dataset,
        )

        return _sanitize_for_json({
            "tool": "event_study",
            "dataset": dataset,
            "n_events": result.n_events,
            "car": float(result.car),
            "car_t_stat": float(result.car_t_stat),
            "car_p_value": float(result.car_p_value),
            "estimation_alpha": float(result.estimation_alpha),
            "estimation_beta": float(result.estimation_beta),
            "event_dates": event_dates,
            **stored,
        })

    @mcp.tool()
    def diff_in_diff(
        dataset: str,
        treatment_col: str,
        post_col: str,
        outcome_col: str,
    ) -> dict[str, Any]:
        """Estimate treatment effect using difference-in-differences.

        Compares the before-after change in the treatment group to
        the before-after change in the control group.

        Parameters:
            dataset: Dataset with treatment, post, and outcome columns.
            treatment_col: Binary column (1 = treatment group).
            post_col: Binary column (1 = post-treatment period).
            outcome_col: Outcome variable column.
        """
        from wraquant.causal.treatment import diff_in_diff as _did

        df = ctx.get_dataset(dataset)
        outcome = df[outcome_col].values
        treatment = df[treatment_col].values
        post = df[post_col].values

        result = _did(outcome, treatment, post)

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
        donor_datasets_json: str = "[]",
        pre_period: int = 60,
        post_period: int = 30,
    ) -> dict[str, Any]:
        """Estimate treatment effect using synthetic control.

        Constructs a weighted combination of donor units that matches
        the treated unit pre-treatment, then measures the gap
        post-treatment.

        Parameters:
            treated_dataset: Dataset with treated unit outcomes.
            donor_datasets_json: JSON array of donor dataset names.
                Each must be loaded in the context.
            pre_period: Number of pre-treatment time periods.
            post_period: Number of post-treatment time periods
                (used for validation, total series should be at least
                pre_period + post_period).
        """
        import numpy as np

        from wraquant.causal.treatment import synthetic_control as _sc

        df_treated = ctx.get_dataset(treated_dataset)

        # Use the first numeric column as the outcome
        treated = df_treated.select_dtypes(include=[np.number]).iloc[:, 0].dropna().values

        donor_names = json.loads(donor_datasets_json)
        if not donor_names:
            return {"tool": "synthetic_control", "error": "No donor datasets provided."}

        # Stack donor series as columns
        donor_cols = []
        for name in donor_names:
            ddf = ctx.get_dataset(name)
            col = ddf.select_dtypes(include=[np.number]).iloc[:, 0].dropna().values
            donor_cols.append(col)

        # Align lengths
        min_len = min(len(treated), *(len(c) for c in donor_cols))
        treated = treated[:min_len]
        donors = np.column_stack([c[:min_len] for c in donor_cols])

        result = _sc(treated, donors, pre_period=pre_period)

        return _sanitize_for_json({
            "tool": "synthetic_control",
            "treated_dataset": treated_dataset,
            "donor_datasets": donor_names,
            "pre_period": pre_period,
            "post_period": post_period,
            "ate": result.ate,
            "weights": result.weights.tolist() if hasattr(result.weights, "tolist") else result.weights,
            "pre_rmse": result.pre_rmse,
        })

    @mcp.tool()
    def instrumental_variable(
        dataset: str,
        y_col: str,
        x_col: str,
        instrument_col: str,
    ) -> dict[str, Any]:
        """Two-stage least squares instrumental variable estimation.

        Addresses endogeneity by using an instrument that is correlated
        with the endogenous regressor but uncorrelated with the error.

        Parameters:
            dataset: Dataset containing outcome, endogenous, and
                instrument columns.
            y_col: Outcome variable column.
            x_col: Endogenous regressor column.
            instrument_col: Instrument variable column.
        """
        from wraquant.causal.treatment import instrumental_variable as _iv

        df = ctx.get_dataset(dataset)
        y = df[y_col].dropna().values
        x = df[x_col].dropna().values
        z = df[instrument_col].dropna().values

        # Align lengths
        n = min(len(y), len(x), len(z))
        y = y[:n]
        x = x[:n]
        z = z[:n].reshape(-1, 1)

        result = _iv(y, x, z)

        return _sanitize_for_json({
            "tool": "instrumental_variable",
            "dataset": dataset,
            "y_col": y_col,
            "x_col": x_col,
            "instrument_col": instrument_col,
            "coefficient": result.coefficient,
            "se": result.se,
            "ci_lower": result.ci_lower,
            "ci_upper": result.ci_upper,
            "first_stage_f": result.first_stage_f,
            "hausman_stat": result.hausman_stat,
            "hausman_p": result.hausman_p,
            "sargan_stat": result.sargan_stat,
            "sargan_p": result.sargan_p,
            "n_obs": result.n_obs,
        })

    @mcp.tool()
    def regression_discontinuity(
        dataset: str,
        running_col: str,
        outcome_col: str,
        cutoff: float = 0.0,
    ) -> dict[str, Any]:
        """Sharp regression discontinuity design.

        Exploits a cutoff in a running variable to identify a local
        treatment effect. Units just above and below the cutoff are
        essentially identical except for treatment status.

        Parameters:
            dataset: Dataset with running variable and outcome.
            running_col: Running variable column (determines treatment).
            outcome_col: Outcome variable column.
            cutoff: Cutoff value in the running variable.
        """
        from wraquant.causal.treatment import regression_discontinuity as _rd

        df = ctx.get_dataset(dataset)
        running = df[running_col].dropna().values
        outcome = df[outcome_col].dropna().values

        n = min(len(running), len(outcome))
        running = running[:n]
        outcome = outcome[:n]

        result = _rd(outcome, running, cutoff=cutoff)

        return _sanitize_for_json({
            "tool": "regression_discontinuity",
            "dataset": dataset,
            "running_col": running_col,
            "outcome_col": outcome_col,
            "cutoff": cutoff,
            "ate": result.ate,
            "se": result.se,
            "t_stat": result.t_stat,
            "p_value": result.p_value,
            "n_treated": result.n_treated,
            "n_control": result.n_control,
        })

    @mcp.tool()
    def mediation_analysis(
        dataset: str,
        treatment_col: str,
        mediator_col: str,
        outcome_col: str,
    ) -> dict[str, Any]:
        """Baron-Kenny mediation analysis with Sobel test.

        Decomposes the total treatment effect into a direct effect and
        an indirect effect through a mediator variable.

        Parameters:
            dataset: Dataset with treatment, mediator, and outcome.
            treatment_col: Treatment variable column.
            mediator_col: Mediator variable column.
            outcome_col: Outcome variable column.
        """
        from wraquant.causal.treatment import mediation_analysis as _mediation

        df = ctx.get_dataset(dataset)
        treatment = df[treatment_col].dropna().values
        mediator = df[mediator_col].dropna().values
        outcome = df[outcome_col].dropna().values

        n = min(len(treatment), len(mediator), len(outcome))
        treatment = treatment[:n]
        mediator = mediator[:n]
        outcome = outcome[:n]

        result = _mediation(outcome, treatment, mediator)

        return _sanitize_for_json({
            "tool": "mediation_analysis",
            "dataset": dataset,
            "treatment_col": treatment_col,
            "mediator_col": mediator_col,
            "outcome_col": outcome_col,
            "total_effect": result.total_effect,
            "direct_effect": result.direct_effect,
            "indirect_effect": result.indirect_effect,
            "sobel_stat": result.sobel_stat,
            "sobel_p": result.sobel_p,
            "proportion_mediated": result.proportion_mediated,
            "path_a": result.path_a,
            "path_b": result.path_b,
        })
