"""Bayesian inference MCP tools.

Tools: bayesian_sharpe, bayesian_regression, bayesian_changepoint,
model_comparison.
"""

from __future__ import annotations

from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_bayes_tools(mcp, ctx: AnalysisContext) -> None:
    """Register Bayesian inference tools on the MCP server."""

    @mcp.tool()
    def bayesian_sharpe(
        dataset: str,
        column: str = "returns",
        n_samples: int = 10_000,
    ) -> dict[str, Any]:
        """Estimate the Sharpe ratio with full Bayesian uncertainty.

        Returns the posterior distribution of the Sharpe ratio including
        credible intervals, probability of positive Sharpe, and
        posterior mean/median.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
            n_samples: Number of posterior samples.
        """
        from wraquant.bayes.models import bayesian_sharpe as _bsharpe

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna().values

        result = _bsharpe(returns, n_samples=n_samples)

        return _sanitize_for_json({
            "tool": "bayesian_sharpe",
            "dataset": dataset,
            "posterior_mean": result.posterior_mean,
            "posterior_median": result.posterior_median,
            "credible_interval_95": result.credible_interval,
            "prob_positive": result.prob_positive,
            "n_samples": n_samples,
        })

    @mcp.tool()
    def bayesian_regression(
        dataset: str,
        y_column: str,
        x_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Conjugate Bayesian linear regression.

        Returns the full posterior distribution of coefficients,
        log marginal likelihood for model comparison, and credible
        intervals.

        Parameters:
            dataset: Dataset with dependent and independent variables.
            y_column: Dependent variable column.
            x_columns: Independent variable columns.
                If None, uses all numeric columns except y.
        """
        import numpy as np

        from wraquant.bayes.models import bayesian_regression as _breg

        df = ctx.get_dataset(dataset)

        if x_columns is None:
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            x_columns = [c for c in numeric if c != y_column]

        y = df[y_column].dropna().values
        X = df[x_columns].dropna().values

        n = min(len(y), len(X))
        y = y[:n]
        X = X[:n]

        result = _breg(y, X)

        model_name = f"bayes_reg_{dataset}"
        stored = ctx.store_model(
            model_name, result,
            model_type="bayesian_regression",
            source_dataset=dataset,
        )

        return _sanitize_for_json({
            "tool": "bayesian_regression",
            "dataset": dataset,
            "y_column": y_column,
            "x_columns": x_columns,
            "posterior_mean": result.posterior_mean,
            "posterior_std": np.sqrt(np.diag(result.posterior_cov)).tolist(),
            "log_marginal_likelihood": result.log_marginal_likelihood,
            **stored,
        })

    @mcp.tool()
    def bayesian_changepoint(
        dataset: str,
        column: str = "returns",
        hazard: float = 0.01,
        threshold: float = 0.3,
    ) -> dict[str, Any]:
        """Bayesian online changepoint detection (Adams & MacKay, 2007).

        Processes observations sequentially and maintains a posterior
        over the run length (time since last changepoint), providing
        probability estimates at each time step.

        Parameters:
            dataset: Dataset containing the time series.
            column: Column to analyze.
            hazard: Prior probability of a changepoint at each step.
            threshold: Probability threshold for declaring a changepoint.
        """
        import pandas as pd

        from wraquant.bayes.models import bayesian_changepoint as _bcp

        df = ctx.get_dataset(dataset)
        data = df[column].dropna().values

        result = _bcp(data, hazard=hazard, threshold=threshold)

        cp_df = pd.DataFrame({
            "changepoint_prob": result.changepoint_probs,
        })
        stored = ctx.store_dataset(
            f"changepoints_{dataset}", cp_df,
            source_op="bayesian_changepoint", parent=dataset,
        )

        return _sanitize_for_json({
            "tool": "bayesian_changepoint",
            "dataset": dataset,
            "n_changepoints": result.n_changepoints,
            "changepoint_indices": result.changepoint_indices,
            "hazard": hazard,
            "threshold": threshold,
            **stored,
        })

    @mcp.tool()
    def model_comparison(
        dataset: str,
        y_column: str,
        models_config: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compare Bayesian regression models via marginal likelihood.

        Fits each candidate model specification and ranks them by
        log marginal likelihood, WAIC, and Bayes factors.

        Parameters:
            dataset: Dataset with all variables.
            y_column: Dependent variable column.
            models_config: List of dicts, each with:
                - 'name': model name (str)
                - 'x_columns': list of regressor column names
        """
        import numpy as np

        from wraquant.bayes.models import model_comparison as _compare

        df = ctx.get_dataset(dataset)
        y = df[y_column].dropna().values

        X_list = []
        model_names = []
        for spec in models_config:
            cols = spec["x_columns"]
            X_list.append(df[cols].dropna().values[:len(y)])
            model_names.append(spec.get("name", "+".join(cols)))

        result = _compare(y, X_list, model_names=model_names)

        return _sanitize_for_json({
            "tool": "model_comparison",
            "dataset": dataset,
            "y_column": y_column,
            "ranking": result.to_dict(orient="records")
            if hasattr(result, "to_dict") else result,
        })
