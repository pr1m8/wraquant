"""Rich results container for experiment analysis.

Provides ``ExperimentResults`` -- a container that wraps the raw run data
from an experiment and provides analysis, comparison, visualization, and
persistence methods.

The results object is the primary interface for post-experiment analysis:
    - ``summary()``: one-row-per-param-combo overview
    - ``best()`` / ``top_n()``: find winning parameter sets
    - ``stability()``: how consistent are results across CV folds
    - ``parameter_sensitivity()``: how sensitive is performance to a parameter
    - ``plot_*()``: visualization methods (require plotly)
    - ``save()`` / ``load()``: persist and restore results
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wraquant.experiment.runner import RunResult

logger = logging.getLogger(__name__)


class ExperimentResults:
    """Results from an experiment run.

    Wraps a list of ``RunResult`` objects and provides analysis,
    comparison, and visualization of experiment outcomes.

    Parameters:
        runs: List of RunResult objects from the experiment.
        experiment_name: Name of the experiment.
        params: Parameter grid specification {name: [values]}.
        benchmark: Optional benchmark return series.
    """

    def __init__(
        self,
        runs: list[RunResult],
        experiment_name: str,
        params: dict[str, list[Any]],
        benchmark: pd.Series | None = None,
    ) -> None:
        self.runs = runs
        self.experiment_name = experiment_name
        self.params = params
        self.benchmark = benchmark
        self._param_names = sorted(params.keys())

    def _param_key(self, params: dict[str, Any]) -> tuple:
        """Create a hashable key from a parameter dict."""
        return tuple(params.get(k) for k in self._param_names)

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """DataFrame with one row per parameter combination.

        Aggregates metrics across CV folds (mean of each metric) and
        includes all parameter values as columns.

        Returns:
            DataFrame with columns for each parameter and each metric
            (prefixed with mean_ for CV-aggregated values).
        """
        grouped: dict[tuple, list[RunResult]] = {}
        for run in self.runs:
            key = self._param_key(run.params)
            grouped.setdefault(key, []).append(run)

        rows: list[dict[str, Any]] = []
        for key, group_runs in grouped.items():
            row: dict[str, Any] = {}
            # Parameters
            for i, name in enumerate(self._param_names):
                row[name] = key[i]

            # Aggregate metrics across folds
            all_metrics: dict[str, list[float]] = {}
            for r in group_runs:
                for metric_name, value in r.metrics.items():
                    all_metrics.setdefault(metric_name, []).append(value)

            for metric_name, values in all_metrics.items():
                row[f"mean_{metric_name}"] = float(np.mean(values))
                if len(values) > 1:
                    row[f"std_{metric_name}"] = float(np.std(values, ddof=1))

            row["n_folds"] = len(group_runs)
            row["total_elapsed"] = sum(r.elapsed_seconds for r in group_runs)
            rows.append(row)

        return pd.DataFrame(rows)

    def best(self, metric: str = "sharpe") -> dict[str, Any]:
        """Best parameter combination by given metric.

        Parameters:
            metric: Metric name to optimize.  Looks for ``mean_{metric}``
                in the summary DataFrame.  Uses "sharpe" by default.

        Returns:
            Dictionary with the best parameters and their metrics.
        """
        df = self.summary()
        col = f"mean_{metric}"
        if col not in df.columns:
            raise ValueError(
                f"Metric '{metric}' not found. Available: "
                f"{[c.replace('mean_', '') for c in df.columns if c.startswith('mean_')]}"
            )
        idx = df[col].idxmax()
        row = df.iloc[idx]
        return {
            "params": {name: row[name] for name in self._param_names},
            "metrics": {
                c.replace("mean_", ""): row[c]
                for c in df.columns
                if c.startswith("mean_")
            },
            "n_folds": int(row["n_folds"]),
        }

    def worst(self, metric: str = "max_drawdown") -> dict[str, Any]:
        """Worst parameter combination by given metric.

        For metrics like max_drawdown where lower (more negative) is
        worse, this returns the minimum.  For return-like metrics,
        returns the minimum as well.

        Parameters:
            metric: Metric name.

        Returns:
            Dictionary with worst parameters and metrics.
        """
        df = self.summary()
        col = f"mean_{metric}"
        if col not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in summary.")
        idx = df[col].idxmin()
        row = df.iloc[idx]
        return {
            "params": {name: row[name] for name in self._param_names},
            "metrics": {
                c.replace("mean_", ""): row[c]
                for c in df.columns
                if c.startswith("mean_")
            },
        }

    def top_n(self, n: int = 5, metric: str = "sharpe") -> pd.DataFrame:
        """Top N parameter combinations by metric.

        Parameters:
            n: Number of top combinations to return.
            metric: Metric to rank by.

        Returns:
            DataFrame with top N rows sorted by the metric descending.
        """
        df = self.summary()
        col = f"mean_{metric}"
        if col not in df.columns:
            raise ValueError(f"Metric '{metric}' not found.")
        return df.nlargest(n, col).reset_index(drop=True)

    def stability(self) -> pd.DataFrame:
        """How stable are results across CV folds.

        For each parameter combination, computes mean, std, min, and max
        of key metrics across folds.  Low std relative to mean indicates
        robust parameters.

        Returns:
            DataFrame with stability statistics per parameter combo.
        """
        grouped: dict[tuple, list[RunResult]] = {}
        for run in self.runs:
            key = self._param_key(run.params)
            grouped.setdefault(key, []).append(run)

        rows: list[dict[str, Any]] = []
        for key, group_runs in grouped.items():
            if len(group_runs) < 2:
                continue

            row: dict[str, Any] = {}
            for i, name in enumerate(self._param_names):
                row[name] = key[i]

            # Compute stability for key metrics
            for metric_name in ["sharpe", "total_return", "max_drawdown", "win_rate"]:
                values = [
                    r.metrics[metric_name]
                    for r in group_runs
                    if metric_name in r.metrics
                ]
                if values:
                    row[f"{metric_name}_mean"] = float(np.mean(values))
                    row[f"{metric_name}_std"] = float(np.std(values, ddof=1))
                    row[f"{metric_name}_min"] = float(np.min(values))
                    row[f"{metric_name}_max"] = float(np.max(values))

            rows.append(row)

        return pd.DataFrame(rows)

    def parameter_sensitivity(self, param_name: str) -> pd.DataFrame:
        """How sensitive is performance to one parameter.

        Groups results by the given parameter's values and computes
        average metrics for each value.  Useful for understanding which
        parameters matter most.

        Parameters:
            param_name: Name of the parameter to analyze.

        Returns:
            DataFrame with one row per unique parameter value.
        """
        if param_name not in self._param_names:
            raise ValueError(
                f"Parameter '{param_name}' not in grid. "
                f"Available: {self._param_names}"
            )

        df = self.summary()
        if param_name not in df.columns:
            raise ValueError(f"Parameter '{param_name}' not in summary.")

        metric_cols = [c for c in df.columns if c.startswith("mean_")]
        group_cols = [param_name] + metric_cols
        available = [c for c in group_cols if c in df.columns]

        return df[available].groupby(param_name).mean().reset_index()

    def regime_breakdown(self) -> pd.DataFrame:
        """Performance broken down by detected market regime.

        Auto-detects regimes on the underlying data and computes
        strategy metrics within each regime.  Requires the data to be
        a pd.Series with a DatetimeIndex.

        Returns:
            DataFrame with one row per regime, showing average metrics.
        """
        try:
            from wraquant.regimes.base import detect_regimes
        except ImportError:
            logger.warning("regimes module not available for regime_breakdown")
            return pd.DataFrame()

        # Get returns from the best-performing runs
        best_info = self.best()
        best_params = best_info["params"]
        best_key = self._param_key(best_params)
        best_runs = [r for r in self.runs if self._param_key(r.params) == best_key]

        if not best_runs:
            return pd.DataFrame()

        # Use the longest run for regime detection
        longest_run = max(best_runs, key=lambda r: len(r.returns))
        returns = longest_run.returns

        if len(returns) < 50:
            logger.warning("Not enough data for regime detection (need >= 50 periods)")
            return pd.DataFrame()

        try:
            regime_result = detect_regimes(returns, method="hmm", n_regimes=2)
            labels = regime_result.labels
        except Exception as exc:
            logger.warning("Regime detection failed: %s", exc)
            return pd.DataFrame()

        # Compute metrics per regime
        from wraquant.experiment.runner import _compute_metrics

        rows: list[dict[str, Any]] = []
        for regime_id in sorted(set(labels)):
            mask = labels == regime_id
            regime_returns = returns[mask]
            if len(regime_returns) < 5:
                continue
            metrics = _compute_metrics(regime_returns)
            metrics["regime"] = int(regime_id)
            metrics["n_periods_in_regime"] = int(mask.sum())
            metrics["pct_time_in_regime"] = float(mask.sum() / len(labels))
            rows.append(metrics)

        return pd.DataFrame(rows)

    def volatility_analysis(self) -> dict[str, Any]:
        """Analyse volatility characteristics of the best strategy.

        Fits an EWMA volatility model to the best strategy's returns
        and reports annualised volatility statistics.  If enough data
        is available, also fits a GARCH(1,1) model and reports
        persistence and half-life.

        Returns:
            Dictionary with:
            - ``annualized_vol``: EWMA annualised volatility at end of series.
            - ``mean_vol``: Mean annualised EWMA volatility over the series.
            - ``max_vol``: Peak annualised volatility observed.
            - ``min_vol``: Minimum annualised volatility observed.
            - ``garch_persistence``: GARCH alpha+beta (if fitted).
            - ``garch_half_life``: GARCH shock half-life (if fitted).
        """
        best_info = self.best()
        best_key = self._param_key(best_info["params"])
        best_runs = [r for r in self.runs if self._param_key(r.params) == best_key]

        if not best_runs:
            return {"error": "No runs found for best params"}

        longest_run = max(best_runs, key=lambda r: len(r.returns))
        returns = longest_run.returns

        if len(returns) < 20:
            return {"error": "Not enough data for volatility analysis"}

        # EWMA volatility analysis
        from wraquant.vol.models import ewma_volatility

        vol_series = ewma_volatility(returns, span=30, annualize=True)
        result: dict[str, Any] = {
            "annualized_vol": float(vol_series.iloc[-1]),
            "mean_vol": float(vol_series.mean()),
            "max_vol": float(vol_series.max()),
            "min_vol": float(vol_series.dropna().min()),
        }

        # Attempt GARCH fit if we have enough data
        if len(returns) >= 100:
            try:
                from wraquant.vol.models import garch_fit

                garch_result = garch_fit(returns)
                result["garch_persistence"] = garch_result["persistence"]
                result["garch_half_life"] = garch_result["half_life"]
            except Exception as exc:
                logger.debug("GARCH fit failed in volatility_analysis: %s", exc)

        return result

    def correlation_with_benchmark(self) -> dict[str, Any]:
        """Correlation of strategy returns with benchmark.

        Returns:
            Dictionary with correlation, beta, and alpha statistics
            for the best parameter combination.
        """
        if self.benchmark is None:
            return {"error": "No benchmark provided"}

        best_info = self.best()
        best_key = self._param_key(best_info["params"])
        best_runs = [r for r in self.runs if self._param_key(r.params) == best_key]

        if not best_runs:
            return {"error": "No runs found for best params"}

        # Concatenate returns from all folds
        all_returns = pd.concat([r.returns for r in best_runs])
        all_returns = all_returns[~all_returns.index.duplicated(keep="first")]

        common = all_returns.index.intersection(self.benchmark.index)
        if len(common) < 10:
            return {"error": "Insufficient overlapping data with benchmark"}

        strat = all_returns.loc[common]
        bench = self.benchmark.loc[common]

        corr = float(strat.corr(bench))
        cov_matrix = np.cov(strat.values, bench.values)
        beta = float(cov_matrix[0, 1] / cov_matrix[1, 1]) if cov_matrix[1, 1] != 0 else 0.0
        alpha = float(strat.mean() - beta * bench.mean()) * 252

        return {
            "correlation": corr,
            "beta": beta,
            "annualized_alpha": alpha,
            "tracking_error": float((strat - bench).std() * np.sqrt(252)),
        }

    def compare_metrics(self) -> pd.DataFrame:
        """Full metrics comparison across all individual runs.

        Unlike ``summary()`` which aggregates across folds, this returns
        one row per (param_combo, fold) with raw metrics.

        Returns:
            DataFrame with one row per run.
        """
        rows: list[dict[str, Any]] = []
        for run in self.runs:
            row = dict(run.params)
            row["fold"] = run.fold
            row.update(run.metrics)
            row["elapsed_seconds"] = run.elapsed_seconds
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_parameter_heatmap(
        self,
        param_x: str,
        param_y: str,
        metric: str = "sharpe",
    ) -> Any:
        """2D heatmap of metric vs two parameters.

        Parameters:
            param_x: Parameter for x-axis.
            param_y: Parameter for y-axis.
            metric: Metric to display.

        Returns:
            Plotly Figure.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("plotly is required for visualization")

        df = self.summary()
        col = f"mean_{metric}"
        pivot = df.pivot_table(values=col, index=param_y, columns=param_x, aggfunc="mean")

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=[str(v) for v in pivot.columns],
                y=[str(v) for v in pivot.index],
                colorscale="RdYlGn",
                text=np.round(pivot.values, 3),
                texttemplate="%{text}",
                colorbar=dict(title=metric),
            )
        )
        fig.update_layout(
            title=f"{self.experiment_name}: {metric} by {param_x} vs {param_y}",
            xaxis_title=param_x,
            yaxis_title=param_y,
        )
        return fig

    def plot_equity_curves(self, top_n: int = 5) -> Any:
        """Overlay equity curves of top N strategies.

        Uses the longest fold for each parameter combination.

        Parameters:
            top_n: Number of top strategies to plot.

        Returns:
            Plotly Figure.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("plotly is required for visualization")

        top_df = self.top_n(n=top_n)
        fig = go.Figure()

        for _, row in top_df.iterrows():
            params = {name: row[name] for name in self._param_names}
            key = self._param_key(params)
            runs = [r for r in self.runs if self._param_key(r.params) == key]
            if not runs:
                continue
            longest = max(runs, key=lambda r: len(r.returns))
            equity = (1 + longest.returns).cumprod()
            label = ", ".join(f"{k}={v}" for k, v in params.items())
            fig.add_trace(go.Scatter(x=list(range(len(equity))), y=equity.values, name=label))

        fig.update_layout(
            title=f"{self.experiment_name}: Top {top_n} Equity Curves",
            xaxis_title="Period",
            yaxis_title="Equity",
        )
        return fig

    def plot_metric_distribution(self, metric: str = "sharpe") -> Any:
        """Distribution of a metric across all runs.

        Parameters:
            metric: Metric to plot.

        Returns:
            Plotly Figure.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("plotly is required for visualization")

        values = [r.metrics.get(metric, np.nan) for r in self.runs]
        values = [v for v in values if not np.isnan(v)]

        fig = go.Figure(data=[go.Histogram(x=values, nbinsx=30)])
        fig.update_layout(
            title=f"{self.experiment_name}: {metric} Distribution",
            xaxis_title=metric,
            yaxis_title="Count",
        )
        return fig

    def plot_stability(self) -> Any:
        """Box plot of metrics across CV folds for each param combo.

        Returns:
            Plotly Figure.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("plotly is required for visualization")

        grouped: dict[str, list[float]] = {}
        for run in self.runs:
            label = ", ".join(f"{k}={v}" for k, v in sorted(run.params.items()))
            grouped.setdefault(label, []).append(run.metrics.get("sharpe", 0.0))

        fig = go.Figure()
        for label, values in sorted(grouped.items()):
            if len(values) > 1:
                fig.add_trace(go.Box(y=values, name=label))

        fig.update_layout(
            title=f"{self.experiment_name}: Sharpe Stability Across Folds",
            yaxis_title="Sharpe Ratio",
        )
        return fig

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def report(self, format: str = "dict") -> Any:
        """Generate comprehensive experiment report.

        Parameters:
            format: Output format -- "dict", "dataframe", or "html".

        Returns:
            Report in the requested format.
        """
        best_info = self.best()
        worst_info = self.worst()
        summary_df = self.summary()

        report_data = {
            "experiment_name": self.experiment_name,
            "n_param_combos": len(summary_df),
            "n_total_runs": len(self.runs),
            "n_folds": int(summary_df["n_folds"].iloc[0]) if len(summary_df) > 0 else 0,
            "best": best_info,
            "worst": worst_info,
            "summary_statistics": {
                "sharpe_mean": float(summary_df["mean_sharpe"].mean())
                if "mean_sharpe" in summary_df.columns
                else None,
                "sharpe_std": float(summary_df["mean_sharpe"].std())
                if "mean_sharpe" in summary_df.columns
                else None,
            },
        }

        if format == "dict":
            return report_data
        elif format == "dataframe":
            return summary_df
        elif format == "html":
            html = f"<h1>Experiment: {self.experiment_name}</h1>\n"
            html += f"<p>Param combos: {len(summary_df)} | Total runs: {len(self.runs)}</p>\n"
            html += "<h2>Best Parameters</h2>\n"
            html += f"<pre>{json.dumps(best_info, indent=2, default=str)}</pre>\n"
            html += "<h2>Summary</h2>\n"
            html += summary_df.to_html()
            return html
        else:
            raise ValueError(f"Unknown format: {format!r}. Use 'dict', 'dataframe', or 'html'.")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist results to disk.

        Saves metadata as JSON and run data as a parquet file.

        Parameters:
            path: Directory to save results to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        meta = {
            "experiment_name": self.experiment_name,
            "params": self.params,
            "n_runs": len(self.runs),
            "run_summaries": [r.to_dict() for r in self.runs],
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        # Save summary as parquet
        summary = self.summary()
        summary.to_parquet(path / "summary.parquet", index=False)

        # Save full comparison as parquet
        compare = self.compare_metrics()
        compare.to_parquet(path / "all_runs.parquet", index=False)

    @classmethod
    def load(cls, path: str | Path) -> ExperimentResults:
        """Load persisted results.

        Parameters:
            path: Directory containing saved results.

        Returns:
            ExperimentResults reconstituted from disk.
        """
        path = Path(path)

        with open(path / "metadata.json") as f:
            meta = json.load(f)

        # Reconstruct RunResult objects (without full return series)
        runs: list[RunResult] = []
        for run_data in meta.get("run_summaries", []):
            runs.append(
                RunResult(
                    params=run_data["params"],
                    fold=run_data["fold"],
                    metrics=run_data["metrics"],
                    returns=pd.Series(dtype=float),
                    train_indices=np.array([]),
                    test_indices=np.array([]),
                    elapsed_seconds=run_data.get("elapsed_seconds", 0.0),
                )
            )

        return cls(
            runs=runs,
            experiment_name=meta["experiment_name"],
            params=meta["params"],
        )


__all__ = [
    "ExperimentResults",
]
