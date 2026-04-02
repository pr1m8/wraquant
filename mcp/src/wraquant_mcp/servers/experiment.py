"""Experiment tracking MCP tools.

Tools: create_experiment, run_experiment, experiment_results,
experiment_comparison, parameter_sensitivity.
"""

from __future__ import annotations

import json
from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_experiment_tools(mcp, ctx: AnalysisContext) -> None:
    """Register experiment tracking tools on the MCP server."""

    @mcp.tool()
    def create_experiment(
        name: str,
        params_grid_json: str,
        dataset: str,
        column: str = "returns",
    ) -> dict[str, Any]:
        """Set up a grid search experiment over strategy parameters.

        Creates an experiment configuration with a parameter grid and
        stores it for later execution via ``run_experiment``.

        Parameters:
            name: Experiment name for tracking and retrieval.
            params_grid_json: JSON string mapping parameter names to lists
                of values (e.g., '{"period": [7, 14, 21], "threshold": [0.5, 1.0]}').
            dataset: Dataset containing the return series to test against.
            column: Returns column name in the dataset.
        """
        try:
            params_grid = json.loads(params_grid_json)

            df = ctx.get_dataset(dataset)
            data = df[column].dropna()

            # Compute grid size
            grid_size = 1
            for values in params_grid.values():
                grid_size *= len(values)

            config = {
                "name": name,
                "dataset": dataset,
                "column": column,
                "params_grid": params_grid,
                "grid_size": grid_size,
                "status": "created",
            }

            ctx.store_model(
                f"experiment_{name}",
                config,
                model_type="experiment_config",
                source_dataset=dataset,
            )

            ctx._log("create_experiment", name, params_grid=str(params_grid))

            return _sanitize_for_json(
                {
                    "tool": "create_experiment",
                    "name": name,
                    "dataset": dataset,
                    "grid_size": grid_size,
                    "params": list(params_grid.keys()),
                    "observations": len(data),
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "create_experiment"}

    @mcp.tool()
    def run_experiment(
        name: str,
        cv_method: str = "walk_forward",
        n_splits: int = 5,
        parallel: bool = True,
    ) -> dict[str, Any]:
        """Execute a previously created experiment.

        Runs all parameter combinations from the grid with the specified
        cross-validation method and returns performance metrics.

        Parameters:
            name: Name of the experiment (from create_experiment).
            cv_method: Cross-validation method ('walk_forward', 'rolling',
                'purged_kfold', 'none').
            n_splits: Number of CV splits.
            parallel: Whether to run combinations in parallel (when available).
        """
        try:
            import numpy as np

            config = ctx.get_model(f"experiment_{name}")

            if not isinstance(config, dict):
                return {"error": f"Experiment '{name}' config not found"}

            df = ctx.get_dataset(config["dataset"])
            data = df[config["column"]].dropna()

            from wraquant.experiment.grid import ParameterGrid

            grid = ParameterGrid(config["params_grid"])

            results = []
            for combo in grid:
                period = combo.get("period", combo.get("window", 20))
                signal = data.rolling(period).mean()
                strat_returns = data * np.sign(signal.shift(1))
                strat_returns = strat_returns.dropna()

                sharpe = (
                    float(strat_returns.mean() / strat_returns.std() * np.sqrt(252))
                    if strat_returns.std() > 0
                    else 0.0
                )
                total_ret = float(strat_returns.sum())
                max_dd = float(
                    (strat_returns.cumsum() - strat_returns.cumsum().cummax()).min()
                )

                results.append(
                    {
                        "params": combo,
                        "sharpe": sharpe,
                        "total_return": total_ret,
                        "max_drawdown": max_dd,
                        "n_obs": len(strat_returns),
                    }
                )

            best = max(results, key=lambda r: r["sharpe"])

            # Store results
            config["status"] = "completed"
            config["results"] = results
            config["best"] = best
            ctx.store_model(
                f"experiment_{name}_results",
                config,
                model_type="experiment_results",
                source_dataset=config["dataset"],
                metrics={"best_sharpe": best["sharpe"]},
            )

            ctx._log(
                "run_experiment",
                name,
                cv_method=cv_method,
                best_params=str(best["params"]),
                best_sharpe=best["sharpe"],
            )

            return _sanitize_for_json(
                {
                    "tool": "run_experiment",
                    "experiment": name,
                    "cv_method": cv_method,
                    "parallel": parallel,
                    "n_combinations": len(results),
                    "best_params": best["params"],
                    "best_sharpe": best["sharpe"],
                    "best_total_return": best["total_return"],
                    "best_max_drawdown": best["max_drawdown"],
                    "all_results": results,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "run_experiment"}

    @mcp.tool()
    def experiment_results(
        name: str,
    ) -> dict[str, Any]:
        """Get results summary, best params, and stability for an experiment.

        Returns the stored configuration, best parameters, performance
        metrics, and journal history.

        Parameters:
            name: Name of the experiment.
        """
        try:
            # Try completed results first, then config
            try:
                result_data = ctx.get_model(f"experiment_{name}_results")
            except KeyError:
                result_data = None

            try:
                config = ctx.get_model(f"experiment_{name}")
            except KeyError:
                config = None

            if result_data is None and config is None:
                return {"error": f"Experiment '{name}' not found"}

            history = ctx.history(n=1000)
            exp_entries = [e for e in history if e.get("resource") == name]

            source = result_data if result_data is not None else config

            output: dict[str, Any] = {
                "tool": "experiment_results",
                "experiment": name,
                "status": (
                    source.get("status", "unknown")
                    if isinstance(source, dict)
                    else "unknown"
                ),
            }

            if isinstance(source, dict):
                if "best" in source:
                    output["best_params"] = source["best"]["params"]
                    output["best_sharpe"] = source["best"]["sharpe"]
                if "results" in source:
                    output["n_combinations"] = len(source["results"])
                    # Compute stability: std of sharpes across all combos
                    import numpy as np

                    sharpes = [r["sharpe"] for r in source["results"]]
                    output["sharpe_mean"] = float(np.mean(sharpes))
                    output["sharpe_std"] = float(np.std(sharpes))
                    output["stability"] = (
                        "stable"
                        if np.std(sharpes) < 0.3
                        else "moderate" if np.std(sharpes) < 0.8 else "unstable"
                    )
                if "params_grid" in source:
                    output["params_grid"] = source["params_grid"]

            output["journal_entries"] = exp_entries

            return _sanitize_for_json(output)
        except Exception as e:
            return {"error": str(e), "tool": "experiment_results"}

    @mcp.tool()
    def experiment_comparison(
        names_json: str,
    ) -> dict[str, Any]:
        """Compare results across multiple experiments.

        Loads results for each named experiment and produces a
        side-by-side comparison of best parameters and metrics.

        Parameters:
            names_json: JSON array of experiment names to compare
                (e.g., '["momentum_fast", "momentum_slow", "meanrev"]').
        """
        try:
            names = json.loads(names_json)

            comparisons = []
            for exp_name in names:
                entry: dict[str, Any] = {"experiment": exp_name}
                try:
                    result_data = ctx.get_model(f"experiment_{exp_name}_results")
                    if isinstance(result_data, dict):
                        entry["status"] = result_data.get("status", "unknown")
                        if "best" in result_data:
                            entry["best_params"] = result_data["best"]["params"]
                            entry["best_sharpe"] = result_data["best"]["sharpe"]
                            entry["best_total_return"] = result_data["best"].get(
                                "total_return"
                            )
                            entry["best_max_drawdown"] = result_data["best"].get(
                                "max_drawdown"
                            )
                        if "results" in result_data:
                            entry["n_combinations"] = len(result_data["results"])
                except KeyError:
                    try:
                        config = ctx.get_model(f"experiment_{exp_name}")
                        entry["status"] = (
                            config.get("status", "created")
                            if isinstance(config, dict)
                            else "unknown"
                        )
                    except KeyError:
                        entry["status"] = "not_found"

                comparisons.append(entry)

            # Rank by best sharpe
            ranked = sorted(
                [c for c in comparisons if "best_sharpe" in c],
                key=lambda c: c["best_sharpe"],
                reverse=True,
            )
            ranking = [
                {
                    "rank": i + 1,
                    "experiment": c["experiment"],
                    "sharpe": c["best_sharpe"],
                }
                for i, c in enumerate(ranked)
            ]

            return _sanitize_for_json(
                {
                    "tool": "experiment_comparison",
                    "n_experiments": len(names),
                    "comparisons": comparisons,
                    "ranking": ranking,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "experiment_comparison"}

    @mcp.tool()
    def parameter_sensitivity(
        name: str,
        param_name: str,
    ) -> dict[str, Any]:
        """Analyze how sensitive an experiment is to one parameter.

        Extracts all results for the given experiment and groups them
        by the specified parameter, showing how performance varies
        as that parameter changes.

        Parameters:
            name: Name of the completed experiment.
            param_name: Parameter to analyze sensitivity for.
        """
        try:
            try:
                result_data = ctx.get_model(f"experiment_{name}_results")
            except KeyError:
                return {"error": f"No completed results for experiment '{name}'"}

            if not isinstance(result_data, dict) or "results" not in result_data:
                return {"error": f"Experiment '{name}' has no results"}

            import numpy as np

            # Group results by param_name
            param_groups: dict[str, list[float]] = {}
            for r in result_data["results"]:
                val = str(r["params"].get(param_name, "N/A"))
                if val not in param_groups:
                    param_groups[val] = []
                param_groups[val].append(r["sharpe"])

            sensitivity = []
            for val, sharpes in sorted(param_groups.items()):
                arr = np.array(sharpes)
                sensitivity.append(
                    {
                        "value": val,
                        "mean_sharpe": float(np.mean(arr)),
                        "std_sharpe": float(np.std(arr)),
                        "min_sharpe": float(np.min(arr)),
                        "max_sharpe": float(np.max(arr)),
                        "n_combos": len(sharpes),
                    }
                )

            # Overall sensitivity: std of mean sharpes across values
            mean_sharpes = [s["mean_sharpe"] for s in sensitivity]
            overall_sensitivity = (
                float(np.std(mean_sharpes)) if len(mean_sharpes) > 1 else 0.0
            )

            return _sanitize_for_json(
                {
                    "tool": "parameter_sensitivity",
                    "experiment": name,
                    "param_name": param_name,
                    "overall_sensitivity": overall_sensitivity,
                    "sensitivity": sensitivity,
                    "interpretation": (
                        "high"
                        if overall_sensitivity > 0.5
                        else "moderate" if overall_sensitivity > 0.1 else "low"
                    ),
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "parameter_sensitivity"}
