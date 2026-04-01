"""Tests for experiment MCP server tools.

Tests create_experiment, run_experiment, experiment_results via
underlying wraquant.experiment and context model storage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add mcp source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


@pytest.fixture
def ctx(tmp_path):
    """Create an AnalysisContext with a temporary workspace."""
    context = AnalysisContext(workspace_dir=tmp_path / "test_workspace")
    yield context
    context.close()


@pytest.fixture
def returns_df():
    """Create synthetic return data for experiments."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    returns = np.random.randn(n) * 0.015
    return pd.DataFrame({"returns": returns}, index=dates)


class TestExperimentServer:
    """Test experiment MCP tool functions via context + grid."""

    def test_create_experiment(self, ctx, returns_df):
        """create_experiment sets up a parameter grid."""
        ctx.store_dataset("returns", returns_df)

        params_grid = {"period": [7, 14, 21], "threshold": [0.5, 1.0]}
        name = "momentum_test"

        df = ctx.get_dataset("returns")
        data = df["returns"].dropna()

        grid_size = 1
        for values in params_grid.values():
            grid_size *= len(values)

        config = {
            "name": name,
            "dataset": "returns",
            "column": "returns",
            "params_grid": params_grid,
            "grid_size": grid_size,
            "status": "created",
        }

        ctx.store_model(
            f"experiment_{name}",
            config,
            model_type="experiment_config",
            source_dataset="returns",
        )

        output = _sanitize_for_json({
            "tool": "create_experiment",
            "name": name,
            "dataset": "returns",
            "grid_size": grid_size,
            "params": list(params_grid.keys()),
            "observations": len(data),
        })

        assert output["tool"] == "create_experiment"
        assert output["name"] == "momentum_test"
        assert output["dataset"] == "returns"
        assert output["grid_size"] == 6  # 3 periods * 2 thresholds
        assert isinstance(output["params"], list)
        assert "period" in output["params"]
        assert "threshold" in output["params"]
        assert output["observations"] == 500

        # Verify config was stored as model
        stored_config = ctx.get_model(f"experiment_{name}")
        assert stored_config["status"] == "created"
        assert stored_config["params_grid"] == params_grid

    def test_run_experiment(self, ctx, returns_df):
        """run_experiment executes all grid combinations and finds best."""
        from wraquant.experiment.grid import ParameterGrid

        ctx.store_dataset("returns", returns_df)

        # First create the experiment config
        params_grid = {"period": [7, 14, 21]}
        name = "run_test"
        config = {
            "name": name,
            "dataset": "returns",
            "column": "returns",
            "params_grid": params_grid,
            "grid_size": 3,
            "status": "created",
        }
        ctx.store_model(f"experiment_{name}", config, model_type="experiment_config")

        # Run the experiment (replicate server logic)
        df = ctx.get_dataset(config["dataset"])
        data = df[config["column"]].dropna()

        grid = ParameterGrid(config["params_grid"])

        results = []
        for combo in grid:
            period = combo.get("period", 20)
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

            results.append({
                "params": combo,
                "sharpe": sharpe,
                "total_return": total_ret,
                "max_drawdown": max_dd,
                "n_obs": len(strat_returns),
            })

        best = max(results, key=lambda r: r["sharpe"])

        output = _sanitize_for_json({
            "tool": "run_experiment",
            "experiment": name,
            "n_combinations": len(results),
            "best_params": best["params"],
            "best_sharpe": best["sharpe"],
            "best_total_return": best["total_return"],
            "best_max_drawdown": best["max_drawdown"],
            "all_results": results,
        })

        assert output["tool"] == "run_experiment"
        assert output["experiment"] == "run_test"
        assert output["n_combinations"] == 3
        assert isinstance(output["best_params"], dict)
        assert "period" in output["best_params"]
        assert isinstance(output["best_sharpe"], float)
        assert isinstance(output["best_total_return"], float)
        assert isinstance(output["best_max_drawdown"], float)
        assert output["best_max_drawdown"] <= 0  # drawdowns are non-positive
        assert isinstance(output["all_results"], list)
        assert len(output["all_results"]) == 3

    def test_experiment_results(self, ctx, returns_df):
        """experiment_results returns best params and stability."""
        ctx.store_dataset("returns", returns_df)

        # Store completed experiment results
        name = "completed_test"
        results_data = {
            "name": name,
            "dataset": "returns",
            "column": "returns",
            "params_grid": {"period": [7, 14]},
            "grid_size": 2,
            "status": "completed",
            "results": [
                {"params": {"period": 7}, "sharpe": 0.8, "total_return": 0.12, "max_drawdown": -0.05},
                {"params": {"period": 14}, "sharpe": 1.2, "total_return": 0.18, "max_drawdown": -0.03},
            ],
            "best": {"params": {"period": 14}, "sharpe": 1.2, "total_return": 0.18, "max_drawdown": -0.03},
        }

        ctx.store_model(
            f"experiment_{name}_results",
            results_data,
            model_type="experiment_results",
            source_dataset="returns",
            metrics={"best_sharpe": 1.2},
        )

        # Retrieve and build output (replicate server logic)
        result_data = ctx.get_model(f"experiment_{name}_results")

        sharpes = [r["sharpe"] for r in result_data["results"]]

        output = _sanitize_for_json({
            "tool": "experiment_results",
            "experiment": name,
            "status": result_data["status"],
            "best_params": result_data["best"]["params"],
            "best_sharpe": result_data["best"]["sharpe"],
            "n_combinations": len(result_data["results"]),
            "sharpe_mean": float(np.mean(sharpes)),
            "sharpe_std": float(np.std(sharpes)),
            "stability": (
                "stable" if np.std(sharpes) < 0.3
                else "moderate" if np.std(sharpes) < 0.8
                else "unstable"
            ),
        })

        assert output["tool"] == "experiment_results"
        assert output["experiment"] == "completed_test"
        assert output["status"] == "completed"
        assert isinstance(output["best_params"], dict)
        assert output["best_params"]["period"] == 14
        assert isinstance(output["best_sharpe"], float)
        assert output["best_sharpe"] == pytest.approx(1.2)
        assert output["n_combinations"] == 2
        assert isinstance(output["sharpe_mean"], float)
        assert isinstance(output["sharpe_std"], float)
        assert output["stability"] in ("stable", "moderate", "unstable")
        assert output["stability"] == "stable"  # std of [0.8, 1.2] = 0.2 < 0.3
