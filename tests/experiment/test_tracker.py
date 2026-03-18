"""Tests for wraquant.experiment.tracker."""

from __future__ import annotations

from pathlib import Path

import pytest

from wraquant.experiment.tracker import Experiment


class TestExperiment:
    def test_creates_directory(self, tmp_path: Path) -> None:
        exp = Experiment("my_exp", base_dir=str(tmp_path / "experiments"))
        assert exp.experiment_dir.exists()
        assert exp.experiment_dir.is_dir()

    def test_run_logs_params_and_metrics(self, tmp_path: Path) -> None:
        exp = Experiment("test_run", base_dir=str(tmp_path))
        with exp.start_run(params={"lr": 0.01}) as run:
            run.log_param("epochs", 100)
            run.log_metric("sharpe_ratio", 1.5)
            run.log_metric("max_drawdown", -0.12)

        run_dict = run.to_dict()
        assert run_dict["params"]["lr"] == 0.01
        assert run_dict["params"]["epochs"] == 100
        assert run_dict["metrics"]["sharpe_ratio"] == 1.5
        assert run_dict["metrics"]["max_drawdown"] == -0.12

    def test_list_runs_returns_logged_runs(self, tmp_path: Path) -> None:
        exp = Experiment("list_test", base_dir=str(tmp_path))

        with exp.start_run(params={"a": 1}) as run:
            run.log_metric("score", 0.9)

        with exp.start_run(params={"a": 2}) as run:
            run.log_metric("score", 0.8)

        runs = exp.list_runs()
        assert len(runs) == 2
        scores = {r["metrics"]["score"] for r in runs}
        assert scores == {0.9, 0.8}

    def test_best_run_finds_correct_best(self, tmp_path: Path) -> None:
        exp = Experiment("best_test", base_dir=str(tmp_path))

        with exp.start_run() as run:
            run.log_metric("sharpe_ratio", 1.0)

        with exp.start_run() as run:
            run.log_metric("sharpe_ratio", 2.5)

        with exp.start_run() as run:
            run.log_metric("sharpe_ratio", 1.8)

        best = exp.best_run(metric="sharpe_ratio", maximize=True)
        assert best["metrics"]["sharpe_ratio"] == 2.5

    def test_best_run_minimize(self, tmp_path: Path) -> None:
        exp = Experiment("min_test", base_dir=str(tmp_path))

        with exp.start_run() as run:
            run.log_metric("loss", 0.5)

        with exp.start_run() as run:
            run.log_metric("loss", 0.1)

        best = exp.best_run(metric="loss", maximize=False)
        assert best["metrics"]["loss"] == 0.1

    def test_best_run_no_runs_raises(self, tmp_path: Path) -> None:
        exp = Experiment("empty", base_dir=str(tmp_path))
        with pytest.raises(ValueError, match="No runs"):
            exp.best_run()

    def test_compare_runs(self, tmp_path: Path) -> None:
        exp = Experiment("compare_test", base_dir=str(tmp_path))

        with exp.start_run(params={"lr": 0.01}) as run:
            run.log_metric("sharpe_ratio", 1.2)

        with exp.start_run(params={"lr": 0.001}) as run:
            run.log_metric("sharpe_ratio", 1.8)

        df = exp.compare_runs()
        assert len(df) == 2
        assert "param_lr" in df.columns
        assert "metric_sharpe_ratio" in df.columns

    def test_log_artifact(self, tmp_path: Path) -> None:
        exp = Experiment("artifact_test", base_dir=str(tmp_path))

        with exp.start_run() as run:
            run.log_artifact("weights", [1.0, 2.0, 3.0])

        artifact_path = run.run_dir / "weights.pkl"
        assert artifact_path.exists()
