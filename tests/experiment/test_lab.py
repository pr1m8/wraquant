"""Tests for the wraquant Experiment Lab.

Tests cover:
    - Lab creation and experiment creation
    - Parameter grid generation (correct count)
    - Single run produces metrics
    - Walk-forward splits are chronological
    - Purged splits have gaps
    - Results.best() returns correct winner
    - Results.summary() has correct columns
    - Results.stability() across folds
    - Results.parameter_sensitivity() grouping
    - Save/load roundtrip
    - Parallel execution matches sequential
    - Regime breakdown runs without error
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wraquant.experiment.cv import (
    combinatorial_purged_splits,
    purged_kfold_splits,
    rolling_splits,
    walk_forward_splits,
)
from wraquant.experiment.lab import Experiment, Lab
from wraquant.experiment.results import ExperimentResults
from wraquant.experiment.runner import ExperimentRunner, GridSpec, RunResult
from wraquant.experiment.tracking import ExperimentStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_prices(n: int = 500, seed: int = 42) -> pd.Series:
    """Generate synthetic price series."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.015, n)
    prices = 100 * np.cumprod(1 + returns)
    return pd.Series(prices, name="price")


def simple_ma_strategy(
    prices: pd.Series, fast: int = 10, slow: int = 30
) -> pd.Series:
    """Simple moving average crossover strategy for testing.

    Buys when fast MA > slow MA, flat otherwise.
    Returns per-period strategy returns.
    """
    sma_fast = prices.rolling(fast).mean()
    sma_slow = prices.rolling(slow).mean()
    signals = (sma_fast > sma_slow).astype(float)
    returns = prices.pct_change() * signals.shift(1)
    return returns.dropna()


def constant_return_strategy(
    prices: pd.Series, scale: float = 1.0
) -> pd.Series:
    """Strategy that adds a scaled constant to returns.

    Higher scale => higher mean return => higher Sharpe.
    Useful for testing where we know the exact ordering.
    """
    returns = prices.pct_change().dropna() + scale * 0.001
    return returns


# ---------------------------------------------------------------------------
# Tests: GridSpec
# ---------------------------------------------------------------------------


class TestGridSpec:
    def test_correct_count(self) -> None:
        grid = GridSpec({"a": [1, 2, 3], "b": [10, 20]})
        assert len(grid) == 6

    def test_iteration(self) -> None:
        grid = GridSpec({"x": [1, 2]})
        combos = list(grid)
        assert len(combos) == 2
        assert combos[0] == {"x": 1}
        assert combos[1] == {"x": 2}

    def test_single_value(self) -> None:
        grid = GridSpec({"a": [42]})
        assert len(grid) == 1
        assert list(grid) == [{"a": 42}]


# ---------------------------------------------------------------------------
# Tests: CV methods
# ---------------------------------------------------------------------------


class TestWalkForwardSplits:
    def test_correct_number_of_splits(self) -> None:
        splits = walk_forward_splits(1000, n_splits=5)
        assert len(splits) == 5

    def test_train_always_starts_at_zero(self) -> None:
        splits = walk_forward_splits(1000, n_splits=5)
        for train_idx, _ in splits:
            assert train_idx[0] == 0

    def test_train_grows(self) -> None:
        splits = walk_forward_splits(1000, n_splits=5)
        sizes = [len(train) for train, _ in splits]
        assert sizes == sorted(sizes)
        assert sizes[0] < sizes[-1]

    def test_no_overlap(self) -> None:
        splits = walk_forward_splits(1000, n_splits=5)
        for train_idx, test_idx in splits:
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0

    def test_chronological_order(self) -> None:
        """Test indices are always going forward -- train before test."""
        splits = walk_forward_splits(1000, n_splits=5)
        for train_idx, test_idx in splits:
            assert train_idx[-1] < test_idx[0]

    def test_min_train_pct(self) -> None:
        splits = walk_forward_splits(1000, n_splits=3, min_train_pct=0.7)
        assert len(splits[0][0]) >= 700

    def test_invalid_min_train_pct_raises(self) -> None:
        with pytest.raises(ValueError):
            walk_forward_splits(1000, min_train_pct=0.0)
        with pytest.raises(ValueError):
            walk_forward_splits(1000, min_train_pct=1.0)


class TestRollingSplits:
    def test_correct_number_of_splits(self) -> None:
        splits = rolling_splits(1000, n_splits=5)
        assert len(splits) == 5

    def test_fixed_train_size(self) -> None:
        splits = rolling_splits(1000, n_splits=5, window_pct=0.6)
        sizes = [len(train) for train, _ in splits]
        assert all(s == sizes[0] for s in sizes)

    def test_no_overlap(self) -> None:
        splits = rolling_splits(1000, n_splits=5)
        for train_idx, test_idx in splits:
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0


class TestPurgedKfoldSplits:
    def test_correct_number_of_folds(self) -> None:
        splits = purged_kfold_splits(1000, n_splits=5)
        assert len(splits) == 5

    def test_embargo_creates_gap(self) -> None:
        """Purged splits must have a gap between train and test boundaries."""
        splits = purged_kfold_splits(1000, n_splits=5, embargo_pct=0.05)
        for train_idx, test_idx in splits:
            # No overlap
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0

            # Check that there is a gap around test boundaries
            test_start = test_idx[0]
            test_end = test_idx[-1]
            embargo = int(1000 * 0.05)

            # Train should not contain indices within embargo of test
            near_test = set(range(max(test_start - embargo, 0), min(test_end + embargo + 1, 1000)))
            train_set = set(train_idx)
            gap = near_test - set(test_idx) - train_set
            # The gap should exist (some indices near test should be excluded from train)
            assert len(gap) > 0 or test_start == 0 or test_end == 999

    def test_invalid_n_splits_raises(self) -> None:
        with pytest.raises(ValueError):
            purged_kfold_splits(1000, n_splits=1)


class TestCombinatorialPurgedSplits:
    def test_correct_number_of_combinations(self) -> None:
        splits = combinatorial_purged_splits(1000, n_splits=6, n_test_groups=2)
        # C(6, 2) = 15
        assert len(splits) == 15

    def test_no_overlap(self) -> None:
        splits = combinatorial_purged_splits(1000, n_splits=5, n_test_groups=2)
        for train_idx, test_idx in splits:
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0

    def test_invalid_n_test_groups_raises(self) -> None:
        with pytest.raises(ValueError):
            combinatorial_purged_splits(1000, n_splits=5, n_test_groups=5)


# ---------------------------------------------------------------------------
# Tests: ExperimentRunner
# ---------------------------------------------------------------------------


class TestExperimentRunner:
    def test_run_single_produces_metrics(self) -> None:
        prices = _make_prices(200)
        runner = ExperimentRunner(
            strategy_fn=simple_ma_strategy,
            data=prices,
            params={"fast": [5, 10], "slow": [20, 30]},
        )

        all_idx = np.arange(len(prices))
        result = runner.run_single(
            param_combo={"fast": 5, "slow": 20},
            train_idx=all_idx[:100],
            test_idx=all_idx[100:],
            fold=0,
        )

        assert isinstance(result, RunResult)
        assert "sharpe" in result.metrics
        assert "total_return" in result.metrics
        assert "max_drawdown" in result.metrics
        assert "win_rate" in result.metrics
        assert "profit_factor" in result.metrics
        assert "omega" in result.metrics
        assert result.fold == 0
        assert result.params == {"fast": 5, "slow": 20}
        assert result.elapsed_seconds >= 0

    def test_run_grid_sequential(self) -> None:
        prices = _make_prices(300)
        runner = ExperimentRunner(
            strategy_fn=simple_ma_strategy,
            data=prices,
            params={"fast": [5, 10], "slow": [20, 30]},
        )

        results = runner.run_grid(cv="walk_forward", n_splits=3, parallel=False)

        # 4 param combos x 3 folds = 12 runs
        assert len(results) == 12
        assert all(isinstance(r, RunResult) for r in results)

    def test_run_grid_no_cv(self) -> None:
        prices = _make_prices(200)
        runner = ExperimentRunner(
            strategy_fn=simple_ma_strategy,
            data=prices,
            params={"fast": [5], "slow": [20]},
        )

        results = runner.run_grid(cv="none", n_splits=1, parallel=False)

        # 1 param combo x 1 fold = 1 run
        assert len(results) == 1

    def test_invalid_cv_raises(self) -> None:
        prices = _make_prices(200)
        runner = ExperimentRunner(
            strategy_fn=simple_ma_strategy,
            data=prices,
            params={"fast": [5], "slow": [20]},
        )
        with pytest.raises(ValueError, match="Unknown CV method"):
            runner.run_grid(cv="invalid_method")


# ---------------------------------------------------------------------------
# Tests: ExperimentResults
# ---------------------------------------------------------------------------


class TestExperimentResults:
    @pytest.fixture()
    def results(self) -> ExperimentResults:
        """Create results from a real strategy run."""
        prices = _make_prices(400)
        runner = ExperimentRunner(
            strategy_fn=constant_return_strategy,
            data=prices,
            params={"scale": [0.5, 1.0, 1.5, 2.0]},
        )
        runs = runner.run_grid(cv="walk_forward", n_splits=3, parallel=False)
        return ExperimentResults(
            runs=runs,
            experiment_name="scale_test",
            params={"scale": [0.5, 1.0, 1.5, 2.0]},
        )

    def test_summary_has_correct_shape(self, results: ExperimentResults) -> None:
        df = results.summary()
        assert len(df) == 4  # 4 scale values
        assert "scale" in df.columns
        assert "mean_sharpe" in df.columns
        assert "mean_total_return" in df.columns
        assert "mean_max_drawdown" in df.columns
        assert "n_folds" in df.columns

    def test_best_returns_correct_winner(self, results: ExperimentResults) -> None:
        """Higher scale = higher returns = higher sharpe (for positive-mean data)."""
        best = results.best(metric="sharpe")
        assert "params" in best
        assert "metrics" in best
        assert "n_folds" in best
        # The best scale should be the highest (2.0) for positive-mean data
        assert best["params"]["scale"] == 2.0

    def test_worst_returns_minimum(self, results: ExperimentResults) -> None:
        worst = results.worst(metric="sharpe")
        assert "params" in worst
        assert worst["params"]["scale"] == 0.5

    def test_top_n(self, results: ExperimentResults) -> None:
        top = results.top_n(n=2, metric="sharpe")
        assert len(top) == 2
        assert top.iloc[0]["scale"] == 2.0

    def test_stability(self, results: ExperimentResults) -> None:
        stability_df = results.stability()
        assert len(stability_df) > 0
        assert "sharpe_mean" in stability_df.columns
        assert "sharpe_std" in stability_df.columns
        assert "sharpe_min" in stability_df.columns
        assert "sharpe_max" in stability_df.columns

    def test_parameter_sensitivity(self, results: ExperimentResults) -> None:
        sens = results.parameter_sensitivity("scale")
        assert len(sens) == 4  # 4 unique scale values
        assert "scale" in sens.columns
        assert "mean_sharpe" in sens.columns

    def test_parameter_sensitivity_invalid_param(
        self, results: ExperimentResults
    ) -> None:
        with pytest.raises(ValueError, match="not in grid"):
            results.parameter_sensitivity("nonexistent")

    def test_compare_metrics(self, results: ExperimentResults) -> None:
        df = results.compare_metrics()
        assert len(df) == 12  # 4 combos x 3 folds
        assert "scale" in df.columns
        assert "fold" in df.columns
        assert "sharpe" in df.columns

    def test_best_invalid_metric_raises(self, results: ExperimentResults) -> None:
        with pytest.raises(ValueError, match="not found"):
            results.best(metric="nonexistent_metric")

    def test_report_dict(self, results: ExperimentResults) -> None:
        report = results.report(format="dict")
        assert "experiment_name" in report
        assert report["experiment_name"] == "scale_test"
        assert "best" in report
        assert "worst" in report
        assert "n_param_combos" in report
        assert report["n_param_combos"] == 4

    def test_report_dataframe(self, results: ExperimentResults) -> None:
        df = results.report(format="dataframe")
        assert isinstance(df, pd.DataFrame)

    def test_report_invalid_format_raises(self, results: ExperimentResults) -> None:
        with pytest.raises(ValueError, match="Unknown format"):
            results.report(format="invalid")


# ---------------------------------------------------------------------------
# Tests: Save / Load roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_results_save_load_roundtrip(self, tmp_path: Path) -> None:
        prices = _make_prices(300)
        runner = ExperimentRunner(
            strategy_fn=constant_return_strategy,
            data=prices,
            params={"scale": [0.5, 1.0]},
        )
        runs = runner.run_grid(cv="walk_forward", n_splits=2, parallel=False)
        original = ExperimentResults(
            runs=runs,
            experiment_name="roundtrip_test",
            params={"scale": [0.5, 1.0]},
        )

        # Save
        save_dir = tmp_path / "saved_experiment"
        original.save(save_dir)

        # Verify files exist
        assert (save_dir / "metadata.json").exists()
        assert (save_dir / "summary.parquet").exists()
        assert (save_dir / "all_runs.parquet").exists()

        # Load
        loaded = ExperimentResults.load(save_dir)
        assert loaded.experiment_name == "roundtrip_test"
        assert len(loaded.runs) == len(original.runs)

        # Metrics should match
        orig_best = original.best()
        loaded_best = loaded.best()
        assert orig_best["params"] == loaded_best["params"]
        assert abs(orig_best["metrics"]["sharpe"] - loaded_best["metrics"]["sharpe"]) < 1e-6

    def test_store_save_load_delete(self, tmp_path: Path) -> None:
        store = ExperimentStore(storage_dir=str(tmp_path / "store"))

        prices = _make_prices(200)
        runner = ExperimentRunner(
            strategy_fn=constant_return_strategy,
            data=prices,
            params={"scale": [1.0]},
        )
        runs = runner.run_grid(cv="none", n_splits=1, parallel=False)
        results = ExperimentResults(
            runs=runs,
            experiment_name="store_test",
            params={"scale": [1.0]},
        )

        # Save
        store.save_experiment("store_test", results)

        # List
        listing = store.list_experiments()
        assert len(listing) == 1
        assert listing.iloc[0]["name"] == "store_test"

        # Load
        loaded = store.load_experiment("store_test")
        assert loaded.experiment_name == "store_test"

        # Delete
        store.delete_experiment("store_test")
        listing = store.list_experiments()
        assert len(listing) == 0

    def test_store_load_nonexistent_raises(self, tmp_path: Path) -> None:
        store = ExperimentStore(storage_dir=str(tmp_path / "empty"))
        with pytest.raises(FileNotFoundError):
            store.load_experiment("nonexistent")

    def test_store_delete_nonexistent_raises(self, tmp_path: Path) -> None:
        store = ExperimentStore(storage_dir=str(tmp_path / "empty2"))
        with pytest.raises(FileNotFoundError):
            store.delete_experiment("nonexistent")

    def test_store_compare_experiments(self, tmp_path: Path) -> None:
        store = ExperimentStore(storage_dir=str(tmp_path / "compare"))
        prices = _make_prices(200)

        for name, scale_vals in [("exp_a", [0.5, 1.0]), ("exp_b", [1.5, 2.0])]:
            runner = ExperimentRunner(
                strategy_fn=constant_return_strategy,
                data=prices,
                params={"scale": scale_vals},
            )
            runs = runner.run_grid(cv="none", n_splits=1, parallel=False)
            results = ExperimentResults(
                runs=runs,
                experiment_name=name,
                params={"scale": scale_vals},
            )
            store.save_experiment(name, results)

        comparison = store.compare_experiments(["exp_a", "exp_b"])
        assert len(comparison) == 2
        assert "experiment" in comparison.columns


# ---------------------------------------------------------------------------
# Tests: Lab (integration)
# ---------------------------------------------------------------------------


class TestLab:
    def test_lab_create_and_run(self, tmp_path: Path) -> None:
        lab = Lab("test_lab", storage_dir=str(tmp_path / "lab"))
        prices = _make_prices(300)

        exp = lab.create(
            "ma_crossover",
            strategy_fn=simple_ma_strategy,
            params={"fast": [5, 10], "slow": [20, 30]},
            data=prices,
        )

        assert exp.grid_size() == 4
        results = exp.run(cv="walk_forward", n_splits=3, parallel=False)

        assert isinstance(results, ExperimentResults)
        assert len(results.runs) == 12

    def test_lab_auto_persistence(self, tmp_path: Path) -> None:
        lab = Lab("persist_lab", storage_dir=str(tmp_path / "persist"))
        prices = _make_prices(200)

        exp = lab.create(
            "auto_save_test",
            strategy_fn=constant_return_strategy,
            params={"scale": [1.0]},
            data=prices,
        )
        exp.run(cv="none", n_splits=1, parallel=False)

        # Should have been auto-saved
        listing = lab.list_experiments()
        assert len(listing) == 1
        assert listing.iloc[0]["name"] == "auto_save_test"

    def test_lab_load_after_run(self, tmp_path: Path) -> None:
        lab = Lab("load_lab", storage_dir=str(tmp_path / "load"))
        prices = _make_prices(200)

        exp = lab.create(
            "load_test",
            strategy_fn=constant_return_strategy,
            params={"scale": [1.0, 2.0]},
            data=prices,
        )
        original_results = exp.run(cv="none", n_splits=1, parallel=False)

        loaded = lab.load("load_test")
        assert loaded.experiment_name == "load_test"
        assert len(loaded.runs) == len(original_results.runs)

    def test_lab_compare(self, tmp_path: Path) -> None:
        lab = Lab("compare_lab", storage_dir=str(tmp_path / "compare"))
        prices = _make_prices(200)

        for name in ["exp_1", "exp_2"]:
            exp = lab.create(
                name,
                strategy_fn=constant_return_strategy,
                params={"scale": [1.0]},
                data=prices,
            )
            exp.run(cv="none", n_splits=1, parallel=False)

        comparison = lab.compare(["exp_1", "exp_2"])
        assert len(comparison) == 2

    def test_lab_delete(self, tmp_path: Path) -> None:
        lab = Lab("delete_lab", storage_dir=str(tmp_path / "delete"))
        prices = _make_prices(200)

        exp = lab.create(
            "to_delete",
            strategy_fn=constant_return_strategy,
            params={"scale": [1.0]},
            data=prices,
        )
        exp.run(cv="none", n_splits=1, parallel=False)

        assert len(lab.list_experiments()) == 1
        lab.delete("to_delete")
        assert len(lab.list_experiments()) == 0


# ---------------------------------------------------------------------------
# Tests: Parallel vs sequential
# ---------------------------------------------------------------------------


class TestParallelExecution:
    def test_parallel_matches_sequential(self) -> None:
        """If joblib is available, parallel results should match sequential."""
        prices = _make_prices(300)
        params = {"fast": [5, 10], "slow": [20, 30]}

        # Sequential
        runner_seq = ExperimentRunner(
            strategy_fn=simple_ma_strategy,
            data=prices,
            params=params,
        )
        results_seq = runner_seq.run_grid(
            cv="walk_forward", n_splits=2, parallel=False
        )

        # Parallel (may fall back to sequential if joblib not installed)
        runner_par = ExperimentRunner(
            strategy_fn=simple_ma_strategy,
            data=prices,
            params=params,
        )
        results_par = runner_par.run_grid(
            cv="walk_forward", n_splits=2, parallel=True, n_jobs=1
        )

        # Same number of results
        assert len(results_seq) == len(results_par)

        # Sort by params+fold for comparison
        def sort_key(r: RunResult) -> tuple:
            return (tuple(sorted(r.params.items())), r.fold)

        seq_sorted = sorted(results_seq, key=sort_key)
        par_sorted = sorted(results_par, key=sort_key)

        for s, p in zip(seq_sorted, par_sorted):
            assert s.params == p.params
            assert s.fold == p.fold
            # Metrics should be identical
            for metric in ["sharpe", "total_return", "max_drawdown"]:
                assert abs(s.metrics[metric] - p.metrics[metric]) < 1e-10, (
                    f"Mismatch in {metric}: {s.metrics[metric]} vs {p.metrics[metric]}"
                )


# ---------------------------------------------------------------------------
# Tests: Regime breakdown (best-effort)
# ---------------------------------------------------------------------------


class TestRegimeBreakdown:
    def test_regime_breakdown_runs_without_error(self) -> None:
        """Regime breakdown should not crash; it may return empty if deps missing."""
        prices = _make_prices(400)
        runner = ExperimentRunner(
            strategy_fn=constant_return_strategy,
            data=prices,
            params={"scale": [1.0, 2.0]},
        )
        runs = runner.run_grid(cv="walk_forward", n_splits=3, parallel=False)
        results = ExperimentResults(
            runs=runs,
            experiment_name="regime_test",
            params={"scale": [1.0, 2.0]},
        )

        # Should not raise -- returns DataFrame (possibly empty)
        df = results.regime_breakdown()
        assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# Tests: Experiment standalone
# ---------------------------------------------------------------------------


class TestExperimentStandalone:
    def test_experiment_without_lab(self) -> None:
        """Experiment should work fine without a parent Lab."""
        prices = _make_prices(200)
        exp = Experiment(
            name="standalone",
            strategy_fn=simple_ma_strategy,
            params={"fast": [5, 10], "slow": [20]},
            data=prices,
        )

        assert exp.grid_size() == 2
        results = exp.run(cv="none", n_splits=1, parallel=False)
        assert len(results.runs) == 2

    def test_experiment_with_benchmark(self) -> None:
        """Experiment should compute benchmark-relative metrics."""
        prices = _make_prices(200)
        benchmark = prices.pct_change().dropna()

        exp = Experiment(
            name="with_bench",
            strategy_fn=constant_return_strategy,
            params={"scale": [1.0]},
            data=prices,
            benchmark=benchmark,
        )

        results = exp.run(cv="none", n_splits=1, parallel=False)
        assert len(results.runs) == 1


# ---------------------------------------------------------------------------
# Tests: CV with real data
# ---------------------------------------------------------------------------


class TestCVWithRealData:
    def test_walk_forward_with_strategy(self) -> None:
        """Full walk-forward CV run with MA strategy."""
        prices = _make_prices(500)
        exp = Experiment(
            name="wf_ma",
            strategy_fn=simple_ma_strategy,
            params={"fast": [5, 10, 15], "slow": [20, 30]},
            data=prices,
        )

        results = exp.run(cv="walk_forward", n_splits=4, parallel=False)
        assert len(results.runs) == 6 * 4  # 6 combos x 4 folds

        summary = results.summary()
        assert len(summary) == 6
        assert all(summary["n_folds"] == 4)

    def test_rolling_cv_with_strategy(self) -> None:
        """Full rolling CV run."""
        prices = _make_prices(500)
        exp = Experiment(
            name="rolling_ma",
            strategy_fn=simple_ma_strategy,
            params={"fast": [5], "slow": [20]},
            data=prices,
        )

        results = exp.run(cv="rolling", n_splits=3, parallel=False)
        assert len(results.runs) == 3

    def test_purged_kfold_with_strategy(self) -> None:
        """Full purged K-fold CV run."""
        prices = _make_prices(500)
        exp = Experiment(
            name="purged_ma",
            strategy_fn=simple_ma_strategy,
            params={"fast": [5], "slow": [20]},
            data=prices,
        )

        results = exp.run(cv="purged_kfold", n_splits=5, parallel=False)
        assert len(results.runs) == 5
