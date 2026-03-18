"""Tests for wraquant.experiment.walk_forward."""

from __future__ import annotations

import numpy as np

from wraquant.experiment.walk_forward import (
    expanding_window_splits,
    rolling_window_splits,
    walk_forward_optimize,
)


class TestRollingWindowSplits:
    def test_correct_number_of_folds(self) -> None:
        splits = rolling_window_splits(n_samples=100, train_size=50, test_size=10)
        # Starts: 0, 10, 20, 30, 40 → 5 folds
        assert len(splits) == 5

    def test_train_size_is_fixed(self) -> None:
        splits = rolling_window_splits(n_samples=100, train_size=30, test_size=10)
        for train_idx, _ in splits:
            assert len(train_idx) == 30

    def test_test_size_is_fixed(self) -> None:
        splits = rolling_window_splits(n_samples=100, train_size=30, test_size=10)
        for _, test_idx in splits:
            assert len(test_idx) == 10

    def test_no_overlap_between_train_and_test(self) -> None:
        splits = rolling_window_splits(n_samples=200, train_size=50, test_size=20)
        for train_idx, test_idx in splits:
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0

    def test_custom_step_size(self) -> None:
        splits = rolling_window_splits(
            n_samples=100, train_size=50, test_size=10, step_size=5
        )
        # More folds with smaller step
        assert len(splits) > 5


class TestExpandingWindowSplits:
    def test_train_size_grows(self) -> None:
        splits = expanding_window_splits(n_samples=100, min_train_size=20, test_size=10)
        train_sizes = [len(train_idx) for train_idx, _ in splits]
        assert train_sizes == sorted(train_sizes)
        assert train_sizes[0] < train_sizes[-1]

    def test_train_always_starts_at_zero(self) -> None:
        splits = expanding_window_splits(n_samples=100, min_train_size=20, test_size=10)
        for train_idx, _ in splits:
            assert train_idx[0] == 0

    def test_no_overlap_between_train_and_test(self) -> None:
        splits = expanding_window_splits(n_samples=150, min_train_size=30, test_size=15)
        for train_idx, test_idx in splits:
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0

    def test_correct_number_of_folds(self) -> None:
        splits = expanding_window_splits(n_samples=100, min_train_size=50, test_size=10)
        # train_end starts at 50, advances by 10 each fold
        # 50, 60, 70, 80, 90 → 5 folds
        assert len(splits) == 5


class TestWalkForwardOptimize:
    def test_output_structure(self) -> None:
        data = np.arange(100, dtype=float)

        def objective(data: np.ndarray, a: int) -> float:
            return float(np.mean(data) * a)

        result = walk_forward_optimize(
            objective_fn=objective,
            param_grid={"a": [1, 2, 3]},
            data=data,
            train_size=30,
            test_size=10,
        )

        assert "fold_results" in result
        assert "aggregate_metrics" in result
        assert "stability" in result
        assert isinstance(result["fold_results"], list)
        assert len(result["fold_results"]) > 0

        # Check fold result keys
        fold = result["fold_results"][0]
        assert "best_params" in fold
        assert "train_score" in fold
        assert "test_score" in fold

        # Aggregate metrics
        agg = result["aggregate_metrics"]
        assert "mean_test_score" in agg
        assert "std_test_score" in agg

    def test_anchored_mode(self) -> None:
        data = np.arange(100, dtype=float)

        def objective(data: np.ndarray, a: int) -> float:
            return float(np.mean(data) * a)

        result = walk_forward_optimize(
            objective_fn=objective,
            param_grid={"a": [1, 2]},
            data=data,
            train_size=30,
            test_size=10,
            anchored=True,
        )

        # In anchored mode, training indices should always start at 0
        for fold in result["fold_results"]:
            assert fold["train_indices"][0] == 0

        # Training sizes should grow
        train_sizes = [len(fold["train_indices"]) for fold in result["fold_results"]]
        assert train_sizes == sorted(train_sizes)
