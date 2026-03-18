"""Tests for wraquant.experiment.grid."""

from __future__ import annotations

from wraquant.experiment.grid import ParameterGrid, grid_search, random_search


class TestParameterGrid:
    def test_generates_correct_number_of_combinations(self) -> None:
        grid = ParameterGrid({"fast_ma": [5, 10, 20], "slow_ma": [50, 100, 200]})
        combos = list(grid)
        assert len(combos) == 9
        assert len(grid) == 9

    def test_single_param(self) -> None:
        grid = ParameterGrid({"x": [1, 2, 3]})
        combos = list(grid)
        assert len(combos) == 3
        assert combos[0] == {"x": 1}

    def test_empty_grid(self) -> None:
        grid = ParameterGrid({})
        combos = list(grid)
        # itertools.product with no iterables yields one empty tuple
        assert len(combos) == 1
        assert combos[0] == {}


class TestGridSearch:
    def test_finds_optimal_parameters(self) -> None:
        def objective(x: int, y: int) -> float:
            return -(float((x - 3) ** 2 + (y - 7) ** 2))

        result = grid_search(
            objective,
            ParameterGrid({"x": [1, 2, 3, 4, 5], "y": [5, 6, 7, 8, 9]}),
        )

        assert result["best_params"] == {"x": 3, "y": 7}
        assert result["best_score"] == 0.0

    def test_results_sorted_by_score_descending(self) -> None:
        def objective(a: int) -> float:
            return float(a)

        result = grid_search(objective, {"a": [1, 2, 3]})
        scores = [r["score"] for r in result["all_results"]]
        assert scores == sorted(scores, reverse=True)

    def test_accepts_dict_param_grid(self) -> None:
        result = grid_search(lambda x: float(x), {"x": [10, 20, 30]})
        assert result["best_params"] == {"x": 30}
        assert result["best_score"] == 30.0


class TestRandomSearch:
    def test_returns_correct_number_of_iterations(self) -> None:
        result = random_search(
            lambda x: float(x),
            {"x": [1, 2, 3]},
            n_iter=50,
            seed=42,
        )
        assert len(result["all_results"]) == 50

    def test_results_sorted_by_score(self) -> None:
        result = random_search(
            lambda x: float(x),
            {"x": (0.0, 10.0)},
            n_iter=30,
            seed=0,
        )
        scores = [r["score"] for r in result["all_results"]]
        assert scores == sorted(scores, reverse=True)

    def test_log_uniform_distribution(self) -> None:
        result = random_search(
            lambda lr: -lr,
            {"lr": {"type": "log-uniform", "low": 1e-5, "high": 1e-1}},
            n_iter=20,
            seed=99,
        )
        # All sampled values should fall in range
        for r in result["all_results"]:
            assert 1e-5 <= r["params"]["lr"] <= 1e-1

    def test_choice_distribution(self) -> None:
        result = random_search(
            lambda c: float(ord(c)),
            {"c": {"type": "choice", "values": ["a", "b", "c"]}},
            n_iter=15,
            seed=7,
        )
        for r in result["all_results"]:
            assert r["params"]["c"] in ("a", "b", "c")

    def test_seed_reproducibility(self) -> None:
        dist = {"x": (0.0, 1.0)}
        r1 = random_search(lambda x: x, dist, n_iter=10, seed=42)
        r2 = random_search(lambda x: x, dist, n_iter=10, seed=42)
        scores1 = [r["score"] for r in r1["all_results"]]
        scores2 = [r["score"] for r in r2["all_results"]]
        assert scores1 == scores2
