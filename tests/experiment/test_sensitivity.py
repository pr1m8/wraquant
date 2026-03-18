"""Tests for wraquant.experiment.sensitivity."""

from __future__ import annotations

import numpy as np

from wraquant.experiment.sensitivity import (
    parameter_heatmap,
    parameter_sensitivity,
    robustness_check,
    stability_score,
)


class TestParameterSensitivity:
    def test_output_shape(self) -> None:
        def objective(x: int, y: int) -> float:
            return float(x + y)

        df = parameter_sensitivity(
            objective,
            base_params={"x": 1, "y": 10},
            param_name="x",
            values=[1, 2, 3, 4, 5],
        )

        assert len(df) == 5
        assert "x" in df.columns
        assert "score" in df.columns

    def test_scores_change_with_parameter(self) -> None:
        def objective(x: int, y: int) -> float:
            return float(x * 10 + y)

        df = parameter_sensitivity(
            objective,
            base_params={"x": 1, "y": 5},
            param_name="x",
            values=[1, 2, 3],
        )

        assert df["score"].iloc[0] == 15.0
        assert df["score"].iloc[1] == 25.0
        assert df["score"].iloc[2] == 35.0


class TestParameterHeatmap:
    def test_output_shape(self) -> None:
        def objective(x: int, y: int) -> float:
            return float(x * y)

        df = parameter_heatmap(
            objective,
            base_params={},
            param1_name="x",
            param1_values=[1, 2, 3],
            param2_name="y",
            param2_values=[10, 20],
        )

        assert df.shape == (3, 2)
        assert df.index.name == "x"
        assert df.columns.name == "y"

    def test_correct_values(self) -> None:
        def objective(x: int, y: int) -> float:
            return float(x + y)

        df = parameter_heatmap(
            objective,
            base_params={},
            param1_name="x",
            param1_values=[1, 2],
            param2_name="y",
            param2_values=[10, 20],
        )

        assert df.loc[1, 10] == 11.0
        assert df.loc[2, 20] == 22.0


class TestStabilityScore:
    def test_perfect_stability(self) -> None:
        wf_results = {
            "fold_results": [
                {"best_params": {"a": 1, "b": 2}},
                {"best_params": {"a": 1, "b": 2}},
                {"best_params": {"a": 1, "b": 2}},
            ]
        }
        assert stability_score(wf_results) == 1.0

    def test_no_stability(self) -> None:
        wf_results = {
            "fold_results": [
                {"best_params": {"a": 1}},
                {"best_params": {"a": 2}},
                {"best_params": {"a": 3}},
            ]
        }
        assert stability_score(wf_results) == 0.0

    def test_single_fold(self) -> None:
        wf_results = {"fold_results": [{"best_params": {"a": 1}}]}
        assert stability_score(wf_results) == 1.0


class TestRobustnessCheck:
    def test_returns_expected_keys(self) -> None:
        def objective(x: float, y: float) -> float:
            return -(x**2 + y**2)

        result = robustness_check(
            objective,
            params={"x": 1.0, "y": 2.0},
            n_perturbations=50,
            noise_std=0.1,
            seed=42,
        )

        assert "base_score" in result
        assert "mean_score" in result
        assert "std_score" in result
        assert "min_score" in result
        assert "max_score" in result
        assert "scores" in result
        assert len(result["scores"]) == 50

    def test_base_score_is_correct(self) -> None:
        def objective(x: float) -> float:
            return x * 2.0

        result = robustness_check(
            objective,
            params={"x": 5.0},
            n_perturbations=10,
            seed=0,
        )
        assert result["base_score"] == 10.0

    def test_seed_reproducibility(self) -> None:
        def objective(x: float) -> float:
            return x**2

        r1 = robustness_check(objective, {"x": 3.0}, n_perturbations=20, seed=123)
        r2 = robustness_check(objective, {"x": 3.0}, n_perturbations=20, seed=123)
        np.testing.assert_array_equal(r1["scores"], r2["scores"])
