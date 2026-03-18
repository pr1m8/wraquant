"""Tests for panel data econometrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.econometrics.panel import (
    between_effects,
    first_difference,
    fixed_effects,
    hausman_test,
    pooled_ols,
    random_effects,
)


def _make_panel(
    n_entities: int = 10,
    n_periods: int = 20,
    seed: int = 42,
) -> tuple[pd.Series, pd.DataFrame]:
    """Generate a panel dataset with known coefficients.

    True DGP: y_it = 2 * x1_it + 3 * x2_it + alpha_i + epsilon_it
    """
    rng = np.random.default_rng(seed)
    n = n_entities * n_periods

    entity = np.repeat(np.arange(n_entities), n_periods)
    time = np.tile(np.arange(n_periods), n_entities)

    # Entity fixed effects
    alpha = rng.normal(0, 2, n_entities)
    alpha_expanded = np.repeat(alpha, n_periods)

    x1 = rng.normal(0, 1, n) + 0.5 * alpha_expanded  # correlated with FE
    x2 = rng.normal(0, 1, n)
    eps = rng.normal(0, 0.5, n)

    y = 2.0 * x1 + 3.0 * x2 + alpha_expanded + eps

    df = pd.DataFrame({
        "entity": entity,
        "time": time,
        "x1": x1,
        "x2": x2,
    })
    y_series = pd.Series(y, name="y")

    return y_series, df


class TestFixedEffects:
    def test_recovers_known_coefficients(self) -> None:
        y, X = _make_panel(n_entities=20, n_periods=50, seed=123)
        result = fixed_effects(y, X, entity_col="entity")

        # Should recover beta1 ~ 2.0, beta2 ~ 3.0
        assert abs(result["coefficients"]["x1"] - 2.0) < 0.3
        assert abs(result["coefficients"]["x2"] - 3.0) < 0.3

    def test_output_structure(self) -> None:
        y, X = _make_panel()
        result = fixed_effects(y, X, entity_col="entity")

        assert "coefficients" in result
        assert "std_errors" in result
        assert "t_stats" in result
        assert "p_values" in result
        assert "r_squared" in result
        assert "entity_effects" in result
        assert "residuals" in result
        assert "nobs" in result
        assert "n_entities" in result

    def test_n_entities(self) -> None:
        y, X = _make_panel(n_entities=15)
        result = fixed_effects(y, X, entity_col="entity")
        assert result["n_entities"] == 15

    def test_two_way_fe(self) -> None:
        y, X = _make_panel()
        result = fixed_effects(y, X, entity_col="entity", time_col="time")
        assert "coefficients" in result
        assert result["r_squared"] >= 0


class TestPooledOLS:
    def test_output_structure(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        y = X @ np.array([1.0, 2.0]) + rng.normal(0, 0.5, n)

        result = pooled_ols(y, X)

        assert "coefficients" in result
        assert "std_errors" in result
        assert "t_stats" in result
        assert "p_values" in result
        assert "r_squared" in result
        assert "adj_r_squared" in result
        assert "residuals" in result
        assert "nobs" in result
        assert result["nobs"] == n

    def test_coefficients_close(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        true_beta = np.array([1.0, 2.0])
        y = X @ true_beta + rng.normal(0, 0.3, n)

        result = pooled_ols(y, X)
        np.testing.assert_allclose(result["coefficients"], true_beta, atol=0.2)


class TestRandomEffects:
    def test_output_structure(self) -> None:
        y, X = _make_panel()
        result = random_effects(y, X, entity_col="entity")

        assert "coefficients" in result
        assert "std_errors" in result
        assert "t_stats" in result
        assert "p_values" in result
        assert "r_squared" in result
        assert "theta" in result
        assert "residuals" in result
        assert "nobs" in result


class TestHausmanTest:
    def test_returns_statistic_and_p_value(self) -> None:
        y, X = _make_panel(n_entities=20, n_periods=30)

        fe_result = fixed_effects(y, X, entity_col="entity")
        re_result = random_effects(y, X, entity_col="entity")

        h = hausman_test(fe_result, re_result)

        assert "statistic" in h
        assert "p_value" in h
        assert "df" in h
        assert "prefer" in h
        assert h["statistic"] >= 0
        assert 0 <= h["p_value"] <= 1
        assert h["prefer"] in ("fe", "re")

    def test_prefers_fe_with_correlated_effects(self) -> None:
        """When entity effects are correlated with X, Hausman should prefer FE."""
        y, X = _make_panel(n_entities=30, n_periods=50, seed=99)

        fe_result = fixed_effects(y, X, entity_col="entity")
        re_result = random_effects(y, X, entity_col="entity")

        h = hausman_test(fe_result, re_result)
        # With our DGP (x1 correlated with alpha_i), FE should be preferred
        assert h["prefer"] == "fe"


class TestBetweenEffects:
    def test_output_structure(self) -> None:
        y, X = _make_panel()
        result = between_effects(y, X, entity_col="entity")

        assert "coefficients" in result
        assert "r_squared" in result
        assert "n_entities" in result


class TestFirstDifference:
    def test_output_structure(self) -> None:
        y, X = _make_panel()
        result = first_difference(y, X, entity_col="entity", time_col="time")

        assert "coefficients" in result
        assert "std_errors" in result
        assert "r_squared" in result
        assert "nobs" in result
        # First differencing loses one observation per entity
        assert result["nobs"] < len(y)
