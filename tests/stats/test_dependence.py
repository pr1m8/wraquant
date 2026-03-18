"""Tests for advanced dependence measures module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wraquant.stats.dependence import (
    concordance_index,
    copula_selection,
    rank_correlation_matrix,
    tail_dependence_coefficient,
)


# ---------------------------------------------------------------------------
# Tail dependence coefficient
# ---------------------------------------------------------------------------


class TestTailDependenceCoefficient:
    def test_bounded(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000)
        y = 0.7 * x + rng.normal(0, 0.71, 1000)
        result = tail_dependence_coefficient(x, y)
        assert 0 <= result["upper_lambda"] <= 1
        assert 0 <= result["lower_lambda"] <= 1

    def test_keys(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        y = rng.normal(0, 1, 200)
        result = tail_dependence_coefficient(x, y)
        assert set(result.keys()) == {"upper_lambda", "lower_lambda"}

    def test_independent_low_tail_dep(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 2000)
        y = rng.normal(0, 1, 2000)
        result = tail_dependence_coefficient(x, y)
        # Independent normal variables should have low tail dependence
        assert result["upper_lambda"] < 0.5
        assert result["lower_lambda"] < 0.5


# ---------------------------------------------------------------------------
# Copula selection
# ---------------------------------------------------------------------------


class TestCopulaSelection:
    def test_returns_dataframe(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 500)
        y = 0.7 * x + rng.normal(0, 0.71, 500)
        result = copula_selection(x, y)
        assert isinstance(result["all_fits"], pd.DataFrame)

    def test_best_copula_is_string(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 500)
        y = 0.5 * x + rng.normal(0, 1, 500)
        result = copula_selection(x, y)
        assert isinstance(result["best_copula"], str)
        assert result["best_copula"] in ["gaussian", "student_t", "clayton", "gumbel"]

    def test_all_fits_has_correct_columns(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 300)
        y = 0.5 * x + rng.normal(0, 1, 300)
        result = copula_selection(x, y)
        df = result["all_fits"]
        if not df.empty:
            expected_cols = {"copula", "parameter", "log_likelihood", "aic"}
            assert expected_cols == set(df.columns)

    def test_sorted_by_aic(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 500)
        y = 0.7 * x + rng.normal(0, 0.71, 500)
        result = copula_selection(x, y)
        df = result["all_fits"]
        if len(df) > 1:
            assert df["aic"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Rank correlation matrix
# ---------------------------------------------------------------------------


class TestRankCorrelationMatrix:
    def test_shape(self) -> None:
        rng = np.random.default_rng(42)
        data = pd.DataFrame(rng.normal(0, 1, (100, 4)), columns=list("ABCD"))
        rcm = rank_correlation_matrix(data)
        assert rcm.shape == (4, 4)

    def test_diagonal_is_one(self) -> None:
        rng = np.random.default_rng(42)
        data = pd.DataFrame(rng.normal(0, 1, (100, 3)), columns=["X", "Y", "Z"])
        rcm = rank_correlation_matrix(data)
        np.testing.assert_allclose(np.diag(rcm.values), 1.0, atol=1e-10)

    def test_bounded(self) -> None:
        rng = np.random.default_rng(42)
        data = pd.DataFrame(rng.normal(0, 1, (100, 3)), columns=["X", "Y", "Z"])
        rcm = rank_correlation_matrix(data)
        assert (rcm.values >= -1.0).all()
        assert (rcm.values <= 1.0).all()


# ---------------------------------------------------------------------------
# Concordance index
# ---------------------------------------------------------------------------


class TestConcordanceIndex:
    def test_perfect_concordance(self) -> None:
        predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        c = concordance_index(predicted, observed)
        assert c == 1.0

    def test_perfect_discordance(self) -> None:
        predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        observed = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        c = concordance_index(predicted, observed)
        assert c == 0.0

    def test_bounded(self) -> None:
        rng = np.random.default_rng(42)
        predicted = rng.normal(0, 1, 100)
        observed = predicted + rng.normal(0, 0.5, 100)
        c = concordance_index(predicted, observed)
        assert 0 <= c <= 1

    def test_good_model(self) -> None:
        rng = np.random.default_rng(42)
        predicted = rng.normal(0, 1, 200)
        observed = predicted + rng.normal(0, 0.3, 200)
        c = concordance_index(predicted, observed)
        assert c > 0.7
