"""Tests for wraquant.ml.preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.ml.preprocessing import (
    combinatorial_purged_kfold,
    denoised_correlation,
    detoned_correlation,
    fractional_differentiation,
    purged_kfold,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_data() -> tuple[pd.DataFrame, pd.Series]:
    np.random.seed(42)
    n = 200
    X = pd.DataFrame(np.random.randn(n, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.random.choice([0, 1], size=n))
    return X, y


@pytest.fixture()
def return_matrix() -> pd.DataFrame:
    np.random.seed(99)
    return pd.DataFrame(
        np.random.randn(200, 10) * 0.01,
        columns=[f"asset_{i}" for i in range(10)],
    )


# ---------------------------------------------------------------------------
# purged_kfold
# ---------------------------------------------------------------------------


class TestPurgedKFold:
    def test_non_overlapping_with_embargo(self, sample_data: tuple) -> None:
        X, y = sample_data
        n = len(X)
        embargo_pct = 0.02
        embargo_size = int(n * embargo_pct)

        folds = list(purged_kfold(X, y, n_splits=5, embargo_pct=embargo_pct))
        assert len(folds) == 5

        for train_idx, test_idx in folds:
            # No index should appear in both train and test
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0

            # Train indices immediately after test should be embargoed
            test_end = test_idx.max()
            embargo_zone = set(range(test_end + 1, min(test_end + 1 + embargo_size, n)))
            leaked = embargo_zone & set(train_idx)
            assert len(leaked) == 0, f"Embargo zone leaked: {leaked}"

    def test_all_samples_tested(self, sample_data: tuple) -> None:
        X, y = sample_data
        all_test = np.concatenate([t for _, t in purged_kfold(X, y, n_splits=5)])
        assert len(set(all_test)) == len(X)


# ---------------------------------------------------------------------------
# combinatorial_purged_kfold
# ---------------------------------------------------------------------------


class TestCombinatorialPurgedKFold:
    def test_number_of_splits(self, sample_data: tuple) -> None:
        X, y = sample_data
        from math import comb

        n_splits, n_test = 5, 2
        folds = list(
            combinatorial_purged_kfold(X, y, n_splits=n_splits, n_test_splits=n_test)
        )
        assert len(folds) == comb(n_splits, n_test)

    def test_no_overlap(self, sample_data: tuple) -> None:
        X, y = sample_data
        for train_idx, test_idx in combinatorial_purged_kfold(
            X, y, n_splits=4, n_test_splits=2
        ):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0


# ---------------------------------------------------------------------------
# fractional_differentiation
# ---------------------------------------------------------------------------


class TestFractionalDifferentiation:
    def test_preserves_index(self) -> None:
        idx = pd.date_range("2020-01-01", periods=200)
        s = pd.Series(np.cumsum(np.random.randn(200)), index=idx, name="price")
        result = fractional_differentiation(s, d=0.5)
        assert result.index.isin(idx).all()
        assert result.name == s.name

    def test_d_zero_approx_original(self) -> None:
        """With d=0 the differentiated series should be very close to the
        original (identity transform)."""
        s = pd.Series(np.cumsum(np.random.randn(200)) + 100)
        result = fractional_differentiation(s, d=0.0, threshold=1e-5)
        # d=0 weights = [1], so result ≈ original
        common = s.loc[result.index]
        np.testing.assert_allclose(result.values, common.values, atol=1e-10)

    def test_d_one_approx_diff(self) -> None:
        """With d=1 the result should approximate the first difference."""
        np.random.seed(7)
        s = pd.Series(np.cumsum(np.random.randn(200)) + 100)
        result = fractional_differentiation(s, d=1.0, threshold=1e-5)
        first_diff = s.diff().loc[result.index]
        np.testing.assert_allclose(result.values, first_diff.values, atol=1e-6)


# ---------------------------------------------------------------------------
# denoised_correlation
# ---------------------------------------------------------------------------


class TestDenoisedCorrelation:
    def test_valid_correlation_matrix(self, return_matrix: pd.DataFrame) -> None:
        corr = denoised_correlation(return_matrix)
        # Shape
        n = return_matrix.shape[1]
        assert corr.shape == (n, n)
        # Symmetric
        np.testing.assert_allclose(corr, corr.T, atol=1e-10)
        # Diagonal is 1
        np.testing.assert_allclose(np.diag(corr), np.ones(n), atol=1e-10)
        # All entries in [-1, 1]
        assert np.all(corr >= -1.0 - 1e-10)
        assert np.all(corr <= 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# detoned_correlation
# ---------------------------------------------------------------------------


class TestDetonedCorrelation:
    def test_valid_output(self, return_matrix: pd.DataFrame) -> None:
        corr = np.array(return_matrix.corr())
        detoned = detoned_correlation(corr, n_components=1)
        n = corr.shape[0]
        assert detoned.shape == (n, n)
        np.testing.assert_allclose(detoned, detoned.T, atol=1e-10)
        np.testing.assert_allclose(np.diag(detoned), np.ones(n), atol=1e-10)
