"""Tests for correlation and covariance estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wraquant.stats.correlation import (
    correlation_matrix,
    rolling_correlation,
    shrunk_covariance,
)


def _make_multi_asset(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    data = rng.normal(0.0003, 0.015, size=(n, 3))
    return pd.DataFrame(data, index=dates, columns=["A", "B", "C"])


class TestCorrelationMatrix:
    def test_shape(self) -> None:
        df = _make_multi_asset()
        corr = correlation_matrix(df)
        assert corr.shape == (3, 3)

    def test_diagonal_is_one(self) -> None:
        df = _make_multi_asset()
        corr = correlation_matrix(df)
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-10)

    def test_symmetric(self) -> None:
        df = _make_multi_asset()
        corr = correlation_matrix(df)
        np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-10)

    def test_spearman(self) -> None:
        df = _make_multi_asset()
        corr = correlation_matrix(df, method="spearman")
        assert corr.shape == (3, 3)


class TestShrunkCovariance:
    def test_ledoit_wolf_shape(self) -> None:
        df = _make_multi_asset()
        cov = shrunk_covariance(df, method="ledoit_wolf")
        assert cov.shape == (3, 3)

    def test_positive_diagonal(self) -> None:
        df = _make_multi_asset()
        cov = shrunk_covariance(df)
        assert (np.diag(cov.values) > 0).all()

    def test_oas(self) -> None:
        df = _make_multi_asset()
        cov = shrunk_covariance(df, method="oas")
        assert cov.shape == (3, 3)

    def test_symmetric(self) -> None:
        df = _make_multi_asset()
        cov = shrunk_covariance(df)
        np.testing.assert_allclose(cov.values, cov.values.T, atol=1e-10)


class TestRollingCorrelation:
    def test_length(self) -> None:
        df = _make_multi_asset()
        rc = rolling_correlation(df["A"], df["B"], window=20)
        assert len(rc) == len(df)

    def test_bounded(self) -> None:
        df = _make_multi_asset()
        rc = rolling_correlation(df["A"], df["B"], window=20).dropna()
        assert (rc >= -1.0).all()
        assert (rc <= 1.0).all()
