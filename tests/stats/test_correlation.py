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


# ---------------------------------------------------------------------------
# Partial correlation
# ---------------------------------------------------------------------------


class TestPartialCorrelation:
    def test_shape(self) -> None:
        from wraquant.stats.correlation import partial_correlation

        df = _make_multi_asset()
        pcorr = partial_correlation(df)
        assert pcorr.shape == (3, 3)

    def test_diagonal_is_one(self) -> None:
        from wraquant.stats.correlation import partial_correlation

        df = _make_multi_asset()
        pcorr = partial_correlation(df)
        np.testing.assert_allclose(np.diag(pcorr.values), 1.0, atol=1e-10)

    def test_symmetric(self) -> None:
        from wraquant.stats.correlation import partial_correlation

        df = _make_multi_asset()
        pcorr = partial_correlation(df)
        np.testing.assert_allclose(pcorr.values, pcorr.values.T, atol=1e-10)

    def test_removes_confounding(self) -> None:
        """Partial correlation should be lower when confounding is removed."""
        from wraquant.stats.correlation import partial_correlation

        rng = np.random.default_rng(42)
        z = rng.normal(0, 1, 300)
        data = pd.DataFrame({
            "A": z + rng.normal(0, 0.3, 300),
            "B": z + rng.normal(0, 0.3, 300),
            "Z": z,
        })
        raw_corr = data[["A", "B"]].corr().iloc[0, 1]
        pcorr = partial_correlation(data)
        # Partial correlation between A and B (controlling for Z) should be
        # smaller in magnitude than raw correlation
        assert abs(pcorr.loc["A", "B"]) < abs(raw_corr)


# ---------------------------------------------------------------------------
# Distance correlation
# ---------------------------------------------------------------------------


class TestDistanceCorrelation:
    def test_positive_for_dependent(self) -> None:
        from wraquant.stats.correlation import distance_correlation

        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        y = x ** 2 + rng.normal(0, 0.3, 200)
        dcor = distance_correlation(x, y)
        assert dcor > 0.3

    def test_near_zero_for_independent(self) -> None:
        from wraquant.stats.correlation import distance_correlation

        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 300)
        y = rng.normal(0, 1, 300)
        dcor = distance_correlation(x, y)
        assert dcor < 0.3

    def test_bounded(self) -> None:
        from wraquant.stats.correlation import distance_correlation

        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        y = 2 * x + rng.normal(0, 0.5, 100)
        dcor = distance_correlation(x, y)
        assert 0 <= dcor <= 1.0


# ---------------------------------------------------------------------------
# Mutual information
# ---------------------------------------------------------------------------


class TestMutualInformation:
    def test_positive_for_dependent(self) -> None:
        from wraquant.stats.correlation import mutual_information

        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 500)
        y = x + rng.normal(0, 0.5, 500)
        mi = mutual_information(x, y)
        assert mi > 0

    def test_near_zero_for_independent(self) -> None:
        from wraquant.stats.correlation import mutual_information

        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000)
        y = rng.normal(0, 1, 1000)
        mi = mutual_information(x, y)
        assert mi < 0.3

    def test_kde_method(self) -> None:
        from wraquant.stats.correlation import mutual_information

        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 300)
        y = x + rng.normal(0, 0.5, 300)
        mi = mutual_information(x, y, method="kde")
        assert mi > 0


# ---------------------------------------------------------------------------
# Correlation significance
# ---------------------------------------------------------------------------


class TestCorrelationSignificance:
    def test_significant_correlation(self) -> None:
        from wraquant.stats.correlation import correlation_significance

        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        y = 0.5 * x + rng.normal(0, 1, 200)
        result = correlation_significance(x, y)
        assert result["p_value"] < 0.05
        assert result["ci_lower"] < result["r"] < result["ci_upper"]

    def test_keys(self) -> None:
        from wraquant.stats.correlation import correlation_significance

        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        y = rng.normal(0, 1, 100)
        result = correlation_significance(x, y)
        assert set(result.keys()) == {"r", "t_stat", "p_value", "ci_lower", "ci_upper"}


# ---------------------------------------------------------------------------
# Minimum spanning tree
# ---------------------------------------------------------------------------


class TestMinimumSpanningTree:
    def test_shape(self) -> None:
        from wraquant.stats.correlation import minimum_spanning_tree_correlation

        df = _make_multi_asset()
        corr = correlation_matrix(df)
        mst = minimum_spanning_tree_correlation(corr)
        assert mst.shape == (3, 3)

    def test_edge_count(self) -> None:
        from wraquant.stats.correlation import minimum_spanning_tree_correlation

        rng = np.random.default_rng(42)
        data = pd.DataFrame(rng.normal(0, 1, (100, 5)), columns=list("ABCDE"))
        corr = correlation_matrix(data)
        mst = minimum_spanning_tree_correlation(corr)
        # MST with p nodes has p-1 edges, each counted twice in adjacency
        n_edges = (mst.values > 0).sum()
        assert n_edges == 2 * (5 - 1)
