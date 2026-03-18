"""Tests for wraquant.ml.clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.ml.clustering import (
    correlation_clustering,
    optimal_clusters,
    regime_clustering,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def return_matrix() -> pd.DataFrame:
    np.random.seed(123)
    # Build two groups with correlated returns
    n = 200
    factor1 = np.random.randn(n) * 0.01
    factor2 = np.random.randn(n) * 0.01
    data = {}
    for i in range(5):
        data[f"g1_{i}"] = factor1 + np.random.randn(n) * 0.002
    for i in range(5):
        data[f"g2_{i}"] = factor2 + np.random.randn(n) * 0.002
    return pd.DataFrame(data)


@pytest.fixture()
def feature_matrix() -> np.ndarray:
    np.random.seed(77)
    # Two clear clusters in 2D
    c1 = np.random.randn(100, 2) + np.array([3, 3])
    c2 = np.random.randn(100, 2) + np.array([-3, -3])
    return np.vstack([c1, c2])


# ---------------------------------------------------------------------------
# correlation_clustering
# ---------------------------------------------------------------------------


class TestCorrelationClustering:
    def test_returns_valid_labels(self, return_matrix: pd.DataFrame) -> None:
        result = correlation_clustering(return_matrix, n_clusters=2)
        assert "labels" in result
        labels = result["labels"]
        assert len(labels) == return_matrix.shape[1]
        assert set(labels).issubset(set(range(result["n_clusters"])))

    def test_auto_clusters(self, return_matrix: pd.DataFrame) -> None:
        result = correlation_clustering(return_matrix, n_clusters=None)
        assert result["n_clusters"] >= 2

    def test_linkage_present(self, return_matrix: pd.DataFrame) -> None:
        result = correlation_clustering(return_matrix, method="hierarchical")
        assert result["linkage_matrix"] is not None


# ---------------------------------------------------------------------------
# regime_clustering
# ---------------------------------------------------------------------------


class TestRegimeClustering:
    def test_gmm_output(self, feature_matrix: np.ndarray) -> None:
        result = regime_clustering(feature_matrix, n_regimes=2, method="gmm")
        assert "labels" in result
        assert len(result["labels"]) == len(feature_matrix)
        assert result["n_regimes"] == 2
        assert result["model"] is not None

    def test_kmeans_output(self, feature_matrix: np.ndarray) -> None:
        result = regime_clustering(feature_matrix, n_regimes=2, method="kmeans")
        assert len(set(result["labels"])) == 2


# ---------------------------------------------------------------------------
# optimal_clusters
# ---------------------------------------------------------------------------


class TestOptimalClusters:
    def test_returns_int_in_range(self, feature_matrix: np.ndarray) -> None:
        k = optimal_clusters(feature_matrix, max_k=5, method="silhouette")
        assert isinstance(k, int)
        assert 2 <= k <= 5

    def test_bic_method(self, feature_matrix: np.ndarray) -> None:
        k = optimal_clusters(feature_matrix, max_k=5, method="bic")
        assert isinstance(k, int)
        assert 2 <= k <= 5
