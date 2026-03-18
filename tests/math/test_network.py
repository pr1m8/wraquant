"""Tests for wraquant.math.network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.math.network import (
    centrality_measures,
    community_detection,
    contagion_simulation,
    correlation_network,
    granger_network,
    minimum_spanning_tree,
    systemic_risk_score,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def returns_df() -> pd.DataFrame:
    """Synthetic 5-asset correlated return data."""
    rng = np.random.default_rng(42)
    n = 500
    # Create correlated returns: assets 0-2 correlated, 3-4 correlated
    common1 = rng.standard_normal(n)
    common2 = rng.standard_normal(n)
    data = {
        "A": 0.01 * (common1 + 0.2 * rng.standard_normal(n)),
        "B": 0.01 * (common1 + 0.3 * rng.standard_normal(n)),
        "C": 0.01 * (common1 + 0.4 * rng.standard_normal(n)),
        "D": 0.01 * (common2 + 0.2 * rng.standard_normal(n)),
        "E": 0.01 * (common2 + 0.3 * rng.standard_normal(n)),
    }
    return pd.DataFrame(data)


@pytest.fixture
def simple_adj() -> np.ndarray:
    """Simple 4-node adjacency matrix (line graph: 0-1-2-3)."""
    adj = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=float)
    return adj


# ---------------------------------------------------------------------------
# Correlation network
# ---------------------------------------------------------------------------

class TestCorrelationNetwork:
    """Tests for correlation_network."""

    def test_returns_correct_keys(self, returns_df: pd.DataFrame) -> None:
        result = correlation_network(returns_df, threshold=0.3)
        assert "adjacency" in result
        assert "correlation" in result
        assert "asset_names" in result

    def test_adjacency_shape(self, returns_df: pd.DataFrame) -> None:
        result = correlation_network(returns_df, threshold=0.3)
        n = len(returns_df.columns)
        assert result["adjacency"].shape == (n, n)

    def test_no_self_loops(self, returns_df: pd.DataFrame) -> None:
        result = correlation_network(returns_df, threshold=0.0)
        np.testing.assert_array_equal(np.diag(result["adjacency"]), 0.0)

    def test_threshold_removes_weak_edges(self, returns_df: pd.DataFrame) -> None:
        """High threshold should produce fewer edges than low threshold."""
        result_low = correlation_network(returns_df, threshold=0.1)
        result_high = correlation_network(returns_df, threshold=0.8)
        assert result_high["adjacency"].sum() <= result_low["adjacency"].sum()

    def test_adjacency_is_symmetric(self, returns_df: pd.DataFrame) -> None:
        result = correlation_network(returns_df, threshold=0.3)
        np.testing.assert_array_equal(result["adjacency"], result["adjacency"].T)

    def test_asset_names_match(self, returns_df: pd.DataFrame) -> None:
        result = correlation_network(returns_df, threshold=0.3)
        assert result["asset_names"] == list(returns_df.columns)


# ---------------------------------------------------------------------------
# Minimum spanning tree
# ---------------------------------------------------------------------------

class TestMinimumSpanningTree:
    """Tests for minimum_spanning_tree."""

    def test_mst_has_n_minus_1_edges(self) -> None:
        """MST of n nodes has exactly n-1 edges."""
        rng = np.random.default_rng(7)
        n = 6
        corr = np.eye(n)
        # Create a positive definite correlation-like matrix
        A = rng.standard_normal((n, n))
        corr = A @ A.T
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)

        mst = minimum_spanning_tree(corr)
        # Each edge counted once in upper triangle
        n_edges = int(mst.sum()) // 2
        assert n_edges == n - 1

    def test_mst_is_symmetric(self) -> None:
        corr = np.array([
            [1.0, 0.8, 0.2],
            [0.8, 1.0, 0.5],
            [0.2, 0.5, 1.0],
        ])
        mst = minimum_spanning_tree(corr)
        np.testing.assert_array_equal(mst, mst.T)

    def test_mst_connects_all_nodes(self) -> None:
        """All nodes reachable in MST (connected)."""
        rng = np.random.default_rng(0)
        n = 5
        A = rng.standard_normal((n, n))
        corr = A @ A.T
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)

        mst = minimum_spanning_tree(corr)
        # BFS from node 0
        visited = set()
        queue = [0]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            neighbours = np.where(mst[node] > 0)[0]
            queue.extend(neighbours)
        assert len(visited) == n


# ---------------------------------------------------------------------------
# Centrality measures
# ---------------------------------------------------------------------------

class TestCentralityMeasures:
    """Tests for centrality_measures."""

    def test_returns_correct_keys(self, simple_adj: np.ndarray) -> None:
        result = centrality_measures(simple_adj)
        assert "degree" in result
        assert "betweenness" in result
        assert "eigenvector" in result

    def test_degree_centrality_values(self, simple_adj: np.ndarray) -> None:
        """In line graph 0-1-2-3, nodes 1,2 have degree 2/3, nodes 0,3 have 1/3."""
        result = centrality_measures(simple_adj)
        # n=4, normalised degree
        expected = np.array([1 / 3, 2 / 3, 2 / 3, 1 / 3])
        np.testing.assert_allclose(result["degree"], expected, atol=1e-10)

    def test_betweenness_centre_higher(self, simple_adj: np.ndarray) -> None:
        """Centre nodes should have higher betweenness."""
        result = centrality_measures(simple_adj)
        # Nodes 1 and 2 are more central
        assert result["betweenness"][1] >= result["betweenness"][0]
        assert result["betweenness"][2] >= result["betweenness"][3]

    def test_eigenvector_nonnegative(self, simple_adj: np.ndarray) -> None:
        result = centrality_measures(simple_adj)
        assert np.all(result["eigenvector"] >= -1e-10)


# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------

class TestCommunityDetection:
    """Tests for community_detection."""

    def test_returns_correct_number_of_labels(self) -> None:
        n = 6
        adj = np.zeros((n, n))
        # Two cliques
        adj[0, 1] = adj[1, 0] = 1
        adj[0, 2] = adj[2, 0] = 1
        adj[1, 2] = adj[2, 1] = 1
        adj[3, 4] = adj[4, 3] = 1
        adj[3, 5] = adj[5, 3] = 1
        adj[4, 5] = adj[5, 4] = 1

        labels = community_detection(adj, n_communities=2)
        assert len(labels) == n
        assert len(np.unique(labels)) == 2

    def test_detects_two_cliques(self) -> None:
        """Two disconnected cliques should be in different communities."""
        n = 6
        adj = np.zeros((n, n))
        adj[0, 1] = adj[1, 0] = 1
        adj[0, 2] = adj[2, 0] = 1
        adj[1, 2] = adj[2, 1] = 1
        adj[3, 4] = adj[4, 3] = 1
        adj[3, 5] = adj[5, 3] = 1
        adj[4, 5] = adj[5, 4] = 1

        labels = community_detection(adj, n_communities=2)
        # Nodes 0,1,2 should have same label; 3,4,5 same label (different)
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] == labels[4] == labels[5]
        assert labels[0] != labels[3]


# ---------------------------------------------------------------------------
# Systemic risk
# ---------------------------------------------------------------------------

class TestSystemicRiskScore:
    """Tests for systemic_risk_score."""

    def test_mes_returns_series(self, returns_df: pd.DataFrame) -> None:
        result = systemic_risk_score(returns_df, method="mes")
        assert isinstance(result, pd.Series)
        assert len(result) == len(returns_df.columns)

    def test_covar_returns_series(self, returns_df: pd.DataFrame) -> None:
        result = systemic_risk_score(returns_df, method="covar")
        assert isinstance(result, pd.Series)
        assert len(result) == len(returns_df.columns)

    def test_connectedness_returns_series(self, returns_df: pd.DataFrame) -> None:
        result = systemic_risk_score(returns_df, method="connectedness")
        assert isinstance(result, pd.Series)

    def test_unknown_method_raises(self, returns_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            systemic_risk_score(returns_df, method="unknown")

    def test_scores_are_finite(self, returns_df: pd.DataFrame) -> None:
        for method in ["mes", "covar", "connectedness"]:
            result = systemic_risk_score(returns_df, method=method)
            assert np.all(np.isfinite(result.values))


# ---------------------------------------------------------------------------
# Contagion simulation
# ---------------------------------------------------------------------------

class TestContagionSimulation:
    """Tests for contagion_simulation."""

    def test_shock_node_always_defaults(self, simple_adj: np.ndarray) -> None:
        result = contagion_simulation(simple_adj, shock_node=0, shock_magnitude=1.0)
        assert result["defaulted"][0]

    def test_cascade_size_gte_1(self, simple_adj: np.ndarray) -> None:
        result = contagion_simulation(simple_adj, shock_node=0, shock_magnitude=1.0)
        assert result["cascade_size"] >= 1

    def test_no_contagion_with_high_threshold(self) -> None:
        """With very high threshold, only shock node defaults."""
        adj = np.ones((3, 3))
        np.fill_diagonal(adj, 0)
        result = contagion_simulation(adj, shock_node=0, shock_magnitude=0.1, threshold=100.0)
        assert result["cascade_size"] == 1


# ---------------------------------------------------------------------------
# Granger network
# ---------------------------------------------------------------------------

class TestGrangerNetwork:
    """Tests for granger_network."""

    def test_returns_correct_keys(self, returns_df: pd.DataFrame) -> None:
        result = granger_network(returns_df, max_lag=2)
        assert "adjacency" in result
        assert "pvalues" in result
        assert "asset_names" in result

    def test_adjacency_shape(self, returns_df: pd.DataFrame) -> None:
        n = len(returns_df.columns)
        result = granger_network(returns_df, max_lag=2)
        assert result["adjacency"].shape == (n, n)

    def test_no_self_causation(self, returns_df: pd.DataFrame) -> None:
        result = granger_network(returns_df, max_lag=2)
        np.testing.assert_array_equal(np.diag(result["adjacency"]), 0.0)

    def test_pvalues_between_0_and_1(self, returns_df: pd.DataFrame) -> None:
        result = granger_network(returns_df, max_lag=2)
        # Off-diagonal
        mask = ~np.eye(len(returns_df.columns), dtype=bool)
        pvals = result["pvalues"][mask]
        assert np.all(pvals >= 0.0)
        assert np.all(pvals <= 1.0)
