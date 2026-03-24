"""Financial network analysis.

Build and analyse networks from correlation structures, Granger causality,
and other dependence measures.  Includes systemic risk scoring and
contagion simulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import linalg as sp_linalg

from wraquant.core._coerce import coerce_array, coerce_dataframe

__all__ = [
    "correlation_network",
    "minimum_spanning_tree",
    "centrality_measures",
    "community_detection",
    "systemic_risk_score",
    "contagion_simulation",
    "granger_network",
]


# ---------------------------------------------------------------------------
# Network construction
# ---------------------------------------------------------------------------

def correlation_network(
    returns_df: pd.DataFrame,
    threshold: float = 0.3,
) -> dict[str, np.ndarray | list[str]]:
    """Build an adjacency matrix from pairwise correlations.

    Edges with absolute correlation below *threshold* are set to zero.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Asset returns, columns are assets.
    threshold : float, optional
        Minimum absolute correlation to retain an edge (default 0.3).

    Returns
    -------
    dict
        ``adjacency``   – (n, n) adjacency matrix (0/1).
        ``correlation`` – (n, n) full correlation matrix.
        ``asset_names`` – list of column names.
    """
    returns_df = coerce_dataframe(returns_df, name="returns_df")
    corr = returns_df.corr().values
    n = corr.shape[0]
    adj = (np.abs(corr) >= threshold).astype(float)
    # Remove self-loops
    np.fill_diagonal(adj, 0.0)

    return {
        "adjacency": adj,
        "correlation": corr,
        "asset_names": list(returns_df.columns),
    }


def minimum_spanning_tree(
    correlation_matrix: ArrayLike,
) -> np.ndarray:
    """Compute the minimum spanning tree of a correlation matrix.

    Uses the distance metric ``d = sqrt(2 * (1 - rho))`` and Prim's algorithm.

    Parameters
    ----------
    correlation_matrix : array_like
        (n, n) symmetric correlation matrix.

    Returns
    -------
    np.ndarray
        (n, n) adjacency matrix of the MST (symmetric, 0/1 entries).
    """
    corr = np.asarray(correlation_matrix, dtype=np.float64)
    n = corr.shape[0]

    # Distance from correlation
    dist = np.sqrt(np.clip(2.0 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)

    # Prim's algorithm
    in_tree = np.zeros(n, dtype=bool)
    mst_adj = np.zeros((n, n), dtype=float)
    min_cost = np.full(n, np.inf)
    min_edge = np.full(n, -1, dtype=int)

    # Start from node 0
    min_cost[0] = 0.0
    for _ in range(n):
        # Pick the cheapest node not yet in the tree
        candidates = np.where(~in_tree)[0]
        u = candidates[np.argmin(min_cost[candidates])]
        in_tree[u] = True

        if min_edge[u] >= 0:
            mst_adj[u, min_edge[u]] = 1.0
            mst_adj[min_edge[u], u] = 1.0

        # Update costs
        for v in range(n):
            if not in_tree[v] and dist[u, v] < min_cost[v]:
                min_cost[v] = dist[u, v]
                min_edge[v] = u

    return mst_adj


# ---------------------------------------------------------------------------
# Centrality
# ---------------------------------------------------------------------------

def centrality_measures(
    adjacency_matrix: ArrayLike,
    asset_names: list[str] | None = None,
) -> dict[str, np.ndarray | list[str]]:
    """Compute degree, betweenness, and eigenvector centrality.

    Parameters
    ----------
    adjacency_matrix : array_like
        (n, n) adjacency matrix (can be weighted; zeros = no edge).
    asset_names : list of str or None, optional
        Labels for the nodes; defaults to ``["0", "1", ...]``.

    Returns
    -------
    dict
        ``degree``       – degree centrality (normalised).
        ``betweenness``  – betweenness centrality.
        ``eigenvector``  – eigenvector centrality.
        ``asset_names``  – node labels.
    """
    adj = np.asarray(adjacency_matrix, dtype=np.float64)
    n = adj.shape[0]
    if asset_names is None:
        asset_names = [str(i) for i in range(n)]

    # --- Degree centrality (normalised) ---
    degree = adj.sum(axis=1) / max(n - 1, 1)

    # --- Betweenness centrality (Brandes-style BFS on unweighted) ---
    binary = (adj != 0).astype(float)
    betweenness = np.zeros(n)

    for s in range(n):
        # BFS from s
        stack: list[int] = []
        pred: list[list[int]] = [[] for _ in range(n)]
        sigma = np.zeros(n)
        sigma[s] = 1.0
        dist_bfs = np.full(n, -1, dtype=int)
        dist_bfs[s] = 0
        queue = [s]
        head = 0

        while head < len(queue):
            v = queue[head]
            head += 1
            stack.append(v)
            neighbours = np.where(binary[v] > 0)[0]
            for w in neighbours:
                if dist_bfs[w] < 0:
                    dist_bfs[w] = dist_bfs[v] + 1
                    queue.append(w)
                if dist_bfs[w] == dist_bfs[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        delta = np.zeros(n)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                c = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                delta[v] += c
            if w != s:
                betweenness[w] += delta[w]

    # Normalise betweenness
    norm_factor = max((n - 1) * (n - 2), 1)
    betweenness /= norm_factor

    # --- Eigenvector centrality (power iteration) ---
    eig_cent = _eigenvector_centrality(adj)

    return {
        "degree": degree,
        "betweenness": betweenness,
        "eigenvector": eig_cent,
        "asset_names": asset_names,
    }


def _eigenvector_centrality(
    adj: np.ndarray,
    max_iter: int = 300,
    tol: float = 1e-8,
) -> np.ndarray:
    """Power-iteration eigenvector centrality."""
    n = adj.shape[0]
    x = np.ones(n) / n
    for _ in range(max_iter):
        x_new = adj @ x
        norm = np.linalg.norm(x_new)
        if norm == 0:
            return np.zeros(n)
        x_new /= norm
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x_new


# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------

def community_detection(
    adjacency_matrix: ArrayLike,
    n_communities: int = 2,
) -> np.ndarray:
    """Detect communities via spectral clustering on the graph Laplacian.

    Parameters
    ----------
    adjacency_matrix : array_like
        (n, n) adjacency matrix.
    n_communities : int, optional
        Number of communities to detect (default 2).

    Returns
    -------
    np.ndarray
        Integer community labels of length *n*.
    """
    adj = np.asarray(adjacency_matrix, dtype=np.float64)
    n = adj.shape[0]
    n_communities = min(n_communities, n)

    # Unnormalised graph Laplacian  L = D - A
    degree_vec = adj.sum(axis=1)
    L = np.diag(degree_vec) - adj

    # Eigen-decomposition: smallest eigenvectors of L (skip first = all-ones)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    # Take eigenvectors 1..n_communities (skip the trivial zero eigenvector)
    V = eigenvectors[:, 1:n_communities]

    # Row-normalise
    row_norms = np.linalg.norm(V, axis=1, keepdims=True)
    row_norms = np.where(row_norms > 0, row_norms, 1.0)
    V = V / row_norms

    # k-means++ clustering
    labels = _kmeans(V, n_communities, max_iter=300, seed=42)
    return labels


def _kmeans(
    X: np.ndarray,
    k: int,
    max_iter: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """Minimal k-means++ for spectral clustering."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    # k-means++ initialization
    centroids = np.empty((k, X.shape[1]))
    centroids[0] = X[rng.integers(n)]
    for c in range(1, k):
        dists = np.min(
            np.linalg.norm(X[:, None, :] - centroids[None, :c, :], axis=2), axis=1
        )
        probs = dists**2
        total = probs.sum()
        probs = probs / total if total > 0 else np.ones(n) / n
        centroids[c] = X[rng.choice(n, p=probs)]

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # Assign
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update centroids
        for c in range(k):
            members = X[labels == c]
            if len(members) > 0:
                centroids[c] = members.mean(axis=0)

    return labels


# ---------------------------------------------------------------------------
# Systemic risk
# ---------------------------------------------------------------------------

def systemic_risk_score(
    returns_df: pd.DataFrame,
    method: str = "mes",
    quantile: float = 0.05,
) -> pd.Series:
    """Compute a systemic importance score for each asset.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Asset returns, columns are assets.
    method : {'mes', 'covar', 'connectedness'}, optional
        Scoring method:
        - ``'mes'`` – Marginal Expected Shortfall: expected loss of asset
          conditional on market being in its lower *quantile* tail.
        - ``'covar'`` – Delta CoVaR: change in system VaR conditional on
          asset being at its VaR vs median.
        - ``'connectedness'`` – Diebold-Yilmaz–style total connectedness
          from variance decomposition (simplified).
    quantile : float, optional
        Tail quantile for MES / CoVaR (default 0.05).

    Returns
    -------
    pd.Series
        Systemic risk score for each asset (higher = more systemic).

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    returns_df = coerce_dataframe(returns_df, name="returns_df")
    if method == "mes":
        return _marginal_expected_shortfall(returns_df, quantile)
    elif method == "covar":
        return _delta_covar(returns_df, quantile)
    elif method == "connectedness":
        return _connectedness_score(returns_df)
    else:
        raise ValueError(
            f"Unknown method {method!r}; use 'mes', 'covar', or 'connectedness'."
        )


def _marginal_expected_shortfall(
    returns_df: pd.DataFrame,
    quantile: float,
) -> pd.Series:
    """MES: expected asset return when the market is in the tail."""
    market = returns_df.mean(axis=1)
    threshold = np.quantile(market.values, quantile)
    tail_mask = market <= threshold
    mes = returns_df[tail_mask].mean()
    # More negative MES = higher systemic risk, so negate for scoring
    return -mes


def _delta_covar(
    returns_df: pd.DataFrame,
    quantile: float,
) -> pd.Series:
    """Simplified Delta CoVaR."""
    market = returns_df.mean(axis=1).values
    scores = {}
    for col in returns_df.columns:
        asset = returns_df[col].values
        # VaR of market conditional on asset being at its VaR
        asset_var = np.quantile(asset, quantile)
        stress_mask = asset <= asset_var
        median_mask = (asset >= np.quantile(asset, 0.45)) & (
            asset <= np.quantile(asset, 0.55)
        )
        if stress_mask.sum() > 0 and median_mask.sum() > 0:
            covar_stress = np.quantile(market[stress_mask], quantile)
            covar_median = np.quantile(market[median_mask], quantile)
            scores[col] = abs(covar_stress - covar_median)
        else:
            scores[col] = 0.0
    return pd.Series(scores)


def _connectedness_score(returns_df: pd.DataFrame) -> pd.Series:
    """Simplified Diebold-Yilmaz connectedness based on correlation."""
    corr = returns_df.corr().values
    n = corr.shape[0]
    # Total connectedness: sum of off-diagonal squared correlations per asset
    sq = corr ** 2
    np.fill_diagonal(sq, 0.0)
    scores = sq.sum(axis=1) / max(n - 1, 1)
    return pd.Series(scores, index=returns_df.columns)


# ---------------------------------------------------------------------------
# Contagion simulation
# ---------------------------------------------------------------------------

def contagion_simulation(
    adjacency_matrix: ArrayLike,
    shock_node: int,
    shock_magnitude: float = 1.0,
    threshold: float = 0.5,
    max_rounds: int = 100,
) -> dict[str, np.ndarray | int]:
    """Simulate shock propagation through a financial network.

    At each round, a node that receives total stress above *threshold*
    "defaults" and propagates the shock to its neighbours weighted by
    the adjacency matrix.

    Parameters
    ----------
    adjacency_matrix : array_like
        (n, n) weighted adjacency matrix.
    shock_node : int
        Index of the initially shocked node.
    shock_magnitude : float, optional
        Size of the initial shock (default 1.0).
    threshold : float, optional
        Stress threshold for contagion (default 0.5).
    max_rounds : int, optional
        Maximum propagation rounds (default 100).

    Returns
    -------
    dict
        ``defaulted``     – boolean array of which nodes defaulted.
        ``stress``        – cumulative stress on each node.
        ``rounds``        – number of propagation rounds.
        ``cascade_size``  – total number of defaults.
    """
    adj = np.asarray(adjacency_matrix, dtype=np.float64)
    n = adj.shape[0]

    stress = np.zeros(n)
    defaulted = np.zeros(n, dtype=bool)

    # Initial shock
    stress[shock_node] = shock_magnitude
    defaulted[shock_node] = True

    rounds = 0
    for _ in range(max_rounds):
        # Find newly defaulted nodes (stress >= threshold, not already defaulted)
        new_defaults = (~defaulted) & (stress >= threshold)
        if not new_defaults.any():
            break
        rounds += 1
        defaulted |= new_defaults
        # Propagate shock from newly defaulted nodes
        for node in np.where(new_defaults)[0]:
            stress += adj[node] * stress[node]

    return {
        "defaulted": defaulted,
        "stress": stress,
        "rounds": rounds,
        "cascade_size": int(defaulted.sum()),
    }


# ---------------------------------------------------------------------------
# Granger causality network
# ---------------------------------------------------------------------------

def granger_network(
    returns_df: pd.DataFrame,
    max_lag: int = 5,
    significance: float = 0.05,
) -> dict[str, np.ndarray | list[str]]:
    """Build a directed network from pairwise Granger causality tests.

    For each pair (i, j) test whether lagged values of *j* help predict *i*
    beyond *i*'s own lags, using an F-test.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Asset returns, columns are assets.
    max_lag : int, optional
        Maximum lag order (default 5).
    significance : float, optional
        p-value threshold for a significant Granger-causal link (default 0.05).

    Returns
    -------
    dict
        ``adjacency`` – (n, n) directed adjacency matrix; entry (i, j) = 1
                        means *j* Granger-causes *i*.
        ``pvalues``   – (n, n) matrix of p-values.
        ``asset_names`` – column names.
    """
    from scipy.stats import f as f_dist

    returns_df = coerce_dataframe(returns_df, name="returns_df")
    cols = list(returns_df.columns)
    n = len(cols)
    data = returns_df.values
    T = data.shape[0]

    pvalues = np.ones((n, n))
    adjacency = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pval = _granger_f_test(data[:, i], data[:, j], max_lag, T)
            pvalues[i, j] = pval
            if pval < significance:
                adjacency[i, j] = 1.0

    return {
        "adjacency": adjacency,
        "pvalues": pvalues,
        "asset_names": cols,
    }


def _granger_f_test(
    y: np.ndarray,
    x: np.ndarray,
    max_lag: int,
    T: int,
) -> float:
    """F-test for Granger causality: does *x* Granger-cause *y*?"""
    from scipy.stats import f as f_dist

    # Build lagged matrices
    n_obs = T - max_lag
    if n_obs <= 2 * max_lag + 1:
        return 1.0  # Not enough data

    Y = y[max_lag:]

    # Restricted model: y ~ y_lags
    X_r = np.column_stack(
        [y[max_lag - lag: T - lag] for lag in range(1, max_lag + 1)]
    )
    X_r = np.column_stack([np.ones(n_obs), X_r])

    # Unrestricted model: y ~ y_lags + x_lags
    X_u = np.column_stack(
        [y[max_lag - lag: T - lag] for lag in range(1, max_lag + 1)]
        + [x[max_lag - lag: T - lag] for lag in range(1, max_lag + 1)]
    )
    X_u = np.column_stack([np.ones(n_obs), X_u])

    # OLS residuals
    try:
        beta_r = np.linalg.lstsq(X_r, Y, rcond=None)[0]
        resid_r = Y - X_r @ beta_r
        ssr_r = resid_r @ resid_r

        beta_u = np.linalg.lstsq(X_u, Y, rcond=None)[0]
        resid_u = Y - X_u @ beta_u
        ssr_u = resid_u @ resid_u
    except np.linalg.LinAlgError:
        return 1.0

    df1 = max_lag
    df2 = n_obs - X_u.shape[1]
    if df2 <= 0 or ssr_u <= 0:
        return 1.0

    f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
    pval = 1.0 - f_dist.cdf(f_stat, df1, df2)
    return float(pval)
