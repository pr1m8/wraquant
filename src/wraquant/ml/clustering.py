"""Financial clustering methods.

Provides correlation-based asset clustering, market-regime detection, and
optimal-cluster-count selection.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from wraquant.core.decorators import requires_extra

__all__ = [
    "correlation_clustering",
    "regime_clustering",
    "optimal_clusters",
]


# ---------------------------------------------------------------------------
# Correlation-based clustering
# ---------------------------------------------------------------------------


def correlation_clustering(
    returns: pd.DataFrame,
    n_clusters: int | None = None,
    method: Literal["hierarchical", "spectral"] = "hierarchical",
) -> dict[str, Any]:
    """Cluster assets by their return correlations.

    Parameters
    ----------
    returns : pd.DataFrame
        T x N return matrix (rows = observations, columns = assets).
    n_clusters : int or None
        Number of clusters.  If ``None`` the optimal number is chosen
        automatically (silhouette score for hierarchical, or defaults to
        ``3`` for spectral).
    method : {'hierarchical', 'spectral'}
        Clustering algorithm.

    Returns
    -------
    dict
        ``labels``: np.ndarray of cluster assignments (length N),
        ``n_clusters``: int,
        ``linkage_matrix``: linkage matrix (hierarchical only, else
        ``None``).
    """
    corr = returns.corr().values
    n = corr.shape[0]

    if method == "hierarchical":
        # Distance = sqrt(0.5 * (1 - rho))
        dist_matrix = np.sqrt(0.5 * (1 - corr))
        np.fill_diagonal(dist_matrix, 0.0)
        # Ensure symmetry
        dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
        condensed = squareform(dist_matrix, checks=False)
        link = linkage(condensed, method="ward")

        if n_clusters is None:
            n_clusters = _optimal_k_from_linkage(dist_matrix, link, max_k=min(10, n))

        labels = fcluster(link, t=n_clusters, criterion="maxclust") - 1
        return {
            "labels": labels,
            "n_clusters": int(n_clusters),
            "linkage_matrix": link,
        }

    if method == "spectral":
        return _spectral_clustering(corr, n_clusters or 3)

    raise ValueError(f"Unknown method '{method}'; use 'hierarchical' or 'spectral'.")


def _optimal_k_from_linkage(
    dist_matrix: np.ndarray,
    link: np.ndarray,
    max_k: int = 10,
) -> int:
    """Find optimal number of clusters via silhouette score."""
    from scipy.spatial.distance import squareform as _squareform

    condensed = _squareform(dist_matrix, checks=False)
    best_k = 2
    best_score = -1.0

    for k in range(2, max_k + 1):
        labels = fcluster(link, t=k, criterion="maxclust")
        if len(set(labels)) < 2:
            continue
        score = _silhouette_score_simple(condensed, labels, dist_matrix.shape[0])
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def _silhouette_score_simple(
    condensed_dist: np.ndarray,
    labels: np.ndarray,
    n: int,
) -> float:
    """Simplified silhouette score from a condensed distance matrix."""
    full_dist = squareform(condensed_dist)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return -1.0

    sil = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        if same.sum() == 0:
            sil[i] = 0.0
            continue
        a_i = full_dist[i, same].mean()

        b_i = np.inf
        for lab in unique_labels:
            if lab == labels[i]:
                continue
            other = labels == lab
            if other.sum() == 0:
                continue
            b_i = min(b_i, full_dist[i, other].mean())

        sil[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0

    return float(sil.mean())


@requires_extra("ml")
def _spectral_clustering(
    corr: np.ndarray,
    n_clusters: int,
) -> dict[str, Any]:
    """Spectral clustering on the correlation matrix."""
    from sklearn.cluster import SpectralClustering

    # Shift correlation to [0, 1] for affinity
    affinity = (corr + 1.0) / 2.0
    np.fill_diagonal(affinity, 1.0)

    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=42,
        assign_labels="kmeans",
    )
    labels = sc.fit_predict(affinity)
    return {
        "labels": labels,
        "n_clusters": n_clusters,
        "linkage_matrix": None,
    }


# ---------------------------------------------------------------------------
# Regime clustering
# ---------------------------------------------------------------------------


@requires_extra("ml")
def regime_clustering(
    features: pd.DataFrame | np.ndarray,
    n_regimes: int = 2,
    method: Literal["gmm", "kmeans"] = "gmm",
) -> dict[str, Any]:
    """Cluster time periods into market regimes.

    Parameters
    ----------
    features : pd.DataFrame or np.ndarray
        Feature matrix where each row is a time observation.
    n_regimes : int
        Number of regimes to identify.
    method : {'gmm', 'kmeans'}
        Clustering algorithm.

    Returns
    -------
    dict
        ``labels``: np.ndarray of regime assignments,
        ``n_regimes``: int,
        ``model``: the fitted clustering model.
    """
    X = np.asarray(features)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if method == "gmm":
        from sklearn.mixture import GaussianMixture

        gm = GaussianMixture(n_components=n_regimes, random_state=42)
        labels = gm.fit_predict(X)
        return {"labels": labels, "n_regimes": n_regimes, "model": gm}

    if method == "kmeans":
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        return {"labels": labels, "n_regimes": n_regimes, "model": km}

    raise ValueError(f"Unknown method '{method}'; use 'gmm' or 'kmeans'.")


# ---------------------------------------------------------------------------
# Optimal cluster count
# ---------------------------------------------------------------------------


@requires_extra("ml")
def optimal_clusters(
    data: pd.DataFrame | np.ndarray,
    max_k: int = 10,
    method: Literal["silhouette", "bic"] = "silhouette",
) -> int:
    """Determine the optimal number of clusters.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Feature matrix.
    max_k : int
        Maximum number of clusters to evaluate.
    method : {'silhouette', 'bic'}
        Selection criterion.  ``'silhouette'`` uses the silhouette score
        with KMeans; ``'bic'`` uses the Bayesian Information Criterion
        with a Gaussian Mixture Model.

    Returns
    -------
    int
        Optimal number of clusters (between 2 and *max_k*).
    """
    X = np.asarray(data)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if method == "silhouette":
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        best_k = 2
        best_score = -1.0
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
        return best_k

    if method == "bic":
        from sklearn.mixture import GaussianMixture

        best_k = 2
        best_bic = np.inf
        for k in range(2, max_k + 1):
            gm = GaussianMixture(n_components=k, random_state=42)
            gm.fit(X)
            bic = gm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_k = k
        return best_k

    raise ValueError(f"Unknown method '{method}'; use 'silhouette' or 'bic'.")
