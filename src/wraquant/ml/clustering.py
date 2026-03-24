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

from wraquant.core._coerce import coerce_dataframe
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

    Use correlation clustering to group assets that move together,
    which is useful for portfolio diversification (allocate across
    clusters), risk management (monitor cluster concentration), and
    statistical arbitrage (trade within-cluster mean-reversion).

    The correlation-based distance is ``d(i,j) = sqrt(0.5 * (1 - rho_ij))``,
    which maps perfect correlation to distance 0 and perfect negative
    correlation to distance 1.

    Parameters
    ----------
    returns : pd.DataFrame
        T x N return matrix (rows = observations, columns = assets).
    n_clusters : int or None
        Number of clusters.  If ``None`` the optimal number is chosen
        automatically (silhouette score for hierarchical, or defaults to
        ``3`` for spectral).
    method : {'hierarchical', 'spectral'}
        Clustering algorithm.  Hierarchical uses Ward linkage and
        produces a dendrogram-compatible linkage matrix.  Spectral uses
        the correlation matrix as affinity and finds clusters via
        eigenvalue decomposition.

    Returns
    -------
    dict
        ``labels`` : np.ndarray
            Cluster assignment for each asset (0-indexed, length N).
            Assets with the same label belong to the same cluster.
        ``n_clusters`` : int
            Number of clusters found or specified.
        ``linkage_matrix`` : np.ndarray or None
            Linkage matrix (hierarchical only).  Pass to
            ``scipy.cluster.hierarchy.dendrogram`` for visualization.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> # 3 groups of correlated assets
    >>> factor = np.random.randn(252, 3)
    >>> returns = pd.DataFrame(
    ...     np.column_stack([factor[:, i % 3] + np.random.randn(252) * 0.5
    ...                      for i in range(9)]),
    ...     columns=[f'asset_{i}' for i in range(9)]
    ... )
    >>> result = correlation_clustering(returns, n_clusters=3)
    >>> result['n_clusters']
    3
    >>> len(result['labels']) == 9
    True

    See Also
    --------
    regime_clustering : Cluster time periods into regimes.
    optimal_clusters : Determine optimal cluster count.
    wraquant.ml.preprocessing.detoned_correlation : Remove market mode before clustering.
    """
    returns = coerce_dataframe(returns, name="returns")
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

    Use regime clustering when you want to identify distinct market
    states (e.g., bull/bear, risk-on/risk-off, high/low volatility)
    from observable features without a pre-defined model.  GMM is
    preferred because it assigns soft probabilities to each regime;
    KMeans provides hard assignments only.

    Parameters
    ----------
    features : pd.DataFrame or np.ndarray
        Feature matrix where each row is a time observation.  Common
        inputs include rolling volatility, returns, spreads, and VIX.
    n_regimes : int
        Number of regimes to identify (default 2, typical for
        risk-on/risk-off).
    method : {'gmm', 'kmeans'}
        Clustering algorithm.  ``'gmm'`` (Gaussian Mixture Model)
        provides probabilistic assignments; ``'kmeans'`` provides
        hard assignments and is faster.

    Returns
    -------
    dict
        ``labels`` : np.ndarray
            Regime assignment for each time period (0-indexed).
        ``n_regimes`` : int
            Number of regimes.
        ``model`` : object
            Fitted GaussianMixture or KMeans model.  For GMM, call
            ``model.predict_proba(X)`` to get regime probabilities.

    Example
    -------
    >>> import numpy as np, pandas as pd
    >>> np.random.seed(42)
    >>> vol = np.concatenate([np.random.randn(100) * 0.5 + 0.1,
    ...                       np.random.randn(100) * 0.5 + 0.3])
    >>> features = pd.DataFrame({'vol': vol, 'vol_sq': vol ** 2})
    >>> result = regime_clustering(features, n_regimes=2)
    >>> result['n_regimes']
    2
    >>> len(result['labels']) == 200
    True

    See Also
    --------
    correlation_clustering : Cluster assets (cross-sectional).
    optimal_clusters : Find the optimal number of clusters/regimes.
    wraquant.regimes : HMM and Markov-switching regime detection.
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

    Use this function before calling ``correlation_clustering`` or
    ``regime_clustering`` to select the number of clusters
    data-adaptively rather than guessing.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Feature matrix.
    max_k : int
        Maximum number of clusters to evaluate (default 10).
    method : {'silhouette', 'bic'}
        Selection criterion.  ``'silhouette'`` uses the silhouette score
        with KMeans (higher is better, range [-1, 1]); ``'bic'`` uses
        the Bayesian Information Criterion with a Gaussian Mixture Model
        (lower is better).  Silhouette is faster; BIC is more principled
        for probabilistic models.

    Returns
    -------
    int
        Optimal number of clusters (between 2 and *max_k*).
        Use this value as ``n_clusters`` in ``correlation_clustering``
        or ``n_regimes`` in ``regime_clustering``.

    Example
    -------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> # Generate data with 3 natural clusters
    >>> data = np.vstack([np.random.randn(50, 2) + [0, 0],
    ...                   np.random.randn(50, 2) + [5, 5],
    ...                   np.random.randn(50, 2) + [10, 0]])
    >>> k = optimal_clusters(data, max_k=6)
    >>> 2 <= k <= 6
    True

    See Also
    --------
    correlation_clustering : Cluster assets by correlation.
    regime_clustering : Cluster time periods into regimes.
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
