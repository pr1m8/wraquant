"""Financial data preprocessing utilities.

Implements purged cross-validation, fractional differentiation, and
random-matrix-theory denoising -- all central to the *Advances in Financial
Machine Learning* workflow (Lopez de Prado).
"""

from __future__ import annotations

from itertools import combinations
from typing import Generator

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

__all__ = [
    "purged_kfold",
    "combinatorial_purged_kfold",
    "fractional_differentiation",
    "denoised_correlation",
    "detoned_correlation",
]


# ---------------------------------------------------------------------------
# Purged K-Fold cross-validation
# ---------------------------------------------------------------------------


def purged_kfold(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Purged K-Fold cross-validation.

    Ensures that training observations that immediately follow a test
    observation are removed (embargo) so that information cannot leak
    through overlapping labels.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix (only its length is used).
    y : pd.Series or np.ndarray
        Target vector (only its length is used).
    n_splits : int
        Number of folds.
    embargo_pct : float
        Fraction of total samples to embargo after each test fold.

    Yields
    ------
    tuple[np.ndarray, np.ndarray]
        ``(train_indices, test_indices)`` for each fold.
    """
    n_samples = len(X)
    embargo_size = int(n_samples * embargo_pct)
    indices = np.arange(n_samples)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    current = 0
    for fold_size in fold_sizes:
        test_start = current
        test_end = current + fold_size
        test_idx = indices[test_start:test_end]

        # Embargo zone immediately after the test set
        embargo_end = min(test_end + embargo_size, n_samples)

        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[test_start:embargo_end] = False
        train_idx = indices[train_mask]

        yield train_idx, test_idx
        current = test_end


def combinatorial_purged_kfold(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    n_splits: int = 5,
    n_test_splits: int = 2,
    embargo_pct: float = 0.01,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Combinatorial purged K-Fold cross-validation.

    Generates all C(n_splits, n_test_splits) train/test combinations,
    applying an embargo after each test group to prevent leakage.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : pd.Series or np.ndarray
        Target vector.
    n_splits : int
        Total number of groups.
    n_test_splits : int
        Number of groups held out for testing in each split.
    embargo_pct : float
        Fraction of total samples to embargo after each test group.

    Yields
    ------
    tuple[np.ndarray, np.ndarray]
        ``(train_indices, test_indices)`` for each combination.
    """
    n_samples = len(X)
    embargo_size = int(n_samples * embargo_pct)
    indices = np.arange(n_samples)

    # Build group boundaries
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    group_bounds: list[tuple[int, int]] = []
    current = 0
    for fs in fold_sizes:
        group_bounds.append((current, current + fs))
        current += fs

    for test_groups in combinations(range(n_splits), n_test_splits):
        test_mask = np.zeros(n_samples, dtype=bool)
        embargo_mask = np.zeros(n_samples, dtype=bool)

        for g in test_groups:
            start, end = group_bounds[g]
            test_mask[start:end] = True
            embargo_end = min(end + embargo_size, n_samples)
            embargo_mask[end:embargo_end] = True

        train_mask = ~(test_mask | embargo_mask)
        yield indices[train_mask], indices[test_mask]


# ---------------------------------------------------------------------------
# Fractional differentiation
# ---------------------------------------------------------------------------


def _frac_diff_weights(d: float, threshold: float) -> np.ndarray:
    """Compute the weights for fractional differentiation.

    Weights are generated until they fall below *threshold*.
    """
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    return np.array(weights[::-1], dtype=np.float64)


def fractional_differentiation(
    series: pd.Series,
    d: float = 0.5,
    threshold: float = 1e-5,
) -> pd.Series:
    """Fractionally differentiate a time series.

    Applies the fractional differentiation operator of order *d*
    (Hosking, 1981) to obtain a (near-)stationary series while
    preserving long-range memory.

    Parameters
    ----------
    series : pd.Series
        Input time series.
    d : float
        Fractional differentiation order (0 < d < 1 for partial
        differentiation; d = 1 is the standard first difference).
    threshold : float
        Minimum absolute weight to retain.

    Returns
    -------
    pd.Series
        Fractionally differentiated series (initial rows where the full
        convolution is not available are dropped).
    """
    weights = _frac_diff_weights(d, threshold)
    width = len(weights)
    values = series.values.astype(float)

    result = np.full(len(values), np.nan)
    for i in range(width - 1, len(values)):
        result[i] = np.dot(weights, values[i - width + 1 : i + 1])

    out = pd.Series(result, index=series.index, name=series.name)
    return out.dropna()


# ---------------------------------------------------------------------------
# Random Matrix Theory denoising
# ---------------------------------------------------------------------------


def _marchenko_pastur_pdf(
    x: np.ndarray,
    q: float,
    sigma2: float = 1.0,
) -> np.ndarray:
    """Marchenko-Pastur probability density function."""
    lambda_plus = sigma2 * (1 + np.sqrt(1 / q)) ** 2
    lambda_minus = sigma2 * (1 - np.sqrt(1 / q)) ** 2
    pdf = np.zeros_like(x)
    mask = (x >= lambda_minus) & (x <= lambda_plus)
    pdf[mask] = (
        q
        / (2 * np.pi * sigma2)
        * np.sqrt((lambda_plus - x[mask]) * (x[mask] - lambda_minus))
        / x[mask]
    )
    return pdf


def _fit_mp_sigma(eigenvalues: np.ndarray, q: float) -> float:
    """Fit sigma^2 to the bulk of eigenvalues via KS-like minimisation."""

    def _neg_loglik(sigma2: float) -> float:
        if sigma2 <= 0:
            return 1e12
        lambda_plus = sigma2 * (1 + np.sqrt(1 / q)) ** 2
        # Count eigenvalues below lambda+
        n_below = int(np.sum(eigenvalues <= lambda_plus))
        expected = int(
            len(eigenvalues) * min(1.0, 1.0 / q) if q >= 1 else len(eigenvalues)
        )
        return float((n_below - expected) ** 2)

    res = minimize_scalar(_neg_loglik, bounds=(0.01, 5.0), method="bounded")
    return float(res.x)


def denoised_correlation(
    returns: pd.DataFrame,
    n_components: int | None = None,
) -> np.ndarray:
    """Denoise a correlation matrix using Random Matrix Theory.

    Eigenvalues that fall within the Marchenko-Pastur distribution are
    replaced by their average, shrinking noise while preserving signal.

    Parameters
    ----------
    returns : pd.DataFrame
        T x N return matrix (rows = observations, columns = assets).
    n_components : int or None
        Number of signal eigenvalues to keep.  If ``None``, they are
        determined automatically from the Marchenko-Pastur bound.

    Returns
    -------
    np.ndarray
        Denoised correlation matrix of shape ``(N, N)``.
    """
    corr = np.array(returns.corr())
    t, n = returns.shape
    q = t / n

    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    if n_components is None:
        sigma2 = _fit_mp_sigma(eigenvalues, q)
        lambda_plus = sigma2 * (1 + np.sqrt(1 / q)) ** 2
        n_components = int(np.sum(eigenvalues > lambda_plus))
        n_components = max(n_components, 1)

    # Replace noise eigenvalues with their mean
    noise_eigenvalues = eigenvalues[n_components:]
    if len(noise_eigenvalues) > 0:
        noise_mean = noise_eigenvalues.mean()
        eigenvalues[n_components:] = noise_mean

    # Reconstruct
    diag = np.diag(eigenvalues)
    corr_denoised = eigenvectors @ diag @ eigenvectors.T

    # Rescale to unit diagonal
    d = np.sqrt(np.diag(corr_denoised))
    d[d == 0] = 1.0
    corr_denoised = corr_denoised / np.outer(d, d)
    np.fill_diagonal(corr_denoised, 1.0)

    return corr_denoised


def detoned_correlation(
    corr: np.ndarray,
    n_components: int = 1,
) -> np.ndarray:
    """Remove the first *n_components* eigenvectors (market mode) from a
    correlation matrix.

    This is useful for removing the dominant market factor from a
    correlation matrix so that residual clustering structure becomes
    visible.

    Parameters
    ----------
    corr : np.ndarray
        Correlation matrix of shape ``(N, N)``.
    n_components : int
        Number of leading eigenvalues/vectors to remove.

    Returns
    -------
    np.ndarray
        De-toned correlation matrix.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Zero-out the leading components
    eigenvalues[:n_components] = 0.0

    corr_detoned = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # Rescale to unit diagonal
    d = np.sqrt(np.diag(corr_detoned))
    d[d == 0] = 1.0
    corr_detoned = corr_detoned / np.outer(d, d)
    np.fill_diagonal(corr_detoned, 1.0)

    return corr_detoned
