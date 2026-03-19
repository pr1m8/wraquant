"""Cross-validation methods for financial time series.

Provides splitting strategies that respect the temporal ordering of financial
data and avoid lookahead bias.  Each function returns a list of
``(train_indices, test_indices)`` tuples suitable for slicing pandas
Series/DataFrames or numpy arrays.

The key methods:

- **Walk-forward** (expanding window): mimics live deployment where the
  training history grows over time.
- **Rolling**: fixed-length training window slides forward.
- **Purged K-fold**: K-fold with a gap (embargo) between train and test to
  prevent information leakage from overlapping labels.
- **Combinatorial purged**: generates many backtest paths from purged
  K-fold splits (Lopez de Prado, *Advances in Financial Machine Learning*).

References:
    - Lopez de Prado (2018). *Advances in Financial Machine Learning*.
      Chapter 7: Cross-Validation in Finance.
    - Bailey et al. (2017). "The Probability of Backtest Overfitting."
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np


def walk_forward_splits(
    n_samples: int,
    n_splits: int = 5,
    min_train_pct: float = 0.5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding window walk-forward splits.

    The training set always starts at index 0 and grows with each fold.
    The test set is a fixed-size window immediately following the training
    set.  This mimics live deployment where you retrain on all available
    history.

    When to use:
        The default for strategy research.  Walk-forward is the most
        realistic CV method for trading strategies because it mirrors
        how you would actually deploy: train on everything up to today,
        trade tomorrow, repeat.

    Parameters:
        n_samples: Total number of observations in the dataset.
        n_splits: Number of train/test splits to generate.
        min_train_pct: Minimum fraction of data used for the first
            training window.  Must be in (0, 1).  Default 0.5 means
            the first fold uses at least 50% of the data for training.

    Returns:
        List of (train_indices, test_indices) tuples.  Training sets
        expand over time; test sets are non-overlapping consecutive
        blocks.

    Example:
        >>> splits = walk_forward_splits(1000, n_splits=5, min_train_pct=0.5)
        >>> len(splits)
        5
        >>> all(s[0][0] == 0 for s in splits)  # train always starts at 0
        True
        >>> len(splits[0][0]) < len(splits[-1][0])  # train grows
        True

    See Also:
        rolling_splits: Fixed-size training window.
        purged_kfold_splits: K-fold with embargo for overlapping labels.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    if not 0.0 < min_train_pct < 1.0:
        raise ValueError(f"min_train_pct must be in (0, 1), got {min_train_pct}")

    min_train_size = int(n_samples * min_train_pct)
    remaining = n_samples - min_train_size
    test_size = max(remaining // n_splits, 1)

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        train_end = min_train_size + i * test_size
        test_start = train_end
        test_end = min(test_start + test_size, n_samples)
        if test_start >= n_samples:
            break
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))

    return splits


def rolling_splits(
    n_samples: int,
    n_splits: int = 5,
    window_pct: float = 0.6,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fixed-size rolling window splits.

    Both the training and test windows are fixed in size and slide
    forward through the data.  Unlike walk-forward, the training set
    does not grow -- this avoids giving extra weight to early data and
    is useful when regime changes make old data less relevant.

    When to use:
        Prefer rolling over walk-forward when you suspect the data
        generating process changes over time (regime shifts) and older
        data may hurt rather than help.

    Parameters:
        n_samples: Total number of observations.
        n_splits: Number of splits to generate.
        window_pct: Fraction of data used for each training window.
            Must be in (0, 1).  Default 0.6.

    Returns:
        List of (train_indices, test_indices) tuples.

    Example:
        >>> splits = rolling_splits(1000, n_splits=5, window_pct=0.6)
        >>> len(splits)
        5
        >>> all(len(s[0]) == len(splits[0][0]) for s in splits)  # fixed size
        True
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    if not 0.0 < window_pct < 1.0:
        raise ValueError(f"window_pct must be in (0, 1), got {window_pct}")

    train_size = int(n_samples * window_pct)
    remaining = n_samples - train_size
    test_size = max(remaining // n_splits, 1)
    step_size = max((n_samples - train_size - test_size) // max(n_splits - 1, 1), 1)

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        train_start = i * step_size
        train_end = train_start + train_size
        test_start = train_end
        test_end = min(test_start + test_size, n_samples)
        if test_end > n_samples or test_start >= n_samples:
            break
        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))

    return splits


def purged_kfold_splits(
    n_samples: int,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Purged K-fold cross-validation with embargo.

    Standard K-fold violates temporal ordering and leaks information
    when labels span multiple time steps (e.g., forward-looking returns).
    Purged K-fold adds an embargo gap between the training and test sets
    to eliminate this leakage.

    The data is split into ``n_splits`` contiguous groups.  For each fold,
    one group is the test set and the remaining groups form the training
    set, with observations within ``embargo_pct * n_samples`` of the
    test boundaries removed from training.

    When to use:
        Use purged K-fold when you have overlapping labels (e.g., returns
        computed over rolling windows) and want to evaluate on multiple
        segments without temporal ordering constraints.

    Parameters:
        n_samples: Total number of observations.
        n_splits: Number of folds.  Default 5.
        embargo_pct: Fraction of samples to exclude around test
            boundaries as an information barrier.  Default 0.01 (1%).

    Returns:
        List of (train_indices, test_indices) tuples.

    References:
        Lopez de Prado (2018), Chapter 7.

    Example:
        >>> splits = purged_kfold_splits(1000, n_splits=5, embargo_pct=0.01)
        >>> len(splits)
        5
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2 for K-fold, got {n_splits}")

    embargo_size = max(int(n_samples * embargo_pct), 0)
    fold_size = n_samples // n_splits
    indices = np.arange(n_samples)

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_splits - 1 else n_samples
        test_idx = indices[test_start:test_end]

        # Purge: remove embargo zone around test boundaries from train
        purge_start = max(test_start - embargo_size, 0)
        purge_end = min(test_end + embargo_size, n_samples)

        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[purge_start:purge_end] = False
        train_idx = indices[train_mask]

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits


def combinatorial_purged_splits(
    n_samples: int,
    n_splits: int = 5,
    n_test_groups: int = 2,
    embargo_pct: float = 0.01,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Combinatorial purged cross-validation (CPCV).

    Generates all C(n_splits, n_test_groups) combinations of contiguous
    groups as test sets, with purging and embargo applied.  This yields
    many more backtest paths than standard K-fold and enables
    statistical testing of the probability of backtest overfitting.

    When to use:
        Use CPCV when you need a large number of independent backtest
        paths to compute the Probability of Backtest Overfitting (PBO)
        or when you want the most thorough cross-validation at the cost
        of compute time.

    Parameters:
        n_samples: Total number of observations.
        n_splits: Number of contiguous groups to partition data into.
        n_test_groups: Number of groups to combine as the test set
            for each split.
        embargo_pct: Fraction of samples to exclude around test
            boundaries.

    Returns:
        List of (train_indices, test_indices) tuples.  Length is
        C(n_splits, n_test_groups).

    References:
        Lopez de Prado (2018), Chapter 12: "Backtesting on Synthetic Data."

    Example:
        >>> splits = combinatorial_purged_splits(1000, n_splits=6, n_test_groups=2)
        >>> len(splits)  # C(6, 2) = 15
        15
    """
    if n_test_groups >= n_splits:
        raise ValueError(
            f"n_test_groups ({n_test_groups}) must be < n_splits ({n_splits})"
        )

    embargo_size = max(int(n_samples * embargo_pct), 0)
    fold_size = n_samples // n_splits
    indices = np.arange(n_samples)

    # Build group boundaries
    groups: list[tuple[int, int]] = []
    for i in range(n_splits):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_splits - 1 else n_samples
        groups.append((start, end))

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for test_combo in combinations(range(n_splits), n_test_groups):
        # Build test indices
        test_ranges: list[tuple[int, int]] = [groups[g] for g in test_combo]
        test_idx_list: list[np.ndarray] = []
        purge_mask = np.zeros(n_samples, dtype=bool)

        for start, end in test_ranges:
            test_idx_list.append(indices[start:end])
            purge_start = max(start - embargo_size, 0)
            purge_end = min(end + embargo_size, n_samples)
            purge_mask[purge_start:purge_end] = True

        test_idx = np.concatenate(test_idx_list)

        # Training: everything not in purge zone
        train_mask = ~purge_mask
        train_idx = indices[train_mask]

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits


__all__ = [
    "combinatorial_purged_splits",
    "purged_kfold_splits",
    "rolling_splits",
    "walk_forward_splits",
]
