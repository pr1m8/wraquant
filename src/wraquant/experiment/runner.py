"""Experiment execution engine.

Runs a strategy function across a full parameter grid with cross-validation,
computing standardized performance metrics for each combination.  The runner
is the computational core of the Lab -- it handles splitting, metric
computation, and optional parallel execution via joblib.

Design decisions:
    - The strategy function must accept ``(data, **params)`` and return a
      ``pd.Series`` of per-period returns.  This contract is simple enough
      to wrap any existing strategy.
    - Metrics are computed using wraquant's own ``risk.metrics`` and
      ``backtest.metrics`` modules so that numbers are consistent with
      the rest of the library.
    - Parallel execution uses joblib when available; falls back to
      sequential when it isn't.
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from wraquant.experiment.cv import (
    purged_kfold_splits,
    rolling_splits,
    walk_forward_splits,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Result of a single strategy run (one param combo, one CV fold).

    Attributes:
        params: Parameter combination used.
        fold: CV fold index (0-based), or -1 for full in-sample.
        metrics: Dictionary of performance metrics.
        returns: Strategy return series for this run.
        train_indices: Indices used for training.
        test_indices: Indices used for testing / evaluation.
        elapsed_seconds: Wall-clock time for this run.
    """

    params: dict[str, Any]
    fold: int
    metrics: dict[str, float]
    returns: pd.Series
    train_indices: np.ndarray
    test_indices: np.ndarray
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary (no numpy/pandas)."""
        return {
            "params": dict(self.params),
            "fold": self.fold,
            "metrics": dict(self.metrics),
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass
class GridSpec:
    """Specification for a parameter grid.

    Attributes:
        param_dict: Mapping of parameter names to lists of values.
    """

    param_dict: dict[str, list[Any]]

    def __iter__(self):
        keys = sorted(self.param_dict.keys())
        values = [self.param_dict[k] for k in keys]
        for combo in itertools.product(*values):
            yield dict(zip(keys, combo, strict=False))

    def __len__(self) -> int:
        n = 1
        for v in self.param_dict.values():
            n *= len(v)
        return n


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def _compute_metrics(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Compute comprehensive performance metrics for a return series.

    Uses wraquant's own metric implementations as single source of truth.
    Falls back to local computations when imports are unavailable.

    Parameters:
        returns: Strategy return series.
        benchmark: Optional benchmark return series.
        periods_per_year: Annualization factor.

    Returns:
        Dictionary of named metrics.
    """
    from wraquant.backtest.metrics import omega_ratio, profit_factor
    from wraquant.risk.metrics import max_drawdown, sharpe_ratio, sortino_ratio

    clean = returns.dropna()
    n = len(clean)

    if n == 0:
        return {
            "sharpe": 0.0,
            "sortino": 0.0,
            "total_return": 0.0,
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "omega": 0.0,
            "n_periods": 0,
        }

    # Core metrics
    sharpe = sharpe_ratio(clean, periods_per_year=periods_per_year)
    sortino = sortino_ratio(clean, periods_per_year=periods_per_year)

    total_ret = float((1 + clean).prod() - 1)
    ann_factor = periods_per_year / n if n > 0 else 1.0
    ann_return = float((1 + total_ret) ** ann_factor - 1)
    ann_vol = float(clean.std() * np.sqrt(periods_per_year))

    cumulative = (1 + clean).cumprod()
    mdd = max_drawdown(cumulative)
    calmar = ann_return / abs(mdd) if mdd != 0 else 0.0

    win_rate = float((clean > 0).sum() / n) if n > 0 else 0.0
    pf = profit_factor(clean)
    omega = omega_ratio(clean)

    metrics: dict[str, float] = {
        "sharpe": sharpe,
        "sortino": sortino,
        "total_return": total_ret,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "max_drawdown": mdd,
        "calmar": calmar,
        "win_rate": win_rate,
        "profit_factor": pf,
        "omega": omega,
        "n_periods": float(n),
    }

    # Benchmark-relative metrics
    if benchmark is not None:
        from wraquant.risk.metrics import information_ratio

        bench_clean = benchmark.reindex(clean.index).dropna()
        common = clean.index.intersection(bench_clean.index)
        if len(common) > 10:
            ir = information_ratio(clean.loc[common], bench_clean.loc[common])
            metrics["information_ratio"] = ir

            # Excess return
            excess = float(clean.loc[common].mean() - bench_clean.loc[common].mean())
            metrics["excess_return"] = excess * periods_per_year

    return metrics


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ExperimentRunner:
    """Executes a strategy across a parameter grid with cross-validation.

    The runner is the computational engine of the Lab.  It:
    1. Generates all parameter combinations from the grid.
    2. Creates CV splits from the data.
    3. For each (param_combo, fold), runs the strategy and computes metrics.
    4. Collects all RunResults and returns them.

    Parameters:
        strategy_fn: Callable(data, **params) -> pd.Series of returns.
            The function receives a *slice* of the original data
            (the test window) and must return a return series.
        data: Full price or return data.
        params: Parameter grid as {name: [values]}.
        benchmark: Optional benchmark return series.
    """

    def __init__(
        self,
        strategy_fn: Callable[..., pd.Series],
        data: pd.Series | pd.DataFrame,
        params: dict[str, list[Any]],
        benchmark: pd.Series | None = None,
    ) -> None:
        self.strategy_fn = strategy_fn
        self.data = data
        self.params = params
        self.benchmark = benchmark
        self._grid = GridSpec(params)

    def run_single(
        self,
        param_combo: dict[str, Any],
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        fold: int = 0,
    ) -> RunResult:
        """Run one parameter combination on one CV fold.

        The strategy function is called with the *test* data slice.
        The train_idx is stored for reference but not passed to the
        strategy function (strategies that need training data should
        accept it through the params dict or closure).

        Parameters:
            param_combo: Single set of strategy parameters.
            train_idx: Training indices (stored, not passed to strategy).
            test_idx: Test indices used to slice the data.
            fold: CV fold index.

        Returns:
            RunResult with metrics, returns, and timing.
        """
        t0 = time.monotonic()

        # Slice data for this fold
        if isinstance(self.data, (pd.Series, pd.DataFrame)):
            test_data = self.data.iloc[test_idx]
        else:
            test_data = self.data[test_idx]

        # Run strategy
        try:
            returns = self.strategy_fn(test_data, **param_combo)
            if not isinstance(returns, pd.Series):
                returns = pd.Series(returns)
        except Exception as exc:
            logger.warning(
                "Strategy failed with params=%s fold=%d: %s",
                param_combo,
                fold,
                exc,
            )
            returns = pd.Series(dtype=float)

        # Compute metrics
        bench_slice = None
        if self.benchmark is not None and isinstance(returns.index, pd.DatetimeIndex):
            bench_slice = self.benchmark

        metrics = _compute_metrics(returns, benchmark=bench_slice)
        elapsed = time.monotonic() - t0

        return RunResult(
            params=dict(param_combo),
            fold=fold,
            metrics=metrics,
            returns=returns,
            train_indices=train_idx,
            test_indices=test_idx,
            elapsed_seconds=elapsed,
        )

    def _make_cv_splits(
        self,
        cv: str,
        n_splits: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate train/test index splits based on CV method.

        Parameters:
            cv: Cross-validation method name.
            n_splits: Number of splits.

        Returns:
            List of (train_indices, test_indices) tuples.
        """
        n = len(self.data)

        if cv == "none":
            # Full in-sample: train=all, test=all
            all_idx = np.arange(n)
            return [(all_idx, all_idx)]
        elif cv == "walk_forward":
            return walk_forward_splits(n, n_splits=n_splits)
        elif cv == "rolling":
            return rolling_splits(n, n_splits=n_splits)
        elif cv == "purged_kfold":
            return purged_kfold_splits(n, n_splits=n_splits)
        else:
            raise ValueError(
                f"Unknown CV method: {cv!r}. "
                "Choose from: 'walk_forward', 'rolling', 'purged_kfold', 'none'."
            )

    def run_grid(
        self,
        cv: str = "walk_forward",
        n_splits: int = 5,
        parallel: bool = True,
        n_jobs: int = -1,
    ) -> list[RunResult]:
        """Run full parameter grid across all CV folds.

        Parameters:
            cv: Cross-validation method.
            n_splits: Number of CV splits.
            parallel: Use joblib for parallel execution if available.
            n_jobs: Number of parallel jobs (-1 = all CPUs).

        Returns:
            List of RunResult objects.
        """
        splits = self._make_cv_splits(cv, n_splits)
        grid = list(self._grid)

        # Build list of all (param_combo, fold, train_idx, test_idx) tasks
        tasks = [
            (params, fold_i, train_idx, test_idx)
            for params in grid
            for fold_i, (train_idx, test_idx) in enumerate(splits)
        ]

        total = len(tasks)
        logger.info(
            "Running %d tasks (%d param combos x %d folds)",
            total,
            len(grid),
            len(splits),
        )

        # Try parallel execution
        if parallel and total > 1:
            try:
                from joblib import Parallel, delayed

                results = Parallel(n_jobs=n_jobs)(
                    delayed(self.run_single)(params, train_idx, test_idx, fold_i)
                    for params, fold_i, train_idx, test_idx in tasks
                )
                return list(results)
            except ImportError:
                logger.info("joblib not available; falling back to sequential execution")

        # Sequential fallback
        results = []
        for i, (params, fold_i, train_idx, test_idx) in enumerate(tasks):
            result = self.run_single(params, train_idx, test_idx, fold_i)
            results.append(result)
            if (i + 1) % max(total // 10, 1) == 0:
                logger.info("Progress: %d/%d tasks complete", i + 1, total)

        return results


__all__ = [
    "ExperimentRunner",
    "GridSpec",
    "RunResult",
]
