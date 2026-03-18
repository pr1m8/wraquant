"""Distributed computing for wraquant.

Provides wrappers for parallelizing quantitative finance workloads
using Dask and Ray.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from wraquant.core.decorators import requires_extra

__all__ = [
    "dask_map",
    "ray_map",
    "parallel_backtest",
]


@requires_extra("scale")
def dask_map(
    func: Callable,
    items: list,
    n_workers: int | None = None,
) -> list:
    """Apply a function to items in parallel using Dask.

    Parameters:
        func: Function to apply to each item.
        items: List of items to process.
        n_workers: Number of Dask workers (None for auto).

    Returns:
        List of results.
    """
    import dask
    from dask.distributed import Client

    client = Client(n_workers=n_workers, silence_logs=True)
    try:
        delayed_results = [dask.delayed(func)(item) for item in items]
        results = dask.compute(*delayed_results)
        return list(results)
    finally:
        client.close()


@requires_extra("scale")
def ray_map(
    func: Callable,
    items: list,
    num_cpus: int | None = None,
) -> list:
    """Apply a function to items in parallel using Ray.

    Parameters:
        func: Function to apply to each item.
        items: List of items to process.
        num_cpus: Number of CPUs to use (None for all).

    Returns:
        List of results.
    """
    import ray

    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus, ignore_reinit_error=True, logging_level="ERROR")

    remote_func = ray.remote(func)
    futures = [remote_func.remote(item) for item in items]
    results = ray.get(futures)
    return results


def parallel_backtest(
    strategy_fn: Callable,
    parameter_grid: list[dict],
    prices: pd.DataFrame,
    backend: str = "joblib",
    n_jobs: int = -1,
) -> list[dict]:
    """Run a backtest across a parameter grid in parallel.

    Uses joblib by default (no extra deps), with optional Dask/Ray backends.

    Parameters:
        strategy_fn: Function(prices, **params) -> dict with results.
        parameter_grid: List of parameter dicts to test.
        prices: Price data to backtest on.
        backend: "joblib", "dask", or "ray".
        n_jobs: Number of parallel jobs (-1 for all CPUs).

    Returns:
        List of result dicts, one per parameter combination.
    """
    if backend == "joblib":
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs)(
            delayed(strategy_fn)(prices, **params) for params in parameter_grid
        )
        return results
    elif backend == "dask":
        return dask_map(lambda p: strategy_fn(prices, **p), parameter_grid)
    elif backend == "ray":
        return ray_map(lambda p: strategy_fn(prices, **p), parameter_grid)
    else:
        raise ValueError(
            f"Unknown backend {backend!r}. Use 'joblib', 'dask', or 'ray'."
        )
