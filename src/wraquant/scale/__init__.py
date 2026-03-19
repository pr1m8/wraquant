"""Distributed computing for wraquant.

Provides wrappers for parallelizing quantitative finance workloads
using joblib, Dask, and Ray.  Every public function defaults to the
``joblib`` backend so it works out of the box; pass ``backend="dask"``
or ``backend="ray"`` (and install the corresponding extra group) for
heavier workloads.

Parallelism trade-offs
----------------------
Parallelization adds per-task overhead (serialization, IPC, worker
start-up).  Rule of thumb:

* **< 100 ms per task** -- stay sequential.
* **100 ms -- 1 s per task** -- ``joblib`` with threading or loky.
* **> 1 s per task** -- ``dask`` / ``ray`` for true distribution.

Functions
---------
Low-level primitives:
    dask_map, ray_map

Quant workflow helpers:
    parallel_backtest, parallel_optimize, parallel_walk_forward,
    parallel_regime_detection, parallel_monte_carlo,
    parallel_feature_compute, distributed_backtest, chunk_apply
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd

from wraquant.core.decorators import requires_extra

logger = logging.getLogger(__name__)

__all__ = [
    "dask_map",
    "ray_map",
    "parallel_backtest",
    "parallel_optimize",
    "parallel_walk_forward",
    "parallel_regime_detection",
    "parallel_monte_carlo",
    "parallel_feature_compute",
    "distributed_backtest",
    "chunk_apply",
]


@requires_extra("scale")
def dask_map(
    func: Callable,
    items: list,
    n_workers: int | None = None,
) -> list:
    """Apply a function to items in parallel using Dask distributed.

    Creates a temporary Dask ``Client``, submits all tasks as delayed
    computations, collects results, and shuts down the client.  Use
    this when tasks are CPU-intensive and you want multi-process
    parallelism with Dask's scheduler.

    Requires the ``scale`` extra group (``pip install wraquant[scale]``).

    Parameters:
        func (callable): Function to apply to each item.
        items (list): List of items to process.
        n_workers (int | None): Number of Dask workers.  ``None``
            auto-detects based on available CPUs.

    Returns:
        list: List of results, in the same order as *items*.

    Example:
        >>> results = dask_map(lambda x: x ** 2, [1, 2, 3])  # doctest: +SKIP
        >>> results
        [1, 4, 9]

    See Also:
        ray_map: Alternative using Ray.
        parallel_backtest: Higher-level parallel backtesting.
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

    Initialises Ray (if not already running), wraps *func* as a remote
    task, submits all items, and collects results.  Ray is ideal for
    heavy workloads that benefit from distributed scheduling and
    object-store-based data sharing.

    Requires the ``scale`` extra group (``pip install wraquant[scale]``).

    Parameters:
        func (callable): Function to apply to each item.
        items (list): List of items to process.
        num_cpus (int | None): Number of CPUs to use.  ``None`` uses
            all available CPUs.

    Returns:
        list: List of results, in the same order as *items*.

    Example:
        >>> results = ray_map(lambda x: x ** 2, [1, 2, 3])  # doctest: +SKIP
        >>> results
        [1, 4, 9]

    See Also:
        dask_map: Alternative using Dask.
        parallel_backtest: Higher-level parallel backtesting.
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

    Each parameter combination is evaluated independently, making
    this embarrassingly parallel.  Uses joblib by default (no extra
    dependencies), with optional Dask or Ray backends for heavier
    workloads or distributed clusters.

    Parameters:
        strategy_fn (callable): Strategy function with signature
            ``strategy_fn(prices, **params) -> dict``.  The returned
            dict should contain performance metrics (e.g.,
            ``{"sharpe": 1.2, "max_dd": -0.15}``).
        parameter_grid (list[dict]): List of parameter dicts to test.
            Each dict is unpacked as keyword arguments to
            *strategy_fn*.
        prices (pd.DataFrame): Price data to backtest on.
        backend (str): ``"joblib"`` (default), ``"dask"``, or
            ``"ray"``.
        n_jobs (int): Number of parallel jobs (-1 for all CPUs, joblib
            only).

    Returns:
        list[dict]: List of result dicts, one per parameter
            combination, in the same order as *parameter_grid*.

    Example:
        >>> def strat(prices, window=20):
        ...     ret = prices["close"].pct_change().rolling(window).mean().iloc[-1]
        ...     return {"sharpe": ret / 0.01}
        >>> grid = [{"window": w} for w in [10, 20, 50]]
        >>> results = parallel_backtest(strat, grid, prices)  # doctest: +SKIP
        >>> len(results)
        3

    See Also:
        distributed_backtest: Enhanced version with auto-backend
            selection and structured output.
        parallel_optimize: Sweep portfolio constraints in parallel.
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dispatch_parallel(
    func: Callable,
    items: list,
    backend: str = "joblib",
    n_jobs: int = -1,
    fail_fast: bool = False,
) -> list[Any]:
    """Run *func* over *items* with the chosen backend.

    Errors in individual tasks are caught and replaced with ``None``
    unless *fail_fast* is ``True``.  A warning is emitted for each
    failed task so callers receive partial results instead of losing
    the entire batch.

    Parameters:
        func: Callable to apply to each element of *items*.
        items: Iterable of inputs.
        backend: ``"joblib"``, ``"dask"``, or ``"ray"``.
        n_jobs: Worker count (joblib only, -1 = all CPUs).
        fail_fast: If ``True``, re-raise the first exception instead
            of collecting partial results.

    Returns:
        List of results (same length as *items*).  Failed items are
        ``None`` unless *fail_fast* is set.
    """
    if not items:
        return []

    def _safe(item: Any) -> Any:
        try:
            return func(item)
        except Exception:
            if fail_fast:
                raise
            logger.warning("Task failed for item %s", item, exc_info=True)
            return None

    if backend == "joblib":
        from joblib import Parallel, delayed

        return Parallel(n_jobs=n_jobs)(delayed(_safe)(item) for item in items)
    elif backend == "dask":
        return dask_map(_safe, items)
    elif backend == "ray":
        return ray_map(_safe, items)
    else:
        raise ValueError(
            f"Unknown backend {backend!r}. Use 'joblib', 'dask', or 'ray'."
        )


# ---------------------------------------------------------------------------
# parallel_optimize
# ---------------------------------------------------------------------------


def parallel_optimize(
    returns: pd.DataFrame,
    constraint_sets: list[dict[str, Any]],
    method: str = "mean_variance",
    backend: str = "joblib",
    n_jobs: int = -1,
    **optimizer_kwargs: Any,
) -> list[dict[str, Any]]:
    """Run portfolio optimization across multiple constraint sets in parallel.

    This is the go-to tool for **robust optimization** and **constraint
    sensitivity analysis**.  Instead of fitting a single optimal portfolio,
    sweep over several constraint configurations (e.g., different max-weight
    caps, turnover limits, sector bounds) and compare the resulting
    allocations side by side.

    When to parallelise
    ^^^^^^^^^^^^^^^^^^^
    Mean-variance with a moderate asset universe (< 200 assets) solves in
    < 50 ms, so parallelism only pays off when you have many constraint
    sets (> 20) or when the solver itself is expensive (risk parity with
    large universes, hierarchical risk parity with bootstrap).

    Parameters:
        returns: ``(T, N)`` DataFrame of asset returns.  Column names
            become asset names in the result.
        constraint_sets: Each dict is forwarded as keyword arguments to
            the optimizer **on top of** *optimizer_kwargs*.  Typical keys:
            ``weight_bounds``, ``max_weight``, ``sector_mapper``,
            ``sector_upper``.
        method: Optimization method name.  One of ``"mean_variance"``,
            ``"min_volatility"``, ``"max_sharpe"``, ``"risk_parity"``,
            ``"black_litterman"``, ``"hierarchical_risk_parity"``,
            ``"equal_weight"``, ``"inverse_volatility"``.
        backend: ``"joblib"`` (default), ``"dask"``, or ``"ray"``.
        n_jobs: Number of parallel workers (joblib only, -1 = all CPUs).
        **optimizer_kwargs: Extra keyword arguments passed to every
            optimization call (e.g., ``risk_free_rate=0.04``).

    Returns:
        List of dicts, one per constraint set, each containing:

        - ``weights`` -- ``np.ndarray`` of optimal weights.
        - ``expected_return`` -- annualized expected return.
        - ``volatility`` -- annualized portfolio volatility.
        - ``sharpe_ratio`` -- Sharpe ratio.
        - ``asset_names`` -- list of asset name strings.
        - ``constraints`` -- the constraint dict that produced this result.
        - ``error`` -- ``None`` on success, error message string on failure.

    Raises:
        ValueError: If *method* is not recognised or *constraint_sets*
            is empty.

    Example:
        >>> import pandas as pd, numpy as np
        >>> returns = pd.DataFrame(
        ...     np.random.randn(252, 4) * 0.01,
        ...     columns=["SPY", "TLT", "GLD", "VWO"],
        ... )
        >>> results = parallel_optimize(
        ...     returns,
        ...     [{"max_weight": w} for w in [0.25, 0.40, 0.60, 1.0]],
        ...     method="mean_variance",
        ... )
        >>> len(results)
        4
    """
    from wraquant import opt as _opt

    _METHODS: dict[str, Callable] = {
        "mean_variance": _opt.mean_variance,
        "min_volatility": _opt.min_volatility,
        "max_sharpe": _opt.max_sharpe,
        "risk_parity": _opt.risk_parity,
        "equal_weight": _opt.equal_weight,
        "inverse_volatility": _opt.inverse_volatility,
        "hierarchical_risk_parity": _opt.hierarchical_risk_parity,
        "black_litterman": _opt.black_litterman,
    }

    if method not in _METHODS:
        raise ValueError(f"Unknown method {method!r}. Choose from: {sorted(_METHODS)}")
    if not constraint_sets:
        raise ValueError("constraint_sets must be a non-empty list of dicts.")

    opt_fn = _METHODS[method]

    def _run_one(constraints: dict[str, Any]) -> dict[str, Any]:
        merged = {**optimizer_kwargs, **constraints}
        result = opt_fn(returns, **merged)
        return {
            "weights": np.asarray(result.weights),
            "expected_return": float(getattr(result, "expected_return", 0.0)),
            "volatility": float(getattr(result, "volatility", 0.0)),
            "sharpe_ratio": float(getattr(result, "sharpe_ratio", 0.0)),
            "asset_names": list(
                getattr(result, "asset_names", returns.columns.tolist())
            ),
            "constraints": constraints,
            "error": None,
        }

    raw = _dispatch_parallel(_run_one, constraint_sets, backend=backend, n_jobs=n_jobs)

    # Replace None (failed) entries with an error sentinel
    results: list[dict[str, Any]] = []
    for idx, r in enumerate(raw):
        if r is None:
            results.append(
                {
                    "weights": np.array([]),
                    "expected_return": float("nan"),
                    "volatility": float("nan"),
                    "sharpe_ratio": float("nan"),
                    "asset_names": [],
                    "constraints": constraint_sets[idx],
                    "error": "Optimization failed -- see logs for traceback.",
                }
            )
        else:
            results.append(r)
    return results


# ---------------------------------------------------------------------------
# parallel_walk_forward
# ---------------------------------------------------------------------------


def parallel_walk_forward(
    model_fn: Callable[..., Any],
    data: pd.DataFrame,
    n_windows: int = 5,
    train_ratio: float = 0.7,
    backend: str = "joblib",
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Parallelised walk-forward validation.

    Walk-forward is the gold standard for time-series model validation:
    the data is split into *n_windows* non-overlapping folds, each fold
    is split into a training and out-of-sample (OOS) segment, the model
    is fitted on the training segment, and predictions are collected on
    the OOS segment.

    Sequential walk-forward can be painfully slow when fitting is
    expensive (GARCH, HMM, ML models).  This function fits each window
    **independently and in parallel**, giving near-linear speed-up with
    the number of windows.

    When to parallelise
    ^^^^^^^^^^^^^^^^^^^
    If each window fits in < 200 ms (e.g., linear regression on a small
    dataset), joblib overhead dominates and sequential may be faster.
    For GARCH, HMMs, gradient-boosted trees, or neural nets, parallel
    walk-forward provides significant speed-up.

    Parameters:
        model_fn: ``model_fn(train_df, test_df) -> dict`` where the
            returned dict must contain at least ``"predictions"`` (array
            or Series of OOS predictions) and optionally ``"metrics"``
            (dict of scalar performance numbers).
        data: Full dataset as a pandas DataFrame.  Rows must be in
            chronological order.
        n_windows: Number of walk-forward windows.
        train_ratio: Fraction of each window used for training (the
            remainder is OOS).
        backend: ``"joblib"`` (default), ``"dask"``, or ``"ray"``.
        n_jobs: Number of parallel workers (joblib only, -1 = all CPUs).

    Returns:
        Dict with keys:

        - ``predictions`` -- concatenated OOS predictions (``np.ndarray``).
        - ``actuals`` -- concatenated OOS actuals (``np.ndarray``),
          taken from the first column of *data* if ``model_fn`` does not
          return an ``"actuals"`` key.
        - ``metrics_per_window`` -- list of per-window metric dicts.
        - ``window_indices`` -- list of ``(train_start, train_end,
          test_start, test_end)`` tuples.

    Raises:
        ValueError: If *n_windows* < 1 or *data* is too small.

    Example:
        >>> def my_model(train, test):
        ...     pred = [train.iloc[:, 0].mean()] * len(test)
        ...     return {"predictions": pred}
        >>> result = parallel_walk_forward(my_model, data, n_windows=4)
        >>> result["predictions"].shape
        (120,)
    """
    n = len(data)
    if n_windows < 1:
        raise ValueError("n_windows must be >= 1.")
    window_size = n // n_windows
    if window_size < 2:
        raise ValueError(f"Data has {n} rows, too small for {n_windows} windows.")

    # Build window boundaries
    windows: list[tuple[int, int, int, int]] = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size if i < n_windows - 1 else n
        split = start + max(1, int((end - start) * train_ratio))
        windows.append((start, split, split, end))

    def _fit_window(window: tuple[int, int, int, int]) -> dict[str, Any]:
        tr_s, tr_e, te_s, te_e = window
        train_df = data.iloc[tr_s:tr_e]
        test_df = data.iloc[te_s:te_e]
        result = model_fn(train_df, test_df)
        preds = np.asarray(result.get("predictions", []))
        actuals = np.asarray(result.get("actuals", test_df.iloc[:, 0].values))
        metrics = result.get("metrics", {})
        return {"predictions": preds, "actuals": actuals, "metrics": metrics}

    raw = _dispatch_parallel(_fit_window, windows, backend=backend, n_jobs=n_jobs)

    all_preds: list[np.ndarray] = []
    all_actuals: list[np.ndarray] = []
    metrics_per_window: list[dict] = []
    for idx, r in enumerate(raw):
        if r is not None:
            all_preds.append(r["predictions"])
            all_actuals.append(r["actuals"])
            metrics_per_window.append(r["metrics"])
        else:
            metrics_per_window.append({"error": f"Window {idx} failed."})

    return {
        "predictions": np.concatenate(all_preds) if all_preds else np.array([]),
        "actuals": np.concatenate(all_actuals) if all_actuals else np.array([]),
        "metrics_per_window": metrics_per_window,
        "window_indices": windows,
    }


# ---------------------------------------------------------------------------
# parallel_regime_detection
# ---------------------------------------------------------------------------


def parallel_regime_detection(
    returns: pd.DataFrame,
    method: str = "gaussian_hmm",
    n_regimes: int = 2,
    backend: str = "joblib",
    n_jobs: int = -1,
    **detect_kwargs: Any,
) -> dict[str, Any]:
    """Run regime detection across multiple assets simultaneously.

    Fitting an HMM or GMM to each asset is independent, so we can
    trivially parallelise across the column axis of a multi-asset
    returns DataFrame.

    When to parallelise
    ^^^^^^^^^^^^^^^^^^^
    HMM fitting via EM is O(T * K^2) per iteration and typically runs
    10-100 iterations.  For a 10-year daily series (T ~ 2 500) with
    K = 3 states, each fit takes ~50-200 ms.  With 50+ assets,
    parallelism shaves significant wall-clock time.

    Parameters:
        returns: ``(T, N)`` DataFrame where each column is an asset's
            return series.
        method: Detection method forwarded to
            ``wraquant.regimes.detect_regimes``.  Common values:
            ``"gaussian_hmm"``, ``"gmm"``, ``"ms_regression"``.
        n_regimes: Number of regimes to detect per asset.
        backend: ``"joblib"`` (default), ``"dask"``, or ``"ray"``.
        n_jobs: Number of parallel workers (joblib only, -1 = all CPUs).
        **detect_kwargs: Extra keyword arguments forwarded to
            ``detect_regimes`` (e.g., ``n_iter=200``).

    Returns:
        Dict mapping asset name (``str``) to a
        ``wraquant.regimes.RegimeResult`` object.  Failed assets map to
        ``None`` with a warning logged.

    Example:
        >>> import pandas as pd, numpy as np
        >>> rets = pd.DataFrame(
        ...     np.random.randn(500, 3) * 0.01,
        ...     columns=["SPY", "TLT", "GLD"],
        ... )
        >>> results = parallel_regime_detection(rets, n_regimes=2)
        >>> results["SPY"].n_regimes
        2
    """
    from wraquant.regimes import detect_regimes

    assets = returns.columns.tolist()

    def _detect_one(asset: str) -> Any:
        series = returns[asset]
        return detect_regimes(
            series, method=method, n_regimes=n_regimes, **detect_kwargs
        )

    raw = _dispatch_parallel(_detect_one, assets, backend=backend, n_jobs=n_jobs)

    return {asset: result for asset, result in zip(assets, raw)}


# ---------------------------------------------------------------------------
# parallel_monte_carlo
# ---------------------------------------------------------------------------


def parallel_monte_carlo(
    simulation_fn: Callable[..., np.ndarray | pd.DataFrame],
    n_simulations: int,
    n_workers: int | None = None,
    backend: str = "joblib",
    **sim_kwargs: Any,
) -> np.ndarray:
    """Split Monte Carlo simulations across parallel workers and merge.

    Monte Carlo is *embarrassingly parallel* -- each path is independent.
    This function divides *n_simulations* into roughly equal chunks,
    runs ``simulation_fn(n, **sim_kwargs)`` in each worker (where *n* is
    the chunk size), and concatenates the results along axis 0.

    Common use cases:

    * **MC VaR / CVaR** -- simulate portfolio returns and compute tail
      quantiles.
    * **Option pricing** -- average discounted payoffs across paths.
    * **Stress testing** -- generate thousands of correlated shock
      scenarios.

    When to parallelise
    ^^^^^^^^^^^^^^^^^^^
    Most MC engines spend 80% of time in numpy/scipy vectorised code
    that already releases the GIL, so ``joblib`` with ``"loky"``
    (process-based) is ideal.  Dask/Ray add value when simulations are
    very long-running or when you need to scale beyond a single machine.

    Parameters:
        simulation_fn: ``simulation_fn(n_sims, **sim_kwargs) -> array``
            where the returned array has shape ``(n_sims, ...)`` (first
            axis is the simulation index).
        n_simulations: Total number of simulations to run.
        n_workers: Number of workers.  ``None`` means auto-detect
            (``os.cpu_count()`` for joblib).
        backend: ``"joblib"`` (default), ``"dask"``, or ``"ray"``.
        **sim_kwargs: Extra keyword arguments forwarded to
            *simulation_fn*.

    Returns:
        ``np.ndarray`` of shape ``(n_simulations, ...)`` with all
        simulation results concatenated along axis 0.

    Raises:
        ValueError: If *n_simulations* < 1.

    Example:
        >>> import numpy as np
        >>> def gbm(n, S0=100, mu=0.05, sigma=0.2, T=1, steps=252):
        ...     dt = T / steps
        ...     z = np.random.randn(n, steps)
        ...     paths = S0 * np.exp(
        ...         np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z, axis=1)
        ...     )
        ...     return paths
        >>> results = parallel_monte_carlo(gbm, n_simulations=10_000, n_workers=4)
        >>> results.shape[0]
        10000
    """
    import os

    if n_simulations < 1:
        raise ValueError("n_simulations must be >= 1.")

    effective_workers = n_workers or os.cpu_count() or 1
    # Divide simulations into chunks as evenly as possible
    base, remainder = divmod(n_simulations, effective_workers)
    chunks = [base + (1 if i < remainder else 0) for i in range(effective_workers)]
    # Remove zero-size chunks (n_simulations < n_workers)
    chunks = [c for c in chunks if c > 0]

    def _run_chunk(n: int) -> np.ndarray:
        result = simulation_fn(n, **sim_kwargs)
        return np.asarray(result)

    raw = _dispatch_parallel(_run_chunk, chunks, backend=backend, n_jobs=len(chunks))

    # Filter out failed chunks
    valid = [r for r in raw if r is not None]
    if not valid:
        return np.array([])
    return np.concatenate(valid, axis=0)


# ---------------------------------------------------------------------------
# parallel_feature_compute
# ---------------------------------------------------------------------------


def parallel_feature_compute(
    prices: pd.DataFrame,
    feature_fn: Callable[[pd.Series], pd.DataFrame | pd.Series],
    backend: str = "joblib",
    n_jobs: int = -1,
) -> dict[str, pd.DataFrame]:
    """Compute features for multiple assets in parallel.

    A common pattern in alpha research is to compute a large feature
    matrix (technical indicators, rolling statistics, microstructure
    features) for each asset independently.  This function parallelises
    that column-by-column loop.

    When to parallelise
    ^^^^^^^^^^^^^^^^^^^
    Feature computation is typically fast per asset (< 50 ms) but
    becomes significant when you have 500+ assets or the feature
    function is expensive (e.g., wavelet decomposition, entropy
    measures, GARCH-fitted volatility).

    Parameters:
        prices: ``(T, N)`` DataFrame where each column is an asset's
            price (or return) series.
        feature_fn: ``feature_fn(series) -> DataFrame | Series``.
            Called once per column of *prices*.  The returned object is
            stored under the column name in the output dict.
        backend: ``"joblib"`` (default), ``"dask"``, or ``"ray"``.
        n_jobs: Number of parallel workers (joblib only, -1 = all CPUs).

    Returns:
        Dict mapping asset name (``str``) to a ``pd.DataFrame`` (or
        ``pd.Series``) of computed features.  Failed assets map to
        ``None`` with a warning logged.

    Example:
        >>> import pandas as pd, numpy as np
        >>> prices = pd.DataFrame(
        ...     np.cumsum(np.random.randn(252, 3), axis=0) + 100,
        ...     columns=["AAPL", "GOOG", "MSFT"],
        ... )
        >>> def my_features(s):
        ...     return pd.DataFrame({
        ...         "sma_20": s.rolling(20).mean(),
        ...         "vol_20": s.rolling(20).std(),
        ...     })
        >>> feats = parallel_feature_compute(prices, my_features)
        >>> sorted(feats.keys())
        ['AAPL', 'GOOG', 'MSFT']
    """
    assets = prices.columns.tolist()

    def _compute_one(asset: str) -> pd.DataFrame | pd.Series:
        return feature_fn(prices[asset])

    raw = _dispatch_parallel(_compute_one, assets, backend=backend, n_jobs=n_jobs)
    return {asset: result for asset, result in zip(assets, raw)}


# ---------------------------------------------------------------------------
# distributed_backtest
# ---------------------------------------------------------------------------


def distributed_backtest(
    strategy_fn: Callable[..., dict[str, Any]],
    parameter_grid: list[dict[str, Any]],
    prices: pd.DataFrame,
    backend: str | None = None,
    n_jobs: int = -1,
    auto_backend_threshold: int = 50,
) -> dict[str, Any]:
    """Run backtests across a parameter grid with progress tracking.

    An enhanced version of :func:`parallel_backtest` that:

    1. **Auto-selects the backend** -- uses ``"joblib"`` for small grids
       and ``"dask"`` or ``"ray"`` (if installed) for large grids.
    2. **Returns structured output** -- a DataFrame with parameter
       columns and metric columns, plus the single best parameter set.
    3. **Handles failures gracefully** -- failed parameter sets are
       reported but do not crash the entire sweep.

    When to parallelise
    ^^^^^^^^^^^^^^^^^^^
    Always, if you have more than one parameter set.  Backtests are
    independent across parameters, making this embarrassingly parallel.
    The only concern is memory: if each backtest holds large price
    arrays, you may want to limit ``n_jobs``.

    Parameters:
        strategy_fn: ``strategy_fn(prices, **params) -> dict`` where
            the returned dict contains at least one numeric metric
            (e.g., ``{"sharpe": 1.2, "max_dd": -0.15}``).
        parameter_grid: List of parameter dicts to sweep.
        prices: Price data passed to *strategy_fn*.
        backend: ``"joblib"``, ``"dask"``, ``"ray"``, or ``None`` for
            auto-selection.
        n_jobs: Number of parallel workers (joblib only, -1 = all CPUs).
        auto_backend_threshold: Grid size above which auto-selection
            prefers ``dask`` or ``ray`` (if available).

    Returns:
        Dict with keys:

        - ``results_df`` -- ``pd.DataFrame`` with one row per param set.
          Columns are the union of parameter keys and metric keys.
        - ``best_params`` -- the parameter dict that produced the highest
          ``"sharpe"`` value (or the first metric key if ``"sharpe"`` is
          absent).
        - ``best_metrics`` -- metric dict for the best run.
        - ``n_succeeded`` -- number of successful runs.
        - ``n_failed`` -- number of failed runs.

    Example:
        >>> def strat(prices, window=20):
        ...     ret = prices["close"].pct_change().rolling(window).mean().iloc[-1]
        ...     return {"sharpe": ret / 0.01, "window": window}
        >>> grid = [{"window": w} for w in range(5, 60, 5)]
        >>> out = distributed_backtest(strat, grid, prices)
        >>> isinstance(out["results_df"], pd.DataFrame)
        True
    """
    # Auto-select backend
    if backend is None:
        if len(parameter_grid) > auto_backend_threshold:
            # Prefer dask/ray for large grids if available
            try:
                import dask  # noqa: F401

                backend = "dask"
            except ImportError:
                try:
                    import ray  # noqa: F401

                    backend = "ray"
                except ImportError:
                    backend = "joblib"
        else:
            backend = "joblib"

    def _run_one(params: dict) -> dict[str, Any]:
        metrics = strategy_fn(prices, **params)
        return {"params": params, "metrics": metrics}

    raw = _dispatch_parallel(_run_one, parameter_grid, backend=backend, n_jobs=n_jobs)

    rows: list[dict[str, Any]] = []
    n_failed = 0
    for idx, r in enumerate(raw):
        if r is not None:
            row = {**r["params"], **r["metrics"]}
            rows.append(row)
        else:
            n_failed += 1

    results_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    n_succeeded = len(rows)

    # Determine best run
    best_params: dict[str, Any] = {}
    best_metrics: dict[str, Any] = {}
    if not results_df.empty:
        # Identify metric columns (those not in parameter keys)
        all_param_keys = set()
        for p in parameter_grid:
            all_param_keys.update(p.keys())
        metric_cols = [c for c in results_df.columns if c not in all_param_keys]

        sort_col = (
            "sharpe"
            if "sharpe" in metric_cols
            else (metric_cols[0] if metric_cols else None)
        )
        if sort_col is not None and sort_col in results_df.columns:
            best_idx = results_df[sort_col].idxmax()
            best_row = results_df.loc[best_idx]
            best_params = {
                k: best_row[k] for k in all_param_keys if k in best_row.index
            }
            best_metrics = {k: best_row[k] for k in metric_cols if k in best_row.index}

    return {
        "results_df": results_df,
        "best_params": best_params,
        "best_metrics": best_metrics,
        "n_succeeded": n_succeeded,
        "n_failed": n_failed,
    }


# ---------------------------------------------------------------------------
# chunk_apply
# ---------------------------------------------------------------------------


def chunk_apply(
    df: pd.DataFrame,
    func: Callable[[pd.DataFrame], pd.DataFrame],
    n_chunks: int | None = None,
    by: str | list[str] | None = None,
    backend: str = "joblib",
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Apply a function to chunks of a large DataFrame in parallel.

    Two chunking strategies:

    * **Row-based** (default) -- split the DataFrame into *n_chunks*
      contiguous slices of roughly equal size.  Best for tick data,
      high-frequency returns, or any operation that is local to each
      row.
    * **Group-based** (``by="column_name"``) -- split by the unique
      values of one or more columns, similar to
      ``df.groupby(by).apply(func)`` but in parallel.  Best for
      per-symbol or per-date processing.

    When to parallelise
    ^^^^^^^^^^^^^^^^^^^
    * Row-based: useful when *func* is expensive and the DataFrame has
      > 1 M rows (e.g., computing microstructure features on tick data).
    * Group-based: useful when each group is independent and there are
      many groups (> 20 symbols, > 100 dates).

    Beware that serialization overhead grows with DataFrame size.  If
    the DataFrame is very wide (> 1 000 columns) or each chunk is
    small (< 1 000 rows), sequential ``apply`` may be faster.

    Parameters:
        df: Input DataFrame.
        func: ``func(chunk_df) -> DataFrame``.  Must return a DataFrame
            that can be concatenated with the other chunks.
        n_chunks: Number of row-based chunks (ignored when ``by`` is
            set).  Defaults to ``os.cpu_count()``.
        by: Column name(s) to group by before applying *func*.  When
            set, each group is processed as a separate chunk.
        backend: ``"joblib"`` (default), ``"dask"``, or ``"ray"``.
        n_jobs: Number of parallel workers (joblib only, -1 = all CPUs).

    Returns:
        Concatenated ``pd.DataFrame`` of results (row-based: same order
        as input; group-based: sorted by group key).

    Raises:
        ValueError: If the DataFrame is empty.

    Example:
        >>> import pandas as pd, numpy as np
        >>> df = pd.DataFrame({"x": np.arange(10_000), "y": np.random.randn(10_000)})
        >>> def process(chunk):
        ...     chunk = chunk.copy()
        ...     chunk["z"] = chunk["y"].cumsum()
        ...     return chunk
        >>> result = chunk_apply(df, process, n_chunks=4)
        >>> len(result) == len(df)
        True
    """
    import os

    if df.empty:
        raise ValueError("Cannot chunk_apply on an empty DataFrame.")

    if by is not None:
        # Group-based chunking
        groups = [group_df for _, group_df in df.groupby(by, sort=True)]
    else:
        # Row-based chunking
        effective_chunks = n_chunks or os.cpu_count() or 1
        indices = np.array_split(np.arange(len(df)), effective_chunks)
        groups = [df.iloc[idx] for idx in indices if len(idx) > 0]

    raw = _dispatch_parallel(func, groups, backend=backend, n_jobs=n_jobs)

    valid = [r for r in raw if r is not None]
    if not valid:
        warnings.warn(
            "All chunks failed in chunk_apply; returning empty DataFrame.",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame()
    return pd.concat(valid, ignore_index=(by is None))
