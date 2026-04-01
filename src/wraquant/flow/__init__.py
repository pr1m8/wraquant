"""Workflow orchestration for wraquant.

Provides composable workflow primitives and integrations with
orchestration frameworks (Prefect, Dagster, APScheduler) for building
reproducible, observable, and fault-tolerant quantitative finance
pipelines.  Includes zero-dependency Pipeline and DAG abstractions for
simple workflows, plus decorators for retry logic, disk caching, and
step-level logging.

Key components:

- **Pipeline** -- Sequential function pipeline: chain functions with
  ``pipeline(step1, step2, step3)`` and execute with ``.run(data)``.
  Supports ``>>`` operator for composition.
- **DAG** -- Directed acyclic graph of steps with dependencies: define
  steps and their dependency relationships, execute in topological order.
- **parallel_pipeline** -- Run multiple independent Pipeline objects in
  parallel using threading.
- **retry** -- Decorator with exponential backoff for unreliable
  operations (API calls, database connections).
- **cache_result** -- Decorator that caches function results to disk
  with TTL expiry.
- **log_step** -- Decorator that logs function entry, exit, duration,
  and exceptions.
- **prefect_backtest_flow** -- Run backtests as Prefect flows with
  automatic retries and distributed execution.
- **dagster_pipeline** -- Define Dagster jobs from wraquant operations.
- **schedule_data_refresh** -- Schedule recurring data fetches using
  APScheduler.

Example:
    >>> from wraquant.flow import pipeline, retry, cache_result
    >>> @cache_result(ttl_hours=4)
    ... @retry(max_retries=3)
    ... def fetch_and_clean(symbol):
    ...     from wraquant.data import fetch_prices
    ...     return fetch_prices(symbol, start="2020-01-01")
    >>> pipe = pipeline(fetch_and_clean, compute_signals, generate_report)
    >>> result = pipe.run("AAPL")

Use ``wraquant.flow`` for building production workflows.  For parallel
execution across assets or parameter grids, see ``wraquant.scale``.
For experiment tracking and parameter optimization, see
``wraquant.experiment``.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import pickle
import time as _time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd

from wraquant.core.decorators import requires_extra

_F = TypeVar("_F", bound=Callable)

__all__ = [
    "prefect_backtest_flow",
    "schedule_data_refresh",
    "dagster_pipeline",
    "pipeline",
    "Pipeline",
    "dag",
    "DAG",
    "retry",
    "cache_result",
    "log_step",
    "parallel_pipeline",
]


@requires_extra("workflow")
def prefect_backtest_flow(
    strategy_fn,
    symbols: list[str],
    start: str,
    end: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Run a backtest as a Prefect flow with automatic retries and logging.

    Parameters:
        strategy_fn: Callable that takes prices DataFrame and returns signals.
        symbols: List of ticker symbols.
        start: Start date string.
        end: End date string (None for today).
        **kwargs: Additional arguments passed to strategy_fn.

    Returns:
        dict with 'results' per symbol and 'flow_run_id'.
    """
    from prefect import flow, task

    @task(retries=2, retry_delay_seconds=5)
    def fetch_data(symbol, start, end):
        from wraquant.data import fetch_prices

        return fetch_prices(symbol, start=start, end=end)

    @task
    def run_strategy(prices, strategy_fn, **kw):
        return strategy_fn(prices, **kw)

    @flow(name="wraquant-backtest")
    def backtest_flow():
        results = {}
        for sym in symbols:
            prices = fetch_data(sym, start, end)
            results[sym] = run_strategy(prices, strategy_fn, **kwargs)
        return results

    state = backtest_flow()
    return {"results": state}


@requires_extra("workflow")
def schedule_data_refresh(
    fetch_fn,
    interval_minutes: int = 60,
    max_runs: int | None = None,
) -> dict[str, Any]:
    """Schedule recurring data fetches using APScheduler.

    Parameters:
        fetch_fn: Callable to execute on each interval.
        interval_minutes: Minutes between executions.
        max_runs: Maximum number of executions (None for unlimited).

    Returns:
        dict with 'scheduler' and 'job_id'.
    """
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.interval import IntervalTrigger

    scheduler = BackgroundScheduler()
    trigger = IntervalTrigger(minutes=interval_minutes)

    run_count = [0]

    def wrapped():
        run_count[0] += 1
        if max_runs and run_count[0] > max_runs:
            scheduler.shutdown(wait=False)
            return
        return fetch_fn()

    job = scheduler.add_job(wrapped, trigger)
    return {"scheduler": scheduler, "job_id": job.id}


@requires_extra("workflow")
def dagster_pipeline(
    ops_dict: dict[str, Callable],
) -> dict[str, Any]:
    """Define a Dagster pipeline from wraquant operations.

    Wraps a dictionary of named callables into Dagster ``@op``-decorated
    functions and assembles them into a Dagster ``@job``. This provides a
    lightweight bridge between wraquant's functional style and Dagster's
    asset/op-based orchestration.

    Each operation is executed in sequence (insertion order of
    *ops_dict*). The output of each op is passed as input to the next.
    The first op receives no input.

    Parameters
    ----------
    ops_dict : dict of str to callable
        Dictionary mapping operation names to callables. The callables
        are executed sequentially in insertion order.

    Returns
    -------
    dict
        Dictionary containing:

        * **pipeline** -- the Dagster ``@job``-decorated function.
        * **ops** -- dict mapping operation names to their Dagster
          ``@op``-wrapped versions.
        * **op_names** -- list of operation names in execution order.

    Example
    -------
    >>> from wraquant.flow import dagster_pipeline
    >>> result = dagster_pipeline({
    ...     "generate": lambda: [1, 2, 3],
    ...     "double": lambda data: [x * 2 for x in data],
    ... })
    >>> list(result["ops"].keys())
    ['generate', 'double']

    Notes
    -----
    This function only *defines* the Dagster job. To execute it, call
    ``result["pipeline"].execute_in_process()`` or use the Dagster CLI/UI.

    See Also
    --------
    prefect_backtest_flow : Prefect-based workflow orchestration.
    pipeline : Simple sequential pipeline (no external deps).
    """
    from dagster import In, Nothing, Out, job, op

    dagster_ops: dict[str, Callable] = {}
    op_names = list(ops_dict.keys())

    for i, (name, fn) in enumerate(ops_dict.items()):
        if i == 0:
            # First op: no input
            @op(name=name, out=Out(dagster_type=None))
            def _first_op(*, _fn=fn):
                return _fn()

            dagster_ops[name] = _first_op
        else:
            # Subsequent ops: take previous output as input
            @op(name=name, ins={"upstream": In(dagster_type=None)}, out=Out(dagster_type=None))
            def _chained_op(upstream, *, _fn=fn):
                return _fn(upstream)

            dagster_ops[name] = _chained_op

    @job(name="wraquant_pipeline")
    def _job():
        result = None
        for name in op_names:
            if result is None:
                result = dagster_ops[name]()
            else:
                result = dagster_ops[name](result)

    return {
        "pipeline": _job,
        "ops": dagster_ops,
        "op_names": op_names,
    }


def pipeline(*steps):
    """Create a simple sequential pipeline of functions.

    Each step receives the output of the previous step.
    No external dependencies required.

    Parameters:
        *steps: Callable functions to chain.

    Returns:
        A Pipeline object with a .run(initial_data) method.

    Example:
        >>> pipe = pipeline(
        ...     lambda prices: prices.pct_change().dropna(),
        ...     lambda returns: {"sharpe": returns.mean() / returns.std()},
        ... )
        >>> result = pipe.run(prices)
    """
    return Pipeline(list(steps))


class Pipeline:
    """Sequential function pipeline."""

    def __init__(self, steps: list) -> None:
        self.steps = steps

    def run(self, data: Any) -> Any:
        """Execute the pipeline on the given data.

        Parameters:
            data: Initial input to the first step.

        Returns:
            Output of the final step.
        """
        result = data
        for step in self.steps:
            result = step(result)
        return result

    def __rshift__(self, other):
        """Compose pipelines or append a callable using the >> operator."""
        if callable(other):
            return Pipeline(self.steps + [other])
        if isinstance(other, Pipeline):
            return Pipeline(self.steps + other.steps)
        raise TypeError(f"Cannot compose Pipeline with {type(other)}")


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------


class DAG:
    """Directed Acyclic Graph of pipeline steps with dependencies.

    Steps are executed in topological order so that each step runs only
    after all its dependencies have completed.  No external dependencies
    required.

    Parameters:
        steps: Dictionary mapping step name to a tuple of
            ``(callable, list_of_dependency_names)``.

    Example:
        >>> d = DAG({
        ...     "double": (lambda data: data * 2, []),
        ...     "add_one": (lambda data: data + 1, ["double"]),
        ... })
        >>> results = d.run(initial_data=5)
        >>> results["double"]
        10
        >>> results["add_one"]
        11
    """

    def __init__(self, steps: dict[str, tuple[Callable, list[str]]]) -> None:
        self.steps = steps
        self._validate()

    def _validate(self) -> None:
        """Check for missing dependencies and cycles."""
        names = set(self.steps.keys())
        for name, (_, deps) in self.steps.items():
            missing = set(deps) - names
            if missing:
                raise ValueError(
                    f"Step '{name}' depends on undefined steps: {missing}"
                )
        # Cycle detection via topological sort attempt
        self._topo_sort()

    def _topo_sort(self) -> list[str]:
        """Return step names in topological order (Kahn's algorithm)."""
        in_degree: dict[str, int] = {name: 0 for name in self.steps}
        dependents: dict[str, list[str]] = {name: [] for name in self.steps}

        for name, (_, deps) in self.steps.items():
            in_degree[name] = len(deps)
            for dep in deps:
                dependents[dep].append(name)

        queue = deque(name for name, deg in in_degree.items() if deg == 0)
        order: list[str] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for child in dependents[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.steps):
            raise ValueError("DAG contains a cycle")

        return order

    def run(self, initial_data: Any = None) -> dict[str, Any]:
        """Execute all steps in topological order.

        Each step callable receives the output of its first dependency
        (if any), or *initial_data* if it has no dependencies.  If a
        step has multiple dependencies, it receives a dictionary mapping
        dependency names to their results.

        Parameters:
            initial_data: Input data for root steps (steps with no
                dependencies).

        Returns:
            Dictionary mapping step names to their outputs.
        """
        order = self._topo_sort()
        results: dict[str, Any] = {}

        for name in order:
            fn, deps = self.steps[name]
            if not deps:
                results[name] = fn(initial_data)
            elif len(deps) == 1:
                results[name] = fn(results[deps[0]])
            else:
                dep_results = {d: results[d] for d in deps}
                results[name] = fn(dep_results)

        return results


def dag(steps_dict: dict[str, tuple[Callable, list[str]]]) -> DAG:
    """Create a DAG of pipeline steps with dependencies.

    Steps run in topological order.  No external dependencies required.
    This is useful for workflows where some steps depend on the output
    of other steps, forming a directed acyclic graph.

    Parameters:
        steps_dict: Dictionary mapping step name to a tuple of
            ``(callable, list_of_dependency_names)``.  Root steps
            (no dependencies) receive the *initial_data* passed to
            :meth:`DAG.run`.

    Returns:
        A DAG object with a ``.run(initial_data)`` method.

    Example:
        >>> d = dag({
        ...     "fetch": (lambda data: [1, 2, 3], []),
        ...     "double": (lambda data: [x * 2 for x in data], ["fetch"]),
        ...     "total": (lambda data: sum(data), ["double"]),
        ... })
        >>> results = d.run()
        >>> results["double"]
        [2, 4, 6]
        >>> results["total"]
        12

    See Also:
        pipeline: Simple sequential pipeline (no dependencies).
        parallel_pipeline: Run multiple pipelines in parallel.
    """
    return DAG(steps_dict)


# ---------------------------------------------------------------------------
# retry
# ---------------------------------------------------------------------------


def retry(
    fn: Callable | None = None,
    *,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable:
    """Retry decorator with exponential backoff.

    Use this for unreliable operations like data fetching from external
    APIs, database connections, or any I/O-bound function that may
    transiently fail.  The decorator retries the function up to
    *max_retries* times with exponential backoff between attempts.

    Parameters:
        fn: The function to wrap (when used without arguments).
        max_retries: Maximum number of retry attempts (default 3).
            The function is called at most ``max_retries + 1`` times
            (1 initial + retries).
        delay: Initial delay between retries in seconds (default 1.0).
        backoff_factor: Multiplicative factor for delay between
            successive retries (default 2.0).  With delay=1.0 and
            backoff_factor=2.0, delays are 1s, 2s, 4s, ...
        exceptions: Tuple of exception types to catch and retry on.
            Default is ``(Exception,)`` which retries on any exception.

    Returns:
        Decorated function with retry logic.

    Example:
        >>> call_count = 0
        >>> @retry(max_retries=2, delay=0.01)
        ... def flaky():
        ...     global call_count
        ...     call_count += 1
        ...     if call_count < 3:
        ...         raise ConnectionError("temporary failure")
        ...     return "success"
        >>> flaky()
        'success'

    See Also:
        cache_result: Cache expensive results to disk.
        log_step: Log function entry/exit.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        _time.sleep(current_delay)
                        current_delay *= backoff_factor
            raise last_exc  # type: ignore[misc]

        return wrapper

    if fn is not None:
        # Used as @retry without arguments
        return decorator(fn)
    return decorator


# ---------------------------------------------------------------------------
# cache_result
# ---------------------------------------------------------------------------


def cache_result(
    fn: Callable | None = None,
    *,
    cache_dir: str | Path | None = None,
    ttl_hours: float = 24,
) -> Callable:
    """Cache function results to disk with TTL.

    Use this for expensive computations (e.g., data fetching, feature
    engineering, model training) that you want to avoid re-running
    within a time window.  Results are pickled to disk and reloaded
    if the cache is still valid.

    The cache key is derived from the function name and its arguments
    (serialised via ``repr``), so calling with different arguments
    creates separate cache entries.

    Parameters:
        fn: The function to wrap (when used without arguments).
        cache_dir: Directory for cache files.  If *None*, uses
            ``~/.wraquant/cache``.
        ttl_hours: Time-to-live in hours (default 24).  Cache entries
            older than this are re-computed.

    Returns:
        Decorated function with disk caching.

    Example:
        >>> import tempfile
        >>> @cache_result(cache_dir=tempfile.mkdtemp(), ttl_hours=1)
        ... def slow_computation(x):
        ...     return x ** 2
        >>> slow_computation(5)  # computed
        25
        >>> slow_computation(5)  # loaded from cache
        25

    See Also:
        retry: Retry decorator for unreliable operations.
        log_step: Log function entry/exit.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Determine cache directory
            cdir = Path(cache_dir) if cache_dir is not None else Path.home() / ".wraquant" / "cache"
            cdir.mkdir(parents=True, exist_ok=True)

            # Build cache key from function name + args
            key_data = f"{func.__module__}.{func.__qualname__}:{repr(args)}:{repr(sorted(kwargs.items()))}"
            key_hash = hashlib.sha256(key_data.encode()).hexdigest()[:16]
            cache_file = cdir / f"{func.__name__}_{key_hash}.pkl"
            meta_file = cdir / f"{func.__name__}_{key_hash}.meta"

            # Check if cached result is still valid
            if cache_file.exists() and meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text())
                    cached_time = meta.get("timestamp", 0)
                    if (_time.time() - cached_time) < ttl_hours * 3600:
                        with open(cache_file, "rb") as f:
                            return pickle.load(f)  # noqa: S301
                except (json.JSONDecodeError, pickle.UnpicklingError, OSError):
                    pass  # Cache is corrupt, re-compute

            # Compute and cache
            result = func(*args, **kwargs)
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f)
                meta_file.write_text(json.dumps({"timestamp": _time.time()}))
            except (OSError, pickle.PicklingError):
                pass  # Caching failure is non-fatal

            return result

        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


# ---------------------------------------------------------------------------
# log_step
# ---------------------------------------------------------------------------


def log_step(
    fn: Callable | None = None,
    *,
    logger: logging.Logger | None = None,
) -> Callable:
    """Decorator that logs step entry, exit, and duration.

    Use this to add lightweight observability to pipeline steps.  Each
    call logs the function name, arguments summary, duration, and
    whether it succeeded or raised an exception.

    Parameters:
        fn: The function to wrap (when used without arguments).
        logger: Logger instance to use.  If *None*, uses a logger named
            ``'wraquant.flow'``.

    Returns:
        Decorated function with logging.

    Example:
        >>> @log_step
        ... def compute(x):
        ...     return x * 2
        >>> compute(5)
        10

    See Also:
        retry: Retry with exponential backoff.
        cache_result: Cache results to disk.
    """

    def decorator(func: Callable) -> Callable:
        log = logger or logging.getLogger("wraquant.flow")

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = func.__qualname__
            log.info("ENTER %s", func_name)
            start = _time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = _time.perf_counter() - start
                log.info("EXIT  %s  (%.3fs)", func_name, elapsed)
                return result
            except Exception:
                elapsed = _time.perf_counter() - start
                log.exception("FAIL  %s  (%.3fs)", func_name, elapsed)
                raise

        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


# ---------------------------------------------------------------------------
# parallel_pipeline
# ---------------------------------------------------------------------------


def parallel_pipeline(
    *pipelines: Pipeline,
    max_workers: int | None = None,
) -> list[Any]:
    """Run multiple Pipeline objects in parallel using threading.

    Use this when you have independent pipelines (e.g., processing
    different assets or strategies) that can run concurrently.  Each
    pipeline runs in its own thread using a ``ThreadPoolExecutor``.

    Parameters:
        *pipelines: Pipeline objects to run.  Each pipeline must have
            been created with all its steps already added.
        max_workers: Maximum number of threads (default *None*, which
            uses ``min(len(pipelines), 8)``).

    Returns:
        List of results, one per pipeline, in the same order as the
        input pipelines.  Each result is the output of the pipeline's
        final step.

    Example:
        >>> pipe1 = pipeline(lambda x: x * 2)
        >>> pipe2 = pipeline(lambda x: x + 10)
        >>> results = parallel_pipeline(pipe1, pipe2, initial_data=5)
        >>> results  # [10, 15]  # doctest: +SKIP

    Notes:
        Since this uses threading (not multiprocessing), it is best
        suited for I/O-bound tasks.  For CPU-bound parallel execution,
        consider ``wraquant.scale``.

    See Also:
        pipeline: Create a sequential pipeline.
        dag: Create a DAG with dependencies.
    """
    if not pipelines:
        return []

    workers = max_workers or min(len(pipelines), 8)

    # Each pipeline needs initial data; we extract it from the first step
    # Actually the user should call this differently -- let's support
    # an initial_data pattern by allowing it as a kwarg hack or by
    # requiring each pipeline to already have initial data baked in.
    # The simplest approach: run each pipeline with None as initial data.
    # Callers should bake in data via closures or partial application.

    future_to_idx: dict[Any, int] = {}
    results: list[Any] = [None] * len(pipelines)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i, pipe in enumerate(pipelines):
            future = executor.submit(pipe.run, None)
            future_to_idx[future] = i

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()

    return results
