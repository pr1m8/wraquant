"""Workflow orchestration for wraquant.

Provides wrappers for common workflow patterns using Prefect, Dagster,
and APScheduler for scheduling recurring quantitative finance tasks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wraquant.core.decorators import requires_extra

__all__ = [
    "prefect_backtest_flow",
    "schedule_data_refresh",
    "pipeline",
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
