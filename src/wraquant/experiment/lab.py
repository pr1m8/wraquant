"""The wraquant Experiment Lab -- systematic strategy research platform.

The Lab provides a framework for running, tracking, comparing, and
analyzing quantitative trading strategies across parameter grids,
cross-validation folds, and market regimes.

The workflow:
    1. Create a Lab (organizes experiments by name).
    2. Create an Experiment (defines strategy + parameter grid + data).
    3. Run the experiment (sweeps all combos across CV folds).
    4. Analyze results (best/worst, stability, sensitivity, plots).
    5. Persist results for later comparison.

Example:
    >>> from wraquant.experiment import Lab
    >>> lab = Lab("momentum_research")
    >>> exp = lab.create("rsi_strategy",
    ...     strategy_fn=my_strategy,
    ...     params={"period": [7, 14, 21], "threshold": [20, 30]},
    ...     data=prices,
    ... )
    >>> results = exp.run(cv="walk_forward", n_splits=5)
    >>> print(results.best())
    >>> results.plot_parameter_heatmap("period", "threshold")

Design Principles:
    - **Single source of truth**: metrics come from ``risk.metrics`` and
      ``backtest.metrics``, not reimplemented here.
    - **Minimal dependencies**: only numpy + pandas required; joblib for
      parallelism and plotly for visualization are optional.
    - **Persistence**: results save as JSON + Parquet; no database needed.
    - **Composable**: each component (Lab, Experiment, Results, Store, CV)
      works independently.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import pandas as pd

from wraquant.experiment.results import ExperimentResults
from wraquant.experiment.runner import ExperimentRunner
from wraquant.experiment.tracking import ExperimentStore

logger = logging.getLogger(__name__)


class Experiment:
    """A single experiment testing a strategy across a parameter grid.

    An Experiment encapsulates a strategy function, a parameter grid,
    and the data to test on.  Call ``.run()`` to execute all parameter
    combinations with cross-validation and get an ``ExperimentResults``
    object for analysis.

    Parameters:
        name: Experiment name for tracking and persistence.
        strategy_fn: Callable(data, **params) -> pd.Series of returns.
            The function receives a data slice (from CV splitting) and
            keyword arguments from the parameter grid.
        params: Dictionary mapping parameter names to lists of values
            to test.  Example: ``{"period": [7, 14, 21]}``.
        data: Price or return data (pd.Series or pd.DataFrame).
        benchmark: Optional benchmark return series for relative metrics.
        lab: Optional reference to the parent Lab for auto-persistence.
    """

    def __init__(
        self,
        name: str,
        strategy_fn: Callable[..., pd.Series],
        params: dict[str, list[Any]],
        data: pd.Series | pd.DataFrame,
        benchmark: pd.Series | None = None,
        lab: Lab | None = None,
    ) -> None:
        self.name = name
        self.strategy_fn = strategy_fn
        self.params = params
        self.data = data
        self.benchmark = benchmark
        self._lab = lab

    def grid_size(self) -> int:
        """Total number of parameter combinations.

        Returns:
            Product of the lengths of all parameter value lists.

        Example:
            >>> exp = Experiment("test", fn, {"a": [1, 2], "b": [3, 4, 5]}, data)
            >>> exp.grid_size()
            6
        """
        n = 1
        for values in self.params.values():
            n *= len(values)
        return n

    def run(
        self,
        cv: str = "walk_forward",
        n_splits: int = 5,
        parallel: bool = True,
        n_jobs: int = -1,
    ) -> ExperimentResults:
        """Run all parameter combinations with cross-validation.

        This is the main entry point.  It creates an ExperimentRunner,
        executes the full grid, wraps results in an ExperimentResults
        container, and optionally persists them via the Lab's store.

        Parameters:
            cv: Cross-validation method:
                - ``"walk_forward"`` -- expanding train, fixed test
                  (default, most realistic for live trading).
                - ``"rolling"`` -- fixed-size rolling window (good when
                  old data may hurt due to regime changes).
                - ``"purged_kfold"`` -- purged K-fold with embargo (good
                  for overlapping labels).
                - ``"none"`` -- full in-sample, no CV (for quick
                  prototyping only; results will be overfit).
            n_splits: Number of CV splits.  More splits = more robust
                estimates but slower execution.
            parallel: Use joblib for parallel execution if available.
            n_jobs: Number of parallel jobs (-1 = all CPUs).

        Returns:
            ExperimentResults object with analysis and visualization
            methods.
        """
        logger.info(
            "Running experiment '%s': %d param combos x %d folds (%s CV)",
            self.name,
            self.grid_size(),
            n_splits if cv != "none" else 1,
            cv,
        )

        runner = ExperimentRunner(
            strategy_fn=self.strategy_fn,
            data=self.data,
            params=self.params,
            benchmark=self.benchmark,
        )

        runs = runner.run_grid(
            cv=cv,
            n_splits=n_splits,
            parallel=parallel,
            n_jobs=n_jobs,
        )

        results = ExperimentResults(
            runs=runs,
            experiment_name=self.name,
            params=self.params,
            benchmark=self.benchmark,
        )

        # Auto-persist if Lab is available
        if self._lab is not None:
            try:
                self._lab._store.save_experiment(self.name, results)
                logger.info("Results auto-saved to %s", self._lab._store.storage_dir)
            except Exception as exc:
                logger.warning("Failed to auto-save results: %s", exc)

        return results


class Lab:
    """Experiment laboratory for systematic strategy research.

    The Lab is the top-level organizer.  It creates experiments, persists
    results, and enables cross-experiment comparison.

    Parameters:
        name: Lab name for organizing experiments.
        storage_dir: Directory for persisting results.
            Default: ``./experiments/``.

    Example:
        >>> lab = Lab("factor_research", storage_dir="./my_experiments")
        >>> exp = lab.create("momentum_v1",
        ...     strategy_fn=momentum_strategy,
        ...     params={"lookback": [20, 60, 120]},
        ...     data=prices,
        ... )
        >>> results = exp.run()
        >>> lab.list_experiments()
    """

    def __init__(
        self,
        name: str,
        storage_dir: str = "./experiments/",
    ) -> None:
        self.name = name
        self._store = ExperimentStore(storage_dir)

    def create(
        self,
        name: str,
        strategy_fn: Callable[..., pd.Series],
        params: dict[str, list[Any]],
        data: pd.Series | pd.DataFrame,
        benchmark: pd.Series | None = None,
        **kwargs: Any,
    ) -> Experiment:
        """Create a new experiment.

        Parameters:
            name: Experiment name (must be unique within the Lab).
            strategy_fn: Strategy function.  Must accept
                ``(data, **params)`` and return a ``pd.Series`` of
                per-period returns.
            params: Parameter grid as ``{name: [values]}``.
            data: Price or return data.
            benchmark: Optional benchmark return series.
            **kwargs: Additional keyword arguments (reserved for future).

        Returns:
            Experiment object ready to ``.run()``.
        """
        return Experiment(
            name=name,
            strategy_fn=strategy_fn,
            params=params,
            data=data,
            benchmark=benchmark,
            lab=self,
        )

    def list_experiments(self) -> pd.DataFrame:
        """List all past experiments.

        Returns:
            DataFrame with experiment name, date, run count, and
            best Sharpe ratio.
        """
        return self._store.list_experiments()

    def load(self, name: str) -> ExperimentResults:
        """Load a past experiment's results.

        Parameters:
            name: Experiment name.

        Returns:
            ExperimentResults loaded from disk.
        """
        return self._store.load_experiment(name)

    def compare(self, experiment_names: list[str]) -> pd.DataFrame:
        """Compare results across multiple experiments.

        Parameters:
            experiment_names: List of experiment names to compare.

        Returns:
            DataFrame with one row per experiment, showing best
            parameters and metrics.
        """
        return self._store.compare_experiments(experiment_names)

    def delete(self, name: str) -> None:
        """Delete an experiment and its results.

        Parameters:
            name: Experiment name.
        """
        self._store.delete_experiment(name)


__all__ = [
    "Experiment",
    "Lab",
]
