"""Simple file-based experiment tracking.

Provides an ``Experiment`` container and ``Run`` objects for logging
parameters, metrics, and artifacts to disk without any external
dependencies.
"""

from __future__ import annotations

import json
import pickle  # noqa: S403 — internal experiment artifact serialization
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import pandas as pd


class Run:
    """An individual experiment run.

    Parameters
    ----------
    run_id : str
        Unique identifier for this run.
    run_dir : Path
        Directory where run data is stored.
    params : dict[str, Any] | None, optional
        Initial parameters to log.  Default is ``None``.
    """

    def __init__(
        self,
        run_id: str,
        run_dir: Path,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.run_id = run_id
        self.run_dir = run_dir
        self._params: dict[str, Any] = {}
        self._metrics: dict[str, Any] = {}
        self._start_time: float = time.time()
        self._end_time: float | None = None

        self.run_dir.mkdir(parents=True, exist_ok=True)

        if params is not None:
            for k, v in params.items():
                self.log_param(k, v)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter.

        Parameters
        ----------
        key : str
            Parameter name.
        value : Any
            Parameter value (must be JSON-serialisable).
        """
        self._params[key] = value

    def log_metric(self, key: str, value: Any) -> None:
        """Log a metric.

        Parameters
        ----------
        key : str
            Metric name.
        value : Any
            Metric value (must be JSON-serialisable).
        """
        self._metrics[key] = value

    def log_artifact(self, name: str, data: Any) -> None:
        """Save arbitrary data as a pickle artifact.

        Parameters
        ----------
        name : str
            Artifact name (used as the filename stem).
        data : Any
            Object to persist.
        """
        artifact_path = self.run_dir / f"{name}.pkl"
        with open(artifact_path, "wb") as f:
            pickle.dump(data, f)

    def to_dict(self) -> dict[str, Any]:
        """Return all run information as a dictionary.

        Returns
        -------
        dict[str, Any]
            Contains ``run_id``, ``params``, ``metrics``,
            ``start_time``, and ``end_time``.
        """
        return {
            "run_id": self.run_id,
            "params": dict(self._params),
            "metrics": dict(self._metrics),
            "start_time": self._start_time,
            "end_time": self._end_time,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _finish(self) -> None:
        """Finalise the run and persist metadata to disk."""
        self._end_time = time.time()
        meta_path = self.run_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class Experiment:
    """Track and log experiment runs to disk.

    Parameters
    ----------
    name : str
        Human-readable experiment name.  Also used as the subdirectory
        under ``base_dir``.
    base_dir : str, optional
        Root directory for experiment storage.  Default is
        ``'.experiments'``.
    """

    def __init__(self, name: str, base_dir: str = ".experiments") -> None:
        self.name = name
        self.base_dir = Path(base_dir)
        self.experiment_dir = self.base_dir / name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def start_run(
        self, params: dict[str, Any] | None = None
    ) -> Generator[Run, None, None]:
        """Start a new tracked run.

        Parameters
        ----------
        params : dict[str, Any] | None, optional
            Initial parameters to log. Default is ``None``.

        Yields
        ------
        Run
            The active run object for logging params, metrics, and
            artifacts.
        """
        run_id = uuid.uuid4().hex[:12]
        run_dir = self.experiment_dir / run_id
        run = Run(run_id=run_id, run_dir=run_dir, params=params)
        try:
            yield run
        finally:
            run._finish()

    def list_runs(self) -> list[dict[str, Any]]:
        """List all completed runs for this experiment.

        Returns
        -------
        list[dict[str, Any]]
            Each entry contains ``run_id``, ``params``, ``metrics``,
            ``start_time``, and ``end_time``.
        """
        runs: list[dict[str, Any]] = []
        for meta_path in sorted(self.experiment_dir.rglob("meta.json")):
            with open(meta_path) as f:
                runs.append(json.load(f))
        return runs

    def best_run(
        self, metric: str = "sharpe_ratio", maximize: bool = True
    ) -> dict[str, Any]:
        """Find the best run according to a metric.

        Parameters
        ----------
        metric : str, optional
            Name of the metric to optimise.  Default is
            ``'sharpe_ratio'``.
        maximize : bool, optional
            If ``True`` (default), the run with the highest metric value
            is returned; otherwise the lowest.

        Returns
        -------
        dict[str, Any]
            The best run's metadata dictionary.

        Raises
        ------
        ValueError
            If no runs have been recorded or no run contains the
            requested metric.
        """
        runs = self.list_runs()
        if not runs:
            raise ValueError("No runs have been recorded.")

        scored = [r for r in runs if metric in r.get("metrics", {})]
        if not scored:
            raise ValueError(f"No runs contain the metric {metric!r}.")

        return (
            max(scored, key=lambda r: r["metrics"][metric])
            if maximize
            else min(scored, key=lambda r: r["metrics"][metric])
        )

    def compare_runs(self, metric_names: list[str] | None = None) -> pd.DataFrame:
        """Create a comparison table of all runs.

        Parameters
        ----------
        metric_names : list[str] | None, optional
            Subset of metric names to include.  If ``None``, all
            metrics found across runs are included.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by ``run_id`` with columns for each
            parameter and metric.
        """
        runs = self.list_runs()
        if not runs:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for r in runs:
            row: dict[str, Any] = {"run_id": r["run_id"]}
            for k, v in r.get("params", {}).items():
                row[f"param_{k}"] = v
            metrics = r.get("metrics", {})
            if metric_names is not None:
                metrics = {k: v for k, v in metrics.items() if k in metric_names}
            for k, v in metrics.items():
                row[f"metric_{k}"] = v
            rows.append(row)

        df = pd.DataFrame(rows)
        if "run_id" in df.columns:
            df = df.set_index("run_id")
        return df
