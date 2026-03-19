"""Experiment persistence and retrieval.

Provides ``ExperimentStore`` for saving, loading, listing, comparing,
and deleting experiment results on disk.  Uses JSON for metadata and
Parquet for time series data -- no database dependency required.

Storage layout::

    storage_dir/
        experiment_name_1/
            metadata.json
            summary.parquet
            all_runs.parquet
        experiment_name_2/
            ...
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from wraquant.experiment.results import ExperimentResults


class ExperimentStore:
    """Persist and retrieve experiment results.

    Uses JSON for metadata and Parquet for time series data.
    No database dependency required.

    Parameters:
        storage_dir: Root directory for persisting experiments.
            Defaults to ``./experiments/``.

    Example:
        >>> store = ExperimentStore("./my_experiments")
        >>> store.save_experiment("rsi_test", results)
        >>> store.list_experiments()
        >>> loaded = store.load_experiment("rsi_test")
    """

    def __init__(self, storage_dir: str | Path = "./experiments/") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_experiment(self, name: str, results: ExperimentResults) -> Path:
        """Save experiment results to disk.

        Parameters:
            name: Experiment name (used as subdirectory name).
            results: ExperimentResults to persist.

        Returns:
            Path to the saved experiment directory.
        """
        exp_dir = self.storage_dir / name
        results.save(exp_dir)
        return exp_dir

    def load_experiment(self, name: str) -> ExperimentResults:
        """Load experiment results from disk.

        Parameters:
            name: Experiment name.

        Returns:
            Reconstituted ExperimentResults.

        Raises:
            FileNotFoundError: If the experiment does not exist.
        """
        exp_dir = self.storage_dir / name
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment '{name}' not found at {exp_dir}")
        return ExperimentResults.load(exp_dir)

    def list_experiments(self) -> pd.DataFrame:
        """List all saved experiments.

        Returns:
            DataFrame with columns: name, date_modified, n_runs,
            n_param_combos, best_sharpe.
        """
        rows: list[dict[str, Any]] = []

        for exp_dir in sorted(self.storage_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            meta_path = exp_dir / "metadata.json"
            if not meta_path.exists():
                continue

            with open(meta_path) as f:
                meta = json.load(f)

            # Extract best sharpe from run summaries
            best_sharpe = 0.0
            run_summaries = meta.get("run_summaries", [])
            for run_data in run_summaries:
                sharpe = run_data.get("metrics", {}).get("sharpe", 0.0)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe

            # Count unique param combos
            param_keys = set()
            for run_data in run_summaries:
                key = tuple(sorted(run_data.get("params", {}).items()))
                param_keys.add(key)

            rows.append(
                {
                    "name": meta.get("experiment_name", exp_dir.name),
                    "date_modified": pd.Timestamp.fromtimestamp(
                        meta_path.stat().st_mtime
                    ),
                    "n_runs": meta.get("n_runs", 0),
                    "n_param_combos": len(param_keys),
                    "best_sharpe": best_sharpe,
                }
            )

        return pd.DataFrame(rows)

    def delete_experiment(self, name: str) -> None:
        """Delete an experiment and its results from disk.

        Parameters:
            name: Experiment name.

        Raises:
            FileNotFoundError: If the experiment does not exist.
        """
        exp_dir = self.storage_dir / name
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment '{name}' not found at {exp_dir}")
        shutil.rmtree(exp_dir)

    def compare_experiments(self, names: list[str]) -> pd.DataFrame:
        """Load multiple experiments and compare their best results.

        Parameters:
            names: List of experiment names to compare.

        Returns:
            DataFrame with one row per experiment, showing the best
            parameter combination and its metrics.
        """
        rows: list[dict[str, Any]] = []
        for name in names:
            try:
                results = self.load_experiment(name)
                best = results.best()
                row: dict[str, Any] = {"experiment": name}
                row.update({f"param_{k}": v for k, v in best["params"].items()})
                row.update(best["metrics"])
                rows.append(row)
            except Exception as exc:
                rows.append({"experiment": name, "error": str(exc)})

        return pd.DataFrame(rows)


__all__ = [
    "ExperimentStore",
]
