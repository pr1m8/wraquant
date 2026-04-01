"""The wraquant Experiment Lab -- systematic strategy research platform.

Provides a structured framework for running, tracking, comparing, and
analyzing trading strategies and models across parameter grids, cross-
validation folds, and walk-forward windows.  Designed to bring
experiment-tracking discipline (like MLflow or Weights & Biases) to
quantitative finance research, ensuring reproducibility and preventing
the common pitfall of p-hacking through untracked parameter sweeps.

The module provides two layers:

**High-level Lab API**:
    ``Lab``, ``Experiment``, ``ExperimentResults``, ``ExperimentStore``
    -- a complete research platform for running, tracking, comparing,
    and analyzing strategies across parameter grids and CV folds.

**Low-level utilities**:
    ``ParameterGrid``, ``grid_search``, ``random_search``,
    ``walk_forward_optimize``, ``parameter_sensitivity``, etc.
    -- building blocks for custom optimization workflows.

Key components:

- **Lab / Experiment** -- Define experiments with parameter grids,
  run them with automatic result capture, and compare outcomes across
  configurations.  ``ExperimentStore`` persists results to disk for
  later analysis.
- **CV methods** -- ``walk_forward_splits``, ``rolling_splits``,
  ``purged_kfold_splits``, ``combinatorial_purged_splits`` for
  generating time-aware train/test splits without lookahead bias.
- **Grid search** -- ``grid_search`` (exhaustive), ``random_search``
  (randomized), and ``walk_forward_optimize`` (rolling optimization)
  for parameter tuning.
- **Sensitivity analysis** -- ``parameter_sensitivity`` (one-at-a-time),
  ``parameter_heatmap`` (two-parameter interaction),
  ``robustness_check``, and ``stability_score`` for understanding how
  fragile a strategy is to parameter perturbation.

Example:
    >>> from wraquant.experiment import Lab, grid_search, parameter_sensitivity
    >>> lab = Lab("momentum_study")
    >>> exp = lab.create_experiment(strategy_fn, param_grid)
    >>> results = exp.run(prices)
    >>> results.best_params

Use ``wraquant.experiment`` for systematic strategy research and parameter
optimization.  For parallel execution of parameter sweeps, see
``wraquant.scale.parallel_backtest``.  For the backtesting engine itself,
see ``wraquant.backtest``.
"""

from __future__ import annotations

# New high-level Lab API
from wraquant.experiment.cv import (
    combinatorial_purged_splits,
    purged_kfold_splits,
    rolling_splits,
    walk_forward_splits,
)
from wraquant.experiment.lab import Experiment, Lab
from wraquant.experiment.results import ExperimentResults
from wraquant.experiment.runner import ExperimentRunner, RunResult
from wraquant.experiment.tracking import ExperimentStore

# Existing low-level utilities (backward compatible)
from wraquant.experiment.grid import ParameterGrid, grid_search, random_search
from wraquant.experiment.sensitivity import (
    parameter_heatmap,
    parameter_sensitivity,
    robustness_check,
    stability_score,
)
from wraquant.experiment.tracker import Experiment as TrackerExperiment
from wraquant.experiment.tracker import Run
from wraquant.experiment.walk_forward import (
    expanding_window_splits,
    rolling_window_splits,
    walk_forward_optimize,
)

__all__ = [
    # High-level Lab API
    "Lab",
    "Experiment",
    "ExperimentResults",
    "ExperimentRunner",
    "ExperimentStore",
    "RunResult",
    # CV methods
    "walk_forward_splits",
    "rolling_splits",
    "purged_kfold_splits",
    "combinatorial_purged_splits",
    # Existing utilities
    "ParameterGrid",
    "Run",
    "TrackerExperiment",
    "expanding_window_splits",
    "grid_search",
    "parameter_heatmap",
    "parameter_sensitivity",
    "random_search",
    "robustness_check",
    "rolling_window_splits",
    "stability_score",
    "walk_forward_optimize",
]
