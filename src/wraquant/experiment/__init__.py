"""The wraquant Experiment Lab -- systematic strategy research platform.

The experiment module provides two layers:

**High-level Lab API** (new):
    ``Lab``, ``Experiment``, ``ExperimentResults``, ``ExperimentStore``
    -- a complete research platform for running, tracking, comparing,
    and analyzing strategies across parameter grids and CV folds.

**Low-level utilities** (existing):
    ``ParameterGrid``, ``grid_search``, ``random_search``,
    ``walk_forward_optimize``, ``parameter_sensitivity``, etc.
    -- building blocks for custom optimization workflows.
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
