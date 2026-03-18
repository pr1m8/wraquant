"""Machine learning utilities for quantitative finance.

This module provides feature engineering, preprocessing, model wrappers,
clustering, and evaluation tools tailored for financial machine-learning
workflows.
"""

from __future__ import annotations

from wraquant.ml.clustering import (
    correlation_clustering,
    optimal_clusters,
    regime_clustering,
)
from wraquant.ml.evaluation import (
    backtest_predictions,
    classification_metrics,
    financial_metrics,
    learning_curve,
)
from wraquant.ml.features import (
    label_fixed_horizon,
    label_triple_barrier,
    microstructure_features,
    return_features,
    rolling_features,
    technical_features,
    volatility_features,
)
from wraquant.ml.models import (
    ensemble_predict,
    feature_importance_mda,
    feature_importance_mdi,
    sequential_feature_selection,
    walk_forward_train,
)
from wraquant.ml.preprocessing import (
    combinatorial_purged_kfold,
    denoised_correlation,
    detoned_correlation,
    fractional_differentiation,
    purged_kfold,
)

__all__ = [
    # features
    "rolling_features",
    "return_features",
    "technical_features",
    "volatility_features",
    "microstructure_features",
    "label_fixed_horizon",
    "label_triple_barrier",
    # preprocessing
    "purged_kfold",
    "combinatorial_purged_kfold",
    "fractional_differentiation",
    "denoised_correlation",
    "detoned_correlation",
    # models
    "walk_forward_train",
    "ensemble_predict",
    "feature_importance_mdi",
    "feature_importance_mda",
    "sequential_feature_selection",
    # clustering
    "correlation_clustering",
    "regime_clustering",
    "optimal_clusters",
    # evaluation
    "classification_metrics",
    "financial_metrics",
    "learning_curve",
    "backtest_predictions",
]
