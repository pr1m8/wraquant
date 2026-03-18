"""Causal inference methods for quantitative finance.

Provides treatment effect estimation (propensity scores, IPW, matching,
doubly robust, regression discontinuity, synthetic control, diff-in-diff,
Granger causality, instrumental variables, event studies, causal forests,
mediation analysis, robust RDD, and partial identification bounds) and
wrappers for external causal inference packages (DoWhy, EconML, DoubleML).
"""

from wraquant.causal.treatment import (
    bounds_analysis,
    causal_forest,
    diff_in_diff,
    doubly_robust_ate,
    event_study,
    granger_causality,
    instrumental_variable,
    ipw_ate,
    matching_ate,
    mediation_analysis,
    propensity_score,
    regression_discontinuity,
    regression_discontinuity_robust,
    synthetic_control,
    synthetic_control_weights,
)

__all__ = [
    # treatment.py — pure numpy/scipy implementations
    "propensity_score",
    "ipw_ate",
    "matching_ate",
    "doubly_robust_ate",
    "regression_discontinuity",
    "synthetic_control",
    "diff_in_diff",
    # New deep implementations
    "granger_causality",
    "instrumental_variable",
    "event_study",
    "synthetic_control_weights",
    "causal_forest",
    "mediation_analysis",
    "regression_discontinuity_robust",
    "bounds_analysis",
]
