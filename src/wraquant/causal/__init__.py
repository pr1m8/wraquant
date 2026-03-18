"""Causal inference methods for quantitative finance.

Provides treatment effect estimation (propensity scores, IPW, matching,
doubly robust, regression discontinuity, synthetic control, diff-in-diff)
and wrappers for external causal inference packages (DoWhy, EconML, DoubleML).
"""

from wraquant.causal.treatment import (
    diff_in_diff,
    doubly_robust_ate,
    ipw_ate,
    matching_ate,
    propensity_score,
    regression_discontinuity,
    synthetic_control,
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
]
