"""Causal inference methods for quantitative finance.

Provides tools for estimating causal effects in financial settings --
where randomized experiments are rare and observational data dominates.
Covers the major causal identification strategies used in empirical
finance and economics: propensity score methods for selection-on-
observables designs, difference-in-differences for policy evaluation,
synthetic control for single-treated-unit studies, regression
discontinuity for threshold-based effects, and instrumental variables
for endogeneity correction.

Key components:

- **Propensity score methods** -- ``propensity_score`` (logistic model),
  ``ipw_ate`` (Inverse Probability Weighting for Average Treatment
  Effect), ``matching_ate`` (nearest-neighbor propensity matching),
  ``doubly_robust_ate`` (combines outcome model with IPW for double
  protection against misspecification).
- **Difference-in-differences** -- ``diff_in_diff`` for estimating
  treatment effects from panel data with parallel trends.
- **Synthetic control** -- ``synthetic_control`` (Abadie et al. method
  for constructing a counterfactual from donor units),
  ``synthetic_control_weights`` (retrieve the donor weights).
- **Regression discontinuity** -- ``regression_discontinuity`` (sharp
  RDD), ``regression_discontinuity_robust`` (bias-corrected with
  robust confidence intervals).
- **Granger causality** -- ``granger_causality`` for testing predictive
  causality in time series.
- **Instrumental variables** -- ``instrumental_variable`` (2SLS for
  endogenous regressors).
- **Event studies** -- ``event_study`` for estimating dynamic treatment
  effects around an event.
- **Advanced methods** -- ``causal_forest`` (heterogeneous treatment
  effects via random forests), ``mediation_analysis`` (decompose
  direct and indirect effects), ``bounds_analysis`` (partial
  identification under weaker assumptions).

Example:
    >>> from wraquant.causal import diff_in_diff, synthetic_control
    >>> att = diff_in_diff(panel, y="returns", treat="treated", post="post_event")
    >>> sc = synthetic_control(treated_series, donor_matrix, pre_periods=60)

Use ``wraquant.causal`` when you need to establish causal relationships
rather than mere correlations -- for example, measuring the effect of
a policy change on asset prices or estimating the impact of an index
inclusion event.  For standard regression without causal identification,
see ``wraquant.stats.regression`` or ``wraquant.econometrics``.
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
