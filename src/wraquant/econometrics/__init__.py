"""Econometric methods for quantitative finance.

Covers panel data estimation, cross-sectional econometrics, time series
econometrics (VAR/VECM), volatility modeling (GARCH family), regression
diagnostics, and event study methodology.
"""

from wraquant.econometrics.cross_section import (
    gmm_estimation,
    quantile_regression,
    robust_ols,
    sargan_test,
    two_stage_least_squares,
)
from wraquant.econometrics.diagnostics import (
    breusch_godfrey,
    breusch_pagan,
    condition_number,
    durbin_watson,
    jarque_bera,
    ramsey_reset,
    vif,
    white_test,
)
from wraquant.econometrics.event_study import (
    buy_and_hold_abnormal_return,
    cumulative_abnormal_return,
    event_study,
)
from wraquant.econometrics.panel import (
    between_effects,
    first_difference,
    fixed_effects,
    hausman_test,
    pooled_ols,
    random_effects,
)
from wraquant.econometrics.timeseries import (
    granger_causality,
    impulse_response,
    structural_break_test,
    var_model,
    variance_decomposition,
    vecm_model,
)
from wraquant.econometrics.volatility import (
    arch_test,
    dcc_garch,
    egarch,
    garch,
    garch_numpy_fallback,
    gjr_garch,
)

__all__ = [
    # cross_section
    "robust_ols",
    "quantile_regression",
    "two_stage_least_squares",
    "gmm_estimation",
    "sargan_test",
    # diagnostics
    "durbin_watson",
    "breusch_godfrey",
    "breusch_pagan",
    "white_test",
    "jarque_bera",
    "ramsey_reset",
    "vif",
    "condition_number",
    # event_study
    "event_study",
    "cumulative_abnormal_return",
    "buy_and_hold_abnormal_return",
    # panel
    "pooled_ols",
    "fixed_effects",
    "random_effects",
    "hausman_test",
    "between_effects",
    "first_difference",
    # timeseries
    "var_model",
    "vecm_model",
    "granger_causality",
    "impulse_response",
    "variance_decomposition",
    "structural_break_test",
    # volatility
    "garch",
    "garch_numpy_fallback",
    "egarch",
    "gjr_garch",
    "dcc_garch",
    "arch_test",
]
