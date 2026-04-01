"""Econometric methods for quantitative finance.

Provides rigorous econometric estimators and tests grounded in the
academic finance literature.  Covers panel data methods for multi-firm
studies, cross-sectional regressions for asset pricing tests, time series
models for macro-financial linkages, volatility modeling via the GARCH
family, regression diagnostics for model validation, and event study
methodology for measuring abnormal returns around corporate actions.

Key sub-modules:

- **Panel data** (``panel``) -- ``fixed_effects``, ``random_effects``,
  ``pooled_ols``, ``between_effects``, ``first_difference``, and
  ``hausman_test`` for choosing between FE and RE.  Use for multi-firm,
  multi-period studies (e.g., Fama-MacBeth cross-sections, earnings
  announcements across firms).
- **Cross-section** (``cross_section``) -- ``robust_ols`` (White or HAC
  standard errors), ``quantile_regression``,
  ``two_stage_least_squares`` (IV/2SLS for endogeneity),
  ``gmm_estimation``, and ``sargan_test`` (over-identification).
- **Time series** (``timeseries``) -- ``var_model`` (Vector
  Autoregression), ``vecm_model`` (Vector Error Correction for
  cointegrated systems), ``granger_causality``, ``impulse_response``,
  ``variance_decomposition``, and ``structural_break_test``.
- **Volatility** (``volatility``) -- ``garch``, ``egarch``,
  ``gjr_garch``, ``dcc_garch``, ``arch_test`` (test for ARCH effects),
  and ``garch_numpy_fallback`` (pure-numpy fallback when arch is
  unavailable).
- **Diagnostics** (``diagnostics``) -- ``durbin_watson``,
  ``breusch_godfrey`` (serial correlation), ``breusch_pagan`` /
  ``white_test`` (heteroskedasticity), ``jarque_bera`` (normality),
  ``ramsey_reset`` (functional form), ``vif`` (multicollinearity),
  and ``condition_number``.
- **Event study** (``event_study``) -- ``event_study`` (market model
  estimation), ``cumulative_abnormal_return`` (CAR), and
  ``buy_and_hold_abnormal_return`` (BHAR).

Example:
    >>> from wraquant.econometrics import fixed_effects, granger_causality
    >>> fe_result = fixed_effects(panel_df, y="returns", x=["beta", "size"])
    >>> gc = granger_causality(gdp_growth, sp500_returns, max_lag=4)

Use ``wraquant.econometrics`` for formal econometric analysis (panel
regressions, IV estimation, event studies, VAR/VECM).  For simpler OLS
and rolling regression, see ``wraquant.stats.regression``.  For GARCH
with full diagnostics and forecasting, prefer ``wraquant.vol``.
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
