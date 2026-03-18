"""Risk management and portfolio risk analytics.

Covers performance metrics, Value-at-Risk, portfolio risk decomposition,
scenario analysis, stress testing, copula dependency models, and
DCC-GARCH dynamic correlation.
"""

from wraquant.risk.integrations import (
    copulas_fit,
    extreme_value_analysis,
    pypfopt_efficient_frontier,
    riskfolio_portfolio,
    skfolio_optimize,
    vine_copula,
)
from wraquant.risk.copulas import (
    copula_simulate,
    fit_clayton_copula,
    fit_frank_copula,
    fit_gaussian_copula,
    fit_gumbel_copula,
    fit_t_copula,
    rank_correlation,
    tail_dependence,
)
from wraquant.risk.dcc import (
    conditional_covariance,
    dcc_garch,
    forecast_correlation,
    rolling_correlation_dcc,
)
from wraquant.risk.metrics import (
    hit_ratio,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from wraquant.risk.portfolio import (
    diversification_ratio,
    portfolio_volatility,
    risk_contribution,
)
from wraquant.risk.scenarios import monte_carlo_var, stress_test
from wraquant.risk.stress import (
    historical_stress_test,
    joint_stress_test,
    marginal_stress_contribution,
    reverse_stress_test,
    sensitivity_ladder,
    spot_stress_test,
    stress_test_returns,
    vol_stress_test,
)
from wraquant.risk.credit import (
    altman_z_score,
    cds_spread,
    credit_spread,
    default_probability,
    expected_loss,
    loss_given_default,
    merton_model,
)
from wraquant.risk.monte_carlo import (
    antithetic_variates,
    block_bootstrap,
    filtered_historical_simulation,
    importance_sampling_var,
    stationary_bootstrap,
    stratified_sampling,
)
from wraquant.risk.survival import (
    cox_partial_likelihood,
    exponential_survival,
    hazard_rate,
    kaplan_meier,
    log_rank_test,
    median_survival_time,
    nelson_aalen,
    weibull_survival,
)
from wraquant.risk.var import conditional_var, value_at_risk

__all__ = [
    # metrics
    "sharpe_ratio",
    "sortino_ratio",
    "information_ratio",
    "max_drawdown",
    "hit_ratio",
    # var
    "value_at_risk",
    "conditional_var",
    # portfolio
    "portfolio_volatility",
    "risk_contribution",
    "diversification_ratio",
    # scenarios
    "monte_carlo_var",
    "stress_test",
    # stress testing
    "stress_test_returns",
    "historical_stress_test",
    "vol_stress_test",
    "spot_stress_test",
    "sensitivity_ladder",
    "reverse_stress_test",
    "joint_stress_test",
    "marginal_stress_contribution",
    # copulas
    "fit_gaussian_copula",
    "fit_t_copula",
    "fit_clayton_copula",
    "fit_gumbel_copula",
    "fit_frank_copula",
    "copula_simulate",
    "tail_dependence",
    "rank_correlation",
    # dcc
    "dcc_garch",
    "rolling_correlation_dcc",
    "forecast_correlation",
    "conditional_covariance",
    # integrations
    "pypfopt_efficient_frontier",
    "riskfolio_portfolio",
    "skfolio_optimize",
    "copulas_fit",
    "vine_copula",
    "extreme_value_analysis",
    # credit
    "merton_model",
    "altman_z_score",
    "default_probability",
    "credit_spread",
    "loss_given_default",
    "expected_loss",
    "cds_spread",
    # survival
    "kaplan_meier",
    "nelson_aalen",
    "hazard_rate",
    "cox_partial_likelihood",
    "exponential_survival",
    "weibull_survival",
    "log_rank_test",
    "median_survival_time",
    # monte carlo
    "importance_sampling_var",
    "antithetic_variates",
    "stratified_sampling",
    "block_bootstrap",
    "stationary_bootstrap",
    "filtered_historical_simulation",
]
