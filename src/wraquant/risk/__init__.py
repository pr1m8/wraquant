"""Risk management and portfolio risk analytics.

This module provides a comprehensive suite of tools for measuring,
decomposing, and stress-testing portfolio risk. It spans the full
spectrum from simple return-based metrics through tail-risk modeling,
credit risk, and survival analysis.

Key concepts
------------
**Risk** in quantitative finance is the possibility that actual returns
deviate from expected returns. This module organises risk tools into
several layers, from simple to sophisticated:

1. **Performance metrics** -- risk-adjusted return ratios.

   - ``sharpe_ratio`` -- excess return per unit of total risk (std dev).
     The most widely used performance measure; a Sharpe > 1 is generally
     considered good for a long-only strategy.
   - ``sortino_ratio`` -- like Sharpe but penalises only downside
     volatility. Preferred when the return distribution is skewed (most
     equity strategies).
   - ``information_ratio`` -- active return per unit of tracking error.
     Measures a manager's skill relative to a benchmark.
   - ``max_drawdown`` -- largest peak-to-trough decline. Captures the
     worst historical loss experience.
   - ``hit_ratio`` -- fraction of periods with positive returns.

2. **Value-at-Risk (VaR) and Expected Shortfall (CVaR)** -- quantile-
   based loss measures.

   - ``value_at_risk`` -- "with X% confidence, the portfolio will not
     lose more than VaR in one period."  Historical VaR uses the
     empirical distribution; parametric VaR assumes normality.
   - ``conditional_var`` (CVaR / Expected Shortfall) -- "given that the
     loss exceeds VaR, what is the expected loss?"  CVaR is coherent
     (satisfies sub-additivity) and is preferred by regulators (Basel
     III/IV) over VaR.

   When to use historical vs parametric: historical is non-parametric
   and captures fat tails, but needs a long sample (>1000 obs).
   Parametric is smooth and works with short samples, but underestimates
   tail risk if returns are non-normal.

3. **Portfolio risk decomposition** -- understand *where* risk comes from.

   - ``portfolio_volatility`` -- portfolio-level standard deviation.
   - ``risk_contribution`` -- Euler decomposition: each asset's
     marginal contribution to portfolio volatility.
   - ``diversification_ratio`` -- ratio of weighted-average individual
     vols to portfolio vol.  Higher is better.

4. **Stress testing and scenario analysis** -- ask "what if?"

   - ``stress_test_returns`` -- apply user-defined additive shocks.
   - ``historical_stress_test`` -- replay historical crises (GFC,
     COVID, dot-com) on your portfolio.
   - ``vol_stress_test`` -- scale volatility by multipliers (1.5x, 2x).
   - ``spot_stress_test`` -- shift price levels.
   - ``sensitivity_ladder`` -- P&L sensitivity to a single factor.
   - ``reverse_stress_test`` -- find scenarios that produce a target
     loss.
   - ``joint_stress_test`` -- simultaneous vol, spot, and correlation
     shocks.
   - ``marginal_stress_contribution`` -- identify the worst-contributing
     asset under a stress scenario.

5. **Copula dependency models** -- model the *joint* tail behaviour of
   multiple assets. Linear correlation understates co-movement in
   crashes; copulas capture this.

   - ``fit_gaussian_copula`` -- symmetric dependence; no tail dependence.
   - ``fit_t_copula`` -- symmetric tail dependence; good for equities.
   - ``fit_clayton_copula`` -- lower-tail dependence (joint crashes).
   - ``fit_gumbel_copula`` -- upper-tail dependence (joint rallies).
   - ``fit_frank_copula`` -- symmetric, no tail dependence; useful
     baseline.
   - ``copula_simulate`` -- Monte Carlo from any fitted copula.
   - ``tail_dependence`` -- empirical tail dependence coefficients.

6. **Dynamic correlation** -- time-varying dependence.

   - ``dcc_garch`` -- DCC-GARCH model for time-varying correlations
     and covariances.
   - ``rolling_correlation_dcc`` -- rolling DCC estimates.
   - ``forecast_correlation`` -- forward-looking correlation forecasts.

7. **Credit risk** -- default probability and credit-sensitive pricing.

   - ``merton_model`` -- structural model (equity as a call on assets).
   - ``altman_z_score`` -- bankruptcy prediction via accounting ratios.
   - ``default_probability`` -- cumulative PD from transition matrices.
   - ``credit_spread``, ``cds_spread`` -- implied spreads.
   - ``loss_given_default``, ``expected_loss`` -- EL = PD x LGD x EAD.

8. **Survival analysis** -- time-to-event modeling for defaults, fund
   closures, and drawdown durations.

   - ``kaplan_meier``, ``nelson_aalen`` -- non-parametric estimators.
   - ``cox_partial_likelihood`` -- semi-parametric Cox PH model.
   - ``exponential_survival``, ``weibull_survival`` -- parametric models.
   - ``log_rank_test`` -- compare survival curves across groups.

9. **Monte Carlo simulation** -- advanced sampling techniques.

   - ``importance_sampling_var`` -- variance reduction for tail
     estimation.
   - ``antithetic_variates``, ``stratified_sampling`` -- variance
     reduction.
   - ``block_bootstrap``, ``stationary_bootstrap`` -- resampling
     preserving serial dependence.
   - ``filtered_historical_simulation`` -- GARCH-filtered bootstrapping.

10. **Third-party integrations** -- wrappers for ``PyPortfolioOpt``,
    ``riskfolio-lib``, ``skfolio``, ``copulas``, and ``pyextremes``.

How to choose
-------------
- **Quick portfolio health check**: ``sharpe_ratio``, ``max_drawdown``,
  ``value_at_risk``.
- **Regulatory reporting (Basel)**: ``conditional_var`` at 97.5%,
  ``stress_test_returns``.
- **Portfolio construction**: ``risk_contribution`` +
  ``diversification_ratio``.
- **Tail-risk hedging**: ``fit_t_copula`` or ``fit_clayton_copula`` +
  ``copula_simulate``.
- **Credit analysis**: ``merton_model`` + ``altman_z_score``.

References
----------
- Artzner et al. (1999), "Coherent Measures of Risk"
- McNeil, Frey & Embrechts (2005), "Quantitative Risk Management"
- Merton (1974), "On the Pricing of Corporate Debt"
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
