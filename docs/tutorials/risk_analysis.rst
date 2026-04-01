Risk Analysis
=============

This tutorial demonstrates a complete risk analysis workflow: computing
risk metrics, estimating Value-at-Risk, stress testing against historical
crises, decomposing portfolio risk, and generating a comprehensive report.

By the end, you will know how to assess a portfolio's risk profile using
wraquant's 95+ risk functions.


Step 1: Load Data and Compute Returns
--------------------------------------

.. code-block:: python

   import wraquant as wq
   import pandas as pd
   import numpy as np

   # Fetch historical prices (requires market-data extra)
   from wraquant.data import fetch_prices

   prices = fetch_prices(["AAPL", "MSFT", "AMZN", "GOOGL"], start="2018-01-01")
   returns = prices.pct_change().dropna()

   # Single asset for initial analysis
   aapl_returns = returns["AAPL"]

   # Quick check
   print(f"Observations: {len(aapl_returns)}")
   print(f"Date range: {aapl_returns.index[0]} to {aapl_returns.index[-1]}")


Step 2: Risk-Adjusted Performance Metrics
------------------------------------------

Compute the standard battery of risk-adjusted return ratios. These tell you
whether the returns compensate for the risk taken.

.. code-block:: python

   from wraquant.risk import (
       sharpe_ratio, sortino_ratio, information_ratio,
       max_drawdown, hit_ratio, capture_ratios,
   )

   sr = sharpe_ratio(aapl_returns)
   print(f"Sharpe ratio: {sr:.4f}")
   # Interpretation: >1.0 is good, >2.0 is excellent for long-only

   so = sortino_ratio(aapl_returns)
   print(f"Sortino ratio: {so:.4f}")
   # Uses downside deviation only -- preferred for skewed distributions.
   # Sortino > Sharpe suggests positive skew (more upside than downside).

   mdd = max_drawdown(aapl_returns)
   print(f"Max drawdown: {mdd:.4f}")
   # The worst peak-to-trough decline. -0.30 means a 30% drawdown.

   hr = hit_ratio(aapl_returns)
   print(f"Hit ratio: {hr:.4f}")
   # Fraction of days with positive returns. ~0.52-0.54 is typical for equities.


Step 3: Value-at-Risk and Expected Shortfall
---------------------------------------------

VaR answers: "with X% confidence, what is the maximum loss in one period?"
CVaR (Expected Shortfall) answers: "given we exceeded VaR, what is the average loss?"

.. code-block:: python

   from wraquant.risk import value_at_risk, conditional_var, garch_var

   # Historical VaR at 95% and 99% confidence
   var_95 = value_at_risk(aapl_returns, confidence=0.95)
   var_99 = value_at_risk(aapl_returns, confidence=0.99)
   print(f"VaR(95%): {var_95:.4f}  |  VaR(99%): {var_99:.4f}")

   # Expected Shortfall (CVaR) -- coherent risk measure, preferred by regulators
   cvar = conditional_var(aapl_returns, confidence=0.95)
   print(f"CVaR(95%): {cvar:.4f}")
   # CVaR is always worse (more negative) than VaR

   # GARCH-based time-varying VaR -- captures volatility clustering
   gvar = garch_var(aapl_returns, vol_model="GJR", dist="t")
   print(f"Current GARCH VaR: {gvar['var'].iloc[-1]:.4f}")
   print(f"Breach rate: {gvar['breach_rate']:.3f}")
   # Breach rate should be close to (1 - confidence). If much higher,
   # the model underestimates tail risk.


Step 4: Stress Testing Against Historical Crises
--------------------------------------------------

Stress tests answer "what if?" by applying historical or hypothetical shocks
to your portfolio.

.. code-block:: python

   from wraquant.risk import (
       historical_stress_test, stress_test_returns,
       vol_stress_test, scenario_library,
   )

   # Replay historical crises on your portfolio
   # Built-in scenarios: GFC 2008, COVID 2020, dot-com, etc.
   scenarios = scenario_library()
   print(f"Available scenarios: {list(scenarios.keys())}")

   crisis_impact = historical_stress_test(aapl_returns)
   for scenario, result in crisis_impact.items():
       print(f"{scenario}: return={result['return']:.4f}, "
             f"max_dd={result['max_drawdown']:.4f}")

   # Custom stress: what if returns drop by 5%?
   custom_shock = stress_test_returns(aapl_returns, shock=-0.05)
   print(f"Post-shock mean: {custom_shock['stressed_mean']:.4f}")

   # Volatility stress: what happens at 2x current vol?
   vol_result = vol_stress_test(aapl_returns, multiplier=2.0)
   print(f"VaR at 2x vol: {vol_result['var']:.4f}")


Step 5: Crisis Drawdown Analysis
---------------------------------

Examine the worst drawdown episodes in detail -- when they started, how
deep they went, and how long recovery took.

.. code-block:: python

   from wraquant.risk import crisis_drawdowns, event_impact, contagion_analysis

   # Top 5 worst drawdowns with full lifecycle
   crises = crisis_drawdowns(aapl_returns, top_n=5)
   for c in crises:
       print(f"Start: {c['start']}, End: {c['end']}, "
             f"Max DD: {c['max_drawdown']:.4f}, "
             f"Duration: {c['duration']} days, "
             f"Recovery: {c['recovery_days']} days")

   # How did correlations change during crises vs normal times?
   contagion = contagion_analysis(returns)
   print(f"Normal-period avg correlation: {contagion['normal_corr']:.4f}")
   print(f"Crisis-period avg correlation: {contagion['crisis_corr']:.4f}")
   # Correlations typically increase during crises (diversification fails
   # when you need it most).


Step 6: Portfolio Risk Decomposition
-------------------------------------

For a multi-asset portfolio, decompose risk to understand which assets
contribute most to portfolio volatility.

.. code-block:: python

   from wraquant.risk import (
       portfolio_volatility, risk_contribution,
       diversification_ratio, component_var,
   )

   # Equal-weight portfolio
   weights = np.array([0.25, 0.25, 0.25, 0.25])

   port_vol = portfolio_volatility(returns, weights)
   print(f"Portfolio volatility: {port_vol:.4f}")

   # Euler risk decomposition: each asset's marginal contribution
   rc = risk_contribution(returns, weights)
   for asset, contrib in zip(returns.columns, rc):
       print(f"  {asset}: {contrib:.4f} ({contrib/port_vol:.1%} of total)")

   # Diversification benefit
   dr = diversification_ratio(returns, weights)
   print(f"Diversification ratio: {dr:.4f}")
   # >1.0 means diversification is working. Higher is better.

   # Component VaR: per-asset VaR contribution
   cvar_decomp = component_var(returns, weights, confidence=0.95)
   for asset, cv in zip(returns.columns, cvar_decomp):
       print(f"  {asset} component VaR: {cv:.4f}")


Step 7: Factor Risk Model
---------------------------

Decompose returns into systematic factor exposures and idiosyncratic risk.

.. code-block:: python

   from wraquant.risk import factor_risk_model, fama_french_regression

   # Fama-French 3-factor regression
   ff = fama_french_regression(aapl_returns, model="3factor")
   print(f"Alpha:    {ff['alpha']:.4f} (p={ff['alpha_pvalue']:.3f})")
   print(f"Market:   {ff['market_beta']:.4f}")
   print(f"SMB:      {ff['smb_beta']:.4f}")
   print(f"HML:      {ff['hml_beta']:.4f}")
   print(f"R-squared: {ff['r_squared']:.4f}")
   # R-squared tells you what fraction of return variance is explained
   # by systematic factors. The remainder is idiosyncratic (stock-specific).


Next Steps
----------

- :doc:`/tutorials/regime_investing` -- Combine risk metrics with regime detection
  for adaptive risk management.
- :doc:`/tutorials/portfolio_construction` -- Use risk decomposition to build
  better portfolios.
- :doc:`/api/risk` -- Full API reference for all 95+ risk functions.
