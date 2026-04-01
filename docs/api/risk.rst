Risk Management (``wraquant.risk``)
====================================

The risk module provides 95+ functions for portfolio risk assessment,
spanning the full spectrum from simple return-based ratios through
tail-risk modeling, factor decomposition, copula dependence, stress
testing, credit risk, and survival analysis.

**Key capabilities:**

- Risk-adjusted performance: Sharpe, Sortino, Treynor, Information Ratio, capture ratios
- VaR and Expected Shortfall: historical, parametric, GARCH-based, Cornish-Fisher
- Portfolio risk decomposition: Euler decomposition, component/marginal/incremental VaR
- Beta estimation: rolling, Blume-adjusted, Vasicek, Dimson, conditional, EWMA
- Factor models: Fama-French, PCA-based, custom factor regression
- Copulas: Gaussian, Student-t, Clayton, Gumbel, Frank, simulation
- Stress testing: historical crisis replay, vol/spot/correlation shocks, reverse stress test
- Credit risk: Merton model, Altman Z-score, CDS spreads
- Survival analysis: Kaplan-Meier, Nelson-Aalen, Cox PH
- Monte Carlo: importance sampling, antithetic variates, filtered historical simulation

Quick Example
-------------

.. code-block:: python

   from wraquant.risk import sharpe_ratio, garch_var, crisis_drawdowns

   # Basic risk metrics
   sr = sharpe_ratio(returns)
   print(f"Sharpe ratio: {sr:.4f}")

   # GARCH-based time-varying VaR
   var_result = garch_var(returns, vol_model="GJR", dist="t")
   print(f"Current VaR: {var_result['var'].iloc[-1]:.4f}")
   print(f"Breach rate: {var_result['breach_rate']:.3f}")

   # Historical crisis analysis
   crises = crisis_drawdowns(returns, top_n=5)
   for c in crises:
       print(f"{c['start']} to {c['end']}: {c['max_drawdown']:.2%}")

Portfolio Risk Decomposition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from wraquant.risk import risk_contribution, diversification_ratio
   import numpy as np

   weights = np.array([0.4, 0.3, 0.2, 0.1])
   rc = risk_contribution(returns_df, weights)
   dr = diversification_ratio(returns_df, weights)
   print(f"Diversification ratio: {dr:.4f}")

Stress Testing
^^^^^^^^^^^^^^^

.. code-block:: python

   from wraquant.risk import historical_stress_test, scenario_library

   # Built-in crisis scenarios (GFC, COVID, dot-com, etc.)
   impact = historical_stress_test(returns)
   for scenario, result in impact.items():
       print(f"{scenario}: {result['return']:.2%}")

.. seealso::

   - :doc:`/tutorials/risk_analysis` -- End-to-end risk analysis tutorial
   - :doc:`vol` -- Volatility models that feed into VaR
   - :doc:`regimes` -- Regime detection for conditional risk management

API Reference
-------------

.. automodule:: wraquant.risk
   :members:
   :undoc-members:
   :show-inheritance:

Value at Risk
~~~~~~~~~~~~~

.. automodule:: wraquant.risk.var
   :members:

Metrics
~~~~~~~

.. automodule:: wraquant.risk.metrics
   :members:

Beta Estimation
~~~~~~~~~~~~~~~

.. automodule:: wraquant.risk.beta
   :members:

Factor Risk
~~~~~~~~~~~

.. automodule:: wraquant.risk.factor
   :members:

Portfolio Analytics
~~~~~~~~~~~~~~~~~~~

.. automodule:: wraquant.risk.portfolio_analytics
   :members:

Portfolio Risk
~~~~~~~~~~~~~~

.. automodule:: wraquant.risk.portfolio
   :members:

Copulas
~~~~~~~

.. automodule:: wraquant.risk.copulas
   :members:

Tail Risk & EVT
~~~~~~~~~~~~~~~~

.. automodule:: wraquant.risk.tail
   :members:

DCC Multivariate
~~~~~~~~~~~~~~~~

.. automodule:: wraquant.risk.dcc
   :members:

Monte Carlo
~~~~~~~~~~~

.. automodule:: wraquant.risk.monte_carlo
   :members:

Stress Testing
~~~~~~~~~~~~~~

.. automodule:: wraquant.risk.stress
   :members:

Scenarios
~~~~~~~~~

.. automodule:: wraquant.risk.scenarios
   :members:

Historical Events
~~~~~~~~~~~~~~~~~

.. automodule:: wraquant.risk.historical
   :members:

Credit Risk
~~~~~~~~~~~

.. automodule:: wraquant.risk.credit
   :members:

Survival Analysis
~~~~~~~~~~~~~~~~~

.. automodule:: wraquant.risk.survival
   :members:

Integrations
~~~~~~~~~~~~

.. automodule:: wraquant.risk.integrations
   :members:
