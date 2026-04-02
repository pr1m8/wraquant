wraquant
========

**The ultimate quantitative finance toolkit for Python.**

.. rst-class:: lead

   1,097 functions | 3,630+ tests | 27 modules | 265 TA indicators | 100K+ LOC

wraquant is a deeply integrated quant finance library that combines risk
management, regime detection, volatility modeling, derivatives pricing,
backtesting, portfolio optimization, fundamental analysis, machine learning,
and technical analysis in one cohesive framework -- with a consistent API,
deep documentation, and production-quality implementations.

.. code-block:: bash

   pip install wraquant

.. code-block:: python

   import wraquant as wq

   # One-liner comprehensive analysis
   report = wq.analyze(daily_returns)
   print(f"Sharpe: {report['risk']['sharpe']:.2f}")
   print(f"Max drawdown: {report['risk']['max_drawdown']:.2%}")
   print(f"Regime: {report['regime']['current']}")

   # Composable workflows -- zero glue code
   result = wq.quick_analysis_workflow().run(prices)
   result.risk       # risk metrics
   result.regimes    # regime detection
   result.garch      # GARCH volatility model


AI-Native Quant Research Lab
-----------------------------

**New in v1.0.0.** wraquant-mcp exposes 218 hand-crafted tools and 327
prompt templates as an MCP server. Point Claude or any AI agent at your
data -- it can fit GARCH models, detect regimes, optimize portfolios, run
backtests, and generate tearsheets through structured tool calls with
persistent DuckDB state. No notebooks, no glue code.

.. code-block:: bash

   pip install wraquant-mcp
   wraquant-mcp  # Start MCP server for Claude Desktop

See the :doc:`mcp` page for configuration and the full tool catalog.


Module Overview
---------------

.. grid:: 3

   .. grid-item-card:: Risk Management
      :link: api/risk
      :link-type: doc

      95+ functions: VaR/CVaR, beta estimation, factor risk,
      portfolio analytics, tail risk, copulas, stress testing,
      credit risk, and survival analysis.

   .. grid-item-card:: Regime Detection
      :link: api/regimes
      :link-type: doc

      38+ functions: Gaussian HMM, Markov-switching,
      Kalman filter/smoother/UKF, changepoint detection,
      regime scoring, and regime-aware portfolios.

   .. grid-item-card:: Volatility Modeling
      :link: api/vol
      :link-type: doc

      28+ functions: Full GARCH family (EGARCH, GJR, FIGARCH,
      HARCH), Hawkes processes, stochastic vol, realized vol
      estimators, DCC, and variance risk premium.

.. grid:: 3

   .. grid-item-card:: Technical Analysis
      :link: api/ta
      :link-type: doc

      263 indicators across 19 modules: momentum, overlap,
      volume, trend, cycles, Fibonacci, candlestick patterns,
      exotic oscillators, support/resistance, and more.

   .. grid-item-card:: Machine Learning
      :link: api/ml
      :link-type: doc

      44+ functions: LSTM/GRU/Transformer, sklearn pipelines,
      walk-forward validation, purged K-fold, triple-barrier
      labeling, SHAP, and online regression.

   .. grid-item-card:: Derivatives Pricing
      :link: api/price
      :link-type: doc

      50+ functions: Black-Scholes, FBSDE solvers, characteristic
      function pricing (Heston, VG, NIG), SABR, rough Bergomi,
      CIR, Vasicek, fixed income.

.. grid:: 3

   .. grid-item-card:: Backtesting
      :link: api/backtest
      :link-type: doc

      Event-driven and vectorized engines, walk-forward,
      position sizing, regime-conditional sizing, tearsheets,
      and 15+ performance metrics.

   .. grid-item-card:: Portfolio Optimization
      :link: api/opt
      :link-type: doc

      MVO, risk parity, Black-Litterman, HRP, inverse volatility,
      convex/nonlinear solvers, multi-objective, and constraint
      utilities.

   .. grid-item-card:: Statistics & Time Series
      :link: api/stats
      :link-type: doc

      Regression, cointegration, distributions, factor analysis,
      robust statistics, decomposition, forecasting, anomaly
      detection, and wavelets.

.. grid:: 3

   .. grid-item-card:: Fundamental Analysis
      :link: api/fundamental
      :link-type: doc

      28+ functions: financial ratios, DCF/DDM/RIM valuation,
      DuPont decomposition, financial health scoring, earnings
      quality, and stock screening (value, growth, quality,
      Piotroski, Magic Formula).

   .. grid-item-card:: News & Sentiment
      :link: api/news
      :link-type: doc

      17 functions: news sentiment scoring, sentiment time
      series and signals, earnings surprises, insider activity,
      institutional ownership, SEC filings search.

   .. grid-item-card:: MCP Server
      :link: mcp
      :link-type: doc

      218 tools, 327 prompts. Point an AI agent at your data:
      it fits GARCH, detects regimes, optimizes portfolios, runs
      backtests, prices derivatives, generates tearsheets.
      Shared DuckDB state. ``pip install wraquant-mcp``

.. grid:: 3

   .. grid-item-card:: Econometrics
      :link: api/econometrics
      :link-type: doc

      Panel data, VAR, VECM, IV/2SLS, event studies, structural
      breaks, impulse response, and Granger causality.

   .. grid-item-card:: Market Microstructure
      :link: api/microstructure
      :link-type: doc

      Liquidity (Amihud, Kyle), toxicity (VPIN), market quality,
      spread decomposition, order flow, and execution cost models.

   .. grid-item-card:: Execution Algorithms
      :link: api/execution
      :link-type: doc

      Almgren-Chriss optimal execution, TWAP, VWAP, POV,
      IS scheduling, transaction cost analysis, and
      slippage estimation.

.. grid:: 3

   .. grid-item-card:: Forex
      :link: api/forex
      :link-type: doc

      Carry trade analysis, currency strength, FX risk,
      session timing, cross rates, and pip calculations.

   .. grid-item-card:: Bayesian Inference
      :link: api/bayes
      :link-type: doc

      PyMC, emcee, BlackJAX, NumPyro backends. Bayesian
      Sharpe, portfolio, volatility, changepoints, and
      model comparison.

   .. grid-item-card:: Causal Inference
      :link: api/causal
      :link-type: doc

      Difference-in-differences, synthetic control, IPW,
      regression discontinuity, and IV estimation.

.. grid:: 3

   .. grid-item-card:: Data
      :link: api/data
      :link-type: doc

      Fetching (yfinance, FRED, NASDAQ, FMP), cleaning,
      validation, transforms, calendar alignment, and
      caching.

   .. grid-item-card:: Advanced Math
      :link: api/math
      :link-type: doc

      Network analysis, Levy processes, optimal stopping,
      spectral methods, PDEs, and entropy measures.

   .. grid-item-card:: Visualization
      :link: api/viz
      :link-type: doc

      Interactive Plotly dashboards for portfolios, regimes,
      risk, volatility surfaces, correlation networks,
      and tearsheets.


Cross-Module Integration
------------------------

wraquant modules are organized as a directed acyclic graph. Each module
feeds its outputs into the next with zero glue code::

   data --> ta/stats/ts --> vol/regimes --> risk --> opt --> backtest --> viz
                                |                    |          |
                                +-------- ml --------+   fundamental/news

Example pipeline:

.. code-block:: python

   from wraquant.vol import garch_fit
   from wraquant.risk import garch_var, historical_stress_test
   from wraquant.viz import plot_var_breaches

   # Fit GJR-GARCH --> time-varying VaR --> stress test --> visualize
   model = garch_fit(returns * 100, model="GJR", dist="t")
   var = garch_var(returns, vol_model="GJR", dist="t", alpha=0.01)
   stress = historical_stress_test(returns, scenarios=["gfc_2008", "covid_2020"])

See the :doc:`getting_started` page for installation, configuration, and
a complete walkthrough.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   getting_started

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Ecosystem
   :hidden:

   mcp
   changelog


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
