wraquant
========

**The ultimate quantitative finance toolkit for Python.**

wraquant provides 27 modules covering risk management, volatility modeling,
regime detection, technical analysis, machine learning, derivatives pricing,
backtesting, portfolio optimization, and more -- all with a consistent API,
deep documentation, and production-quality implementations.

**New in v1.0.0: AI-native quant research lab.** wraquant-mcp exposes 218
hand-crafted tools and 327 prompt templates as an MCP server. Point Claude
or any AI agent at your data -- it can fit GARCH models, detect regimes,
optimize portfolios, run backtests, and generate tearsheets through structured
tool calls with persistent DuckDB state. No notebooks, no glue code.

.. code-block:: bash

   pip install wraquant-mcp
   wraquant-mcp  # Start MCP server for Claude Desktop

.. code-block:: python

   import wraquant as wq

   # One-liner comprehensive analysis
   report = wq.analyze(daily_returns)
   print(f"Sharpe: {report['risk']['sharpe']:.2f}")
   print(f"Max drawdown: {report['risk']['max_drawdown']:.2%}")

   # Composable workflows
   result = wq.quick_analysis_workflow().run(prices)
   result.risk       # risk metrics
   result.regimes    # regime detection
   result.garch      # GARCH volatility model

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

   .. grid-item-card:: MCP Server -- AI Quant Research Lab

      218 tools, 327 prompts. Point an AI agent at your data:
      it fits GARCH, detects regimes, optimizes portfolios, runs
      backtests, prices derivatives, generates tearsheets.
      Shared DuckDB state. ``pip install wraquant-mcp``

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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
