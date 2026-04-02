Tutorials
=========

End-to-end guides that show how wraquant's modules work together for
real quantitative finance workflows. Each tutorial walks through a
complete pipeline with runnable code and interpretation of results.

These tutorials assume you have wraquant installed with the relevant
extras. See :doc:`/getting_started` for installation instructions.

.. grid:: 2

   .. grid-item-card:: Risk Analysis
      :link: risk_analysis
      :link-type: doc

      Compute risk-adjusted performance metrics (Sharpe, Sortino, Treynor),
      estimate VaR and CVaR with multiple methods, stress test against
      historical crises (GFC, COVID), decompose portfolio risk with Euler
      decomposition, and generate a risk report.

      *Modules used:* ``risk``, ``vol``, ``data``, ``viz``

   .. grid-item-card:: Regime-Based Investing
      :link: regime_investing
      :link-type: doc

      Detect bull/bear market regimes with a Gaussian HMM, analyze
      per-regime statistics (mean, volatility, Sharpe), build a
      regime-conditional portfolio that adjusts allocation by regime
      probability, and backtest against buy-and-hold.

      *Modules used:* ``regimes``, ``opt``, ``backtest``, ``risk``

   .. grid-item-card:: Volatility Modeling
      :link: volatility_modeling
      :link-type: doc

      Fit GARCH, EGARCH, and GJR-GARCH models to daily returns. Compare
      models with BIC-based selection. Compute news impact curves to
      visualize asymmetric volatility response. Forecast volatility
      10 days ahead and run rolling GARCH for out-of-sample evaluation.

      *Modules used:* ``vol``, ``risk``, ``viz``

   .. grid-item-card:: Portfolio Construction
      :link: portfolio_construction
      :link-type: doc

      Optimize portfolios with Mean-Variance (max Sharpe), risk parity,
      and Black-Litterman. Decompose risk contributions per asset.
      Adjust allocations for market regimes using regime-aware
      optimization. Compare portfolio strategies on a backtest.

      *Modules used:* ``opt``, ``risk``, ``regimes``, ``backtest``

   .. grid-item-card:: Backtesting Strategies
      :link: backtesting_strategies
      :link-type: doc

      Define a moving average crossover strategy, backtest with the
      event-driven engine, analyze performance with 15+ metrics,
      generate a full tearsheet, and run walk-forward validation
      to evaluate out-of-sample robustness.

      *Modules used:* ``backtest``, ``ta``, ``risk``, ``viz``

   .. grid-item-card:: ML Alpha Research
      :link: ml_alpha_research
      :link-type: doc

      Engineer features from 263 TA indicators, label returns with
      triple-barrier labeling, train a gradient boosted model with
      purged K-fold cross-validation, walk-forward validate, evaluate
      with financial metrics (Sharpe, profit factor), and track
      experiments.

      *Modules used:* ``ml``, ``ta``, ``backtest``, ``risk``, ``experiment``


.. toctree::
   :maxdepth: 1
   :hidden:

   risk_analysis
   regime_investing
   volatility_modeling
   portfolio_construction
   backtesting_strategies
   ml_alpha_research
