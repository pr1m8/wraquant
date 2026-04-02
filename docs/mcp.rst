MCP Server (wraquant-mcp)
=========================

wraquant-mcp exposes wraquant's 1,097 quantitative finance functions as
MCP (Model Context Protocol) tools that Claude, LangChain, and other AI
agents can call directly. Instead of writing Python scripts, an AI agent
can fit GARCH models, detect market regimes, compute VaR, optimize
portfolios, and generate tearsheets -- all through structured tool calls
with persistent DuckDB state.


Quick Start
-----------

Install
^^^^^^^^

.. code-block:: bash

   pip install wraquant-mcp

Claude Desktop
^^^^^^^^^^^^^^^

Add the following to your Claude Desktop config file
(``~/.claude/claude_desktop_config.json`` on macOS/Linux):

.. code-block:: json

   {
     "mcpServers": {
       "wraquant": {
         "command": "wraquant-mcp",
         "env": {
           "PYTHONUNBUFFERED": "1"
         }
       }
     }
   }

Then restart Claude Desktop. All 218 tools and 327 prompts are
immediately available.

CLI (stdio)
^^^^^^^^^^^^

.. code-block:: bash

   wraquant-mcp                     # Start stdio server for Claude Desktop

HTTP (LangChain / hosted deployments)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   wraquant-mcp --transport http    # Start HTTP server on port 8000
   wraquant-mcp --transport http --port 9000  # Custom port

Python (programmatic)
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from wraquant_mcp import create_server

   mcp = create_server("my-quant-server")
   mcp.run()  # stdio
   # or
   mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)


Architecture
------------

wraquant-mcp is designed to compose with other MCP servers for end-to-end
quant workflows:

.. code-block:: text

                            AI Agent (Claude / LangChain)
                                       |
                       ----------------+----------------
                       |               |               |
                 wraquant-mcp     OpenBB MCP      DuckDB MCP
                 (analysis)       (market data)   (SQL queries)
                       |               |               |
                       +-------+-------+-------+-------+
                               |               |
                        Shared DuckDB      Jupyter MCP
                        (data.duckdb)      (notebooks)
                               |
                        Alpaca MCP
                        (execution)

**How it works:**

1. **OpenBB MCP** fetches market data and stores it in DuckDB
2. **wraquant-mcp** reads data from DuckDB, runs analysis, stores results back
3. **DuckDB MCP** lets the agent query any stored dataset with SQL
4. **Jupyter MCP** connects to the same DuckDB file for notebook exploration
5. **Alpaca MCP** executes trades based on wraquant analysis

All MCPs share the same DuckDB file
(``~/.wraquant/workspaces/default/data.duckdb``), enabling seamless
composition without data copying.


Tool Tiers
----------

.. list-table::
   :header-rows: 1
   :widths: 15 50 10 15

   * - Tier
     - Description
     - Count
     - Loading
   * - **Tier 1**
     - Discovery and workspace tools
     - 5
     - Always loaded
   * - **Tier 2**
     - Hand-crafted analysis tools across 22 modules
     - 205
     - Always loaded
   * - **Tier 3**
     - Auto-registered wraquant functions via ToolAdaptor
     - ~900+
     - Always loaded
   * - **Prompts**
     - Guided multi-step workflow templates
     - 327
     - Always loaded

Tier 2 tools are curated, optimized wrappers with domain-specific logic.
Tier 3 tools are auto-generated from wraquant's ``__all__`` exports --
every public function is available, wrapped with automatic DuckDB dataset
resolution and JSON-safe output serialization.


Tool Reference by Module
-------------------------

Discovery tools (Tier 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``list_modules`` -- List all 22+ analysis modules with descriptions and function counts
- ``list_tools`` -- List available tools in a specific module
- ``workspace_status`` -- Show current datasets, models, and journal state
- ``workspace_history`` -- Show recent operations in the journal
- ``add_note`` -- Add a research note to the workspace journal

Common operations (Tier 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``analyze`` -- Comprehensive analysis (stats, risk, stationarity, regime, GARCH)
- ``compute_returns`` -- Compute simple or log returns from prices
- ``compute_indicator`` -- Compute any of 265 TA indicators
- ``fit_garch`` -- Fit GARCH/EGARCH/GJR volatility models
- ``detect_regimes`` -- Detect market regimes (HMM, GMM, changepoint)
- ``risk_metrics`` -- Compute Sharpe, Sortino, max drawdown, hit ratio
- ``dataset_info`` -- Get schema, stats, lineage, and sample rows for a dataset
- ``store_data`` -- Store inline data as a workspace dataset

Module tools (Tier 2, 22 modules)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 18 7 75

   * - Module
     - Tools
     - Key tools
   * - **risk**
     - 15
     - ``var_analysis``, ``stress_test``, ``beta_analysis``, ``factor_analysis``,
       ``crisis_drawdowns``, ``portfolio_risk``, ``tail_risk``, ``copula_fit``,
       ``credit_analysis``, ``monte_carlo_var``
   * - **data**
     - 17
     - ``fetch_yahoo``, ``fetch_ohlcv``, ``load_csv``, ``load_json``,
       ``export_dataset``, ``merge_datasets``, ``filter_dataset``,
       ``clean_dataset``, ``resample_ohlcv``
   * - **microstructure**
     - 16
     - ``liquidity_metrics``, ``toxicity_analysis``, ``market_quality``,
       ``spread_decomposition``, ``price_impact``, ``depth_analysis``
   * - **viz**
     - 14
     - ``plot_equity_curve``, ``plot_drawdown``, ``plot_regime``,
       ``plot_correlation``, ``plot_candlestick``, ``plot_tearsheet``
   * - **math**
     - 14
     - ``correlation_network``, ``systemic_risk``, ``levy_simulate``,
       ``optimal_stopping``, ``spectral_analysis``
   * - **regimes**
     - 12
     - ``regime_statistics``, ``fit_gaussian_hmm``, ``fit_ms_autoregression``,
       ``kalman_filter``, ``rolling_regime_probability``
   * - **vol**
     - 11
     - ``forecast_volatility``, ``news_impact_curve``, ``model_selection``,
       ``realized_volatility``, ``hawkes_fit``, ``garch_rolling``
   * - **stats**
     - 11
     - ``correlation_analysis``, ``regression``, ``distribution_fit``,
       ``stationarity_tests``, ``cointegration_test``
   * - **ta**
     - 11
     - ``list_indicators``, ``multi_indicator``, ``scan_signals``,
       ``momentum_indicators``, ``ta_summary``
   * - **ts**
     - 10
     - ``forecast``, ``decompose``, ``changepoint_detect``,
       ``anomaly_detect``, ``ensemble_forecast``
   * - **backtest**
     - 10
     - ``run_backtest``, ``backtest_metrics``, ``comprehensive_tearsheet``,
       ``walk_forward``, ``strategy_comparison``
   * - **execution**
     - 10
     - ``optimal_schedule``, ``execution_cost``, ``almgren_chriss``,
       ``transaction_cost_analysis``, ``slippage_estimate``
   * - **ml**
     - 9
     - ``build_features``, ``train_model``, ``feature_importance``,
       ``walk_forward_ml``, ``pca_factors``
   * - **price**
     - 9
     - ``price_option``, ``compute_greeks``, ``implied_volatility``,
       ``simulate_process``, ``sabr_calibrate``
   * - **opt**
     - 8
     - ``optimize_portfolio``, ``efficient_frontier``, ``black_litterman``,
       ``hierarchical_risk_parity``, ``risk_budgeting``
   * - **causal**
     - 7
     - ``granger_causality``, ``event_study``, ``diff_in_diff``,
       ``synthetic_control``, ``instrumental_variable``
   * - **bayes**
     - 7
     - ``bayesian_sharpe``, ``bayesian_regression``, ``bayesian_changepoint``,
       ``bayesian_portfolio``, ``model_comparison_bayesian``
   * - **econometrics**
     - 6
     - ``var_model``, ``panel_regression``, ``structural_break``,
       ``impulse_response``, ``event_study_econometric``
   * - **forex**
     - 6
     - ``carry_analysis``, ``fx_risk``, ``currency_strength``,
       ``session_info``, ``pip_calculator``
   * - **experiment**
     - 5
     - ``create_experiment``, ``run_experiment``, ``experiment_results``,
       ``parameter_sensitivity``
   * - **fundamental**
     - 5
     - ``piotroski_score``, ``altman_z``, ``dcf_valuation``,
       ``fundamental_ratios``, ``quality_screen``
   * - **news**
     - 5
     - ``sentiment_score``, ``sentiment_aggregate``, ``news_signal``,
       ``news_impact``, ``earnings_surprise``

Supervisor tools (Tier 2)
^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``recommend_workflow`` -- Given an analysis goal, recommend tools and step order
- ``module_guide`` -- Get a usage guide for any wraquant module

Workspace management (Tier 2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``create_workspace`` -- Create a new isolated research workspace
- ``open_workspace`` -- Open an existing workspace, restoring all state
- ``list_workspaces`` -- List all workspaces with metadata
- ``snapshot`` -- Create a named checkpoint of the current workspace
- ``restore_snapshot`` -- Restore workspace to a previous snapshot
- ``delete_workspace`` -- Delete a workspace and all its data
- ``query_data`` -- Run SQL (SELECT) against the workspace DuckDB


Prompt Catalog
--------------

327 prompt templates organized across 16 categories. Each template guides
an AI agent through a multi-step quantitative finance workflow, explaining
which tools to call, in what order, and how to interpret results.

**Categories:**

- **Risk analysis** -- VaR estimation, stress testing, risk decomposition,
  tail risk, copula analysis, credit risk
- **Volatility** -- GARCH fitting, model selection, forecasting, realized
  vol, stochastic vol, news impact
- **Regime detection** -- HMM fitting, state interpretation, regime
  statistics, regime-conditional portfolios
- **Portfolio optimization** -- MVO, risk parity, Black-Litterman, HRP,
  efficient frontier, rebalancing
- **Technical analysis** -- Indicator computation, signal scanning,
  pattern recognition, multi-timeframe
- **Backtesting** -- Strategy backtesting, walk-forward, regime
  backtesting, performance analysis
- **Machine learning** -- Feature engineering, training, walk-forward ML,
  PCA, isolation forest, online learning
- **Pricing** -- Options pricing, Greeks, implied vol, SABR, bond
  duration, yield curve, process simulation
- **Statistical analysis** -- Regression, correlation, distribution
  fitting, cointegration, stationarity
- **Time series** -- Forecasting, decomposition, changepoints, anomaly
  detection, SSA
- **Econometrics** -- VAR, panel regression, structural breaks, impulse
  response, event studies
- **Microstructure** -- Liquidity metrics, toxicity, market quality,
  spread decomposition
- **Execution** -- Optimal scheduling, transaction cost analysis,
  Almgren-Chriss, slippage
- **Causal inference** -- Granger causality, DID, synthetic control,
  regression discontinuity
- **Fundamental analysis** -- Financial ratios, DCF valuation, stock
  screening, financial health
- **News & sentiment** -- Sentiment scoring, earnings analysis, insider
  tracking, SEC filings


Full Documentation
-------------------

For the complete tool catalog, prompt listing, and advanced configuration
options, see the `wraquant-mcp README on GitHub
<https://github.com/pr1m8/wraquant/blob/main/mcp/README.md>`_.

.. seealso::

   - :doc:`getting_started` -- Installing wraquant and wraquant-mcp
   - :doc:`changelog` -- Version history and release notes
   - :doc:`api/index` -- Full Python API reference
