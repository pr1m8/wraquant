# wraquant-mcp

MCP server for wraquant -- a quant analysis engine for AI agents.

[![PyPI](https://img.shields.io/pypi/v/wraquant-mcp)](https://pypi.org/project/wraquant-mcp/)
[![Python](https://img.shields.io/pypi/pyversions/wraquant-mcp)](https://pypi.org/project/wraquant-mcp/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](../LICENSE)

**218 hand-crafted tools | 217 prompt templates | 22 module servers | shared DuckDB state**

---

## What is this?

wraquant-mcp exposes wraquant's 1,097 quantitative finance functions as MCP
(Model Context Protocol) tools that Claude, LangChain, and other AI agents
can call directly.

Instead of writing Python scripts, an AI agent can:

- Fit a GARCH model, detect market regimes, compute VaR, and generate a tearsheet
  -- all through structured tool calls
- Maintain state across operations via a shared DuckDB database
- Follow guided multi-step workflows using 217 prompt templates
- Compose with OpenBB (data), DuckDB MCP (SQL), Jupyter MCP (notebooks),
  and Alpaca MCP (execution) for end-to-end quant workflows

Every wraquant function -- from `risk.sharpe_ratio` to `vol.garch_fit` to
`regimes.detect_regimes` -- is available as a typed MCP tool with automatic
data resolution, result storage, and JSON-safe output.

---

## Architecture

```
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
```

### How it works

1. **OpenBB MCP** fetches market data and stores it in DuckDB
2. **wraquant-mcp** reads data from DuckDB, runs analysis, stores results back
3. **DuckDB MCP** lets the agent query any stored dataset with SQL
4. **Jupyter MCP** can connect to the same DuckDB file for notebook-based exploration
5. **Alpaca MCP** executes trades based on wraquant analysis

All MCPs share the same DuckDB file (`~/.wraquant/workspaces/default/data.duckdb`),
enabling seamless composition without data copying.

### Tool tiers

| Tier        | Description                                        | Count | Loading       |
| ----------- | -------------------------------------------------- | ----- | ------------- |
| **Tier 1**  | Discovery and workspace tools                      | 5     | Always loaded |
| **Tier 2**  | Hand-crafted analysis tools across 22 modules      | 205   | Always loaded |
| **Tier 3**  | Auto-registered wraquant functions via ToolAdaptor | ~900+ | Always loaded |
| **Prompts** | Guided multi-step workflow templates               | 217   | Always loaded |

Tier 2 tools are curated, optimized wrappers with domain-specific logic.
Tier 3 tools are auto-generated from wraquant's `__all__` exports -- every
public function in wraquant is available, wrapped with automatic DuckDB
dataset resolution and JSON-safe output serialization.

---

## Quick Start

### Claude Desktop

Add to `~/.claude/claude_desktop_config.json`:

```json
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
```

### CLI (stdio)

```bash
pip install wraquant-mcp
wraquant-mcp                    # Start stdio server for Claude Desktop
```

### HTTP (for LangChain / hosted deployments)

```bash
wraquant-mcp --transport http   # Start HTTP server on port 8000
wraquant-mcp --transport http --port 9000  # Custom port
```

### Python (module entry)

```bash
python -m wraquant_mcp          # Same as wraquant-mcp
```

### Programmatic

```python
from wraquant_mcp import create_server

mcp = create_server("my-quant-server")
mcp.run()  # stdio
# or
mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
```

---

## Available Tools by Module

### Tier 1: Discovery (5 tools)

| Tool                | Description                                                         |
| ------------------- | ------------------------------------------------------------------- |
| `list_modules`      | List all 22+ analysis modules with descriptions and function counts |
| `list_tools`        | List available tools in a specific module                           |
| `workspace_status`  | Show current datasets, models, and journal state                    |
| `workspace_history` | Show recent operations in the journal                               |
| `add_note`          | Add a research note to the workspace journal                        |

### Tier 1: Common Operations (8 tools)

| Tool                | Description                                                           |
| ------------------- | --------------------------------------------------------------------- |
| `analyze`           | Run comprehensive analysis (stats, risk, stationarity, regime, GARCH) |
| `compute_returns`   | Compute simple or log returns from prices                             |
| `compute_indicator` | Compute any of 265 TA indicators                                      |
| `fit_garch`         | Fit GARCH/EGARCH/GJR volatility models                                |
| `detect_regimes`    | Detect market regimes (HMM, GMM, changepoint)                         |
| `risk_metrics`      | Compute Sharpe, Sortino, max drawdown, hit ratio                      |
| `dataset_info`      | Get schema, stats, lineage, and sample rows for a dataset             |
| `store_data`        | Store inline data as a workspace dataset                              |

### Tier 2: Module Servers (22 modules, 218 tools)

| Module             | Tools | Key Tools                                                                                                                                                                                                                                                                                                                                                             |
| ------------------ | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **risk**           | 15    | `var_analysis`, `stress_test`, `beta_analysis`, `factor_analysis`, `crisis_drawdowns`, `portfolio_risk`, `tail_risk`, `copula_fit`, `credit_analysis`, `monte_carlo_var`, `dcc_correlation`, `cornish_fisher_var`, `rolling_beta`, `survival_analysis`, `expected_shortfall_decomposition`                                                                            |
| **vol**            | 11    | `forecast_volatility`, `news_impact_curve`, `model_selection`, `realized_volatility`, `ewma_volatility`, `hawkes_fit`, `stochastic_vol`, `variance_risk_premium`, `bipower_variation`, `jump_detection`, `garch_rolling`                                                                                                                                              |
| **data**           | 17    | `fetch_yahoo`, `fetch_ohlcv`, `load_csv`, `load_json`, `export_dataset`, `merge_datasets`, `filter_dataset`, `clean_dataset`, `validate_returns_tool`, `resample_ohlcv`, `resample_dataset`, `add_column`, `describe_dataset`, `split_dataset`, `rename_columns`, `compute_log_returns`, `align_datasets`                                                             |
| **microstructure** | 16    | `liquidity_metrics`, `toxicity_analysis`, `market_quality`, `spread_decomposition`, `price_impact`, `depth_analysis`, `kyle_lambda_rolling`, `amihud_rolling`, `corwin_schultz_spread`, `roll_spread_tool`, `effective_spread_tool`, `trade_classification_tool`, `order_flow_imbalance`, `information_share`, `intraday_volatility_pattern`, `liquidity_commonality` |
| **viz**            | 14    | `plot_equity_curve`, `plot_drawdown`, `plot_regime`, `plot_correlation`, `plot_candlestick`, `plot_returns`, `plot_distribution`, `plot_heatmap`, `plot_vol_surface`, `plot_factor_exposure`, `plot_rolling_metrics`, `plot_tearsheet`, `portfolio_dashboard`, `regime_dashboard`                                                                                     |
| **math**           | 14    | `correlation_network`, `systemic_risk`, `levy_simulate`, `optimal_stopping`, `hawkes_fit`, `spectral_analysis`, `minimum_spanning_tree`, `community_detection`, `contagion_simulation`, `variance_gamma_simulate`, `nig_simulate`, `longstaff_schwartz`, `cusum_detect`, `entropy_analysis`                                                                           |
| **regimes**        | 12    | `regime_statistics`, `select_n_states`, `rolling_regime_probability`, `regime_scoring`, `fit_gaussian_hmm`, `fit_ms_autoregression`, `gaussian_mixture_regimes`, `kalman_filter`, `kalman_regression`, `regime_conditional_moments`, `regime_transition`, `regime_labels`                                                                                             |
| **stats**          | 11    | `correlation_analysis`, `regression`, `distribution_fit`, `stationarity_tests`, `cointegration_test`, `cointegration_johansen`, `partial_correlation`, `distance_correlation`, `mutual_information`, `kde_estimate`, `best_fit_distribution`                                                                                                                          |
| **ta**             | 11    | `list_indicators`, `multi_indicator`, `scan_signals`, `momentum_indicators`, `trend_indicators`, `volatility_indicators`, `volume_indicators`, `pattern_recognition`, `support_resistance`, `ta_summary`, `ta_screening`                                                                                                                                              |
| **ts**             | 10    | `forecast`, `decompose`, `stationarity_tests`, `changepoint_detect`, `anomaly_detect`, `ssa_decompose`, `arima_diagnostics`, `rolling_forecast`, `ensemble_forecast`, `ornstein_uhlenbeck`                                                                                                                                                                            |
| **backtest**       | 10    | `run_backtest`, `backtest_metrics`, `comprehensive_tearsheet`, `walk_forward`, `strategy_comparison`, `omega_ratio`, `kelly_fraction`, `regime_backtest`, `vectorized_backtest`, `drawdown_analysis`                                                                                                                                                                  |
| **execution**      | 10    | `optimal_schedule`, `execution_cost`, `almgren_chriss`, `transaction_cost_analysis`, `close_auction`, `is_schedule_tool`, `pov_schedule_tool`, `expected_cost_model_tool`, `bertsimas_lo_tool`, `slippage_estimate`                                                                                                                                                   |
| **ml**             | 9     | `build_features`, `train_model`, `feature_importance`, `walk_forward_ml`, `pca_factors`, `isolation_forest`, `svm_classify`, `online_regression`, `cross_asset_features`                                                                                                                                                                                              |
| **price**          | 9     | `price_option`, `compute_greeks`, `implied_volatility`, `simulate_process`, `sabr_calibrate`, `fbsde_price`, `bond_duration`, `yield_curve_analysis`, `simulate_heston`                                                                                                                                                                                               |
| **opt**            | 8     | `optimize_portfolio`, `efficient_frontier`, `black_litterman`, `hierarchical_risk_parity`, `min_volatility`, `max_sharpe`, `risk_budgeting`, `rebalance_analysis`                                                                                                                                                                                                     |
| **causal**         | 7     | `granger_causality`, `event_study`, `diff_in_diff`, `synthetic_control`, `regression_discontinuity`, `mediation_analysis`, `instrumental_variable`                                                                                                                                                                                                                    |
| **bayes**          | 7     | `bayesian_sharpe`, `bayesian_regression`, `bayesian_changepoint`, `bayesian_portfolio`, `model_comparison_bayesian`, `bayesian_volatility`, `hmc_sample`                                                                                                                                                                                                              |
| **econometrics**   | 6     | `var_model`, `panel_regression`, `structural_break`, `impulse_response`, `event_study_econometric`, `instrumental_variable`                                                                                                                                                                                                                                           |
| **forex**          | 6     | `carry_analysis`, `fx_risk`, `currency_strength`, `cross_rate`, `session_info`, `pip_calculator`                                                                                                                                                                                                                                                                      |
| **experiment**     | 5     | `create_experiment`, `run_experiment`, `experiment_results`, `experiment_comparison`, `parameter_sensitivity`                                                                                                                                                                                                                                                         |
| **fundamental**    | 5     | `piotroski_score`, `altman_z`, `dcf_valuation`, `fundamental_ratios`, `quality_screen`                                                                                                                                                                                                                                                                                |
| **news**           | 5     | `sentiment_score`, `sentiment_aggregate`, `news_signal`, `news_impact`, `earnings_surprise`                                                                                                                                                                                                                                                                           |

### Tier 2: Supervisor (2 tools)

| Tool                 | Description                                            |
| -------------------- | ------------------------------------------------------ |
| `recommend_workflow` | Given an analysis goal, recommend tools and step order |
| `module_guide`       | Get a usage guide for any wraquant module              |

### Tier 2: Workspace Management (7 tools)

| Tool               | Description                                        |
| ------------------ | -------------------------------------------------- |
| `create_workspace` | Create a new isolated research workspace           |
| `open_workspace`   | Open an existing workspace, restoring all state    |
| `list_workspaces`  | List all workspaces with metadata                  |
| `snapshot`         | Create a named checkpoint of the current workspace |
| `restore_snapshot` | Restore workspace to a previous snapshot           |
| `delete_workspace` | Delete a workspace and all its data                |
| `query_data`       | Run SQL (SELECT) against the workspace DuckDB      |

### Tier 3: Auto-Registered (~900+ tools)

Every public function in wraquant's `__all__` exports is auto-registered as
an MCP tool via the `ToolAdaptor` pattern. These tools automatically:

- Resolve dataset names to DataFrames from DuckDB
- Call the wraquant function with resolved data
- Store DataFrame results back in DuckDB
- Store model results as joblib files
- Return JSON-safe metadata (never raw DataFrames)

---

## Prompt Templates

217 prompt templates organized across 16 categories. Each template guides an
AI agent through a multi-step quantitative finance workflow, explaining which
tools to use, in what order, and how to interpret results.

### System (1 prompt)

| Prompt                    | Description                                                                                          |
| ------------------------- | ---------------------------------------------------------------------------------------------------- |
| `wraquant_system_context` | Comprehensive system context: all modules, state model, tool chaining patterns, interpretation guide |

### Analysis (9 prompts)

| Prompt                 | Description                                                             |
| ---------------------- | ----------------------------------------------------------------------- |
| `equity_deep_dive`     | 7-phase single-stock analysis: stats, vol, regimes, TA, risk, synthesis |
| `sector_comparison`    | Multi-sector side-by-side comparison with cointegration check           |
| `macro_analysis`       | Macro regime and cross-asset correlation analysis                       |
| `earnings_impact`      | Event study around earnings announcements                               |
| `ipo_analysis`         | Post-IPO behavior and stabilization analysis                            |
| `market_breadth`       | Advance/decline, McClellan oscillator, percent above MA                 |
| `cross_asset_study`    | Cross-asset correlation regimes and flight-to-quality                   |
| `seasonality_analysis` | Day-of-week, month effects, holiday patterns                            |
| `liquidity_screen`     | Rank assets by Amihud, spread, turnover with composite score            |

### Risk & Volatility (10 prompts)

| Prompt                   | Description                                          |
| ------------------------ | ---------------------------------------------------- |
| `volatility_deep_dive`   | GARCH model selection, forecasting, news impact      |
| `risk_report`            | Institutional-grade portfolio risk report (8 phases) |
| `tail_risk_assessment`   | Extreme value theory and tail dependence             |
| `stress_test_battery`    | All 7 crisis scenarios ranked by severity            |
| `correlation_breakdown`  | DCC dynamics and contagion analysis                  |
| `vol_surface_analysis`   | Implied vol surface, skew, SABR calibration          |
| `credit_risk_assessment` | Merton model, Altman Z, KMV default probability      |
| `copula_risk`            | Copula-based tail dependence and crash co-movement   |
| `liquidity_risk`         | Amihud crisis comparison, spread widening scenarios  |
| `var_backtesting`        | Kupiec, Christoffersen, breach severity testing      |

### Regime Detection (7 prompts)

| Prompt                  | Description                                            |
| ----------------------- | ------------------------------------------------------ |
| `regime_detection`      | Multi-method regime detection with comparison          |
| `market_regime_monitor` | Current market state and regime probability            |
| `regime_backtest`       | Regime-conditional vs unconditional strategy           |
| `changepoint_analysis`  | Structural break detection with pre/post comparison    |
| `multi_asset_regime`    | Joint regime detection across asset classes            |
| `volatility_regime`     | Vol-based regime classification with GARCH persistence |
| `correlation_regime`    | DCC and regime-conditional correlation analysis        |

### Portfolio Construction (9 prompts)

| Prompt                   | Description                                            |
| ------------------------ | ------------------------------------------------------ |
| `portfolio_construction` | Full multi-method portfolio optimization (7 phases)    |
| `portfolio_rebalance`    | Rebalancing cost analysis and optimal execution        |
| `factor_attribution`     | Factor exposure and risk attribution                   |
| `portfolio_stress_test`  | Multi-scenario portfolio stress testing                |
| `asset_allocation`       | Strategic multi-asset allocation with regime awareness |
| `tactical_allocation`    | Regime-aware tactical tilts from strategic weights     |
| `risk_parity_deep_dive`  | Equal risk contribution construction and validation    |
| `factor_tilt`            | Tilt portfolio toward desired factor exposures         |
| `esg_screen`             | ESG-constrained portfolio construction                 |

### Trading Strategies (9 prompts)

| Prompt                  | Description                                                            |
| ----------------------- | ---------------------------------------------------------------------- |
| `pairs_trading`         | Full pairs trading workflow: cointegration through backtest (8 phases) |
| `momentum_strategy`     | RSI + MACD + ROC with regime filter                                    |
| `mean_reversion`        | Stationarity, OU fit, Bollinger Band signals                           |
| `trend_following`       | MA crossover, ADX filter, PSAR stops                                   |
| `statistical_arbitrage` | PCA factors, residual alpha, capacity estimation                       |
| `carry_trade`           | FX carry portfolio construction and crash risk                         |
| `volatility_selling`    | Put selling, straddle selling, VRP capture                             |
| `market_making`         | Spread capture, Avellaneda-Stoikov, VPIN toxicity                      |
| `sector_rotation`       | Momentum + fundamentals-based sector selection                         |

### Machine Learning (8 prompts)

| Prompt                 | Description                                                     |
| ---------------------- | --------------------------------------------------------------- |
| `ml_alpha_research`    | Full ML pipeline: features, model, backtest (8 phases)          |
| `feature_engineering`  | Comprehensive feature construction for ML                       |
| `model_comparison`     | Compare RF, GBM, SVM, LSTM with walk-forward                    |
| `hyperparameter_sweep` | Grid search with walk-forward validation                        |
| `anomaly_detection`    | Isolation forest, z-score, regime-based anomalies               |
| `regime_ml`            | Regime-enhanced ML: regime labels and probabilities as features |
| `ensemble_strategy`    | Combine RF, GBM, LSTM predictions                               |
| `feature_selection`    | RFE, SHAP, correlation filtering, stability check               |

### Pricing & Fixed Income (6 prompts)

| Prompt                 | Description                                           |
| ---------------------- | ----------------------------------------------------- |
| `option_pricing`       | BS, Greeks, vol smile, Heston calibration             |
| `yield_curve_analysis` | Bootstrap, forward rates, duration, scenario analysis |
| `exotic_pricing`       | Characteristic function and FBSDE pricing             |
| `vol_arbitrage`        | Implied vs realized vol, VRP signal, delta hedging    |
| `convertible_bond`     | Hybrid pricing: bond floor + equity option            |
| `structured_product`   | Principal-protected note pricing and sensitivity      |

### Reporting & Monitoring (14 prompts)

| Prompt                    | Description                                        |
| ------------------------- | -------------------------------------------------- |
| `daily_risk_monitor`      | Daily risk checklist with GREEN/YELLOW/RED status  |
| `weekly_portfolio_review` | Weekly attribution and outlook                     |
| `strategy_tearsheet`      | Full strategy performance tearsheet                |
| `research_summary`        | Summarize workspace: datasets, models, findings    |
| `sentiment_analysis`      | News sentiment scoring and impact                  |
| `fundamental_screen`      | Piotroski F-score quality screening                |
| `microstructure_analysis` | Liquidity, toxicity, market quality                |
| `execution_optimization`  | Optimal execution scheduling and cost              |
| `bayesian_analysis`       | Bayesian inference with credible intervals         |
| `causal_analysis`         | DID, synthetic control, event studies              |
| `monthly_report`          | Comprehensive monthly performance report           |
| `attribution_report`      | Brinson allocation, selection, interaction effects |
| `compliance_check`        | Verify portfolio against risk limits               |
| `client_report`           | Client-facing performance summary                  |

### Execution & Microstructure (5 prompts)

| Prompt                            | Description                                                       |
| --------------------------------- | ----------------------------------------------------------------- |
| `optimal_execution`               | Full execution optimization: schedule, cost model, Almgren-Chriss |
| `market_microstructure_deep_dive` | Comprehensive liquidity, toxicity, and market quality             |
| `dark_pool_analysis`              | Dark pool impact, routing decisions, venue selection              |
| `intraday_trading`                | Intraday strategy with microstructure signals                     |
| `transaction_cost_deep_dive`      | Detailed post-trade TCA with cost attribution                     |

### Econometrics (6 prompts)

| Prompt                           | Description                                                   |
| -------------------------------- | ------------------------------------------------------------- |
| `panel_data_analysis`            | Panel regression: pooled OLS, fixed/random effects, Hausman   |
| `var_model_analysis`             | VAR: lag selection, IRFs, variance decomposition, forecasting |
| `structural_break_analysis`      | Detect structural breaks with sub-period comparison           |
| `cointegration_analysis`         | Full cointegration: Engle-Granger, Johansen, VECM, trading    |
| `event_study_analysis`           | Event study with abnormal returns and statistical tests       |
| `instrumental_variable_analysis` | IV/2SLS for causal identification                             |

### Forex (4 prompts)

| Prompt                       | Description                                               |
| ---------------------------- | --------------------------------------------------------- |
| `fx_pairs_analysis`          | Currency pair dynamics: trend, vol, carry, sessions       |
| `carry_trade_deep_dive`      | Carry trade construction, crash risk, regime conditioning |
| `fx_risk_management`         | Multi-currency hedging and risk analysis                  |
| `currency_strength_analysis` | Multi-currency strength ranking and signals               |

### Data Workflows (4 prompts)

| Prompt               | Description                                                |
| -------------------- | ---------------------------------------------------------- |
| `data_pipeline`      | Fetch, clean, validate, compute returns, store             |
| `data_quality_audit` | Check data for missing values, outliers, structural issues |
| `multi_asset_setup`  | Set up multi-asset workspace with aligned returns          |
| `data_exploration`   | Explore and summarize an unfamiliar dataset                |

### Advanced Math (4 prompts)

| Prompt                        | Description                                                    |
| ----------------------------- | -------------------------------------------------------------- |
| `network_analysis`            | Financial network: correlation networks, centrality, contagion |
| `levy_process_modeling`       | Lévy process calibration and simulation                        |
| `information_theory_analysis` | Entropy, mutual information, transfer entropy                  |
| `optimal_stopping_analysis`   | Optimal stopping for American options and exit timing          |

### Bayesian (5 prompts)

| Prompt                      | Description                                                |
| --------------------------- | ---------------------------------------------------------- |
| `bayesian_portfolio`        | Bayesian portfolio optimization with parameter uncertainty |
| `bayesian_regime_detection` | Bayesian changepoint and regime analysis                   |
| `bayesian_factor_model`     | Bayesian factor model with uncertainty                     |
| `model_selection`           | Bayesian model comparison: WAIC, LOO, Bayes factors        |
| `bayesian_vol`              | Bayesian stochastic volatility estimation                  |

### Tool Guides (116 prompts)

Short, focused guides for individual tools. Each walks an agent through using
a specific tool correctly, including parameter choices and interpretation.
Covers: risk, vol, stats, ts, backtest, ml, price, opt, microstructure,
execution, data, causal, bayes, forex, viz, workspace tools.

### Example: Using a prompt

When connected to wraquant-mcp, an AI agent can load any prompt:

```
Use the equity_deep_dive prompt for AAPL
```

This returns a detailed, multi-phase workflow that guides the agent through
loading data, running statistical analysis, fitting GARCH models, detecting
regimes, computing technical indicators, assessing risk, and synthesizing
findings into an actionable assessment. Each phase explains which tools to
call, what parameters to use, and how to interpret the results.

---

## State Model

### DuckDB Shared State

All tabular data lives in a shared DuckDB database. Every tool that produces
tabular output automatically stores it as a named DuckDB table.

```
~/.wraquant/workspaces/default/
    data.duckdb          # All datasets (zero-copy pandas registration)
    models/              # Fitted models (joblib files)
    notebooks/           # Jupyter notebooks
    snapshots/           # Named workspace checkpoints
    journal.jsonl        # Append-only operation log
    manifest.json        # Workspace metadata
```

### How data flows

1. **Store**: Data enters DuckDB via `store_data`, `compute_returns`,
   or any tool that produces a DataFrame
2. **Reference**: Tools accept dataset names as strings, not raw data.
   Pass `"prices_aapl"` and the tool resolves it from DuckDB automatically
3. **Query**: Use `query_data("SELECT * FROM prices_aapl WHERE rsi < 30")`
   for ad-hoc SQL inspection
4. **Lineage**: Every dataset tracks its parent. `prices_aapl` spawns
   `returns_aapl`, which spawns `features_aapl` -- the full chain is tracked
5. **Versioning**: Name collisions auto-version: `prices_aapl` then
   `prices_aapl_v2`, then `prices_aapl_v3`
6. **Models**: Fitted models (GARCH, HMM, etc.) are stored both in memory
   and as joblib files on disk

### Naming conventions

| Pattern                   | Example                | When                 |
| ------------------------- | ---------------------- | -------------------- |
| `prices_{ticker}`         | `prices_aapl`          | Raw price data       |
| `returns_{ticker}`        | `returns_aapl`         | Computed returns     |
| `features_{name}`         | `features_aapl`        | ML feature matrices  |
| `{model}_{descriptor}`    | `garch_gjr_aapl`       | Fitted models        |
| `{analysis}_{descriptor}` | `var_portfolio`        | Analysis results     |
| `multi_{name}`            | `multi_sector_returns` | Multi-asset datasets |

### Resource ID system

The `IDRegistry` tracks every resource (dataset, model, result) with metadata:

- **Auto-versioning**: Name collisions get `_v2`, `_v3` suffixes
- **Lineage tracking**: Every resource knows its parent and the operation
  that created it
- **Type metadata**: Datasets track rows/columns/dtypes; models track
  model type, source dataset, and fit metrics

---

## Workspace Management

Workspaces are isolated research environments. Each has its own DuckDB
database, models directory, journal, and notebooks.

```python
# Create a new workspace for a specific research topic
create_workspace("pairs_research", description="GLD/GDX pairs trading analysis")

# Switch between workspaces
open_workspace("regime_study")

# List all workspaces with metadata
list_workspaces()

# Checkpoint before risky operations
snapshot("before_rebalance")

# Roll back if needed
restore_snapshot("before_rebalance")

# Run SQL against workspace data
query_data("SELECT date, close, rsi FROM prices_aapl_rsi WHERE rsi < 30")
```

---

## Composition with Other MCPs

wraquant-mcp is designed to compose with existing MCPs through the shared
DuckDB database. Here is the recommended MCP stack:

### OpenBB MCP -- Market Data

OpenBB MCP fetches market data. Store it in DuckDB via `store_data` and
wraquant-mcp can analyze it immediately.

```
Agent: Use OpenBB to fetch AAPL daily prices for the last 5 years
Agent: Store the data as "prices_aapl" in wraquant
Agent: Run equity_deep_dive on AAPL
```

### DuckDB MCP -- SQL Queries

Both wraquant-mcp and DuckDB MCP can connect to the same `.duckdb` file.
The agent can use SQL for data exploration alongside wraquant analysis.

```
Agent: wraquant → compute_returns("prices_aapl")
Agent: DuckDB MCP → SELECT * FROM returns_aapl WHERE returns < -0.03
Agent: wraquant → var_analysis("returns_aapl")
```

### Jupyter MCP -- Notebooks

Notebooks in the workspace directory can connect to the same DuckDB file
for visualization and custom analysis.

```python
# In a Jupyter notebook in the workspace
import duckdb
con = duckdb.connect("~/.wraquant/workspaces/default/data.duckdb")
df = con.sql("SELECT * FROM returns_aapl").df()
```

### Alpaca MCP -- Trade Execution

After wraquant computes optimal weights, Alpaca MCP can execute the trades.

```
Agent: wraquant → optimize_portfolio("multi_asset_returns", method="risk_parity")
Agent: wraquant → execution_cost("portfolio_weights")
Agent: Alpaca MCP → execute trades according to target weights
```

### Claude Desktop config (full stack)

```json
{
  "mcpServers": {
    "wraquant": {
      "command": "wraquant-mcp",
      "env": { "PYTHONUNBUFFERED": "1" }
    },
    "openbb": {
      "command": "openbb-mcp",
      "env": { "OPENBB_TOKEN": "your_token" }
    },
    "duckdb": {
      "command": "duckdb-mcp",
      "args": ["--db", "~/.wraquant/workspaces/default/data.duckdb"]
    }
  }
}
```

---

## Examples

### Example 1: Full Equity Analysis

The agent calls wraquant-mcp tools in sequence:

```
1. workspace_status()
   → No datasets yet

2. store_data("prices_aapl", {"date": [...], "close": [...], "volume": [...]})
   → Stored: prices_aapl (1258 rows, 3 columns)

3. compute_returns("prices_aapl")
   → Stored: returns_aapl (1257 rows)
   → mean=0.0008, std=0.018, annualized_vol=0.286

4. analyze("returns_aapl")
   → Sharpe: 0.72, skewness: -0.34, kurtosis: 7.2
   → ADF p-value: 0.001 (stationary)
   → Jarque-Bera: reject normality

5. fit_garch("returns_aapl", model="GJR", dist="t")
   → Model: garch_returns_aapl_gjr
   → Persistence: 0.971, half-life: 23.5 days
   → Gamma (leverage): 0.089, AIC: -5.23

6. detect_regimes("returns_aapl", n_regimes=2)
   → Current regime: 0 (low-vol bull), probability: 0.87
   → Transition matrix: [[0.97, 0.03], [0.05, 0.95]]
   → Expected bull duration: 33 days, bear duration: 20 days

7. risk_metrics("returns_aapl")
   → Sharpe: 0.72, Sortino: 1.01, max_drawdown: -0.32
   → Hit ratio: 0.54, annualized_return: 0.206
```

### Example 2: Regime-Aware Portfolio

```
1. store_data for SPY, TLT, GLD, VIX returns
   → 4 datasets stored

2. correlation_analysis("multi_asset_returns")
   → SPY-TLT: -0.35 (negative = good hedge)
   → SPY-GLD: 0.05 (low = good diversifier)

3. optimize_portfolio("multi_asset_returns", method="risk_parity")
   → Weights: SPY=0.18, TLT=0.42, GLD=0.40
   → Portfolio vol: 7.2%, Sharpe: 0.89

4. detect_regimes("returns_spy", n_regimes=2)
   → Current: bull regime (probability 0.82)
   → Bull Sharpe: 1.4, Bear Sharpe: -0.8

5. stress_test("portfolio_returns")
   → GFC 2008: -12.3%, COVID 2020: -8.7%
   → Rate hike: -6.1%, Vol spike: -4.2%

6. comprehensive_tearsheet("portfolio_returns")
   → Full report: equity curve, drawdowns, monthly returns, rolling metrics
```

### Example 3: ML Alpha Research

```
1. compute_returns("prices_aapl")

2. build_features("returns_aapl", types=["returns", "volatility", "ta"])
   → 42 features generated, stored as features_aapl

3. train_model("features_aapl", model="gradient_boost", walk_forward=True)
   → 5 walk-forward splits
   → Out-of-sample hit rate: 0.534, Sharpe: 0.61

4. feature_importance("gradient_boost_model")
   → Top 5: vol_21d (0.12), rsi_14 (0.09), return_5d (0.08),
            macd_histogram (0.07), regime_prob (0.06)

5. run_backtest("ml_signals_aapl")
   → Ann. return: 12.3%, Max DD: -18.7%, Sharpe: 0.61

6. risk_metrics("backtest_ml_aapl")
   → Sharpe: 0.61, Sortino: 0.84, hit_ratio: 0.534
```

---

## Project Structure

```
mcp/
    pyproject.toml                    # Package metadata and dependencies
    src/wraquant_mcp/
        __init__.py                   # Package entry point (create_server, main)
        __main__.py                   # python -m wraquant_mcp support
        server.py                     # FastMCP server builder + Tier 1/2 tools
        context.py                    # AnalysisContext: DuckDB state manager
        ids.py                        # Resource ID system (auto-versioning, lineage)
        adaptor.py                    # ToolAdaptor: auto-wraps wraquant functions
        auto_register.py              # Auto-registers ALL wraquant functions as tools
        workspace.py                  # Workspace management tools
        supervisor.py                 # Orchestrator: recommend_workflow, module_guide
        servers/                      # 22 module servers (218 tools total)
            __init__.py               # register_all() aggregator
            risk.py                   # 15 risk tools
            vol.py                    # 11 volatility tools
            data.py                   # 17 data management tools
            microstructure.py         # 16 microstructure tools
            viz.py                    # 14 visualization tools
            math.py                   # 14 advanced math tools
            regimes.py                # 12 regime detection tools
            stats.py                  # 11 statistics tools
            ta.py                     # 11 technical analysis tools
            ts.py                     # 10 time series tools
            backtest.py               # 10 backtesting tools
            execution.py              # 10 execution algorithm tools
            ml.py                     # 9 ML tools
            price.py                  # 9 pricing tools
            opt.py                    # 8 optimization tools
            causal.py                 # 7 causal inference tools
            bayes.py                  # 7 Bayesian tools
            econometrics.py           # 6 econometrics tools
            forex.py                  # 6 forex tools
            experiment.py             # 5 experiment tracking tools
            fundamental.py            # 5 fundamental analysis tools
            news.py                   # 5 news/sentiment tools
        prompts/                      # 16 prompt categories (327 prompts total)
            __init__.py               # register_all_prompts() aggregator
            tools_guide.py            # 226 per-tool usage guides (100% coverage)
            system.py                 # 1 system context prompt
            analysis.py               # 9 analysis workflow prompts
            risk.py                   # 10 risk & volatility prompts
            regime.py                 # 7 regime detection prompts
            portfolio.py              # 9 portfolio construction prompts
            strategy.py               # 9 trading strategy prompts
            ml.py                     # 8 machine learning prompts
            pricing.py                # 6 pricing & fixed income prompts
            reporting.py              # 14 reporting & monitoring prompts
            execution.py              # 5 execution & microstructure prompts
            econometrics.py           # 6 econometrics prompts
            bayes.py                  # 5 Bayesian prompts
            data.py                   # 4 data workflow prompts
            forex.py                  # 4 forex prompts
            math.py                   # 4 advanced math prompts
    tests/                            # 357 tests, 0 failures
        test_e2e.py                   # End-to-end integration tests
        test_*_server.py              # Per-module server tests (22 files)
    examples/
        claude_desktop_config.json    # Claude Desktop configuration
        equity_analysis.md            # Full equity analysis workflow example
        portfolio_construction.md     # Portfolio construction workflow example
```

---

## Development

### Install

```bash
cd mcp/
pdm install
```

### Run tests

```bash
PYTHONPATH=mcp/src:src pdm run pytest mcp/tests/    # 357 tests
```

### Run the server locally

```bash
pdm run wraquant-mcp
```

### Key dependencies

| Package    | Version | Purpose                     |
| ---------- | ------- | --------------------------- |
| `wraquant` | >=0.1.1 | Core quant analysis library |
| `fastmcp`  | >=3.2.0 | MCP server framework        |
| `duckdb`   | >=1.5.1 | Shared state database       |

### Adding a new module server

1. Create `src/wraquant_mcp/servers/mymodule.py`
2. Define `register_mymodule_tools(mcp, ctx)` with `@mcp.tool()` decorated functions
3. Import and call from `servers/__init__.py`
4. Add tests in `tests/test_mymodule_server.py`

### Adding a new prompt category

1. Create `src/wraquant_mcp/prompts/mycategory.py`
2. Define `register_mycategory_prompts(mcp)` with `@mcp.prompt()` decorated functions
3. Import and call from `prompts/__init__.py`

---

## Quality

| Metric              | Value                 |
| ------------------- | --------------------- |
| Hand-crafted tools  | 218                   |
| Prompt templates    | 327                   |
| Tool guide coverage | 218/218 (100%)        |
| Error handling      | 22/22 servers (100%)  |
| Test suite          | 357 tests, 0 failures |
| Test files          | 22 (every server)     |

Every tool function is wrapped in `try/except` with graceful error
returns. Every tool has a corresponding `guide_*` prompt that explains
parameters, interpretation, and follow-up actions.

---

## License

MIT
