# wraquant — Master Plan

**Version:** 1.0 | **Date:** 2026-04-01 | **Author:** William Astley

---

## Where We Are

### The Library (wraquant v0.1.1 — published on PyPI)
- 97K LOC, 1,097 functions, 3,630+ tests, 27 modules, 265 TA indicators
- Deep implementations: GARCH family, HMM regimes, FBSDE pricing, 95 risk functions
- Coerce-first type system (~95% adopted)
- Financial frame types (PriceSeries, ReturnSeries, OHLCVFrame)
- Result dataclasses with chaining (GARCHResult, RegimeResult, etc.)
- Experiment lab, Streamlit dashboard, compose workflows
- Sphinx docs with Furo theme, tutorials, API reference
- GitHub Actions CI/CD with trusted PyPI publishing

### Known Problems
1. **Integration gaps** — 11 files still isolated, some cross-module imports missing
2. **Paramspec inconsistency** — `period` vs `window`, `seed` vs `random_state`
3. **Float coercion** — some functions return np.float64 instead of native float
4. **Result types** — GARCHResult works, but backtest/forecast don't fully use theirs
5. **Frame adoption** — PriceSeries/ReturnSeries built but only used in data/ and risk/
6. **Viz-indicator linkage** — no auto-plot decorator for TA indicators
7. **Some docstrings still thin** — especially in io/, data/, frame/

---

## The Vision

```
┌─────────────────────────────────────────────────────┐
│                   USER LAYER                         │
│                                                      │
│  Human (Jupyter/Dashboard)    AI Agent (Claude/LC)   │
│         │                            │               │
│    import wraquant              MCP Protocol         │
│         │                            │               │
└─────────┼────────────────────────────┼───────────────┘
          │                            │
┌─────────▼────────────────────────────▼───────────────┐
│                   ACCESS LAYER                        │
│                                                      │
│  wraquant (Python API)    wraquant-mcp (MCP Server)  │
│  pip install wraquant     pip install wraquant-mcp    │
│                                                      │
│  Compose with:                                       │
│  • OpenBB MCP (data)     • DuckDB MCP (SQL state)   │
│  • Alpaca MCP (trades)   • Jupyter MCP (notebooks)  │
└─────────┬────────────────────────────────────────────┘
          │
┌─────────▼────────────────────────────────────────────┐
│                   STATE LAYER                         │
│                                                      │
│  Workspace (~/.wraquant/workspaces/{name}/)           │
│  ├── data.duckdb        (shared tabular state)       │
│  ├── models/            (fitted models, joblib)      │
│  ├── journal.jsonl      (operation log, lineage)     │
│  ├── notebooks/         (Jupyter, linked to state)   │
│  └── snapshots/         (checkpoints)                │
│                                                      │
│  ID System: everything referenced by name            │
│  "prices_aapl" → "returns_aapl" → "garch_v1"       │
└──────────────────────────────────────────────────────┘
```

---

## Module Analysis Matrix

| Module | Funcs | Complexity | Domain | Process Stage | MCP Tools | Prompts Using |
|--------|-------|-----------|--------|---------------|-----------|---------------|
| **data/** | 41 | Low | Infrastructure | 1. Ingest | 5 | All (data entry) |
| **io/** | 21 | Low | Infrastructure | 1. Ingest | 3 | reporting |
| **frame/** | 10 | Medium | Infrastructure | 1. Ingest | 0 (internal) | N/A |
| **core/** | 11 | Low | Infrastructure | All | 0 (internal) | N/A |
| **ta/** | 265 | Low-Med | Signals | 2. Analyze | 1 (dispatcher) | momentum, mean_rev, trend |
| **stats/** | 79 | Medium | Analysis | 2. Analyze | 8 | equity, pairs, macro |
| **ts/** | 52 | Medium | Analysis | 2. Analyze | 6 | forecast, decompose |
| **econometrics/** | 34 | High | Analysis | 2. Analyze | 5 | event_study, policy, granger |
| **vol/** | 28 | High | Modeling | 3. Model | 6 | vol_deep_dive, risk_report |
| **risk/** | 95 | High | Risk | 3. Model | 10 | risk_report, stress, tail |
| **regimes/** | 38 | High | Modeling | 3. Model | 5 | regime_*, market_monitor |
| **ml/** | 44 | High | Modeling | 3. Model | 6 | ml_alpha, feature_eng |
| **bayes/** | 29 | High | Modeling | 3. Model | 4 | bayesian_* |
| **opt/** | 26 | Medium | Decision | 4. Decide | 5 | portfolio_*, asset_alloc |
| **price/** | 50 | High | Pricing | 3. Model | 5 | option_*, yield_curve |
| **backtest/** | 38 | Medium | Validation | 5. Validate | 5 | all strategy prompts |
| **experiment/** | 13 | Medium | Validation | 5. Validate | 3 | hp_sweep, model_compare |
| **microstructure/** | 33 | High | Market | 2. Analyze | 4 | liquidity, execution |
| **execution/** | 21 | Medium | Execution | 6. Execute | 4 | execution_opt, rebalance |
| **causal/** | 19 | High | Analysis | 2. Analyze | 4 | event_study, policy |
| **forex/** | 23 | Medium | Domain | 2. Analyze | 3 | macro, carry_trade |
| **math/** | 55 | High | Foundation | Support | 2 | exotic_pricing, network |
| **viz/** | 47 | Medium | Output | 7. Report | 5 | all (chart output) |
| **dashboard/** | 15 | Medium | Output | 7. Report | 0 (UI) | N/A |
| **flow/** | 8 | Low | Infra | Support | 2 | pipeline orchestration |
| **scale/** | 10 | Medium | Infra | Support | 2 | parallel_backtest |
| **compose/** | 14 | Medium | Orchestration | All | 1 | run_workflow |

### Process Stages (how modules link in a workflow):

```
1. INGEST    →  data/, io/, frame/
                 ↓
2. ANALYZE   →  stats/, ts/, ta/, econometrics/, microstructure/, causal/, forex/
                 ↓
3. MODEL     →  vol/, risk/, regimes/, ml/, bayes/, price/
                 ↓
4. DECIDE    →  opt/ (portfolio weights, allocation)
                 ↓
5. VALIDATE  →  backtest/, experiment/ (test the decision)
                 ↓
6. EXECUTE   →  execution/ (trade scheduling, cost)
                 ↓
7. REPORT    →  viz/, dashboard/ (communicate results)
```

### Complexity Guide:
- **Low**: Pure computation, simple I/O, no optimization
- **Medium**: Statistical estimation, moderate math, some iteration
- **High**: Optimization loops, MCMC, matrix decomposition, MLE fitting

---

## Phase 1: Polish wraquant Core (1-2 sessions)

### 1A. Fix remaining integration gaps
- [ ] Wire 11 remaining isolated files (io, data, frame/ops, experiment/cv)
- [ ] Replace remaining lstsq calls in econometrics/ (panel, cross_section, event_study)
- [ ] Replace remaining lstsq in stats/cointegration
- [ ] Ensure all numeric outputs are native `float()` not np.float64
- [ ] Verify all `__init__.py` have proper `__all__` exports

### 1B. Paramspec consistency pass
- [ ] Document the convention: `period` in ta/, `window` in vol/stats
- [ ] Standardize `seed` (not `random_state`) — it's the numpy convention
- [ ] Standardize `risk_free` (not `rf` or `rf_rate`)
- [ ] Add deprecation aliases where names differ

### 1C. Result type completion
- [ ] backtest/engine.py Backtest.run() → return BacktestResult consistently
- [ ] ts/forecasting.py auto_forecast → return ForecastResult consistently
- [ ] vol/models.py — verify GARCHResult is returned (already done)
- [ ] Add `__getitem__` compat to all result dataclasses (already done)

### 1D. Frame type adoption
- [ ] risk/metrics auto-detect ReturnSeries.periods_per_year (already done)
- [ ] data/loaders return PriceSeries/OHLCVFrame (already done)
- [ ] ta/ indicators accept PriceSeries (works via coercion)
- [ ] vol/realized accept PriceSeries
- [ ] Document the pattern in ARCHITECTURE.md

### 1E. Float coercion
- [ ] Audit all functions returning dict values — wrap in float()
- [ ] Critical for MCP (JSON serialization fails on np.float64)
- [ ] Can be systematic: grep for common patterns

### 1F. Viz decorator for TA indicators
- [ ] Create `@plotable` decorator or registry
- [ ] Each indicator type (oscillator, overlay, band) gets appropriate chart
- [ ] `indicator.plot()` returns Plotly figure
- [ ] Integrate with dashboard TA screener

---

## Phase 2: Build wraquant-mcp (2-3 sessions)

### 2A. Core infrastructure (session 1)
- [ ] `mcp/src/wraquant_mcp/server.py` — FastMCP server with mount composition
- [ ] `context.py` — AnalysisContext (DuckDB state manager)
- [ ] `adaptor.py` — Auto-wrap wraquant functions as MCP tools
- [ ] `ids.py` — ResourceID, IDRegistry, auto-versioning, lineage
- [ ] `workspace.py` — create/open/list/snapshot/restore
- [ ] `registry.py` — Tool discovery (list_modules, list_tools, describe_tool)

### 2B. Module servers (session 2)
- [ ] `servers/risk.py` — risk_metrics, var_analysis, stress_test, beta, factor
- [ ] `servers/vol.py` — fit_garch, forecast_vol, news_impact, model_selection
- [ ] `servers/regimes.py` — detect_regimes, regime_stats, regime_portfolio
- [ ] `servers/ta.py` — compute_indicator (dispatches to 265 indicators)
- [ ] `servers/stats.py` — summary_stats, correlation, distribution_fit, regression
- [ ] `servers/ts.py` — forecast, decompose, stationarity_test
- [ ] `servers/opt.py` — optimize_portfolio (MVO, RP, BL, HRP)
- [ ] `servers/backtest.py` — run_backtest, metrics, tearsheet
- [ ] `servers/price.py` — price_option, simulate_process, yield_curve
- [ ] `servers/ml.py` — build_features, walk_forward, train_model
- [ ] `servers/viz.py` — plot tools returning PNG via Image()
- [ ] `servers/data.py` — workspace management, dataset operations

### 2C. Prompt templates (session 2-3)

#### Analysis & Research
- [ ] `equity_deep_dive` — full single-stock analysis: stats → distribution → vol → regimes → TA signals → risk → report
- [ ] `sector_comparison` — compare multiple stocks/ETFs: relative performance → correlation → regime co-movement → factor exposure
- [ ] `macro_analysis` — macro regime: yield curve → rate sensitivity → cross-asset correlations → recession indicators
- [ ] `earnings_impact` — event study around earnings: pre/post returns → abnormal returns → vol spike → recovery
- [ ] `ipo_analysis` — post-IPO behavior: return distribution → vol decay → regime settling → institutional flow

#### Volatility & Risk
- [ ] `volatility_deep_dive` — GARCH → model selection → forecast → NIC → term structure → realized vs implied
- [ ] `risk_report` — full portfolio risk: VaR → CVaR → component VaR → stress test → crisis scenarios → factor decomposition
- [ ] `tail_risk_assessment` — EVT → tail index → CDaR → Cornish-Fisher → copula tail dependence → contagion
- [ ] `stress_test_battery` — run all 7 built-in scenarios + custom → compare → rank by severity → hedging suggestions
- [ ] `correlation_breakdown` — DCC → rolling correlation → regime-conditional → contagion analysis → diversification score
- [ ] `vol_surface_analysis` — implied vol → SABR calibration → skew → term structure → smile dynamics

#### Regime & Market State
- [ ] `regime_detection` — HMM → GMM → Markov-switching → compare methods → current state → transition probabilities
- [ ] `market_regime_monitor` — current regime → historical comparison → expected duration → allocation implications
- [ ] `regime_backtest` — detect regimes → regime-conditional strategy → compare to unconditional → regime-aware sizing
- [ ] `changepoint_analysis` — PELT → binary segmentation → CUSUM → identify structural breaks → pre/post comparison

#### Portfolio & Optimization
- [ ] `portfolio_construction` — data → covariance → optimize (MVO/RP/BL/HRP) → risk decompose → regime adjust → report
- [ ] `portfolio_rebalance` — current weights → drift analysis → rebalance cost → optimal trade schedule → execution plan
- [ ] `factor_attribution` — factor exposure → Fama-French → risk contribution → alpha decomposition → tracking error
- [ ] `portfolio_stress_test` — existing portfolio → crisis scenarios → correlation stress → liquidity stress → margin analysis
- [ ] `asset_allocation` — multi-asset → efficient frontier → regime-aware → BL with views → risk parity comparison

#### Trading & Strategy
- [ ] `pairs_trading` — cointegration test → spread analysis → half-life → entry/exit signals → backtest → risk
- [ ] `momentum_strategy` — RSI/MACD/ROC signals → combine → regime filter → backtest → walk-forward → metrics
- [ ] `mean_reversion` — stationarity test → OU fit → half-life → Bollinger entry/exit → backtest → regime breakdown
- [ ] `trend_following` — MA crossover → ADX filter → PSAR stops → position sizing → backtest → drawdown analysis
- [ ] `statistical_arbitrage` — PCA factors → residual alpha → z-score signals → backtest → transaction costs → capacity

#### Machine Learning
- [ ] `ml_alpha_research` — features → labels → purged CV → walk-forward → feature importance → SHAP → financial metrics
- [ ] `feature_engineering` — price → returns → TA features → vol features → regime features → correlation → interaction terms
- [ ] `model_comparison` — RF vs GBM vs LSTM → walk-forward each → compare Sharpe/hit rate → ensemble → best model
- [ ] `hyperparameter_sweep` — experiment lab → grid search → stability analysis → parameter sensitivity → best config

#### Pricing & Fixed Income
- [ ] `option_pricing` — BS → Greeks → vol smile → Heston calibration → FBSDE → comparison
- [ ] `yield_curve_analysis` — bootstrap → interpolation → forward rates → duration/convexity → rate scenario
- [ ] `exotic_pricing` — characteristic function → FFT/COS → Lévy process comparison → American via FBSDE

#### Microstructure & Execution
- [ ] `liquidity_analysis` — spread → depth → Amihud → Kyle lambda → VPIN → toxicity score → execution cost
- [ ] `execution_optimization` — TWAP vs VWAP vs IS → Almgren-Chriss → impact model → cost analysis
- [ ] `market_quality` — variance ratio → efficiency → information share → intraday pattern

#### Causal & Econometric
- [ ] `event_study` — define event → estimation window → CAR → cross-sectional test → significance
- [ ] `policy_impact` — DID → parallel trends → synthetic control → placebo tests → effect size
- [ ] `granger_analysis` — pairwise Granger → network → lead-lag → causality graph

#### Bayesian
- [ ] `bayesian_portfolio` — posterior returns → BL with uncertainty → credible intervals on weights → model comparison
- [ ] `bayesian_regime` — Bayesian HMM → posterior regime probs → uncertainty on transition matrix

#### Reporting & Monitoring
- [ ] `daily_risk_monitor` — current VaR → regime check → correlation change → stress breach → alert summary
- [ ] `weekly_portfolio_review` — performance → attribution → rebalance signal → risk budget → regime outlook
- [ ] `strategy_tearsheet` — full tearsheet → monthly returns → drawdown table → regime breakdown → comparison
- [ ] `research_summary` — summarize workspace: datasets, models, experiments, key findings

### 2D. Testing (session 3)
- [ ] Test with MCP Inspector (npx @modelcontextprotocol/inspector)
- [ ] Test with Claude Desktop (stdio transport)
- [ ] Test composition: wraquant-mcp + DuckDB MCP together
- [ ] Test composition: wraquant-mcp + OpenBB MCP for data
- [ ] LangChain integration test (langchain-mcp-adapters)
- [ ] Publish wraquant-mcp to PyPI

---

## Phase 3: Dashboard Evolution (1 session)

### 3A. Dashboard connects to MCP
- [ ] Dashboard calls wraquant-mcp server instead of wraquant directly
- [ ] Shared DuckDB state — dashboard and MCP see same data
- [ ] Dashboard shows workspace contents (datasets, models, journal)

### 3B. Enhanced dashboard pages
- [ ] Workspace browser (list, open, compare workspaces)
- [ ] Dataset explorer (SQL queries, preview, lineage graph)
- [ ] Model registry (compare fitted models, diagnostics)
- [ ] Journal viewer (operation history, undo)

### 3C. Viz registry
- [ ] `@dashboard_panel("risk")` decorator on viz functions
- [ ] Auto-registers viz functions as dashboard components
- [ ] Panel/Dash consideration for more flexible layout (vs Streamlit)

---

## Phase 4: Production Hardening (1-2 sessions)

### 4A. Performance
- [ ] Benchmark TA indicators (coercion overhead on hot paths)
- [ ] Profile GARCH fitting, regime detection
- [ ] Torch overloads where GPU helps (large correlation matrices)
- [ ] Caching layer for expensive computations

### 4B. Error handling
- [ ] All functions raise WQError subclasses (not bare ValueError)
- [ ] MCP tools never crash — always return error dicts
- [ ] Validation at API boundaries (Pydantic models for MCP inputs)

### 4C. Documentation
- [ ] ReadTheDocs live and rendering properly (autoapi working)
- [ ] All 27 modules have tutorial-quality API pages
- [ ] Example notebooks in examples/ actually run
- [ ] Changelog auto-generates on release

### 4D. Testing
- [ ] pytest-cov ≥ 80% coverage target
- [ ] Property-based tests (hypothesis) for core math functions
- [ ] Integration tests for cross-module workflows
- [ ] MCP end-to-end tests

---

## Phase 5: Ecosystem (ongoing)

### 5A. Community
- [ ] Contributing guide
- [ ] Issue templates
- [ ] Discussion forum or Discord
- [ ] Example strategies/notebooks

### 5B. Extensions
- [ ] wraquant-dash (enhanced dashboard, separate package if needed)
- [ ] Notebook templates for common workflows
- [ ] Claude Desktop config generator
- [ ] LangChain agent examples

### 5C. Scaling
- [ ] wrafin umbrella package (if justified by adoption)
- [ ] Hosted wraquant-mcp (Docker, cloud deployment)
- [ ] Multi-user workspace support
- [ ] API authentication for hosted MCP

---

## Decision Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Type system | Coerce-first (isinstance + coerce_*) | sklearn pattern, fast, explicit |
| Frame types | pd.Series subclasses with _metadata | Lightweight, backward compat |
| MCP composition | Don't duplicate data/execution MCPs | OpenBB, DuckDB, Alpaca exist |
| State management | Shared DuckDB file + joblib models | Zero-copy pandas, SQL-queryable |
| Package structure | Same repo, separate pyproject | Simple, one PR workflow |
| Tool exposure | All tools via tiered discovery | Agents choose what to load |
| ID system | Human-readable, auto-versioned | "prices_aapl_v2" not UUIDs |
| wrafin refactor | Not yet | Current extras pattern works |
| Notebooks | In workspace, same DuckDB | Human + agent see same state |
| Float coercion | native float() for all outputs | JSON serialization for MCP |

---

## File Index (docs/_internal/)

| File | What it covers |
|------|---------------|
| MASTER_PLAN.md | This document — the single source of truth |
| ARCHITECTURE.md | Module boundaries, 6-layer DAG, conventions |
| MODULE_GRAPH.md | Data flow patterns, integration points |
| MODULE_STATUS.md | Per-module maturity assessment |
| INTEGRATION_INDEX.md | 5-level integration status map |
| COERCION_ADOPTION_STATUS.md | Which modules use core/_coerce |
| TYPE_SYSTEM_ANALYSIS.md | Coerce-first decision + frame redesign |
| MCP_DEFINITIVE_ARCHITECTURE.md | Final MCP architecture |
| MCP_STATE_ARCHITECTURE.md | DuckDB hybrid state design |
| wraquant-mcp-design.md | FastMCP + LangChain patterns |
| MONOREPO_STRUCTURE.md | Package separation + ID system |
| WRAFIN_REFACTOR_CONSIDERATION.md | Umbrella package (not yet) |
| SESSION_SUMMARY.md | This session's work summary |
