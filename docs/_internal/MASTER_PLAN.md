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
- [ ] `prompts/equity_analysis.py` — multi-step equity deep-dive
- [ ] `prompts/pairs_trading.py` — cointegration → spread → backtest
- [ ] `prompts/portfolio_construction.py` — data → optimize → risk decompose
- [ ] `prompts/risk_report.py` — VaR → stress → crisis → factor
- [ ] `prompts/volatility_deepdive.py` — GARCH → compare → forecast → NIC

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
