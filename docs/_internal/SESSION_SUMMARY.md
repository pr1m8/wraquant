# Session Summary — 2026-03-18/19

## What Was Built (wraquant core)

### Numbers
- 97K LOC, 1,097 functions, 3,630+ tests, 27 modules, 265 TA indicators
- Published to PyPI as wraquant v0.1.1
- GitHub Actions CI (tests, coverage, lint, docs, changelog, PyPI publish)
- Sphinx docs with Furo theme, autoapi, tutorials

### Deep Implementations
- **risk/** — 95 functions: beta (6 methods), factor risk, portfolio analytics
  (component/marginal/incremental VaR), tail risk, historical crisis analysis,
  stress testing with 7 built-in scenarios, copulas, DCC, credit, survival
- **vol/** — 28 functions: full GARCH family with diagnostics, Hawkes, stochastic
  vol, realized vol (bipower, jump detection, TSRV, realized kernel)
- **regimes/** — 38 functions: Gaussian HMM, Markov-switching AR, Kalman/UKF,
  RegimeResult dataclass, detect_regimes() unified interface, scoring, labels
- **ta/** — 265 indicators across 19 modules with trading interpretation
- **ml/** — 44 functions: LSTM/GRU/TFT, sklearn pipeline, walk-forward, SHAP
- **price/** — 50 functions: FBSDE, characteristic function pricing, SABR,
  rough Bergomi, CIR, Vasicek
- **ts/** — 52 functions: auto_forecast, SSA/EMD, OU, jump-diffusion, VAR
- **stats/** — 79 functions: robust stats, KDE, distance correlation, VIF
- **backtest/** — 38 functions: 15+ custom metrics, VectorizedBacktest
- **experiment/** — Lab, Experiment, ExperimentResults with CV and persistence
- **+ 17 more modules** (econometrics, microstructure, execution, causal,
  bayes, forex, viz, math, data, io, flow, scale, dashboard, compose, etc.)

### Integration Work Done
- **Coercion** (~95%): core/_coerce.py adopted across most modules
- **Frame types**: PriceSeries, ReturnSeries, OHLCVFrame built and working
- **Result dataclasses**: GARCHResult, BacktestResult, ForecastResult, RegimeResult
  with chaining methods (.to_var(), .plot(), .summary())
- **Cross-module imports**: OLS consolidated (22 lstsq→ols), drawdown/sharpe
  consolidated, backtest→risk, opt→stats, data→frame, risk auto-detects
  ReturnSeries.periods_per_year
- **Compose system**: Workflow + 14 steps + 4 pre-built workflows
- **Recipes**: analyze(), regime_aware_backtest(), garch_risk_pipeline()

### Integration Still TODO
- 11 files still isolated (mostly infra: io, data, frame/ops)
- Some remaining lstsq calls in econometrics, stats/cointegration
- viz/ doesn't auto-compute metrics
- Result dataclasses not returned by all functions yet
- Frame types adopted in data/ and risk/ but not everywhere

---

## Key Decisions Made

### 1. Type System: Coerce-first (DECIDED)
- Pattern: isinstance + coerce_array/coerce_series at function entry
- No plum-dispatch, no custom ExtensionDtype, no beartype in hot paths
- PriceSeries/ReturnSeries as pd.Series subclasses with _metadata
- See: TYPE_SYSTEM_ANALYSIS.md

### 2. Architecture: 6-layer DAG (DECIDED)
- Foundation → Domain → Quantitative → Modeling → Analysis → Application
- Each layer imports only from below
- Cross-layer = lazy imports
- See: ARCHITECTURE.md, MODULE_GRAPH.md

### 3. MCP: Compose, don't duplicate (DECIDED)
- wraquant-mcp = quant analysis engine (GARCH, regimes, risk, TA, etc.)
- Data fetching → OpenBB/EODHD MCP (don't rebuild)
- SQL queries → DuckDB MCP (share the .duckdb file)
- Trade execution → Alpaca MCP (don't rebuild)
- See: MCP_DEFINITIVE_ARCHITECTURE.md

### 4. State: Shared DuckDB file (DECIDED)
- All tabular data in DuckDB, referenced by name/ID
- Fitted models in joblib files, referenced by ID
- Operation journal for lineage + undo
- Workspaces for multi-session persistence
- Notebooks can connect to same DuckDB
- See: MCP_STATE_ARCHITECTURE.md

### 5. Package: Separate wraquant-mcp in mcp/ directory (DECIDED)
- Same repo, separate pyproject.toml
- pip install wraquant-mcp (depends on wraquant + fastmcp + duckdb)
- Feature branch: feat/wraquant-mcp
- See: MONOREPO_STRUCTURE.md

### 6. wrafin refactor: Not yet (DECIDED)
- Current extras pattern works fine for now
- MCP doesn't require restructuring
- Can create wrafin as meta-package later if needed
- See: WRAFIN_REFACTOR_CONSIDERATION.md

### 7. All tools available, tiered discovery (DECIDED)
- Tier 1: Discovery meta-tools (5)
- Tier 2: Common operations (30, loaded by default)
- Tier 3: Full module tools (200+, lazy loaded on demand)
- Agent asks "what risk tools exist?" before calling them

### 8. ID system for everything (DECIDED)
- Datasets: "prices_aapl", "returns_aapl_v2"
- Models: "garch_aapl_gjr_t"
- Results: "backtest_rsi_regime"
- Auto-versioning on update
- Lineage tracking through journal

---

## What's Next (Priority Order)

### Immediate (next session)
1. Build wraquant-mcp core on feat/wraquant-mcp branch:
   - server.py (FastMCP entry point)
   - context.py (AnalysisContext with DuckDB)
   - adaptor.py (auto-wraps wraquant functions as tools)
   - ids.py (ID registry + resolution)
   - workspace.py (create/open/list/snapshot)

2. Build first module servers:
   - servers/risk.py, servers/vol.py, servers/ta.py
   - Test with MCP Inspector

3. Build prompt templates:
   - equity_analysis, risk_report, portfolio_construction

### After MCP is working
4. Test composition with DuckDB MCP + OpenBB MCP
5. LangChain integration test
6. Dashboard connecting to MCP instead of wraquant directly

### Ongoing (wraquant core)
7. Remaining integration (11 isolated files, result type adoption)
8. Paramspec consistency (float conversion, naming)
9. Viz decorator for auto-plotting indicators
10. More docstring depth where needed

---

## Research Completed (saved in docs/_internal/)

- TYPE_SYSTEM_ANALYSIS.md — singledispatch vs plum vs isinstance patterns
- MCP_STATE_ARCHITECTURE.md — DuckDB hybrid, session persistence
- wraquant-mcp-design.md — FastMCP + LangChain architecture (60 tools)
- MCP_DEFINITIVE_ARCHITECTURE.md — final architecture with composition
- MONOREPO_STRUCTURE.md — package separation + ID system
- WRAFIN_REFACTOR_CONSIDERATION.md — umbrella package analysis
- Financial MCP landscape (OpenBB, QuantConnect, Alpaca, DuckDB, Pandas, Jupyter MCPs)
