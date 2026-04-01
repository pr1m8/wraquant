# wraquant-mcp — Definitive Architecture

**Status:** APPROVED — Ready to implement on feature branch

## Core Principle

wraquant-mcp is the **quant analysis engine** MCP server. It does NOT
fetch data (OpenBB does that) or execute trades (Alpaca does that).
It takes data in and produces deep quantitative analysis out.

## Composition Pattern

```
Agent (Claude / LangChain / any MCP client)
  ├── OpenBB MCP      → data fetching (prices, fundamentals, macro)
  ├── DuckDB MCP      → SQL queries on shared state
  ├── wraquant MCP    → quant analysis (GARCH, regimes, risk, TA, etc.)
  ├── Jupyter MCP     → notebook interaction (optional)
  └── Alpaca MCP      → trade execution (optional)
```

All share the same DuckDB file as the state layer.

## Shared State: DuckDB File

```
~/.wraquant/workspaces/{name}/
├── data.duckdb           # ALL tabular data lives here
├── manifest.json         # Workspace metadata
├── journal.jsonl         # Operation log (append-only)
├── models/               # Fitted models (joblib serialized)
│   ├── garch_v1.joblib
│   └── hmm_2state.joblib
├── notebooks/            # Jupyter notebooks for this workspace
│   └── analysis.ipynb
└── snapshots/            # Named checkpoints
```

- DuckDB MCP reads/writes SQL against `data.duckdb`
- wraquant MCP reads datasets, computes, writes results back to `data.duckdb`
- wraquant MCP stores fitted models in `models/` (not tabular → joblib)
- Jupyter notebooks can `import duckdb; con = duckdb.connect("data.duckdb")`
  for the same data — human and agent see identical state

## Data Flow (Hybrid SQL + Operations)

```
1. [OpenBB MCP] fetch_prices("AAPL") → writes "prices_aapl" table
2. [DuckDB MCP] "SELECT * FROM prices_aapl LIMIT 5" → agent sees schema
3. [wraquant MCP] compute_indicator("prices_aapl", "rsi", period=14)
                  → reads table, computes, writes "prices_aapl_rsi" table
4. [DuckDB MCP] "SELECT date, close, rsi FROM prices_aapl_rsi
                  WHERE rsi < 30" → agent sees oversold dates
5. [wraquant MCP] fit_garch("prices_aapl_rsi", column="close")
                  → stores model in models/, writes "garch_diagnostics" table
6. [DuckDB MCP] "SELECT persistence, half_life FROM garch_diagnostics"
7. [wraquant MCP] detect_regimes("prices_aapl_rsi", method="hmm")
                  → writes "regime_probs" and "regime_stats" tables
```

## What wraquant MCP Builds (vs Reuses)

### DO NOT BUILD (reuse existing MCPs):
- Data fetching → OpenBB / EODHD / Alpha Vantage
- SQL queries on DataFrames → DuckDB MCP
- DataFrame preview/schema → DuckDB MCP
- Trade execution → Alpaca MCP
- Notebook interaction → Jupyter MCP

### BUILD (wraquant's unique value):
1. **Quant operation tools** (~100 tools covering all 27 modules)
   - Each tool: reads from DuckDB → calls wraquant → writes result to DuckDB
   - Named operations, not code execution (safe)

2. **Model management tools**
   - fit_model, forecast, compare_models, model_info
   - Models stored as joblib files, referenced by name

3. **Workspace tools**
   - create/open/list/snapshot/restore workspace
   - Workspace history (journal)
   - Research notes

4. **Discovery tools**
   - list_modules, list_tools(module), describe_tool(name)
   - Agent asks "what risk tools exist?" before calling them

5. **Prompt templates**
   - Guided multi-step workflows (equity analysis, pairs trading,
     portfolio construction, risk report, vol deep-dive)

## Tool Design

Every quant tool follows the same pattern:

```python
@mcp.tool()
async def compute_indicator(
    dataset: str,        # Name of DuckDB table
    indicator: str,      # e.g., "rsi", "macd", "bollinger_bands"
    column: str = "close",
    period: int = 14,
    ctx: Context = None,
) -> dict:
    """Compute a technical indicator and add it to the dataset.

    Reads the specified column from the dataset, computes the
    indicator using wraquant.ta, and stores the result as a new
    table (original + indicator column).
    """
    # 1. Read from shared DuckDB
    df = ctx.state.db.sql(f"SELECT * FROM {dataset}").df()

    # 2. Compute using wraquant
    from wraquant.ta import get_indicator  # dispatcher
    result = get_indicator(indicator, df[column], period=period)

    # 3. Add to DataFrame
    df[indicator] = result

    # 4. Write back to DuckDB
    new_name = f"{dataset}_{indicator}"
    ctx.state.db.register(new_name, df)

    # 5. Log operation
    ctx.state.log(op="compute_indicator", input=dataset, output=new_name)

    # 6. Return metadata (not data)
    return {
        "dataset": new_name,
        "rows": len(df),
        "new_column": indicator,
        "summary": {
            "mean": float(result.mean()),
            "min": float(result.min()),
            "max": float(result.max()),
        },
    }
```

## All Tools (not just curated 30)

Every wraquant function is accessible. But organized in tiers:

**Tier 1: Discovery (always available, ~5 tools)**
- list_modules, list_tools, describe_tool, workspace_status, help

**Tier 2: Common operations (~30 tools, loaded by default)**
- compute_returns, compute_indicator, fit_garch, detect_regimes,
  risk_metrics, optimize_portfolio, backtest, forecast, analyze

**Tier 3: Full module tools (~200+ tools, loaded on demand)**
- Agent calls list_tools("risk.copulas") → server loads copula tools
- Lazy loading prevents context bloat
- Every function in wraquant.__all__ is available

## Output: Always Metadata, Never Raw Data

Tools return:
- Dataset ID (for SQL queries via DuckDB MCP)
- Shape (rows, columns)
- Summary statistics
- Key metrics (persistence, sharpe, etc.)
- Model ID (for fitted models)

The agent uses DuckDB MCP to inspect actual data when needed.

## Notebook Integration

Notebooks in the workspace can:
```python
import duckdb
import wraquant as wq

# Same DuckDB the MCP uses
con = duckdb.connect("~/.wraquant/workspaces/my_research/data.duckdb")

# Query what the agent created
df = con.sql("SELECT * FROM prices_aapl_rsi WHERE rsi < 30").df()

# Use wraquant directly
result = wq.vol.garch_fit(df["close"].pct_change().dropna())

# Write back (agent sees it too)
con.register("garch_from_notebook", result_df)
```

## Float Conversion

All numeric outputs wrapped in `float()` for JSON serialization:
- No np.float64, np.int64 in tool responses
- Coerce at the adaptor layer, not in wraquant core

## Implementation Plan

1. Create feature branch: `git checkout -b feat/wraquant-mcp`
2. Create `src/wraquant_mcp/` package structure
3. Build AnalysisContext (DuckDB state manager)
4. Build ToolAdaptor (auto-wraps wraquant functions)
5. Build module servers (one per wraquant module)
6. Build discovery + workspace tools
7. Build prompt templates
8. Test with MCP Inspector
9. Test with Claude Desktop
10. Test composition with DuckDB MCP + OpenBB MCP
