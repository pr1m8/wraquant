# wraquant-mcp: Design Document

> Comprehensive architecture for exposing wraquant as an MCP (Model Context Protocol) server,
> consumable by Claude Desktop, LangChain/LangGraph agents, and any MCP-compliant client.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Technology Landscape](#2-technology-landscape)
3. [FastMCP Server Architecture](#3-fastmcp-server-architecture)
4. [LangChain Integration Pattern](#4-langchain-integration-pattern)
5. [Tool Design by Module](#5-tool-design-by-module)
6. [Prompt Templates for Common Workflows](#6-prompt-templates-for-common-workflows)
7. [State Management](#7-state-management)
8. [Visualization and Rich Output](#8-visualization-and-rich-output)
9. [Authentication and Security](#9-authentication-and-security)
10. [Deployment Options](#10-deployment-options)
11. [Project Structure](#11-project-structure)
12. [Implementation Roadmap](#12-implementation-roadmap)

---

## 1. Executive Summary

**wraquant-mcp** is a standalone package that wraps wraquant's 25+ modules into a FastMCP
server, exposing 50-80 curated tools, dynamic resources, and guided prompt templates. It
enables AI agents (Claude, LangChain/LangGraph, OpenAI Agents SDK, or any MCP client) to
perform end-to-end quantitative finance analysis through natural language.

### Key design principles

- **Curated, not exhaustive**: Expose ~60 high-value tools, not all 500+ wraquant functions.
  Each tool wraps a complete workflow step, not a raw function call.
- **Composable via server mounting**: One parent FastMCP server mounts per-module child
  servers with namespaces (`data_*`, `risk_*`, `vol_*`, etc.).
- **State-first**: A session-scoped `AnalysisContext` holds DataFrames, fitted models, and
  intermediate results so tools can chain without re-fetching data.
- **Dual transport**: stdio for local Claude Desktop/CLI usage; Streamable HTTP for hosted
  deployment and LangChain consumption.
- **LangChain-native**: `langchain-mcp-adapters` converts all tools to LangChain `BaseTool`
  instances with zero additional code.

---

## 2. Technology Landscape

### 2.1 FastMCP (v3.x)

FastMCP is the official high-level Python framework for building MCP servers, now maintained
under PrefectHQ and bundled into the MCP Python SDK. Key capabilities used in this design:

| Feature | How we use it |
|---|---|
| `@mcp.tool()` decorator | Register each wraquant operation as a typed, documented tool |
| `@mcp.resource()` decorator | Expose stored DataFrames, model params, and configs as URI-addressable resources |
| `@mcp.prompt()` decorator | Guided multi-step analysis workflows |
| `Context` dependency injection | Access session state, logging, progress reporting within tools |
| `server.mount(child, namespace=...)` | Compose per-module servers into a single parent |
| `FileSystemProvider` | Auto-discover tool files from a directory (dev convenience) |
| Tags and `enable()`/`disable()` | Let users control which module groups are active |
| `Image` return type | Return Plotly charts as base64 PNG |
| Bearer token / API key auth | Secure hosted deployments |
| Streamable HTTP transport | Production deployment with SSE streaming |

**Version**: Target FastMCP >= 3.0 (GA January 2026). Leverage the provider/transform
architecture, component versioning, and authorization system.

### 2.2 LangChain MCP Adapters

The `langchain-mcp-adapters` package (by LangChain, released March 2025) provides:

| Component | Purpose |
|---|---|
| `load_mcp_tools(session)` | Convert all MCP tools from a session into `langchain.tools.BaseTool` instances |
| `convert_mcp_tool_to_langchain_tool()` | Convert a single MCP tool (for selective loading) |
| `MultiServerMCPClient` | Connect to multiple MCP servers (stdio, SSE, HTTP) and aggregate tools |
| Schema conversion | MCP input schemas become Pydantic models automatically |
| Result transformation | MCP content blocks (text, image, embedded resource) become `ToolMessage` |

**Key insight**: wraquant-mcp does NOT need to know about LangChain. It is a standard MCP
server. LangChain consumes it via the adapters package with zero wraquant-specific code.

### 2.3 MCP Protocol Fundamentals

MCP defines three primitives:

1. **Tools** — Functions the LLM can invoke. Accept typed JSON input, return
   `TextContent | ImageContent | EmbeddedResource`. This is the primary interface.
2. **Resources** — URI-addressable data the LLM can read. Static (`config://version`) or
   templated (`data://dataframe/{name}`). Used to inspect stored state.
3. **Prompts** — Reusable message templates that guide multi-step workflows. The LLM
   requests a prompt, gets back a structured message sequence.

---

## 3. FastMCP Server Architecture

### 3.1 Composable server design

wraquant-mcp uses FastMCP's `mount()` to compose focused sub-servers into a unified parent:

```python
from fastmcp import FastMCP

# Parent server
mcp = FastMCP(
    "wraquant",
    description="Quantitative finance analysis toolkit",
)

# Import and mount module servers
from wraquant_mcp.servers.data import data_server
from wraquant_mcp.servers.stats import stats_server
from wraquant_mcp.servers.risk import risk_server
from wraquant_mcp.servers.vol import vol_server
from wraquant_mcp.servers.ta import ta_server
from wraquant_mcp.servers.opt import opt_server
from wraquant_mcp.servers.regimes import regimes_server
from wraquant_mcp.servers.ts import ts_server
from wraquant_mcp.servers.backtest import backtest_server
from wraquant_mcp.servers.price import price_server
from wraquant_mcp.servers.viz import viz_server
from wraquant_mcp.servers.forex import forex_server

mcp.mount(data_server,    namespace="data")
mcp.mount(stats_server,   namespace="stats")
mcp.mount(risk_server,    namespace="risk")
mcp.mount(vol_server,     namespace="vol")
mcp.mount(ta_server,      namespace="ta")
mcp.mount(opt_server,     namespace="opt")
mcp.mount(regimes_server, namespace="regimes")
mcp.mount(ts_server,      namespace="ts")
mcp.mount(backtest_server,namespace="backtest")
mcp.mount(price_server,   namespace="price")
mcp.mount(viz_server,     namespace="viz")
mcp.mount(forex_server,   namespace="forex")
```

This produces tools like `data_fetch_prices`, `risk_sharpe_ratio`, `vol_garch_fit`, etc.
The parent server exposes cross-cutting tools (analyze, list datasets, session management).

### 3.2 Context and state injection

Every tool receives an injected `Context` object for logging, progress, and state:

```python
from fastmcp import FastMCP, Context
from wraquant_mcp.state import AnalysisContext

data_server = FastMCP("wraquant-data")

@data_server.tool(
    tags={"data", "fetch"},
    description="Fetch OHLCV price data for a ticker symbol and store it in the session."
)
async def fetch_prices(
    ticker: str,
    start: str = "2020-01-01",
    end: str | None = None,
    interval: str = "1d",
    ctx: Context = Context(),
) -> str:
    """Fetch price data from Yahoo Finance and store in session state."""
    await ctx.info(f"Fetching {ticker} prices from {start}...")

    from wraquant.data import fetch_prices as wq_fetch

    df = wq_fetch(ticker, start=start, end=end, interval=interval)

    # Store in session state
    analysis = AnalysisContext.from_ctx(ctx)
    analysis.store_dataframe(ticker, df)

    await ctx.info(f"Stored {len(df)} rows for {ticker}")
    return (
        f"Fetched {len(df)} rows of OHLCV data for {ticker} "
        f"({df.index[0].date()} to {df.index[-1].date()}). "
        f"Columns: {list(df.columns)}. "
        f"Stored as dataset '{ticker}'."
    )
```

### 3.3 Tool design principles

Each MCP tool follows these rules:

1. **Coarse-grained**: A tool is a workflow step, not a raw function. `risk_portfolio_report`
   computes VaR + CVaR + drawdown + Sharpe in one call.
2. **String-in, string-out**: Inputs are simple types (str, float, int, list[str]).
   DataFrames are referenced by name from session state, not passed inline.
3. **Self-documenting returns**: Return human-readable text with key metrics formatted for
   the LLM to interpret. Include units and interpretation guidance.
4. **Side-effect: store results**: Tools store computed results (fitted models, transformed
   data) in session state for downstream tools to consume.
5. **Progress reporting**: Long-running tools (GARCH fitting, backtesting) report progress
   via `ctx.report_progress()`.

### 3.4 Tag-based module control

Users can enable/disable module groups:

```python
# Server-level configuration
mcp.enable(tags={"data", "stats", "risk"}, only=True)  # minimal mode
mcp.disable(tags={"experimental"})                       # hide beta tools
```

Tags follow a hierarchy: `{"data"}`, `{"data", "fetch"}`, `{"data", "clean"}`,
`{"risk", "var"}`, `{"risk", "portfolio"}`, `{"viz", "chart"}`, etc.

---

## 4. LangChain Integration Pattern

### 4.1 LangChain as MCP client (primary pattern)

wraquant-mcp is consumed by LangChain agents via `langchain-mcp-adapters`. The MCP server
does not import or depend on LangChain.

**Stdio transport (local development):**

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model

model = init_chat_model("anthropic:claude-sonnet-4-20250514")

async with MultiServerMCPClient({
    "wraquant": {
        "command": "python",
        "args": ["-m", "wraquant_mcp"],
        "transport": "stdio",
    }
}) as client:
    tools = client.get_tools()
    agent = create_react_agent(model=model, tools=tools)

    result = await agent.ainvoke({
        "messages": [
            "Fetch AAPL and MSFT data for 2023, compute their Sharpe ratios, "
            "run a cointegration test, and if cointegrated, show me the spread."
        ]
    })
```

**HTTP transport (production):**

```python
async with MultiServerMCPClient({
    "wraquant": {
        "url": "https://mcp.wraquant.io/mcp",
        "transport": "http",
        "headers": {"Authorization": "Bearer <token>"},
    }
}) as client:
    tools = client.get_tools()
    agent = create_react_agent(model=model, tools=tools)
    # ... same agent usage
```

### 4.2 Multi-server composition with LangChain

LangChain can compose wraquant-mcp with other MCP servers:

```python
async with MultiServerMCPClient({
    "wraquant": {
        "command": "python",
        "args": ["-m", "wraquant_mcp"],
        "transport": "stdio",
    },
    "financial-data": {
        "url": "https://api.financial-datasets.ai/mcp",
        "transport": "http",
    },
    "slack": {
        "command": "npx",
        "args": ["-y", "@anthropic/slack-mcp-server"],
        "transport": "stdio",
    },
}) as client:
    tools = client.get_tools()
    # Agent can: fetch data -> analyze with wraquant -> post results to Slack
```

### 4.3 LangGraph workflow with MCP tools

For structured multi-step workflows, use LangGraph's StateGraph:

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Build graph with wraquant MCP tools
tools = client.get_tools()
model_with_tools = model.bind_tools(tools)

def analyst(state: MessagesState):
    """The main analyst node."""
    return {"messages": [model_with_tools.invoke(state["messages"])]}

graph = StateGraph(MessagesState)
graph.add_node("analyst", analyst)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "analyst")
graph.add_conditional_edges("analyst", tools_condition)
graph.add_edge("tools", "analyst")

app = graph.compile()
```

### 4.4 Schema compatibility

The conversion pipeline is:

```
wraquant-mcp tool                    LangChain BaseTool
─────────────────                    ──────────────────
@mcp.tool() with type hints    →    Pydantic input model (auto-generated)
str/float/int/list params      →    JSON schema properties
Google docstring               →    tool.description
TextContent return             →    ToolMessage.content (str)
ImageContent return            →    ToolMessage.content (base64 image)
```

No manual schema mapping is needed. FastMCP's type introspection produces valid JSON Schema
that `langchain-mcp-adapters` consumes directly.

---

## 5. Tool Design by Module

### 5.1 Tool inventory

Each module exposes a curated set of tools (not all functions). Tools are grouped by
cognitive task: fetch, analyze, model, optimize, visualize.

#### data (namespace: `data_`)

| Tool | Input | Output | Session side-effect |
|---|---|---|---|
| `fetch_prices` | ticker, start, end, interval | Summary stats | Stores OHLCV DataFrame |
| `fetch_macro` | series_id, source | Summary | Stores macro series |
| `list_datasets` | — | Table of stored datasets | — |
| `compute_returns` | dataset, method (simple/log) | Stats | Stores return series |
| `clean_data` | dataset, methods[] | Report of fixes | Updates stored DataFrame |
| `data_quality_report` | dataset | Quality metrics | — |

#### stats (namespace: `stats_`)

| Tool | Input | Output | Session side-effect |
|---|---|---|---|
| `summary_statistics` | dataset | Full descriptive stats table | — |
| `correlation_matrix` | datasets[] | Correlation table | Stores corr matrix |
| `cointegration_test` | dataset1, dataset2 | Engle-Granger / Johansen results | Stores spread |
| `distribution_fit` | dataset | Best-fit distribution + params | — |
| `regression` | y_dataset, x_datasets[], model | Coefficients, R2, diagnostics | Stores model |
| `rolling_statistics` | dataset, window, metrics[] | Rolling stats summary | Stores results |

#### risk (namespace: `risk_`)

| Tool | Input | Output | Session side-effect |
|---|---|---|---|
| `risk_metrics` | dataset, benchmark? | Sharpe, Sortino, MaxDD, VaR, CVaR | — |
| `portfolio_risk` | weights, datasets[] | Portfolio vol, risk decomposition | — |
| `value_at_risk` | dataset, confidence, method | VaR + CVaR | — |
| `stress_test` | dataset, scenarios[] | Stress P&L table | — |
| `risk_attribution` | weights, datasets[] | Marginal/component VaR | — |

#### vol (namespace: `vol_`)

| Tool | Input | Output | Session side-effect |
|---|---|---|---|
| `realized_volatility` | dataset, estimator, window | Vol estimate + chart data | — |
| `garch_fit` | dataset, model, p, q | Fitted params, persistence, diagnostics | Stores model |
| `garch_forecast` | model_name, horizon | Forecast table + CI | Stores forecast |
| `vol_surface` | ticker | Implied vol surface data | — |
| `news_impact_curve` | model_name | NIC data for visualization | — |

#### ta (namespace: `ta_`)

| Tool | Input | Output | Session side-effect |
|---|---|---|---|
| `compute_indicator` | dataset, indicator, params{} | Indicator values + interpretation | Stores series |
| `indicator_scan` | dataset, category | All indicators in category | Stores results |
| `signal_summary` | dataset | Bull/bear signal counts across indicators | — |
| `support_resistance` | dataset, method | S/R levels | — |
| `pattern_scan` | dataset | Detected candlestick patterns | — |

#### opt (namespace: `opt_`)

| Tool | Input | Output | Session side-effect |
|---|---|---|---|
| `optimize_portfolio` | datasets[], method, constraints{} | Weights + metrics | Stores weights |
| `efficient_frontier` | datasets[], n_points | Frontier data | — |
| `black_litterman` | datasets[], views{} | BL weights + metrics | Stores weights |
| `risk_parity` | datasets[] | RP weights + risk contributions | Stores weights |

#### regimes (namespace: `regimes_`)

| Tool | Input | Output | Session side-effect |
|---|---|---|---|
| `detect_regimes` | dataset, n_regimes, model | Regime labels + params | Stores model |
| `current_regime` | model_name | Current regime + probability | — |
| `regime_statistics` | model_name | Per-regime mean/vol/duration | — |
| `regime_portfolio` | model_name, datasets[] | Regime-conditional weights | Stores weights |

#### ts (namespace: `ts_`)

| Tool | Input | Output | Session side-effect |
|---|---|---|---|
| `decompose` | dataset, method | Trend + seasonal + residual | Stores components |
| `stationarity_test` | dataset | ADF + KPSS results | — |
| `forecast` | dataset, model, horizon | Point + interval forecasts | Stores forecast |
| `detect_changepoints` | dataset | Changepoint dates + confidence | — |
| `seasonality` | dataset | Detected periods + strength | — |

#### backtest (namespace: `backtest_`)

| Tool | Input | Output | Session side-effect |
|---|---|---|---|
| `run_backtest` | strategy, dataset, params{} | Performance summary | Stores results |
| `tearsheet` | backtest_name | Full performance report | — |
| `compare_strategies` | backtest_names[] | Comparison table | — |
| `walk_forward` | strategy, dataset, params{} | WF results | Stores results |

#### price (namespace: `price_`)

| Tool | Input | Output | Session side-effect |
|---|---|---|---|
| `option_price` | S, K, T, r, sigma, style, type | Price + Greeks | — |
| `implied_volatility` | S, K, T, r, market_price, type | IV + Greeks | — |
| `bond_price` | face, coupon, ytm, maturity, freq | Price + duration + convexity | — |
| `yield_curve` | rates{}, method | Interpolated curve | Stores curve |

#### viz (namespace: `viz_`)

| Tool | Input | Output | Session side-effect |
|---|---|---|---|
| `chart_prices` | dataset, indicators[] | PNG image | — |
| `chart_returns` | dataset | Distribution + time series PNG | — |
| `chart_correlation` | datasets[] | Heatmap PNG | — |
| `chart_drawdown` | dataset | Drawdown chart PNG | — |
| `chart_efficient_frontier` | frontier_name | EF plot PNG | — |
| `chart_regimes` | model_name, dataset | Regime overlay PNG | — |
| `chart_vol_surface` | surface_name | 3D vol surface PNG | — |

#### forex (namespace: `forex_`)

| Tool | Input | Output | Session side-effect |
|---|---|---|---|
| `pip_calculator` | pair, entry, exit, lot_size | P&L in pips and currency | — |
| `carry_analysis` | pairs[], rates{} | Carry return table | — |
| `session_info` | — | Current session + overlaps | — |

### 5.2 Cross-cutting tools (no namespace, on parent server)

| Tool | Input | Output | Session side-effect |
|---|---|---|---|
| `analyze` | dataset, benchmark? | Comprehensive multi-module report | — |
| `list_stored` | type? (datasets/models/forecasts) | Inventory of session state | — |
| `clear_session` | — | Confirmation | Clears all state |
| `export_results` | name, format (csv/json/parquet) | Download link or data | — |

### 5.3 Tool implementation pattern

Every tool follows this template:

```python
from __future__ import annotations
from fastmcp import FastMCP, Context

vol_server = FastMCP("wraquant-vol")

@vol_server.tool(
    tags={"vol", "garch", "modeling"},
    description=(
        "Fit a GARCH-family volatility model to a return series. "
        "Supports GARCH, EGARCH, GJR-GARCH, FIGARCH, and HARCH. "
        "Returns fitted parameters, persistence, half-life, and diagnostics. "
        "The fitted model is stored in session state for forecasting."
    ),
)
async def garch_fit(
    dataset: str,
    model: str = "GARCH",
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
    ctx: Context = Context(),
) -> str:
    """Fit a GARCH model to the named return series.

    Parameters:
        dataset: Name of a stored return series (from compute_returns or fetch_prices).
        model: Model type. One of: GARCH, EGARCH, GJR, FIGARCH, HARCH.
        p: ARCH order (number of lagged squared residuals). Default 1.
        q: GARCH order (number of lagged conditional variances). Default 1.
        dist: Error distribution. One of: normal, t, skewt, ged. Use 't' for
            heavy-tailed assets (equities), 'skewt' for asymmetric (FX).
    """
    from wraquant_mcp.state import AnalysisContext

    analysis = AnalysisContext.from_ctx(ctx)
    returns = analysis.get_returns(dataset)
    if returns is None:
        return f"Error: No return series named '{dataset}'. Use data_compute_returns first."

    await ctx.info(f"Fitting {model}({p},{q}) with {dist} distribution...")
    await ctx.report_progress(0.1, 1.0)

    from wraquant.vol import garch_fit as wq_garch_fit

    result = wq_garch_fit(returns, model=model, p=p, q=q, dist=dist)

    await ctx.report_progress(0.9, 1.0)

    # Store the fitted model
    model_name = f"{dataset}_{model}_{p}_{q}"
    analysis.store_model(model_name, result)

    await ctx.report_progress(1.0, 1.0)

    # Format rich text output
    return (
        f"## {model}({p},{q}) fitted to '{dataset}'\n\n"
        f"**Parameters:**\n"
        f"- omega (constant): {result['omega']:.6f}\n"
        f"- alpha (ARCH): {result['alpha']:.4f}\n"
        f"- beta (GARCH): {result['beta']:.4f}\n"
        f"- persistence (alpha+beta): {result['persistence']:.4f}\n"
        f"- half-life: {result['half_life']:.1f} periods\n\n"
        f"**Diagnostics:**\n"
        f"- Log-likelihood: {result['log_likelihood']:.2f}\n"
        f"- AIC: {result['aic']:.2f}\n"
        f"- BIC: {result['bic']:.2f}\n"
        f"- Ljung-Box p-value (residuals): {result['ljung_box_p']:.4f}\n\n"
        f"**Interpretation:**\n"
        f"- {'High' if result['persistence'] > 0.95 else 'Moderate'} persistence "
        f"({'integrated' if result['persistence'] > 0.99 else 'mean-reverting'}).\n"
        f"- Shocks have a half-life of {result['half_life']:.0f} periods.\n\n"
        f"Stored as model '{model_name}'. Use vol_garch_forecast to generate forecasts."
    )
```

---

## 6. Prompt Templates for Common Workflows

MCP prompts guide the LLM through multi-step analysis patterns. They return structured
message sequences that tell the LLM which tools to call in which order.

### 6.1 Prompt definitions

```python
from fastmcp import FastMCP

mcp = FastMCP("wraquant")

@mcp.prompt(
    name="equity_analysis",
    description="Complete equity analysis: fetch data, compute stats, fit volatility model, assess risk.",
    tags={"workflow", "equity"},
)
def equity_analysis_prompt(
    ticker: str,
    start: str = "2020-01-01",
    benchmark: str = "SPY",
) -> str:
    return f"""You are a quantitative analyst. Perform a complete analysis of {ticker}
from {start} to today, benchmarked against {benchmark}.

Follow these steps in order:

1. **Fetch data**: Call `data_fetch_prices` for both {ticker} and {benchmark}.
2. **Compute returns**: Call `data_compute_returns` for both.
3. **Summary statistics**: Call `stats_summary_statistics` for {ticker}.
4. **Risk metrics**: Call `risk_risk_metrics` with {benchmark} as benchmark.
5. **Distribution analysis**: Call `stats_distribution_fit` to check for fat tails.
6. **Volatility model**: Call `vol_garch_fit` with model=EGARCH, dist=t.
7. **Regime detection**: Call `regimes_detect_regimes` with n_regimes=2.
8. **Visualization**: Call `viz_chart_returns` and `viz_chart_regimes`.
9. **Synthesis**: Combine all findings into a structured investment memo with:
   - Risk/return profile
   - Current volatility regime
   - Distribution characteristics
   - Key risks and opportunities"""


@mcp.prompt(
    name="pairs_trading",
    description="Pairs trading analysis: cointegration, spread, signals, backtest.",
    tags={"workflow", "pairs", "strategy"},
)
def pairs_trading_prompt(
    ticker1: str,
    ticker2: str,
    start: str = "2018-01-01",
) -> str:
    return f"""You are a pairs trading specialist. Analyze the {ticker1}/{ticker2} pair
for mean-reversion trading opportunities.

Follow these steps:

1. **Fetch data**: Call `data_fetch_prices` for {ticker1} and {ticker2} from {start}.
2. **Compute returns**: Call `data_compute_returns` for both.
3. **Cointegration test**: Call `stats_cointegration_test` with both datasets.
4. **Correlation analysis**: Call `stats_correlation_matrix` for both.
5. **If cointegrated**:
   a. Compute the spread and z-score.
   b. Call `ta_compute_indicator` with indicator=bollinger_bands on the spread.
   c. Call `backtest_run_backtest` with a mean-reversion strategy.
   d. Call `backtest_tearsheet` to evaluate performance.
6. **If NOT cointegrated**: Explain why and suggest alternative pairs.
7. **Visualize**: Call `viz_chart_prices` overlaying both tickers.
8. **Risk assessment**: Call `risk_risk_metrics` on the strategy returns."""


@mcp.prompt(
    name="portfolio_construction",
    description="Build an optimized portfolio: data, optimization, risk analysis, visualization.",
    tags={"workflow", "portfolio"},
)
def portfolio_construction_prompt(
    tickers: str,  # comma-separated
    method: str = "max_sharpe",
    start: str = "2019-01-01",
) -> str:
    ticker_list = [t.strip() for t in tickers.split(",")]
    return f"""You are a portfolio manager. Build an optimized portfolio from:
{', '.join(ticker_list)}

Follow these steps:

1. **Fetch data**: Call `data_fetch_prices` for each of: {', '.join(ticker_list)}.
2. **Compute returns**: Call `data_compute_returns` for each.
3. **Correlation structure**: Call `stats_correlation_matrix` for all assets.
4. **Optimize**: Call `opt_optimize_portfolio` with method={method}.
5. **Risk decomposition**: Call `risk_portfolio_risk` with the optimized weights.
6. **Efficient frontier**: Call `opt_efficient_frontier` for context.
7. **Regime analysis**: Call `regimes_detect_regimes` on the portfolio returns.
8. **Visualize**:
   a. Call `viz_chart_efficient_frontier`.
   b. Call `viz_chart_correlation`.
9. **Report**: Summarize weights, expected return, volatility, Sharpe, max drawdown,
   and regime-conditional behavior."""


@mcp.prompt(
    name="risk_report",
    description="Comprehensive risk report: VaR, stress tests, tail risk, regime analysis.",
    tags={"workflow", "risk"},
)
def risk_report_prompt(
    dataset: str,
    confidence: float = 0.95,
) -> str:
    return f"""You are a risk manager. Produce a comprehensive risk report for '{dataset}'.

Follow these steps:

1. **Risk metrics**: Call `risk_risk_metrics` for the dataset.
2. **VaR analysis**: Call `risk_value_at_risk` at {confidence*100}% confidence
   using both historical and parametric methods.
3. **Tail risk**: Call `stats_distribution_fit` to assess fat tails.
4. **Stress testing**: Call `risk_stress_test` with scenarios:
   - Market crash (-20% equity shock)
   - Rate spike (+200bps)
   - Vol spike (VIX to 40)
5. **Regime analysis**: Call `regimes_detect_regimes` to identify risk regimes.
6. **Drawdown analysis**: Call `viz_chart_drawdown`.
7. **Report**: Present findings in a structured risk report with traffic-light
   ratings (green/amber/red) for each risk dimension."""


@mcp.prompt(
    name="volatility_deep_dive",
    description="Deep volatility analysis: realized vol, GARCH, term structure, regimes.",
    tags={"workflow", "volatility"},
)
def volatility_deep_dive_prompt(
    ticker: str,
    start: str = "2018-01-01",
) -> str:
    return f"""You are a volatility specialist. Perform a deep volatility analysis
for {ticker} from {start}.

Follow these steps:

1. **Fetch data**: Call `data_fetch_prices` for {ticker}.
2. **Compute returns**: Call `data_compute_returns`.
3. **Realized vol**: Call `vol_realized_volatility` with estimators:
   close-to-close, Parkinson, Garman-Klass, Yang-Zhang.
4. **GARCH modeling**: Fit three models:
   a. Call `vol_garch_fit` with model=GARCH, dist=t.
   b. Call `vol_garch_fit` with model=EGARCH, dist=t.
   c. Call `vol_garch_fit` with model=GJR, dist=t.
5. **Compare models**: Select the best by AIC/BIC.
6. **Forecast**: Call `vol_garch_forecast` with the best model, horizon=30.
7. **Regime detection**: Call `regimes_detect_regimes` with n_regimes=2 to
   identify high/low volatility regimes.
8. **Visualization**: Call `viz_chart_vol_surface` if options data is available.
9. **Report**: Summarize current vol level, regime, forecast, and the
   leverage effect (asymmetric response to positive vs negative returns)."""
```

---

## 7. State Management

### 7.1 AnalysisContext: session-scoped state

MCP's `Context` provides per-request state, but tools need to share data across multiple
requests within a session. We build `AnalysisContext` on top of `ctx.set_state` / `ctx.get_state`
plus an in-process store for the current session.

```python
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from fastmcp import Context


# Module-level session store (in-memory, per-process)
_sessions: dict[str, "AnalysisContext"] = {}


@dataclass
class AnalysisContext:
    """Session-scoped analysis state.

    Stores DataFrames, fitted models, forecasts, and computed results
    so that tools can chain without re-fetching or re-computing data.

    Lifecycle: created on first tool call in a session, persists until
    the session ends or clear_session is called.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dataframes: dict[str, pd.DataFrame] = field(default_factory=dict)
    returns: dict[str, pd.Series] = field(default_factory=dict)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    forecasts: dict[str, dict[str, Any]] = field(default_factory=dict)
    weights: dict[str, dict[str, float]] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_ctx(cls, ctx: Context) -> "AnalysisContext":
        """Get or create the AnalysisContext for the current session."""
        session_id = getattr(ctx, "session_id", None) or "default"
        if session_id not in _sessions:
            _sessions[session_id] = cls(session_id=session_id)
        return _sessions[session_id]

    def store_dataframe(self, name: str, df: pd.DataFrame) -> None:
        self.dataframes[name] = df

    def store_returns(self, name: str, returns: pd.Series) -> None:
        self.returns[name] = returns

    def get_returns(self, name: str) -> pd.Series | None:
        """Get returns, computing from prices if needed."""
        if name in self.returns:
            return self.returns[name]
        if name in self.dataframes:
            df = self.dataframes[name]
            if "Close" in df.columns:
                ret = df["Close"].pct_change().dropna()
                self.returns[name] = ret
                return ret
        return None

    def store_model(self, name: str, model: dict[str, Any]) -> None:
        self.models[name] = model

    def store_weights(self, name: str, weights: dict[str, float]) -> None:
        self.weights[name] = weights

    def store_result(self, name: str, result: Any) -> None:
        self.results[name] = result

    def inventory(self) -> str:
        """Return a formatted inventory of all stored items."""
        lines = []
        if self.dataframes:
            lines.append("**DataFrames:**")
            for name, df in self.dataframes.items():
                lines.append(f"  - {name}: {len(df)} rows, cols={list(df.columns)}")
        if self.returns:
            lines.append("**Return Series:**")
            for name, ret in self.returns.items():
                lines.append(f"  - {name}: {len(ret)} observations")
        if self.models:
            lines.append("**Fitted Models:**")
            for name in self.models:
                lines.append(f"  - {name}")
        if self.weights:
            lines.append("**Portfolio Weights:**")
            for name, w in self.weights.items():
                lines.append(f"  - {name}: {w}")
        if self.forecasts:
            lines.append("**Forecasts:**")
            for name in self.forecasts:
                lines.append(f"  - {name}")
        return "\n".join(lines) if lines else "Session is empty. Fetch some data first."

    def clear(self) -> None:
        self.dataframes.clear()
        self.returns.clear()
        self.models.clear()
        self.forecasts.clear()
        self.weights.clear()
        self.results.clear()
        self.metadata.clear()
```

### 7.2 Resources for state inspection

MCP resources let the LLM browse stored state without calling tools:

```python
@mcp.resource("session://datasets")
def list_datasets() -> str:
    """List all datasets stored in the current session."""
    # Note: resource functions cannot access Context in all cases,
    # so we use a module-level default session.
    analysis = _sessions.get("default", AnalysisContext())
    return analysis.inventory()


@mcp.resource("session://datasets/{name}")
def get_dataset_info(name: str) -> str:
    """Get detailed info about a specific stored dataset."""
    analysis = _sessions.get("default", AnalysisContext())
    if name in analysis.dataframes:
        df = analysis.dataframes[name]
        return (
            f"Dataset: {name}\n"
            f"Shape: {df.shape}\n"
            f"Columns: {list(df.columns)}\n"
            f"Date range: {df.index[0]} to {df.index[-1]}\n"
            f"Head:\n{df.head().to_string()}\n"
            f"Describe:\n{df.describe().to_string()}"
        )
    return f"No dataset named '{name}' found."


@mcp.resource("session://models/{name}")
def get_model_info(name: str) -> str:
    """Get details of a fitted model."""
    analysis = _sessions.get("default", AnalysisContext())
    if name in analysis.models:
        model = analysis.models[name]
        return f"Model: {name}\nParameters: {model}"
    return f"No model named '{name}' found."
```

### 7.3 State lifecycle and limitations

- **Per-process, per-session**: State lives in memory on the server process. It does not
  persist across server restarts.
- **Horizontal scaling**: When running multiple workers behind a load balancer, session
  affinity (sticky sessions) is required, or sessions must be externalized to Redis/SQLite.
- **Memory management**: Large DataFrames consume memory. The `clear_session` tool and
  a configurable max-datasets limit prevent unbounded growth.
- **No serialization over the wire**: DataFrames are never sent to the LLM. Tools return
  summaries, statistics, and chart images. The data stays server-side.

---

## 8. Visualization and Rich Output

### 8.1 Returning images from tools

MCP supports `ImageContent` with base64-encoded PNG/JPEG data. FastMCP's `Image` class
handles encoding automatically:

```python
from fastmcp import FastMCP, Context, Image
import io

viz_server = FastMCP("wraquant-viz")

@viz_server.tool(
    tags={"viz", "chart"},
    description="Generate a returns distribution chart for a stored dataset.",
)
async def chart_returns(
    dataset: str,
    ctx: Context = Context(),
) -> Image:
    """Create a returns distribution and time series chart."""
    from wraquant_mcp.state import AnalysisContext

    analysis = AnalysisContext.from_ctx(ctx)
    returns = analysis.get_returns(dataset)
    if returns is None:
        raise ValueError(f"No dataset '{dataset}'. Fetch data first.")

    from wraquant.viz.interactive import plotly_returns, plotly_distribution

    fig = plotly_returns(returns)

    # Convert Plotly figure to PNG bytes
    img_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)

    return Image(data=img_bytes, media_type="image/png")
```

### 8.2 Plotly figure serialization strategy

Three output modes for visualizations:

1. **PNG image** (default): Tool returns `Image(data=png_bytes, media_type="image/png")`.
   Works with all MCP clients. Claude can "see" the chart via vision.

2. **JSON spec** (for web clients): Tool returns the Plotly JSON spec as `TextContent`.
   Web-based MCP clients can render interactive charts. Return as structured text:
   ```python
   return f"```plotly\n{fig.to_json()}\n```"
   ```

3. **HTML file** (for local usage): Write to a temp file, return the path. The user can
   open it in a browser for interactive exploration.

### 8.3 Text formatting for LLM consumption

All tool outputs are formatted for LLM readability:

```python
def format_risk_report(metrics: dict) -> str:
    """Format risk metrics for LLM consumption."""
    return (
        f"## Risk Metrics\n\n"
        f"| Metric | Value | Rating |\n"
        f"|--------|-------|--------|\n"
        f"| Sharpe Ratio | {metrics['sharpe']:.3f} | {'Good' if metrics['sharpe'] > 1 else 'Fair' if metrics['sharpe'] > 0.5 else 'Poor'} |\n"
        f"| Sortino Ratio | {metrics['sortino']:.3f} | {'Good' if metrics['sortino'] > 1.5 else 'Fair'} |\n"
        f"| Max Drawdown | {metrics['max_drawdown']:.1%} | {'Low risk' if abs(metrics['max_drawdown']) < 0.1 else 'Moderate' if abs(metrics['max_drawdown']) < 0.2 else 'High risk'} |\n"
        f"| VaR (95%) | {metrics['var_95']:.1%} | |\n"
        f"| CVaR (95%) | {metrics['cvar_95']:.1%} | |\n"
    )
```

---

## 9. Authentication and Security

### 9.1 Local usage (stdio transport)

No authentication needed. The server runs as a subprocess of the client (Claude Desktop,
CLI, or LangChain), and communication happens over stdin/stdout.

### 9.2 Hosted deployment (HTTP transport)

FastMCP 3.0 supports multiple authentication patterns:

**API key authentication (simplest):**

```python
from fastmcp import FastMCP
from fastmcp.server.auth import APIKeyAuth

mcp = FastMCP(
    "wraquant",
    auth=APIKeyAuth(
        keys=["wq-key-abc123"],  # or load from env
        header="X-API-Key",
    ),
)
```

**Bearer token / JWT (production):**

```python
from fastmcp import FastMCP
from fastmcp.server.auth import BearerAuth

mcp = FastMCP(
    "wraquant",
    auth=BearerAuth(
        issuer="https://auth.wraquant.io",
        audience="wraquant-mcp",
    ),
)
```

**Per-tool authorization:**

```python
@mcp.tool(
    tags={"data", "fetch"},
    auth={"scopes": ["data:read"]},
)
async def fetch_prices(ticker: str, ctx: Context = Context()) -> str:
    ...
```

### 9.3 Security considerations

- **No code execution**: Tools call wraquant functions with validated parameters. The LLM
  cannot execute arbitrary Python code.
- **Input validation**: All tool parameters are typed and validated by FastMCP's schema
  system before reaching wraquant functions.
- **Rate limiting**: The HTTP deployment should sit behind a reverse proxy (nginx, Caddy)
  with rate limiting configured.
- **Data privacy**: Price data fetched from Yahoo Finance is public. User-uploaded data
  stays in the server process and is never sent to external services.
- **No persistent storage by default**: Session state is in-memory only. No data is written
  to disk unless the user explicitly calls `export_results`.

---

## 10. Deployment Options

### 10.1 Local stdio (development / Claude Desktop)

The simplest deployment. The MCP server runs as a subprocess:

**Claude Desktop config** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "wraquant": {
      "command": "python",
      "args": ["-m", "wraquant_mcp"],
      "env": {
        "WRAQUANT_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

**Or via `uvx` / `pipx` for zero-install:**
```json
{
  "mcpServers": {
    "wraquant": {
      "command": "uvx",
      "args": ["wraquant-mcp"]
    }
  }
}
```

### 10.2 Local HTTP (development server)

```python
# wraquant_mcp/__main__.py
from wraquant_mcp.server import mcp

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8642,
        log_level="debug",
    )
```

```bash
python -m wraquant_mcp
# Server at http://127.0.0.1:8642/mcp
```

### 10.3 Production HTTP (uvicorn + nginx)

**Dockerfile:**
```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install wraquant + wraquant-mcp
COPY pyproject.toml pdm.lock ./
RUN pip install pdm && pdm install --prod --no-self
COPY src/ src/

# Install kaleido for Plotly PNG export
RUN pip install kaleido

EXPOSE 8642

CMD ["uvicorn", "wraquant_mcp.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8642", \
     "--workers", "1"]
```

Note: `--workers 1` because session state is in-memory. For multi-worker deployments,
externalize state to Redis.

**nginx config (excerpt):**
```nginx
location /mcp {
    proxy_pass http://wraquant-mcp:8642;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_buffering off;           # Required for SSE
    proxy_cache off;
    proxy_read_timeout 300s;       # Long-running tools

    # Rate limiting
    limit_req zone=mcp burst=20 nodelay;
}
```

### 10.4 Docker Compose (full stack)

```yaml
version: "3.9"
services:
  wraquant-mcp:
    build: .
    ports:
      - "8642:8642"
    environment:
      - WRAQUANT_LOG_LEVEL=INFO
      - WRAQUANT_MAX_DATASETS=50
    volumes:
      - ./exports:/app/exports
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - wraquant-mcp
```

---

## 11. Project Structure

```
wraquant-mcp/
├── pyproject.toml              # Package metadata, dependencies
├── README.md                   # Usage docs
├── Dockerfile
├── docker-compose.yml
├── nginx.conf
│
├── src/
│   └── wraquant_mcp/
│       ├── __init__.py         # Package init, version
│       ├── __main__.py         # `python -m wraquant_mcp` entry point
│       ├── server.py           # Parent FastMCP server + mounts
│       ├── state.py            # AnalysisContext session state
│       ├── formatting.py       # Output formatting helpers
│       ├── config.py           # Server configuration (env vars, defaults)
│       │
│       ├── servers/            # Per-module child servers
│       │   ├── __init__.py
│       │   ├── data.py         # data_server: fetch, clean, transform
│       │   ├── stats.py        # stats_server: descriptive, correlation, cointegration
│       │   ├── risk.py         # risk_server: metrics, VaR, stress testing
│       │   ├── vol.py          # vol_server: realized, GARCH, forecast
│       │   ├── ta.py           # ta_server: indicators, signals, patterns
│       │   ├── opt.py          # opt_server: portfolio optimization
│       │   ├── regimes.py      # regimes_server: HMM, regime detection
│       │   ├── ts.py           # ts_server: decomposition, forecasting
│       │   ├── backtest.py     # backtest_server: engine, tearsheets
│       │   ├── price.py        # price_server: options, bonds, curves
│       │   ├── viz.py          # viz_server: charts, dashboards
│       │   └── forex.py        # forex_server: FX-specific tools
│       │
│       ├── prompts/            # Prompt templates
│       │   ├── __init__.py
│       │   ├── equity.py       # Equity analysis workflow
│       │   ├── pairs.py        # Pairs trading workflow
│       │   ├── portfolio.py    # Portfolio construction workflow
│       │   ├── risk.py         # Risk report workflow
│       │   └── volatility.py   # Volatility deep-dive workflow
│       │
│       └── resources/          # Resource definitions
│           ├── __init__.py
│           ├── session.py      # Session state resources
│           └── config.py       # Server config resources
│
├── tests/
│   ├── test_server.py          # Server startup and tool listing
│   ├── test_tools/             # Per-module tool tests
│   │   ├── test_data_tools.py
│   │   ├── test_risk_tools.py
│   │   ├── test_vol_tools.py
│   │   └── ...
│   ├── test_state.py           # AnalysisContext tests
│   ├── test_langchain.py       # LangChain adapter integration tests
│   └── test_prompts.py         # Prompt template tests
│
├── examples/
│   ├── claude_desktop_config.json
│   ├── langchain_agent.py      # LangChain ReAct agent example
│   ├── langgraph_workflow.py   # LangGraph multi-step workflow
│   └── multi_server.py         # Composing with other MCP servers
│
└── docs/
    ├── quickstart.md
    ├── tools-reference.md      # Auto-generated tool reference
    ├── deployment.md
    └── langchain-guide.md
```

### 11.1 Dependencies

```toml
[project]
name = "wraquant-mcp"
version = "0.1.0"
description = "MCP server for wraquant quantitative finance toolkit"
requires-python = ">=3.13"
dependencies = [
    "wraquant>=0.1.0",
    "fastmcp>=3.0.0",
]

[project.optional-dependencies]
viz = [
    "kaleido>=0.2.1",  # Plotly PNG export
]
langchain = [
    "langchain-mcp-adapters>=0.1.0",
    "langgraph>=0.2.0",
]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]

[project.scripts]
wraquant-mcp = "wraquant_mcp.__main__:main"
```

### 11.2 Entry point

```python
# src/wraquant_mcp/__main__.py
from __future__ import annotations

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="wraquant MCP server")
    parser.add_argument(
        "--transport", choices=["stdio", "http", "sse"],
        default="stdio",
        help="Transport mode (default: stdio for Claude Desktop)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8642)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    from wraquant_mcp.server import mcp

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(
            transport="streamable-http",
            host=args.host,
            port=args.port,
            log_level=args.log_level,
        )


if __name__ == "__main__":
    main()
```

---

## 12. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

- [ ] Create `wraquant-mcp` package scaffolding
- [ ] Implement `AnalysisContext` state management
- [ ] Build parent server with mount architecture
- [ ] Implement `data_server` (fetch_prices, compute_returns, list_datasets)
- [ ] Implement `stats_server` (summary_statistics, correlation_matrix)
- [ ] Implement `risk_server` (risk_metrics, value_at_risk)
- [ ] Test with Claude Desktop via stdio
- [ ] Test with `langchain-mcp-adapters` via stdio

### Phase 2: Core Modules (Week 3-4)

- [ ] Implement `vol_server` (realized_volatility, garch_fit, garch_forecast)
- [ ] Implement `ta_server` (compute_indicator, signal_summary, pattern_scan)
- [ ] Implement `opt_server` (optimize_portfolio, efficient_frontier)
- [ ] Implement `regimes_server` (detect_regimes, current_regime)
- [ ] Add prompt templates (equity_analysis, pairs_trading, portfolio_construction)
- [ ] Add resource templates for session inspection

### Phase 3: Full Coverage + Viz (Week 5-6)

- [ ] Implement `ts_server` (forecast, decompose, stationarity_test)
- [ ] Implement `backtest_server` (run_backtest, tearsheet)
- [ ] Implement `price_server` (option_price, bond_price)
- [ ] Implement `forex_server` (pip_calculator, carry_analysis)
- [ ] Implement `viz_server` with Plotly PNG export
- [ ] Add remaining prompt templates (risk_report, volatility_deep_dive)

### Phase 4: Production Hardening (Week 7-8)

- [ ] Dockerfile and docker-compose
- [ ] Authentication (API key + JWT)
- [ ] Rate limiting and input validation hardening
- [ ] Session memory management (max datasets, TTL)
- [ ] Error handling and graceful degradation
- [ ] Comprehensive test suite
- [ ] Auto-generated tool reference documentation
- [ ] LangChain integration examples and guide
- [ ] PyPI publication

---

## Appendix A: Answers to Key Questions

### How does LangChain consume MCP tools?

Via `langchain-mcp-adapters`. The `MultiServerMCPClient` connects to wraquant-mcp over
stdio or HTTP, calls `list_tools()` to discover available tools, and wraps each one as a
`langchain.tools.BaseTool` with an auto-generated Pydantic input model. LangChain agents
(ReAct, LangGraph) then invoke these tools like any other LangChain tool. No wraquant-specific
adapter code is needed.

### Can one MCP server expose tools from multiple wraquant modules?

Yes. FastMCP's `mount()` composes child servers into a single parent. All 12+ module servers
appear as one unified tool set with namespace prefixes. The client sees a flat list of tools
like `data_fetch_prices`, `risk_risk_metrics`, `vol_garch_fit`.

### How to handle tool chaining?

Through session state. Tool A stores its output in `AnalysisContext` (e.g., `fetch_prices`
stores a DataFrame as `"AAPL"`). Tool B reads it by name (e.g., `garch_fit(dataset="AAPL")`).
The LLM orchestrates the chain based on tool descriptions and prompt templates. No explicit
piping mechanism is needed; the LLM's reasoning handles sequencing.

### Can we define MCP prompts that guide multi-step workflows?

Yes. MCP prompts return structured instructions that tell the LLM which tools to call in
which order. See Section 6 for five production-ready prompt templates. The LLM requests a
prompt (e.g., `equity_analysis(ticker="AAPL")`), receives the step-by-step plan, and
executes each tool call in sequence.

### What is the recommended project structure?

A standalone `wraquant-mcp` package that depends on `wraquant` and `fastmcp`. See Section 11.
Separate package avoids bloating wraquant's core dependencies and allows independent versioning.

### How to make tools configurable?

Three levels: (1) Tool parameters with defaults (e.g., `window=20`), (2) Server-level
config via environment variables loaded in `config.py` (e.g., `WRAQUANT_DEFAULT_START_DATE`),
(3) Session-level preferences stored in `AnalysisContext.metadata` that tools read as overrides.

### How to handle authentication for hosted MCP servers?

FastMCP 3.0 natively supports API key auth (via custom headers), Bearer/JWT auth (with JWKS
validation), and OAuth 2.1. For wraquant-mcp, start with API key auth for simplicity; upgrade
to JWT for multi-tenant production deployments. See Section 9.

### Can MCP tools return Plotly figures?

Yes. Convert Plotly figures to PNG via `fig.to_image()` (requires `kaleido`), then return
`Image(data=png_bytes, media_type="image/png")`. Claude processes the image via vision.
For web-based clients, additionally return the Plotly JSON spec as text for interactive
rendering. See Section 8.

---

## Appendix B: Sources

### FastMCP
- [FastMCP GitHub (PrefectHQ)](https://github.com/jlowin/fastmcp)
- [FastMCP Documentation](https://gofastmcp.com)
- [FastMCP Composing Servers](https://gofastmcp.com/servers/composition)
- [FastMCP Context](https://gofastmcp.com/servers/context)
- [FastMCP Tools](https://gofastmcp.com/servers/tools)
- [FastMCP Prompts](https://gofastmcp.com/servers/prompts)
- [FastMCP Resources](https://gofastmcp.com/servers/resources)
- [FastMCP HTTP Deployment](https://gofastmcp.com/deployment/http)
- [FastMCP Authentication](https://gofastmcp.com/servers/auth/authentication)
- [FastMCP FileSystem Provider](https://gofastmcp.com/servers/providers/filesystem)
- [FastMCP OpenAPI Integration](https://gofastmcp.com/integrations/openapi)
- [What's New in FastMCP 3.0](https://www.jlowin.dev/blog/fastmcp-3-whats-new)
- [Introducing FastMCP 3.0](https://www.jlowin.dev/blog/fastmcp-3)
- [FastMCP 3.0 GA](https://www.jlowin.dev/blog/fastmcp-3-launch)
- [FastMCP Decorating Methods](https://gofastmcp.com/v2/patterns/decorating-methods)
- [FastMCP on PyPI](https://pypi.org/project/fastmcp/)

### LangChain MCP Adapters
- [langchain-mcp-adapters GitHub](https://github.com/langchain-ai/langchain-mcp-adapters)
- [langchain-mcp-adapters on PyPI](https://pypi.org/project/langchain-mcp-adapters/)
- [LangChain MCP Documentation](https://docs.langchain.com/oss/python/langchain/mcp)
- [LangChain MCP Reference](https://reference.langchain.com/python/langchain-mcp-adapters)
- [convert_mcp_tool_to_langchain_tool Reference](https://reference.langchain.com/python/langchain-mcp-adapters/tools/convert_mcp_tool_to_langchain_tool)
- [MCP Adapters Announcement](https://changelog.langchain.com/announcements/mcp-adapters-for-langchain-and-langgraph)
- [LangChain MCP Adapters DeepWiki](https://deepwiki.com/langchain-ai/langchain-mcp-adapters)

### MCP Protocol and Financial MCP Servers
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Tools Specification](https://modelcontextprotocol.info/docs/concepts/tools/)
- [Financial Datasets MCP Server](https://github.com/financial-datasets/mcp-server)
- [Alpaca MCP Server](https://github.com/alpacahq/alpaca-mcp-server)
- [Finance Trading AI Agents MCP](https://github.com/aitrados/finance-trading-ai-agents-mcp)
- [QuantConnect MCP Server](https://github.com/QuantConnect/mcp-server)

### Tool Chaining and Agent Patterns
- [Advanced MCP: Agent Orchestration, Chaining, and Handoffs](https://www.getknit.dev/blog/advanced-mcp-agent-orchestration-chaining-and-handoffs)
- [MCP Tool Chainer](https://playbooks.com/mcp/thirdstrandstudio/mcp-tool-chainer)
- [Part 4: Advanced MCP Patterns and Tool Chaining](https://dev.to/techstuff/part-4-advanced-mcp-patterns-and-tool-chaining-4ll7)
- [LangGraph + MCP Stock Analysis Agent](https://medium.com/@sitabjapal03/langgraph-mcp-build-a-stock-analysis-agent-part-1-34bbf431610d)
- [FastMCP Tutorial (Firecrawl)](https://www.firecrawl.dev/blog/fastmcp-tutorial-building-mcp-servers-python)
- [Building MCP Server Complete Guide (Scrapfly)](https://scrapfly.io/blog/posts/how-to-build-an-mcp-server-in-python-a-complete-guide)
- [Deploy MCP Servers to Production (Ekamoira)](https://www.ekamoira.com/blog/mcp-servers-cloud-deployment-guide)
