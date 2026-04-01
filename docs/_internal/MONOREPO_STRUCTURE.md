# Monorepo Structure — wraquant + wraquant-mcp

## Current: Single package
```
wraquant/
├── src/wraquant/          # The library (97K LOC)
├── tests/
├── docs/
├── pyproject.toml         # One package: wraquant
└── README.md
```

## Target: Monorepo with separate packages
```
wraquant/                  # Root repo
├── packages/
│   ├── wraquant/          # Core library
│   │   ├── src/wraquant/
│   │   ├── tests/
│   │   └── pyproject.toml # pip install wraquant
│   │
│   ├── wraquant-mcp/      # MCP server (separate package)
│   │   ├── src/wraquant_mcp/
│   │   │   ├── __init__.py
│   │   │   ├── server.py          # FastMCP entry point
│   │   │   ├── context.py         # AnalysisContext (DuckDB state)
│   │   │   ├── adaptor.py         # Auto-wraps wraquant functions
│   │   │   ├── registry.py        # Tool registry + discovery
│   │   │   ├── ids.py             # ID generation + resolution
│   │   │   ├── workspace.py       # Workspace management
│   │   │   ├── prompts/           # MCP prompt templates
│   │   │   │   ├── equity_analysis.py
│   │   │   │   ├── pairs_trading.py
│   │   │   │   ├── portfolio_construction.py
│   │   │   │   └── risk_report.py
│   │   │   └── servers/           # Module-specific servers
│   │   │       ├── data.py
│   │   │       ├── risk.py
│   │   │       ├── vol.py
│   │   │       ├── regimes.py
│   │   │       ├── ta.py
│   │   │       ├── stats.py
│   │   │       ├── ts.py
│   │   │       ├── opt.py
│   │   │       ├── backtest.py
│   │   │       ├── price.py
│   │   │       ├── ml.py
│   │   │       └── viz.py
│   │   ├── tests/
│   │   └── pyproject.toml # pip install wraquant-mcp
│   │                      # depends on: wraquant, fastmcp, duckdb
│   │
│   └── wraquant-dash/     # Dashboard (optional, future)
│       ├── src/wraquant_dash/
│       └── pyproject.toml # pip install wraquant-dash
│
├── docs/                  # Shared docs
├── .github/workflows/     # Shared CI
└── README.md              # Root README
```

## Alternative: Keep it simpler (recommended for now)

Don't move wraquant yet. Just add wraquant-mcp alongside:

```
wraquant/                  # Root repo (existing)
├── src/wraquant/          # Core library (unchanged)
├── src/wraquant_mcp/      # MCP server (new, separate package)
│   ├── __init__.py
│   ├── __main__.py        # python -m wraquant_mcp
│   ├── server.py
│   ├── context.py
│   ├── adaptor.py
│   ├── registry.py
│   ├── ids.py
│   ├── workspace.py
│   ├── prompts/
│   └── servers/
├── tests/
├── tests_mcp/             # MCP-specific tests
├── pyproject.toml         # wraquant (existing)
├── pyproject.mcp.toml     # wraquant-mcp (new, separate build)
└── ...
```

Or even simpler: wraquant-mcp in its own repo that `pip install wraquant`.

## ID System

Everything gets an ID. IDs are human-readable, auto-generated, namespaced:

```python
# Dataset IDs
"prices_aapl"              # from fetch
"prices_aapl_rsi_14"       # after adding RSI
"returns_aapl"             # computed returns
"features_aapl_v3"         # feature engineering output

# Model IDs
"garch_aapl_gjr_t"         # GARCH(1,1) GJR with t-dist
"hmm_aapl_2state"          # 2-state HMM
"rf_momentum_wf5"          # Random forest, walk-forward 5 splits

# Result IDs
"backtest_rsi_regime"      # Backtest result
"tearsheet_momentum_v2"    # Tearsheet output
"var_portfolio_20240319"   # VaR analysis

# Workspace IDs
"ws_aapl_momentum_2024"    # Workspace
"snap_baseline"            # Snapshot within workspace
```

### ID Generation

```python
from dataclasses import dataclass
from datetime import datetime
import hashlib

@dataclass
class ResourceID:
    """Unique identifier for any resource in the workspace."""

    namespace: str    # "dataset", "model", "result", "workspace"
    name: str         # human-readable name
    version: int = 1  # auto-incremented on updates

    @property
    def id(self) -> str:
        if self.version > 1:
            return f"{self.name}_v{self.version}"
        return self.name

    @property
    def qualified(self) -> str:
        return f"{self.namespace}:{self.id}"


class IDRegistry:
    """Track all resource IDs in a workspace."""

    def __init__(self):
        self._datasets: dict[str, DatasetMeta] = {}
        self._models: dict[str, ModelMeta] = {}
        self._results: dict[str, ResultMeta] = {}

    def register_dataset(self, name, df, source_op=None, parent=None):
        """Register a dataset, auto-version if name exists."""
        if name in self._datasets:
            # Auto-version: prices_aapl → prices_aapl_v2
            version = self._datasets[name].version + 1
            name = f"{name}_v{version}"

        self._datasets[name] = DatasetMeta(
            name=name,
            rows=len(df),
            columns=list(df.columns),
            created=datetime.now(),
            source_op=source_op,
            parent=parent,  # lineage
        )
        return name

    def resolve(self, id_str: str):
        """Resolve an ID to its resource.

        Supports:
          "prices_aapl"           → dataset
          "dataset:prices_aapl"   → explicit namespace
          "garch_aapl"            → model (auto-detected)
          "latest"                → most recently created
        """
        ...

    def lineage(self, id_str: str) -> list[str]:
        """Trace the derivation chain of a resource.

        Returns: ["prices_aapl", "returns_aapl", "features_aapl_v2"]
        """
        ...


@dataclass
class DatasetMeta:
    name: str
    rows: int
    columns: list[str]
    created: datetime
    source_op: str | None    # "fetch_prices", "compute_indicator", etc.
    parent: str | None       # ID of parent dataset (lineage)

@dataclass
class ModelMeta:
    name: str
    model_type: str          # "garch", "hmm", "random_forest"
    created: datetime
    source_dataset: str      # which dataset it was trained on
    params: dict             # hyperparameters used
    metrics: dict            # performance metrics
```

## Decision: Start simple

For the feature branch, start with the "simpler" layout:
- wraquant-mcp as `src/wraquant_mcp/` in the same repo
- Separate pyproject for building (`pyproject.mcp.toml`)
- Or just a `[project.optional-dependencies] mcp = [...]` group

Monorepo split can happen later when it's proven.
