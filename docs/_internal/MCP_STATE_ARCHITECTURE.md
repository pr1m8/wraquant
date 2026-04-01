# wraquant-mcp: Stateful MCP Server Design Document

## Problem Statement

When an LLM agent uses wraquant via MCP, it cannot pass large DataFrames (10K+ rows)
as tool arguments -- JSON serialization would consume the entire context window.
We need an architecture where:

1. DataFrames live server-side, referenced by string ID (not by value)
2. Agents say `"fit GARCH on dataset_123"` not `"fit GARCH on [10000 rows of JSON]"`
3. Mutations happen in-place on the server; the agent sees only metadata
4. Multiple DataFrames coexist in a session workspace
5. Derived results (fitted models, regime labels, signals) are also stored by reference

---

## 1. Recommended State Architecture: Hybrid DuckDB + Python Dict

### Why not pure dict?

A plain `dict[str, pd.DataFrame]` works for storage but gives you nothing for
free -- no SQL queries, no cross-table joins, no columnar aggregation, no disk
persistence. Every analytical operation must be hand-written in Python.

### Why not pure DuckDB?

DuckDB cannot store arbitrary Python objects (fitted GARCH models, sklearn
pipelines, Plotly figures). It is excellent for tabular data but not for the
full workspace of a quant agent.

### The hybrid approach

```
WorkspaceState
  |
  +-- DuckDB in-memory connection       # All tabular data
  |     - Named tables (prices, returns, features, signals)
  |     - SQL queryable by the agent
  |     - Zero-copy pandas registration via replacement scans
  |     - Optional disk persistence (.duckdb file)
  |
  +-- Python object store (dict)         # Non-tabular objects
  |     - Fitted models (GARCH, HMM, sklearn)
  |     - Result dataclasses (GARCHResult, BacktestResult, ForecastResult)
  |     - Configuration snapshots
  |     - Plotly figures (serialized as JSON)
  |
  +-- Operation log (list[OperationRecord])  # Audit trail
        - Every mutation recorded
        - Enables undo/rollback
        - Tracks dataset lineage (parent -> child)
```

### DuckDB specifics

**Registration (zero-copy, <1ms):**
```python
import duckdb

con = duckdb.connect()  # in-memory
df = pd.DataFrame({"close": [...], "volume": [...]})

# Method 1: Replacement scan (auto-detected from Python scope)
con.sql("SELECT * FROM df WHERE close > 100")

# Method 2: Explicit registration as virtual view (preferred for MCP)
con.register("spy_prices", df)
con.sql("SELECT AVG(close) FROM spy_prices")

# Method 3: Materialized copy (for persistence to disk)
con.execute("CREATE TABLE spy_prices AS SELECT * FROM df")
```

**Performance characteristics:**
- Registration: <1ms (just a pointer, zero copy)
- Simple filter on 1M rows: ~400 microseconds
- Group-by on 100M rows: ~3.8 seconds (vs pandas ~19.6s)
- Memory: reads pandas memory structures directly, no duplication
- Concurrent reads: fully supported (single writer)

**Disk persistence:**
```python
# Persistent workspace that survives process restart
con = duckdb.connect("/path/to/workspace.duckdb")
con.execute("CREATE TABLE IF NOT EXISTS spy_prices AS SELECT * FROM df")
# On next session startup: tables already present
```

**Why DuckDB is ideal for MCP:**
- The agent can run arbitrary SQL against stored DataFrames without the LLM
  seeing any data rows -- just schema metadata and aggregate results
- DuckDB already has its own MCP extension (`duckdb_mcp`) with `mcp_server_start()`
  and `mcp_publish_table()`, proving the pattern works
- The MotherDuck MCP server (github.com/motherduckdb/mcp-server-motherduck) is
  a production reference implementation

---

## 2. Tool API Patterns: How Tools Reference Data

### Core principle: IDs in, IDs out

Every tool that touches data takes a `dataset_id: str` parameter and returns
either a new dataset ID or a summary dict. The agent never sees raw data.

### Dataset ID conventions

```
Naming pattern: {source}_{content}_{transform}

Examples:
  "spy_prices"           -- raw OHLCV from data fetch
  "spy_returns"          -- derived from spy_prices via log returns
  "spy_features"         -- derived from spy_returns via feature engineering
  "spy_garch_vol"        -- conditional volatility column added
  "spy_regime_labels"    -- regime detection output
  "portfolio_weights"    -- optimization result
```

### Tool taxonomy

**Category 1: Data ingestion (creates new datasets)**
```python
@mcp.tool()
async def load_prices(
    ticker: str,
    start: str,
    end: str,
    dataset_id: str | None = None,  # auto-generated if omitted
    ctx: Context = None,
) -> str:
    """Fetch OHLCV prices and store in workspace.

    Returns: JSON with dataset_id, shape, date_range, columns.
    """
    from wraquant.data import fetch_prices
    ws: Workspace = ctx.request_context.lifespan_context
    df = fetch_prices(ticker, start=start, end=end)
    did = dataset_id or f"{ticker.lower()}_prices"
    ws.store_dataframe(did, df, source="yfinance", parent=None)
    return json.dumps({
        "dataset_id": did,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "date_range": [str(df.index.min()), str(df.index.max())],
    })
```

**Category 2: Transformations (dataset in, dataset out)**
```python
@mcp.tool()
async def compute_returns(
    dataset_id: str,
    method: str = "log",       # "log" | "simple"
    output_id: str | None = None,
    ctx: Context = None,
) -> str:
    """Compute return series from price data. Stores result as new dataset."""
    ws: Workspace = ctx.request_context.lifespan_context
    df = ws.get_dataframe(dataset_id)
    from wraquant.stats import compute_returns as _compute
    returns = _compute(df["close"], method=method)
    out_id = output_id or f"{dataset_id.replace('_prices', '')}_returns"
    ws.store_dataframe(out_id, returns.to_frame("returns"), parent=dataset_id)
    return json.dumps({
        "dataset_id": out_id,
        "shape": [len(returns), 1],
        "stats": {
            "mean": float(returns.mean()),
            "std": float(returns.std()),
            "min": float(returns.min()),
            "max": float(returns.max()),
        },
    })
```

**Category 3: Model fitting (dataset in, model + dataset out)**
```python
@mcp.tool()
async def fit_garch(
    dataset_id: str,
    column: str = "returns",
    p: int = 1,
    q: int = 1,
    model_id: str | None = None,
    ctx: Context = None,
) -> str:
    """Fit a GARCH(p,q) model. Stores fitted model and conditional vol series."""
    ws: Workspace = ctx.request_context.lifespan_context
    df = ws.get_dataframe(dataset_id)
    from wraquant.vol import fit_garch as _fit
    result: GARCHResult = _fit(df[column], p=p, q=q)

    mid = model_id or f"{dataset_id}_garch_{p}_{q}"
    ws.store_model(mid, result, parent=dataset_id)

    # Also store conditional vol as a queryable dataset
    vol_id = f"{mid}_vol"
    ws.store_dataframe(vol_id, result.conditional_volatility.to_frame("cond_vol"),
                       parent=dataset_id)

    return json.dumps({
        "model_id": mid,
        "vol_dataset_id": vol_id,
        "params": result.params,
        "aic": result.aic,
        "bic": result.bic,
        "persistence": result.persistence,
        "half_life": result.half_life,
    })
```

**Category 4: Queries (read-only, returns summaries)**
```python
@mcp.tool()
async def query_dataset(
    sql: str,
    ctx: Context = None,
) -> str:
    """Run SQL against workspace datasets. Returns up to 50 rows as JSON."""
    ws: Workspace = ctx.request_context.lifespan_context
    result = ws.query(sql, max_rows=50)
    return json.dumps(result)

@mcp.tool()
async def describe_dataset(
    dataset_id: str,
    ctx: Context = None,
) -> str:
    """Return schema, shape, and summary statistics for a dataset."""
    ws: Workspace = ctx.request_context.lifespan_context
    info = ws.describe(dataset_id)
    return json.dumps(info)
```

**Category 5: Workspace management**
```python
@mcp.tool()
async def list_datasets(ctx: Context = None) -> str:
    """List all datasets in the workspace with shapes and lineage."""

@mcp.tool()
async def list_models(ctx: Context = None) -> str:
    """List all fitted models in the workspace."""

@mcp.tool()
async def drop_dataset(dataset_id: str, ctx: Context = None) -> str:
    """Remove a dataset from the workspace."""

@mcp.tool()
async def rename_dataset(old_id: str, new_id: str, ctx: Context = None) -> str:
    """Rename a dataset."""

@mcp.tool()
async def save_workspace(path: str, ctx: Context = None) -> str:
    """Persist entire workspace to disk."""

@mcp.tool()
async def load_workspace(path: str, ctx: Context = None) -> str:
    """Restore workspace from disk."""
```

### MCP Resources (complementary to tools)

In addition to tools, expose datasets as MCP resources so agents can
"read" metadata without calling a tool:

```python
@mcp.resource("workspace://datasets/{dataset_id}/schema")
async def dataset_schema(dataset_id: str, ctx: Context) -> str:
    """Return column names, dtypes, and row count."""

@mcp.resource("workspace://datasets/{dataset_id}/preview")
async def dataset_preview(dataset_id: str, ctx: Context) -> str:
    """Return first 5 rows as formatted table."""

@mcp.resource("workspace://models/{model_id}/summary")
async def model_summary(model_id: str, ctx: Context) -> str:
    """Return model summary string."""
```

---

## 3. Mutation Handling

### Recommendation: Append-only operation log with snapshot checkpoints

Three approaches were evaluated:

| Approach | Pros | Cons |
|----------|------|------|
| **In-place mutation** | Fast, simple | No undo, no lineage |
| **Copy-on-write** | Safe, pandas 3.0 native | Memory doubles on every mutation |
| **Event log + snapshots** | Full lineage, undo, audit | Slightly more complex |

**Decision: Event log + snapshots.** This aligns with pandas 3.0's CoW semantics
(mutations create new objects anyway) and provides the audit trail agents need.

### How pandas 3.0 CoW affects us

In pandas 3.0 (which wraquant already requires as `pandas>=3.0.1`), Copy-on-Write
is enabled by default and cannot be turned off. Key implications:

- `df2 = df[df["col"] > 0]` shares memory with `df` until either is modified
- Any mutation on `df2` triggers a lazy copy -- the original is never corrupted
- Chained indexing (`df[mask]["col"] = value`) now raises an error
- This means our workspace can safely hand out DataFrame references to tools
  without worrying about accidental cross-contamination

### Operation log schema

```python
@dataclass
class OperationRecord:
    """Immutable record of a workspace mutation."""
    timestamp: datetime
    operation: str          # "store_dataframe", "store_model", "drop", "mutate"
    target_id: str          # dataset or model ID affected
    parent_id: str | None   # what this was derived from
    tool_name: str          # MCP tool that triggered this
    params: dict[str, Any]  # tool parameters (for replay)
    snapshot_hash: str      # hash of data at this point (for integrity)
    reversible: bool        # can this operation be undone?
```

### Undo mechanism

```python
async def undo_last(ctx: Context) -> str:
    """Undo the most recent reversible operation.

    For 'store_dataframe': drops the created dataset
    For 'mutate': restores from the pre-mutation snapshot
    For 'drop': restores from the pre-drop snapshot
    """
    ws = ctx.request_context.lifespan_context
    last_op = ws.op_log.pop()
    if last_op.operation == "store_dataframe":
        ws._drop_internal(last_op.target_id)
    elif last_op.operation == "mutate":
        ws._restore_snapshot(last_op.target_id, last_op.snapshot_hash)
    return f"Undid {last_op.operation} on {last_op.target_id}"
```

### Lineage tracking

Every dataset knows its parent(s), enabling the agent to ask
"what was this derived from?" and enabling cascade operations:

```python
def get_lineage(self, dataset_id: str) -> list[str]:
    """Return ordered list of ancestor dataset IDs."""
    chain = []
    current = dataset_id
    while current:
        chain.append(current)
        parent = self._metadata[current].get("parent")
        current = parent
    return list(reversed(chain))

# Example lineage:
# spy_prices -> spy_returns -> spy_features -> spy_garch_1_1_vol
```

---

## 4. Session Persistence

### On-disk format

```
workspace_2025-01-15/
  +-- workspace.duckdb          # All tabular data (DuckDB native format)
  +-- models/
  |     +-- spy_garch_1_1.pkl   # Fitted models (joblib serialized)
  |     +-- spy_hmm_3.pkl
  +-- metadata.json             # Dataset metadata, lineage, column info
  +-- op_log.jsonl              # Operation log (append-only, newline-delimited)
  +-- manifest.json             # Workspace manifest: version, created_at, checksum
```

### Serialization strategy

| Object type | Format | Library | Rationale |
|-------------|--------|---------|-----------|
| DataFrames | DuckDB tables | duckdb | Fast, columnar, SQL-queryable |
| Fitted sklearn models | .pkl | joblib | Standard, handles numpy arrays |
| Fitted GARCH/arch models | .pkl | joblib | arch models are picklable |
| GARCHResult dataclasses | .pkl + .json | joblib + json | pkl for model, json for params |
| Operation log | .jsonl | json | Human-readable, append-only |
| Metadata | .json | json | Human-readable |

### Save/restore implementation

```python
async def save_workspace(self, path: str) -> dict:
    """Persist entire workspace to disk.

    1. EXPORT DATABASE to DuckDB file (all tables)
    2. Serialize each model with joblib
    3. Write metadata.json
    4. Write op_log.jsonl
    5. Compute manifest checksum
    """
    workspace_dir = Path(path)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # 1. DuckDB tables -> file
    disk_con = duckdb.connect(str(workspace_dir / "workspace.duckdb"))
    for table_name in self.list_tables():
        df = self.get_dataframe(table_name)
        disk_con.register("__tmp", df)
        disk_con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM __tmp")
        disk_con.unregister("__tmp")
    disk_con.close()

    # 2. Models -> joblib
    models_dir = workspace_dir / "models"
    models_dir.mkdir(exist_ok=True)
    for model_id, model_obj in self._models.items():
        joblib.dump(model_obj, models_dir / f"{model_id}.pkl")

    # 3-5. Metadata, log, manifest
    ...

async def load_workspace(self, path: str) -> dict:
    """Restore workspace from disk. Inverse of save_workspace."""
    workspace_dir = Path(path)
    disk_con = duckdb.connect(str(workspace_dir / "workspace.duckdb"))
    for table_name in disk_con.execute("SHOW TABLES").fetchall():
        name = table_name[0]
        df = disk_con.execute(f"SELECT * FROM {name}").fetchdf()
        self.store_dataframe(name, df)
    disk_con.close()
    # Restore models, metadata, etc.
    ...
```

### Security for persistence

- **Model deserialization**: joblib/pickle is inherently unsafe for untrusted
  data. Only load workspaces the user themselves saved. Consider `skops.io`
  for sklearn models if workspace sharing is needed.
- **Path validation**: Workspace paths must be within a configured base
  directory. No `../` traversal.
- **Checksum verification**: Manifest includes SHA-256 of all files. Reject
  tampered workspaces.

---

## 5. Workspace Class: Complete Implementation Skeleton

```python
from __future__ import annotations

import json
import hashlib
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


@dataclass
class DatasetMetadata:
    """Metadata for a stored dataset."""
    dataset_id: str
    shape: tuple[int, int]
    columns: list[str]
    dtypes: dict[str, str]
    created_at: str
    parent_id: str | None = None
    source: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class OperationRecord:
    """Immutable record of a workspace mutation."""
    timestamp: str
    operation: str
    target_id: str
    parent_id: str | None
    tool_name: str
    params: dict[str, Any]
    reversible: bool = True


class Workspace:
    """Server-side workspace for MCP agent sessions.

    Holds DataFrames in DuckDB (queryable via SQL) and arbitrary
    Python objects in a dict store. All mutations are logged.
    """

    def __init__(self, persist_path: str | None = None):
        # DuckDB connection: in-memory by default, file-backed if path given
        self._con = duckdb.connect(persist_path or ":memory:")
        self._models: dict[str, Any] = {}
        self._metadata: dict[str, DatasetMetadata] = {}
        self._op_log: list[OperationRecord] = []
        self._snapshots: dict[str, pd.DataFrame] = {}  # for undo

    # ── Dataset operations ─────────────────────────────────────

    def store_dataframe(
        self,
        dataset_id: str,
        df: pd.DataFrame,
        parent: str | None = None,
        source: str | None = None,
    ) -> DatasetMetadata:
        """Store a DataFrame in the workspace, queryable via SQL."""
        # Snapshot existing data if overwriting (for undo)
        if dataset_id in self._metadata:
            old_df = self.get_dataframe(dataset_id)
            self._snapshots[dataset_id] = old_df.copy()

        # Register in DuckDB (zero-copy via replacement scan)
        self._con.register(dataset_id, df)

        # Store metadata
        meta = DatasetMetadata(
            dataset_id=dataset_id,
            shape=df.shape,
            columns=list(df.columns),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            created_at=datetime.now(timezone.utc).isoformat(),
            parent_id=parent,
            source=source,
        )
        self._metadata[dataset_id] = meta

        self._log("store_dataframe", dataset_id, parent)
        return meta

    def get_dataframe(self, dataset_id: str) -> pd.DataFrame:
        """Retrieve a DataFrame by ID."""
        if dataset_id not in self._metadata:
            raise KeyError(f"Dataset '{dataset_id}' not found in workspace")
        return self._con.execute(f"SELECT * FROM \"{dataset_id}\"").fetchdf()

    def query(self, sql: str, max_rows: int = 100) -> dict:
        """Execute SQL against workspace datasets. Returns dict with columns+rows."""
        result = self._con.execute(sql)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchmany(max_rows)
        return {
            "columns": columns,
            "rows": [list(row) for row in rows],
            "truncated": len(rows) == max_rows,
        }

    def describe(self, dataset_id: str) -> dict:
        """Return schema, shape, and summary statistics."""
        meta = self._metadata[dataset_id]
        stats = self._con.execute(
            f"SELECT * FROM (SUMMARIZE \"{dataset_id}\")"
        ).fetchdf().to_dict(orient="records")
        return {
            "dataset_id": dataset_id,
            "shape": list(meta.shape),
            "columns": meta.columns,
            "dtypes": meta.dtypes,
            "parent": meta.parent_id,
            "source": meta.source,
            "created_at": meta.created_at,
            "statistics": stats,
        }

    def list_datasets(self) -> list[dict]:
        """List all datasets with metadata."""
        return [
            {
                "dataset_id": m.dataset_id,
                "shape": list(m.shape),
                "columns": m.columns,
                "parent": m.parent_id,
                "created_at": m.created_at,
            }
            for m in self._metadata.values()
        ]

    def drop_dataset(self, dataset_id: str) -> None:
        """Remove a dataset from the workspace."""
        self._snapshots[dataset_id] = self.get_dataframe(dataset_id)
        self._con.unregister(dataset_id)
        del self._metadata[dataset_id]
        self._log("drop_dataset", dataset_id, None)

    # ── Model operations ───────────────────────────────────────

    def store_model(
        self, model_id: str, model: Any, parent: str | None = None
    ) -> None:
        """Store a fitted model or result object."""
        self._models[model_id] = model
        self._log("store_model", model_id, parent)

    def get_model(self, model_id: str) -> Any:
        """Retrieve a fitted model by ID."""
        if model_id not in self._models:
            raise KeyError(f"Model '{model_id}' not found in workspace")
        return self._models[model_id]

    def list_models(self) -> list[dict]:
        """List all models with type info."""
        return [
            {
                "model_id": mid,
                "type": type(model).__name__,
                "module": type(model).__module__,
            }
            for mid, model in self._models.items()
        ]

    # ── Lineage ────────────────────────────────────────────────

    def get_lineage(self, dataset_id: str) -> list[str]:
        """Return ancestry chain: [root, ..., parent, self]."""
        chain = []
        current = dataset_id
        visited = set()
        while current and current not in visited:
            visited.add(current)
            chain.append(current)
            meta = self._metadata.get(current)
            current = meta.parent_id if meta else None
        return list(reversed(chain))

    # ── Operation log ──────────────────────────────────────────

    def _log(self, operation: str, target_id: str, parent_id: str | None) -> None:
        self._op_log.append(OperationRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            operation=operation,
            target_id=target_id,
            parent_id=parent_id,
            tool_name="",  # filled by tool wrapper
        ))

    def undo_last(self) -> str:
        """Undo the most recent operation, if reversible."""
        if not self._op_log:
            return "Nothing to undo"
        last = self._op_log.pop()
        if last.operation == "store_dataframe" and last.target_id in self._metadata:
            self._con.unregister(last.target_id)
            del self._metadata[last.target_id]
            return f"Undid store of {last.target_id}"
        elif last.operation == "drop_dataset" and last.target_id in self._snapshots:
            self.store_dataframe(last.target_id, self._snapshots.pop(last.target_id))
            return f"Restored dropped {last.target_id}"
        return f"Cannot undo {last.operation} on {last.target_id}"

    # ── Cleanup ────────────────────────────────────────────────

    def close(self) -> None:
        """Release DuckDB connection."""
        self._con.close()
```

### FastMCP integration

```python
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from mcp.server.fastmcp import FastMCP, Context

@asynccontextmanager
async def workspace_lifespan(server: FastMCP) -> AsyncIterator[Workspace]:
    """Manage workspace lifecycle for the MCP server."""
    ws = Workspace()
    try:
        yield ws
    finally:
        ws.close()

mcp = FastMCP("wraquant-mcp", lifespan=workspace_lifespan)

@mcp.tool()
async def load_prices(ticker: str, start: str, end: str, ctx: Context) -> str:
    ws: Workspace = ctx.request_context.lifespan_context
    # ... use ws.store_dataframe(), ws.get_dataframe(), etc.
```

---

## 6. Security Considerations

### Threat model

The MCP server runs as a subprocess with the user's permissions. An agent
could potentially:

1. **Read sensitive data** from the filesystem via `load_csv("/etc/passwd")`
2. **Delete workspace data** accidentally or via confused prompts
3. **Execute arbitrary code** if we expose a "run Python" tool
4. **Exhaust memory** by loading enormous datasets
5. **Deserialize malicious pickles** from untrusted workspace files

### Mitigations

**1. Path sandboxing**
```python
class Workspace:
    ALLOWED_BASE_DIRS: list[Path] = [
        Path.home() / "wraquant-workspaces",
        Path("/tmp/wraquant"),
    ]

    def _validate_path(self, path: str) -> Path:
        resolved = Path(path).resolve()
        if not any(resolved.is_relative_to(base) for base in self.ALLOWED_BASE_DIRS):
            raise PermissionError(f"Path {path} outside allowed directories")
        return resolved
```

**2. No arbitrary code execution**
- Do NOT expose a "run Python code" tool. Every operation must go through
  a typed, validated tool function.
- The `query_dataset` tool only runs SQL through DuckDB, which has no
  filesystem access by default (unless extensions are loaded).
- DuckDB SQL is safe: no `COPY TO`, no `LOAD`, no shell access.

**3. Resource limits**
```python
MAX_DATASETS = 100
MAX_DATASET_ROWS = 10_000_000
MAX_DATASET_COLS = 1_000
MAX_MODELS = 50
MAX_WORKSPACE_SIZE_MB = 2048

def store_dataframe(self, dataset_id, df, ...):
    if len(self._metadata) >= MAX_DATASETS:
        raise ResourceError("Maximum dataset count reached")
    if df.shape[0] > MAX_DATASET_ROWS:
        raise ResourceError(f"Dataset exceeds {MAX_DATASET_ROWS} row limit")
    ...
```

**4. Read-only mode**
```python
# For production/shared deployments, disable mutations
mcp = FastMCP("wraquant-mcp", lifespan=workspace_lifespan)

@mcp.tool()
async def drop_dataset(dataset_id: str, ctx: Context) -> str:
    ws = ctx.request_context.lifespan_context
    if ws.read_only:
        return "ERROR: Workspace is in read-only mode"
    ...
```

**5. Safe deserialization**
- Only load workspace files from paths the user explicitly provides.
- Log a warning when deserializing pickled models.
- Consider `skops.io` for sklearn models (validates before loading).
- Verify manifest checksums before loading any workspace file.

**6. DuckDB SQL safety**
```python
BLOCKED_SQL_PATTERNS = [
    r"\bCOPY\b",
    r"\bEXPORT\b",
    r"\bIMPORT\b",
    r"\bLOAD\b",
    r"\bINSTALL\b",
    r"\bATTACH\b",
]

def query(self, sql: str, max_rows: int = 100) -> dict:
    for pattern in BLOCKED_SQL_PATTERNS:
        if re.search(pattern, sql, re.IGNORECASE):
            raise PermissionError(f"SQL statement contains blocked keyword")
    ...
```

---

## 7. Arrow Flight / IPC Assessment

### Verdict: Not needed for wraquant-mcp v1

Arrow Flight is designed for high-throughput data transfer between separate
processes or across the network. In the MCP architecture:

- The MCP server and wraquant library run in the **same Python process**
- DataFrame handoff is a **Python object reference** (zero-copy by default)
- There is no serialization boundary between the MCP tool function and wraquant

Arrow Flight would matter if:
- wraquant ran as a separate microservice (it does not)
- Multiple MCP servers needed to share DataFrames across processes
- We needed to stream data to a remote client (the agent never sees raw data)

**Recommendation:** Revisit Arrow Flight only if wraquant-mcp evolves into a
multi-process or distributed architecture. For now, DuckDB's zero-copy
pandas integration provides all the efficiency we need.

---

## 8. Multi-DataFrame Workspace Patterns

### Auto-linking related datasets

When a tool derives a dataset from another, the parent-child relationship
is automatically recorded. The agent can traverse lineage:

```
Agent: "What datasets do I have?"
Server: [
  {id: "spy_prices", shape: [2520, 6], parent: null},
  {id: "spy_returns", shape: [2519, 1], parent: "spy_prices"},
  {id: "spy_features", shape: [2519, 15], parent: "spy_returns"},
  {id: "spy_garch_1_1_vol", shape: [2519, 1], parent: "spy_returns"},
  {id: "qqq_prices", shape: [2520, 6], parent: null},
]

Agent: "What is spy_features derived from?"
Server: lineage = ["spy_prices", "spy_returns", "spy_features"]
```

### Cross-dataset operations

DuckDB enables natural cross-dataset queries:

```sql
-- Agent asks: "correlate SPY and QQQ returns"
SELECT corr(s.returns, q.returns)
FROM spy_returns s
JOIN qqq_returns q ON s.date = q.date

-- Agent asks: "which dates had both high vol and regime 2?"
SELECT v.date, v.cond_vol, r.regime
FROM spy_garch_1_1_vol v
JOIN spy_regime_labels r ON v.date = r.date
WHERE r.regime = 2 AND v.cond_vol > 0.03
```

### Workspace templates

Pre-configured workspaces for common workflows:

```python
async def create_equity_workspace(tickers: list[str], start: str, end: str):
    """Create a standard equity analysis workspace.

    For each ticker:
      1. {ticker}_prices  (OHLCV)
      2. {ticker}_returns (log returns)
      3. {ticker}_features (TA indicators)

    Plus:
      - universe_returns (wide DataFrame, all tickers)
      - correlation_matrix (stored as dataset)
    """
```

---

## 9. Comparison with Existing MCP Servers

| Server | State model | Data reference | SQL support | Model storage |
|--------|------------|----------------|-------------|---------------|
| **postgres-mcp** | External DB | Table names | Full SQL | No |
| **pandas-mcp-server** | In-memory dict | File paths | No | No |
| **motherduck-mcp** | DuckDB/MotherDuck | Table names | Full SQL | No |
| **duckdb_mcp extension** | DuckDB native | Table names | Full SQL | No |
| **wraquant-mcp (proposed)** | DuckDB + dict | Dataset IDs | Full SQL | Yes (fitted models) |

wraquant-mcp is unique in combining SQL-queryable tabular storage with
Python object storage for fitted models -- essential for quant workflows
where a GARCH fit, regime detector, or portfolio optimizer needs to persist
across tool calls.

---

## 10. Implementation Roadmap

### Phase 1: Core workspace (MVP)
- [ ] `Workspace` class with DuckDB backend
- [ ] `store_dataframe`, `get_dataframe`, `query`, `describe`
- [ ] FastMCP server with lifespan context
- [ ] 5 essential tools: `load_prices`, `compute_returns`, `describe_dataset`,
      `query_dataset`, `list_datasets`
- [ ] Basic operation log

### Phase 2: Model storage + analytics tools
- [ ] `store_model`, `get_model`, `list_models`
- [ ] `fit_garch`, `detect_regimes`, `compute_ta_indicators`
- [ ] Lineage tracking
- [ ] `undo_last`

### Phase 3: Persistence + security
- [ ] `save_workspace`, `load_workspace`
- [ ] Path sandboxing
- [ ] Resource limits
- [ ] SQL safety filters

### Phase 4: Advanced
- [ ] Workspace templates
- [ ] MCP resources (schema, preview)
- [ ] Multi-session support (separate workspace per session ID)
- [ ] Streaming results for large queries

---

## 11. Dependencies

Required (add to `pyproject.toml` as a new `mcp` dependency group):

```toml
[project.optional-dependencies]
mcp = [
    "mcp>=1.9.0",            # Official MCP Python SDK (includes FastMCP v1)
    "duckdb>=1.4.4",          # Already in 'accelerate' group
    "joblib>=1.5.3",          # Already in 'accelerate' group
]
```

Optional (for advanced features):
- `fastmcp>=3.0` -- standalone FastMCP with OpenTelemetry, OAuth, versioning
- `skops` -- safe model deserialization
- `pyarrow` -- already in 'accelerate' group, for Arrow-native DuckDB paths

---

## Research Sources

- [DuckDB Python Data Ingestion](https://duckdb.org/docs/current/clients/python/data_ingestion)
- [DuckDB Python Quickstart Part 2 (Pandas, Arrow, Polars)](https://motherduck.com/learn-more/duckdb-python-quickstart-part2/)
- [MotherDuck MCP Server](https://github.com/motherduckdb/mcp-server-motherduck)
- [DuckDB MCP Extension](https://duckdb.org/community_extensions/extensions/duckdb_mcp)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [FastMCP Context Documentation](https://gofastmcp.com/servers/context)
- [FastMCP Server Documentation](https://gofastmcp.com/python-sdk/fastmcp-server-server)
- [pandas-mcp-server](https://github.com/marlonluo2018/pandas-mcp-server)
- [Pandas 3.0 Copy-on-Write Documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/copy_on_write.html)
- [Pandas 3.0 CoW Changes](https://dev.to/serada/2026-latest-pandas-30-is-here-copy-on-write-pyarrow-and-what-you-need-to-know-hme)
- [Apache Arrow IPC Documentation](https://arrow.apache.org/docs/python/ipc.html)
- [Apache Arrow Flight RPC](https://arrow.apache.org/docs/python/flight.html)
- [MCP Session State Management](https://codesignal.com/learn/courses/developing-and-integrating-an-mcp-server-in-typescript/lessons/stateful-mcp-server-sessions)
- [MCP Session Isolation Issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1087)
- [DuckDB vs Pandas Benchmarks (KDnuggets)](https://www.kdnuggets.com/we-benchmarked-duckdb-sqlite-and-pandas-on-1m-rows-heres-what-happened)
- [DuckDB + PyArrow: 2900x Faster](https://codecut.ai/efficiently-handle-large-datasets-with-duckdb-and-pyarrow/)
- [DataLineagePy](https://github.com/Arbaznazir/DataLineagePy)
- [MCP Server Security Best Practices](https://toolradar.com/blog/mcp-server-security-best-practices)
- [MCP Server Sandbox Isolation](https://claudecodeguides.com/mcp-server-sandbox-isolation-security-guide/)
- [scikit-learn Model Persistence](https://scikit-learn.org/stable/model_persistence.html)
- [Python Event Sourcing Library](https://eventsourcing.readthedocs.io/)
- [Pydantic AI State Management](https://github.com/pydantic/pydantic-ai/issues/4322)
