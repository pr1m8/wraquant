# wrafin Refactor Consideration

**Status:** DISCUSSION — Do NOT implement without full design review

## The Idea

Refactor wraquant into **wrafin** — a broader financial platform with wraquant[quant]
as one installable extra. This separates concerns and makes each piece independently
useful.

## Proposed Package Structure

```
wrafin/
├── wrafin[core]          # Core types, coercion, frame, config
├── wrafin[data]          # Data fetching, cleaning, validation, EDA
├── wrafin[viz]           # Plotly dashboards, charts, viz registry
├── wrafin[stats]         # Stats, TS, causal inference
├── wrafin[ta]            # Technical analysis (265 indicators)
├── wrafin[risk]          # Risk + vol (GARCH, VaR, copulas, beta)
├── wrafin[quant]         # Everything: opt, backtest, ml, pricing,
│                         # regimes, econometrics, experiment lab
├── wrafin[micro]         # Microstructure, execution
├── wrafin[scale]         # DuckDB, Dask, Ray, io
├── wrafin[mcp]           # MCP server + adaptors
└── wrafin[all]           # Everything
```

## Questions to Answer First

1. **Is the refactor worth it?**
   - Pro: cleaner separation, smaller installs, each piece usable alone
   - Con: massive refactoring, breaking changes, import path changes
   - Con: monorepo vs multi-package complexity

2. **Monorepo vs separate packages?**
   - Monorepo (one repo, multiple pyproject.toml): easier to develop
   - Separate repos: cleaner but harder to coordinate releases
   - Or: keep as one package with optional extras (current pattern works)

3. **What existing MCPs should we integrate/learn from?**
   - Pandas MCP server — how do they handle DataFrame state?
   - DuckDB MCP server — SQL-first approach
   - QuantConnect MCP — financial data + backtesting
   - OpenBB MCP — financial data platform
   Need to research these before designing our approach.

4. **Naming: wraquant vs wrafin?**
   - wraquant is already published on PyPI
   - wrafin could be the umbrella, wraquant stays as the quant module
   - Or: keep wraquant as the name, just restructure extras

5. **Migration path?**
   - Can we do this incrementally? (rename extras, move modules)
   - Or does it require a clean break? (new package, deprecation period)

## Current Extras (what we have)

wraquant already has 30 optional groups. The question is whether to
restructure these into a hierarchical `wrafin[component]` pattern
or keep the flat extras.

## Research Needed

- How does OpenBB structure their platform?
- How does QuantConnect's MCP work?
- Existing DuckDB MCP servers and their patterns
- How to use pre-existing MCPs alongside our own
- Monorepo tooling (hatch workspaces, PDM workspaces, uv workspaces)

## Decision Criteria

1. Does restructuring make the MCP story cleaner?
2. Does it help users install only what they need?
3. Is the migration cost justified by the architectural benefit?
4. Can we do it incrementally or is it all-or-nothing?

## My Current Lean

Keep wraquant as-is for now. The 30 optional extras already provide
granular install. The MCP server is a separate package (wraquant-mcp)
that depends on wraquant. If we later want wrafin as an umbrella,
we can create it as a thin meta-package that depends on wraquant.

The bigger win is getting wraquant-mcp working well, which doesn't
require restructuring the core library.

BUT: worth researching OpenBB and existing financial MCPs first to
see if they reveal a better architecture.
