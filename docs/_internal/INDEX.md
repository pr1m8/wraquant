# Internal Documentation Index

## Architecture & Design
- [ARCHITECTURE.md](ARCHITECTURE.md) — Module boundaries, 6-layer DAG, parameter conventions
- [MODULE_GRAPH.md](MODULE_GRAPH.md) — Data flow patterns, integration points
- [MODULE_STATUS.md](MODULE_STATUS.md) — Per-module maturity assessment

## Integration Status
- [INTEGRATION_INDEX.md](INTEGRATION_INDEX.md) — 5-level integration map (coercion → frame types)
- [COERCION_ADOPTION_STATUS.md](COERCION_ADOPTION_STATUS.md) — Which modules use core/_coerce

## MCP / Platform
- [wraquant-mcp-design.md](wraquant-mcp-design.md) — FastMCP + LangChain architecture (60 tools, composable mount)
- [MCP_STATE_ARCHITECTURE.md](MCP_STATE_ARCHITECTURE.md) — DuckDB hybrid state, session persistence, DataFrame-by-reference
- [WRAFIN_REFACTOR_CONSIDERATION.md](WRAFIN_REFACTOR_CONSIDERATION.md) — Whether to restructure as wrafin umbrella (decision: not yet)

## Superseded (merged into above)
- ~~INTEGRATION_DEBT.md~~ → merged into INTEGRATION_INDEX.md
- ~~INTEGRATION_PLAN.md~~ → merged into INTEGRATION_INDEX.md
- ~~TYPE_SYSTEM_PLAN.md~~ → merged into TYPE_SYSTEM_ANALYSIS.md
- ~~APPLICATION_INTEGRATION_PLAN.md~~ → merged into ARCHITECTURE.md
- ~~NEXT_SESSION_PRIORITIES.md~~ → merged into INTEGRATION_INDEX.md

## Decision: Type System
- [TYPE_SYSTEM_ANALYSIS.md](TYPE_SYSTEM_ANALYSIS.md) — Coerce-first pattern, frame/ redesign, torch overloads
  - Decision: Use isinstance + coerce_array/coerce_series (sklearn pattern)
  - Decision: PriceSeries/ReturnSeries as pd.Series subclasses with metadata
  - Decision: No plum-dispatch, no custom ExtensionDtype, no beartype in hot paths
