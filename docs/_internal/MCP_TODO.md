# MCP TODO — Next Session

## Fix test failures (11 failed, 63 errors)

### Root causes identified:
1. `RDResult` missing `t_stat` attribute — causal/treatment.py returns different dataclass
2. `bulk_volume_classification()` signature mismatch — needs high, low, volume not just prices
3. `granger_causality` import error — function name or module path wrong
4. Panel regression — Series truth value ambiguity
5. Almgren-Chriss — assertion value out of range
6. Various attribute mismatches between test expectations and actual wraquant API

### Fix approach:
- Read each failing test, check actual wraquant function signature, fix test expectations
- Don't change wraquant core — fix the test/server to match actual API

## Prompt expansion (user request: "WAY bigger and longer with system")

Current prompts are short step lists. Need to be:
1. **System-level prompts** with deep context about wraquant capabilities
2. **Much longer** — 50-100 lines per prompt, not 10-15
3. **Include wraquant module docstrings** in the system context
4. **Include DuckDB state model** explanation
5. **Include tool chaining patterns** with examples
6. **Include interpretation guidance** for each step

### Add MCP system prompt:
```python
@mcp.prompt()
def system_context():
    """Comprehensive system prompt with full wraquant context."""
    # Include: module overview, tool inventory, state model,
    # chaining patterns, interpretation guide
```

## Module docstring injection

Auto-inject wraquant module docstrings into MCP tool descriptions
so agents get rich context when they call list_tools().

## Test with real MCP clients

1. MCP Inspector: `npx @modelcontextprotocol/inspector python -m wraquant_mcp`
2. Claude Desktop config
3. DuckDB MCP composition test
