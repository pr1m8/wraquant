# FMP Docs Compiler Pro

A docs-first compiler for Financial Modeling Prep that turns documentation pages into a normalized internal catalog, then generates:

- `catalog.json`
- `manifest.json`
- `openapi.json`
- `tools.json`
- a generated FastAPI wrapper scaffold

## Improvements in this version

- Rich + Typer CLI
- async `httpx` crawling with retries, backoff, token-bucket rate limiting, and `Retry-After` support
- cacheable raw HTML snapshots for index pages and endpoint pages
- manifest generation with crawl stats and warnings
- stronger parameter parsing from tables, lists, and definition lists
- optional verification mode for classifying endpoints as verified, rate limited, plan gated, invalid API key, skipped, or failed
- better generated FastAPI wrapper code with request models and an upstream client adapter
- offline fixture-backed end-to-end tests

## Quick start

```bash
pdm install
pdm run fmp-docs compile-fixtures --fixtures-dir tests/fixtures/site --out-dir artifacts/fixtures
pdm run fmp-docs build-all --catalog artifacts/fixtures/catalog.json --out-dir artifacts/build
pdm run pytest
```

## Live compile

```bash
pdm run fmp-docs compile-live --out-dir artifacts/live --cache-dir artifacts/cache --max-pages 20
```

## Verification

```bash
pdm run fmp-docs verify-catalog --catalog artifacts/live/catalog.json --out artifacts/live/catalog.verified.json --api-key YOUR_KEY
```

## Inspection and health checks

```bash
pdm run fmp-docs inspect-endpoint --catalog artifacts/fixtures/catalog.json --name get_latest_sec_filings
pdm run fmp-docs doctor --catalog artifacts/fixtures/catalog.json
```

## MCP integration

Render generated FastMCP files:

```bash
pdm run fmp-docs render-mcp --catalog artifacts/fixtures/catalog.json --out-dir artifacts/mcp
```

Build everything, including the combined REST + MCP wrapper:

```bash
pdm run fmp-docs build-all --catalog artifacts/fixtures/catalog.json --out-dir artifacts/build
```

Generated files now include:

- `generated_wrapper/mcp_auto.py`
- `generated_wrapper/mcp_manual.py`
- `generated_wrapper/combined_app.py`
