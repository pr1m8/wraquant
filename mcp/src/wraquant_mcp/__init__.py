"""wraquant-mcp — Quant analysis engine for AI agents.

MCP server exposing wraquant's 1,097 quantitative finance functions
as tools for Claude, LangChain, and other MCP-compatible AI agents.

Compose with:
- OpenBB MCP for data fetching
- DuckDB MCP for SQL queries on shared state
- Alpaca MCP for trade execution
- Jupyter MCP for notebook interaction

Quick start:
    $ wraquant-mcp                          # stdio (Claude Desktop)
    $ wraquant-mcp --transport http          # HTTP (hosted/LangChain)
    $ python -m wraquant_mcp                # module entry

Configuration:
    ~/.wraquant/config.json                 # Global settings
    ~/.wraquant/workspaces/                 # Persistent workspaces
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["create_server", "main"]


def create_server(name: str = "wraquant"):
    """Create and configure the wraquant MCP server."""
    from wraquant_mcp.server import build_server

    return build_server(name)


def main():
    """CLI entry point."""
    from wraquant_mcp.server import run

    run()
