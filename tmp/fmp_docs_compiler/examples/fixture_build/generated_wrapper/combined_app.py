"""Combined FastAPI and FastMCP ASGI application.

Purpose:
    Serve both the generated REST wrapper and the generated MCP interface from
    one FastAPI application.

Design:
    The combined app uses FastMCP's ``http_app`` integration and merges the MCP
    routes with the REST routes. The MCP interface is exposed under ``/mcp``.

Attributes:
    combined_app:
        FastAPI application containing both route sets.

Examples:
    ::
        >>> combined_app.title
        'FMP Wrapper with MCP'
"""

from __future__ import annotations

from fastapi import FastAPI

from .app import app
from .mcp_auto import mcp

mcp_app = mcp.http_app(path="/mcp")

combined_app = FastAPI(
    title="FMP Wrapper with MCP",
    routes=[*mcp_app.routes, *app.routes],
    lifespan=mcp_app.lifespan,
)
