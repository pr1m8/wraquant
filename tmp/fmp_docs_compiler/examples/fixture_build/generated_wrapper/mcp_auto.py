"""Generated FastMCP server derived from the FastAPI wrapper.

Purpose:
    Expose the generated REST wrapper as MCP tools using FastMCP's FastAPI
    integration.

Design:
    This is the quickest bridge from the generated FastAPI application to MCP.
    It is best for bootstrapping and prototyping. For more curated LLM-facing
    behavior, see :mod:`~generated_wrapper.mcp_manual`.

Attributes:
    mcp:
        FastMCP server instance generated from the FastAPI app.

Examples:
    ::
        >>> callable(run)
        True
"""

from __future__ import annotations

from fastmcp import FastMCP

from .app import app

mcp = FastMCP.from_fastapi(app=app, name="FMP Wrapper MCP")


def run() -> None:
    """Run the generated MCP server.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.

    Examples:
        ::
            >>> callable(run)
            True
    """
    mcp.run()


if __name__ == "__main__":
    run()
