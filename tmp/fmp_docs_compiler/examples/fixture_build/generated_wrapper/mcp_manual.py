"""Generated manual FastMCP server.

Purpose:
    Expose a more curated MCP surface than the automatic FastAPI conversion.

Design:
    Each endpoint is emitted as an explicit MCP tool with parameter names,
    descriptions, and tags derived from the docs catalog. A catalog resource is
    also exposed so clients can inspect the available operations.

Attributes:
    mcp:
        Manual FastMCP server instance.

Examples:
    ::
        >>> callable(run)
        True
"""

from __future__ import annotations

from fastmcp import FastMCP

from .settings import get_settings
from .upstream import get_upstream_client

mcp = FastMCP("FMP Manual MCP")


@mcp.resource("fmp://catalog")
def catalog_resource() -> list[dict[str, object]]:
    """Return the generated endpoint catalog."""
    return [
        {
            "operation_name": "get_available_countries",
            "title": "Available Countries API",
            "summary": "Access country-based data for supported companies and markets.",
            "wrapper_path": "/fmp/reference-data/get-available-countries",
            "category": "reference-data",
        },
        {
            "operation_name": "get_available_sectors",
            "title": "Available Sectors API",
            "summary": "Access a complete list of industry sectors.",
            "wrapper_path": "/fmp/reference-data/get-available-sectors",
            "category": "reference-data",
        },
        {
            "operation_name": "get_latest_sec_filings",
            "title": "Latest SEC Filings API",
            "summary": "Stay updated with the most recent SEC filings from publicly traded companies.",
            "wrapper_path": "/fmp/sec/get-latest-sec-filings",
            "category": "sec",
        },
        {
            "operation_name": "get_symbol_search",
            "title": "Symbol Search API",
            "summary": "Search symbols and company names.",
            "wrapper_path": "/fmp/search/get-symbol-search",
            "category": "search",
        },
    ]


@mcp.tool(name="list_available_endpoints", tags={"catalog", "discovery"})
def list_available_endpoints() -> list[dict[str, object]]:
    """List operation names, wrapper paths, and summaries for available tools."""
    return [
        {
            "operation_name": "get_available_countries",
            "title": "Available Countries API",
            "summary": "Access country-based data for supported companies and markets.",
            "wrapper_path": "/fmp/reference-data/get-available-countries",
            "category": "reference-data",
        },
        {
            "operation_name": "get_available_sectors",
            "title": "Available Sectors API",
            "summary": "Access a complete list of industry sectors.",
            "wrapper_path": "/fmp/reference-data/get-available-sectors",
            "category": "reference-data",
        },
        {
            "operation_name": "get_latest_sec_filings",
            "title": "Latest SEC Filings API",
            "summary": "Stay updated with the most recent SEC filings from publicly traded companies.",
            "wrapper_path": "/fmp/sec/get-latest-sec-filings",
            "category": "sec",
        },
        {
            "operation_name": "get_symbol_search",
            "title": "Symbol Search API",
            "summary": "Search symbols and company names.",
            "wrapper_path": "/fmp/search/get-symbol-search",
            "category": "search",
        },
    ]


@mcp.tool(
    name="get_available_countries",
    description="The API allows filtering companies by country and supports localized research workflows.",
    tags={"fmp", "generated", "reference-data"},
)
async def get_available_countries() -> dict[str, object]:
    """Access country-based data for supported companies and markets."""
    client = get_upstream_client(get_settings())
    query = {
        # No query parameters for this endpoint.
    }
    return await client.call(path="/stable/available-countries", query=query)


@mcp.tool(
    name="get_available_sectors",
    description="This endpoint returns supported sectors. Endpoint: https://financialmodelingprep.com/stable/available-sectors",
    tags={"fmp", "generated", "reference-data"},
)
async def get_available_sectors() -> dict[str, object]:
    """Access a complete list of industry sectors."""
    client = get_upstream_client(get_settings())
    query = {
        # No query parameters for this endpoint.
    }
    return await client.call(path="/stable/available-sectors", query=query)


@mcp.tool(
    name="get_latest_sec_filings",
    description="Retrieve recent 8-K, 10-K, and 10-Q filings with date filters and pagination.",
    tags={"fmp", "generated", "sec"},
)
async def get_latest_sec_filings(
    from_: str | None = None,
    to: str | None = None,
    page: int | None = None,
    limit: int | None = None,
) -> dict[str, object]:
    """Stay updated with the most recent SEC filings from publicly traded companies."""
    client = get_upstream_client(get_settings())
    query = {
        "from": from_,
        "to": to,
        "page": page,
        "limit": limit,
    }
    return await client.call(path="/stable/sec-filings-financials", query=query)


@mcp.tool(
    name="get_symbol_search",
    description="Find companies by partial ticker or company name. Endpoint: https://financialmodelingprep.com/stable/search-symbol?query=AAPL&exchange=NASDAQ&limit=10",
    tags={"fmp", "generated", "search"},
)
async def get_symbol_search(
    query: str, exchange: str | None = None, limit: int | None = None
) -> dict[str, object]:
    """Search symbols and company names."""
    client = get_upstream_client(get_settings())
    query = {
        "query": query,
        "exchange": exchange,
        "limit": limit,
    }
    return await client.call(path="/stable/search-symbol", query=query)


def run() -> None:
    """Run the manual MCP server.

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
