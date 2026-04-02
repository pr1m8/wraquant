"""FastMCP scaffold generation for :mod:`~fmp_docs_compiler`.

Purpose:
    Generate FastMCP integration artifacts from the normalized catalog and the
    generated FastAPI wrapper.

Design:
    The generator emits two integration styles:

    - an automatic MCP server derived from the FastAPI wrapper
    - a manual MCP server with curated tools and a catalog resource

    A combined ASGI application is also generated so the REST wrapper and MCP
    interface can be served from a single FastAPI process.

Attributes:
    None.

Examples:
    ::
        >>> from fmp_docs_compiler.models import CatalogIR, ManifestIR
        >>> files = render_mcp_project(CatalogIR(source="x", source_urls=["y"], endpoints=[], manifest=ManifestIR(source="x", source_urls=["y"])))
        >>> 'generated_wrapper/mcp_auto.py' in files
        True
"""

from __future__ import annotations

from .generator_fastapi import _py_type, pythonize_identifier
from .models import CatalogIR, EndpointIR, ParameterIR


def render_mcp_project(catalog: CatalogIR) -> dict[str, str]:
    """Render FastMCP integration files.

    Args:
        catalog:
            Normalized catalog.

    Returns:
        A mapping of relative file path to file content.

    Raises:
        None.

    Examples:
        ::
            >>> from fmp_docs_compiler.models import CatalogIR, ManifestIR
            >>> files = render_mcp_project(CatalogIR(source="x", source_urls=["y"], endpoints=[], manifest=ManifestIR(source="x", source_urls=["y"])))
            >>> 'generated_wrapper/combined_app.py' in files
            True
    """
    manual_tools = "\n\n".join(
        _render_manual_tool(endpoint) for endpoint in catalog.endpoints
    )
    catalog_entries = "\n".join(
        _render_catalog_entry(endpoint) for endpoint in catalog.endpoints
    )

    mcp_auto = '''"""Generated FastMCP server derived from the FastAPI wrapper.

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

mcp = FastMCP.from_fastapi(app=app, name='FMP Wrapper MCP')


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


if __name__ == '__main__':
    run()
'''

    mcp_manual = f'''"""Generated manual FastMCP server.

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

mcp = FastMCP('FMP Manual MCP')


@mcp.resource('fmp://catalog')
def catalog_resource() -> list[dict[str, object]]:
    """Return the generated endpoint catalog."""
    return [
{catalog_entries}
    ]


@mcp.tool(name='list_available_endpoints', tags={{'catalog', 'discovery'}})
def list_available_endpoints() -> list[dict[str, object]]:
    """List operation names, wrapper paths, and summaries for available tools."""
    return [
{catalog_entries}
    ]


{manual_tools}


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


if __name__ == '__main__':
    run()
'''

    combined_app = '''"""Combined FastAPI and FastMCP ASGI application.

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

mcp_app = mcp.http_app(path='/mcp')

combined_app = FastAPI(
    title='FMP Wrapper with MCP',
    routes=[*mcp_app.routes, *app.routes],
    lifespan=mcp_app.lifespan,
)
'''

    return {
        "generated_wrapper/mcp_auto.py": mcp_auto,
        "generated_wrapper/mcp_manual.py": mcp_manual,
        "generated_wrapper/combined_app.py": combined_app,
    }


def _render_catalog_entry(endpoint: EndpointIR) -> str:
    return (
        "        {\n"
        f'            "operation_name": {endpoint.operation_name!r},\n'
        f'            "title": {endpoint.title!r},\n'
        f'            "summary": {endpoint.summary!r},\n'
        f'            "wrapper_path": {endpoint.wrapper_path!r},\n'
        f'            "category": {endpoint.category!r},\n'
        "        },"
    )


def _render_manual_param(parameter: ParameterIR) -> tuple[str, str]:
    py_name = pythonize_identifier(parameter.name)
    annotation = _py_type(parameter.type_hint)
    if parameter.required:
        return py_name, f"{py_name}: {annotation}"
    return py_name, f"{py_name}: {annotation} | None = None"


def _render_manual_tool(endpoint: EndpointIR) -> str:
    arguments: list[str] = []
    query_lines: list[str] = []
    for parameter in endpoint.parameters:
        py_name, argument = _render_manual_param(parameter)
        arguments.append(argument)
        query_lines.append(f"        {parameter.name!r}: {py_name},")

    signature = ", ".join(arguments)
    query_block = (
        "\n".join(query_lines)
        if query_lines
        else "        # No query parameters for this endpoint."
    )
    description = endpoint.about or endpoint.summary
    tool_name = endpoint.operation_name
    tags = (
        "{"
        + ", ".join(
            repr(tag) for tag in sorted({endpoint.category, "fmp", "generated"})
        )
        + "}"
    )
    return f'''@mcp.tool(name={tool_name!r}, description={description!r}, tags={tags})
async def {tool_name}({signature}) -> dict[str, object]:
    """{endpoint.summary}"""
    client = get_upstream_client(get_settings())
    query = {{
{query_block}
    }}
    return await client.call(path={endpoint.upstream_path!r}, query=query)'''
