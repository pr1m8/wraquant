from __future__ import annotations

from pathlib import Path

from fmp_docs_compiler.compiler import FMPDocsCompiler
from fmp_docs_compiler.generator_fastapi import render_fastapi_project
from fmp_docs_compiler.generator_mcp import render_mcp_project
from fmp_docs_compiler.generator_openapi import build_openapi_document
from fmp_docs_compiler.generator_tools import build_tool_schemas
from fmp_docs_compiler.http import RetryConfig


def test_fixture_pipeline_generates_artifacts() -> None:
    compiler = FMPDocsCompiler(retry_config=RetryConfig())
    catalog = compiler.compile_fixtures(fixtures_dir=Path("tests/fixtures/site"))
    assert catalog.endpoint_count == 4
    assert catalog.manifest.stats.discovered_pages == 4
    assert any(endpoint.premium_signals for endpoint in catalog.endpoints)

    openapi = build_openapi_document(catalog)
    tools = build_tool_schemas(catalog)
    project = render_fastapi_project(catalog)
    mcp_project = render_mcp_project(catalog)

    assert "/fmp/sec/get-latest-sec-filings" in openapi["paths"]
    assert any(tool["function"]["name"] == "get_symbol_search" for tool in tools)
    assert "generated_wrapper/upstream.py" in project
    assert "generated_wrapper/mcp_auto.py" in mcp_project
    assert "FastMCP.from_fastapi" in mcp_project["generated_wrapper/mcp_auto.py"]
    assert "@mcp.tool" in mcp_project["generated_wrapper/mcp_manual.py"]
