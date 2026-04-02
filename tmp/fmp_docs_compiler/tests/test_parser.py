from __future__ import annotations

from pathlib import Path

from fmp_docs_compiler.models import DiscoveredDocPage
from fmp_docs_compiler.parser import EndpointPageParser


def test_parse_table_parameters() -> None:
    parser = EndpointPageParser()
    page = DiscoveredDocPage(
        url="https://example.test/docs/sec",
        category="sec",
        label="SEC",
        source_index_url="https://example.test/docs",
    )
    html = Path("tests/fixtures/site/latest-sec-filings.html").read_text(
        encoding="utf-8"
    )
    parsed = parser.parse(page=page, html=html)
    names = {parameter.name for parameter in parsed.parsed_parameters}
    assert {"from", "to", "page", "limit"} <= names
    assert any(
        example.content_type == "application/json" for example in parsed.examples
    )


def test_parse_definition_list_and_list_parameters() -> None:
    parser = EndpointPageParser()
    page = DiscoveredDocPage(
        url="https://example.test/docs/search",
        category="search",
        label="Search",
        source_index_url="https://example.test/docs",
    )
    html = Path("tests/fixtures/site/symbol-search.html").read_text(encoding="utf-8")
    parsed = parser.parse(page=page, html=html)
    by_name = {parameter.name: parameter for parameter in parsed.parsed_parameters}
    assert by_name["query"].description == "Search text. Required."
    assert by_name["limit"].type_hint == "integer"
