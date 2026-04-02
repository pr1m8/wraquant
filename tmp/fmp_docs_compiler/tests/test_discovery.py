from __future__ import annotations

from pathlib import Path

from fmp_docs_compiler.discovery import DEFAULT_INDEX_URLS, FMPDocDiscoverer


def test_discover_from_fixture_html() -> None:
    discoverer = FMPDocDiscoverer(client=None, index_urls=DEFAULT_INDEX_URLS)
    html = Path("tests/fixtures/site/index.html").read_text(encoding="utf-8")
    pages = discoverer.discover_from_html(
        html, f'file://{(Path("tests/fixtures/site/index.html")).resolve()}'
    )
    assert len(pages) == 4
    assert {page.category for page in pages} >= {"reference-data", "sec", "search"}
