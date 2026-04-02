"""Compiler orchestration for :mod:`~fmp_docs_compiler`.

Purpose:
    Coordinate discovery, parsing, normalization, optional caching, and
    manifest generation.

Design:
    The compiler separates live crawling from local fixture compilation so the
    same normalization and generation layers can be reused in both contexts.

Attributes:
    None.

Examples:
    ::
        >>> from fmp_docs_compiler.http import RetryConfig
        >>> isinstance(FMPDocsCompiler(RetryConfig()), FMPDocsCompiler)
        True
"""

from __future__ import annotations

from pathlib import Path

from .cache import CacheStore
from .discovery import DEFAULT_INDEX_URLS, FMPDocDiscoverer
from .http import ResilientAsyncClient, RetryConfig
from .models import CatalogIR, CrawlStats, ManifestIR, ParsedEndpointPage
from .normalizer import EndpointNormalizer
from .parser import EndpointPageParser
from .verify import CatalogVerifier


class FMPDocsCompiler:
    """High-level docs compiler."""

    def __init__(self, retry_config: RetryConfig) -> None:
        self.retry_config = retry_config
        self.parser = EndpointPageParser()
        self.normalizer = EndpointNormalizer()
        self.verifier = CatalogVerifier(retry_config)

    async def compile_live(
        self, max_pages: int | None = None, cache_dir: Path | None = None
    ) -> CatalogIR:
        manifest = ManifestIR(
            source="financialmodelingprep_live_docs",
            source_urls=list(DEFAULT_INDEX_URLS),
        )
        cache_store = CacheStore(cache_dir) if cache_dir is not None else None
        async with ResilientAsyncClient(retry_config=self.retry_config) as client:
            discoverer = FMPDocDiscoverer(client=client, index_urls=DEFAULT_INDEX_URLS)
            discovered_pages = await discoverer.discover()
            manifest.stats.discovered_pages = len(discovered_pages)
            if max_pages is not None:
                discovered_pages = discovered_pages[:max_pages]
            parsed_pages: list[ParsedEndpointPage] = []
            for index_url in DEFAULT_INDEX_URLS:
                if cache_store is not None:
                    cache_store.write_html(index_url, await client.get_text(index_url))
                    manifest.stats.cached_pages += 1
            for page in discovered_pages:
                html = await client.get_text(page.url)
                if cache_store is not None:
                    cache_store.write_html(page.url, html)
                    manifest.stats.cached_pages += 1
                parsed_pages.append(self.parser.parse(page=page, html=html))

        manifest.stats = CrawlStats(
            discovered_pages=manifest.stats.discovered_pages,
            parsed_pages=len(parsed_pages),
            cached_pages=manifest.stats.cached_pages,
            premium_flagged_pages=sum(
                1 for page in parsed_pages if page.premium_signals
            ),
            warning_count=sum(len(page.parse_warnings) for page in parsed_pages),
        )
        return self.normalizer.normalize_many(
            parsed_pages,
            "financialmodelingprep_live_docs",
            list(DEFAULT_INDEX_URLS),
            manifest,
        )

    def compile_fixtures(
        self, fixtures_dir: Path, index_name: str = "index.html"
    ) -> CatalogIR:
        index_path = fixtures_dir / index_name
        source_index_url = f"file://{index_path.resolve()}"
        manifest = ManifestIR(source="fixture_docs", source_urls=[source_index_url])
        discoverer = FMPDocDiscoverer(client=None, index_urls=(source_index_url,))
        discovered_pages = discoverer.discover_from_html(
            index_path.read_text(encoding="utf-8"), source_index_url
        )
        manifest.stats.discovered_pages = len(discovered_pages)
        parsed_pages: list[ParsedEndpointPage] = []
        for page in discovered_pages:
            parsed_pages.append(
                self.parser.parse(
                    page=page, html=Path(page.url).read_text(encoding="utf-8")
                )
            )
        manifest.stats = CrawlStats(
            discovered_pages=len(discovered_pages),
            parsed_pages=len(parsed_pages),
            cached_pages=0,
            premium_flagged_pages=sum(
                1 for page in parsed_pages if page.premium_signals
            ),
            warning_count=sum(len(page.parse_warnings) for page in parsed_pages),
        )
        return self.normalizer.normalize_many(
            parsed_pages, "fixture_docs", [source_index_url], manifest
        )

    async def verify_catalog(
        self, catalog: CatalogIR, api_key: str | None
    ) -> CatalogIR:
        return await self.verifier.verify_catalog(catalog=catalog, api_key=api_key)
