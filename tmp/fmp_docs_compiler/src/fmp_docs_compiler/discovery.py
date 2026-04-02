"""Discovery for :mod:`~fmp_docs_compiler`.

Purpose:
    Discover endpoint documentation pages from FMP index pages or local
    fixture HTML.

Design:
    Discovery intentionally favors resilience over strict selectors. It scans
    headings and links together so endpoints inherit the most recent visible
    category heading.

Attributes:
    DEFAULT_INDEX_URLS:
        Default FMP docs entry points.

Examples:
    ::
        >>> isinstance(DEFAULT_INDEX_URLS, tuple)
        True
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag

from .models import DiscoveredDocPage

DOC_HOST = "https://site.financialmodelingprep.com"
DEFAULT_INDEX_URLS: tuple[str, ...] = (
    f"{DOC_HOST}/developer/docs",
    f"{DOC_HOST}/developer/docs/legacy-endpoints",
)
_EXCLUDED_PATHS = {
    "/developer/docs",
    "/developer/docs/pricing",
    "/developer/docs/quickstart",
    "/developer/docs/legacy-endpoints",
    "/developer/docs/bulk-endpoints",
}


class FMPDocDiscoverer:
    """Discover endpoint docs pages from HTML.

    Args:
        client:
            Optional resilient async client for live fetches.
        index_urls:
            Index URLs to crawl.

    Returns:
        A discoverer instance.

    Raises:
        None.

    Examples:
        ::
            >>> discoverer = FMPDocDiscoverer(client=None, index_urls=DEFAULT_INDEX_URLS)
            >>> len(discoverer.index_urls) >= 1
            True
    """

    def __init__(self, client, index_urls: tuple[str, ...]) -> None:
        self.client = client
        self.index_urls = index_urls

    async def discover(self) -> list[DiscoveredDocPage]:
        """Discover pages from live index URLs.

        Args:
            None.

        Returns:
            A deduplicated list of discovered doc pages.

        Raises:
            RuntimeError:
                Raised when no client is configured.

        Examples:
            ::
                >>> callable(FMPDocDiscoverer.discover)
                True
        """
        if self.client is None:
            raise RuntimeError("A client is required for live discovery.")

        pages: list[DiscoveredDocPage] = []
        for index_url in self.index_urls:
            html = await self.client.get_text(index_url)
            pages.extend(
                self.discover_from_html(index_html=html, source_index_url=index_url)
            )

        deduped: dict[str, DiscoveredDocPage] = {}
        for page in pages:
            deduped.setdefault(page.url, page)
        return list(deduped.values())

    def discover_from_html(
        self, index_html: str, source_index_url: str
    ) -> list[DiscoveredDocPage]:
        """Discover pages from supplied index HTML.

        Args:
            index_html:
                Raw index HTML.
            source_index_url:
                Source index URL or fixture URL.

        Returns:
            A list of discovered doc pages.

        Raises:
            None.

        Examples:
            ::
                >>> html = '<h2>Reference</h2><a href="a.html">A</a>'
                >>> pages = FMPDocDiscoverer(client=None, index_urls=DEFAULT_INDEX_URLS).discover_from_html(html, 'file:///tmp/index.html')
                >>> pages[0].category
                'reference'
        """
        soup = BeautifulSoup(index_html, "lxml")
        current_category = "uncategorized"
        pages: list[DiscoveredDocPage] = []

        body = soup.body or soup
        for node in body.descendants:
            if not isinstance(node, Tag):
                continue
            if node.name in {"h1", "h2", "h3"}:
                text = " ".join(node.get_text(" ", strip=True).split()).strip()
                if text:
                    current_category = text.lower()
            if node.name != "a":
                continue
            href = str(node.get("href", "")).strip()
            label = " ".join(node.get_text(" ", strip=True).split()).strip()
            if not href or not label:
                continue
            target_url = self._resolve_url(source_index_url=source_index_url, href=href)
            if self._skip_url(target_url):
                continue
            pages.append(
                DiscoveredDocPage(
                    url=target_url,
                    category=current_category.replace(" ", "-"),
                    label=label,
                    source_index_url=source_index_url,
                    legacy="legacy" in source_index_url,
                )
            )
        return pages

    def _resolve_url(self, source_index_url: str, href: str) -> str:
        if source_index_url.startswith("file://"):
            base_path = Path(urlparse(source_index_url).path).parent
            return str((base_path / href).resolve())
        return urljoin(source_index_url, href)

    def _skip_url(self, url: str) -> bool:
        if url.startswith("http"):
            parsed = urlparse(url)
            return parsed.path in _EXCLUDED_PATHS
        return False
