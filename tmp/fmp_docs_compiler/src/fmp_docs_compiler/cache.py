"""Caching utilities for :mod:`~fmp_docs_compiler`.

Purpose:
    Persist raw HTML snapshots and build artifacts for reproducible docs
    compilation runs.

Design:
    The cache is intentionally simple and file-based. It stores deterministic
    filenames derived from URLs so the package remains easy to inspect and
    portable across environments.

Attributes:
    None.

Examples:
    ::
        >>> from pathlib import Path
        >>> store = CacheStore(root=Path('tmp/cache'))
        >>> store.root.name
        'cache'
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .io_utils import write_text


class CacheStore:
    """Small file-based cache for HTML snapshots.

    Args:
        root:
            Cache root directory.

    Returns:
        A cache store instance.

    Raises:
        OSError:
            Raised when directories cannot be created.

    Examples:
        ::
            >>> from pathlib import Path
            >>> CacheStore(root=Path('tmp/cache')).root.name
            'cache'
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def key_for_url(self, url: str) -> str:
        """Return a deterministic cache key for a URL.

        Args:
            url:
                Input URL.

        Returns:
            A stable SHA-256-based key.

        Raises:
            None.

        Examples:
            ::
                >>> len(CacheStore(root=Path('tmp/cache')).key_for_url('https://example.test')) > 10
                True
        """
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def html_path_for_url(self, url: str) -> Path:
        """Return the cache path for HTML content.

        Args:
            url:
                Input URL.

        Returns:
            A filesystem path.

        Raises:
            None.

        Examples:
            ::
                >>> from pathlib import Path
                >>> CacheStore(root=Path('tmp/cache')).html_path_for_url('https://example.test').suffix
                '.html'
        """
        return self.root / "html" / f"{self.key_for_url(url)}.html"

    def write_html(self, url: str, html: str) -> Path:
        """Write HTML for a URL to cache.

        Args:
            url:
                Source URL.
            html:
                HTML content.

        Returns:
            The written cache path.

        Raises:
            OSError:
                Raised when writing fails.

        Examples:
            ::
                >>> callable(CacheStore.write_html)
                True
        """
        path = self.html_path_for_url(url)
        write_text(path, html)
        return path
