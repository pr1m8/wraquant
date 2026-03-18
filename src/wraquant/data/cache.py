"""Caching infrastructure for data fetches.

Provides both in-memory TTL cache and optional disk caching via diskcache.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import pandas as pd

from wraquant.core.config import get_config
from wraquant.core.logging import get_logger

log = get_logger(__name__)


class MemoryCache:
    """Simple in-memory cache with TTL."""

    def __init__(self, ttl: int | None = None) -> None:
        self._store: dict[str, tuple[float, Any]] = {}
        self._ttl = ttl

    @property
    def ttl(self) -> int:
        if self._ttl is not None:
            return self._ttl
        return get_config().cache_ttl_seconds

    def _make_key(self, *args: Any, **kwargs: Any) -> str:
        raw = f"{args}:{sorted(kwargs.items())}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        """Get a cached value if it exists and hasn't expired."""
        if key in self._store:
            ts, value = self._store[key]
            if time.monotonic() - ts < self.ttl:
                return value
            del self._store[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Store a value in the cache."""
        self._store[key] = (time.monotonic(), value)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._store.clear()


class DiskCache:
    """Disk-based cache for persisting fetched data across sessions.

    Falls back to no-op if diskcache is not installed.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir
        self._disk_cache: Any = None
        self._initialized = False

    def _init(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        if not get_config().cache_enabled:
            return

        try:
            import diskcache

            cache_dir = self._cache_dir or get_config().cache_dir
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._disk_cache = diskcache.Cache(str(cache_dir))
        except ImportError:
            log.debug("diskcache not installed, disk caching disabled")

    def get(self, key: str) -> pd.Series | pd.DataFrame | None:
        """Retrieve cached data from disk."""
        self._init()
        if self._disk_cache is None:
            return None
        return self._disk_cache.get(key)

    def set(
        self, key: str, value: pd.Series | pd.DataFrame, ttl: int | None = None
    ) -> None:
        """Store data to disk cache."""
        self._init()
        if self._disk_cache is None:
            return
        expire = ttl or get_config().cache_ttl_seconds
        self._disk_cache.set(key, value, expire=expire)

    def clear(self) -> None:
        """Clear the disk cache."""
        self._init()
        if self._disk_cache is not None:
            self._disk_cache.clear()


# Shared instances
memory_cache = MemoryCache()
disk_cache = DiskCache()
