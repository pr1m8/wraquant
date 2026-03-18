"""Tests for data caching."""

from __future__ import annotations

from wraquant.data.cache import MemoryCache


class TestMemoryCache:
    def test_set_and_get(self) -> None:
        cache = MemoryCache(ttl=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_missing_key(self) -> None:
        cache = MemoryCache(ttl=60)
        assert cache.get("nonexistent") is None

    def test_clear(self) -> None:
        cache = MemoryCache(ttl=60)
        cache.set("k", "v")
        cache.clear()
        assert cache.get("k") is None

    def test_expired_entry(self) -> None:
        cache = MemoryCache(ttl=0)  # Immediately expires
        cache.set("k", "v")
        # With ttl=0 this should be considered expired
        # (time.monotonic() - ts >= 0 is always true)
        assert cache.get("k") is None
