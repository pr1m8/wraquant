"""Tests for wraquant.core.decorators."""

from __future__ import annotations

import pytest

from wraquant.core.decorators import cache_result, requires_extra
from wraquant.core.exceptions import MissingDependencyError


class TestRequiresExtra:
    def test_available_group_passes(self) -> None:
        @requires_extra("viz")
        def plot() -> str:
            return "plotted"

        # viz (matplotlib) should be installed in the dev environment
        try:
            result = plot()
            assert result == "plotted"
        except MissingDependencyError:
            pytest.skip("viz dependencies not installed")

    def test_missing_group_raises(self) -> None:
        @requires_extra("nonexistent-group-xyz")
        def do_thing() -> str:
            return "done"

        # nonexistent group should always fail
        with pytest.raises(MissingDependencyError):
            do_thing()


class TestCacheResult:
    def test_caches_result(self) -> None:
        call_count = 0

        @cache_result(ttl=60)
        def expensive(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        assert expensive(5) == 10
        assert expensive(5) == 10
        assert call_count == 1

    def test_different_args_miss_cache(self) -> None:
        call_count = 0

        @cache_result(ttl=60)
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x + 1

        assert compute(1) == 2
        assert compute(2) == 3
        assert call_count == 2

    def test_cache_clear(self) -> None:
        call_count = 0

        @cache_result(ttl=60)
        def func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x

        func(1)
        func(1)
        assert call_count == 1

        func.cache_clear()
        func(1)
        assert call_count == 2
