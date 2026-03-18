"""Tests for data provider registry."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from wraquant.core.types import DateLike
from wraquant.data.base import DataProvider, ProviderRegistry


class MockProvider(DataProvider):
    name = "mock"

    def fetch_prices(
        self,
        symbol: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
        **kwargs: Any,
    ) -> pd.Series:
        return pd.Series([100, 101, 102], name=symbol)


class TestProviderRegistry:
    def test_register_and_get(self) -> None:
        reg = ProviderRegistry()
        mock = MockProvider()
        reg.register(mock)
        assert reg.get("mock") is mock

    def test_default_provider(self) -> None:
        reg = ProviderRegistry()
        mock = MockProvider()
        reg.register(mock, default=True)
        assert reg.get() is mock
        assert reg.default == "mock"

    def test_first_registered_is_default(self) -> None:
        reg = ProviderRegistry()
        mock = MockProvider()
        reg.register(mock)
        assert reg.default == "mock"

    def test_unknown_provider_raises(self) -> None:
        reg = ProviderRegistry()
        with pytest.raises(KeyError, match="not found"):
            reg.get("nonexistent")

    def test_list_providers(self) -> None:
        reg = ProviderRegistry()
        reg.register(MockProvider())
        assert "mock" in reg.list_providers()

    def test_empty_registry_raises(self) -> None:
        reg = ProviderRegistry()
        with pytest.raises(KeyError, match="No providers"):
            reg.get()


class TestMockProvider:
    def test_fetch_prices(self) -> None:
        provider = MockProvider()
        prices = provider.fetch_prices("TEST")
        assert len(prices) == 3
        assert prices.name == "TEST"

    def test_ohlcv_not_implemented(self) -> None:
        provider = MockProvider()
        with pytest.raises(NotImplementedError):
            provider.fetch_ohlcv("TEST")

    def test_macro_not_implemented(self) -> None:
        provider = MockProvider()
        with pytest.raises(NotImplementedError):
            provider.fetch_macro("GDP")
