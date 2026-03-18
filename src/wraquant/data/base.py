"""Abstract base classes and provider registry for data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from wraquant.core.types import DateLike


class DataProvider(ABC):
    """Abstract base class for all data providers.

    Subclasses must implement ``fetch_prices`` and declare their ``name``.
    """

    name: str = ""

    @abstractmethod
    def fetch_prices(
        self,
        symbol: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
        **kwargs: Any,
    ) -> pd.Series:
        """Fetch closing prices for a symbol.

        Parameters:
            symbol: Ticker or identifier.
            start: Start date.
            end: End date.

        Returns:
            Price series with DatetimeIndex.
        """

    def fetch_ohlcv(
        self,
        symbol: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol.

        Parameters:
            symbol: Ticker or identifier.
            start: Start date.
            end: End date.

        Returns:
            DataFrame with open, high, low, close, volume columns.
        """
        raise NotImplementedError(
            f"Provider '{self.name}' does not support OHLCV data."
        )

    def fetch_macro(
        self,
        series_id: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
        **kwargs: Any,
    ) -> pd.Series:
        """Fetch macroeconomic data series.

        Parameters:
            series_id: Series identifier (e.g., 'GDP', 'UNRATE').
            start: Start date.
            end: End date.

        Returns:
            Macro data series with DatetimeIndex.
        """
        raise NotImplementedError(
            f"Provider '{self.name}' does not support macro data."
        )


class ProviderRegistry:
    """Registry for data providers.

    Allows registering, retrieving, and listing data providers by name.
    """

    def __init__(self) -> None:
        self._providers: dict[str, DataProvider] = {}
        self._default: str | None = None

    def register(self, provider: DataProvider, *, default: bool = False) -> None:
        """Register a data provider.

        Parameters:
            provider: DataProvider instance to register.
            default: If True, make this the default provider.
        """
        self._providers[provider.name] = provider
        if default or self._default is None:
            self._default = provider.name

    def get(self, name: str | None = None) -> DataProvider:
        """Get a provider by name, or the default.

        Parameters:
            name: Provider name. None returns the default.

        Returns:
            The requested DataProvider.

        Raises:
            KeyError: If the provider is not registered.
        """
        if name is None:
            if self._default is None:
                raise KeyError("No providers registered.")
            name = self._default
        if name not in self._providers:
            available = list(self._providers.keys())
            raise KeyError(f"Provider '{name}' not found. Available: {available}")
        return self._providers[name]

    def list_providers(self) -> list[str]:
        """List all registered provider names.

        Returns:
            List of provider name strings.
        """
        return list(self._providers.keys())

    @property
    def default(self) -> str | None:
        """Name of the default provider."""
        return self._default


# Global registry singleton
registry = ProviderRegistry()
