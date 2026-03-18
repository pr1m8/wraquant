"""FRED (Federal Reserve Economic Data) provider."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

from wraquant.core.decorators import requires_extra
from wraquant.core.types import DateLike
from wraquant.data.base import DataProvider
from wraquant.data.utils import parse_date


class FREDProvider(DataProvider):
    """Data provider for FRED macroeconomic data.

    Requires a FRED API key set via the ``FRED_API_KEY`` environment variable
    or passed directly.

    Example:
        >>> provider = FREDProvider(api_key="your_key")  # doctest: +SKIP
        >>> gdp = provider.fetch_macro("GDP")  # doctest: +SKIP
    """

    name = "fred"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("FRED_API_KEY")

    def _get_fred(self) -> Any:
        import fredapi

        if not self._api_key:
            raise ValueError(
                "FRED API key required. Set FRED_API_KEY environment variable "
                "or pass api_key to FREDProvider."
            )
        return fredapi.Fred(api_key=self._api_key)

    @requires_extra("market-data")
    def fetch_prices(
        self,
        symbol: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
        **kwargs: Any,
    ) -> pd.Series:
        """Fetch a FRED series as prices.

        Parameters:
            symbol: FRED series ID (e.g., 'DFF', 'DEXUSEU').
            start: Start date.
            end: End date.

        Returns:
            Data series with DatetimeIndex.
        """
        return self.fetch_macro(symbol, start=start, end=end, **kwargs)

    @requires_extra("market-data")
    def fetch_macro(
        self,
        series_id: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
        **kwargs: Any,
    ) -> pd.Series:
        """Fetch macroeconomic data from FRED.

        Parameters:
            series_id: FRED series ID (e.g., 'GDP', 'UNRATE', 'DFF').
            start: Start date.
            end: End date.

        Returns:
            Macro data series with DatetimeIndex.
        """
        fred = self._get_fred()
        s = parse_date(start)
        e = parse_date(end)
        data = fred.get_series(series_id, observation_start=s, observation_end=e)
        data.name = series_id
        return data
