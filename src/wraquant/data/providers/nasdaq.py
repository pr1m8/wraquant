"""NASDAQ Data Link (formerly Quandl) provider."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

from wraquant.core.decorators import requires_extra
from wraquant.core.types import DateLike
from wraquant.data.base import DataProvider
from wraquant.data.utils import parse_date


class NasdaqProvider(DataProvider):
    """Data provider for NASDAQ Data Link datasets.

    Requires API key via ``NASDAQ_DATA_LINK_API_KEY`` environment variable.
    """

    name = "nasdaq"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("NASDAQ_DATA_LINK_API_KEY")

    @requires_extra("market-data")
    def fetch_prices(
        self,
        symbol: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
        **kwargs: Any,
    ) -> pd.Series:
        """Fetch data from NASDAQ Data Link.

        Parameters:
            symbol: Dataset code (e.g., 'WIKI/AAPL').
            start: Start date.
            end: End date.

        Returns:
            Closing price series.
        """
        import nasdaqdatalink

        if self._api_key:
            nasdaqdatalink.ApiConfig.api_key = self._api_key

        s = str(parse_date(start)) if start else None
        e = str(parse_date(end)) if end else None

        data = nasdaqdatalink.get(
            symbol,
            start_date=s,
            end_date=e,
            **kwargs,
        )

        if isinstance(data, pd.DataFrame):
            # Try to find a close/value column
            for col in ["Close", "close", "Value", "value", "Settle", "settle"]:
                if col in data.columns:
                    result = data[col]
                    result.name = symbol
                    return result
            # Fall back to last column
            result = data.iloc[:, -1]
            result.name = symbol
            return result
        return data
