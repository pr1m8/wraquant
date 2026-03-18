"""CSV file data provider for local data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from wraquant.core.types import DateLike
from wraquant.data.base import DataProvider
from wraquant.data.utils import parse_date


class CSVProvider(DataProvider):
    """Data provider that reads from local CSV files.

    Always available (no optional dependencies).

    Example:
        >>> provider = CSVProvider()
        >>> prices = provider.fetch_prices("/path/to/prices.csv")  # doctest: +SKIP
    """

    name = "csv"

    def fetch_prices(
        self,
        symbol: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
        *,
        date_column: str = "date",
        price_column: str = "close",
        **kwargs: Any,
    ) -> pd.Series:
        """Read prices from a CSV file.

        Parameters:
            symbol: Path to CSV file.
            start: Start date filter.
            end: End date filter.
            date_column: Name of date column.
            price_column: Name of price column.

        Returns:
            Price series with DatetimeIndex.
        """
        path = Path(symbol)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {symbol}")

        df = pd.read_csv(path, parse_dates=[date_column], **kwargs)
        df = df.set_index(date_column).sort_index()

        s = parse_date(start)
        e = parse_date(end)
        if s:
            df = df[df.index >= s]
        if e:
            df = df[df.index <= e]

        result = df[price_column]
        result.name = path.stem
        return result

    def fetch_ohlcv(
        self,
        symbol: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
        *,
        date_column: str = "date",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Read OHLCV data from a CSV file.

        Parameters:
            symbol: Path to CSV file.
            start: Start date filter.
            end: End date filter.
            date_column: Name of date column.

        Returns:
            DataFrame with open, high, low, close, volume columns.
        """
        path = Path(symbol)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {symbol}")

        df = pd.read_csv(path, parse_dates=[date_column], **kwargs)
        df = df.set_index(date_column).sort_index()
        df.columns = [c.lower() for c in df.columns]

        s = parse_date(start)
        e = parse_date(end)
        if s:
            df = df[df.index >= s]
        if e:
            df = df[df.index <= e]

        return df[["open", "high", "low", "close", "volume"]]
