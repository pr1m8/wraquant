"""Yahoo Finance data provider via yfinance."""

from __future__ import annotations

from typing import Any

import pandas as pd

from wraquant.core.decorators import requires_extra
from wraquant.core.types import DateLike
from wraquant.data.base import DataProvider
from wraquant.data.utils import clean_symbol, parse_date


class YahooProvider(DataProvider):
    """Data provider using Yahoo Finance (yfinance).

    Supports equities, ETFs, forex pairs (e.g., 'EURUSD=X'),
    crypto (e.g., 'BTC-USD'), and indices.
    """

    name = "yahoo"

    @requires_extra("market-data")
    def fetch_prices(
        self,
        symbol: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
        **kwargs: Any,
    ) -> pd.Series:
        """Fetch adjusted close prices from Yahoo Finance.

        Parameters:
            symbol: Ticker symbol (e.g., 'AAPL', 'EURUSD=X').
            start: Start date.
            end: End date.

        Returns:
            Adjusted close price series.
        """
        import yfinance as yf

        ticker = clean_symbol(symbol)
        s = str(parse_date(start)) if start else None
        e = str(parse_date(end)) if end else None

        data = yf.download(
            ticker,
            start=s,
            end=e,
            progress=False,
            auto_adjust=True,
            **kwargs,
        )

        if data.empty:
            return pd.Series(dtype=float, name=ticker)

        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.name = ticker
        return close

    @requires_extra("market-data")
    def fetch_ohlcv(
        self,
        symbol: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance.

        Parameters:
            symbol: Ticker symbol.
            start: Start date.
            end: End date.

        Returns:
            DataFrame with open, high, low, close, volume columns.
        """
        import yfinance as yf

        ticker = clean_symbol(symbol)
        s = str(parse_date(start)) if start else None
        e = str(parse_date(end)) if end else None

        data = yf.download(
            ticker,
            start=s,
            end=e,
            progress=False,
            auto_adjust=True,
            **kwargs,
        )

        if data.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Flatten multi-level columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.columns = [c.lower() for c in data.columns]
        return data[["open", "high", "low", "close", "volume"]]
