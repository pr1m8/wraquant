"""Real-time data streaming utilities.

Provides a WebSocket client for consuming streaming market data and a
tick buffer for aggregating raw ticks into OHLCV bars.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Callable

import pandas as pd

__all__ = [
    "WebSocketClient",
    "TickBuffer",
]


class WebSocketClient:
    """Async WebSocket client for streaming market data.

    Wraps the ``websockets`` library to provide a simple interface for
    subscribing to real-time data feeds.

    Parameters:
        url: WebSocket server URL (e.g., ``"wss://stream.example.com"``).
        on_message: Optional callback invoked with each received message.
        on_error: Optional callback invoked when an error occurs.

    Example:
        >>> client = WebSocketClient("wss://stream.example.com/v1/ws")
        >>> client.on_message = lambda msg: print(msg)
        >>> client.run()  # blocks until disconnected
    """

    def __init__(
        self,
        url: str,
        on_message: Callable[[str], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self.url = url
        self.on_message = on_message
        self.on_error = on_error
        self._ws: Any = None
        self._running: bool = False
        self._subscriptions: set[str] = set()

    async def connect(self) -> None:
        """Open the WebSocket connection.

        Requires the ``websockets`` package (part of the ``ingestion``
        extra).
        """
        import websockets

        self._ws = await websockets.connect(self.url)
        self._running = True

    async def disconnect(self) -> None:
        """Close the WebSocket connection gracefully."""
        self._running = False
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def subscribe(self, channels: list[str]) -> None:
        """Subscribe to one or more data channels.

        Parameters:
            channels: List of channel identifiers to subscribe to.
        """
        import json

        self._subscriptions.update(channels)
        if self._ws is not None:
            message = json.dumps({"action": "subscribe", "channels": channels})
            await self._ws.send(message)

    async def unsubscribe(self, channels: list[str]) -> None:
        """Unsubscribe from one or more data channels.

        Parameters:
            channels: List of channel identifiers to unsubscribe from.
        """
        import json

        self._subscriptions.difference_update(channels)
        if self._ws is not None:
            message = json.dumps({"action": "unsubscribe", "channels": channels})
            await self._ws.send(message)

    async def _listen(self) -> None:
        """Internal listener that dispatches incoming messages."""
        try:
            async for message in self._ws:
                if not self._running:
                    break
                if self.on_message is not None:
                    self.on_message(message)
        except Exception as exc:
            if self.on_error is not None:
                self.on_error(exc)
            else:
                raise

    def run(self) -> None:
        """Start the WebSocket event loop (blocking).

        Connects to the server and listens for messages until the
        connection is closed or :meth:`disconnect` is called.
        """
        asyncio.get_event_loop().run_until_complete(self._run_async())

    async def _run_async(self) -> None:
        """Internal coroutine that manages the connection lifecycle."""
        await self.connect()
        try:
            await self._listen()
        finally:
            await self.disconnect()


class TickBuffer:
    """Buffer incoming ticks and aggregate them into OHLCV bars.

    Stores raw tick data and groups it into time-based bars at the
    requested interval.

    Parameters:
        bar_interval: Pandas-compatible frequency string for bar
            aggregation (e.g., ``'1min'``, ``'5min'``, ``'1h'``).

    Example:
        >>> buf = TickBuffer(bar_interval="1min")
        >>> buf.add_tick(pd.Timestamp("2024-01-02 09:30:00.100"), 150.25, 100)
        >>> buf.add_tick(pd.Timestamp("2024-01-02 09:30:00.500"), 150.50, 200)
        >>> bars = buf.get_bars()
    """

    def __init__(self, bar_interval: str = "1min") -> None:
        self.bar_interval = bar_interval
        self._timestamps: list[datetime | pd.Timestamp] = []
        self._prices: list[float] = []
        self._volumes: list[float] = []

    def add_tick(
        self,
        timestamp: datetime | pd.Timestamp,
        price: float,
        volume: float = 0,
    ) -> None:
        """Add a single tick to the buffer.

        Parameters:
            timestamp: Tick timestamp.
            price: Tick price.
            volume: Tick volume. Defaults to 0.
        """
        self._timestamps.append(timestamp)
        self._prices.append(price)
        self._volumes.append(volume)

    def get_bars(self) -> pd.DataFrame:
        """Aggregate buffered ticks into OHLCV bars.

        Returns:
            DataFrame with columns ``open``, ``high``, ``low``,
            ``close``, ``volume`` indexed by the bar period start time.
            Returns an empty DataFrame if no ticks have been added.
        """
        if not self._timestamps:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        ticks = pd.DataFrame(
            {
                "price": self._prices,
                "volume": self._volumes,
            },
            index=pd.DatetimeIndex(self._timestamps, name="timestamp"),
        )

        grouper = pd.Grouper(freq=self.bar_interval)
        bars = ticks.groupby(grouper).agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("volume", "sum"),
        )

        # Drop rows with no ticks (all NaN from empty groups)
        bars = bars.dropna(subset=["open"])

        return bars

    def flush(self) -> pd.DataFrame:
        """Return completed bars and clear the internal buffer.

        Returns:
            DataFrame with OHLCV bars from all buffered ticks.
        """
        bars = self.get_bars()
        self.clear()
        return bars

    def clear(self) -> None:
        """Clear all buffered ticks without returning bars."""
        self._timestamps.clear()
        self._prices.clear()
        self._volumes.clear()

    def __len__(self) -> int:
        """Return the number of buffered ticks."""
        return len(self._timestamps)
