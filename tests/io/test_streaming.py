"""Tests for wraquant.io.streaming — TickBuffer aggregation logic."""

from __future__ import annotations

import pandas as pd
import pytest

from wraquant.io.streaming import TickBuffer


@pytest.fixture
def populated_buffer() -> TickBuffer:
    """Create a TickBuffer with several ticks across two 1-minute bars."""
    buf = TickBuffer(bar_interval="1min")

    # Bar 1: 09:30:00 - 09:30:59
    buf.add_tick(pd.Timestamp("2024-01-02 09:30:00.100"), 150.00, 100)
    buf.add_tick(pd.Timestamp("2024-01-02 09:30:15.000"), 150.50, 200)
    buf.add_tick(pd.Timestamp("2024-01-02 09:30:30.000"), 149.75, 150)
    buf.add_tick(pd.Timestamp("2024-01-02 09:30:45.000"), 150.25, 300)

    # Bar 2: 09:31:00 - 09:31:59
    buf.add_tick(pd.Timestamp("2024-01-02 09:31:00.000"), 150.30, 250)
    buf.add_tick(pd.Timestamp("2024-01-02 09:31:20.000"), 151.00, 400)
    buf.add_tick(pd.Timestamp("2024-01-02 09:31:40.000"), 150.80, 350)

    return buf


class TestTickBufferAddTick:
    """Tests for TickBuffer.add_tick."""

    def test_add_single_tick(self) -> None:
        """Adding one tick should increase buffer length to 1."""
        buf = TickBuffer()
        buf.add_tick(pd.Timestamp("2024-01-02 09:30:00"), 100.0, 50)
        assert len(buf) == 1

    def test_add_multiple_ticks(self) -> None:
        """Adding multiple ticks should accumulate."""
        buf = TickBuffer()
        for i in range(10):
            buf.add_tick(
                pd.Timestamp("2024-01-02 09:30:00") + pd.Timedelta(seconds=i),
                100.0 + i,
                10,
            )
        assert len(buf) == 10

    def test_default_volume_zero(self) -> None:
        """Volume should default to 0 when not provided."""
        buf = TickBuffer()
        buf.add_tick(pd.Timestamp("2024-01-02 09:30:00"), 100.0)
        bars = buf.get_bars()
        assert bars["volume"].iloc[0] == 0


class TestTickBufferGetBars:
    """Tests for TickBuffer.get_bars."""

    def test_empty_buffer_returns_empty_df(self) -> None:
        """get_bars on an empty buffer returns an empty DataFrame with correct columns."""
        buf = TickBuffer()
        bars = buf.get_bars()
        assert isinstance(bars, pd.DataFrame)
        assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
        assert len(bars) == 0

    def test_produces_correct_number_of_bars(
        self, populated_buffer: TickBuffer
    ) -> None:
        """Two minutes of data should yield two 1-minute bars."""
        bars = populated_buffer.get_bars()
        assert len(bars) == 2

    def test_ohlcv_values_bar1(self, populated_buffer: TickBuffer) -> None:
        """Verify OHLCV aggregation for the first bar."""
        bars = populated_buffer.get_bars()
        bar1 = bars.iloc[0]

        assert bar1["open"] == 150.00
        assert bar1["high"] == 150.50
        assert bar1["low"] == 149.75
        assert bar1["close"] == 150.25
        assert bar1["volume"] == 750  # 100 + 200 + 150 + 300

    def test_ohlcv_values_bar2(self, populated_buffer: TickBuffer) -> None:
        """Verify OHLCV aggregation for the second bar."""
        bars = populated_buffer.get_bars()
        bar2 = bars.iloc[1]

        assert bar2["open"] == 150.30
        assert bar2["high"] == 151.00
        assert bar2["low"] == 150.30
        assert bar2["close"] == 150.80
        assert bar2["volume"] == 1000  # 250 + 400 + 350

    def test_get_bars_does_not_clear(self, populated_buffer: TickBuffer) -> None:
        """get_bars should not clear the internal buffer."""
        populated_buffer.get_bars()
        assert len(populated_buffer) == 7

    def test_single_tick_bar(self) -> None:
        """A single tick should produce a bar where O=H=L=C."""
        buf = TickBuffer(bar_interval="1min")
        buf.add_tick(pd.Timestamp("2024-01-02 09:30:00"), 100.0, 50)
        bars = buf.get_bars()

        assert len(bars) == 1
        bar = bars.iloc[0]
        assert bar["open"] == bar["high"] == bar["low"] == bar["close"] == 100.0
        assert bar["volume"] == 50


class TestTickBufferFlush:
    """Tests for TickBuffer.flush."""

    def test_flush_returns_bars(self, populated_buffer: TickBuffer) -> None:
        """flush should return the aggregated bars."""
        bars = populated_buffer.flush()
        assert isinstance(bars, pd.DataFrame)
        assert len(bars) == 2

    def test_flush_clears_buffer(self, populated_buffer: TickBuffer) -> None:
        """flush should empty the internal buffer."""
        populated_buffer.flush()
        assert len(populated_buffer) == 0

    def test_flush_then_get_bars_empty(self, populated_buffer: TickBuffer) -> None:
        """After flush, get_bars should return an empty DataFrame."""
        populated_buffer.flush()
        bars = populated_buffer.get_bars()
        assert len(bars) == 0


class TestTickBufferInterval:
    """Tests for different bar intervals."""

    def test_five_minute_bars(self) -> None:
        """5-minute interval should group ticks correctly."""
        buf = TickBuffer(bar_interval="5min")
        base = pd.Timestamp("2024-01-02 09:30:00")

        # Ticks across 10 minutes -> should produce 2 five-minute bars
        for i in range(10):
            buf.add_tick(base + pd.Timedelta(minutes=i), 100.0 + i, 10)

        bars = buf.get_bars()
        assert len(bars) == 2

    def test_hourly_bars(self) -> None:
        """Hourly interval groups ticks into 1-hour bars."""
        buf = TickBuffer(bar_interval="1h")
        base = pd.Timestamp("2024-01-02 09:00:00")

        # One tick per 30 min across 2 hours -> 2 bars
        for i in range(4):
            buf.add_tick(base + pd.Timedelta(minutes=30 * i), 100.0 + i, 50)

        bars = buf.get_bars()
        assert len(bars) == 2
