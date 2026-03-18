"""Tests for backtest event tracking system."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from wraquant.backtest.events import (
    Event,
    EventTracker,
    EventType,
    detect_drawdown_events,
    detect_regime_changes,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_returns(n: int = 500, seed: int = 42) -> pd.Series:
    """Create a synthetic daily return series."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(rng.normal(0.0003, 0.015, size=n), index=dates)


def _make_crisis_returns(n: int = 500, seed: int = 42) -> pd.Series:
    """Create returns with a severe drawdown in the middle."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    rets = rng.normal(0.0005, 0.01, size=n)
    # Inject a 20-day crash in the middle
    crash_start = n // 2
    rets[crash_start: crash_start + 20] = -0.03
    return pd.Series(rets, index=dates)


# ------------------------------------------------------------------
# EventTracker tests
# ------------------------------------------------------------------

class TestEventTracker:
    def test_log_trade(self) -> None:
        tracker = EventTracker()
        ts = datetime(2024, 1, 2, 9, 30)
        event = tracker.log_trade(ts, "AAPL", "buy", 100, 150.0)
        assert isinstance(event, Event)
        assert event.event_type == EventType.TRADE
        assert event.data["asset"] == "AAPL"
        assert event.data["side"] == "buy"
        assert event.data["quantity"] == 100
        assert event.data["price"] == 150.0

    def test_log_trade_with_metadata(self) -> None:
        tracker = EventTracker()
        ts = datetime(2024, 1, 2)
        meta = {"order_id": "abc123", "slippage": 0.001}
        event = tracker.log_trade(ts, "GOOG", "sell", 50, 120.0, metadata=meta)
        assert event.data["metadata"]["order_id"] == "abc123"

    def test_log_rebalance(self) -> None:
        tracker = EventTracker()
        ts = datetime(2024, 3, 1)
        old = {"SPY": 0.6, "AGG": 0.4}
        new = {"SPY": 0.5, "AGG": 0.5}
        event = tracker.log_rebalance(ts, old, new, reason="quarterly")
        assert event.event_type == EventType.REBALANCE
        assert event.data["reason"] == "quarterly"
        assert event.data["old_weights"]["SPY"] == 0.6
        assert event.data["new_weights"]["AGG"] == 0.5

    def test_log_signal(self) -> None:
        tracker = EventTracker()
        ts = datetime(2024, 1, 15)
        event = tracker.log_signal(ts, "momentum_z", 2.3)
        assert event.event_type == EventType.SIGNAL
        assert event.data["signal_name"] == "momentum_z"
        assert event.data["value"] == 2.3

    def test_log_signal_with_metadata(self) -> None:
        tracker = EventTracker()
        ts = datetime(2024, 1, 15)
        event = tracker.log_signal(ts, "rsi", 75.0, metadata={"asset": "MSFT"})
        assert event.data["metadata"]["asset"] == "MSFT"

    def test_log_risk_event(self) -> None:
        tracker = EventTracker()
        ts = datetime(2024, 2, 10)
        event = tracker.log_risk_event(ts, "var_breach", details={"var": -0.03})
        assert event.event_type == EventType.RISK
        assert event.data["risk_event_type"] == "var_breach"
        assert event.data["details"]["var"] == -0.03

    def test_get_events_no_filter(self) -> None:
        tracker = EventTracker()
        ts1 = datetime(2024, 1, 1)
        ts2 = datetime(2024, 1, 2)
        tracker.log_trade(ts1, "A", "buy", 10, 100)
        tracker.log_signal(ts2, "sig", 1.0)
        assert len(tracker.get_events()) == 2

    def test_get_events_by_type(self) -> None:
        tracker = EventTracker()
        ts = datetime(2024, 1, 1)
        tracker.log_trade(ts, "A", "buy", 10, 100)
        tracker.log_trade(ts, "B", "sell", 5, 200)
        tracker.log_signal(ts, "sig", 1.0)
        trades = tracker.get_events(event_type=EventType.TRADE)
        assert len(trades) == 2
        signals = tracker.get_events(event_type="signal")
        assert len(signals) == 1

    def test_get_events_by_time_range(self) -> None:
        tracker = EventTracker()
        ts1 = datetime(2024, 1, 1)
        ts2 = datetime(2024, 6, 15)
        ts3 = datetime(2024, 12, 31)
        tracker.log_trade(ts1, "A", "buy", 10, 100)
        tracker.log_trade(ts2, "B", "buy", 10, 100)
        tracker.log_trade(ts3, "C", "buy", 10, 100)
        mid = tracker.get_events(start=datetime(2024, 3, 1), end=datetime(2024, 9, 1))
        assert len(mid) == 1
        assert mid[0].data["asset"] == "B"

    def test_summary(self) -> None:
        tracker = EventTracker()
        ts = datetime(2024, 1, 1)
        tracker.log_trade(ts, "A", "buy", 1, 1)
        tracker.log_trade(ts, "B", "sell", 1, 1)
        tracker.log_signal(ts, "s", 1.0)
        tracker.log_risk_event(ts, "drawdown_threshold")
        s = tracker.summary()
        assert s["total_events"] == 4
        assert s["by_type"]["trade"] == 2
        assert s["by_type"]["signal"] == 1
        assert s["by_type"]["risk"] == 1

    def test_to_dataframe(self) -> None:
        tracker = EventTracker()
        ts = datetime(2024, 1, 1)
        tracker.log_trade(ts, "AAPL", "buy", 100, 150.0)
        tracker.log_signal(ts, "rsi", 70.0)
        df = tracker.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "timestamp" in df.columns
        assert "event_type" in df.columns

    def test_to_dataframe_empty(self) -> None:
        tracker = EventTracker()
        df = tracker.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_len(self) -> None:
        tracker = EventTracker()
        assert len(tracker) == 0
        tracker.log_trade(datetime(2024, 1, 1), "A", "buy", 1, 1)
        assert len(tracker) == 1


# ------------------------------------------------------------------
# Regime change detection tests
# ------------------------------------------------------------------

class TestDetectRegimeChanges:
    def test_rolling_vol_returns_dataframe(self) -> None:
        rets = _make_crisis_returns(n=500)
        result = detect_regime_changes(rets, method="rolling_vol", window=20)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"timestamp", "regime", "indicator"}

    def test_mean_shift_returns_dataframe(self) -> None:
        rets = _make_crisis_returns(n=500)
        result = detect_regime_changes(rets, method="mean_shift", window=20, threshold=1.0)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"timestamp", "regime", "indicator"}

    def test_short_series_returns_empty(self) -> None:
        rets = pd.Series([0.01, -0.01, 0.005])
        result = detect_regime_changes(rets, window=20)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_invalid_method(self) -> None:
        rets = _make_returns(n=500)
        with pytest.raises(ValueError, match="Unknown method"):
            detect_regime_changes(rets, method="invalid")


# ------------------------------------------------------------------
# Drawdown event detection tests
# ------------------------------------------------------------------

class TestDetectDrawdownEvents:
    def test_crisis_triggers_drawdown(self) -> None:
        rets = _make_crisis_returns(n=500)
        result = detect_drawdown_events(rets, threshold=-0.05)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert all(result["depth"] < -0.05)

    def test_empty_series(self) -> None:
        rets = pd.Series(dtype=float)
        result = detect_drawdown_events(rets)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_no_drawdown_below_threshold(self) -> None:
        # Constant positive returns — no drawdown
        rets = pd.Series(np.full(100, 0.01), index=pd.bdate_range("2020-01-01", periods=100))
        result = detect_drawdown_events(rets, threshold=-0.05)
        assert len(result) == 0

    def test_columns_present(self) -> None:
        rets = _make_crisis_returns()
        result = detect_drawdown_events(rets, threshold=-0.01)
        expected_cols = {"start", "end", "trough_date", "depth", "duration"}
        assert expected_cols == set(result.columns)
