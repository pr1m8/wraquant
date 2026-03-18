"""Event tracking system for portfolio backtesting.

Provides structured logging and querying of portfolio events including
trades, rebalances, signals, risk events, regime changes, and drawdowns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "EventType",
    "Event",
    "EventTracker",
    "detect_regime_changes",
    "detect_drawdown_events",
]


class EventType(str, Enum):
    """Enumeration of trackable event types."""

    TRADE = "trade"
    REBALANCE = "rebalance"
    SIGNAL = "signal"
    RISK = "risk"
    REGIME_CHANGE = "regime_change"
    DRAWDOWN = "drawdown"


@dataclass
class Event:
    """A single portfolio event.

    Parameters
    ----------
    timestamp : datetime
        When the event occurred.
    event_type : EventType
        Category of the event.
    data : dict[str, Any]
        Event-specific payload.
    """

    timestamp: datetime
    event_type: EventType
    data: dict[str, Any] = field(default_factory=dict)


class EventTracker:
    """Track and query portfolio events during a backtest.

    Maintains an ordered log of events (trades, rebalances, signals,
    risk events) and provides query and summary facilities.

    Example
    -------
    >>> tracker = EventTracker()
    >>> tracker.log_trade(datetime(2024, 1, 2), "AAPL", "buy", 100, 150.0)
    >>> tracker.summary()["total_events"]
    1
    """

    def __init__(self) -> None:
        self._events: list[Event] = []

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log_trade(
        self,
        timestamp: datetime,
        asset: str,
        side: str,
        quantity: float,
        price: float,
        metadata: dict[str, Any] | None = None,
    ) -> Event:
        """Log a trade event.

        Parameters
        ----------
        timestamp : datetime
            Trade execution time.
        asset : str
            Instrument identifier.
        side : str
            ``"buy"`` or ``"sell"``.
        quantity : float
            Number of units traded.
        price : float
            Execution price per unit.
        metadata : dict, optional
            Additional trade metadata.

        Returns
        -------
        Event
            The logged event.
        """
        data: dict[str, Any] = {
            "asset": asset,
            "side": side,
            "quantity": quantity,
            "price": price,
        }
        if metadata:
            data["metadata"] = metadata
        event = Event(timestamp=timestamp, event_type=EventType.TRADE, data=data)
        self._events.append(event)
        return event

    def log_rebalance(
        self,
        timestamp: datetime,
        old_weights: dict[str, float],
        new_weights: dict[str, float],
        reason: str = "",
    ) -> Event:
        """Log a portfolio rebalance event.

        Parameters
        ----------
        timestamp : datetime
            When the rebalance occurred.
        old_weights : dict[str, float]
            Pre-rebalance weights by asset.
        new_weights : dict[str, float]
            Post-rebalance weights by asset.
        reason : str
            Reason for the rebalance (e.g., ``"scheduled"``, ``"drift"``).

        Returns
        -------
        Event
            The logged event.
        """
        data: dict[str, Any] = {
            "old_weights": old_weights,
            "new_weights": new_weights,
            "reason": reason,
        }
        event = Event(timestamp=timestamp, event_type=EventType.REBALANCE, data=data)
        self._events.append(event)
        return event

    def log_signal(
        self,
        timestamp: datetime,
        signal_name: str,
        value: float,
        metadata: dict[str, Any] | None = None,
    ) -> Event:
        """Log a signal generation event.

        Parameters
        ----------
        timestamp : datetime
            When the signal was generated.
        signal_name : str
            Name of the signal.
        value : float
            Signal value.
        metadata : dict, optional
            Additional signal metadata.

        Returns
        -------
        Event
            The logged event.
        """
        data: dict[str, Any] = {"signal_name": signal_name, "value": value}
        if metadata:
            data["metadata"] = metadata
        event = Event(timestamp=timestamp, event_type=EventType.SIGNAL, data=data)
        self._events.append(event)
        return event

    def log_risk_event(
        self,
        timestamp: datetime,
        event_type: str,
        details: dict[str, Any] | None = None,
    ) -> Event:
        """Log a risk event (VaR breach, drawdown threshold, etc.).

        Parameters
        ----------
        timestamp : datetime
            When the risk event was detected.
        event_type : str
            Type of risk event (e.g., ``"var_breach"``, ``"drawdown_threshold"``).
        details : dict, optional
            Event-specific details.

        Returns
        -------
        Event
            The logged event.
        """
        data: dict[str, Any] = {"risk_event_type": event_type}
        if details:
            data["details"] = details
        event = Event(timestamp=timestamp, event_type=EventType.RISK, data=data)
        self._events.append(event)
        return event

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_events(
        self,
        event_type: EventType | str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Event]:
        """Query events by type and/or time range.

        Parameters
        ----------
        event_type : EventType or str, optional
            Filter to a specific event type.
        start : datetime, optional
            Inclusive lower bound on timestamp.
        end : datetime, optional
            Inclusive upper bound on timestamp.

        Returns
        -------
        list[Event]
            Matching events in chronological order.
        """
        if isinstance(event_type, str):
            event_type = EventType(event_type)

        result = self._events
        if event_type is not None:
            result = [e for e in result if e.event_type == event_type]
        if start is not None:
            result = [e for e in result if e.timestamp >= start]
        if end is not None:
            result = [e for e in result if e.timestamp <= end]
        return result

    def summary(self) -> dict[str, Any]:
        """Return summary statistics for all logged events.

        Returns
        -------
        dict[str, Any]
            Dictionary with ``total_events`` and per-type counts.
        """
        counts: dict[str, int] = {}
        for e in self._events:
            key = e.event_type.value
            counts[key] = counts.get(key, 0) + 1
        return {"total_events": len(self._events), "by_type": counts}

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the event log to a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``timestamp``, ``event_type``, and
            one column per data key.
        """
        if not self._events:
            return pd.DataFrame(columns=["timestamp", "event_type"])

        rows: list[dict[str, Any]] = []
        for e in self._events:
            row: dict[str, Any] = {
                "timestamp": e.timestamp,
                "event_type": e.event_type.value,
            }
            row.update(e.data)
            rows.append(row)
        return pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self._events)


# ------------------------------------------------------------------
# Standalone detection helpers
# ------------------------------------------------------------------


def detect_regime_changes(
    returns: pd.Series,
    method: str = "rolling_vol",
    window: int = 63,
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Detect regime transitions in a return series.

    Parameters
    ----------
    returns : pd.Series
        Asset or portfolio returns.
    method : str
        Detection method.  Supported: ``"rolling_vol"`` (default) uses
        a ratio of short-term to long-term rolling volatility,
        ``"mean_shift"`` uses a shift in the rolling mean.
    window : int
        Lookback window (number of periods).
    threshold : float
        Multiplier above which a regime change is flagged.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``timestamp``, ``regime``, and
        ``indicator`` for each detected transition point.
    """
    returns = returns.dropna()
    if len(returns) < window * 2:
        return pd.DataFrame(columns=["timestamp", "regime", "indicator"])

    if method == "rolling_vol":
        short_vol = returns.rolling(window).std()
        long_vol = returns.rolling(window * 2).std()
        ratio = short_vol / long_vol.replace(0, np.nan)
        ratio = ratio.dropna()
        high_vol = ratio > threshold
        low_vol = ratio < (1.0 / threshold)
        regime = pd.Series("normal", index=ratio.index)
        regime[high_vol] = "high_vol"
        regime[low_vol] = "low_vol"
        changes = regime != regime.shift(1)
        change_points = regime[changes].iloc[1:]  # skip first trivial change
        records = [
            {"timestamp": idx, "regime": val, "indicator": float(ratio.loc[idx])}
            for idx, val in change_points.items()
        ]
    elif method == "mean_shift":
        rolling_mean = returns.rolling(window).mean().dropna()
        overall_mean = returns.mean()
        overall_std = returns.std()
        z = (rolling_mean - overall_mean) / (overall_std if overall_std > 0 else 1.0)
        regime = pd.Series("normal", index=rolling_mean.index)
        regime[z > threshold] = "bull"
        regime[z < -threshold] = "bear"
        changes = regime != regime.shift(1)
        change_points = regime[changes].iloc[1:]
        records = [
            {"timestamp": idx, "regime": val, "indicator": float(z.loc[idx])}
            for idx, val in change_points.items()
        ]
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'rolling_vol' or 'mean_shift'.")

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["timestamp", "regime", "indicator"]
    )


def detect_drawdown_events(
    returns: pd.Series,
    threshold: float = -0.05,
) -> pd.DataFrame:
    """Detect drawdown events that exceed a given threshold.

    Parameters
    ----------
    returns : pd.Series
        Asset or portfolio return series.
    threshold : float
        Drawdown level (negative) below which events are recorded.
        Default ``-0.05`` (5 % drawdown).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``start``, ``end``, ``trough_date``,
        ``depth``, and ``duration`` for each drawdown event.
    """
    returns = returns.dropna()
    if returns.empty:
        return pd.DataFrame(
            columns=["start", "end", "trough_date", "depth", "duration"]
        )

    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak

    events: list[dict[str, Any]] = []
    dd_start = None
    trough_val = 0.0
    trough_date = None

    for i, (idx, val) in enumerate(drawdown.items()):
        if val < threshold:
            if dd_start is None:
                dd_start = idx
                trough_val = val
                trough_date = idx
            if val < trough_val:
                trough_val = val
                trough_date = idx
        else:
            if dd_start is not None:
                events.append(
                    {
                        "start": dd_start,
                        "end": idx,
                        "trough_date": trough_date,
                        "depth": float(trough_val),
                        "duration": i - list(drawdown.index).index(dd_start),
                    }
                )
                dd_start = None

    # Handle drawdown that extends to the end of the series
    if dd_start is not None:
        events.append(
            {
                "start": dd_start,
                "end": drawdown.index[-1],
                "trough_date": trough_date,
                "depth": float(trough_val),
                "duration": len(drawdown) - list(drawdown.index).index(dd_start),
            }
        )

    return pd.DataFrame(events) if events else pd.DataFrame(
        columns=["start", "end", "trough_date", "depth", "duration"]
    )
