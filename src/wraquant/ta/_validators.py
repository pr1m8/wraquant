"""Shared validation utilities for technical analysis indicators."""

from __future__ import annotations

import pandas as pd


def validate_series(data: pd.Series, name: str = "data") -> pd.Series:
    """Ensure *data* is a ``pd.Series``; raise ``TypeError`` otherwise."""
    if not isinstance(data, pd.Series):
        raise TypeError(f"{name} must be a pd.Series, got {type(data).__name__}")
    return data


def validate_period(period: int, name: str = "period") -> int:
    """Ensure *period* is a positive integer; raise ``ValueError`` otherwise."""
    if not isinstance(period, int) or period < 1:
        raise ValueError(f"{name} must be a positive integer, got {period}")
    return period
