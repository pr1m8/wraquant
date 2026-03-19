"""Shared validation utilities for technical analysis indicators."""

from __future__ import annotations

import pandas as pd

from wraquant.core._coerce import coerce_series as _coerce


def validate_series(data, name: str = "data") -> pd.Series:
    """Coerce input to pd.Series. Accepts Series, ndarray, list, tuple."""
    if isinstance(data, pd.Series):
        return data
    # Auto-convert instead of raising
    return _coerce(data, name=name)


def validate_period(period: int, name: str = "period") -> int:
    """Ensure *period* is a positive integer; raise ``ValueError`` otherwise."""
    if not isinstance(period, int) or period < 1:
        raise ValueError(f"{name} must be a positive integer, got {period}")
    return period
