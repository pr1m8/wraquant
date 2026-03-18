"""Data utility functions — date parsing, symbol cleaning, etc."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from wraquant.core.types import DateLike


def parse_date(d: DateLike | None) -> pd.Timestamp | None:
    """Parse a date-like value into a pd.Timestamp.

    Parameters:
        d: Date string, date, datetime, Timestamp, or None.

    Returns:
        pd.Timestamp or None if input is None.
    """
    if d is None:
        return None
    if isinstance(d, pd.Timestamp):
        return d
    if isinstance(d, (date, datetime)):
        return pd.Timestamp(d)
    if isinstance(d, np.datetime64):
        return pd.Timestamp(d)
    if isinstance(d, str):
        return pd.Timestamp(d)
    raise TypeError(f"Cannot parse {type(d)} as date: {d}")


def clean_symbol(symbol: str) -> str:
    """Normalize a ticker symbol.

    Parameters:
        symbol: Raw ticker symbol.

    Returns:
        Cleaned, uppercase ticker symbol.
    """
    return symbol.strip().upper()


def infer_frequency(index: pd.DatetimeIndex) -> str | None:
    """Attempt to infer the frequency of a DatetimeIndex.

    Parameters:
        index: DatetimeIndex to analyze.

    Returns:
        Frequency string or None if cannot be determined.
    """
    if len(index) < 2:
        return None
    freq = pd.infer_freq(index)
    return freq
