"""Trading calendar utilities.

Wraps exchange-calendars and pandas-market-calendars for trading day
schedules, market hours, and holiday detection.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from wraquant.core.decorators import requires_extra
from wraquant.core.types import DateLike
from wraquant.data.utils import parse_date


@requires_extra("market-data")
def get_trading_calendar(exchange: str = "XNYS") -> Any:
    """Get a trading calendar for an exchange.

    Parameters:
        exchange: Exchange MIC code (default: NYSE = 'XNYS').

    Returns:
        exchange_calendars.ExchangeCalendar instance.

    Example:
        >>> cal = get_trading_calendar("XNYS")  # doctest: +SKIP
    """
    import exchange_calendars

    return exchange_calendars.get_calendar(exchange)


@requires_extra("market-data")
def trading_days(
    start: DateLike,
    end: DateLike,
    exchange: str = "XNYS",
) -> pd.DatetimeIndex:
    """Get trading days between two dates.

    Parameters:
        start: Start date.
        end: End date.
        exchange: Exchange MIC code.

    Returns:
        DatetimeIndex of valid trading days.
    """
    cal = get_trading_calendar(exchange)
    s = parse_date(start)
    e = parse_date(end)
    sessions = cal.sessions_in_range(s, e)
    return pd.DatetimeIndex(sessions)


def is_business_day(dt: DateLike) -> bool:
    """Check if a date is a business day (Mon-Fri).

    Parameters:
        dt: Date to check.

    Returns:
        True if weekday, False if weekend.
    """
    ts = parse_date(dt)
    if ts is None:
        raise ValueError("Date cannot be None")
    return ts.weekday() < 5
