"""Forex trading session utilities.

Defines the four major forex sessions and their overlaps.
"""

from __future__ import annotations

from datetime import datetime, time, timezone
from enum import StrEnum


class ForexSession(StrEnum):
    """Major forex trading sessions."""

    SYDNEY = "sydney"
    TOKYO = "tokyo"
    LONDON = "london"
    NEW_YORK = "new_york"


# Session times in UTC
_SESSION_TIMES: dict[ForexSession, tuple[time, time]] = {
    ForexSession.SYDNEY: (time(21, 0), time(6, 0)),  # 9 PM - 6 AM UTC
    ForexSession.TOKYO: (time(0, 0), time(9, 0)),  # 12 AM - 9 AM UTC
    ForexSession.LONDON: (time(7, 0), time(16, 0)),  # 7 AM - 4 PM UTC
    ForexSession.NEW_YORK: (time(12, 0), time(21, 0)),  # 12 PM - 9 PM UTC
}


def current_session(
    dt: datetime | None = None,
) -> list[ForexSession]:
    """Determine which forex sessions are active.

    Parameters:
        dt: Datetime in UTC. Defaults to now.

    Returns:
        List of active ForexSession values.

    Example:
        >>> current_session(datetime(2024, 1, 15, 14, 0, tzinfo=timezone.utc))
        [<ForexSession.LONDON: 'london'>, <ForexSession.NEW_YORK: 'new_york'>]
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    current_time = dt.time()
    active = []

    for session, (start, end) in _SESSION_TIMES.items():
        if start <= end:
            if start <= current_time < end:
                active.append(session)
        else:
            # Crosses midnight
            if current_time >= start or current_time < end:
                active.append(session)

    return active


def session_overlaps() -> list[tuple[ForexSession, ForexSession, time, time]]:
    """Return the major session overlap periods.

    Returns:
        List of tuples: (session1, session2, start_utc, end_utc).
    """
    return [
        (ForexSession.TOKYO, ForexSession.LONDON, time(7, 0), time(9, 0)),
        (ForexSession.LONDON, ForexSession.NEW_YORK, time(12, 0), time(16, 0)),
        (ForexSession.NEW_YORK, ForexSession.SYDNEY, time(21, 0), time(21, 0)),
    ]


def session_hours(session: ForexSession) -> tuple[time, time]:
    """Get the UTC start and end times for a session.

    Parameters:
        session: Forex trading session.

    Returns:
        Tuple of (start_time, end_time) in UTC.
    """
    return _SESSION_TIMES[session]
