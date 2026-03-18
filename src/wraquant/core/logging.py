"""Structured logging setup for wraquant.

Uses structlog for structured logging with loguru as the output handler.
"""

from __future__ import annotations

import structlog


def configure_logging(level: str = "WARNING") -> None:
    """Configure structlog for wraquant.

    Parameters:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            structlog.get_level_from_name(level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger instance.

    Parameters:
        name: Logger name, typically ``__name__``.

    Returns:
        A bound structlog logger.

    Example:
        >>> from wraquant.core.logging import get_logger
        >>> log = get_logger(__name__)
        >>> log.info("fetching data", symbol="AAPL")
    """
    return structlog.get_logger(name)
