"""Decorators for optional dependency gating, caching, and validation."""

from __future__ import annotations

import functools
import hashlib
import time
from typing import Any, Callable, TypeVar

from wraquant._lazy import check_extra
from wraquant.core.exceptions import MissingDependencyError

F = TypeVar("F", bound=Callable[..., Any])


def requires_extra(group: str) -> Callable[[F], F]:
    """Decorator that raises MissingDependencyError if an optional group is missing.

    Parameters:
        group: The PDM optional dependency group name (e.g., 'market-data').

    Returns:
        Decorated function that checks for the dependency before execution.

    Example:
        >>> @requires_extra('market-data')
        ... def fetch_yahoo(symbol: str) -> pd.DataFrame:
        ...     import yfinance
        ...     return yfinance.download(symbol)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not check_extra(group):
                raise MissingDependencyError(
                    package=group,
                    extra_group=group,
                )
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def cache_result(ttl: int = 3600) -> Callable[[F], F]:
    """Simple in-memory cache with TTL for expensive computations.

    Parameters:
        ttl: Time-to-live in seconds. Defaults to 1 hour.

    Returns:
        Decorated function with caching behavior.
    """
    cache: dict[str, tuple[float, Any]] = {}

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = hashlib.md5(
                f"{func.__name__}:{args}:{sorted(kwargs.items())}".encode(),
                usedforsecurity=False,
            ).hexdigest()

            now = time.monotonic()
            if key in cache:
                timestamp, result = cache[key]
                if now - timestamp < ttl:
                    return result

            result = func(*args, **kwargs)
            cache[key] = (now, result)
            return result

        wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


def validate_input(func: F) -> F:
    """Decorator that validates function inputs using type hints.

    Currently a pass-through that documents the intent for pydantic
    validation integration in the future.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
