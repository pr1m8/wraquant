"""Custom exception hierarchy for wraquant."""

from __future__ import annotations


class WQError(Exception):
    """Base exception for all wraquant errors."""


class MissingDependencyError(WQError, ImportError):
    """Raised when an optional dependency is not installed.

    Parameters:
        package: Name of the missing package.
        extra_group: The PDM extra group that provides it.
    """

    def __init__(self, package: str, extra_group: str | None = None) -> None:
        self.package = package
        self.extra_group = extra_group
        msg = f"'{package}' is not installed."
        if extra_group:
            msg += f" Install with: pdm install -G {extra_group}"
        super().__init__(msg)


class DataFetchError(WQError):
    """Raised when data fetching from a provider fails."""

    def __init__(self, provider: str, symbol: str, reason: str = "") -> None:
        self.provider = provider
        self.symbol = symbol
        msg = f"Failed to fetch '{symbol}' from {provider}"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class ValidationError(WQError, ValueError):
    """Raised when input data fails validation."""


class ConfigError(WQError):
    """Raised when there is a configuration error."""


class OptimizationError(WQError):
    """Raised when an optimization problem fails to solve."""

    def __init__(self, solver: str, reason: str = "") -> None:
        self.solver = solver
        msg = f"Optimization failed (solver={solver})"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class BacktestError(WQError):
    """Raised when a backtesting operation fails."""


class PricingError(WQError):
    """Raised when a pricing calculation fails."""
