"""Core utilities: configuration, types, exceptions, logging, and decorators.

This module provides the foundational infrastructure that every other
wraquant module depends on.  It defines the configuration system, the
type coercion layer, canonical exception hierarchy, reusable decorators,
and structured result dataclasses.

Key components:

- **WQConfig / get_config** -- Global configuration singleton controlling
  the compute backend (pandas, polars, torch), default frequency, and
  logging verbosity.
- **coerce_series / coerce_array / coerce_returns / coerce_dataframe** --
  Coerce-first type system that normalizes heterogeneous inputs (lists,
  numpy arrays, polars Series) into pandas types before computation.
- **WQError** and subclasses (``DataFetchError``, ``ValidationError``,
  ``ConfigError``, ``OptimizationError``) -- Structured exception
  hierarchy for clear error handling across all modules.
- **@requires_extra** -- Decorator that gates functions behind optional
  dependency groups, raising ``MissingDependencyError`` with install
  instructions when the dependency is absent.
- **@cache_result / @validate_input** -- Utility decorators for memoization
  and input validation.
- **GARCHResult / BacktestResult / ForecastResult** -- Typed result
  dataclasses returned by modeling functions for consistent downstream
  consumption.
- **Frequency / AssetClass / Currency / ReturnType** and other enums --
  Canonical enumerations used as parameters throughout the library.

Example:
    >>> from wraquant.core import get_config, coerce_series
    >>> cfg = get_config()
    >>> cfg.backend
    <Backend.PANDAS: 'pandas'>
    >>> import numpy as np
    >>> s = coerce_series(np.array([1.0, 2.0, 3.0]), name="prices")

Use ``wraquant.core`` when you need to configure global settings, coerce
raw inputs into standard types, or handle wraquant-specific exceptions.
Most users interact with this module indirectly through higher-level
modules like ``wraquant.stats`` or ``wraquant.risk``.
"""

from wraquant.core._coerce import (
    coerce_array,
    coerce_dataframe,
    coerce_returns,
    coerce_series,
)
from wraquant.core.config import WQConfig, get_config
from wraquant.core.decorators import cache_result, requires_extra, validate_input
from wraquant.core.exceptions import (
    ConfigError,
    DataFetchError,
    MissingDependencyError,
    OptimizationError,
    ValidationError,
    WQError,
)
from wraquant.core.results import BacktestResult, ForecastResult, GARCHResult
from wraquant.core.types import (
    AssetClass,
    Currency,
    Frequency,
    OptionType,
    OrderSide,
    ReturnType,
)

__all__ = [
    # Coercion
    "coerce_array",
    "coerce_series",
    "coerce_returns",
    "coerce_dataframe",
    # Config
    "WQConfig",
    "get_config",
    # Exceptions
    "WQError",
    "MissingDependencyError",
    "DataFetchError",
    "ValidationError",
    "ConfigError",
    "OptimizationError",
    # Types
    "AssetClass",
    "Currency",
    "Frequency",
    "OptionType",
    "OrderSide",
    "ReturnType",
    # Decorators
    "requires_extra",
    "cache_result",
    "validate_input",
    # Results
    "GARCHResult",
    "BacktestResult",
    "ForecastResult",
]
