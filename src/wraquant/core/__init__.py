"""Core utilities: configuration, types, exceptions, logging, and decorators."""

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
