"""Global configuration for wraquant.

Uses pydantic-settings for environment variable support and validation.
Configuration is a singleton accessible via ``get_config()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import Field
from pydantic_settings import BaseSettings

from wraquant._compat import Backend


class WQConfig(BaseSettings):
    """Global wraquant configuration.

    Settings can be overridden via environment variables prefixed with ``WQ_``.

    Example:
        >>> import wraquant as wq
        >>> cfg = wq.get_config()
        >>> cfg.backend
        <Backend.PANDAS: 'pandas'>
        >>> cfg.backend = Backend.POLARS
    """

    model_config: ClassVar[dict] = {  # type: ignore[misc]
        "env_prefix": "WQ_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    backend: Backend = Field(
        default=Backend.PANDAS,
        description="Default DataFrame backend (pandas, polars, numpy, torch, jax).",
    )

    cache_enabled: bool = Field(
        default=True,
        description="Enable disk caching for data fetches.",
    )

    cache_dir: Path = Field(
        default=Path.home() / ".cache" / "wraquant",
        description="Directory for cached data.",
    )

    cache_ttl_seconds: int = Field(
        default=3600,
        description="Default cache TTL in seconds.",
    )

    float_precision: int = Field(
        default=64,
        description="Default float precision (32 or 64).",
    )

    log_level: str = Field(
        default="WARNING",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )

    max_retries: int = Field(
        default=3,
        description="Maximum retries for data fetching operations.",
    )

    risk_free_rate: float = Field(
        default=0.0,
        description="Default annual risk-free rate for calculations.",
    )

    trading_days_per_year: int = Field(
        default=252,
        description="Number of trading days per year.",
    )

    base_currency: str = Field(
        default="USD",
        description="Default base currency for FX operations.",
    )


# Singleton instance
_config: WQConfig | None = None


def get_config() -> WQConfig:
    """Get the global wraquant configuration singleton.

    Returns:
        The global WQConfig instance.

    Example:
        >>> from wraquant.core import get_config
        >>> cfg = get_config()
        >>> cfg.backend
        <Backend.PANDAS: 'pandas'>
    """
    global _config  # noqa: PLW0603
    if _config is None:
        _config = WQConfig()
    return _config


def reset_config() -> None:
    """Reset the global configuration to defaults."""
    global _config  # noqa: PLW0603
    _config = None
