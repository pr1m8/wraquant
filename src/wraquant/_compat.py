"""Backend detection and compatibility helpers.

Detects which DataFrame/tensor backends are available and provides
utilities for backend-agnostic operations.
"""

from __future__ import annotations

from enum import StrEnum

from wraquant._lazy import is_available

# Always available (core deps)
HAS_PANDAS = True
HAS_POLARS = True
HAS_NUMPY = True

# Optional backends
HAS_TORCH = is_available("torch")
HAS_JAX = is_available("jax")
HAS_DASK = is_available("dask")


class Backend(StrEnum):
    """Supported DataFrame/computation backends."""

    PANDAS = "pandas"
    POLARS = "polars"
    NUMPY = "numpy"
    TORCH = "torch"
    JAX = "jax"


def get_available_backends() -> list[Backend]:
    """Return list of currently available backends.

    Returns:
        List of Backend enum values for installed backends.
    """
    backends = [Backend.PANDAS, Backend.POLARS, Backend.NUMPY]
    if HAS_TORCH:
        backends.append(Backend.TORCH)
    if HAS_JAX:
        backends.append(Backend.JAX)
    return backends
