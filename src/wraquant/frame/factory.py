"""Factory functions for creating Frame and Series objects.

Auto-detects input type or uses the configured default backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from wraquant._compat import Backend
from wraquant._lazy import is_available
from wraquant.core.config import get_config

if TYPE_CHECKING:
    import polars as pl


def _import_polars() -> Any:
    """Import Polars only when the Polars backend is requested."""
    try:
        import polars as pl
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Optional dependency 'polars' is not installed. "
            "Install it with: pdm install -G accelerate"
        ) from exc
    return pl


def series(
    data: Any,
    *,
    name: str | None = None,
    index: Any = None,
    backend: Backend | str | None = None,
) -> pd.Series | "pl.Series":
    """Create a Series using the specified or default backend.

    Parameters:
        data: Input data (list, ndarray, pd.Series, pl.Series).
        name: Series name.
        index: Index for pandas backend (ignored for polars).
        backend: Override the default backend.

    Returns:
        A Series in the requested backend format.

    Example:
        >>> s = series([1.0, 2.0, 3.0], name="prices")
        >>> type(s)
        <class 'pandas.core.series.Series'>
    """
    be = Backend(backend) if backend else get_config().backend

    if be == Backend.POLARS:
        pl = _import_polars()
        if isinstance(data, pl.Series):
            return data.alias(name) if name else data
        if isinstance(data, pd.Series):
            return pl.from_pandas(data)
        return pl.Series(name=name or "", values=data)

    # Default: pandas
    if isinstance(data, pd.Series):
        s = data.copy()
        if name:
            s.name = name
        return s
    if is_available("polars"):
        pl = _import_polars()
        if isinstance(data, pl.Series):
            return data.to_pandas()
    return pd.Series(data, index=index, name=name)


def frame(
    data: Any,
    *,
    columns: list[str] | None = None,
    index: Any = None,
    backend: Backend | str | None = None,
) -> pd.DataFrame | "pl.DataFrame":
    """Create a DataFrame using the specified or default backend.

    Parameters:
        data: Input data (dict, ndarray, pd.DataFrame, pl.DataFrame).
        columns: Column names.
        index: Index for pandas backend (ignored for polars).
        backend: Override the default backend.

    Returns:
        A DataFrame in the requested backend format.

    Example:
        >>> df = frame({"a": [1, 2], "b": [3, 4]})
        >>> type(df)
        <class 'pandas.core.frame.DataFrame'>
    """
    be = Backend(backend) if backend else get_config().backend

    if be == Backend.POLARS:
        pl = _import_polars()
        if isinstance(data, pl.DataFrame):
            return data
        if isinstance(data, pd.DataFrame):
            return pl.from_pandas(data)
        if isinstance(data, np.ndarray):
            cols = columns or [f"col_{i}" for i in range(data.shape[1])]
            return pl.DataFrame({c: data[:, i] for i, c in enumerate(cols)})
        if isinstance(data, dict):
            return pl.DataFrame(data)
        return pl.DataFrame(data)

    # Default: pandas
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if is_available("polars"):
        pl = _import_polars()
        if isinstance(data, pl.DataFrame):
            return data.to_pandas()
    if isinstance(data, np.ndarray):
        return pd.DataFrame(data, columns=columns, index=index)
    if isinstance(data, dict):
        return pd.DataFrame(data, index=index)
    return pd.DataFrame(data, columns=columns, index=index)
