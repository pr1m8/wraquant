"""Abstract base protocols for Frame and Series.

Defines the interface that all backend implementations must satisfy.
Uses Protocol for structural subtyping — backends don't need to inherit.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class AbstractSeries(Protocol):
    """Protocol for a 1-D labeled array of financial data."""

    @property
    def name(self) -> str | None: ...

    @property
    def dtype(self) -> Any: ...

    def __len__(self) -> int: ...

    def to_numpy(self) -> npt.NDArray[np.floating]: ...

    def to_pandas(self) -> Any: ...

    def to_polars(self) -> Any: ...


@runtime_checkable
class AbstractFrame(Protocol):
    """Protocol for a 2-D labeled table of financial data."""

    @property
    def columns(self) -> list[str]: ...

    @property
    def shape(self) -> tuple[int, int]: ...

    def __len__(self) -> int: ...

    def to_numpy(self) -> npt.NDArray[np.floating]: ...

    def to_pandas(self) -> Any: ...

    def to_polars(self) -> Any: ...
