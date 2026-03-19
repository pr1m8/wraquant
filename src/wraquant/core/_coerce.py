"""Input coercion utilities for wraquant.

Normalizes diverse input types (pd.Series, np.ndarray, torch.Tensor,
lists, pd.DataFrame) into canonical forms for computation. Every
public wraquant function should coerce inputs at entry rather than
requiring users to convert manually.

The "coerce-first" pattern: accept anything reasonable, normalize
once at the top of the function, compute on the canonical form.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def coerce_array(data: Any, name: str = "data") -> NDArray[np.floating]:
    """Coerce any array-like input to a 1D float64 numpy array.

    Accepts: pd.Series, pd.DataFrame (first column), np.ndarray,
    torch.Tensor, list, tuple, scalar.

    Parameters:
        data: Input data in any supported format.
        name: Name for error messages.

    Returns:
        1D float64 numpy array.

    Raises:
        TypeError: If data cannot be converted.

    Example:
        >>> import numpy as np
        >>> from wraquant.core._coerce import coerce_array
        >>> coerce_array([1.0, 2.0, 3.0])
        array([1., 2., 3.])
        >>> import pandas as pd
        >>> coerce_array(pd.Series([10, 20, 30]))
        array([10., 20., 30.])
    """
    if isinstance(data, np.ndarray):
        return np.asarray(data, dtype=np.float64).ravel()
    if isinstance(data, pd.Series):
        return data.to_numpy(dtype=np.float64, na_value=np.nan)
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0].to_numpy(dtype=np.float64, na_value=np.nan)
    if hasattr(data, "numpy"):  # torch.Tensor
        return data.detach().cpu().numpy().astype(np.float64).ravel()
    try:
        return np.asarray(data, dtype=np.float64).ravel()
    except (ValueError, TypeError) as e:
        raise TypeError(f"{name} cannot be converted to array: {e}") from e


def coerce_series(data: Any, name: str = "data") -> pd.Series:
    """Coerce any array-like input to a pd.Series.

    Preserves index if input is already a pd.Series.
    Creates integer index for other types.

    Parameters:
        data: Input data.
        name: Name for the resulting Series.

    Returns:
        pd.Series with float64 dtype.

    Example:
        >>> import numpy as np
        >>> from wraquant.core._coerce import coerce_series
        >>> s = coerce_series([1.0, 2.0, 3.0], name="prices")
        >>> s.name
        'prices'
        >>> len(s)
        3
    """
    if isinstance(data, pd.Series):
        return data.astype(float)
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0].astype(float)
    arr = coerce_array(data, name)
    return pd.Series(arr, name=name)


def coerce_returns(
    data: Any,
    is_prices: bool | None = None,
    name: str = "returns",
) -> NDArray[np.floating]:
    """Coerce input to return series.

    Auto-detects whether input is prices or returns based on:
    - If all values > 0 and mean > 0.5: likely prices -> compute pct_change
    - Otherwise: likely returns -> pass through

    Parameters:
        data: Price series, return series, or any array-like.
        is_prices: Force interpretation. None = auto-detect.
        name: Name for error messages.

    Returns:
        1D float64 array of returns.

    Example:
        >>> import numpy as np
        >>> from wraquant.core._coerce import coerce_returns
        >>> prices = [100, 102, 101, 103]
        >>> returns = coerce_returns(prices, is_prices=True)
        >>> len(returns)
        3
    """
    arr = coerce_array(data, name)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return arr

    if is_prices is True or (
        is_prices is None and np.all(arr > 0) and np.mean(arr) > 0.5
    ):
        # Looks like prices -- convert to returns
        return np.diff(arr) / arr[:-1]

    return arr


def coerce_dataframe(data: Any, name: str = "data") -> pd.DataFrame:
    """Coerce input to pd.DataFrame.

    Parameters:
        data: DataFrame, dict, ndarray, or Series.
        name: Name for error messages.

    Returns:
        pd.DataFrame with float64 dtypes.

    Raises:
        TypeError: If data cannot be converted to a DataFrame.

    Example:
        >>> import numpy as np
        >>> from wraquant.core._coerce import coerce_dataframe
        >>> df = coerce_dataframe({"a": [1, 2], "b": [3, 4]})
        >>> df.shape
        (2, 2)
    """
    if isinstance(data, pd.DataFrame):
        return data.astype(float)
    if isinstance(data, pd.Series):
        return data.to_frame().astype(float)
    if isinstance(data, dict):
        return pd.DataFrame(data).astype(float)
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            return pd.DataFrame({name: data.astype(float)})
        return pd.DataFrame(data.astype(float))
    raise TypeError(
        f"{name} must be DataFrame, dict, or array, got {type(data).__name__}"
    )
