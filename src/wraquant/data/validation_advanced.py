"""Advanced data validation using pandera.

Provides pandera schema validation for DataFrames and pre-built
schemas for common financial data formats (OHLCV, returns).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from wraquant.core.decorators import requires_extra

__all__ = [
    "pandera_validate",
    "create_ohlcv_schema",
    "create_returns_schema",
]


@requires_extra("validation")
def pandera_validate(
    df: pd.DataFrame,
    schema: Any,
) -> pd.DataFrame:
    """Validate a DataFrame against a pandera schema.

    Wraps ``schema.validate()`` and returns the validated DataFrame
    (which may include coerced dtypes).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    schema : pandera.DataFrameSchema
        Pandera schema defining the expected structure, dtypes, and
        value constraints.

    Returns
    -------
    pd.DataFrame
        The validated (and potentially coerced) DataFrame.

    Raises
    ------
    pandera.errors.SchemaError
        If validation fails.
    """
    return schema.validate(df)


@requires_extra("validation")
def create_ohlcv_schema(
    strict: bool = False,
    coerce: bool = True,
) -> Any:
    """Create a pandera schema for OHLCV financial data.

    The schema enforces:

    * Columns ``open``, ``high``, ``low``, ``close`` are positive floats.
    * Column ``volume`` is a non-negative integer or float.
    * ``high >= low`` for every row.
    * ``close`` is within ``[low, high]`` for every row.

    Parameters
    ----------
    strict : bool, default False
        If *True*, extra columns not in the schema cause validation to
        fail.
    coerce : bool, default True
        If *True*, attempt to coerce column dtypes before validation.

    Returns
    -------
    pandera.DataFrameSchema
        Schema suitable for passing to :func:`pandera_validate`.
    """
    import pandera as pa

    return pa.DataFrameSchema(
        columns={
            "open": pa.Column(float, pa.Check.gt(0), coerce=coerce, nullable=False),
            "high": pa.Column(float, pa.Check.gt(0), coerce=coerce, nullable=False),
            "low": pa.Column(float, pa.Check.gt(0), coerce=coerce, nullable=False),
            "close": pa.Column(float, pa.Check.gt(0), coerce=coerce, nullable=False),
            "volume": pa.Column(float, pa.Check.ge(0), coerce=coerce, nullable=False),
        },
        checks=[
            pa.Check(
                lambda df: df["high"] >= df["low"],
                error="high must be >= low",
            ),
            pa.Check(
                lambda df: (df["close"] >= df["low"]) & (df["close"] <= df["high"]),
                error="close must be within [low, high]",
            ),
        ],
        strict=strict,
        coerce=coerce,
    )


@requires_extra("validation")
def create_returns_schema(
    max_abs_return: float = 1.0,
    allow_nan: bool = False,
    strict: bool = False,
    coerce: bool = True,
) -> Any:
    """Create a pandera schema for financial return data.

    The schema enforces:

    * All return columns are float type.
    * Return values are within ``[-max_abs_return, max_abs_return]``.

    Parameters
    ----------
    max_abs_return : float, default 1.0
        Maximum allowed absolute return value. Values outside
        ``[-max_abs_return, max_abs_return]`` fail validation.
    allow_nan : bool, default False
        Whether NaN values are allowed in return columns.
    strict : bool, default False
        If *True*, extra columns cause failure.
    coerce : bool, default True
        If *True*, attempt dtype coercion before validation.

    Returns
    -------
    pandera.DataFrameSchema
        Schema suitable for passing to :func:`pandera_validate`.
    """
    import pandera as pa

    return pa.DataFrameSchema(
        columns={
            "returns": pa.Column(
                float,
                checks=[
                    pa.Check.in_range(
                        -max_abs_return,
                        max_abs_return,
                    ),
                ],
                coerce=coerce,
                nullable=allow_nan,
            ),
        },
        strict=strict,
        coerce=coerce,
    )
