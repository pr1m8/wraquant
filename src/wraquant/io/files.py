"""File format I/O with financial defaults.

Provides convenience wrappers around pandas I/O functions with sensible
defaults for financial time-series data (date parsing, index handling, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from wraquant.core._coerce import coerce_dataframe, coerce_series

__all__ = [
    "read_csv",
    "write_csv",
    "read_parquet",
    "write_parquet",
    "read_hdf",
    "write_hdf",
    "read_excel",
    "write_excel",
]


def read_csv(
    path: str | Path,
    date_column: str = "Date",
    parse_dates: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a CSV file with financial defaults.

    By default, parses a ``Date`` column and sets it as the index, which
    matches the most common layout for financial time-series CSVs
    (Yahoo Finance downloads, FRED exports, etc.).  Pass
    ``parse_dates=False`` to disable automatic date parsing.

    Parameters:
        path (str | Path): Path to the CSV file.
        date_column (str): Name of the date column to parse and set as
            index (default ``'Date'``).  Ignored when *parse_dates* is
            ``False``.
        parse_dates (bool): If ``True``, parse *date_column* as
            datetime and use it as the DataFrame index.
        **kwargs: Additional keyword arguments forwarded to
            :func:`pandas.read_csv`.

    Returns:
        pd.DataFrame: DataFrame with the CSV contents, optionally
            date-indexed.

    Example:
        >>> df = read_csv("prices.csv")  # doctest: +SKIP
        >>> df = read_csv("data.csv", parse_dates=False)  # doctest: +SKIP

    See Also:
        write_csv: Write a DataFrame to CSV.
        read_parquet: Faster columnar format for large datasets.
    """
    path = Path(path)

    if parse_dates:
        kwargs.setdefault("parse_dates", [date_column])
        kwargs.setdefault("index_col", date_column)

    return pd.read_csv(path, **kwargs)


def write_csv(
    data: pd.DataFrame | pd.Series,
    path: str | Path,
    **kwargs: Any,
) -> None:
    """Write a DataFrame or Series to CSV.

    Creates parent directories automatically if they do not exist.

    Parameters:
        data (pd.DataFrame | pd.Series): Data to write.
        path (str | Path): Destination file path.
        **kwargs: Additional keyword arguments forwarded to
            :meth:`pandas.DataFrame.to_csv`.

    Example:
        >>> write_csv(df, "output/prices.csv")  # doctest: +SKIP

    See Also:
        read_csv: Read a CSV file with financial defaults.
    """
    if isinstance(data, pd.Series):
        data = coerce_series(data, name="data")
    else:
        data = coerce_dataframe(data, name="data")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, **kwargs)


def read_parquet(
    path: str | Path,
    columns: list[str] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a Parquet file.

    Parquet is the recommended format for large financial datasets: it
    is columnar, compressed, and preserves dtypes (including datetime
    indices) without the ambiguity of CSV parsing.

    Parameters:
        path (str | Path): Path to the Parquet file.
        columns (list[str] | None): Subset of columns to read.
            ``None`` reads all columns.  Selecting a subset is much
            faster for wide datasets.
        **kwargs: Additional keyword arguments forwarded to
            :func:`pandas.read_parquet`.

    Returns:
        pd.DataFrame: DataFrame with the Parquet contents.

    Example:
        >>> df = read_parquet("prices.parquet")  # doctest: +SKIP
        >>> df = read_parquet("prices.parquet", columns=["close", "volume"])  # doctest: +SKIP

    See Also:
        write_parquet: Write a DataFrame to Parquet.
    """
    path = Path(path)
    return pd.read_parquet(path, columns=columns, **kwargs)


def write_parquet(
    data: pd.DataFrame,
    path: str | Path,
    **kwargs: Any,
) -> None:
    """Write a DataFrame to Parquet format.

    Creates parent directories automatically.  Parquet preserves dtypes
    exactly and is significantly faster to read/write than CSV for
    large datasets.

    Parameters:
        data (pd.DataFrame): DataFrame to write.
        path (str | Path): Destination file path.
        **kwargs: Additional keyword arguments forwarded to
            :meth:`pandas.DataFrame.to_parquet`.

    Example:
        >>> write_parquet(df, "output/prices.parquet")  # doctest: +SKIP

    See Also:
        read_parquet: Read a Parquet file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(path, **kwargs)


def read_hdf(
    path: str | Path,
    key: str = "data",
    **kwargs: Any,
) -> pd.DataFrame:
    """Read an HDF5 file.

    Requires the ``tables`` (PyTables) package to be installed.

    Parameters:
        path: Path to the HDF5 file.
        key: The group identifier in the HDF5 store.
        **kwargs: Additional keyword arguments forwarded to
            :func:`pandas.read_hdf`.

    Returns:
        DataFrame with the HDF5 contents.
    """
    path = Path(path)
    return pd.read_hdf(path, key=key, **kwargs)


def write_hdf(
    data: pd.DataFrame | pd.Series,
    path: str | Path,
    key: str = "data",
    **kwargs: Any,
) -> None:
    """Write a DataFrame or Series to HDF5 format.

    Requires the ``tables`` (PyTables) package to be installed.

    Parameters:
        data: Data to write.
        path: Destination file path.
        key: The group identifier in the HDF5 store.
        **kwargs: Additional keyword arguments forwarded to
            :meth:`pandas.DataFrame.to_hdf`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_hdf(path, key=key, **kwargs)


def read_excel(
    path: str | Path,
    sheet_name: str | int = 0,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read an Excel file.

    Parameters:
        path: Path to the Excel file (``.xlsx`` or ``.xls``).
        sheet_name: Name or index of the sheet to read. Defaults to the
            first sheet.
        **kwargs: Additional keyword arguments forwarded to
            :func:`pandas.read_excel`.

    Returns:
        DataFrame with the Excel sheet contents.
    """
    path = Path(path)
    return pd.read_excel(path, sheet_name=sheet_name, **kwargs)


def write_excel(
    data: pd.DataFrame,
    path: str | Path,
    sheet_name: str = "Sheet1",
    **kwargs: Any,
) -> None:
    """Write a DataFrame to Excel format.

    Parameters:
        data: DataFrame to write.
        path: Destination file path.
        sheet_name: Name of the worksheet.
        **kwargs: Additional keyword arguments forwarded to
            :meth:`pandas.DataFrame.to_excel`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_excel(path, sheet_name=sheet_name, **kwargs)
