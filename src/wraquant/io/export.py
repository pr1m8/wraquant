"""Export and reporting utilities.

Functions for converting financial data into various output formats
suitable for reporting, serialization, and display.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wraquant.core._coerce import coerce_series

__all__ = [
    "to_tearsheet",
    "to_json",
    "to_dict",
    "format_table",
]


def to_tearsheet(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a performance tearsheet from a return series.

    Computes the key performance and risk metrics that every portfolio
    analysis should include, and returns them as a serialisable
    dictionary.  Optionally writes the result to a JSON file for
    reporting or downstream consumption.

    Metrics computed: total return, annualized return, annualized
    volatility, Sharpe ratio, maximum drawdown, and Calmar ratio.
    When a benchmark is provided, also computes correlation and
    information ratio.

    Parameters:
        returns (pd.Series): Series of portfolio returns (simple, not
            cumulative), indexed by datetime.
        benchmark (pd.Series | None): Optional benchmark return series
            for relative metrics.  When provided, the two series are
            aligned by index.
        output_path (str | Path | None): If provided, write the
            tearsheet dict to this JSON file.

    Returns:
        dict[str, Any]: Dictionary with keys ``total_return``,
            ``annualized_return``, ``annualized_volatility``,
            ``sharpe_ratio``, ``max_drawdown``, ``calmar_ratio``,
            ``n_periods``, and optionally ``benchmark_correlation``
            and ``information_ratio``.

    Example:
        >>> import pandas as pd, numpy as np
        >>> returns = pd.Series(np.random.randn(252) * 0.01)
        >>> sheet = to_tearsheet(returns)
        >>> "sharpe_ratio" in sheet
        True

    See Also:
        to_json: Serialize any data to JSON.
        format_table: Pretty-print a DataFrame.
    """
    returns = coerce_series(returns, name="returns").dropna()
    n_periods = len(returns)

    # Assume 252 trading days per year for annualization
    trading_days = 252

    total_return = float((1 + returns).prod() - 1)

    n_years = n_periods / trading_days
    annualized_return = (
        float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0
    )

    annualized_vol = float(returns.std() * np.sqrt(trading_days))

    sharpe = annualized_return / annualized_vol if annualized_vol != 0 else 0.0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(drawdown.min())

    calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    result: dict[str, Any] = {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "n_periods": n_periods,
    }

    if benchmark is not None:
        benchmark = coerce_series(benchmark, name="benchmark").dropna()
        # Align the two series
        aligned_returns, aligned_bench = returns.align(benchmark, join="inner")
        if len(aligned_returns) > 1:
            result["benchmark_correlation"] = float(aligned_returns.corr(aligned_bench))
            excess = aligned_returns - aligned_bench
            tracking_error = float(excess.std() * np.sqrt(trading_days))
            info_ratio = (
                float(excess.mean() * trading_days) / tracking_error
                if tracking_error != 0
                else 0.0
            )
            result["information_ratio"] = info_ratio

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, default=str))

    return result


def to_json(
    data: pd.DataFrame | pd.Series | dict[str, Any],
    path: str | Path | None = None,
    orient: str = "records",
) -> str | None:
    """Export data to JSON format.

    Handles DataFrames, Series, and plain dictionaries.  When a file
    path is provided, the JSON is written to disk; otherwise the JSON
    string is returned for further use (e.g., sending via an API).

    Parameters:
        data (pd.DataFrame | pd.Series | dict): Data to serialize.
            DataFrames and Series use the pandas JSON serializer; plain
            dicts use the stdlib ``json`` module.
        path (str | Path | None): If provided, write the JSON string
            to this file and return ``None``.  Otherwise, return the
            JSON string.
        orient (str): Orientation for :meth:`pandas.DataFrame.to_json`
            (e.g., ``'records'``, ``'index'``, ``'columns'``).

    Returns:
        str | None: JSON string when *path* is ``None``; otherwise
            ``None``.

    Example:
        >>> json_str = to_json({"sharpe": 1.2, "max_dd": -0.15})
        >>> isinstance(json_str, str)
        True

    See Also:
        to_dict: Convert to a nested dictionary.
        to_tearsheet: Generate a full performance report.
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        json_str = data.to_json(orient=orient, date_format="iso", indent=2)
    else:
        json_str = json.dumps(data, indent=2, default=str)

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json_str)
        return None

    return json_str


def to_dict(
    data: pd.DataFrame | pd.Series,
) -> dict[str, Any]:
    """Convert a DataFrame or Series to a nested dictionary.

    For a DataFrame, produces ``{column: {index: value, ...}, ...}``.
    For a Series, produces ``{index: value, ...}``.  Useful for
    serialization, API responses, or interop with non-pandas code.

    Parameters:
        data (pd.DataFrame | pd.Series): DataFrame or Series to
            convert.

    Returns:
        dict[str, Any]: Nested dictionary representation of the data.

    Example:
        >>> import pandas as pd
        >>> s = pd.Series([1, 2, 3], index=["a", "b", "c"])
        >>> to_dict(s)
        {'a': 1, 'b': 2, 'c': 3}

    See Also:
        to_json: Serialize to JSON string.
    """
    if isinstance(data, pd.Series):
        return data.to_dict()
    return data.to_dict()


def format_table(
    data: pd.DataFrame,
    precision: int = 4,
    pct_columns: list[str] | None = None,
) -> str:
    """Format a DataFrame as a pretty-printed table string.

    Produces a human-readable text table suitable for console output,
    log files, or email reports.  Numeric columns are formatted to a
    fixed number of decimal places, and designated columns are
    displayed as percentages.

    Parameters:
        data (pd.DataFrame): DataFrame to format.
        precision (int): Number of decimal places for numeric columns
            (default 4).
        pct_columns (list[str] | None): Column names to format as
            percentages (values are multiplied by 100 and suffixed
            with ``%``).

    Returns:
        str: String representation of the formatted table.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"return": [0.05], "vol": [0.15]})
        >>> print(format_table(df, pct_columns=["return", "vol"]))
        ...
    """
    formatted = data.copy()

    if pct_columns is not None:
        for col in pct_columns:
            if col in formatted.columns:
                formatted[col] = formatted[col].apply(
                    lambda x: f"{x * 100:.{precision}f}%" if pd.notna(x) else ""
                )

    # Format remaining numeric columns
    for col in formatted.columns:
        if col in (pct_columns or []):
            continue
        if pd.api.types.is_numeric_dtype(formatted[col]):
            formatted[col] = formatted[col].apply(
                lambda x: f"{x:.{precision}f}" if pd.notna(x) else ""
            )

    return formatted.to_string()
