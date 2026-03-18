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
    """Generate performance tearsheet data from a return series.

    Computes key performance and risk metrics and returns them as a
    dictionary.  Optionally writes the result to a JSON file.

    Parameters:
        returns: Series of portfolio returns (simple, not cumulative),
            indexed by datetime.
        benchmark: Optional benchmark return series for relative
            metrics.
        output_path: If provided, write the tearsheet dict to this
            JSON file.

    Returns:
        Dictionary with keys such as ``total_return``,
        ``annualized_return``, ``annualized_volatility``,
        ``sharpe_ratio``, ``max_drawdown``, ``calmar_ratio``, and
        optionally ``benchmark_correlation`` and
        ``information_ratio``.
    """
    returns = returns.dropna()
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
        benchmark = benchmark.dropna()
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

    Parameters:
        data: Data to serialize.  DataFrames and Series use the pandas
            JSON serializer; plain dicts use the stdlib ``json`` module.
        path: If provided, write the JSON string to this file and
            return ``None``.  Otherwise, return the JSON string.
        orient: Orientation for :meth:`pandas.DataFrame.to_json`
            (e.g., ``'records'``, ``'index'``, ``'columns'``).

    Returns:
        JSON string when *path* is ``None``; otherwise ``None``.
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
    For a Series, produces ``{index: value, ...}``.

    Parameters:
        data: DataFrame or Series to convert.

    Returns:
        Nested dictionary representation of the data.
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

    Parameters:
        data: DataFrame to format.
        precision: Number of decimal places for numeric columns.
        pct_columns: Column names to format as percentages (multiplied
            by 100 and suffixed with ``%``).

    Returns:
        String representation of the formatted table.
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
