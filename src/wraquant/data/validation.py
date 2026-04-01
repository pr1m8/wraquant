"""Data quality checks and validation for financial time series."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wraquant.core._coerce import coerce_dataframe, coerce_series


def validate_ohlcv(df: pd.DataFrame) -> dict[str, Any]:
    """Validate OHLCV data for common issues.

    Checks performed:

    * **high_lt_low** -- rows where *high* < *low*
    * **close_outside_range** -- rows where *close* is outside [*low*, *high*]
    * **negative_volume** -- rows with negative volume
    * **missing_values** -- count of NaN values per column
    * **gaps** -- missing business days in the index

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``open``, ``high``, ``low``, ``close``, and
        ``volume`` (case-insensitive).

    Returns
    -------
    dict
        Dictionary keyed by check name with details of any issues found.
    """
    df = coerce_dataframe(df, name="ohlcv")
    cols = {c.lower(): c for c in df.columns}

    high = df[cols["high"]]
    low = df[cols["low"]]
    close = df[cols["close"]]
    volume = df[cols["volume"]]

    high_lt_low_mask = high < low
    close_outside = (close < low) | (close > high)
    neg_volume = volume < 0

    # Detect gaps in business-day index.
    if isinstance(df.index, pd.DatetimeIndex):
        expected = pd.bdate_range(df.index.min(), df.index.max())
        missing_dates = expected.difference(df.index)
    else:
        missing_dates = pd.DatetimeIndex([])

    return {
        "high_lt_low": df.index[high_lt_low_mask].tolist(),
        "close_outside_range": df.index[close_outside].tolist(),
        "negative_volume": df.index[neg_volume].tolist(),
        "missing_values": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
        "gaps": missing_dates.tolist(),
    }


def validate_returns(
    returns: pd.Series | pd.DataFrame,
    max_abs: float = 0.5,
) -> dict[str, Any]:
    """Validate a return series for suspicious values.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Return series (simple or log).
    max_abs : float, default 0.5
        Returns with absolute value greater than this are flagged.

    Returns
    -------
    dict
        Dictionary containing:

        * **suspicious** -- indices where |return| > *max_abs*
        * **has_nan** -- whether any NaN values exist
        * **nan_count** -- number of NaN values
        * **min** -- minimum return value
        * **max** -- maximum return value
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        returns = coerce_series(returns, name="returns")
    if isinstance(returns, pd.DataFrame):
        flat = returns.stack()
    else:
        flat = returns

    suspicious_mask = flat.abs() > max_abs

    return {
        "suspicious": flat.index[suspicious_mask].tolist(),
        "has_nan": bool(flat.isna().any()),
        "nan_count": int(flat.isna().sum()),
        "min": float(flat.min()) if len(flat) > 0 else np.nan,
        "max": float(flat.max()) if len(flat) > 0 else np.nan,
    }


def check_completeness(
    data: pd.Series | pd.DataFrame,
    expected_freq: str = "B",
) -> dict[str, Any]:
    """Report on data completeness relative to an expected frequency.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Time-series data with a DatetimeIndex.
    expected_freq : str, default 'B'
        Expected frequency (e.g. ``'B'`` for business days, ``'D'`` for
        calendar days).

    Returns
    -------
    dict
        Dictionary containing:

        * **expected_count** -- number of expected periods
        * **actual_count** -- number of actual observations
        * **missing_count** -- number of missing periods
        * **missing_dates** -- list of missing dates
        * **completeness_pct** -- percentage of expected dates present
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Data must have a DatetimeIndex")

    expected_index = pd.date_range(
        start=data.index.min(),
        end=data.index.max(),
        freq=expected_freq,
    )

    missing = expected_index.difference(data.index)
    expected_count = len(expected_index)
    actual_count = len(data.index.intersection(expected_index))

    completeness = (
        (actual_count / expected_count * 100.0) if expected_count > 0 else 100.0
    )

    return {
        "expected_count": expected_count,
        "actual_count": actual_count,
        "missing_count": len(missing),
        "missing_dates": missing.tolist(),
        "completeness_pct": completeness,
    }


def check_staleness(
    data: pd.Series | pd.DataFrame,
    max_unchanged: int = 5,
) -> dict[str, Any]:
    """Detect stale (stuck/unchanged) values in a time series.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Time-series data.
    max_unchanged : int, default 5
        Number of consecutive identical values before flagging as stale.

    Returns
    -------
    dict
        Dictionary containing:

        * **stale_periods** -- list of ``(start, end, length)`` tuples for
          each run of identical values exceeding *max_unchanged*.
        * **total_stale_rows** -- total number of rows within stale periods.
    """
    if isinstance(data, pd.DataFrame):
        # Collapse to a single "changed" flag: any column changed.
        changed = data.ne(data.shift()).any(axis=1)
    else:
        changed = data.ne(data.shift())

    # Group consecutive unchanged values.
    groups = changed.cumsum()
    stale_periods: list[tuple[Any, Any, int]] = []
    total_stale = 0

    for _, group in data.groupby(groups):
        run_length = len(group)
        if run_length > max_unchanged:
            start = group.index[0]
            end = group.index[-1]
            stale_periods.append((start, end, run_length))
            total_stale += run_length

    return {
        "stale_periods": stale_periods,
        "total_stale_rows": total_stale,
    }


def data_quality_report(
    data: pd.DataFrame,
    freq: str = "B",
) -> dict[str, Any]:
    """Generate a comprehensive data quality report.

    Combines completeness, staleness, and value-range checks into a
    single report dictionary.

    Parameters
    ----------
    data : pd.DataFrame
        Time-series data with a DatetimeIndex.
    freq : str, default 'B'
        Expected frequency for completeness checking.

    Returns
    -------
    dict
        Dictionary containing:

        * **completeness** -- output of :func:`check_completeness`
        * **staleness** -- output of :func:`check_staleness`
        * **missing_values** -- NaN counts per column
        * **duplicated_dates** -- number of duplicate index entries
        * **date_range** -- ``(first_date, last_date)``
        * **shape** -- ``(rows, cols)``
        * **dtypes** -- column data types
    """
    completeness = check_completeness(data, expected_freq=freq)
    staleness = check_staleness(data)

    duplicated_count = int(data.index.duplicated().sum())

    return {
        "completeness": completeness,
        "staleness": staleness,
        "missing_values": {k: int(v) for k, v in data.isna().sum().to_dict().items()},
        "duplicated_dates": duplicated_count,
        "date_range": (data.index.min(), data.index.max()),
        "shape": data.shape,
        "dtypes": data.dtypes.to_dict(),
    }
