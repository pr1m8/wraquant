"""Data cleaning utilities for financial time series."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def remove_outliers(
    data: pd.DataFrame | pd.Series,
    method: Literal["zscore", "iqr", "mad"] = "zscore",
    threshold: float = 3.0,
) -> pd.DataFrame | pd.Series:
    """Remove rows containing outlier values from the data.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Input data with a DatetimeIndex.
    method : {'zscore', 'iqr', 'mad'}, default 'zscore'
        Outlier detection method.
    threshold : float, default 3.0
        Sensitivity threshold. For z-score and MAD this is the number of
        standard deviations; for IQR it is the multiplier applied to the
        interquartile range.

    Returns
    -------
    pd.DataFrame or pd.Series
        Data with outlier rows removed.
    """
    mask = detect_outliers(data, method=method, threshold=threshold)
    return data.loc[~mask]


def winsorize(
    data: pd.DataFrame | pd.Series,
    limits: tuple[float, float] = (0.01, 0.01),
) -> pd.DataFrame | pd.Series:
    """Clip extreme values at the given percentile limits.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Input data.
    limits : tuple of float, default (0.01, 0.01)
        Lower and upper percentile fractions to clip.  ``(0.01, 0.01)``
        clips the bottom 1 % and top 1 % of values.

    Returns
    -------
    pd.DataFrame or pd.Series
        Winsorized data with the same shape as the input.
    """
    lower_frac, upper_frac = limits

    if isinstance(data, pd.Series):
        lower = data.quantile(lower_frac)
        upper = data.quantile(1.0 - upper_frac)
        return data.clip(lower=lower, upper=upper)

    result = data.copy()
    for col in result.columns:
        lower = result[col].quantile(lower_frac)
        upper = result[col].quantile(1.0 - upper_frac)
        result[col] = result[col].clip(lower=lower, upper=upper)
    return result


def fill_missing(
    data: pd.DataFrame | pd.Series,
    method: Literal["ffill", "bfill", "interpolate", "drop"] = "ffill",
    limit: int | None = None,
) -> pd.DataFrame | pd.Series:
    """Fill or remove missing values.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Input data possibly containing NaN values.
    method : {'ffill', 'bfill', 'interpolate', 'drop'}, default 'ffill'
        Strategy for handling missing values.
    limit : int or None, default None
        Maximum number of consecutive NaN values to fill.  Only used
        with ``'ffill'``, ``'bfill'``, and ``'interpolate'``.

    Returns
    -------
    pd.DataFrame or pd.Series
        Data with missing values handled.
    """
    if method == "ffill":
        return data.ffill(limit=limit)
    if method == "bfill":
        return data.bfill(limit=limit)
    if method == "interpolate":
        return data.interpolate(limit=limit)
    if method == "drop":
        return data.dropna()
    raise ValueError(f"Unknown method: {method!r}")


def detect_outliers(
    data: pd.DataFrame | pd.Series,
    method: Literal["zscore", "iqr", "mad"] = "zscore",
    threshold: float = 3.0,
) -> pd.Series:
    """Flag rows that contain outlier values.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Input data.
    method : {'zscore', 'iqr', 'mad'}, default 'zscore'
        Detection method.
    threshold : float, default 3.0
        Sensitivity threshold.

    Returns
    -------
    pd.Series
        Boolean series with ``True`` for outlier rows.
    """
    if isinstance(data, pd.DataFrame):
        # A row is an outlier if *any* column is an outlier.
        flags = pd.DataFrame(
            {col: _detect_series(data[col], method, threshold) for col in data.columns}
        )
        return flags.any(axis=1)
    return _detect_series(data, method, threshold)


def _detect_series(
    s: pd.Series,
    method: str,
    threshold: float,
) -> pd.Series:
    """Detect outliers in a single series."""
    if method == "zscore":
        mean = s.mean()
        std = s.std()
        if std == 0 or np.isnan(std):
            return pd.Series(False, index=s.index)
        z = (s - mean).abs() / std
        return z > threshold

    if method == "iqr":
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return (s < lower) | (s > upper)

    if method == "mad":
        median = s.median()
        mad = (s - median).abs().median()
        if mad == 0:
            return pd.Series(False, index=s.index)
        modified_z = 0.6745 * (s - median) / mad
        return modified_z.abs() > threshold

    raise ValueError(f"Unknown method: {method!r}")


def handle_splits_dividends(
    prices: pd.Series,
    splits: pd.Series | None = None,
    dividends: pd.Series | None = None,
) -> pd.Series:
    """Adjust a price series for stock splits and dividends.

    Parameters
    ----------
    prices : pd.Series
        Raw (unadjusted) price series indexed by date.
    splits : pd.Series or None, default None
        Split ratios indexed by date.  A 2-for-1 split is represented as
        ``2.0``.  Dates not present in *prices* are ignored.
    dividends : pd.Series or None, default None
        Cash dividend amounts indexed by ex-date.

    Returns
    -------
    pd.Series
        Adjusted price series.
    """
    adjusted = prices.copy().astype(float)

    if splits is not None:
        # Walk backwards so that each adjustment accumulates.
        cumulative_split = 1.0
        for date in sorted(splits.index, reverse=True):
            if date in adjusted.index:
                cumulative_split *= splits[date]
                adjusted.loc[adjusted.index < date] /= cumulative_split

    if dividends is not None:
        for date in sorted(dividends.index, reverse=True):
            if date in adjusted.index:
                factor = 1.0 - dividends[date] / adjusted.loc[date]
                adjusted.loc[adjusted.index < date] *= factor

    return adjusted


def remove_duplicates(
    data: pd.DataFrame,
    keep: Literal["first", "last", False] = "last",
) -> pd.DataFrame:
    """Remove duplicate index entries.

    Parameters
    ----------
    data : pd.DataFrame
        Data whose index may contain duplicates.
    keep : {'first', 'last', False}, default 'last'
        Which duplicate to keep.

    Returns
    -------
    pd.DataFrame
        Data with unique index values.
    """
    return data[~data.index.duplicated(keep=keep)]


def align_series(
    *series: pd.Series,
    method: Literal["inner", "outer"] = "inner",
) -> tuple[pd.Series, ...]:
    """Align multiple series to a common index.

    Parameters
    ----------
    *series : pd.Series
        Two or more series to align.
    method : {'inner', 'outer'}, default 'inner'
        Join method.  ``'inner'`` keeps only dates present in all series;
        ``'outer'`` keeps all dates (filling gaps with NaN).

    Returns
    -------
    tuple of pd.Series
        Aligned series sharing the same index.
    """
    if len(series) < 2:
        raise ValueError("At least two series are required")

    combined_index: pd.Index = series[0].index
    for s in series[1:]:
        if method == "inner":
            combined_index = combined_index.intersection(s.index)
        else:
            combined_index = combined_index.union(s.index)

    combined_index = combined_index.sort_values()
    return tuple(s.reindex(combined_index) for s in series)


def resample_ohlcv(
    ohlcv: pd.DataFrame,
    freq: str = "W",
) -> pd.DataFrame:
    """Resample OHLCV data to a lower frequency.

    The aggregation follows standard financial conventions:

    * **open** -- first value in the period
    * **high** -- maximum value in the period
    * **low** -- minimum value in the period
    * **close** -- last value in the period
    * **volume** -- sum over the period

    Parameters
    ----------
    ohlcv : pd.DataFrame
        DataFrame with columns ``open``, ``high``, ``low``, ``close``, and
        ``volume`` (case-insensitive) indexed by date.
    freq : str, default 'W'
        Target frequency (any pandas offset alias).

    Returns
    -------
    pd.DataFrame
        Resampled OHLCV data.
    """
    # Normalise column names to lowercase for lookup.
    col_map: dict[str, str] = {}
    for col in ohlcv.columns:
        col_map[col.lower()] = col

    agg: dict[str, str] = {
        col_map["open"]: "first",
        col_map["high"]: "max",
        col_map["low"]: "min",
        col_map["close"]: "last",
        col_map["volume"]: "sum",
    }

    resampled = ohlcv.resample(freq).agg(agg)
    return resampled.dropna(how="all")
