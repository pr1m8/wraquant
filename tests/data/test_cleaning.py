"""Tests for wraquant.data.cleaning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.data.cleaning import (
    align_series,
    fill_missing,
    remove_duplicates,
    remove_outliers,
    resample_ohlcv,
    winsorize,
)


@pytest.fixture()
def sample_series() -> pd.Series:
    """A simple numeric series with one extreme value."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=100)
    values = rng.normal(loc=0, scale=1, size=100)
    # Inject a clear outlier.
    values[50] = 100.0
    return pd.Series(values, index=dates, name="price")


@pytest.fixture()
def ohlcv_df() -> pd.DataFrame:
    """Daily OHLCV data for two weeks."""
    dates = pd.bdate_range("2024-01-02", periods=10)
    rng = np.random.default_rng(7)
    close = 100.0 + rng.normal(0, 1, size=10).cumsum()
    high = close + rng.uniform(0.5, 2, size=10)
    low = close - rng.uniform(0.5, 2, size=10)
    open_ = close + rng.uniform(-1, 1, size=10)
    volume = rng.integers(1_000, 10_000, size=10)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class TestRemoveOutliers:
    def test_removes_extreme_value(self, sample_series: pd.Series) -> None:
        cleaned = remove_outliers(sample_series, method="zscore", threshold=3.0)
        assert len(cleaned) < len(sample_series)
        assert 100.0 not in cleaned.values

    def test_iqr_method(self, sample_series: pd.Series) -> None:
        cleaned = remove_outliers(sample_series, method="iqr", threshold=1.5)
        assert 100.0 not in cleaned.values

    def test_mad_method(self, sample_series: pd.Series) -> None:
        cleaned = remove_outliers(sample_series, method="mad", threshold=3.0)
        assert 100.0 not in cleaned.values

    def test_dataframe_input(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=50)
        df = pd.DataFrame(
            {"a": np.zeros(50), "b": np.zeros(50)},
            index=dates,
        )
        df.iloc[10, 0] = 100.0
        cleaned = remove_outliers(df, method="zscore", threshold=3.0)
        assert len(cleaned) < 50


class TestWinsorize:
    def test_clips_at_percentiles(self, sample_series: pd.Series) -> None:
        result = winsorize(sample_series, limits=(0.05, 0.05))
        lower = sample_series.quantile(0.05)
        upper = sample_series.quantile(0.95)
        assert result.min() >= lower
        assert result.max() <= upper

    def test_dataframe(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=100)
        df = pd.DataFrame(
            {"a": np.arange(100, dtype=float), "b": np.arange(100, dtype=float)},
            index=dates,
        )
        result = winsorize(df, limits=(0.1, 0.1))
        assert result["a"].min() >= df["a"].quantile(0.1)
        assert result["a"].max() <= df["a"].quantile(0.9)


class TestFillMissing:
    def test_ffill(self) -> None:
        s = pd.Series([1.0, np.nan, np.nan, 4.0])
        result = fill_missing(s, method="ffill")
        assert not result.isna().any()
        assert result.iloc[1] == 1.0

    def test_bfill(self) -> None:
        s = pd.Series([np.nan, np.nan, 3.0, 4.0])
        result = fill_missing(s, method="bfill")
        assert result.iloc[0] == 3.0

    def test_interpolate(self) -> None:
        s = pd.Series([1.0, np.nan, 3.0])
        result = fill_missing(s, method="interpolate")
        assert abs(result.iloc[1] - 2.0) < 1e-10

    def test_drop(self) -> None:
        s = pd.Series([1.0, np.nan, 3.0])
        result = fill_missing(s, method="drop")
        assert len(result) == 2

    def test_ffill_with_limit(self) -> None:
        s = pd.Series([1.0, np.nan, np.nan, np.nan, 5.0])
        result = fill_missing(s, method="ffill", limit=1)
        assert result.iloc[1] == 1.0
        assert pd.isna(result.iloc[2])


class TestAlignSeries:
    def test_inner_join(self) -> None:
        idx1 = pd.date_range("2020-01-01", periods=5, freq="D")
        idx2 = pd.date_range("2020-01-03", periods=5, freq="D")
        s1 = pd.Series(range(5), index=idx1, dtype=float)
        s2 = pd.Series(range(5), index=idx2, dtype=float)
        a1, a2 = align_series(s1, s2, method="inner")
        assert len(a1) == len(a2)
        # Inner join of these ranges: Jan 3, 4, 5 = 3 days
        assert len(a1) == 3

    def test_outer_join(self) -> None:
        idx1 = pd.date_range("2020-01-01", periods=3, freq="D")
        idx2 = pd.date_range("2020-01-03", periods=3, freq="D")
        s1 = pd.Series([1.0, 2.0, 3.0], index=idx1)
        s2 = pd.Series([4.0, 5.0, 6.0], index=idx2)
        a1, a2 = align_series(s1, s2, method="outer")
        # Union: Jan 1..5 = 5 days
        assert len(a1) == 5
        assert pd.isna(a1.iloc[-1])

    def test_requires_two_series(self) -> None:
        s = pd.Series([1.0, 2.0])
        with pytest.raises(ValueError):
            align_series(s)


class TestResampleOhlcv:
    def test_aggregation_rules(self, ohlcv_df: pd.DataFrame) -> None:
        weekly = resample_ohlcv(ohlcv_df, freq="W")
        assert len(weekly) < len(ohlcv_df)

        # For the first week grab the original rows.
        first_week_end = weekly.index[0]
        mask = ohlcv_df.index <= first_week_end
        original_week = ohlcv_df.loc[mask]

        assert weekly["open"].iloc[0] == original_week["open"].iloc[0]
        assert weekly["high"].iloc[0] == original_week["high"].max()
        assert weekly["low"].iloc[0] == original_week["low"].min()
        assert weekly["close"].iloc[0] == original_week["close"].iloc[-1]
        assert weekly["volume"].iloc[0] == original_week["volume"].sum()


class TestRemoveDuplicates:
    def test_removes_duplicates(self) -> None:
        idx = pd.DatetimeIndex(["2020-01-01", "2020-01-01", "2020-01-02"])
        df = pd.DataFrame({"a": [1, 2, 3]}, index=idx)
        result = remove_duplicates(df, keep="last")
        assert len(result) == 2
        # 'last' keeps the second occurrence of Jan 1
        assert result.loc["2020-01-01", "a"] == 2

    def test_keep_first(self) -> None:
        idx = pd.DatetimeIndex(["2020-01-01", "2020-01-01", "2020-01-02"])
        df = pd.DataFrame({"a": [1, 2, 3]}, index=idx)
        result = remove_duplicates(df, keep="first")
        assert result.loc["2020-01-01", "a"] == 1
