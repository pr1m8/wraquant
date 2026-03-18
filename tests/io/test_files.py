"""Tests for wraquant.io.files — file format I/O with financial defaults."""

from __future__ import annotations

import pandas as pd
import pytest

from wraquant.io.files import read_csv, read_parquet, write_csv, write_parquet


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a small sample DataFrame with a Date index."""
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [101.0, 102.5, 103.0, 104.0, 105.0],
            "Low": [99.0, 100.5, 101.0, 102.0, 103.0],
            "Close": [100.5, 102.0, 102.5, 103.5, 104.5],
            "Volume": [1000, 1100, 1200, 1300, 1400],
        },
        index=dates,
    )


class TestCSV:
    """CSV read/write roundtrip tests."""

    def test_csv_roundtrip(self, tmp_path: object, sample_df: pd.DataFrame) -> None:
        """Write then read a CSV and verify data integrity."""
        csv_path = tmp_path / "test.csv"  # type: ignore[operator]
        sample_df.index.name = "Date"

        write_csv(sample_df, csv_path)
        result = read_csv(csv_path, date_column="Date")

        pd.testing.assert_frame_equal(
            result, sample_df, check_names=True, check_freq=False
        )

    def test_csv_roundtrip_no_date_parse(
        self, tmp_path: object, sample_df: pd.DataFrame
    ) -> None:
        """Roundtrip with parse_dates=False preserves raw data."""
        csv_path = tmp_path / "test_nodate.csv"  # type: ignore[operator]
        sample_df.index.name = "Date"

        write_csv(sample_df, csv_path)
        result = read_csv(csv_path, parse_dates=False)

        # Should have Date as a regular column (string), not as index
        assert "Date" in result.columns or result.index.name != "Date"

    def test_date_parsing_sets_index(
        self, tmp_path: object, sample_df: pd.DataFrame
    ) -> None:
        """Verify that read_csv sets the date column as the index."""
        csv_path = tmp_path / "test_dates.csv"  # type: ignore[operator]
        sample_df.index.name = "Date"

        write_csv(sample_df, csv_path)
        result = read_csv(csv_path, date_column="Date")

        assert result.index.name == "Date"
        assert pd.api.types.is_datetime64_any_dtype(result.index)

    def test_csv_creates_parent_dirs(
        self, tmp_path: object, sample_df: pd.DataFrame
    ) -> None:
        """write_csv should create intermediate directories."""
        nested_path = tmp_path / "a" / "b" / "test.csv"  # type: ignore[operator]
        sample_df.index.name = "Date"

        write_csv(sample_df, nested_path)
        assert nested_path.exists()


class TestParquet:
    """Parquet read/write roundtrip tests."""

    def test_parquet_roundtrip(self, tmp_path: object, sample_df: pd.DataFrame) -> None:
        """Write then read a Parquet file and verify data integrity."""
        pq_path = tmp_path / "test.parquet"  # type: ignore[operator]
        sample_df.index.name = "Date"

        write_parquet(sample_df, pq_path)
        result = read_parquet(pq_path)

        pd.testing.assert_frame_equal(result, sample_df, check_freq=False)

    def test_parquet_column_selection(
        self, tmp_path: object, sample_df: pd.DataFrame
    ) -> None:
        """read_parquet with columns should return only the requested subset."""
        pq_path = tmp_path / "test_cols.parquet"  # type: ignore[operator]
        sample_df.index.name = "Date"

        write_parquet(sample_df, pq_path)
        result = read_parquet(pq_path, columns=["Close", "Volume"])

        assert list(result.columns) == ["Close", "Volume"]
        assert len(result) == len(sample_df)

    def test_parquet_creates_parent_dirs(
        self, tmp_path: object, sample_df: pd.DataFrame
    ) -> None:
        """write_parquet should create intermediate directories."""
        nested_path = tmp_path / "x" / "y" / "test.parquet"  # type: ignore[operator]
        sample_df.index.name = "Date"

        write_parquet(sample_df, nested_path)
        assert nested_path.exists()
