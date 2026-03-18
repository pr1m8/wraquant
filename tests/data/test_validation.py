"""Tests for wraquant.data.validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.data.validation import (
    check_completeness,
    data_quality_report,
    validate_ohlcv,
    validate_returns,
)


@pytest.fixture()
def good_ohlcv() -> pd.DataFrame:
    """Well-formed OHLCV data."""
    dates = pd.bdate_range("2024-01-02", periods=10)
    return pd.DataFrame(
        {
            "open": [100.0] * 10,
            "high": [105.0] * 10,
            "low": [95.0] * 10,
            "close": [102.0] * 10,
            "volume": [1000] * 10,
        },
        index=dates,
    )


@pytest.fixture()
def bad_ohlcv() -> pd.DataFrame:
    """OHLCV data with known issues."""
    dates = pd.bdate_range("2024-01-02", periods=5)
    return pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [90.0, 105.0, 105.0, 105.0, 105.0],  # row 0: high < low
            "low": [95.0, 95.0, 95.0, 95.0, 95.0],
            "close": [102.0, 102.0, 102.0, 102.0, 102.0],
            "volume": [1000, -500, 1000, 1000, 1000],  # row 1: negative volume
        },
        index=dates,
    )


class TestValidateOhlcv:
    def test_catches_high_lt_low(self, bad_ohlcv: pd.DataFrame) -> None:
        report = validate_ohlcv(bad_ohlcv)
        assert len(report["high_lt_low"]) >= 1

    def test_catches_negative_volume(self, bad_ohlcv: pd.DataFrame) -> None:
        report = validate_ohlcv(bad_ohlcv)
        assert len(report["negative_volume"]) >= 1

    def test_good_data_has_no_issues(self, good_ohlcv: pd.DataFrame) -> None:
        report = validate_ohlcv(good_ohlcv)
        assert len(report["high_lt_low"]) == 0
        assert len(report["negative_volume"]) == 0
        assert len(report["close_outside_range"]) == 0

    def test_detects_close_outside_range(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=3)
        df = pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0],
                "high": [105.0, 105.0, 105.0],
                "low": [95.0, 95.0, 95.0],
                "close": [110.0, 102.0, 90.0],  # rows 0 and 2 outside [low, high]
                "volume": [1000, 1000, 1000],
            },
            index=dates,
        )
        report = validate_ohlcv(df)
        assert len(report["close_outside_range"]) == 2


class TestValidateReturns:
    def test_flags_large_returns(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=5)
        rets = pd.Series([0.01, 0.02, 0.8, -0.6, 0.01], index=dates)
        report = validate_returns(rets, max_abs=0.5)
        assert len(report["suspicious"]) == 2

    def test_detects_nan(self) -> None:
        rets = pd.Series([0.01, np.nan, 0.02])
        report = validate_returns(rets)
        assert report["has_nan"] is True
        assert report["nan_count"] == 1


class TestCheckCompleteness:
    def test_known_gaps(self) -> None:
        # Create business-day data with a gap.
        all_dates = pd.bdate_range("2024-01-02", periods=10)
        # Remove two dates to create gaps.
        kept = all_dates[[0, 1, 2, 3, 6, 7, 8, 9]]
        data = pd.Series(range(len(kept)), index=kept, dtype=float)
        report = check_completeness(data, expected_freq="B")
        assert report["missing_count"] == 2
        assert report["completeness_pct"] < 100.0

    def test_complete_data(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=10)
        data = pd.Series(range(10), index=dates, dtype=float)
        report = check_completeness(data, expected_freq="B")
        assert report["missing_count"] == 0
        assert report["completeness_pct"] == pytest.approx(100.0)


class TestDataQualityReport:
    def test_returns_expected_keys(self, good_ohlcv: pd.DataFrame) -> None:
        report = data_quality_report(good_ohlcv, freq="B")
        expected_keys = {
            "completeness",
            "staleness",
            "missing_values",
            "duplicated_dates",
            "date_range",
            "shape",
            "dtypes",
        }
        assert expected_keys == set(report.keys())

    def test_shape_matches(self, good_ohlcv: pd.DataFrame) -> None:
        report = data_quality_report(good_ohlcv)
        assert report["shape"] == good_ohlcv.shape
