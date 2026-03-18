"""Tests for wraquant.io.export — export and reporting utilities."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from wraquant.io.export import format_table, to_dict, to_json, to_tearsheet


@pytest.fixture
def sample_returns() -> pd.Series:
    """Daily return series for testing."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-02", periods=252)
    returns = pd.Series(
        np.random.normal(0.0005, 0.01, size=252),
        index=dates,
        name="returns",
    )
    return returns


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small DataFrame for export tests."""
    return pd.DataFrame(
        {
            "asset": ["AAPL", "GOOG", "MSFT"],
            "weight": [0.4, 0.35, 0.25],
            "return": [0.123, -0.045, 0.087],
        }
    )


class TestToJson:
    """Tests for to_json."""

    def test_returns_valid_json_string(self, sample_df: pd.DataFrame) -> None:
        """to_json should produce a valid JSON string."""
        result = to_json(sample_df)
        assert result is not None
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    def test_dict_input(self) -> None:
        """to_json handles plain dict input."""
        data = {"key": "value", "number": 42}
        result = to_json(data)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["key"] == "value"
        assert parsed["number"] == 42

    def test_series_input(self) -> None:
        """to_json handles Series input."""
        s = pd.Series([1, 2, 3], index=["a", "b", "c"], name="vals")
        result = to_json(s)
        assert result is not None
        parsed = json.loads(result)
        assert isinstance(parsed, (dict, list))

    def test_write_to_file(self, tmp_path: object, sample_df: pd.DataFrame) -> None:
        """to_json with path should write file and return None."""
        out_path = tmp_path / "out.json"  # type: ignore[operator]
        result = to_json(sample_df, path=out_path)
        assert result is None
        assert out_path.exists()

        content = json.loads(out_path.read_text())
        assert isinstance(content, list)


class TestToDict:
    """Tests for to_dict."""

    def test_dataframe_to_dict(self, sample_df: pd.DataFrame) -> None:
        """to_dict on a DataFrame returns a dict of dicts."""
        result = to_dict(sample_df)
        assert isinstance(result, dict)
        assert "asset" in result
        assert "weight" in result

    def test_series_to_dict(self) -> None:
        """to_dict on a Series returns index -> value mapping."""
        s = pd.Series([10, 20, 30], index=["a", "b", "c"])
        result = to_dict(s)
        assert result == {"a": 10, "b": 20, "c": 30}


class TestFormatTable:
    """Tests for format_table."""

    def test_returns_string(self, sample_df: pd.DataFrame) -> None:
        """format_table should return a string."""
        result = format_table(sample_df)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_precision(self) -> None:
        """format_table should respect the precision parameter."""
        df = pd.DataFrame({"value": [1.123456789]})
        result = format_table(df, precision=2)
        assert "1.12" in result

    def test_pct_columns(self) -> None:
        """format_table should format percentage columns."""
        df = pd.DataFrame({"ret": [0.1234], "price": [100.0]})
        result = format_table(df, precision=2, pct_columns=["ret"])
        assert "12.34%" in result


class TestToTearsheet:
    """Tests for to_tearsheet."""

    def test_returns_expected_keys(self, sample_returns: pd.Series) -> None:
        """to_tearsheet should include all core metric keys."""
        result = to_tearsheet(sample_returns)
        expected_keys = {
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "calmar_ratio",
            "n_periods",
        }
        assert expected_keys.issubset(result.keys())

    def test_return_types(self, sample_returns: pd.Series) -> None:
        """All metric values should be numeric."""
        result = to_tearsheet(sample_returns)
        for key, value in result.items():
            assert isinstance(value, (int, float)), f"{key} is {type(value)}"

    def test_with_benchmark(self, sample_returns: pd.Series) -> None:
        """to_tearsheet with benchmark adds correlation and info ratio."""
        np.random.seed(99)
        benchmark = pd.Series(
            np.random.normal(0.0003, 0.009, size=252),
            index=sample_returns.index,
            name="benchmark",
        )
        result = to_tearsheet(sample_returns, benchmark=benchmark)
        assert "benchmark_correlation" in result
        assert "information_ratio" in result

    def test_max_drawdown_negative(self, sample_returns: pd.Series) -> None:
        """Max drawdown should be non-positive."""
        result = to_tearsheet(sample_returns)
        assert result["max_drawdown"] <= 0

    def test_write_to_file(self, tmp_path: object, sample_returns: pd.Series) -> None:
        """to_tearsheet with output_path writes valid JSON."""
        out_path = tmp_path / "tearsheet.json"  # type: ignore[operator]
        result = to_tearsheet(sample_returns, output_path=out_path)
        assert out_path.exists()
        content = json.loads(out_path.read_text())
        assert content["total_return"] == result["total_return"]
