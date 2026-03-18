"""Tests for wraquant.frame.factory."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from wraquant.frame.factory import frame, series


class TestSeriesFactory:
    def test_from_list_pandas(self) -> None:
        s = series([1.0, 2.0, 3.0], name="test", backend="pandas")
        assert isinstance(s, pd.Series)
        assert s.name == "test"
        assert len(s) == 3

    def test_from_list_polars(self) -> None:
        s = series([1.0, 2.0, 3.0], name="test", backend="polars")
        assert isinstance(s, pl.Series)
        assert s.name == "test"
        assert len(s) == 3

    def test_from_pandas_series(self) -> None:
        ps = pd.Series([1, 2, 3], name="orig")
        result = series(ps, backend="pandas")
        assert isinstance(result, pd.Series)

    def test_from_polars_to_pandas(self) -> None:
        ps = pl.Series("x", [1, 2, 3])
        result = series(ps, backend="pandas")
        assert isinstance(result, pd.Series)

    def test_from_pandas_to_polars(self) -> None:
        ps = pd.Series([1, 2, 3], name="x")
        result = series(ps, backend="polars")
        assert isinstance(result, pl.Series)


class TestFrameFactory:
    def test_from_dict_pandas(self) -> None:
        df = frame({"a": [1, 2], "b": [3, 4]}, backend="pandas")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["a", "b"]

    def test_from_dict_polars(self) -> None:
        df = frame({"a": [1, 2], "b": [3, 4]}, backend="polars")
        assert isinstance(df, pl.DataFrame)
        assert df.columns == ["a", "b"]

    def test_from_numpy_pandas(self) -> None:
        arr = np.array([[1, 2], [3, 4]], dtype=float)
        df = frame(arr, columns=["x", "y"], backend="pandas")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["x", "y"]

    def test_from_numpy_polars(self) -> None:
        arr = np.array([[1, 2], [3, 4]], dtype=float)
        df = frame(arr, columns=["x", "y"], backend="polars")
        assert isinstance(df, pl.DataFrame)
        assert df.columns == ["x", "y"]

    def test_cross_backend_conversion(self) -> None:
        pdf = pd.DataFrame({"a": [1, 2]})
        pldf = frame(pdf, backend="polars")
        assert isinstance(pldf, pl.DataFrame)

        back = frame(pldf, backend="pandas")
        assert isinstance(back, pd.DataFrame)
