"""Tests for advanced validation integrations."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

_has_pandera = importlib.util.find_spec("pandera") is not None


class TestPanderaValidate:
    @pytest.mark.skipif(not _has_pandera, reason="pandera not installed")
    def test_valid_dataframe_passes(self) -> None:
        import pandera as pa

        from wraquant.data.validation_advanced import pandera_validate

        schema = pa.DataFrameSchema({
            "x": pa.Column(float, pa.Check.gt(0)),
        })
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = pandera_validate(df, schema)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    @pytest.mark.skipif(not _has_pandera, reason="pandera not installed")
    def test_invalid_dataframe_raises(self) -> None:
        import pandera as pa

        from wraquant.data.validation_advanced import pandera_validate

        schema = pa.DataFrameSchema({
            "x": pa.Column(float, pa.Check.gt(0)),
        })
        df = pd.DataFrame({"x": [-1.0, 2.0]})
        with pytest.raises(pa.errors.SchemaError):
            pandera_validate(df, schema)


class TestCreateOhlcvSchema:
    @pytest.mark.skipif(not _has_pandera, reason="pandera not installed")
    def test_valid_ohlcv(self) -> None:
        from wraquant.data.validation_advanced import create_ohlcv_schema, pandera_validate

        schema = create_ohlcv_schema()
        df = pd.DataFrame({
            "open": [100.0, 101.0],
            "high": [105.0, 106.0],
            "low": [99.0, 100.0],
            "close": [103.0, 104.0],
            "volume": [1000.0, 2000.0],
        })
        result = pandera_validate(df, schema)
        assert len(result) == 2

    @pytest.mark.skipif(not _has_pandera, reason="pandera not installed")
    def test_negative_price_fails(self) -> None:
        import pandera as pa

        from wraquant.data.validation_advanced import create_ohlcv_schema, pandera_validate

        schema = create_ohlcv_schema()
        df = pd.DataFrame({
            "open": [-1.0],
            "high": [5.0],
            "low": [1.0],
            "close": [3.0],
            "volume": [100.0],
        })
        with pytest.raises(pa.errors.SchemaError):
            pandera_validate(df, schema)

    @pytest.mark.skipif(not _has_pandera, reason="pandera not installed")
    def test_high_lt_low_fails(self) -> None:
        import pandera as pa

        from wraquant.data.validation_advanced import create_ohlcv_schema, pandera_validate

        schema = create_ohlcv_schema()
        df = pd.DataFrame({
            "open": [100.0],
            "high": [98.0],  # less than low
            "low": [99.0],
            "close": [98.5],
            "volume": [100.0],
        })
        with pytest.raises(pa.errors.SchemaError):
            pandera_validate(df, schema)

    @pytest.mark.skipif(not _has_pandera, reason="pandera not installed")
    def test_negative_volume_fails(self) -> None:
        import pandera as pa

        from wraquant.data.validation_advanced import create_ohlcv_schema, pandera_validate

        schema = create_ohlcv_schema()
        df = pd.DataFrame({
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [103.0],
            "volume": [-10.0],
        })
        with pytest.raises(pa.errors.SchemaError):
            pandera_validate(df, schema)


class TestCreateReturnsSchema:
    @pytest.mark.skipif(not _has_pandera, reason="pandera not installed")
    def test_valid_returns(self) -> None:
        from wraquant.data.validation_advanced import (
            create_returns_schema,
            pandera_validate,
        )

        schema = create_returns_schema(max_abs_return=0.5)
        df = pd.DataFrame({"returns": [0.01, -0.02, 0.05, -0.1]})
        result = pandera_validate(df, schema)
        assert len(result) == 4

    @pytest.mark.skipif(not _has_pandera, reason="pandera not installed")
    def test_extreme_return_fails(self) -> None:
        import pandera as pa

        from wraquant.data.validation_advanced import (
            create_returns_schema,
            pandera_validate,
        )

        schema = create_returns_schema(max_abs_return=0.5)
        df = pd.DataFrame({"returns": [0.01, 0.9]})  # 0.9 > 0.5
        with pytest.raises(pa.errors.SchemaError):
            pandera_validate(df, schema)

    @pytest.mark.skipif(not _has_pandera, reason="pandera not installed")
    def test_nan_not_allowed_by_default(self) -> None:
        import pandera as pa

        from wraquant.data.validation_advanced import (
            create_returns_schema,
            pandera_validate,
        )

        schema = create_returns_schema(allow_nan=False)
        df = pd.DataFrame({"returns": [0.01, np.nan]})
        with pytest.raises(pa.errors.SchemaError):
            pandera_validate(df, schema)

    @pytest.mark.skipif(not _has_pandera, reason="pandera not installed")
    def test_nan_allowed_when_enabled(self) -> None:
        from wraquant.data.validation_advanced import (
            create_returns_schema,
            pandera_validate,
        )

        schema = create_returns_schema(allow_nan=True)
        df = pd.DataFrame({"returns": [0.01, np.nan]})
        result = pandera_validate(df, schema)
        assert len(result) == 2
