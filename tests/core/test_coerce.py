"""Tests for the type coercion system (core/_coerce.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.core._coerce import (
    coerce_array,
    coerce_dataframe,
    coerce_returns,
    coerce_series,
)


# -----------------------------------------------------------------------
# coerce_array
# -----------------------------------------------------------------------


class TestCoerceArray:
    """Tests for coerce_array."""

    def test_from_ndarray(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = coerce_array(arr)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, arr)

    def test_from_ndarray_2d(self):
        arr = np.array([[1.0], [2.0], [3.0]])
        result = coerce_array(arr)
        assert result.ndim == 1
        assert len(result) == 3

    def test_from_int_ndarray(self):
        arr = np.array([1, 2, 3])
        result = coerce_array(arr)
        assert result.dtype == np.float64

    def test_from_series(self):
        s = pd.Series([10.0, 20.0, 30.0], index=["a", "b", "c"])
        result = coerce_array(s)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [10.0, 20.0, 30.0])

    def test_from_series_with_nan(self):
        s = pd.Series([1.0, np.nan, 3.0])
        result = coerce_array(s)
        assert np.isnan(result[1])

    def test_from_dataframe(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = coerce_array(df)
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_from_list(self):
        result = coerce_array([1, 2, 3])
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_from_tuple(self):
        result = coerce_array((4.0, 5.0, 6.0))
        np.testing.assert_array_equal(result, [4.0, 5.0, 6.0])

    def test_from_scalar(self):
        result = coerce_array(42.0)
        assert result.shape == (1,)
        assert result[0] == 42.0

    def test_error_on_invalid(self):
        with pytest.raises(TypeError, match="cannot be converted"):
            coerce_array({"a": 1}, name="test_input")

    def test_name_in_error_message(self):
        with pytest.raises(TypeError, match="my_data"):
            coerce_array(object(), name="my_data")


# -----------------------------------------------------------------------
# coerce_series
# -----------------------------------------------------------------------


class TestCoerceSeries:
    """Tests for coerce_series."""

    def test_from_series_preserves_index(self):
        idx = pd.date_range("2020-01-01", periods=3)
        s = pd.Series([1.0, 2.0, 3.0], index=idx, name="original")
        result = coerce_series(s)
        assert isinstance(result, pd.Series)
        pd.testing.assert_index_equal(result.index, idx)

    def test_from_ndarray(self):
        arr = np.array([10.0, 20.0, 30.0])
        result = coerce_series(arr, name="test")
        assert isinstance(result, pd.Series)
        assert result.name == "test"
        assert len(result) == 3

    def test_from_list(self):
        result = coerce_series([1, 2, 3], name="vals")
        assert isinstance(result, pd.Series)
        assert result.dtype == np.float64

    def test_from_dataframe_takes_first_col(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = coerce_series(df)
        np.testing.assert_array_equal(result.values, [1.0, 2.0])

    def test_series_cast_to_float(self):
        s = pd.Series([1, 2, 3], dtype=int)
        result = coerce_series(s)
        assert result.dtype == np.float64


# -----------------------------------------------------------------------
# coerce_returns
# -----------------------------------------------------------------------


class TestCoerceReturns:
    """Tests for coerce_returns."""

    def test_pass_through_returns(self):
        rets = np.array([0.01, -0.02, 0.005, -0.01])
        result = coerce_returns(rets)
        np.testing.assert_array_equal(result, rets)

    def test_auto_detect_prices(self):
        prices = np.array([100.0, 102.0, 101.0, 103.0])
        result = coerce_returns(prices)
        assert len(result) == 3
        expected = np.diff(prices) / prices[:-1]
        np.testing.assert_array_almost_equal(result, expected)

    def test_force_prices(self):
        prices = [100.0, 105.0, 103.0]
        result = coerce_returns(prices, is_prices=True)
        assert len(result) == 2

    def test_force_returns(self):
        data = [0.01, -0.02, 0.005]
        result = coerce_returns(data, is_prices=False)
        assert len(result) == 3

    def test_empty_after_nan_removal(self):
        data = np.array([np.nan, np.nan])
        result = coerce_returns(data)
        assert len(result) == 0

    def test_from_series(self):
        s = pd.Series([0.01, -0.005, 0.003])
        result = coerce_returns(s)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3


# -----------------------------------------------------------------------
# coerce_dataframe
# -----------------------------------------------------------------------


class TestCoerceDataFrame:
    """Tests for coerce_dataframe."""

    def test_from_dataframe(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = coerce_dataframe(df)
        assert isinstance(result, pd.DataFrame)
        assert result.dtypes["a"] == np.float64

    def test_from_dict(self):
        d = {"x": [1, 2, 3], "y": [4, 5, 6]}
        result = coerce_dataframe(d)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 2)

    def test_from_series(self):
        s = pd.Series([1.0, 2.0, 3.0], name="col")
        result = coerce_dataframe(s)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 1)

    def test_from_1d_ndarray(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = coerce_dataframe(arr, name="vals")
        assert isinstance(result, pd.DataFrame)
        assert "vals" in result.columns

    def test_from_2d_ndarray(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = coerce_dataframe(arr)
        assert result.shape == (2, 2)

    def test_error_on_invalid(self):
        with pytest.raises(TypeError, match="must be DataFrame"):
            coerce_dataframe("not a dataframe")


# -----------------------------------------------------------------------
# Integration: verify TA indicators accept numpy arrays
# -----------------------------------------------------------------------


class TestTACoercionIntegration:
    """Verify that TA validators now accept numpy arrays and lists."""

    def test_validate_series_accepts_ndarray(self):
        from wraquant.ta._validators import validate_series

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = validate_series(arr)
        assert isinstance(result, pd.Series)

    def test_validate_series_accepts_list(self):
        from wraquant.ta._validators import validate_series

        result = validate_series([10.0, 20.0, 30.0])
        assert isinstance(result, pd.Series)

    def test_validate_series_passthrough_series(self):
        from wraquant.ta._validators import validate_series

        s = pd.Series([1.0, 2.0, 3.0], name="test")
        result = validate_series(s)
        assert result is s  # Same object, not a copy


# -----------------------------------------------------------------------
# Integration: verify risk/metrics accepts numpy arrays
# -----------------------------------------------------------------------


class TestRiskMetricsCoercionIntegration:
    """Verify risk metrics now accept numpy arrays and lists."""

    def test_sharpe_ratio_with_ndarray(self):
        from wraquant.risk.metrics import sharpe_ratio

        returns = np.random.default_rng(42).normal(0.001, 0.01, 252)
        result = sharpe_ratio(returns)
        assert isinstance(result, float)

    def test_sharpe_ratio_with_list(self):
        from wraquant.risk.metrics import sharpe_ratio

        returns = [0.01, -0.005, 0.008, -0.003, 0.012, 0.002]
        result = sharpe_ratio(returns)
        assert isinstance(result, float)

    def test_max_drawdown_with_ndarray(self):
        from wraquant.risk.metrics import max_drawdown

        prices = np.array([100.0, 110.0, 105.0, 95.0, 108.0, 102.0])
        result = max_drawdown(prices)
        assert isinstance(result, float)
        assert result < 0

    def test_hit_ratio_with_ndarray(self):
        from wraquant.risk.metrics import hit_ratio

        returns = np.array([0.01, -0.005, 0.008, -0.003, 0.012])
        result = hit_ratio(returns)
        assert result == pytest.approx(0.6)


# -----------------------------------------------------------------------
# Integration: verify stats functions accept numpy arrays
# -----------------------------------------------------------------------


class TestStatsCoercionIntegration:
    """Verify stats functions now accept numpy arrays."""

    def test_summary_stats_with_ndarray(self):
        from wraquant.stats.descriptive import summary_stats

        returns = np.random.default_rng(42).normal(0, 0.01, 100)
        result = summary_stats(returns)
        assert "mean" in result
        assert "std" in result

    def test_annualized_return_with_list(self):
        from wraquant.stats.descriptive import annualized_return

        returns = [0.01, -0.005, 0.008, -0.003, 0.012] * 50
        result = annualized_return(returns)
        assert isinstance(result, float)
