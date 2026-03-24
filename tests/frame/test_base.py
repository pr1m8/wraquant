"""Tests for frame/base.py financial types.

Covers PriceSeries, ReturnSeries, OHLCVFrame, ReturnFrame creation,
frequency detection, roundtrip conversions, and financial analytics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.frame.base import (
    OHLCVFrame,
    PriceSeries,
    ReturnFrame,
    ReturnSeries,
    _detect_frequency,
)


# ---------------------------------------------------------------------------
# Frequency detection
# ---------------------------------------------------------------------------


class TestDetectFrequency:
    def test_daily_bday(self):
        idx = pd.bdate_range("2023-01-02", periods=100)
        label, ppy = _detect_frequency(idx)
        assert label == "daily"
        assert ppy == 252

    def test_weekly(self):
        idx = pd.date_range("2023-01-02", periods=52, freq="W")
        label, ppy = _detect_frequency(idx)
        assert label == "weekly"
        assert ppy == 52

    def test_monthly(self):
        idx = pd.date_range("2023-01-31", periods=24, freq="ME")
        label, ppy = _detect_frequency(idx)
        assert label == "monthly"
        assert ppy == 12

    def test_short_index_defaults(self):
        idx = pd.DatetimeIndex(["2023-01-01", "2023-01-02"])
        label, ppy = _detect_frequency(idx)
        assert label == "unknown"
        assert ppy == 252

    def test_non_datetime_defaults(self):
        idx = pd.RangeIndex(10)
        label, ppy = _detect_frequency(idx)
        assert label == "unknown"
        assert ppy == 252


# ---------------------------------------------------------------------------
# PriceSeries
# ---------------------------------------------------------------------------


class TestPriceSeries:
    @pytest.fixture()
    def daily_prices(self):
        idx = pd.bdate_range("2023-01-02", periods=100)
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, 100)))
        return PriceSeries(prices, index=idx, name="SPY")

    def test_creation(self, daily_prices):
        assert isinstance(daily_prices, pd.Series)
        assert isinstance(daily_prices, PriceSeries)
        assert len(daily_prices) == 100

    def test_frequency_detection(self, daily_prices):
        assert daily_prices.frequency == "daily"
        assert daily_prices.periods_per_year == 252

    def test_explicit_frequency(self):
        p = PriceSeries([100, 101, 102], frequency="monthly")
        assert p.frequency == "monthly"
        assert p.periods_per_year == 12

    def test_currency(self, daily_prices):
        assert daily_prices.currency == "USD"
        p = PriceSeries([100, 101], currency="EUR")
        assert p.currency == "EUR"

    def test_to_returns_simple(self, daily_prices):
        r = daily_prices.to_returns(method="simple")
        assert isinstance(r, ReturnSeries)
        assert len(r) == 99
        assert r.return_type == "simple"
        assert r.frequency == daily_prices.frequency
        assert r.currency == daily_prices.currency

    def test_to_returns_log(self, daily_prices):
        r = daily_prices.to_returns(method="log")
        assert isinstance(r, ReturnSeries)
        assert r.return_type == "log"
        # Log returns should be close to simple returns for small moves
        r_simple = daily_prices.to_returns(method="simple")
        np.testing.assert_allclose(r.values, r_simple.values, atol=0.001)

    def test_slicing_preserves_type(self, daily_prices):
        sliced = daily_prices.iloc[10:50]
        assert isinstance(sliced, PriceSeries)

    def test_arithmetic_preserves_type(self, daily_prices):
        result = daily_prices * 2
        assert isinstance(result, PriceSeries)


# ---------------------------------------------------------------------------
# ReturnSeries
# ---------------------------------------------------------------------------


class TestReturnSeries:
    @pytest.fixture()
    def daily_returns(self):
        idx = pd.bdate_range("2023-01-03", periods=252)
        np.random.seed(42)
        rets = np.random.normal(0.0005, 0.01, 252)
        return ReturnSeries(rets, index=idx, name="SPY_ret")

    def test_creation(self, daily_returns):
        assert isinstance(daily_returns, pd.Series)
        assert isinstance(daily_returns, ReturnSeries)
        assert daily_returns.return_type == "simple"

    def test_frequency(self, daily_returns):
        assert daily_returns.frequency == "daily"
        assert daily_returns.periods_per_year == 252

    def test_sharpe(self, daily_returns):
        sr = daily_returns.sharpe()
        assert isinstance(sr, float)
        assert np.isfinite(sr)

    def test_sharpe_with_risk_free(self, daily_returns):
        sr0 = daily_returns.sharpe(risk_free=0.0)
        sr5 = daily_returns.sharpe(risk_free=0.05)
        # Higher risk-free rate should reduce Sharpe
        assert sr5 < sr0

    def test_annualized_vol(self, daily_returns):
        vol = daily_returns.annualized_vol()
        assert isinstance(vol, float)
        assert vol > 0

    def test_annualized_return(self, daily_returns):
        ret = daily_returns.annualized_return()
        assert isinstance(ret, float)
        assert np.isfinite(ret)

    def test_to_prices(self, daily_returns):
        prices = daily_returns.to_prices(initial_price=100.0)
        assert isinstance(prices, PriceSeries)
        assert len(prices) == len(daily_returns)
        assert prices.iloc[0] == pytest.approx(100.0 * (1 + daily_returns.iloc[0]))

    def test_roundtrip(self):
        """PriceSeries -> ReturnSeries -> PriceSeries should roundtrip."""
        idx = pd.bdate_range("2023-01-02", periods=50)
        original = PriceSeries(
            100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 50))),
            index=idx,
        )
        returns = original.to_returns(method="simple")
        recovered = returns.to_prices(initial_price=float(original.iloc[0]))
        # Should be very close (not exact due to floating point)
        np.testing.assert_allclose(
            recovered.values, original.values[1:], rtol=1e-10
        )

    def test_log_roundtrip(self):
        """Log return roundtrip should also work."""
        idx = pd.bdate_range("2023-01-02", periods=50)
        original = PriceSeries(
            100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 50))),
            index=idx,
        )
        returns = original.to_returns(method="log")
        recovered = returns.to_prices(initial_price=float(original.iloc[0]))
        np.testing.assert_allclose(
            recovered.values, original.values[1:], rtol=1e-10
        )

    def test_slicing_preserves_type(self, daily_returns):
        sliced = daily_returns.iloc[10:50]
        assert isinstance(sliced, ReturnSeries)


# ---------------------------------------------------------------------------
# OHLCVFrame
# ---------------------------------------------------------------------------


class TestOHLCVFrame:
    @pytest.fixture()
    def ohlcv(self):
        idx = pd.bdate_range("2023-01-02", periods=50)
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.normal(0, 1, 50))
        return OHLCVFrame(
            {
                "open": close + np.random.uniform(-0.5, 0.5, 50),
                "high": close + np.abs(np.random.normal(0, 1, 50)),
                "low": close - np.abs(np.random.normal(0, 1, 50)),
                "close": close,
                "volume": np.random.randint(100, 10000, 50),
            },
            index=idx,
        )

    def test_creation(self, ohlcv):
        assert isinstance(ohlcv, pd.DataFrame)
        assert isinstance(ohlcv, OHLCVFrame)
        assert ohlcv.shape == (50, 5)

    def test_close_accessor(self, ohlcv):
        c = ohlcv.close
        assert isinstance(c, PriceSeries)
        assert len(c) == 50

    def test_open_accessor(self, ohlcv):
        o = ohlcv.open
        assert isinstance(o, PriceSeries)

    def test_high_accessor(self, ohlcv):
        h = ohlcv.high
        assert isinstance(h, PriceSeries)

    def test_low_accessor(self, ohlcv):
        lo = ohlcv.low
        assert isinstance(lo, PriceSeries)

    def test_volume_accessor(self, ohlcv):
        v = ohlcv.volume
        assert isinstance(v, pd.Series)

    def test_case_insensitive_columns(self):
        idx = pd.bdate_range("2023-01-02", periods=3)
        df = OHLCVFrame(
            {
                "Open": [100, 101, 99],
                "High": [102, 103, 101],
                "Low": [99, 100, 98],
                "Close": [101, 99, 100],
                "Volume": [1000, 1200, 900],
            },
            index=idx,
        )
        assert isinstance(df.close, PriceSeries)
        assert isinstance(df.open, PriceSeries)

    def test_missing_column_raises(self, ohlcv):
        df = ohlcv.drop(columns=["close"])
        with pytest.raises(KeyError, match="close"):
            _ = OHLCVFrame(df).close

    def test_frequency(self, ohlcv):
        assert ohlcv.frequency == "daily"
        assert ohlcv.periods_per_year == 252

    def test_close_to_returns(self, ohlcv):
        """close -> to_returns should work end-to-end."""
        r = ohlcv.close.to_returns()
        assert isinstance(r, ReturnSeries)
        assert len(r) == 49


# ---------------------------------------------------------------------------
# ReturnFrame
# ---------------------------------------------------------------------------


class TestReturnFrame:
    @pytest.fixture()
    def multi_asset_returns(self):
        idx = pd.bdate_range("2023-01-03", periods=252)
        np.random.seed(42)
        return ReturnFrame(
            {
                "AAPL": np.random.normal(0.001, 0.015, 252),
                "GOOGL": np.random.normal(0.0008, 0.012, 252),
                "MSFT": np.random.normal(0.0009, 0.013, 252),
            },
            index=idx,
        )

    def test_creation(self, multi_asset_returns):
        assert isinstance(multi_asset_returns, pd.DataFrame)
        assert isinstance(multi_asset_returns, ReturnFrame)
        assert multi_asset_returns.shape == (252, 3)

    def test_frequency(self, multi_asset_returns):
        assert multi_asset_returns.frequency == "daily"
        assert multi_asset_returns.periods_per_year == 252

    def test_correlation(self, multi_asset_returns):
        corr = multi_asset_returns.correlation()
        assert corr.shape == (3, 3)
        # Diagonal should be 1.0
        np.testing.assert_allclose(np.diag(corr.values), 1.0)

    def test_covariance(self, multi_asset_returns):
        cov = multi_asset_returns.covariance()
        assert cov.shape == (3, 3)
        # Should be positive semi-definite
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert np.all(eigenvalues >= -1e-10)

    def test_covariance_annualized(self, multi_asset_returns):
        cov = multi_asset_returns.covariance(annualize=False)
        cov_ann = multi_asset_returns.covariance(annualize=True)
        np.testing.assert_allclose(cov_ann.values, cov.values * 252)

    def test_return_type(self):
        rf = ReturnFrame({"A": [0.01, -0.02]}, return_type="log")
        assert rf.return_type == "log"


# ---------------------------------------------------------------------------
# Coercion integration
# ---------------------------------------------------------------------------


class TestCoercionIntegration:
    def test_coerce_series_preserves_price_series(self):
        from wraquant.core._coerce import coerce_series

        idx = pd.bdate_range("2023-01-02", periods=10)
        p = PriceSeries([100 + i for i in range(10)], index=idx, currency="EUR")
        result = coerce_series(p, "test")
        assert isinstance(result, PriceSeries)
        assert result.currency == "EUR"

    def test_coerce_series_preserves_return_series(self):
        from wraquant.core._coerce import coerce_series

        r = ReturnSeries([0.01, -0.02, 0.03], return_type="log")
        result = coerce_series(r, "test")
        assert isinstance(result, ReturnSeries)
        assert result.return_type == "log"

    def test_coerce_array_from_price_series(self):
        from wraquant.core._coerce import coerce_array

        p = PriceSeries([100.0, 101.0, 102.0])
        arr = coerce_array(p, "test")
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, [100.0, 101.0, 102.0])
