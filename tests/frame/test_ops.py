"""Tests for wraquant.frame.ops."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from wraquant.frame.ops import (
    cumulative_returns,
    drawdowns,
    ewm_mean,
    log_returns,
    resample,
    returns,
    rolling_mean,
    rolling_std,
)


@pytest.fixture
def prices() -> pd.Series:
    return pd.Series(
        [100.0, 102.0, 101.0, 105.0, 103.0],
        index=pd.bdate_range("2020-01-01", periods=5),
        name="price",
    )


class TestReturns:
    def test_simple_returns(self, prices: pd.Series) -> None:
        r = returns(prices)
        assert r.iloc[0] != r.iloc[0]  # NaN
        assert_allclose(r.iloc[1], 0.02, atol=1e-10)

    def test_log_returns(self, prices: pd.Series) -> None:
        lr = log_returns(prices)
        assert lr.iloc[0] != lr.iloc[0]  # NaN
        assert_allclose(lr.iloc[1], np.log(102 / 100), atol=1e-10)

    def test_returns_multi_period(self, prices: pd.Series) -> None:
        r = returns(prices, periods=2)
        assert_allclose(r.iloc[2], (101 - 100) / 100, atol=1e-10)


class TestCumulativeReturns:
    def test_cumulative(self, prices: pd.Series) -> None:
        r = returns(prices).dropna()
        cr = cumulative_returns(r)
        # Final cumulative return should match total return from first return onwards
        # returns start from prices.iloc[1], so base is prices.iloc[0]
        expected = (prices.iloc[-1] / prices.iloc[0]) - 1
        assert_allclose(cr.iloc[-1], expected, atol=1e-10)


class TestDrawdowns:
    def test_drawdowns_negative(self, prices: pd.Series) -> None:
        dd = drawdowns(prices)
        assert (dd <= 0).all()

    def test_drawdown_at_peak_is_zero(self) -> None:
        prices = pd.Series([100, 105, 110, 108, 112])
        dd = drawdowns(prices)
        assert dd.iloc[2] == 0  # 110 is a peak
        assert dd.iloc[4] == 0  # 112 is a new peak


class TestRolling:
    def test_rolling_mean(self, prices: pd.Series) -> None:
        rm = rolling_mean(prices, window=3)
        assert rm.iloc[:2].isna().all()
        assert_allclose(rm.iloc[2], (100 + 102 + 101) / 3, atol=1e-10)

    def test_rolling_std(self, prices: pd.Series) -> None:
        rs = rolling_std(prices, window=3)
        assert rs.iloc[:2].isna().all()
        assert rs.iloc[2] > 0


class TestEWM:
    def test_ewm_mean(self, prices: pd.Series) -> None:
        em = ewm_mean(prices, span=3)
        assert len(em) == len(prices)
        assert not em.isna().any()


class TestResample:
    def test_resample_weekly(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=20)
        prices = pd.Series(range(20), index=dates, dtype=float)
        weekly = resample(prices, "W", "last")
        assert len(weekly) < len(prices)
