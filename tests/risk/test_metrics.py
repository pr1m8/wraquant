"""Tests for risk and performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wraquant.risk.metrics import (
    hit_ratio,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)


def _make_returns(n: int = 252, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(rng.normal(0.0005, 0.02, size=n), index=dates, name="returns")


def _make_prices(n: int = 252, seed: int = 42) -> pd.Series:
    returns = _make_returns(n, seed)
    return (100 * (1 + returns).cumprod()).rename("price")


class TestSharpeRatio:
    def test_returns_float(self) -> None:
        ret = _make_returns()
        sr = sharpe_ratio(ret)
        assert isinstance(sr, float)

    def test_zero_vol_returns_zero(self) -> None:
        ret = pd.Series([0.0] * 100)
        assert sharpe_ratio(ret) == 0.0

    def test_higher_mean_higher_sharpe(self) -> None:
        rng = np.random.default_rng(42)
        low = pd.Series(rng.normal(0.0001, 0.02, 252))
        high = pd.Series(rng.normal(0.002, 0.02, 252))
        assert sharpe_ratio(high) > sharpe_ratio(low)


class TestSortinoRatio:
    def test_returns_float(self) -> None:
        ret = _make_returns()
        sr = sortino_ratio(ret)
        assert isinstance(sr, float)

    def test_all_positive_returns(self) -> None:
        ret = pd.Series([0.01] * 50)
        # No downside returns, so sortino should be +inf
        sr = sortino_ratio(ret)
        assert sr == float("inf")


class TestInformationRatio:
    def test_returns_float(self) -> None:
        ret = _make_returns(seed=42)
        bench = _make_returns(seed=99)
        ir = information_ratio(ret, bench)
        assert isinstance(ir, float)

    def test_identical_returns_zero(self) -> None:
        ret = _make_returns()
        assert information_ratio(ret, ret) == 0.0


class TestMaxDrawdown:
    def test_negative_or_zero(self) -> None:
        prices = _make_prices()
        mdd = max_drawdown(prices)
        assert mdd <= 0

    def test_monotone_up(self) -> None:
        prices = pd.Series(range(1, 101), dtype=float)
        assert max_drawdown(prices) == 0.0


class TestHitRatio:
    def test_range(self) -> None:
        ret = _make_returns()
        hr = hit_ratio(ret)
        assert 0.0 <= hr <= 1.0

    def test_all_positive(self) -> None:
        ret = pd.Series([0.01, 0.02, 0.03])
        assert hit_ratio(ret) == 1.0

    def test_all_negative(self) -> None:
        ret = pd.Series([-0.01, -0.02, -0.03])
        assert hit_ratio(ret) == 0.0
