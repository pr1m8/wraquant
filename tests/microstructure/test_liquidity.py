"""Tests for microstructure liquidity analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.microstructure.liquidity import (
    amihud_illiquidity,
    effective_spread,
    kyle_lambda,
    price_impact,
    realized_spread,
    roll_spread,
    turnover_ratio,
)


def _make_data(n: int = 200) -> dict[str, pd.Series]:
    rng = np.random.default_rng(42)
    prices = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
        index=pd.bdate_range("2020-01-01", periods=n),
    )
    returns = prices.pct_change().dropna()
    volume = pd.Series(
        rng.integers(100_000, 1_000_000, n),
        index=prices.index,
        dtype=np.float64,
    )
    return {"prices": prices, "returns": returns, "volume": volume}


class TestAmihudIlliquidity:
    def test_scalar_output(self) -> None:
        d = _make_data()
        result = amihud_illiquidity(d["returns"], d["volume"][1:])
        assert isinstance(result, float)
        assert result > 0

    def test_rolling_output(self) -> None:
        d = _make_data()
        result = amihud_illiquidity(d["returns"], d["volume"][1:], window=20)
        assert isinstance(result, pd.Series)
        assert len(result) == len(d["returns"])

    def test_higher_volume_means_lower_illiquidity(self) -> None:
        d = _make_data()
        low_vol = d["volume"][1:] * 0.1
        high_vol = d["volume"][1:] * 10.0
        ill_low = amihud_illiquidity(d["returns"], low_vol)
        ill_high = amihud_illiquidity(d["returns"], high_vol)
        assert ill_low > ill_high


class TestKyleLambda:
    def test_returns_series(self) -> None:
        d = _make_data()
        signed_vol = d["volume"] * np.where(np.random.default_rng(1).random(len(d["volume"])) > 0.5, 1, -1)
        signed_vol = pd.Series(signed_vol, index=d["volume"].index)
        result = kyle_lambda(d["prices"], signed_vol, window=20)
        assert isinstance(result, pd.Series)
        assert len(result) == len(d["prices"])

    def test_window_size_affects_output(self) -> None:
        d = _make_data()
        signed_vol = pd.Series(
            d["volume"].values * np.where(np.random.default_rng(1).random(len(d["volume"])) > 0.5, 1, -1),
            index=d["volume"].index,
        )
        lam10 = kyle_lambda(d["prices"], signed_vol, window=10)
        lam50 = kyle_lambda(d["prices"], signed_vol, window=50)
        # Different windows should give different smoothness
        assert lam10.dropna().std() != lam50.dropna().std()


class TestRollSpread:
    def test_returns_float(self) -> None:
        d = _make_data()
        result = roll_spread(d["prices"])
        assert isinstance(result, float) or np.isnan(result)

    def test_positive_spread_for_mean_reverting(self) -> None:
        # Create mean-reverting series (ensures negative autocovariance)
        rng = np.random.default_rng(42)
        mid = 100.0
        noise = rng.normal(0, 0.5, 500)
        spread_half = 0.05
        # Alternate buy/sell to simulate bid-ask bounce
        prices = np.empty(500)
        for i in range(500):
            prices[i] = mid + noise[i] + (spread_half if i % 2 == 0 else -spread_half)
        result = roll_spread(pd.Series(prices))
        assert isinstance(result, float)
        if not np.isnan(result):
            assert result > 0

    def test_nan_for_trending_series(self) -> None:
        # Monotonically trending should give non-negative autocov
        prices = pd.Series(np.arange(1.0, 201.0))
        result = roll_spread(prices)
        assert np.isnan(result)


class TestEffectiveSpread:
    def test_basic_computation(self) -> None:
        trade = np.array([100.05, 99.95, 100.10])
        mid = np.array([100.0, 100.0, 100.0])
        result = effective_spread(trade, mid)
        np.testing.assert_allclose(result, [0.10, 0.10, 0.20])

    def test_zero_when_at_midpoint(self) -> None:
        trade = np.array([100.0])
        mid = np.array([100.0])
        result = effective_spread(trade, mid)
        assert result[0] == 0.0


class TestRealizedSpread:
    def test_output_length(self) -> None:
        n = 50
        idx = pd.RangeIndex(n)
        trades = pd.Series(np.linspace(100, 101, n), index=idx)
        mids = pd.Series(np.linspace(100, 101, n), index=idx)
        result = realized_spread(trades, mids, delay=5)
        assert len(result) == n

    def test_nan_at_end(self) -> None:
        n = 50
        delay = 5
        idx = pd.RangeIndex(n)
        trades = pd.Series(np.ones(n) * 100, index=idx)
        mids = pd.Series(np.ones(n) * 99.95, index=idx)
        result = realized_spread(trades, mids, delay=delay)
        # Last `delay` values should be NaN
        assert result.iloc[-delay:].isna().all()


class TestPriceImpact:
    def test_output_type(self) -> None:
        n = 30
        idx = pd.RangeIndex(n)
        trades = pd.Series(np.linspace(100, 101, n), index=idx)
        vol = pd.Series(np.ones(n) * 1000, index=idx)
        direction = pd.Series(np.ones(n), index=idx)
        result = price_impact(trades, vol, direction)
        assert isinstance(result, pd.Series)
        assert len(result) == n


class TestTurnoverRatio:
    def test_scalar_shares_outstanding(self) -> None:
        vol = pd.Series([100_000, 200_000, 150_000], dtype=np.float64)
        result = turnover_ratio(vol, 1_000_000.0)
        np.testing.assert_allclose(result.values, [0.1, 0.2, 0.15])

    def test_series_shares_outstanding(self) -> None:
        vol = pd.Series([100_000, 200_000], dtype=np.float64)
        so = pd.Series([1_000_000, 2_000_000], dtype=np.float64)
        result = turnover_ratio(vol, so)
        np.testing.assert_allclose(result.values, [0.1, 0.1])
