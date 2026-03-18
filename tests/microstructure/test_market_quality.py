"""Tests for market quality metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.microstructure.market_quality import (
    depth,
    quoted_spread,
    relative_spread,
    resiliency,
    variance_ratio,
)


class TestQuotedSpread:
    def test_basic(self) -> None:
        bid = np.array([99.0, 100.0])
        ask = np.array([101.0, 102.0])
        result = quoted_spread(bid, ask)
        np.testing.assert_allclose(result, [2.0, 2.0])

    def test_series_input(self) -> None:
        bid = pd.Series([99.0, 100.0])
        ask = pd.Series([101.0, 102.0])
        result = quoted_spread(bid, ask)
        np.testing.assert_allclose(result, [2.0, 2.0])


class TestRelativeSpread:
    def test_basic(self) -> None:
        bid = np.array([99.0, 100.0])
        ask = np.array([101.0, 102.0])
        result = relative_spread(bid, ask)
        expected = np.array([2.0 / 100.0, 2.0 / 101.0])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_series_output(self) -> None:
        bid = pd.Series([99.0, 100.0])
        ask = pd.Series([101.0, 102.0])
        result = relative_spread(bid, ask)
        assert isinstance(result, pd.Series)

    def test_narrow_spread(self) -> None:
        bid = np.array([100.00])
        ask = np.array([100.01])
        result = relative_spread(bid, ask)
        assert result[0] < 0.001


class TestDepth:
    def test_1d_input(self) -> None:
        bid_vol = np.array([100, 200, 300, 400, 500], dtype=np.float64)
        ask_vol = np.array([150, 250, 350, 450, 550], dtype=np.float64)
        result = depth(bid_vol, ask_vol, levels=3)
        expected = (100 + 200 + 300) + (150 + 250 + 350)
        assert result == expected

    def test_2d_input(self) -> None:
        bid_vol = np.array([[100, 200, 300], [400, 500, 600]], dtype=np.float64)
        ask_vol = np.array([[150, 250, 350], [450, 550, 650]], dtype=np.float64)
        result = depth(bid_vol, ask_vol, levels=2)
        expected = np.array([100 + 200 + 150 + 250, 400 + 500 + 450 + 550], dtype=np.float64)
        np.testing.assert_allclose(result, expected)

    def test_dataframe_input(self) -> None:
        bid_vol = pd.DataFrame({"L1": [100, 200], "L2": [300, 400]})
        ask_vol = pd.DataFrame({"L1": [150, 250], "L2": [350, 450]})
        result = depth(bid_vol, ask_vol, levels=2)
        assert isinstance(result, pd.Series)


class TestResiliency:
    def test_output_type(self) -> None:
        rng = np.random.default_rng(42)
        spreads = pd.Series(rng.uniform(0.01, 0.05, 100))
        result = resiliency(spreads, window=20)
        assert isinstance(result, pd.Series)
        assert len(result) == 100


class TestVarianceRatio:
    def test_random_walk(self) -> None:
        rng = np.random.default_rng(42)
        # Geometric random walk: always positive
        prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 5000))))
        result = variance_ratio(prices, short_period=2, long_period=10)
        assert 0.8 < result["vr"] < 1.2  # Should be near 1

    def test_returns_dict(self) -> None:
        prices = pd.Series(np.linspace(100, 200, 500))
        result = variance_ratio(prices, short_period=2, long_period=10)
        assert "vr" in result
        assert "z_stat" in result
        assert "p_value" in result
