"""Tests for microstructure liquidity analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.microstructure.liquidity import (
    amihud_illiquidity,
    amihud_rolling,
    closing_quoted_spread,
    corwin_schultz_spread,
    depth_imbalance,
    effective_spread,
    kyle_lambda,
    lambda_kyle_rolling,
    liquidity_commonality,
    price_impact,
    realized_spread,
    roll_spread,
    spread_decomposition,
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


# ---------------------------------------------------------------------------
# Tests for enhanced liquidity analytics
# ---------------------------------------------------------------------------


class TestCorwinSchultzSpread:
    def test_positive_spread_estimate(self) -> None:
        """Corwin-Schultz should produce a positive spread for realistic data."""
        rng = np.random.default_rng(42)
        n = 200
        mid = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
        spread_half = 0.10
        high = pd.Series(mid + spread_half + rng.uniform(0, 0.2, n))
        low = pd.Series(mid - spread_half - rng.uniform(0, 0.2, n))
        result = corwin_schultz_spread(high, low, window=5)
        valid = result.dropna()
        # At least some values should be positive
        assert (valid > 0).any(), "Expected positive spread estimates"

    def test_zero_floor(self) -> None:
        """Negative estimates should be floored at zero."""
        rng = np.random.default_rng(42)
        n = 100
        high = pd.Series(100.0 + rng.uniform(0, 5, n))
        low = pd.Series(100.0 - rng.uniform(0, 5, n))
        result = corwin_schultz_spread(high, low)
        assert (result.dropna() >= 0).all()

    def test_wider_spread_detected(self) -> None:
        """Wider true spread should produce larger estimated spread."""
        rng = np.random.default_rng(42)
        n = 200
        mid = 100.0 * np.ones(n)

        # Narrow spread
        h_narrow = pd.Series(mid + 0.01 + rng.uniform(0, 0.01, n))
        l_narrow = pd.Series(mid - 0.01 - rng.uniform(0, 0.01, n))

        # Wide spread
        h_wide = pd.Series(mid + 0.50 + rng.uniform(0, 0.01, n))
        l_wide = pd.Series(mid - 0.50 - rng.uniform(0, 0.01, n))

        cs_narrow = corwin_schultz_spread(h_narrow, l_narrow, window=20)
        cs_wide = corwin_schultz_spread(h_wide, l_wide, window=20)

        assert cs_wide.dropna().mean() > cs_narrow.dropna().mean()


class TestClosingQuotedSpread:
    def test_basic(self) -> None:
        bid = pd.Series([99.0, 100.0, 101.0])
        ask = pd.Series([101.0, 102.0, 103.0])
        result = closing_quoted_spread(bid, ask)
        np.testing.assert_allclose(result.values, [2.0, 2.0, 2.0])

    def test_name(self) -> None:
        bid = pd.Series([99.0])
        ask = pd.Series([101.0])
        result = closing_quoted_spread(bid, ask)
        assert result.name == "closing_quoted_spread"


class TestDepthImbalance:
    def test_balanced(self) -> None:
        bid = pd.Series([100.0, 200.0])
        ask = pd.Series([100.0, 200.0])
        result = depth_imbalance(bid, ask)
        np.testing.assert_allclose(result.values, [0.0, 0.0])

    def test_all_bid(self) -> None:
        bid = np.array([100.0])
        ask = np.array([0.0])
        result = depth_imbalance(bid, ask)
        # bid_depth > 0, ask_depth = 0 -> imbalance = 1.0
        np.testing.assert_allclose(result, [1.0])

    def test_all_ask(self) -> None:
        bid = np.array([0.0])
        ask = np.array([100.0])
        result = depth_imbalance(bid, ask)
        np.testing.assert_allclose(result, [-1.0])

    def test_bounded(self) -> None:
        rng = np.random.default_rng(42)
        bid = pd.Series(rng.uniform(1, 100, 50))
        ask = pd.Series(rng.uniform(1, 100, 50))
        result = depth_imbalance(bid, ask)
        assert (result >= -1).all()
        assert (result <= 1).all()


class TestLambdaKyleRolling:
    def test_returns_dataframe(self) -> None:
        d = _make_data()
        rng = np.random.default_rng(1)
        signed_vol = pd.Series(
            d["volume"].values * np.where(rng.random(len(d["volume"])) > 0.5, 1, -1),
            index=d["volume"].index,
        )
        result = lambda_kyle_rolling(d["prices"], signed_vol, window=20)
        assert isinstance(result, pd.DataFrame)
        assert "lambda" in result.columns
        assert "std_err" in result.columns
        assert "ci_lower" in result.columns
        assert "ci_upper" in result.columns

    def test_ci_contains_lambda(self) -> None:
        d = _make_data()
        rng = np.random.default_rng(1)
        signed_vol = pd.Series(
            d["volume"].values * np.where(rng.random(len(d["volume"])) > 0.5, 1, -1),
            index=d["volume"].index,
        )
        result = lambda_kyle_rolling(d["prices"], signed_vol, window=20)
        valid = result.dropna()
        assert (valid["ci_lower"] <= valid["lambda"]).all()
        assert (valid["ci_upper"] >= valid["lambda"]).all()


class TestAmihudRolling:
    def test_output_type(self) -> None:
        d = _make_data()
        result = amihud_rolling(d["returns"], d["volume"][1:], window=21)
        assert isinstance(result, pd.Series)

    def test_normalized_mean_near_one(self) -> None:
        d = _make_data()
        result = amihud_rolling(d["returns"], d["volume"][1:], window=21, normalize=True)
        valid = result.dropna()
        # Mean of normalized series should be near 1.0
        assert 0.5 < valid.mean() < 2.0


class TestLiquidityCommonality:
    def test_r_squared_bounded(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        market = pd.Series(rng.uniform(0, 1, n))
        # Asset with some correlation to market
        asset = market * 0.5 + pd.Series(rng.uniform(0, 1, n)) * 0.5
        result = liquidity_commonality(asset, market, window=30)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()


class TestSpreadDecomposition:
    def test_components_sum_to_one(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        mid = 100.0 + np.cumsum(rng.normal(0, 0.05, n))
        spread_half = 0.05
        bid = pd.Series(mid - spread_half)
        ask = pd.Series(mid + spread_half)

        # Simulate trades at ask (buys) and bid (sells)
        direction = pd.Series(np.where(rng.random(n) > 0.5, 1, -1))
        trade_prices = pd.Series(
            np.where(direction == 1, mid + spread_half * 0.8, mid - spread_half * 0.8)
        )

        result = spread_decomposition(trade_prices, bid, ask, direction, delay=5)
        assert "adverse_selection" in result
        assert "order_processing" in result
        assert "inventory_holding" in result

        total = result["adverse_selection"] + result["order_processing"] + result["inventory_holding"]
        np.testing.assert_allclose(total, 1.0, atol=1e-10)

    def test_positive_effective_spread(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        mid = 100.0 * np.ones(n)
        bid = pd.Series(mid - 0.05)
        ask = pd.Series(mid + 0.05)
        direction = pd.Series(np.ones(n))
        trade_prices = pd.Series(mid + 0.03)

        result = spread_decomposition(trade_prices, bid, ask, direction, delay=5)
        assert result["effective_spread_mean"] > 0
