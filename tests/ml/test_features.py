"""Tests for wraquant.ml.features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.ml.features import (
    cross_asset_features,
    interaction_features,
    label_fixed_horizon,
    label_triple_barrier,
    microstructure_features,
    regime_features,
    return_features,
    rolling_features,
    technical_features,
    volatility_features,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def price_series() -> pd.Series:
    """Deterministic upward-trending price series."""
    np.random.seed(0)
    n = 200
    prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    return pd.Series(prices, name="close")


@pytest.fixture()
def ohlcv(price_series: pd.Series) -> dict[str, pd.Series]:
    """Synthetic OHLCV data derived from the price series."""
    close = price_series
    high = close * (1 + np.abs(np.random.default_rng(1).normal(0, 0.005, len(close))))
    low = close * (1 - np.abs(np.random.default_rng(2).normal(0, 0.005, len(close))))
    volume = pd.Series(
        np.random.default_rng(3).integers(1_000, 100_000, size=len(close)),
        dtype=float,
    )
    return {"high": high, "low": low, "close": close, "volume": volume}


# ---------------------------------------------------------------------------
# rolling_features
# ---------------------------------------------------------------------------


class TestRollingFeatures:
    def test_output_shape(self, price_series: pd.Series) -> None:
        windows = [5, 10, 21]
        result = rolling_features(price_series, windows=windows)
        n_stats = 6  # mean, std, skew, kurt, min, max
        assert result.shape[1] == len(windows) * n_stats

    def test_dataframe_input(self) -> None:
        df = pd.DataFrame({"a": range(100), "b": range(100, 200)}, dtype=float)
        windows = [5, 10]
        result = rolling_features(df, windows=windows)
        # 2 columns * 2 windows * 6 stats = 24
        assert result.shape[1] == 2 * 2 * 6

    def test_index_preserved(self, price_series: pd.Series) -> None:
        result = rolling_features(price_series, windows=[5])
        assert result.index.equals(price_series.index)


# ---------------------------------------------------------------------------
# return_features
# ---------------------------------------------------------------------------


class TestReturnFeatures:
    def test_known_data(self) -> None:
        prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        lags = [1, 2]
        result = return_features(prices, lags=lags)
        # Should contain ret_lag1, ret_lag2, cum_ret_1, cum_ret_2
        assert "ret_lag1" in result.columns
        assert "cum_ret_2" in result.columns
        assert len(result) == len(prices)

    def test_column_count(self, price_series: pd.Series) -> None:
        lags = [1, 2, 3, 5, 10, 21]
        result = return_features(price_series, lags=lags)
        assert result.shape[1] == 2 * len(lags)


# ---------------------------------------------------------------------------
# technical_features
# ---------------------------------------------------------------------------


class TestTechnicalFeatures:
    def test_columns_without_volume(self, ohlcv: dict) -> None:
        result = technical_features(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert set(result.columns) == {"rsi", "macd_hist", "bb_pctb", "atr"}

    def test_columns_with_volume(self, ohlcv: dict) -> None:
        result = technical_features(**ohlcv)
        assert "obv" in result.columns


# ---------------------------------------------------------------------------
# volatility_features
# ---------------------------------------------------------------------------


class TestVolatilityFeatures:
    def test_output_columns(self, price_series: pd.Series) -> None:
        rets = price_series.pct_change().dropna()
        result = volatility_features(rets, windows=[5, 10])
        assert "realized_vol_w5" in result.columns
        assert "vol_of_vol_w10" in result.columns
        assert "vol_ratio_w5_w10" in result.columns


# ---------------------------------------------------------------------------
# microstructure_features
# ---------------------------------------------------------------------------


class TestMicrostructureFeatures:
    def test_output_columns(self, ohlcv: dict) -> None:
        result = microstructure_features(**ohlcv)
        expected = {
            "amihud_illiq",
            "kyle_lambda",
            "log_volume",
            "volume_ma_ratio",
            "dollar_volume",
        }
        assert set(result.columns) == expected


# ---------------------------------------------------------------------------
# label_fixed_horizon
# ---------------------------------------------------------------------------


class TestLabelFixedHorizon:
    def test_binary_labels(self) -> None:
        returns = pd.Series(
            [0.01, -0.02, 0.03, 0.01, -0.01, 0.02, -0.03, 0.01, 0.02, 0.01]
        )
        labels = label_fixed_horizon(returns, horizon=2, threshold=0.0)
        # Should produce 1 or 0
        valid = labels.dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_ternary_labels(self) -> None:
        returns = pd.Series(
            [0.01, -0.02, 0.03, 0.01, -0.01, 0.02, -0.03, 0.01, 0.02, 0.01]
        )
        labels = label_fixed_horizon(returns, horizon=2, threshold=0.01)
        valid = labels.dropna()
        assert set(valid.unique()).issubset({-1, 0, 1})

    def test_trailing_nan(self) -> None:
        returns = pd.Series(np.random.randn(50) * 0.01)
        labels = label_fixed_horizon(returns, horizon=5)
        # Last entries should be NaN
        assert labels.iloc[-1] is pd.NA


# ---------------------------------------------------------------------------
# label_triple_barrier
# ---------------------------------------------------------------------------


class TestLabelTripleBarrier:
    def test_output_values(self, price_series: pd.Series) -> None:
        labels = label_triple_barrier(
            price_series, upper=0.02, lower=0.02, max_holding=10
        )
        valid = labels.dropna()
        assert set(valid.unique()).issubset({-1, 0, 1})

    def test_no_horizontal_barriers(self, price_series: pd.Series) -> None:
        """When barriers are None, only vertical barrier is active."""
        labels = label_triple_barrier(
            price_series, upper=None, lower=None, max_holding=5
        )
        valid = labels.dropna()
        assert len(valid) > 0
        assert set(valid.unique()).issubset({-1, 0, 1})


# ---------------------------------------------------------------------------
# interaction_features
# ---------------------------------------------------------------------------


class TestInteractionFeatures:
    def test_output_columns(self) -> None:
        df = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0]}
        )
        result = interaction_features(df, columns=["a", "b"])
        assert "a_x_b" in result.columns
        assert "a_div_b" in result.columns
        assert result.shape[0] == 3

    def test_all_columns_default(self) -> None:
        df = pd.DataFrame(
            {"x": [1.0, 2.0], "y": [3.0, 4.0], "z": [5.0, 6.0]}
        )
        result = interaction_features(df)
        # 3 choose 2 = 3 pairs, 2 features each (product + ratio) = 6
        assert result.shape[1] == 6

    def test_product_values(self) -> None:
        df = pd.DataFrame({"a": [2.0, 3.0], "b": [4.0, 5.0]})
        result = interaction_features(df)
        np.testing.assert_array_equal(result["a_x_b"].values, [8.0, 15.0])

    def test_ratio_values(self) -> None:
        df = pd.DataFrame({"a": [10.0, 20.0], "b": [2.0, 5.0]})
        result = interaction_features(df)
        np.testing.assert_array_equal(result["a_div_b"].values, [5.0, 4.0])

    def test_division_by_zero_gives_nan(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [0.0, 3.0]})
        result = interaction_features(df)
        assert np.isnan(result["a_div_b"].iloc[0])
        assert result["a_div_b"].iloc[1] == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# cross_asset_features
# ---------------------------------------------------------------------------


class TestCrossAssetFeatures:
    def test_output_columns(self) -> None:
        np.random.seed(0)
        asset = pd.Series(np.random.randn(200) * 0.01, name="asset")
        bench = pd.Series(np.random.randn(200) * 0.01, name="bench")
        result = cross_asset_features(asset, bench, windows=[10, 21])
        assert "rolling_corr_w10" in result.columns
        assert "rolling_corr_w21" in result.columns
        assert "rolling_beta_w10" in result.columns
        assert "rolling_beta_w21" in result.columns
        assert "relative_strength_w10" in result.columns
        assert "relative_strength_w21" in result.columns

    def test_output_length(self) -> None:
        np.random.seed(0)
        n = 100
        asset = pd.Series(np.random.randn(n) * 0.01)
        bench = pd.Series(np.random.randn(n) * 0.01)
        result = cross_asset_features(asset, bench, windows=[10])
        assert len(result) == n

    def test_corr_in_range(self) -> None:
        np.random.seed(42)
        asset = pd.Series(np.random.randn(200) * 0.01)
        bench = pd.Series(np.random.randn(200) * 0.01)
        result = cross_asset_features(asset, bench, windows=[20])
        valid_corr = result["rolling_corr_w20"].dropna()
        assert (valid_corr.abs() <= 1.0 + 1e-10).all()


# ---------------------------------------------------------------------------
# regime_features
# ---------------------------------------------------------------------------


class TestRegimeFeatures:
    def test_output_columns(self) -> None:
        np.random.seed(42)
        probs = pd.DataFrame(
            {
                "bull": np.random.dirichlet([5, 2], size=100)[:, 0],
                "bear": np.random.dirichlet([5, 2], size=100)[:, 1],
            }
        )
        result = regime_features(probs)
        assert "current_regime" in result.columns
        assert "regime_duration" in result.columns
        assert "regime_change" in result.columns
        assert "transition_prob_w5" in result.columns
        assert "transition_prob_w10" in result.columns
        assert "transition_prob_w21" in result.columns
        assert "prob_bull" in result.columns
        assert "prob_bear" in result.columns

    def test_regime_duration_increases(self) -> None:
        # All same regime -> duration should be 1, 2, 3, ...
        probs = pd.DataFrame(
            {"a": [0.9, 0.9, 0.9, 0.9, 0.9], "b": [0.1, 0.1, 0.1, 0.1, 0.1]}
        )
        result = regime_features(probs)
        np.testing.assert_array_equal(
            result["regime_duration"].values, [1, 2, 3, 4, 5]
        )

    def test_regime_change_on_switch(self) -> None:
        probs = pd.DataFrame(
            {"a": [0.9, 0.9, 0.1, 0.1], "b": [0.1, 0.1, 0.9, 0.9]}
        )
        result = regime_features(probs)
        # Change at index 2
        assert result["regime_change"].iloc[0] == 0
        assert result["regime_change"].iloc[1] == 0
        assert result["regime_change"].iloc[2] == 1
        assert result["regime_change"].iloc[3] == 0

    def test_with_explicit_labels(self) -> None:
        probs = pd.DataFrame({"a": [0.5, 0.5, 0.5], "b": [0.5, 0.5, 0.5]})
        labels = pd.Series([0, 1, 1])
        result = regime_features(probs, regime_labels=labels)
        assert result["current_regime"].iloc[0] == 0
        assert result["current_regime"].iloc[1] == 1
        assert result["regime_change"].iloc[1] == 1

    def test_output_length(self) -> None:
        np.random.seed(0)
        n = 50
        probs = pd.DataFrame(
            {
                "state0": np.random.dirichlet([3, 3], size=n)[:, 0],
                "state1": np.random.dirichlet([3, 3], size=n)[:, 1],
            }
        )
        result = regime_features(probs)
        assert len(result) == n
