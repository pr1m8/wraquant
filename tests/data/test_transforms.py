"""Tests for wraquant.data.transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.data.transforms import (
    normalize_prices,
    rank_transform,
    rolling_zscore,
    to_prices,
    to_returns,
)


@pytest.fixture()
def price_series() -> pd.Series:
    """A simple upward-trending price series."""
    dates = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(99)
    log_returns = rng.normal(0.0003, 0.01, size=300)
    prices = 100.0 * np.exp(np.cumsum(log_returns))
    return pd.Series(prices, index=dates, name="price")


class TestToReturnsAndToPrices:
    def test_simple_roundtrip(self, price_series: pd.Series) -> None:
        """to_returns and to_prices should be approximate inverses."""
        rets = to_returns(price_series, method="simple")
        reconstructed = to_prices(
            rets, initial_price=price_series.iloc[0], method="simple"
        )
        pd.testing.assert_series_equal(
            price_series,
            reconstructed,
            check_names=False,
            atol=1e-10,
        )

    def test_log_roundtrip(self, price_series: pd.Series) -> None:
        rets = to_returns(price_series, method="log")
        reconstructed = to_prices(
            rets, initial_price=price_series.iloc[0], method="log"
        )
        pd.testing.assert_series_equal(
            price_series,
            reconstructed,
            check_names=False,
            atol=1e-10,
        )

    def test_first_return_is_nan(self, price_series: pd.Series) -> None:
        rets = to_returns(price_series)
        assert pd.isna(rets.iloc[0])


class TestNormalizePrices:
    def test_starts_at_base(self, price_series: pd.Series) -> None:
        normed = normalize_prices(price_series, base=100.0)
        assert normed.iloc[0] == pytest.approx(100.0)

    def test_custom_base(self, price_series: pd.Series) -> None:
        normed = normalize_prices(price_series, base=1.0)
        assert normed.iloc[0] == pytest.approx(1.0)

    def test_preserves_relative_changes(self, price_series: pd.Series) -> None:
        normed = normalize_prices(price_series, base=100.0)
        original_ratio = price_series.iloc[-1] / price_series.iloc[0]
        normed_ratio = normed.iloc[-1] / normed.iloc[0]
        assert original_ratio == pytest.approx(normed_ratio, rel=1e-10)


class TestRollingZscore:
    def test_mean_and_std_after_warmup(self, price_series: pd.Series) -> None:
        window = 60
        zs = rolling_zscore(price_series, window=window)
        # After the warmup period the z-scores should have roughly mean 0
        # and std 1 (not exact because the window shifts).
        valid = zs.iloc[window:]
        assert abs(valid.mean()) < 0.5
        assert 0.5 < valid.std() < 1.5

    def test_output_length(self, price_series: pd.Series) -> None:
        zs = rolling_zscore(price_series, window=20)
        assert len(zs) == len(price_series)


class TestRankTransform:
    def test_output_range(self) -> None:
        s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        ranked = rank_transform(s)
        assert ranked.min() > 0.0
        assert ranked.max() <= 1.0

    def test_dataframe_cross_sectional(self) -> None:
        df = pd.DataFrame({"a": [1.0, 4.0], "b": [2.0, 3.0], "c": [3.0, 2.0]})
        ranked = rank_transform(df)
        # Each row should sum to (n+1)/2 * n / n^2 ... just check range.
        assert ranked.min().min() > 0.0
        assert ranked.max().max() <= 1.0

    def test_preserves_order(self) -> None:
        s = pd.Series([5.0, 1.0, 3.0])
        ranked = rank_transform(s)
        # 5 should have highest rank, 1 lowest.
        assert ranked.iloc[0] > ranked.iloc[2] > ranked.iloc[1]
