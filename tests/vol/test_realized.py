"""Tests for realized volatility estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wraquant.vol.realized import (
    garman_klass,
    parkinson,
    realized_volatility,
    rogers_satchell,
    yang_zhang,
)


def _make_ohlcv(n: int = 100) -> dict[str, pd.Series]:
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    high = close * (1 + rng.uniform(0.001, 0.03, n))
    low = close * (1 - rng.uniform(0.001, 0.03, n))
    open_ = close * (1 + rng.normal(0, 0.01, n))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return {
        "open": pd.Series(open_, index=idx),
        "high": pd.Series(high, index=idx),
        "low": pd.Series(low, index=idx),
        "close": pd.Series(close, index=idx),
    }


class TestRealizedVolatility:
    def test_returns_series(self) -> None:
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.02, 100))
        vol = realized_volatility(returns, window=20)
        assert isinstance(vol, pd.Series)
        assert len(vol) == 100

    def test_annualized_is_larger(self) -> None:
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.02, 100))
        vol_ann = realized_volatility(returns, window=20, annualize=True)
        vol_raw = realized_volatility(returns, window=20, annualize=False)
        # Annualized should be larger (sqrt(252) factor)
        assert vol_ann.dropna().mean() > vol_raw.dropna().mean()


class TestParkinson:
    def test_positive_vol(self) -> None:
        data = _make_ohlcv()
        vol = parkinson(data["high"], data["low"], window=20)
        assert (vol.dropna() > 0).all()


class TestGarmanKlass:
    def test_positive_vol(self) -> None:
        data = _make_ohlcv()
        vol = garman_klass(
            data["open"], data["high"], data["low"], data["close"], window=20
        )
        assert (vol.dropna() >= 0).all()


class TestRogersSatchell:
    def test_positive_vol(self) -> None:
        data = _make_ohlcv()
        vol = rogers_satchell(
            data["open"], data["high"], data["low"], data["close"], window=20
        )
        assert (vol.dropna() >= 0).all()


class TestYangZhang:
    def test_positive_vol(self) -> None:
        data = _make_ohlcv()
        vol = yang_zhang(
            data["open"], data["high"], data["low"], data["close"], window=20
        )
        assert (vol.dropna() >= 0).all()
