"""Tests for portfolio optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from wraquant.opt.base import OptimizationResult
from wraquant.opt.portfolio import (
    black_litterman,
    equal_weight,
    hierarchical_risk_parity,
    inverse_volatility,
    max_sharpe,
    mean_variance,
    min_volatility,
    risk_parity,
)


def _make_returns(n_assets: int = 4, n_periods: int = 252) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = rng.normal(0.0003, 0.015, size=(n_periods, n_assets))
    cols = [f"asset_{i}" for i in range(n_assets)]
    return pd.DataFrame(data, columns=cols)


class TestMeanVariance:
    def test_weights_sum_to_one(self) -> None:
        ret = _make_returns()
        result = mean_variance(ret)
        assert_allclose(result.weights.sum(), 1.0, atol=1e-6)

    def test_returns_optimization_result(self) -> None:
        ret = _make_returns()
        result = mean_variance(ret)
        assert isinstance(result, OptimizationResult)
        assert result.volatility > 0

    def test_to_dict(self) -> None:
        ret = _make_returns()
        result = mean_variance(ret)
        d = result.to_dict()
        assert len(d) == 4
        assert all(k.startswith("asset_") for k in d)


class TestMinVolatility:
    def test_lower_vol_than_equal_weight(self) -> None:
        ret = _make_returns()
        mv = min_volatility(ret)
        ew = equal_weight(ret)
        assert mv.volatility <= ew.volatility + 1e-6


class TestMaxSharpe:
    def test_positive_sharpe(self) -> None:
        rng = np.random.default_rng(42)
        # Create returns with positive drift
        n = 252
        data = rng.normal(0.001, 0.01, size=(n, 3))
        ret = pd.DataFrame(data, columns=["A", "B", "C"])
        result = max_sharpe(ret)
        assert result.weights.sum() > 0.99


class TestRiskParity:
    def test_weights_sum_to_one(self) -> None:
        ret = _make_returns()
        result = risk_parity(ret)
        assert_allclose(result.weights.sum(), 1.0, atol=1e-4)

    def test_all_positive_weights(self) -> None:
        ret = _make_returns()
        result = risk_parity(ret)
        assert (result.weights > 0).all()


class TestEqualWeight:
    def test_equal_weights(self) -> None:
        ret = _make_returns(n_assets=5)
        result = equal_weight(ret)
        assert_allclose(result.weights, 0.2, atol=1e-10)


class TestInverseVolatility:
    def test_weights_sum_to_one(self) -> None:
        ret = _make_returns()
        result = inverse_volatility(ret)
        assert_allclose(result.weights.sum(), 1.0, atol=1e-10)


class TestHRP:
    def test_weights_sum_to_one(self) -> None:
        ret = _make_returns()
        result = hierarchical_risk_parity(ret)
        assert_allclose(result.weights.sum(), 1.0, atol=1e-6)

    def test_all_positive_weights(self) -> None:
        ret = _make_returns()
        result = hierarchical_risk_parity(ret)
        assert (result.weights > 0).all()


class TestBlackLitterman:
    def test_with_views(self) -> None:
        ret = _make_returns()
        views = {"asset_0": 0.10, "asset_1": 0.05}
        result = black_litterman(ret, views=views)
        assert_allclose(result.weights.sum(), 1.0, atol=1e-4)

    def test_empty_views(self) -> None:
        ret = _make_returns()
        result = black_litterman(ret, views={})
        assert_allclose(result.weights.sum(), 1.0, atol=1e-6)
