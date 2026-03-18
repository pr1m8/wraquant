"""Tests for portfolio risk analytics."""

from __future__ import annotations

import numpy as np

from wraquant.risk.portfolio import (
    diversification_ratio,
    portfolio_volatility,
    risk_contribution,
)


def _make_cov(n: int = 3, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    return A @ A.T / n


class TestPortfolioVolatility:
    def test_positive(self) -> None:
        cov = _make_cov()
        w = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        vol = portfolio_volatility(w, cov)
        assert vol > 0

    def test_single_asset(self) -> None:
        cov = np.array([[0.04]])
        w = np.array([1.0])
        vol = portfolio_volatility(w, cov)
        np.testing.assert_allclose(vol, 0.2, atol=1e-10)


class TestRiskContribution:
    def test_sums_to_one(self) -> None:
        cov = _make_cov()
        w = np.array([0.4, 0.3, 0.3])
        rc = risk_contribution(w, cov)
        np.testing.assert_allclose(rc.sum(), 1.0, atol=1e-10)

    def test_equal_weight_symmetric_cov(self) -> None:
        # Equal weight on identity cov should give equal risk
        cov = np.eye(3) * 0.04
        w = np.ones(3) / 3
        rc = risk_contribution(w, cov)
        np.testing.assert_allclose(rc, [1.0 / 3, 1.0 / 3, 1.0 / 3], atol=1e-10)


class TestDiversificationRatio:
    def test_identity_cov(self) -> None:
        # Diagonal (uncorrelated) cov: diversification ratio = sqrt(n)
        cov = np.eye(3) * 0.04
        w = np.ones(3) / 3
        dr = diversification_ratio(w, cov)
        np.testing.assert_allclose(dr, np.sqrt(3), atol=1e-10)

    def test_correlated_assets_higher_ratio(self) -> None:
        # Non-diagonal covariance should give ratio > 1
        cov = np.array(
            [
                [0.04, 0.01, 0.005],
                [0.01, 0.04, 0.01],
                [0.005, 0.01, 0.04],
            ]
        )
        w = np.ones(3) / 3
        dr = diversification_ratio(w, cov)
        assert dr >= 1.0
