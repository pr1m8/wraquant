"""Tests for extended risk metrics (treynor, m_squared, jensens_alpha, appraisal_ratio, capture_ratios)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.risk.metrics import (
    appraisal_ratio,
    capture_ratios,
    jensens_alpha,
    m_squared,
    treynor_ratio,
)


@pytest.fixture()
def market_portfolio_data():
    """Generate synthetic market and portfolio returns."""
    np.random.seed(42)
    n = 500
    market = pd.Series(np.random.normal(0.0005, 0.01, n))
    portfolio = 1.1 * market + np.random.normal(0.0002, 0.005, n)
    return pd.Series(portfolio), market


class TestTreynorRatio:
    """Tests for treynor_ratio."""

    def test_returns_float(self, market_portfolio_data):
        portfolio, market = market_portfolio_data
        tr = treynor_ratio(portfolio, market)
        assert isinstance(tr, float)

    def test_higher_return_higher_treynor(self):
        np.random.seed(42)
        market = pd.Series(np.random.normal(0.0005, 0.01, 500))
        low = 1.0 * market + np.random.normal(0.0001, 0.005, 500)
        high = 1.0 * market + np.random.normal(0.001, 0.005, 500)
        tr_low = treynor_ratio(pd.Series(low), market)
        tr_high = treynor_ratio(pd.Series(high), market)
        assert tr_high > tr_low

    def test_with_risk_free(self, market_portfolio_data):
        portfolio, market = market_portfolio_data
        tr = treynor_ratio(portfolio, market, risk_free=0.04)
        assert isinstance(tr, float)


class TestMSquared:
    """Tests for m_squared."""

    def test_returns_float(self, market_portfolio_data):
        portfolio, market = market_portfolio_data
        m2 = m_squared(portfolio, market)
        assert isinstance(m2, float)

    def test_with_risk_free(self, market_portfolio_data):
        portfolio, market = market_portfolio_data
        m2 = m_squared(portfolio, market, risk_free=0.04)
        assert isinstance(m2, float)


class TestJensensAlpha:
    """Tests for jensens_alpha."""

    def test_returns_float(self, market_portfolio_data):
        portfolio, market = market_portfolio_data
        alpha = jensens_alpha(portfolio, market)
        assert isinstance(alpha, float)

    def test_positive_alpha_for_outperformer(self):
        """Portfolio with added constant should have positive alpha."""
        np.random.seed(42)
        market = pd.Series(np.random.normal(0.0005, 0.01, 500))
        # Add extra return
        portfolio = 1.0 * market + 0.001 + np.random.normal(0, 0.003, 500)
        alpha = jensens_alpha(pd.Series(portfolio), market)
        assert alpha > 0

    def test_zero_alpha_for_market(self):
        """Market itself should have near-zero alpha vs itself."""
        np.random.seed(42)
        market = pd.Series(np.random.normal(0.0005, 0.01, 500))
        alpha = jensens_alpha(market, market)
        assert abs(alpha) < 0.01


class TestAppraisalRatio:
    """Tests for appraisal_ratio."""

    def test_returns_float(self, market_portfolio_data):
        portfolio, market = market_portfolio_data
        ar = appraisal_ratio(portfolio, market)
        assert isinstance(ar, float)

    def test_identical_returns(self):
        """Same returns should give zero residual vol -> zero ratio."""
        np.random.seed(42)
        market = pd.Series(np.random.normal(0.0005, 0.01, 100))
        ar = appraisal_ratio(market, market)
        assert ar == 0.0


class TestCaptureRatios:
    """Tests for capture_ratios."""

    def test_returns_dict(self, market_portfolio_data):
        portfolio, market = market_portfolio_data
        caps = capture_ratios(portfolio, market)
        assert "up_capture" in caps
        assert "down_capture" in caps
        assert "capture_ratio" in caps

    def test_perfect_tracking(self):
        """Portfolio = benchmark should give 100% capture both ways."""
        np.random.seed(42)
        market = pd.Series(np.random.normal(0.0005, 0.01, 500))
        caps = capture_ratios(market, market)
        assert caps["up_capture"] == pytest.approx(100.0, rel=0.01)
        assert caps["down_capture"] == pytest.approx(100.0, rel=0.01)

    def test_defensive_portfolio(self):
        """Portfolio with beta < 1 should have lower capture ratios."""
        np.random.seed(42)
        market = pd.Series(np.random.normal(0.0005, 0.01, 500))
        defensive = 0.5 * market + np.random.normal(0, 0.002, 500)
        caps = capture_ratios(pd.Series(defensive), market)
        assert caps["up_capture"] < 100
        assert caps["down_capture"] < 100

    def test_capture_ratio_positive(self, market_portfolio_data):
        portfolio, market = market_portfolio_data
        caps = capture_ratios(portfolio, market)
        assert caps["capture_ratio"] > 0
