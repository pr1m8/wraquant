"""Tests for credit risk models."""

from __future__ import annotations

import numpy as np
import pytest

from wraquant.risk.credit import (
    altman_z_score,
    cds_spread,
    credit_spread,
    default_probability,
    expected_loss,
    loss_given_default,
    merton_model,
)

# ---------------------------------------------------------------------------
# Merton model
# ---------------------------------------------------------------------------


class TestMertonModel:
    def test_returns_dict(self) -> None:
        result = merton_model(equity=100, debt=80, vol=0.3, rf_rate=0.05, maturity=1)
        expected_keys = {
            "asset_value",
            "asset_vol",
            "d1",
            "d2",
            "distance_to_default",
            "default_probability",
            "credit_spread",
        }
        assert set(result.keys()) == expected_keys

    def test_default_prob_between_0_and_1(self) -> None:
        result = merton_model(equity=100, debt=80, vol=0.3, rf_rate=0.05, maturity=1)
        assert 0 <= result["default_probability"] <= 1

    def test_high_leverage_higher_default_prob(self) -> None:
        low_lev = merton_model(equity=100, debt=50, vol=0.3, rf_rate=0.05, maturity=1)
        high_lev = merton_model(equity=100, debt=200, vol=0.3, rf_rate=0.05, maturity=1)
        assert high_lev["default_probability"] > low_lev["default_probability"]

    def test_higher_vol_higher_default_prob(self) -> None:
        low_vol = merton_model(equity=100, debt=80, vol=0.1, rf_rate=0.05, maturity=1)
        high_vol = merton_model(equity=100, debt=80, vol=0.6, rf_rate=0.05, maturity=1)
        assert high_vol["default_probability"] > low_vol["default_probability"]

    def test_credit_spread_nonnegative(self) -> None:
        result = merton_model(equity=100, debt=80, vol=0.3, rf_rate=0.05, maturity=1)
        assert result["credit_spread"] >= 0

    def test_asset_value_exceeds_equity(self) -> None:
        result = merton_model(equity=100, debt=80, vol=0.3, rf_rate=0.05, maturity=1)
        assert result["asset_value"] > 100

    def test_invalid_maturity(self) -> None:
        with pytest.raises(ValueError, match="maturity"):
            merton_model(equity=100, debt=80, vol=0.3, rf_rate=0.05, maturity=0)

    def test_invalid_equity(self) -> None:
        with pytest.raises(ValueError, match="equity"):
            merton_model(equity=-10, debt=80, vol=0.3, rf_rate=0.05, maturity=1)


# ---------------------------------------------------------------------------
# Altman Z-Score
# ---------------------------------------------------------------------------


class TestAltmanZScore:
    def test_safe_zone(self) -> None:
        result = altman_z_score(
            working_capital=50,
            total_assets=200,
            retained_earnings=80,
            ebit=40,
            market_cap=300,
            total_liabilities=100,
            sales=250,
        )
        assert result["zone"] == "safe"
        assert result["z_score"] > 2.99

    def test_distress_zone(self) -> None:
        result = altman_z_score(
            working_capital=-10,
            total_assets=200,
            retained_earnings=5,
            ebit=2,
            market_cap=20,
            total_liabilities=180,
            sales=50,
        )
        assert result["zone"] == "distress"
        assert result["z_score"] < 1.81

    def test_grey_zone(self) -> None:
        result = altman_z_score(
            working_capital=30,
            total_assets=200,
            retained_earnings=40,
            ebit=15,
            market_cap=120,
            total_liabilities=100,
            sales=180,
        )
        assert result["zone"] == "grey"
        assert 1.81 <= result["z_score"] <= 2.99

    def test_component_ratios(self) -> None:
        result = altman_z_score(
            working_capital=50,
            total_assets=200,
            retained_earnings=80,
            ebit=40,
            market_cap=300,
            total_liabilities=100,
            sales=250,
        )
        assert result["x1"] == pytest.approx(50 / 200)
        assert result["x5"] == pytest.approx(250 / 200)

    def test_invalid_total_assets(self) -> None:
        with pytest.raises(ValueError, match="total_assets"):
            altman_z_score(0, 0, 0, 0, 0, 100, 0)


# ---------------------------------------------------------------------------
# Default probability from transition matrix
# ---------------------------------------------------------------------------


class TestDefaultProbability:
    def test_identity_matrix(self) -> None:
        # No transitions — default prob should be 0 for non-default states
        n = 4
        mat = np.eye(n)
        probs = default_probability(mat, horizon=5)
        np.testing.assert_allclose(probs, 0.0, atol=1e-12)

    def test_absorbing_default(self) -> None:
        # Simple 3-state matrix: AA, BB, Default
        mat = np.array(
            [
                [0.9, 0.05, 0.05],
                [0.05, 0.85, 0.10],
                [0.0, 0.0, 1.0],
            ]
        )
        probs = default_probability(mat, horizon=1)
        assert probs[0] == pytest.approx(0.05)
        assert probs[1] == pytest.approx(0.10)

    def test_multi_period(self) -> None:
        mat = np.array(
            [
                [0.9, 0.05, 0.05],
                [0.05, 0.85, 0.10],
                [0.0, 0.0, 1.0],
            ]
        )
        p1 = default_probability(mat, horizon=1)
        p5 = default_probability(mat, horizon=5)
        # Default prob should increase over time
        assert np.all(p5 > p1)

    def test_invalid_horizon(self) -> None:
        with pytest.raises(ValueError, match="horizon"):
            default_probability(np.eye(3), horizon=0)


# ---------------------------------------------------------------------------
# Credit spread
# ---------------------------------------------------------------------------


class TestCreditSpread:
    def test_zero_default_prob(self) -> None:
        assert credit_spread(0.0, 0.4) == pytest.approx(0.0)

    def test_positive_spread(self) -> None:
        s = credit_spread(0.02, 0.4)
        assert s > 0

    def test_higher_pd_higher_spread(self) -> None:
        s1 = credit_spread(0.01, 0.4)
        s2 = credit_spread(0.05, 0.4)
        assert s2 > s1

    def test_full_recovery_zero_spread(self) -> None:
        s = credit_spread(0.05, 1.0)
        assert s == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Loss given default
# ---------------------------------------------------------------------------


class TestLossGivenDefault:
    def test_basic(self) -> None:
        assert loss_given_default(1_000_000, 0.4) == pytest.approx(600_000)

    def test_zero_recovery(self) -> None:
        assert loss_given_default(100, 0.0) == pytest.approx(100)

    def test_full_recovery(self) -> None:
        assert loss_given_default(100, 1.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Expected loss
# ---------------------------------------------------------------------------


class TestExpectedLoss:
    def test_basic(self) -> None:
        el = expected_loss(pd_val=0.02, lgd=0.6, ead=1_000_000)
        assert el == pytest.approx(12_000)

    def test_zero_pd(self) -> None:
        assert expected_loss(0.0, 0.5, 100) == pytest.approx(0.0)

    def test_invalid_pd(self) -> None:
        with pytest.raises(ValueError, match="pd_val"):
            expected_loss(-0.1, 0.5, 100)


# ---------------------------------------------------------------------------
# CDS spread
# ---------------------------------------------------------------------------


class TestCDSSpread:
    def test_nonnegative(self) -> None:
        s = cds_spread(0.02, 0.4, 5.0)
        assert s > 0

    def test_higher_intensity_higher_spread(self) -> None:
        s1 = cds_spread(0.01, 0.4, 5.0)
        s2 = cds_spread(0.05, 0.4, 5.0)
        assert s2 > s1

    def test_approximation(self) -> None:
        # For low intensity, spread ≈ lambda * (1 - R)
        lam = 0.01
        r = 0.4
        s = cds_spread(lam, r, 5.0)
        assert s == pytest.approx(lam * (1 - r), rel=0.1)

    def test_zero_intensity(self) -> None:
        s = cds_spread(0.0, 0.4, 5.0)
        assert s == pytest.approx(0.0)

    def test_invalid_maturity(self) -> None:
        with pytest.raises(ValueError, match="maturity"):
            cds_spread(0.01, 0.4, 0.0)
