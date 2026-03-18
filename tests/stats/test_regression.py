"""Tests for regression models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wraquant.stats.regression import (
    fama_macbeth,
    newey_west_ols,
    ols,
    rolling_ols,
    wls,
)


def _make_linear_data(n: int = 200, seed: int = 42) -> tuple[pd.Series, pd.DataFrame]:
    """Create y = 2*x + 3 + noise."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    x = rng.normal(0, 1, size=n)
    noise = rng.normal(0, 0.5, size=n)
    y = 2 * x + 3 + noise
    return (
        pd.Series(y, index=dates, name="y"),
        pd.DataFrame({"x": x}, index=dates),
    )


class TestOLS:
    def test_recovers_known_coefficients(self) -> None:
        y, X = _make_linear_data()
        result = ols(y, X, add_constant=True)
        coeffs = result["coefficients"]
        # coeffs[0] = intercept ~ 3, coeffs[1] = slope ~ 2
        assert abs(coeffs[0] - 3.0) < 0.5
        assert abs(coeffs[1] - 2.0) < 0.5

    def test_r_squared_between_0_and_1(self) -> None:
        y, X = _make_linear_data()
        result = ols(y, X)
        assert 0 <= result["r_squared"] <= 1
        assert 0 <= result["adj_r_squared"] <= 1

    def test_output_keys(self) -> None:
        y, X = _make_linear_data()
        result = ols(y, X)
        expected_keys = {
            "coefficients",
            "t_stats",
            "p_values",
            "r_squared",
            "adj_r_squared",
            "residuals",
        }
        assert set(result.keys()) == expected_keys

    def test_residuals_length(self) -> None:
        y, X = _make_linear_data(n=100)
        result = ols(y, X)
        assert len(result["residuals"]) == 100

    def test_high_r_squared_for_strong_signal(self) -> None:
        y, X = _make_linear_data()
        result = ols(y, X)
        # y = 2x + 3 + small noise => R^2 should be high
        assert result["r_squared"] > 0.8

    def test_no_constant(self) -> None:
        y, X = _make_linear_data()
        result = ols(y, X, add_constant=False)
        # Only one coefficient (no intercept)
        assert len(result["coefficients"]) == 1


class TestRollingOLS:
    def test_output_shapes(self) -> None:
        y, X = _make_linear_data(n=200)
        result = rolling_ols(y, X, window=60)
        assert result["coefficients"].shape[0] == 200
        assert len(result["r_squared"]) == 200

    def test_nan_in_warmup(self) -> None:
        y, X = _make_linear_data(n=200)
        result = rolling_ols(y, X, window=60)
        # First 59 values should be NaN
        assert result["r_squared"].iloc[:59].isna().all()

    def test_valid_after_warmup(self) -> None:
        y, X = _make_linear_data(n=200)
        result = rolling_ols(y, X, window=60)
        # After warmup, values should be valid
        r2_valid = result["r_squared"].dropna()
        assert len(r2_valid) > 0
        assert (r2_valid >= 0).all()
        assert (r2_valid <= 1).all()


class TestWLS:
    def test_output_keys(self) -> None:
        y, X = _make_linear_data()
        rng = np.random.default_rng(42)
        weights = rng.uniform(0.5, 2.0, size=len(y))
        result = wls(y, X, weights=weights)
        expected_keys = {
            "coefficients",
            "t_stats",
            "p_values",
            "r_squared",
            "adj_r_squared",
            "residuals",
        }
        assert set(result.keys()) == expected_keys

    def test_r_squared_valid(self) -> None:
        y, X = _make_linear_data()
        weights = np.ones(len(y))
        result = wls(y, X, weights=weights)
        assert 0 <= result["r_squared"] <= 1


class TestFamaMacBeth:
    def test_output_structure(self) -> None:
        rng = np.random.default_rng(42)
        n_periods = 60
        n_assets = 20
        dates = pd.bdate_range("2020-01-01", periods=n_periods)
        assets = [f"asset_{i}" for i in range(n_assets)]

        # Panel of returns
        panel_y = pd.DataFrame(
            rng.normal(0.001, 0.02, size=(n_periods, n_assets)),
            index=dates,
            columns=assets,
        )

        # Single factor exposure
        panel_X = pd.DataFrame(
            rng.normal(0, 1, size=(n_periods, n_assets)),
            index=dates,
            columns=assets,
        )

        result = fama_macbeth(panel_y, panel_X)
        assert "risk_premia" in result
        assert "t_stats" in result
        assert "r_squared" in result
        assert "gamma_series" in result

    def test_risk_premia_shape(self) -> None:
        rng = np.random.default_rng(42)
        n_periods = 60
        n_assets = 20
        dates = pd.bdate_range("2020-01-01", periods=n_periods)
        assets = [f"asset_{i}" for i in range(n_assets)]

        panel_y = pd.DataFrame(
            rng.normal(0.001, 0.02, size=(n_periods, n_assets)),
            index=dates,
            columns=assets,
        )
        panel_X = pd.DataFrame(
            rng.normal(0, 1, size=(n_periods, n_assets)),
            index=dates,
            columns=assets,
        )

        result = fama_macbeth(panel_y, panel_X)
        # risk_premia should have 2 entries: intercept + 1 factor
        assert len(result["risk_premia"]) == 2


class TestNeweyWestOLS:
    def test_output_keys(self) -> None:
        y, X = _make_linear_data()
        result = newey_west_ols(y, X)
        expected_keys = {
            "coefficients",
            "t_stats",
            "p_values",
            "r_squared",
            "adj_r_squared",
            "residuals",
            "hac_se",
        }
        assert set(result.keys()) == expected_keys

    def test_r_squared_matches_ols(self) -> None:
        y, X = _make_linear_data()
        nw_result = newey_west_ols(y, X)
        ols_result = ols(y, X)
        # R-squared should be the same; only standard errors differ
        np.testing.assert_allclose(
            nw_result["r_squared"], ols_result["r_squared"], atol=1e-10
        )

    def test_hac_se_positive(self) -> None:
        y, X = _make_linear_data()
        result = newey_west_ols(y, X)
        assert (result["hac_se"] > 0).all()
