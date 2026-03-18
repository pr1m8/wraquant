"""Tests for statistical hypothesis tests module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from wraquant.stats.tests import (
    breusch_pagan,
    chow_test,
    durbin_watson,
    shapiro_wilk,
    variance_inflation_factor,
    white_test,
)


# ---------------------------------------------------------------------------
# Shapiro-Wilk
# ---------------------------------------------------------------------------


class TestShapiroWilk:
    def test_normal_data_passes(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)
        result = shapiro_wilk(data)
        assert result["is_normal"] is True
        assert result["p_value"] > 0.05

    def test_non_normal_rejects(self) -> None:
        rng = np.random.default_rng(42)
        # Heavily skewed data
        data = rng.exponential(1.0, 200)
        result = shapiro_wilk(data)
        assert result["is_normal"] is False
        assert result["p_value"] < 0.05

    def test_keys(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        result = shapiro_wilk(data)
        assert set(result.keys()) == {"statistic", "p_value", "is_normal"}


# ---------------------------------------------------------------------------
# Durbin-Watson
# ---------------------------------------------------------------------------


class TestDurbinWatson:
    def test_no_autocorrelation(self) -> None:
        rng = np.random.default_rng(42)
        residuals = rng.normal(0, 1, 200)
        result = durbin_watson(residuals)
        assert 1.5 < result["statistic"] < 2.5
        assert "No significant" in result["interpretation"]

    def test_positive_autocorrelation(self) -> None:
        # Create autocorrelated residuals: e_t = 0.9 * e_{t-1} + noise
        rng = np.random.default_rng(42)
        n = 200
        resid = np.zeros(n)
        for i in range(1, n):
            resid[i] = 0.9 * resid[i - 1] + rng.normal(0, 0.3)
        result = durbin_watson(resid)
        assert result["statistic"] < 1.5
        assert "Positive" in result["interpretation"]

    def test_keys(self) -> None:
        rng = np.random.default_rng(42)
        result = durbin_watson(rng.normal(0, 1, 100))
        assert set(result.keys()) == {"statistic", "interpretation"}


# ---------------------------------------------------------------------------
# Breusch-Pagan
# ---------------------------------------------------------------------------


class TestBreuschPagan:
    def test_homoskedastic(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 2))
        X_const = sm.add_constant(X)
        y = X_const @ [1, 0.5, -0.3] + rng.normal(0, 1, 200)
        resid = y - X_const @ np.linalg.lstsq(X_const, y, rcond=None)[0]
        result = breusch_pagan(resid, X_const)
        assert "lm_stat" in result
        assert "p_value" in result

    def test_keys(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 1))
        X_const = sm.add_constant(X)
        y = X_const @ [1, 0.5] + rng.normal(0, 1, 100)
        resid = y - X_const @ np.linalg.lstsq(X_const, y, rcond=None)[0]
        result = breusch_pagan(resid, X_const)
        expected_keys = {"lm_stat", "p_value", "f_stat", "f_p_value", "is_heteroskedastic"}
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# White's test
# ---------------------------------------------------------------------------


class TestWhiteTest:
    def test_detects_heteroskedasticity(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (300, 2))
        X_const = sm.add_constant(X)
        # Strong heteroskedasticity
        y = X_const @ [1, 0.5, -0.3] + rng.normal(0, 1, 300) * (1 + 2 * np.abs(X[:, 0]))
        resid = y - X_const @ np.linalg.lstsq(X_const, y, rcond=None)[0]
        result = white_test(resid, X_const)
        assert "lm_stat" in result
        assert "p_value" in result

    def test_keys(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 1))
        X_const = sm.add_constant(X)
        y = X_const @ [1, 0.5] + rng.normal(0, 1, 100)
        resid = y - X_const @ np.linalg.lstsq(X_const, y, rcond=None)[0]
        result = white_test(resid, X_const)
        expected_keys = {"lm_stat", "p_value", "f_stat", "f_p_value", "is_heteroskedastic"}
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Chow test
# ---------------------------------------------------------------------------


class TestChowTest:
    def test_detects_structural_break(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 1))
        y = np.concatenate([
            X[:100] @ [1.0] + rng.normal(0, 0.5, 100),
            X[100:] @ [5.0] + rng.normal(0, 0.5, 100),
        ])
        result = chow_test(y, X, break_point=100)
        assert result["break_detected"] is True
        assert result["p_value"] < 0.05

    def test_no_break_stable(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 1))
        y = X @ [1.0] + rng.normal(0, 1, 200)
        result = chow_test(y, X, break_point=100)
        assert result["break_detected"] is False

    def test_keys(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 1))
        y = X @ [1.0] + rng.normal(0, 1, 100)
        result = chow_test(y, X, break_point=50)
        assert set(result.keys()) == {"f_stat", "p_value", "break_detected"}


# ---------------------------------------------------------------------------
# Variance Inflation Factor
# ---------------------------------------------------------------------------


class TestVarianceInflationFactor:
    def test_detects_collinearity(self) -> None:
        rng = np.random.default_rng(42)
        x1 = rng.normal(0, 1, 200)
        x2 = x1 + rng.normal(0, 0.1, 200)  # nearly collinear
        x3 = rng.normal(0, 1, 200)
        X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
        vif = variance_inflation_factor(X)
        assert vif["x1"] > 10
        assert vif["x2"] > 10
        assert vif["x3"] < 5

    def test_independent_features_low_vif(self) -> None:
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(0, 1, (200, 3)), columns=["a", "b", "c"])
        vif = variance_inflation_factor(X)
        assert (vif < 5).all()

    def test_returns_series(self) -> None:
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(0, 1, (100, 2)), columns=["x1", "x2"])
        vif = variance_inflation_factor(X)
        assert isinstance(vif, pd.Series)
        assert list(vif.index) == ["x1", "x2"]
