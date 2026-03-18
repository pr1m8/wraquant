"""Tests for regression diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.econometrics.diagnostics import (
    breusch_godfrey,
    breusch_pagan,
    condition_number,
    durbin_watson,
    jarque_bera,
    ramsey_reset,
    vif,
    white_test,
)


def _ols_residuals(
    n: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate OLS residuals from a well-specified model."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
    y = X @ np.array([1.0, 2.0]) + rng.normal(0, 0.5, n)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    return resid, X


class TestDurbinWatson:
    def test_bounds(self) -> None:
        resid, _ = _ols_residuals()
        dw = durbin_watson(resid)
        assert 0.0 <= dw <= 4.0

    def test_no_autocorrelation(self) -> None:
        """Independent errors should produce DW near 2."""
        resid, _ = _ols_residuals(n=1000)
        dw = durbin_watson(resid)
        assert 1.5 < dw < 2.5

    def test_positive_autocorrelation(self) -> None:
        """AR(1) errors with positive rho should give DW < 2."""
        rng = np.random.default_rng(42)
        n = 500
        eps = np.empty(n)
        eps[0] = rng.normal()
        for t in range(1, n):
            eps[t] = 0.8 * eps[t - 1] + rng.normal(0, 0.5)
        dw = durbin_watson(eps)
        assert dw < 1.5

    def test_accepts_series(self) -> None:
        resid, _ = _ols_residuals()
        dw = durbin_watson(pd.Series(resid))
        assert 0.0 <= dw <= 4.0


class TestBreuschGodfrey:
    def test_output_structure(self) -> None:
        resid, X = _ols_residuals()
        result = breusch_godfrey(resid, X, nlags=4)

        assert "lm_statistic" in result
        assert "lm_p_value" in result
        assert "f_statistic" in result
        assert "f_p_value" in result
        assert "is_autocorrelated" in result

    def test_no_autocorrelation(self) -> None:
        resid, X = _ols_residuals(n=500)
        result = breusch_godfrey(resid, X, nlags=4)
        assert result["is_autocorrelated"] is False


class TestBreuschPagan:
    def test_output_structure(self) -> None:
        resid, X = _ols_residuals()
        result = breusch_pagan(resid, X)

        assert "lm_statistic" in result
        assert "lm_p_value" in result
        assert "f_statistic" in result
        assert "f_p_value" in result
        assert "is_heteroskedastic" in result

    def test_homoskedastic_data(self) -> None:
        resid, X = _ols_residuals(n=500)
        result = breusch_pagan(resid, X)
        # Should not reject for homoskedastic data
        assert result["is_heteroskedastic"] is False

    def test_heteroskedastic_data(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        x = rng.normal(0, 1, n)
        X = np.column_stack([np.ones(n), x])
        eps = rng.normal(0, 1, n) * (1 + 2 * np.abs(x))
        y = X @ np.array([1.0, 2.0]) + eps
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta

        result = breusch_pagan(resid, X)
        assert result["is_heteroskedastic"] is True


class TestWhiteTest:
    def test_output_structure(self) -> None:
        resid, X = _ols_residuals()
        result = white_test(resid, X)

        assert "lm_statistic" in result
        assert "lm_p_value" in result
        assert "is_heteroskedastic" in result


class TestJarqueBera:
    def test_with_normal_data(self) -> None:
        rng = np.random.default_rng(42)
        resid = rng.normal(0, 1, 1000)
        result = jarque_bera(resid)

        assert "statistic" in result
        assert "p_value" in result
        assert "skewness" in result
        assert "kurtosis" in result
        assert "is_normal" in result
        assert result["is_normal"] is True

    def test_with_non_normal_data(self) -> None:
        rng = np.random.default_rng(42)
        # Exponential distribution is clearly non-normal
        resid = rng.exponential(1.0, 1000)
        result = jarque_bera(resid)
        assert result["is_normal"] is False

    def test_skewness_near_zero_for_normal(self) -> None:
        rng = np.random.default_rng(42)
        resid = rng.normal(0, 1, 5000)
        result = jarque_bera(resid)
        assert abs(result["skewness"]) < 0.2


class TestRamseyReset:
    def test_output_structure(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        y = X @ np.array([1.0, 2.0]) + rng.normal(0, 0.5, n)

        result = ramsey_reset(y, X)

        assert "f_statistic" in result
        assert "p_value" in result
        assert "df_num" in result
        assert "df_denom" in result
        assert "is_misspecified" in result

    def test_correctly_specified(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        y = X @ np.array([1.0, 2.0]) + rng.normal(0, 0.5, n)

        result = ramsey_reset(y, X)
        assert result["is_misspecified"] is False


class TestVIF:
    def test_output_length(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(0, 1, (n, 3))
        result = vif(X)
        assert len(result) == 3

    def test_uncorrelated_low_vif(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        X = rng.normal(0, 1, (n, 3))
        result = vif(X)
        # Uncorrelated variables should have VIF near 1
        assert (result < 2.0).all()

    def test_collinear_high_vif(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        x1 = rng.normal(0, 1, n)
        x2 = x1 + rng.normal(0, 0.01, n)  # nearly collinear
        x3 = rng.normal(0, 1, n)
        X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

        result = vif(X)
        # x1 and x2 should have very high VIF
        assert result["x1"] > 10
        assert result["x2"] > 10

    def test_with_dataframe(self) -> None:
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(0, 1, (100, 2)), columns=["a", "b"])
        result = vif(X)
        assert isinstance(result, pd.Series)
        assert list(result.index) == ["a", "b"]


class TestConditionNumber:
    def test_positive(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 3))
        cn = condition_number(X)
        assert cn > 0

    def test_collinear_high(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        x1 = rng.normal(0, 1, n)
        X = np.column_stack([x1, x1 + rng.normal(0, 0.001, n)])
        cn = condition_number(X)
        assert cn > 100  # should be very large
