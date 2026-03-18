"""Tests for cross-sectional econometrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.econometrics.cross_section import (
    gmm_estimation,
    quantile_regression,
    robust_ols,
    sargan_test,
    two_stage_least_squares,
)


def _make_ols_data(
    n: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate OLS data with known coefficients [1, 2]."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
    y = X @ np.array([1.0, 2.0]) + rng.normal(0, 0.5, n)
    return y, X


class TestRobustOLS:
    def test_output_matches_ols_coefficients(self) -> None:
        """Robust OLS should have same point estimates as plain OLS."""
        import statsmodels.api as sm

        y, X = _make_ols_data()
        robust = robust_ols(y, X, cov_type="HC1")
        plain = sm.OLS(y, X).fit()

        np.testing.assert_allclose(
            robust["coefficients"], plain.params, atol=1e-10
        )

    def test_different_standard_errors(self) -> None:
        """HC1 standard errors should differ from non-robust ones."""
        import statsmodels.api as sm

        rng = np.random.default_rng(42)
        n = 200
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        # Heteroskedastic errors: variance proportional to X
        eps = rng.normal(0, 1, n) * (1 + np.abs(X[:, 1]))
        y = X @ np.array([1.0, 2.0]) + eps

        robust = robust_ols(y, X, cov_type="HC1")
        plain = sm.OLS(y, X).fit()

        # Standard errors should not be identical
        assert not np.allclose(robust["std_errors"], plain.bse)

    def test_output_structure(self) -> None:
        y, X = _make_ols_data()
        result = robust_ols(y, X)

        assert "coefficients" in result
        assert "std_errors" in result
        assert "t_stats" in result
        assert "p_values" in result
        assert "r_squared" in result
        assert "adj_r_squared" in result
        assert "residuals" in result
        assert "cov_type" in result
        assert "nobs" in result
        assert result["cov_type"] == "HC1"

    def test_all_hc_types(self) -> None:
        y, X = _make_ols_data()
        for hc in ("HC0", "HC1", "HC2", "HC3"):
            result = robust_ols(y, X, cov_type=hc)
            assert result["cov_type"] == hc
            assert len(result["coefficients"]) == 2


class TestQuantileRegression:
    def test_output_structure(self) -> None:
        y, X = _make_ols_data()
        result = quantile_regression(y, X, quantile=0.5)

        assert "coefficients" in result
        assert "std_errors" in result
        assert "t_stats" in result
        assert "p_values" in result
        assert "quantile" in result
        assert "pseudo_r_squared" in result
        assert "nobs" in result
        assert result["quantile"] == 0.5

    def test_median_close_to_ols(self) -> None:
        """For symmetric, homoskedastic errors, median ~ mean."""
        y, X = _make_ols_data(n=500)
        q50 = quantile_regression(y, X, quantile=0.5)
        ols = robust_ols(y, X)

        np.testing.assert_allclose(
            q50["coefficients"], ols["coefficients"], atol=0.3
        )

    def test_different_quantiles(self) -> None:
        y, X = _make_ols_data(n=500)
        q25 = quantile_regression(y, X, quantile=0.25)
        q75 = quantile_regression(y, X, quantile=0.75)

        # Intercepts should differ (lower quantile => lower intercept)
        assert q25["coefficients"][0] < q75["coefficients"][0]


class TestTwoStageLeastSquares:
    def test_with_valid_instruments(self) -> None:
        rng = np.random.default_rng(42)
        n = 500

        # z is instrument, correlated with x (endogenous) but not with eps
        z = rng.normal(0, 1, n)
        eps = rng.normal(0, 0.5, n)
        x = 0.8 * z + rng.normal(0, 0.3, n)  # first stage: x correlated with z
        y = 1.0 + 2.0 * x + eps  # true beta = 2

        X = np.column_stack([np.ones(n), x])
        instruments = z.reshape(-1, 1)

        result = two_stage_least_squares(y, X, instruments, endog_vars=[1])

        assert "coefficients" in result
        assert "std_errors" in result
        assert "first_stage_f" in result
        assert result["nobs"] == n

        # Should recover slope close to 2.0
        assert abs(result["coefficients"][1] - 2.0) < 0.5

    def test_order_condition_violated(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n), rng.normal(0, 1, n)])
        y = rng.normal(0, 1, n)
        instruments = rng.normal(0, 1, (n, 1))

        with pytest.raises(ValueError, match="Order condition"):
            two_stage_least_squares(y, X, instruments, endog_vars=[1, 2])

    def test_output_structure(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        z = rng.normal(0, 1, n)
        x = 0.8 * z + rng.normal(0, 0.3, n)
        y = 1.0 + 2.0 * x + rng.normal(0, 0.5, n)
        X = np.column_stack([np.ones(n), x])

        result = two_stage_least_squares(y, X, z.reshape(-1, 1), endog_vars=[1])

        assert "coefficients" in result
        assert "std_errors" in result
        assert "t_stats" in result
        assert "p_values" in result
        assert "r_squared" in result
        assert "residuals" in result
        assert "first_stage_f" in result


class TestGMMEstimation:
    def test_recovers_mean(self) -> None:
        """GMM with moment condition E[x - mu] = 0 should recover the mean."""
        rng = np.random.default_rng(42)
        data = rng.normal(5.0, 1.0, 200)

        def moment_cond(params, data):
            return (data - params[0]).reshape(-1, 1)

        result = gmm_estimation(moment_cond, np.array([0.0]), data=data)

        assert "params" in result
        assert abs(result["params"][0] - 5.0) < 0.3

    def test_output_structure(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)

        def moment_cond(params, data):
            return (data - params[0]).reshape(-1, 1)

        result = gmm_estimation(moment_cond, np.array([0.0]), data=data)

        assert "params" in result
        assert "objective" in result
        assert "moment_conditions_mean" in result
        assert "W" in result
        assert "nobs" in result
        assert "n_moments" in result


class TestSarganTest:
    def test_output_structure(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        residuals = rng.normal(0, 1, n)
        instruments = rng.normal(0, 1, (n, 3))

        result = sargan_test(residuals, instruments)

        assert "statistic" in result
        assert "p_value" in result
        assert "df" in result
        assert "is_valid" in result
        assert result["statistic"] >= 0

    def test_valid_instruments(self) -> None:
        """With truly exogenous instruments, should not reject."""
        rng = np.random.default_rng(42)
        n = 200
        residuals = rng.normal(0, 1, n)
        instruments = rng.normal(0, 1, (n, 2))  # uncorrelated with resids

        result = sargan_test(residuals, instruments)
        # Should generally not reject (p > 0.05)
        assert result["p_value"] > 0.01
