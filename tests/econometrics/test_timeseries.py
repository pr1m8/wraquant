"""Tests for time series econometrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.econometrics.timeseries import (
    granger_causality,
    impulse_response,
    structural_break_test,
    var_model,
    variance_decomposition,
    vecm_model,
)


def _make_var_data(
    n: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate bivariate VAR(1) data with known coefficients.

    y1_t = 0.5 * y1_{t-1} + 0.3 * y2_{t-1} + e1_t
    y2_t = 0.0 * y1_{t-1} + 0.4 * y2_{t-1} + e2_t

    So y2 Granger-causes y1, but y1 does NOT Granger-cause y2.
    """
    rng = np.random.default_rng(seed)
    y = np.zeros((n, 2))
    y[0] = rng.normal(0, 1, 2)

    for t in range(1, n):
        y[t, 0] = 0.5 * y[t - 1, 0] + 0.3 * y[t - 1, 1] + rng.normal(0, 0.5)
        y[t, 1] = 0.4 * y[t - 1, 1] + rng.normal(0, 0.5)

    return pd.DataFrame(y, columns=["y1", "y2"])


class TestVARModel:
    def test_output_structure(self) -> None:
        data = _make_var_data()
        result = var_model(data, max_lags=4, ic="aic")

        assert "coefficients" in result
        assert "lag_order" in result
        assert "residuals" in result
        assert "sigma_u" in result
        assert "aic" in result
        assert "bic" in result
        assert "forecast" in result
        assert callable(result["forecast"])

    def test_lag_order_positive(self) -> None:
        data = _make_var_data()
        result = var_model(data, max_lags=4)
        assert result["lag_order"] >= 1

    def test_coefficient_shape(self) -> None:
        data = _make_var_data()
        result = var_model(data, max_lags=4)
        k = data.shape[1]
        p = result["lag_order"]
        coef = result["coefficients"]
        assert coef.shape[0] == k
        assert coef.shape[1] == k * p + 1  # +1 for intercept

    def test_forecast_callable(self) -> None:
        data = _make_var_data()
        result = var_model(data, max_lags=4)
        fc = result["forecast"](5)
        assert fc.shape == (5, 2)


class TestVECM:
    def test_output_structure(self) -> None:
        # Generate cointegrated data
        rng = np.random.default_rng(42)
        n = 300
        x = np.cumsum(rng.normal(0, 1, n))
        y = x + rng.normal(0, 0.5, n)  # cointegrated with x
        data = pd.DataFrame({"x": x, "y": y})

        result = vecm_model(data, k_ar_diff=1)

        assert "alpha" in result
        assert "beta" in result
        assert "gamma" in result
        assert "coint_rank" in result
        assert "residuals" in result
        assert result["coint_rank"] >= 1


class TestGrangerCausality:
    def test_with_known_causal_relationship(self) -> None:
        data = _make_var_data(n=500, seed=123)
        result = granger_causality(data, max_lag=5)

        # y2 should Granger-cause y1
        key_causal = "y2 -> y1"
        assert key_causal in result
        assert result[key_causal]["is_causal"] is True

    def test_output_structure(self) -> None:
        data = _make_var_data()
        result = granger_causality(data, max_lag=5)

        for key, val in result.items():
            assert "f_statistic" in val
            assert "p_value" in val
            assert "lag_order" in val
            assert "is_causal" in val


class TestImpulseResponse:
    def test_output_shape(self) -> None:
        # Simple 2-variable VAR(1) coefficient matrix
        A = np.array([[0.5, 0.3], [0.0, 0.4]])
        irf = impulse_response(A, n_periods=20, shock_var=0)
        assert irf.shape == (21, 2)

    def test_initial_shock(self) -> None:
        A = np.array([[0.5, 0.3], [0.0, 0.4]])
        irf = impulse_response(A, n_periods=10, shock_var=0)
        # At t=0, only the shocked variable should be 1
        assert irf[0, 0] == 1.0
        assert irf[0, 1] == 0.0

    def test_decays_to_zero(self) -> None:
        # Stationary VAR should decay
        A = np.array([[0.3, 0.1], [0.1, 0.3]])
        irf = impulse_response(A, n_periods=50, shock_var=0)
        # Last period should be near zero
        assert np.all(np.abs(irf[-1]) < 0.01)


class TestVarianceDecomposition:
    def test_output_shape(self) -> None:
        A = np.array([[0.5, 0.3], [0.0, 0.4]])
        fevd = variance_decomposition(A, n_periods=20)
        assert fevd.shape == (21, 2, 2)

    def test_sums_to_one(self) -> None:
        A = np.array([[0.5, 0.3], [0.0, 0.4]])
        fevd = variance_decomposition(A, n_periods=20)
        # At each horizon and for each variable, contributions sum to 1
        for h in range(1, 21):
            for i in range(2):
                np.testing.assert_allclose(fevd[h, i, :].sum(), 1.0, atol=1e-10)


class TestStructuralBreak:
    def test_chow_test(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        # True break at t=100
        y = np.empty(n)
        y[:100] = X[:100] @ np.array([1.0, 2.0]) + rng.normal(0, 0.3, 100)
        y[100:] = X[100:] @ np.array([1.0, 5.0]) + rng.normal(0, 0.3, 100)

        result = structural_break_test(y, X, method="chow", break_point=100)

        assert "f_statistic" in result
        assert "p_value" in result
        assert "is_break" in result
        assert result["is_break"] is True

    def test_chow_no_break(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        y = X @ np.array([1.0, 2.0]) + rng.normal(0, 0.3, n)

        result = structural_break_test(y, X, method="chow", break_point=100)
        # Should not detect a break
        assert result["is_break"] is False

    def test_chow_requires_break_point(self) -> None:
        with pytest.raises(ValueError, match="break_point"):
            structural_break_test(np.ones(100), method="chow")

    def test_sup_f(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        y = np.empty(n)
        y[:100] = X[:100] @ np.array([1.0, 2.0]) + rng.normal(0, 0.3, 100)
        y[100:] = X[100:] @ np.array([1.0, 5.0]) + rng.normal(0, 0.3, 100)

        result = structural_break_test(y, X, method="sup_f")
        assert "f_statistic" in result
        assert "break_point" in result
