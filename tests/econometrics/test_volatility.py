"""Tests for volatility econometrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.econometrics.volatility import (
    arch_test,
    garch_numpy_fallback,
)

# Conditionally import arch-dependent functions
try:
    from arch import arch_model as _arch_model  # noqa: F401

    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False


def _make_garch_returns(n: int = 1000, seed: int = 42) -> np.ndarray:
    """Simulate GARCH(1,1) returns."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 0.00001, 0.08, 0.90
    sigma2 = np.empty(n)
    returns = np.empty(n)
    sigma2[0] = omega / (1 - alpha - beta)
    returns[0] = rng.normal(0, np.sqrt(sigma2[0]))

    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        returns[t] = rng.normal(0, np.sqrt(sigma2[t]))

    return returns


class TestArchTest:
    def test_output_structure(self) -> None:
        returns = _make_garch_returns()
        result = arch_test(returns, nlags=5)

        assert "statistic" in result
        assert "p_value" in result
        assert "f_statistic" in result
        assert "f_p_value" in result
        assert "is_arch" in result

    def test_detects_arch_effects(self) -> None:
        returns = _make_garch_returns()
        result = arch_test(returns, nlags=5)
        assert result["is_arch"] is True

    def test_no_arch_in_iid_data(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 1000)
        result = arch_test(returns, nlags=5)
        assert result["is_arch"] is False

    def test_accepts_series(self) -> None:
        returns = pd.Series(_make_garch_returns())
        result = arch_test(returns)
        assert "statistic" in result


class TestGarchNumpyFallback:
    def test_output_structure(self) -> None:
        returns = _make_garch_returns()
        result = garch_numpy_fallback(returns)

        assert "params" in result
        assert "conditional_volatility" in result
        assert "standardized_residuals" in result
        assert "forecast" in result
        assert "loglikelihood" in result

    def test_params_reasonable(self) -> None:
        returns = _make_garch_returns(n=2000)
        result = garch_numpy_fallback(returns)

        params = result["params"]
        assert params["omega"] > 0
        assert 0 < params["alpha[1]"] < 1
        assert 0 < params["beta[1]"] < 1
        assert params["alpha[1]"] + params["beta[1]"] < 1

    def test_conditional_vol_positive(self) -> None:
        returns = _make_garch_returns()
        result = garch_numpy_fallback(returns)
        assert np.all(result["conditional_volatility"] > 0)

    def test_forecast_positive(self) -> None:
        returns = _make_garch_returns()
        result = garch_numpy_fallback(returns)
        assert result["forecast"] > 0


@pytest.mark.skipif(not HAS_ARCH, reason="arch library not installed")
class TestGarch:
    def test_output_structure(self) -> None:
        from wraquant.econometrics.volatility import garch

        returns = _make_garch_returns()
        result = garch(returns, p=1, q=1)

        assert "params" in result
        assert "conditional_volatility" in result
        assert "standardized_residuals" in result
        assert "forecast" in result
        assert "aic" in result
        assert "bic" in result
        assert "loglikelihood" in result

    def test_conditional_vol_shape(self) -> None:
        from wraquant.econometrics.volatility import garch

        returns = _make_garch_returns()
        result = garch(returns)
        assert len(result["conditional_volatility"]) == len(returns)


@pytest.mark.skipif(not HAS_ARCH, reason="arch library not installed")
class TestEGarch:
    def test_output_structure(self) -> None:
        from wraquant.econometrics.volatility import egarch

        returns = _make_garch_returns()
        result = egarch(returns)

        assert "params" in result
        assert "conditional_volatility" in result
        assert "standardized_residuals" in result
        assert "forecast" in result
        assert "aic" in result
        assert "bic" in result


@pytest.mark.skipif(not HAS_ARCH, reason="arch library not installed")
class TestGJRGarch:
    def test_output_structure(self) -> None:
        from wraquant.econometrics.volatility import gjr_garch

        returns = _make_garch_returns()
        result = gjr_garch(returns)

        assert "params" in result
        assert "conditional_volatility" in result
        assert "standardized_residuals" in result
        assert "forecast" in result
        assert "aic" in result
        assert "bic" in result
