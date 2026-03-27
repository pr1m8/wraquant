"""Tests for advanced risk and portfolio optimisation integrations."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

_has_pypfopt = importlib.util.find_spec("pypfopt") is not None
_has_riskfolio = importlib.util.find_spec("riskfolio") is not None
_has_skfolio = importlib.util.find_spec("skfolio") is not None
_has_copulas = importlib.util.find_spec("copulas") is not None
_has_copulae = importlib.util.find_spec("copulae") is not None
_has_pyvinecopulib = importlib.util.find_spec("pyvinecopulib") is not None
_has_pyextremes = importlib.util.find_spec("pyextremes") is not None


def _make_returns(n: int = 500, n_assets: int = 3, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    data = rng.normal(0.0003, 0.015, (n, n_assets))
    return pd.DataFrame(data, index=dates, columns=[f"A{i}" for i in range(n_assets)])


class TestPypfoptEfficientFrontier:
    @pytest.mark.skipif(not _has_pypfopt, reason="pypfopt not installed")
    def test_weights_sum_to_one(self) -> None:
        from wraquant.risk.integrations import pypfopt_efficient_frontier

        returns = _make_returns()
        mu = returns.mean() * 252
        cov = returns.cov() * 252
        result = pypfopt_efficient_frontier(mu, cov)
        assert sum(result["weights"].values()) == pytest.approx(1.0, abs=0.01)

    @pytest.mark.skipif(not _has_pypfopt, reason="pypfopt not installed")
    def test_returns_expected_keys(self) -> None:
        from wraquant.risk.integrations import pypfopt_efficient_frontier

        returns = _make_returns()
        mu = returns.mean() * 252
        cov = returns.cov() * 252
        result = pypfopt_efficient_frontier(mu, cov)
        assert "weights" in result
        assert "expected_return" in result
        assert "volatility" in result
        assert "sharpe_ratio" in result


class TestRiskfolioPortfolio:
    @pytest.mark.skipif(not _has_riskfolio, reason="riskfolio-lib not installed")
    def test_weights_sum_to_one(self) -> None:
        from wraquant.risk.integrations import riskfolio_portfolio

        returns = _make_returns()
        result = riskfolio_portfolio(returns, method="MV")
        assert sum(result["weights"].values()) == pytest.approx(1.0, abs=0.01)

    @pytest.mark.skipif(not _has_riskfolio, reason="riskfolio-lib not installed")
    def test_returns_method(self) -> None:
        from wraquant.risk.integrations import riskfolio_portfolio

        returns = _make_returns()
        result = riskfolio_portfolio(returns, method="CVaR")
        assert result["method"] == "CVaR"


class TestSkfolioOptimize:
    @pytest.mark.skipif(not _has_skfolio, reason="skfolio not installed")
    def test_weights_sum_to_one(self) -> None:
        from wraquant.risk.integrations import skfolio_optimize

        returns = _make_returns()
        result = skfolio_optimize(returns, objective="min_variance")
        total = sum(result["weights"].values())
        assert total == pytest.approx(1.0, abs=0.05)

    @pytest.mark.skipif(not _has_skfolio, reason="skfolio not installed")
    def test_returns_objective(self) -> None:
        from wraquant.risk.integrations import skfolio_optimize

        returns = _make_returns()
        result = skfolio_optimize(returns, objective="min_variance")
        assert result["objective"] == "min_variance"


class TestCopulasFit:
    @pytest.mark.skipif(not _has_copulas, reason="copulas not installed")
    def test_gaussian_copula(self) -> None:
        from wraquant.risk.integrations import copulas_fit

        rng = np.random.default_rng(42)
        data = pd.DataFrame(rng.normal(0, 1, (100, 3)), columns=["x", "y", "z"])
        result = copulas_fit(data, copula_type="gaussian")
        assert result["copula_type"] == "gaussian"
        assert result["columns"] == ["x", "y", "z"]
        assert result["n_samples"] == 100

    @pytest.mark.skipif(not _has_copulas, reason="copulas not installed")
    def test_unknown_copula_raises(self) -> None:
        from wraquant.risk.integrations import copulas_fit

        data = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        with pytest.raises(ValueError, match="Unknown copula_type"):
            copulas_fit(data, copula_type="invalid")


class TestVineCopula:
    @pytest.mark.skipif(not _has_pyvinecopulib, reason="pyvinecopulib not installed")
    def test_returns_expected_keys(self) -> None:
        from wraquant.risk.integrations import vine_copula

        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (200, 3))
        result = vine_copula(data)
        assert "vinecop" in result
        assert "n_vars" in result
        assert result["n_vars"] == 3
        assert "loglik" in result

    @pytest.mark.skipif(not _has_pyvinecopulib, reason="pyvinecopulib not installed")
    def test_dataframe_input(self) -> None:
        from wraquant.risk.integrations import vine_copula

        rng = np.random.default_rng(42)
        data = pd.DataFrame(rng.normal(0, 1, (200, 3)), columns=["a", "b", "c"])
        result = vine_copula(data)
        assert result["n_vars"] == 3


class TestExtremeValueAnalysis:
    @pytest.mark.skipif(not _has_pyextremes, reason="pyextremes not installed")
    def test_returns_expected_keys(self) -> None:
        from wraquant.risk.integrations import extreme_value_analysis

        rng = np.random.default_rng(42)
        dates = pd.date_range("2000-01-01", periods=5000, freq="D")
        data = pd.Series(rng.standard_t(df=3, size=5000), index=dates, name="losses")
        result = extreme_value_analysis(data)
        assert "shape" in result
        assert "loc" in result
        assert "scale" in result
        assert "return_levels" in result

    @pytest.mark.skipif(not _has_pyextremes, reason="pyextremes not installed")
    def test_return_levels_dict(self) -> None:
        from wraquant.risk.integrations import extreme_value_analysis

        rng = np.random.default_rng(42)
        dates = pd.date_range("2000-01-01", periods=5000, freq="D")
        data = pd.Series(rng.standard_t(df=3, size=5000), index=dates, name="losses")
        result = extreme_value_analysis(data)
        assert isinstance(result["return_levels"], dict)


# ---------------------------------------------------------------------------
# copulae library
# ---------------------------------------------------------------------------


class TestCopulaeFit:
    @pytest.mark.skipif(not _has_copulae, reason="copulae not installed")
    def test_gaussian_returns_expected_keys(self) -> None:
        from wraquant.risk.integrations import copulae_fit

        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (200, 3))
        result = copulae_fit(data, family="gaussian")
        assert "params" in result
        assert "log_likelihood" in result
        assert "aic" in result
        assert "bic" in result
        assert "fitted_copula" in result

    @pytest.mark.skipif(not _has_copulae, reason="copulae not installed")
    def test_dataframe_input(self) -> None:
        from wraquant.risk.integrations import copulae_fit

        rng = np.random.default_rng(42)
        data = pd.DataFrame(rng.normal(0, 1, (200, 3)), columns=["a", "b", "c"])
        result = copulae_fit(data, family="gaussian")
        assert "log_likelihood" in result
        assert isinstance(result["log_likelihood"], float)

    @pytest.mark.skipif(not _has_copulae, reason="copulae not installed")
    def test_unknown_family_raises(self) -> None:
        from wraquant.risk.integrations import copulae_fit

        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (100, 2))
        with pytest.raises(ValueError, match="Unknown family"):
            copulae_fit(data, family="invalid")

    @pytest.mark.skipif(not _has_copulae, reason="copulae not installed")
    def test_student_copula(self) -> None:
        from wraquant.risk.integrations import copulae_fit

        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (200, 2))
        result = copulae_fit(data, family="student")
        assert "params" in result
        assert "fitted_copula" in result
