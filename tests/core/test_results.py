"""Tests for standardized result containers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from wraquant.core.results import BacktestResult, ForecastResult, GARCHResult


class TestGARCHResult:
    def test_construction(self) -> None:
        result = GARCHResult(
            params={"omega": 0.01, "alpha[1]": 0.08, "beta[1]": 0.90},
            conditional_volatility=pd.Series([0.01, 0.012, 0.011]),
            standardized_residuals=np.array([0.5, -1.2, 0.3]),
            aic=100.0,
            bic=110.0,
            log_likelihood=-48.0,
            persistence=0.98,
            half_life=34.3,
            unconditional_variance=0.0005,
        )
        assert result.aic == 100.0
        assert result.bic == 110.0
        assert result.persistence == 0.98
        assert result.half_life == 34.3
        assert result.model is None
        assert result.ljung_box is None

    def test_optional_fields(self) -> None:
        result = GARCHResult(
            params={"omega": 0.01},
            conditional_volatility=pd.Series([0.01]),
            standardized_residuals=np.array([0.5]),
            aic=50.0,
            bic=55.0,
            log_likelihood=-24.0,
            persistence=0.95,
            half_life=13.5,
            unconditional_variance=0.001,
            model="mock_model",
            ljung_box={"statistic": 5.0, "p_value": 0.42},
        )
        assert result.model == "mock_model"
        assert result.ljung_box["p_value"] == 0.42

    def test_params_dict(self) -> None:
        params = {"omega": 0.01, "alpha[1]": 0.08, "beta[1]": 0.90}
        result = GARCHResult(
            params=params,
            conditional_volatility=pd.Series([0.01]),
            standardized_residuals=np.array([0.5]),
            aic=50.0,
            bic=55.0,
            log_likelihood=-24.0,
            persistence=0.98,
            half_life=34.3,
            unconditional_variance=0.0005,
        )
        assert result.params == params
        assert result.params["alpha[1]"] == 0.08


class TestBacktestResult:
    def test_construction(self) -> None:
        returns = pd.Series([0.01, -0.005, 0.008, -0.002])
        equity = pd.Series([100, 101, 100.5, 101.3, 101.1])
        metrics = {"sharpe_ratio": 1.5, "max_drawdown": -0.02}
        result = BacktestResult(
            returns=returns,
            equity_curve=equity,
            metrics=metrics,
        )
        assert len(result.returns) == 4
        assert len(result.equity_curve) == 5
        assert result.trades == 0
        assert result.signals is None

    def test_sharpe_property(self) -> None:
        result = BacktestResult(
            returns=pd.Series([0.01]),
            equity_curve=pd.Series([100, 101]),
            metrics={"sharpe_ratio": 1.5, "max_drawdown": -0.02},
        )
        assert result.sharpe == 1.5

    def test_max_drawdown_property(self) -> None:
        result = BacktestResult(
            returns=pd.Series([0.01]),
            equity_curve=pd.Series([100, 101]),
            metrics={"sharpe_ratio": 1.5, "max_drawdown": -0.02},
        )
        assert result.max_drawdown == -0.02

    def test_missing_metric_returns_nan(self) -> None:
        result = BacktestResult(
            returns=pd.Series([0.01]),
            equity_curve=pd.Series([100, 101]),
            metrics={},
        )
        assert math.isnan(result.sharpe)
        assert math.isnan(result.max_drawdown)

    def test_with_trades_and_signals(self) -> None:
        trades = [{"entry": 100, "exit": 105, "pnl": 5}]
        signals = pd.Series([1, 0, -1, 0])
        result = BacktestResult(
            returns=pd.Series([0.01, -0.005, 0.008, -0.002]),
            equity_curve=pd.Series([100, 101, 100.5, 101.3, 101.1]),
            metrics={"sharpe_ratio": 1.2},
            trades=trades,
            signals=signals,
        )
        assert len(result.trades) == 1
        assert result.trades[0]["pnl"] == 5
        assert len(result.signals) == 4


class TestForecastResult:
    def test_construction(self) -> None:
        fc = pd.Series([1.1, 1.2, 1.3])
        result = ForecastResult(forecast=fc)
        assert len(result.forecast) == 3
        assert result.confidence_lower is None
        assert result.confidence_upper is None
        assert result.method == ""
        assert result.fitted_values is None
        assert result.residuals is None
        assert result.metrics == {}
        assert result.model is None

    def test_full_construction(self) -> None:
        fc = np.array([1.1, 1.2, 1.3])
        lower = np.array([0.9, 1.0, 1.1])
        upper = np.array([1.3, 1.4, 1.5])
        result = ForecastResult(
            forecast=fc,
            confidence_lower=lower,
            confidence_upper=upper,
            method="ARIMA",
            fitted_values=np.array([1.0, 1.05]),
            residuals=np.array([0.01, -0.02]),
            metrics={"aic": 50.0, "rmse": 0.015},
            model="mock_arima",
        )
        assert result.method == "ARIMA"
        np.testing.assert_array_equal(result.confidence_lower, lower)
        np.testing.assert_array_equal(result.confidence_upper, upper)
        assert result.metrics["aic"] == 50.0
        assert result.model == "mock_arima"

    def test_default_metrics_is_empty_dict(self) -> None:
        r1 = ForecastResult(forecast=np.array([1.0]))
        r2 = ForecastResult(forecast=np.array([2.0]))
        # Ensure default_factory creates independent dicts
        r1.metrics["test"] = 1.0
        assert "test" not in r2.metrics
