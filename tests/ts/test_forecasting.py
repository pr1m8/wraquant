"""Tests for ARIMA diagnostics, model selection, and deep forecasting."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.arima.model import ARIMA

from wraquant.ts.forecasting import (
    arima_diagnostics,
    arima_model_selection,
    auto_forecast,
    ensemble_forecast,
    forecast_evaluation,
    holt_winters_forecast,
    rolling_forecast,
    ses_forecast,
    theta_forecast,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_walk(n: int = 300, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    values = np.cumsum(rng.normal(0, 1, n))
    return pd.Series(values, index=dates, name="rw")


def _stationary_ar1(n: int = 300, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    values = np.zeros(n)
    for i in range(1, n):
        values[i] = 0.5 * values[i - 1] + rng.normal(0, 1)
    return pd.Series(values, index=dates, name="ar1")


def _fit_arima(data: pd.Series, order: tuple = (1, 1, 1)):
    """Fit ARIMA and return result, suppressing convergence warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(data, order=order)
        return model.fit()


# ---------------------------------------------------------------------------
# arima_diagnostics
# ---------------------------------------------------------------------------


class TestArimaDiagnostics:
    def test_output_keys(self) -> None:
        data = _random_walk()
        fit = _fit_arima(data)
        result = arima_diagnostics(fit)
        expected = {
            "ljung_box",
            "jarque_bera",
            "arch_lm",
            "durbin_watson",
            "acf_values",
            "pacf_values",
            "model_adequate",
        }
        assert expected == set(result.keys())

    def test_ljung_box_structure(self) -> None:
        data = _random_walk()
        fit = _fit_arima(data)
        result = arima_diagnostics(fit)
        lb = result["ljung_box"]
        assert "statistic" in lb
        assert "p_value" in lb
        assert "pass" in lb
        assert isinstance(lb["pass"], bool)

    def test_jarque_bera_structure(self) -> None:
        data = _random_walk()
        fit = _fit_arima(data)
        result = arima_diagnostics(fit)
        jb = result["jarque_bera"]
        assert "statistic" in jb
        assert "p_value" in jb
        assert "pass" in jb

    def test_arch_lm_structure(self) -> None:
        data = _random_walk()
        fit = _fit_arima(data)
        result = arima_diagnostics(fit)
        arch = result["arch_lm"]
        assert "statistic" in arch
        assert "p_value" in arch
        assert "pass" in arch

    def test_durbin_watson_near_two(self) -> None:
        data = _random_walk()
        fit = _fit_arima(data, order=(1, 1, 0))
        result = arima_diagnostics(fit)
        dw = result["durbin_watson"]
        # DW should be near 2 for well-fitted model
        assert 0.5 < dw < 3.5

    def test_acf_pacf_arrays(self) -> None:
        data = _random_walk()
        fit = _fit_arima(data)
        result = arima_diagnostics(fit, nlags=5)
        assert len(result["acf_values"]) > 1
        assert len(result["pacf_values"]) > 1

    def test_model_adequate_is_bool(self) -> None:
        data = _random_walk()
        fit = _fit_arima(data)
        result = arima_diagnostics(fit)
        assert isinstance(result["model_adequate"], (bool, np.bool_))

    def test_good_model_passes_ljungbox(self) -> None:
        # AR(1) data fitted with AR(1) should pass autocorrelation test
        data = _stationary_ar1(500)
        fit = _fit_arima(data, order=(1, 0, 0))
        result = arima_diagnostics(fit)
        assert result["ljung_box"]["pass"] is True

    def test_raises_for_bad_input(self) -> None:
        with pytest.raises(AttributeError):
            arima_diagnostics("not a model")


# ---------------------------------------------------------------------------
# arima_model_selection
# ---------------------------------------------------------------------------


class TestArimaModelSelection:
    def test_returns_dataframe(self) -> None:
        data = _random_walk(200)
        result = arima_model_selection(
            data, p_range=range(0, 2), d_range=range(0, 2), q_range=range(0, 2)
        )
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self) -> None:
        data = _random_walk(200)
        result = arima_model_selection(
            data, p_range=range(0, 2), d_range=[1], q_range=range(0, 2)
        )
        expected_cols = {"order", "aic", "bic", "converged"}
        assert expected_cols.issubset(set(result.columns))

    def test_sorted_by_aic(self) -> None:
        data = _random_walk(200)
        result = arima_model_selection(
            data, p_range=range(0, 2), d_range=[1], q_range=range(0, 2)
        )
        aics = result["aic"].values
        assert np.all(aics[:-1] <= aics[1:])

    def test_sorted_by_bic(self) -> None:
        data = _random_walk(200)
        result = arima_model_selection(
            data,
            p_range=range(0, 2),
            d_range=[1],
            q_range=range(0, 2),
            criterion="bic",
        )
        bics = result["bic"].values
        assert np.all(bics[:-1] <= bics[1:])

    def test_includes_order_tuples(self) -> None:
        data = _random_walk(200)
        result = arima_model_selection(
            data, p_range=range(0, 2), d_range=[1], q_range=range(0, 2)
        )
        for order in result["order"]:
            assert isinstance(order, tuple)
            assert len(order) == 3

    def test_converged_column(self) -> None:
        data = _random_walk(200)
        result = arima_model_selection(
            data, p_range=range(0, 2), d_range=[1], q_range=range(0, 2)
        )
        assert result["converged"].dtype == bool

    def test_correct_number_of_rows(self) -> None:
        data = _random_walk(200)
        result = arima_model_selection(
            data, p_range=range(0, 3), d_range=[1], q_range=range(0, 3)
        )
        # 3 p values * 1 d value * 3 q values = 9 combinations
        assert len(result) == 9


# ---------------------------------------------------------------------------
# Seasonal data helper
# ---------------------------------------------------------------------------


def _seasonal_series(n: int = 200, seed: int = 42) -> pd.Series:
    """Create a synthetic seasonal series for testing."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="MS")
    trend = np.arange(n) * 0.3
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 12)
    noise = rng.normal(0, 1, n)
    return pd.Series(trend + seasonal + noise + 100, index=idx, name="seasonal")


# ---------------------------------------------------------------------------
# theta_forecast
# ---------------------------------------------------------------------------


class TestThetaForecast:
    def test_output_fields(self) -> None:
        from wraquant.core.results import ForecastResult

        data = _random_walk(200)
        result = theta_forecast(data, h=10)
        assert isinstance(result, ForecastResult)
        assert result.forecast is not None
        assert result.fitted_values is not None
        assert result.method == "theta"

    def test_forecast_length(self) -> None:
        data = _random_walk(200)
        result = theta_forecast(data, h=15)
        assert len(result.forecast) == 15

    def test_fitted_values_length(self) -> None:
        data = _random_walk(200)
        result = theta_forecast(data, h=5)
        assert len(result.fitted_values) == len(data)

    def test_theta_params(self) -> None:
        data = _random_walk(200)
        result = theta_forecast(data, h=10)
        params = result.metrics
        assert "theta" in params
        assert "ses_alpha" in params
        assert "drift" in params
        assert 0 <= params["ses_alpha"] <= 1


# ---------------------------------------------------------------------------
# ses_forecast
# ---------------------------------------------------------------------------


class TestSESForecast:
    def test_output_fields(self) -> None:
        from wraquant.core.results import ForecastResult

        data = _random_walk(200)
        result = ses_forecast(data, h=10)
        assert isinstance(result, ForecastResult)
        assert result.forecast is not None
        assert result.fitted_values is not None
        assert result.residuals is not None
        assert result.method == "ses"

    def test_forecast_length(self) -> None:
        data = _random_walk(200)
        result = ses_forecast(data, h=7)
        assert len(result.forecast) == 7

    def test_alpha_range(self) -> None:
        data = _random_walk(200)
        result = ses_forecast(data, h=5)
        assert 0 < result.metrics["alpha"] <= 1

    def test_fixed_alpha(self) -> None:
        data = _random_walk(200)
        result = ses_forecast(data, h=5, alpha=0.3)
        assert abs(result.metrics["alpha"] - 0.3) < 0.01

    def test_flat_forecast(self) -> None:
        """SES should produce a flat forecast (all values equal)."""
        data = _random_walk(200)
        result = ses_forecast(data, h=10)
        fcast = result.forecast.values
        assert np.allclose(fcast, fcast[0], atol=1e-10)


# ---------------------------------------------------------------------------
# holt_winters_forecast
# ---------------------------------------------------------------------------


class TestHoltWintersForecast:
    def test_output_fields(self) -> None:
        from wraquant.core.results import ForecastResult

        data = _seasonal_series()
        result = holt_winters_forecast(data, h=12, seasonal_periods=12)
        assert isinstance(result, ForecastResult)
        assert result.forecast is not None
        assert result.fitted_values is not None
        assert result.method == "holt_winters"

    def test_forecast_length(self) -> None:
        data = _seasonal_series()
        result = holt_winters_forecast(data, h=24, seasonal_periods=12)
        assert len(result.forecast) == 24

    def test_params_structure(self) -> None:
        data = _seasonal_series()
        result = holt_winters_forecast(data, h=12, seasonal_periods=12)
        params = result.metrics
        assert "alpha" in params
        assert "beta" in params
        assert "gamma" in params


# ---------------------------------------------------------------------------
# forecast_evaluation
# ---------------------------------------------------------------------------


class TestForecastEvaluation:
    def test_all_metrics_present(self) -> None:
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = np.array([1.1, 2.2, 2.8, 4.3, 4.7])
        result = forecast_evaluation(actual, pred)
        expected_keys = {
            "rmse", "mae", "mape", "smape", "mdape",
            "mase", "directional_accuracy",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_perfect_forecast(self) -> None:
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = forecast_evaluation(actual, actual)
        assert result["rmse"] == 0.0
        assert result["mae"] == 0.0

    def test_rmse_positive(self) -> None:
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        result = forecast_evaluation(actual, pred)
        assert result["rmse"] > 0
        assert result["mae"] > 0

    def test_diebold_mariano(self) -> None:
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred1 = np.array([1.1, 2.2, 2.8, 4.3, 4.7])
        pred2 = np.array([1.5, 2.5, 2.5, 4.5, 4.5])
        result = forecast_evaluation(actual, pred1, benchmark_forecast=pred2)
        assert "diebold_mariano" in result
        assert "statistic" in result["diebold_mariano"]
        assert "p_value" in result["diebold_mariano"]

    def test_directional_accuracy_range(self) -> None:
        rng = np.random.default_rng(42)
        actual = rng.normal(0, 1, 100).cumsum()
        pred = actual + rng.normal(0, 0.1, 100)
        result = forecast_evaluation(actual, pred)
        assert 0 <= result["directional_accuracy"] <= 100


# ---------------------------------------------------------------------------
# auto_forecast
# ---------------------------------------------------------------------------


class TestAutoForecast:
    def test_output_fields(self) -> None:
        from wraquant.core.results import ForecastResult

        data = _random_walk(200)
        result = auto_forecast(data, h=10)
        assert isinstance(result, ForecastResult)
        assert result.forecast is not None
        assert result.method != ""
        assert result.confidence_lower is not None
        assert result.confidence_upper is not None

    def test_best_model_is_valid(self) -> None:
        data = _random_walk(200)
        result = auto_forecast(data, h=10)
        valid_models = {"ses", "holt_winters", "theta", "seasonal_naive", "drift"}
        assert result.method in valid_models

    def test_forecast_length(self) -> None:
        data = _random_walk(200)
        result = auto_forecast(data, h=15)
        assert len(result.forecast) == 15

    def test_confidence_intervals(self) -> None:
        data = _random_walk(200)
        result = auto_forecast(data, h=10)
        assert len(result.confidence_lower) == 10
        assert len(result.confidence_upper) == 10
        # Upper should be above lower
        assert np.all(result.confidence_upper >= result.confidence_lower)

    def test_model_comparison_is_dataframe(self) -> None:
        data = _random_walk(200)
        result = auto_forecast(data, h=10)
        comp = result.model  # model holds the comparison DataFrame
        assert isinstance(comp, pd.DataFrame)
        assert "model" in comp.columns
        assert "rmse" in comp.columns


# ---------------------------------------------------------------------------
# ensemble_forecast
# ---------------------------------------------------------------------------


class TestEnsembleForecast:
    def test_output_keys(self) -> None:
        data = _random_walk(200)
        result = ensemble_forecast(data, h=10)
        expected = {"forecast", "weights", "individual_forecasts", "rmse_per_model"}
        assert expected.issubset(set(result.keys()))

    def test_weights_sum_to_one(self) -> None:
        data = _random_walk(200)
        result = ensemble_forecast(data, h=10)
        assert abs(sum(result["weights"].values()) - 1.0) < 1e-6

    def test_forecast_length(self) -> None:
        data = _random_walk(200)
        result = ensemble_forecast(data, h=12)
        assert len(result["forecast"]) == 12

    def test_individual_forecasts_present(self) -> None:
        data = _random_walk(200)
        result = ensemble_forecast(data, h=10)
        assert len(result["individual_forecasts"]) > 0


# ---------------------------------------------------------------------------
# rolling_forecast
# ---------------------------------------------------------------------------


class TestRollingForecast:
    def test_output_keys(self) -> None:
        data = _random_walk(100)

        def naive(train, h):
            return np.full(h, train.iloc[-1])

        result = rolling_forecast(data, naive, h=1, initial_window=50)
        expected = {"forecasts", "actuals", "errors", "cumulative_metrics"}
        assert expected.issubset(set(result.keys()))

    def test_forecasts_not_empty(self) -> None:
        data = _random_walk(100)

        def naive(train, h):
            return np.full(h, train.iloc[-1])

        result = rolling_forecast(data, naive, h=1, initial_window=50)
        assert len(result["forecasts"]) > 0

    def test_cumulative_metrics(self) -> None:
        data = _random_walk(100)

        def naive(train, h):
            return np.full(h, train.iloc[-1])

        result = rolling_forecast(data, naive, h=1, initial_window=50)
        assert "rmse" in result["cumulative_metrics"]
        assert "mae" in result["cumulative_metrics"]
        assert result["cumulative_metrics"]["rmse"] > 0
