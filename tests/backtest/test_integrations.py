"""Tests for advanced backtesting integrations."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

_has_vectorbt = importlib.util.find_spec("vectorbt") is not None
_has_quantstats = importlib.util.find_spec("quantstats") is not None
_has_empyrical = importlib.util.find_spec("empyrical") is not None
_has_ffn = importlib.util.find_spec("ffn") is not None


def _make_returns(n: int = 252, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(rng.normal(0.0005, 0.02, n), index=dates, name="returns")


def _make_prices(n: int = 252, seed: int = 42) -> pd.Series:
    returns = _make_returns(n, seed)
    prices = 100 * np.exp(returns.cumsum())
    prices.name = "price"
    return prices


class TestVectorbtBacktest:
    @pytest.mark.skipif(not _has_vectorbt, reason="vectorbt not installed")
    def test_returns_expected_keys(self) -> None:
        from wraquant.backtest.integrations import vectorbt_backtest

        prices = _make_prices()
        entries = pd.Series(False, index=prices.index)
        exits = pd.Series(False, index=prices.index)
        entries.iloc[::20] = True
        exits.iloc[10::20] = True
        result = vectorbt_backtest(prices, entries, exits)
        assert "total_return" in result
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result
        assert "total_trades" in result
        assert "portfolio" in result


class TestQuantstatsReport:
    @pytest.mark.skipif(not _has_quantstats, reason="quantstats not installed")
    def test_returns_metrics(self) -> None:
        from wraquant.backtest.integrations import quantstats_report

        returns = _make_returns()
        result = quantstats_report(returns)
        assert "sharpe" in result
        assert "sortino" in result
        assert "max_drawdown" in result
        assert "cagr" in result
        assert "volatility" in result
        assert "calmar" in result

    @pytest.mark.skipif(not _has_quantstats, reason="quantstats not installed")
    def test_metrics_are_floats(self) -> None:
        from wraquant.backtest.integrations import quantstats_report

        returns = _make_returns()
        result = quantstats_report(returns)
        for key in ("sharpe", "sortino", "max_drawdown", "cagr", "volatility"):
            assert isinstance(result[key], float)


class TestEmpyricalMetrics:
    @pytest.mark.skipif(not _has_empyrical, reason="empyrical not installed")
    def test_all_keys_present(self) -> None:
        from wraquant.backtest.integrations import empyrical_metrics

        returns = _make_returns()
        result = empyrical_metrics(returns)
        expected_keys = {
            "annual_return",
            "annual_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "omega_ratio",
            "tail_ratio",
            "stability",
        }
        assert expected_keys.issubset(result.keys())

    @pytest.mark.skipif(not _has_empyrical, reason="empyrical not installed")
    def test_values_are_floats(self) -> None:
        from wraquant.backtest.integrations import empyrical_metrics

        returns = _make_returns()
        result = empyrical_metrics(returns)
        for key, val in result.items():
            assert isinstance(val, float), f"{key} is not float"

    @pytest.mark.skipif(not _has_empyrical, reason="empyrical not installed")
    def test_max_drawdown_negative(self) -> None:
        from wraquant.backtest.integrations import empyrical_metrics

        returns = _make_returns()
        result = empyrical_metrics(returns)
        assert result["max_drawdown"] <= 0


class TestPyfolioTearsheetData:
    @pytest.mark.skipif(not _has_empyrical, reason="empyrical not installed")
    def test_returns_expected_keys(self) -> None:
        from wraquant.backtest.integrations import pyfolio_tearsheet_data

        returns = _make_returns()
        result = pyfolio_tearsheet_data(returns)
        assert "returns" in result
        assert "cum_returns" in result
        assert "drawdown" in result

    @pytest.mark.skipif(not _has_empyrical, reason="empyrical not installed")
    def test_cum_returns_starts_at_one(self) -> None:
        from wraquant.backtest.integrations import pyfolio_tearsheet_data

        returns = _make_returns()
        result = pyfolio_tearsheet_data(returns)
        assert result["cum_returns"].iloc[0] == pytest.approx(1.0, rel=0.1)

    @pytest.mark.skipif(not _has_empyrical, reason="empyrical not installed")
    def test_drawdown_nonpositive(self) -> None:
        from wraquant.backtest.integrations import pyfolio_tearsheet_data

        returns = _make_returns()
        result = pyfolio_tearsheet_data(returns)
        assert (result["drawdown"] <= 0).all()


class TestFfnStats:
    @pytest.mark.skipif(not _has_ffn, reason="ffn not installed")
    def test_returns_expected_keys(self) -> None:
        from wraquant.backtest.integrations import ffn_stats

        prices = _make_prices()
        result = ffn_stats(prices)
        assert "total_return" in result
        assert "cagr" in result
        assert "daily_sharpe" in result
        assert "max_drawdown" in result
        assert "stats_object" in result

    @pytest.mark.skipif(not _has_ffn, reason="ffn not installed")
    def test_values_are_floats(self) -> None:
        from wraquant.backtest.integrations import ffn_stats

        prices = _make_prices()
        result = ffn_stats(prices)
        for key in ("total_return", "cagr", "daily_sharpe", "max_drawdown"):
            assert isinstance(result[key], float), f"{key} is not float"
