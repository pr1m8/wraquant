"""Tests for backtesting engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.backtest.engine import Backtest, BacktestResult
from wraquant.backtest.metrics import performance_summary
from wraquant.backtest.strategy import BuyAndHold, MomentumStrategy


def _make_prices(n_assets: int = 3, n_periods: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = 100 * np.exp(
        np.cumsum(rng.normal(0.0005, 0.02, size=(n_periods, n_assets)), axis=0)
    )
    cols = [f"asset_{i}" for i in range(n_assets)]
    idx = pd.bdate_range("2020-01-01", periods=n_periods)
    return pd.DataFrame(data, index=idx, columns=cols)


class TestBacktest:
    def test_buy_and_hold(self) -> None:
        prices = _make_prices()
        bt = Backtest(BuyAndHold(), initial_capital=100_000)
        result = bt.run(prices)
        assert isinstance(result, BacktestResult)
        assert len(result.portfolio_value) == len(prices)
        assert result.portfolio_value.iloc[0] == 100_000

    def test_with_commission(self) -> None:
        prices = _make_prices()
        bt_no_cost = Backtest(BuyAndHold())
        bt_with_cost = Backtest(BuyAndHold(), commission=0.01)
        r1 = bt_no_cost.run(prices)
        r2 = bt_with_cost.run(prices)
        # Commission should reduce returns
        assert r2.portfolio_value.iloc[-1] <= r1.portfolio_value.iloc[-1]

    def test_momentum_strategy(self) -> None:
        prices = _make_prices(n_periods=200)
        bt = Backtest(MomentumStrategy(lookback=20))
        result = bt.run(prices)
        assert isinstance(result, BacktestResult)
        assert result.trades > 0

    def test_metrics_populated(self) -> None:
        prices = _make_prices()
        bt = Backtest(BuyAndHold())
        result = bt.run(prices)
        assert "sharpe_ratio" in result.metrics
        assert "max_drawdown" in result.metrics
        assert result.metrics["max_drawdown"] <= 0


class TestPerformanceSummary:
    def test_positive_returns(self) -> None:
        returns = pd.Series(np.full(252, 0.001))
        metrics = performance_summary(returns)
        assert metrics["total_return"] > 0
        assert metrics["annualized_return"] > 0
        assert metrics["sharpe_ratio"] > 0

    def test_zero_returns(self) -> None:
        returns = pd.Series(np.zeros(100))
        metrics = performance_summary(returns)
        assert metrics["total_return"] == 0.0

    def test_win_rate(self) -> None:
        returns = pd.Series([0.01, -0.005, 0.02, 0.01, -0.01])
        metrics = performance_summary(returns)
        assert metrics["win_rate"] == 0.6


# ------------------------------------------------------------------
# VectorizedBacktest
# ------------------------------------------------------------------

from wraquant.backtest.engine import VectorizedBacktest, walk_forward_backtest


class TestVectorizedBacktest:
    def _make_data(
        self, n_assets: int = 2, n_periods: int = 100
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(42)
        data = 100 * np.exp(
            np.cumsum(rng.normal(0.0005, 0.02, size=(n_periods, n_assets)), axis=0)
        )
        cols = [f"asset_{i}" for i in range(n_assets)]
        idx = pd.bdate_range("2023-01-01", periods=n_periods)
        prices = pd.DataFrame(data, index=idx, columns=cols)
        signals = pd.DataFrame(
            1.0 / n_assets, index=idx, columns=cols
        )
        return prices, signals

    def test_returns_correct_keys(self) -> None:
        prices, signals = self._make_data()
        bt = VectorizedBacktest()
        result = bt.run(prices, signals)
        for key in ["returns", "net_returns", "equity_curve", "turnover", "costs", "metrics"]:
            assert key in result, f"Missing key: {key}"

    def test_equity_curve_shape(self) -> None:
        prices, signals = self._make_data()
        bt = VectorizedBacktest()
        result = bt.run(prices, signals)
        assert len(result["equity_curve"]) == len(prices)

    def test_costs_reduce_returns(self) -> None:
        prices, signals = self._make_data()
        bt_free = VectorizedBacktest(commission=0.0, slippage=0.0)
        bt_costly = VectorizedBacktest(commission=0.01, slippage=0.005)
        r_free = bt_free.run(prices, signals)
        r_costly = bt_costly.run(prices, signals)
        # Net returns should be lower with costs
        assert r_costly["equity_curve"].iloc[-1] <= r_free["equity_curve"].iloc[-1]

    def test_rebalance_frequency(self) -> None:
        prices, signals = self._make_data(n_periods=200)
        # Vary signals to ensure turnover differs
        rng = np.random.default_rng(99)
        signals = pd.DataFrame(
            rng.uniform(0.3, 0.7, size=(200, 2)),
            index=prices.index,
            columns=prices.columns,
        )
        bt_daily = VectorizedBacktest(rebalance_frequency=1)
        bt_weekly = VectorizedBacktest(rebalance_frequency=5)
        r_daily = bt_daily.run(prices, signals)
        r_weekly = bt_weekly.run(prices, signals)
        # Weekly should have lower turnover
        assert r_weekly["turnover"].sum() < r_daily["turnover"].sum()

    def test_metrics_populated(self) -> None:
        prices, signals = self._make_data()
        bt = VectorizedBacktest()
        result = bt.run(prices, signals)
        assert "sharpe_ratio" in result["metrics"]
        assert "max_drawdown" in result["metrics"]

    def test_invalid_rebalance_frequency(self) -> None:
        with pytest.raises(ValueError):
            VectorizedBacktest(rebalance_frequency=0)


# ------------------------------------------------------------------
# walk_forward_backtest
# ------------------------------------------------------------------


class TestWalkForwardBacktest:
    def _make_prices(self, n_periods: int = 600) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        data = 100 * np.exp(
            np.cumsum(rng.normal(0.0003, 0.015, size=(n_periods, 2)), axis=0)
        )
        idx = pd.bdate_range("2020-01-01", periods=n_periods)
        return pd.DataFrame(data, index=idx, columns=["A", "B"])

    def test_returns_expected_keys(self) -> None:
        prices = self._make_prices()
        grid = [{"lookback": 10}, {"lookback": 20}]
        result = walk_forward_backtest(
            prices,
            strategy_factory=lambda **kw: MomentumStrategy(**kw),
            param_grid=grid,
            train_size=200,
            test_size=50,
        )
        for key in [
            "oos_returns",
            "is_returns",
            "oos_equity_curve",
            "params_per_window",
            "oos_metrics",
            "stability_ratio",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_oos_returns_non_empty(self) -> None:
        prices = self._make_prices()
        grid = [{"lookback": 10}, {"lookback": 20}]
        result = walk_forward_backtest(
            prices,
            strategy_factory=lambda **kw: MomentumStrategy(**kw),
            param_grid=grid,
            train_size=200,
            test_size=50,
        )
        assert len(result["oos_returns"]) > 0

    def test_params_per_window_populated(self) -> None:
        prices = self._make_prices()
        grid = [{"lookback": 10}, {"lookback": 20}]
        result = walk_forward_backtest(
            prices,
            strategy_factory=lambda **kw: MomentumStrategy(**kw),
            param_grid=grid,
            train_size=200,
            test_size=50,
        )
        assert len(result["params_per_window"]) > 0
        for params in result["params_per_window"]:
            assert "lookback" in params

    def test_stability_ratio_bounded(self) -> None:
        prices = self._make_prices()
        grid = [{"lookback": 10}, {"lookback": 20}]
        result = walk_forward_backtest(
            prices,
            strategy_factory=lambda **kw: MomentumStrategy(**kw),
            param_grid=grid,
            train_size=200,
            test_size=50,
        )
        assert 0.0 <= result["stability_ratio"] <= 1.0

    def test_insufficient_data_raises(self) -> None:
        prices = self._make_prices(n_periods=100)
        grid = [{"lookback": 10}]
        with pytest.raises(ValueError, match="Not enough data"):
            walk_forward_backtest(
                prices,
                strategy_factory=lambda **kw: MomentumStrategy(**kw),
                param_grid=grid,
                train_size=200,
                test_size=50,
            )
