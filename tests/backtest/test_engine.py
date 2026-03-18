"""Tests for backtesting engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

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
