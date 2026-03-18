"""Vectorized backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from wraquant.backtest.metrics import performance_summary
from wraquant.backtest.strategy import Strategy


@dataclass
class BacktestResult:
    """Results from a backtest run.

    Parameters:
        portfolio_value: Time series of portfolio value.
        returns: Time series of portfolio returns.
        positions: Position weights over time.
        trades: Number of trades executed.
        metrics: Performance metrics dict.
    """

    portfolio_value: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: int = 0
    metrics: dict = field(default_factory=dict)


class Backtest:
    """Vectorized backtesting engine.

    Parameters:
        strategy: Strategy instance.
        initial_capital: Starting portfolio value.
        commission: Commission per trade as fraction (e.g., 0.001 = 10bps).
        slippage: Slippage per trade as fraction.

    Example:
        >>> from wraquant.backtest import Backtest
        >>> from wraquant.backtest.strategy import BuyAndHold
        >>> bt = Backtest(BuyAndHold(), initial_capital=100_000)
        >>> result = bt.run(prices_df)  # doctest: +SKIP
    """

    def __init__(
        self,
        strategy: Strategy,
        initial_capital: float = 100_000.0,
        commission: float = 0.0,
        slippage: float = 0.0,
    ) -> None:
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    def run(self, prices: pd.DataFrame) -> BacktestResult:
        """Run the backtest on historical price data.

        Parameters:
            prices: DataFrame of asset prices (columns = assets).

        Returns:
            BacktestResult with portfolio value, returns, positions, metrics.
        """
        signals = self.strategy.generate_signals(prices)

        # Calculate returns
        asset_returns = prices.pct_change().fillna(0)

        # Shift signals to avoid look-ahead bias
        positions = signals.shift(1).fillna(0)

        # Count trades (signal changes)
        trades = int((positions.diff().abs() > 0.01).sum().sum())

        # Portfolio returns = sum of position-weighted asset returns
        portfolio_returns = (positions * asset_returns).sum(axis=1)

        # Apply transaction costs
        turnover = positions.diff().abs().sum(axis=1)
        costs = turnover * (self.commission + self.slippage)
        portfolio_returns = portfolio_returns - costs

        # Portfolio value
        portfolio_value = self.initial_capital * (1 + portfolio_returns).cumprod()
        portfolio_value.name = "portfolio_value"
        portfolio_returns.name = "portfolio_returns"

        # Calculate metrics
        metrics = performance_summary(portfolio_returns)

        return BacktestResult(
            portfolio_value=portfolio_value,
            returns=portfolio_returns,
            positions=positions,
            trades=trades,
            metrics=metrics,
        )
