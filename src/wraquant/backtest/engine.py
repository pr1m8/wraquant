"""Vectorized backtesting engine."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from wraquant.backtest.metrics import performance_summary
from wraquant.backtest.strategy import Strategy
from wraquant.core.results import BacktestResult


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
            returns=portfolio_returns,
            equity_curve=portfolio_value,
            metrics=metrics,
            trades=trades,
            positions=positions,
        )


class VectorizedBacktest:
    """Fast vectorized backtesting engine for signal-based strategies.

    Unlike the event-driven ``Backtest`` class (which requires a
    ``Strategy`` object), ``VectorizedBacktest`` operates directly on a
    pre-computed signal / weight matrix.  This is ideal for research
    workflows where signals are generated upstream (e.g., from a
    machine learning model or factor pipeline) and you need fast,
    repeatable backtest evaluation.

    The engine computes portfolio returns accounting for:

    - **Transaction costs** (fixed commission per unit of turnover).
    - **Slippage** (additional cost proportional to turnover).
    - **Rebalancing frequency** (skip rebalancing on off-days to
      reduce turnover).

    Parameters:
        initial_capital: Starting portfolio value in currency units.
        commission: One-way commission as a fraction of turnover
            (e.g., 0.001 = 10 bps).
        slippage: Slippage as a fraction of turnover.
        rebalance_frequency: Rebalance every N periods.  ``1`` means
            every period (daily for daily data), ``5`` means weekly, etc.

    Example:
        >>> import pandas as pd, numpy as np
        >>> rng = np.random.default_rng(42)
        >>> prices = pd.DataFrame(
        ...     100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, (100, 2)), axis=0)),
        ...     columns=["A", "B"],
        ...     index=pd.bdate_range("2023-01-01", periods=100),
        ... )
        >>> signals = pd.DataFrame(
        ...     0.5, index=prices.index, columns=prices.columns
        ... )
        >>> bt = VectorizedBacktest(commission=0.001, slippage=0.0005)
        >>> result = bt.run(prices, signals)
        >>> isinstance(result, BacktestResult)
        True

    See Also:
        Backtest: Strategy-object-based backtesting engine.
        walk_forward_backtest: Walk-forward optimisation + backtest.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission: float = 0.0,
        slippage: float = 0.0,
        rebalance_frequency: int = 1,
    ) -> None:
        if rebalance_frequency < 1:
            raise ValueError("rebalance_frequency must be >= 1")
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.rebalance_frequency = rebalance_frequency

    def run(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> BacktestResult:
        """Execute the vectorized backtest.

        Parameters:
            prices: Asset price DataFrame (rows = dates, columns = assets).
            signals: Weight / signal DataFrame with the same shape as
                ``prices``.  Values represent desired portfolio weights
                (e.g., 0.5 = 50 % allocation to that asset).  Signals
                are shifted by one period internally to avoid look-ahead
                bias.

        Returns:
            BacktestResult with equity curve, returns, positions, and
            metrics. Additional fields (turnover, costs, net_returns,
            gross_returns) are available via dict-like access on the
            metrics dict.
        """
        asset_returns = prices.pct_change().fillna(0)

        # Apply rebalance frequency: hold positions constant between
        # rebalance dates
        if self.rebalance_frequency > 1:
            rebal_mask = np.arange(len(signals)) % self.rebalance_frequency == 0
            effective_signals = signals.copy()
            for i in range(len(signals)):
                if not rebal_mask[i] and i > 0:
                    effective_signals.iloc[i] = effective_signals.iloc[i - 1]
        else:
            effective_signals = signals

        # Shift to avoid look-ahead bias
        positions = effective_signals.shift(1).fillna(0)

        # Gross portfolio returns
        gross_returns = (positions * asset_returns).sum(axis=1)

        # Turnover and costs
        turnover = positions.diff().abs().sum(axis=1)
        costs = turnover * (self.commission + self.slippage)

        # Net returns
        net_returns = gross_returns - costs

        # Equity curve
        equity_curve = self.initial_capital * (1 + net_returns).cumprod()
        equity_curve.name = "equity_curve"
        net_returns.name = "net_returns"

        metrics = performance_summary(net_returns)

        return BacktestResult(
            returns=net_returns,
            equity_curve=equity_curve,
            metrics=metrics,
            positions=positions,
            signals=signals,
        )


def walk_forward_backtest(
    prices: pd.DataFrame,
    strategy_factory: Callable[..., Strategy],
    param_grid: list[dict[str, Any]],
    train_size: int = 252,
    test_size: int = 63,
    step_size: int | None = None,
    metric: str = "sharpe_ratio",
    initial_capital: float = 100_000.0,
    commission: float = 0.0,
    slippage: float = 0.0,
) -> dict[str, Any]:
    """Walk-forward optimisation and backtesting.

    Walk-forward analysis is the gold standard for evaluating
    parameter-dependent strategies.  It splits the data into rolling
    train/test windows, optimises strategy parameters on the training
    set, and evaluates on the out-of-sample test set.  The result is a
    true out-of-sample equity curve that avoids look-ahead bias and
    parameter overfitting.

    Algorithm:
        1. For each window starting at ``t``:
           a. Train set: ``prices[t : t + train_size]``
           b. Test set: ``prices[t + train_size : t + train_size + test_size]``
        2. For each parameter combination in ``param_grid``, backtest
           on the train set and record the optimisation metric.
        3. Select the best parameters and backtest on the test set.
        4. Slide forward by ``step_size`` and repeat.
        5. Concatenate all out-of-sample test returns.

    Parameters:
        prices: Asset price DataFrame.
        strategy_factory: Callable that accepts keyword arguments
            (from ``param_grid``) and returns a ``Strategy`` instance.
        param_grid: List of parameter dictionaries to search over.
        train_size: Number of periods in each training window.
        test_size: Number of periods in each test window.
        step_size: Number of periods to slide the window forward.
            Defaults to ``test_size`` (non-overlapping test windows).
        metric: Performance metric to optimise (key from
            ``performance_summary`` output, e.g., ``"sharpe_ratio"``).
        initial_capital: Starting capital for each window backtest.
        commission: Commission fraction.
        slippage: Slippage fraction.

    Returns:
        Dictionary with:

        - ``oos_returns``: Concatenated out-of-sample return series.
        - ``is_returns``: Concatenated in-sample return series.
        - ``oos_equity_curve``: Equity curve from OOS returns.
        - ``params_per_window``: List of best params per window.
        - ``is_metrics_per_window``: In-sample metrics per window.
        - ``oos_metrics``: Overall OOS performance summary.
        - ``stability_ratio``: Fraction of windows where OOS Sharpe > 0.

    Example:
        >>> import pandas as pd, numpy as np
        >>> from wraquant.backtest.strategy import MomentumStrategy
        >>> rng = np.random.default_rng(42)
        >>> prices = pd.DataFrame(
        ...     100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, (600, 2)), axis=0)),
        ...     columns=["A", "B"],
        ...     index=pd.bdate_range("2020-01-01", periods=600),
        ... )
        >>> grid = [{"lookback": 10}, {"lookback": 20}, {"lookback": 40}]
        >>> result = walk_forward_backtest(
        ...     prices,
        ...     strategy_factory=lambda **kw: MomentumStrategy(**kw),
        ...     param_grid=grid,
        ...     train_size=200,
        ...     test_size=50,
        ... )
        >>> "oos_returns" in result
        True

    See Also:
        Backtest: Single-pass backtesting engine.
        VectorizedBacktest: Signal-matrix-based backtesting.
    """
    if step_size is None:
        step_size = test_size

    n = len(prices)
    if n < train_size + test_size:
        raise ValueError(
            f"Not enough data: {n} periods < train_size({train_size}) "
            f"+ test_size({test_size})"
        )

    oos_returns_parts: list[pd.Series] = []
    is_returns_parts: list[pd.Series] = []
    params_per_window: list[dict[str, Any]] = []
    is_metrics_per_window: list[dict[str, Any]] = []
    oos_sharpes: list[float] = []

    t = 0
    while t + train_size + test_size <= n:
        train_prices = prices.iloc[t : t + train_size]
        test_prices = prices.iloc[t + train_size : t + train_size + test_size]

        # --- Optimise on training set ---
        best_metric_val = -np.inf
        best_params: dict[str, Any] = param_grid[0] if param_grid else {}
        best_is_returns: pd.Series | None = None
        best_is_metrics: dict[str, Any] = {}

        for params in param_grid:
            strategy = strategy_factory(**params)
            bt = Backtest(
                strategy,
                initial_capital=initial_capital,
                commission=commission,
                slippage=slippage,
            )
            result = bt.run(train_prices)
            val = result.metrics.get(metric, 0.0)
            if val > best_metric_val:
                best_metric_val = val
                best_params = params
                best_is_returns = result.returns
                best_is_metrics = result.metrics

        # --- Evaluate on test set ---
        strategy = strategy_factory(**best_params)
        bt = Backtest(
            strategy,
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
        )
        oos_result = bt.run(test_prices)

        oos_returns_parts.append(oos_result.returns)
        if best_is_returns is not None:
            is_returns_parts.append(best_is_returns)
        params_per_window.append(best_params)
        is_metrics_per_window.append(best_is_metrics)
        oos_sharpes.append(oos_result.metrics.get("sharpe_ratio", 0.0))

        t += step_size

    # --- Concatenate OOS returns ---
    oos_returns = pd.concat(oos_returns_parts) if oos_returns_parts else pd.Series(dtype=float)
    is_returns = pd.concat(is_returns_parts) if is_returns_parts else pd.Series(dtype=float)

    # Equity curve from OOS returns
    oos_equity = initial_capital * (1 + oos_returns).cumprod() if len(oos_returns) > 0 else pd.Series(dtype=float)

    # Stability: fraction of windows where OOS Sharpe > 0
    n_windows = len(oos_sharpes)
    stability_ratio = (
        sum(1 for s in oos_sharpes if s > 0) / n_windows if n_windows > 0 else 0.0
    )

    oos_metrics = performance_summary(oos_returns) if len(oos_returns) > 0 else {}

    return {
        "oos_returns": oos_returns,
        "is_returns": is_returns,
        "oos_equity_curve": oos_equity,
        "params_per_window": params_per_window,
        "is_metrics_per_window": is_metrics_per_window,
        "oos_metrics": oos_metrics,
        "stability_ratio": stability_ratio,
    }
