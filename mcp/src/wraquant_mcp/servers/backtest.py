"""Backtesting MCP tools.

Tools: run_backtest, backtest_metrics, comprehensive_tearsheet,
walk_forward, strategy_comparison.
"""

from __future__ import annotations

from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_backtest_tools(mcp, ctx: AnalysisContext) -> None:
    """Register backtesting-specific tools on the MCP server."""

    @mcp.tool()
    def run_backtest(
        dataset: str,
        signal_column: str = "signal",
        price_column: str = "close",
        initial_capital: float = 100_000.0,
    ) -> dict[str, Any]:
        """Run a vectorized backtest from a signal column.

        The dataset must contain a price column and a signal column
        (1 = long, -1 = short, 0 = flat).

        Parameters:
            dataset: Dataset with price and signal columns.
            signal_column: Column with trading signals.
            price_column: Column with prices.
            initial_capital: Starting capital.
        """
        import numpy as np
        import pandas as pd

        df = ctx.get_dataset(dataset)

        if signal_column not in df.columns:
            return {"error": f"Signal column '{signal_column}' not found"}
        if price_column not in df.columns:
            return {"error": f"Price column '{price_column}' not found"}

        prices = df[price_column]
        signals = df[signal_column]
        returns = prices.pct_change().fillna(0)
        strategy_returns = signals.shift(1).fillna(0) * returns
        equity = initial_capital * (1 + strategy_returns).cumprod()

        # Store equity curve
        eq_df = pd.DataFrame({
            "equity": equity,
            "returns": strategy_returns,
            "signal": signals,
        })
        stored = ctx.store_dataset(
            f"backtest_{dataset}", eq_df,
            source_op="run_backtest", parent=dataset,
        )

        total_return = float(equity.iloc[-1] / initial_capital - 1)
        ann_return = float(strategy_returns.mean() * 252)
        ann_vol = float(strategy_returns.std() * np.sqrt(252))
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

        cum_max = equity.cummax()
        drawdown = (equity - cum_max) / cum_max
        max_dd = float(drawdown.min())

        return _sanitize_for_json({
            "tool": "run_backtest",
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_vol": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "n_trades": int((signals.diff().abs() > 0).sum()),
            **stored,
        })

    @mcp.tool()
    def backtest_metrics(
        dataset: str,
        column: str = "returns",
    ) -> dict[str, Any]:
        """Compute comprehensive backtest performance metrics.

        Requires a dataset with strategy returns. Returns 15+
        performance metrics including Sharpe, Sortino, Omega,
        Kelly fraction, and more.

        Parameters:
            dataset: Dataset containing strategy returns.
            column: Returns column name.
        """
        from wraquant.backtest.metrics import performance_summary

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()

        result = performance_summary(returns)

        return _sanitize_for_json({
            "tool": "backtest_metrics",
            "dataset": dataset,
            **result,
        })

    @mcp.tool()
    def comprehensive_tearsheet(
        dataset: str,
        returns_column: str = "returns",
        benchmark_dataset: str | None = None,
        benchmark_column: str = "returns",
    ) -> dict[str, Any]:
        """Generate a comprehensive performance tearsheet.

        Returns monthly returns table, drawdown analysis, rolling
        metrics, and trade statistics.

        Parameters:
            dataset: Dataset with strategy returns.
            returns_column: Strategy returns column.
            benchmark_dataset: Optional benchmark dataset.
            benchmark_column: Benchmark returns column.
        """
        from wraquant.backtest.tearsheet import (
            comprehensive_tearsheet as _tearsheet,
            drawdown_table,
            monthly_returns_table,
            rolling_metrics_table,
        )

        df = ctx.get_dataset(dataset)
        returns = df[returns_column].dropna()

        benchmark = None
        if benchmark_dataset:
            bdf = ctx.get_dataset(benchmark_dataset)
            benchmark = bdf[benchmark_column].dropna()

        monthly = monthly_returns_table(returns)
        drawdowns = drawdown_table(returns)
        rolling = rolling_metrics_table(returns)

        result = {
            "tool": "comprehensive_tearsheet",
            "dataset": dataset,
            "monthly_returns": monthly,
            "drawdowns": drawdowns,
            "rolling_metrics": rolling,
        }

        if benchmark is not None:
            n = min(len(returns), len(benchmark))
            tearsheet = _tearsheet(
                returns.iloc[-n:], benchmark=benchmark.iloc[-n:],
            )
            result["tearsheet"] = tearsheet

        return _sanitize_for_json(result)

    @mcp.tool()
    def walk_forward(
        dataset: str,
        signal_column: str = "signal",
        price_column: str = "close",
        train_size: int = 252,
        test_size: int = 63,
    ) -> dict[str, Any]:
        """Run a walk-forward backtest.

        Splits data into rolling train/test windows and evaluates
        out-of-sample performance at each step.

        Parameters:
            dataset: Dataset with price and signal columns.
            signal_column: Signal column.
            price_column: Price column.
            train_size: Training window in periods.
            test_size: Test window in periods.
        """
        from wraquant.backtest.engine import walk_forward_backtest

        df = ctx.get_dataset(dataset)

        result = walk_forward_backtest(
            df,
            signal_column=signal_column,
            price_column=price_column,
            train_size=train_size,
            test_size=test_size,
        )

        model_name = f"wf_{dataset}"
        stored = ctx.store_model(
            model_name, result,
            model_type="walk_forward",
            source_dataset=dataset,
        )

        return _sanitize_for_json({
            "tool": "walk_forward",
            "train_size": train_size,
            "test_size": test_size,
            **stored,
            "result": result if isinstance(result, dict) else str(result),
        })

    @mcp.tool()
    def strategy_comparison(
        datasets: list[str],
        returns_column: str = "returns",
    ) -> dict[str, Any]:
        """Compare performance across multiple strategy backtests.

        Parameters:
            datasets: List of dataset names, each containing
                strategy returns.
            returns_column: Returns column name (same for all).
        """
        from wraquant.backtest.tearsheet import strategy_comparison as _compare

        import pandas as pd

        all_returns = {}
        for name in datasets:
            df = ctx.get_dataset(name)
            if returns_column in df.columns:
                all_returns[name] = df[returns_column].dropna()

        if len(all_returns) < 2:
            return {"error": "Need at least 2 datasets with returns for comparison"}

        returns_df = pd.DataFrame(all_returns)
        result = _compare(returns_df)

        return _sanitize_for_json({
            "tool": "strategy_comparison",
            "strategies": list(all_returns.keys()),
            "comparison": result,
        })
