"""Backtesting MCP tools.

Tools: run_backtest, backtest_metrics, comprehensive_tearsheet,
walk_forward, strategy_comparison, omega_ratio, kelly_fraction,
regime_backtest, vectorized_backtest, drawdown_analysis.
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

    @mcp.tool()
    def omega_ratio(
        dataset: str,
        column: str = "returns",
        threshold: float = 0.0,
    ) -> dict[str, Any]:
        """Compute the Omega ratio: probability-weighted gain/loss ratio.

        Uses the entire return distribution (all moments), making
        it more appropriate than Sharpe for non-normal returns.

        Parameters:
            dataset: Dataset containing strategy returns.
            column: Returns column name.
            threshold: Return threshold (gains above vs losses below).
        """
        from wraquant.backtest.metrics import omega_ratio as _omega

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()

        result = _omega(returns, threshold=threshold)

        return _sanitize_for_json({
            "tool": "omega_ratio",
            "dataset": dataset,
            "column": column,
            "threshold": threshold,
            "omega_ratio": float(result),
            "observations": len(returns),
        })

    @mcp.tool()
    def kelly_fraction(
        dataset: str,
        column: str = "returns",
    ) -> dict[str, Any]:
        """Compute the Kelly fraction for optimal bet sizing.

        Determines the fraction of capital to risk per trade
        to maximize geometric growth rate. Full Kelly is aggressive;
        practitioners typically use half-Kelly.

        Parameters:
            dataset: Dataset containing strategy returns.
            column: Returns column name.
        """
        from wraquant.backtest.metrics import kelly_fraction as _kelly

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()

        wins = returns[returns > 0]
        losses = returns[returns < 0]

        if len(wins) == 0 or len(losses) == 0:
            return _sanitize_for_json({
                "tool": "kelly_fraction",
                "dataset": dataset,
                "error": "Need both winning and losing trades",
            })

        win_rate = float(len(wins) / len(returns))
        avg_win = float(wins.mean())
        avg_loss = float(abs(losses.mean()))

        full_kelly = _kelly(win_rate, avg_win, avg_loss)

        return _sanitize_for_json({
            "tool": "kelly_fraction",
            "dataset": dataset,
            "column": column,
            "full_kelly": float(full_kelly),
            "half_kelly": float(full_kelly / 2),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "observations": len(returns),
        })

    @mcp.tool()
    def regime_backtest(
        dataset: str,
        column: str = "returns",
        n_regimes: int = 2,
        bull_weight: float = 1.0,
        bear_weight: float = 0.0,
    ) -> dict[str, Any]:
        """Run a regime-filtered backtest using HMM regime detection.

        Detects market regimes via Hidden Markov Model and applies
        different position weights per regime (e.g., full long in
        bull, flat in bear).

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column.
            n_regimes: Number of regimes for HMM.
            bull_weight: Portfolio weight in the bull regime.
            bear_weight: Portfolio weight in the bear regime.
        """
        import numpy as np
        import pandas as pd

        from wraquant.regimes.hmm import fit_hmm, predict_regime

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()

        model = fit_hmm(returns, n_states=n_regimes)
        regimes = predict_regime(model, returns)

        # Identify bull regime as the one with higher mean return
        regime_means = {}
        for r in range(n_regimes):
            mask = regimes == r
            if mask.sum() > 0:
                regime_means[r] = float(returns.iloc[mask].mean()) \
                    if hasattr(returns, "iloc") else float(returns[mask].mean())

        sorted_regimes = sorted(regime_means.items(), key=lambda x: x[1], reverse=True)
        bull_regime = sorted_regimes[0][0]

        # Apply weights
        weights = np.where(regimes == bull_regime, bull_weight, bear_weight)
        strategy_returns = returns * pd.Series(weights, index=returns.index).shift(1).fillna(0)

        equity = (1 + strategy_returns).cumprod()
        ann_return = float(strategy_returns.mean() * 252)
        ann_vol = float(strategy_returns.std() * np.sqrt(252))
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

        cum_max = equity.cummax()
        max_dd = float(((equity - cum_max) / cum_max).min())

        eq_df = pd.DataFrame({
            "equity": equity,
            "returns": strategy_returns,
            "regime": regimes,
        })
        stored = ctx.store_dataset(
            f"regime_bt_{dataset}", eq_df,
            source_op="regime_backtest", parent=dataset,
        )

        return _sanitize_for_json({
            "tool": "regime_backtest",
            "dataset": dataset,
            "n_regimes": n_regimes,
            "bull_regime": int(bull_regime),
            "regime_means": regime_means,
            "annualized_return": ann_return,
            "annualized_vol": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            **stored,
        })

    @mcp.tool()
    def vectorized_backtest(
        dataset: str,
        signals_dataset: str,
        commission: float = 0.001,
    ) -> dict[str, Any]:
        """Run a fast vectorized backtest from separate price and signal datasets.

        Multiplies signals by returns, accounting for transaction costs
        on trades.

        Parameters:
            dataset: Dataset with price data (must have 'close' column).
            signals_dataset: Dataset with signal column
                (1 = long, -1 = short, 0 = flat).
            commission: Commission per trade (as fraction, e.g., 0.001 = 10bps).
        """
        import numpy as np
        import pandas as pd

        df = ctx.get_dataset(dataset)
        sig_df = ctx.get_dataset(signals_dataset)

        if "close" not in df.columns:
            return {"error": "Price dataset must have a 'close' column"}

        signal_col = None
        for col in ["signal", "signals", "position"]:
            if col in sig_df.columns:
                signal_col = col
                break
        if signal_col is None:
            signal_col = sig_df.columns[0]

        prices = df["close"]
        signals = sig_df[signal_col]

        n = min(len(prices), len(signals))
        prices = prices.iloc[:n]
        signals = signals.iloc[:n]

        returns = prices.pct_change().fillna(0)
        strategy_returns = signals.shift(1).fillna(0) * returns

        # Transaction costs
        trades = signals.diff().abs().fillna(0)
        costs = trades * commission
        strategy_returns = strategy_returns - costs

        equity = (1 + strategy_returns).cumprod()
        total_return = float(equity.iloc[-1] - 1)
        ann_return = float(strategy_returns.mean() * 252)
        ann_vol = float(strategy_returns.std() * np.sqrt(252))
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

        cum_max = equity.cummax()
        max_dd = float(((equity - cum_max) / cum_max).min())
        n_trades = int((trades > 0).sum())

        eq_df = pd.DataFrame({
            "equity": equity,
            "returns": strategy_returns,
            "signal": signals,
        })
        stored = ctx.store_dataset(
            f"vbt_{dataset}", eq_df,
            source_op="vectorized_backtest", parent=dataset,
        )

        return _sanitize_for_json({
            "tool": "vectorized_backtest",
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_vol": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "n_trades": n_trades,
            "total_commission": float(costs.sum()),
            "commission_rate": commission,
            **stored,
        })

    @mcp.tool()
    def drawdown_analysis(
        dataset: str,
        column: str = "returns",
        top_n: int = 5,
    ) -> dict[str, Any]:
        """Detailed analysis of the worst drawdown periods.

        Returns a table of the top N drawdowns with start/end dates,
        depth, duration, and recovery time.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column.
            top_n: Number of worst drawdowns to analyze.
        """
        from wraquant.backtest.tearsheet import drawdown_table

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()

        result = drawdown_table(returns, top_n=top_n)

        import pandas as pd

        if isinstance(result, pd.DataFrame):
            stored = ctx.store_dataset(
                f"drawdowns_{dataset}", result,
                source_op="drawdown_analysis", parent=dataset,
            )
            drawdowns = result.to_dict(orient="records")
        else:
            stored = {}
            drawdowns = result

        # Compute current drawdown
        equity = (1 + returns).cumprod()
        cum_max = equity.cummax()
        current_dd = float(((equity.iloc[-1] - cum_max.iloc[-1]) / cum_max.iloc[-1]))

        return _sanitize_for_json({
            "tool": "drawdown_analysis",
            "dataset": dataset,
            "column": column,
            "top_n": top_n,
            "current_drawdown": current_dd,
            "drawdowns": drawdowns,
            **stored,
        })
