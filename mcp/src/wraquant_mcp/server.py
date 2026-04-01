"""wraquant MCP server — quant analysis engine for AI agents.

Composes with:
- OpenBB MCP / EODHD MCP → data fetching
- DuckDB MCP → SQL queries on shared state (same .duckdb file)
- Pandas MCP → DataFrame manipulation (shares DuckDB state)
- Jupyter MCP → notebook interaction (notebooks in workspace)
- Alpaca MCP → trade execution

The shared DuckDB file is the bridge between all MCPs.
Jupyter notebooks in the workspace can also connect to the same file.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from wraquant_mcp.context import AnalysisContext


def build_server(name: str = "wraquant") -> Any:
    """Build and configure the wraquant MCP server.

    Returns a FastMCP server instance with all tools registered.
    """
    from fastmcp import FastMCP

    from wraquant_mcp.servers import register_all

    mcp = FastMCP(name)

    # Shared context (DuckDB state + models + journal)
    ctx = AnalysisContext()

    # ------------------------------------------------------------------
    # Tier 1: Discovery tools (always available)
    # ------------------------------------------------------------------

    @mcp.tool()
    def list_modules() -> dict[str, Any]:
        """List all available wraquant analysis modules.

        Each module provides specialized quant finance tools.
        Call list_tools(module) to see available tools in a module.
        """
        return {
            "modules": {
                "risk": "Risk management — VaR, beta, factor models, stress testing (95 functions)",
                "vol": "Volatility modeling — GARCH family, Hawkes, realized vol (28 functions)",
                "regimes": "Regime detection — HMM, Markov-switching, Kalman, scoring (38 functions)",
                "ta": "Technical analysis — 265 indicators across 19 sub-modules",
                "stats": "Statistical analysis — regression, correlation, distributions (79 functions)",
                "ts": "Time series — forecasting, decomposition, stationarity (52 functions)",
                "opt": "Portfolio optimization — MVO, risk parity, Black-Litterman (26 functions)",
                "backtest": "Backtesting — engine, metrics, tearsheets (38 functions)",
                "price": "Derivatives pricing — options, FBSDEs, stochastic models (50 functions)",
                "ml": "Machine learning — features, pipelines, deep learning (44 functions)",
                "causal": "Causal inference — DID, IV, event studies, RDD (19 functions)",
                "bayes": "Bayesian inference — MCMC, model comparison (29 functions)",
                "microstructure": "Market microstructure — liquidity, toxicity (33 functions)",
                "execution": "Execution algorithms — TWAP, VWAP, Almgren-Chriss (21 functions)",
                "forex": "Forex analysis — pairs, carry, sessions (23 functions)",
                "econometrics": "Econometrics — panel, VAR, event studies (34 functions)",
                "math": "Advanced math — Lévy, networks, optimal stopping (55 functions)",
                "viz": "Visualization — Plotly dashboards, charts (47 functions)",
            },
            "total_functions": 1097,
            "hint": "Use list_tools('module_name') to see specific tools",
        }

    @mcp.tool()
    def list_tools(module: str) -> dict[str, Any]:
        """List available tools in a specific module.

        Parameters:
            module: Module name (e.g., 'risk', 'vol', 'ta').
        """
        try:
            import importlib

            mod = importlib.import_module(f"wraquant.{module}")
            all_funcs = getattr(mod, "__all__", [])
            descriptions = {}
            for fname in all_funcs[:50]:  # Cap at 50 to avoid context bloat
                func = getattr(mod, fname, None)
                if func and callable(func):
                    doc = func.__doc__ or ""
                    descriptions[fname] = doc.split("\n")[0].strip()
            return {
                "module": module,
                "tools": descriptions,
                "count": len(all_funcs),
                "hint": f"Call wraquant tool with module='{module}', function='name'",
            }
        except ImportError:
            return {"error": f"Module '{module}' not found"}

    @mcp.tool()
    def workspace_status() -> dict[str, Any]:
        """Show current workspace state — datasets, models, history.

        Shows what data and models are available in the current session.
        Use this to understand what you can work with.
        """
        return ctx.workspace_status()

    @mcp.tool()
    def workspace_history(n: int = 20) -> list[dict[str, Any]]:
        """Show recent operations in the workspace journal.

        Parameters:
            n: Number of recent entries to show.
        """
        return ctx.history(n=n)

    @mcp.tool()
    def add_note(text: str) -> dict[str, str]:
        """Add a research note to the workspace journal.

        Notes persist across sessions and help maintain context.

        Parameters:
            text: The note text.
        """
        return ctx.add_note(text)

    # ------------------------------------------------------------------
    # Tier 2: Common operations (loaded by default)
    # ------------------------------------------------------------------

    @mcp.tool()
    def analyze(
        dataset: str,
        column: str = "close",
        benchmark_dataset: str | None = None,
        benchmark_column: str = "close",
    ) -> dict[str, Any]:
        """Run comprehensive analysis on a dataset.

        Computes: descriptive stats, risk metrics, stationarity test,
        regime detection (if enough data), and GARCH volatility.

        Parameters:
            dataset: Name of dataset in workspace (DuckDB table).
            column: Column to analyze (default 'close').
            benchmark_dataset: Optional benchmark dataset for relative metrics.
            benchmark_column: Benchmark column name.
        """
        import wraquant as wq

        df = ctx.get_dataset(dataset)
        if column not in df.columns:
            return {"error": f"Column '{column}' not in {list(df.columns)}"}

        prices = df[column]
        returns = prices.pct_change().dropna()

        benchmark = None
        if benchmark_dataset:
            bdf = ctx.get_dataset(benchmark_dataset)
            benchmark = bdf[benchmark_column].pct_change().dropna()

        result = wq.analyze(returns, benchmark=benchmark)

        # Store returns as a dataset for follow-up analysis
        ctx.store_dataset(
            f"returns_{dataset}",
            returns.to_frame(name="returns"),
            source_op="analyze",
            parent=dataset,
        )

        from wraquant_mcp.context import _sanitize_for_json

        return _sanitize_for_json(result)

    @mcp.tool()
    def compute_indicator(
        dataset: str,
        indicator: str,
        column: str = "close",
        period: int = 14,
    ) -> dict[str, Any]:
        """Compute a technical indicator and store the result.

        Dispatches to any of wraquant's 265 TA indicators.

        Parameters:
            dataset: Source dataset name in workspace.
            indicator: Indicator name (e.g., 'rsi', 'macd', 'bollinger_bands').
            column: Price column to use (default 'close').
            period: Lookback period (default 14).
        """
        import wraquant.ta as ta

        df = ctx.get_dataset(dataset)
        if column not in df.columns:
            return {"error": f"Column '{column}' not in {list(df.columns)}"}

        func = getattr(ta, indicator, None)
        if func is None:
            return {"error": f"Indicator '{indicator}' not found. Use list_tools('ta')"}

        try:
            result = func(df[column], period=period)
        except TypeError:
            # Some indicators need different params (OHLC etc.)
            try:
                result = func(df[column])
            except Exception as e:
                return {"error": f"Failed to compute {indicator}: {e}"}

        # Handle multi-output (dict of Series)
        if isinstance(result, dict):
            for key, series in result.items():
                if hasattr(series, "values"):
                    df[f"{indicator}_{key}"] = series.values[: len(df)]
        elif hasattr(result, "values"):
            df[indicator] = result.values[: len(df)]

        new_name = f"{dataset}_{indicator}"
        stored = ctx.store_dataset(new_name, df, source_op="compute_indicator", parent=dataset)

        from wraquant_mcp.context import _sanitize_for_json

        summary = {}
        if hasattr(result, "mean"):
            summary = {
                "mean": float(result.mean()),
                "min": float(result.min()),
                "max": float(result.max()),
            }

        return {**stored, "indicator": indicator, "summary": _sanitize_for_json(summary)}

    @mcp.tool()
    def compute_returns(
        dataset: str,
        column: str = "close",
        method: str = "simple",
    ) -> dict[str, Any]:
        """Compute returns from a price series.

        Parameters:
            dataset: Source dataset with prices.
            column: Price column name.
            method: 'simple' (pct_change) or 'log' (log returns).
        """
        import numpy as np

        df = ctx.get_dataset(dataset)
        prices = df[column]

        if method == "log":
            returns = np.log(prices / prices.shift(1)).dropna()
        else:
            returns = prices.pct_change().dropna()

        result_df = returns.to_frame(name="returns")
        new_name = f"returns_{dataset}"
        stored = ctx.store_dataset(new_name, result_df, source_op="compute_returns", parent=dataset)

        return {
            **stored,
            "method": method,
            "mean": float(returns.mean()),
            "std": float(returns.std()),
            "annualized_vol": float(returns.std() * np.sqrt(252)),
        }

    @mcp.tool()
    def fit_garch(
        dataset: str,
        column: str = "returns",
        model: str = "GARCH",
        p: int = 1,
        q: int = 1,
        dist: str = "normal",
    ) -> dict[str, Any]:
        """Fit a GARCH-family volatility model.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
            model: Model type ('GARCH', 'EGARCH', 'GJR').
            p: GARCH lag order.
            q: ARCH lag order.
            dist: Error distribution ('normal', 't', 'skewt').
        """
        from wraquant.vol.models import garch_fit, egarch_fit, gjr_garch_fit

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()

        fit_fns = {"GARCH": garch_fit, "EGARCH": egarch_fit, "GJR": gjr_garch_fit}
        fit_fn = fit_fns.get(model, garch_fit)

        result = fit_fn(returns.values, p=p, q=q, dist=dist)

        model_name = f"garch_{dataset}_{model.lower()}"
        stored = ctx.store_model(
            model_name,
            result,
            model_type=model,
            source_dataset=dataset,
            metrics={
                "persistence": float(result["persistence"]),
                "half_life": float(result["half_life"]),
                "aic": float(result["aic"]),
                "bic": float(result["bic"]),
            },
        )

        # Store conditional vol as dataset
        if "conditional_volatility" in result:
            vol_df = result["conditional_volatility"].to_frame(name="conditional_vol")
            ctx.store_dataset(
                f"{model_name}_vol", vol_df, source_op="fit_garch", parent=dataset,
            )

        return stored

    @mcp.tool()
    def detect_regimes(
        dataset: str,
        column: str = "returns",
        method: str = "hmm",
        n_regimes: int = 2,
    ) -> dict[str, Any]:
        """Detect market regimes in a return series.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
            method: Detection method ('hmm', 'gmm', 'changepoint').
            n_regimes: Number of regimes to detect.
        """
        from wraquant.regimes.base import detect_regimes as _detect

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna().values

        result = _detect(returns, method=method, n_regimes=n_regimes)

        model_name = f"regime_{dataset}_{method}_{n_regimes}state"
        stored = ctx.store_model(
            model_name,
            result,
            model_type=f"{method}_{n_regimes}state",
            source_dataset=dataset,
        )

        # Store regime states and probabilities as datasets
        import pandas as pd
        import numpy as np

        states_df = pd.DataFrame({"regime": result.states})
        ctx.store_dataset(f"{model_name}_states", states_df, source_op="detect_regimes")

        if result.probabilities is not None:
            probs_df = pd.DataFrame(
                result.probabilities,
                columns=[f"regime_{i}_prob" for i in range(n_regimes)],
            )
            ctx.store_dataset(f"{model_name}_probs", probs_df, source_op="detect_regimes")

        from wraquant_mcp.context import _sanitize_for_json

        return {
            **stored,
            "current_regime": int(result.current_regime),
            "statistics": _sanitize_for_json(result.statistics.to_dict()) if result.statistics is not None else {},
        }

    @mcp.tool()
    def risk_metrics(
        dataset: str,
        column: str = "returns",
        benchmark_dataset: str | None = None,
        benchmark_column: str = "returns",
    ) -> dict[str, Any]:
        """Compute comprehensive risk metrics.

        Parameters:
            dataset: Dataset with returns.
            column: Returns column.
            benchmark_dataset: Optional benchmark for relative metrics.
            benchmark_column: Benchmark returns column.
        """
        from wraquant.risk.metrics import (
            sharpe_ratio, sortino_ratio, max_drawdown,
            information_ratio, hit_ratio,
        )

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()
        prices = (1 + returns).cumprod()

        result = {
            "sharpe": float(sharpe_ratio(returns)),
            "sortino": float(sortino_ratio(returns)),
            "max_drawdown": float(max_drawdown(prices)),
            "hit_ratio": float(hit_ratio(returns)),
            "annualized_return": float(returns.mean() * 252),
            "annualized_vol": float(returns.std() * (252 ** 0.5)),
        }

        if benchmark_dataset:
            bdf = ctx.get_dataset(benchmark_dataset)
            benchmark = bdf[benchmark_column].dropna()
            n = min(len(returns), len(benchmark))
            result["information_ratio"] = float(
                information_ratio(returns.iloc[-n:], benchmark.iloc[-n:])
            )

        return {"tool": "risk_metrics", "dataset": dataset, **result}

    @mcp.tool()
    def dataset_info(dataset: str) -> dict[str, Any]:
        """Get detailed information about a dataset.

        Shows schema, shape, stats, lineage, and sample rows.
        Use this before running analysis to understand the data.

        Parameters:
            dataset: Dataset name in workspace.
        """
        return ctx.dataset_info(dataset)

    @mcp.tool()
    def store_data(
        name: str,
        data: dict[str, list],
    ) -> dict[str, Any]:
        """Store inline data as a workspace dataset.

        Use this when you have small data to store directly
        (e.g., from another MCP or user input).
        For large data, use OpenBB MCP to fetch and store.

        Parameters:
            name: Name for the dataset.
            data: Dict of column_name → list of values.
        """
        import pandas as pd

        df = pd.DataFrame(data)
        return ctx.store_dataset(name, df, source_op="store_data")

    # ------------------------------------------------------------------
    # Tier 3: Module-specific tools (deep capabilities)
    # ------------------------------------------------------------------

    register_all(mcp, ctx)

    return mcp


def run() -> None:
    """CLI entry point for wraquant-mcp."""
    parser = argparse.ArgumentParser(description="wraquant MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport mode (default: stdio for Claude Desktop)",
    )
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    parser.add_argument("--workspace", type=str, default=None, help="Workspace directory")
    args = parser.parse_args()

    mcp = build_server()

    if args.transport == "http":
        mcp.run(transport="streamable-http", host="0.0.0.0", port=args.port)
    else:
        mcp.run()
