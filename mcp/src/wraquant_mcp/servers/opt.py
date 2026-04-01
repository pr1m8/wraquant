"""Portfolio optimization MCP tools.

Tools: optimize_portfolio, efficient_frontier, rebalance_analysis,
black_litterman, hierarchical_risk_parity, min_volatility,
max_sharpe, risk_budgeting.
"""

from __future__ import annotations

from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_opt_tools(mcp, ctx: AnalysisContext) -> None:
    """Register optimization-specific tools on the MCP server."""

    @mcp.tool()
    def optimize_portfolio(
        dataset: str,
        method: str = "max_sharpe",
        risk_free_rate: float = 0.04,
        target_return: float | None = None,
        views: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Optimize portfolio weights.

        Requires a multi-column dataset (each column = asset returns).

        Parameters:
            dataset: Dataset with multi-asset returns.
            method: Optimization method. Options:
                'max_sharpe' (tangency portfolio),
                'min_vol' (minimum variance),
                'risk_parity' (equal risk contribution),
                'mvo' (mean-variance with target return),
                'hrp' (Hierarchical Risk Parity),
                'bl' (Black-Litterman, requires views),
                'equal_weight' (1/N benchmark),
                'inverse_vol' (inverse volatility).
            risk_free_rate: Risk-free rate for Sharpe calculation.
            target_return: Target return for MVO.
            views: Asset views for Black-Litterman.
                e.g. {"AAPL": 0.10, "MSFT": 0.05}
        """
        import numpy as np

        from wraquant.opt.portfolio import (
            black_litterman,
            equal_weight,
            hierarchical_risk_parity,
            inverse_volatility,
            max_sharpe,
            mean_variance,
            min_volatility,
            risk_parity,
        )

        df = ctx.get_dataset(dataset)
        returns = df.select_dtypes(include=[np.number]).dropna()

        methods = {
            "max_sharpe": lambda: max_sharpe(returns, risk_free_rate=risk_free_rate),
            "min_vol": lambda: min_volatility(returns),
            "risk_parity": lambda: risk_parity(returns),
            "mvo": lambda: mean_variance(
                returns, target_return=target_return or 0.10,
                risk_free_rate=risk_free_rate,
            ),
            "hrp": lambda: hierarchical_risk_parity(returns),
            "bl": lambda: black_litterman(returns, views=views or {}),
            "equal_weight": lambda: equal_weight(returns),
            "inverse_vol": lambda: inverse_volatility(returns),
        }

        func = methods.get(method)
        if func is None:
            return {"error": f"Unknown method '{method}'. Options: {list(methods)}"}

        result = func()

        model_name = f"opt_{dataset}_{method}"
        stored = ctx.store_model(
            model_name, result,
            model_type=f"portfolio_{method}",
            source_dataset=dataset,
        )

        # Extract weights
        weights = {}
        if hasattr(result, "weights"):
            w = result.weights
            if hasattr(w, "items"):
                weights = {str(k): float(v) for k, v in w.items()}
            elif hasattr(w, "tolist"):
                weights = dict(zip(returns.columns, w.tolist()))
            else:
                weights = dict(zip(returns.columns, list(w)))

        return _sanitize_for_json({
            **stored,
            "method": method,
            "weights": weights,
            "assets": list(returns.columns),
        })

    @mcp.tool()
    def efficient_frontier(
        dataset: str,
        n_points: int = 20,
        risk_free_rate: float = 0.04,
    ) -> dict[str, Any]:
        """Compute the efficient frontier.

        Returns risk-return pairs for portfolios along the frontier,
        plus the tangency (max Sharpe) portfolio.

        Parameters:
            dataset: Dataset with multi-asset returns.
            n_points: Number of points on the frontier.
            risk_free_rate: Risk-free rate.
        """
        import numpy as np

        from wraquant.opt.portfolio import max_sharpe, mean_variance, min_volatility

        df = ctx.get_dataset(dataset)
        returns = df.select_dtypes(include=[np.number]).dropna()

        # Get min-vol and max-return bounds
        min_vol_result = min_volatility(returns)
        max_sharpe_result = max_sharpe(returns, risk_free_rate=risk_free_rate)

        # Compute frontier points
        min_ret = float(returns.mean().min()) * 252
        max_ret = float(returns.mean().max()) * 252
        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier = []
        for target in target_returns:
            try:
                result = mean_variance(
                    returns, target_return=target,
                    risk_free_rate=risk_free_rate,
                )
                if hasattr(result, "volatility") and hasattr(result, "expected_return"):
                    frontier.append({
                        "return": float(result.expected_return),
                        "risk": float(result.volatility),
                    })
            except Exception:
                continue

        import pandas as pd

        if frontier:
            ef_df = pd.DataFrame(frontier)
            stored = ctx.store_dataset(
                f"frontier_{dataset}", ef_df,
                source_op="efficient_frontier", parent=dataset,
            )
        else:
            stored = {}

        return _sanitize_for_json({
            "tool": "efficient_frontier",
            "dataset": dataset,
            "n_points": len(frontier),
            "frontier": frontier,
            "tangency": {
                "weights": dict(zip(
                    returns.columns,
                    max_sharpe_result.weights.tolist()
                    if hasattr(max_sharpe_result.weights, "tolist")
                    else list(max_sharpe_result.weights),
                )) if hasattr(max_sharpe_result, "weights") else {},
            },
            **stored,
        })

    @mcp.tool()
    def rebalance_analysis(
        dataset: str,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        portfolio_value: float = 1_000_000.0,
    ) -> dict[str, Any]:
        """Analyze the trades needed to rebalance a portfolio.

        Parameters:
            dataset: Dataset with multi-asset returns (for volatility context).
            current_weights: Current portfolio weights by asset name.
            target_weights: Target portfolio weights by asset name.
            portfolio_value: Total portfolio value for trade sizing.
        """
        import numpy as np

        df = ctx.get_dataset(dataset)

        trades = {}
        total_turnover = 0.0

        all_assets = set(list(current_weights.keys()) + list(target_weights.keys()))

        for asset in all_assets:
            current = current_weights.get(asset, 0.0)
            target = target_weights.get(asset, 0.0)
            delta = target - current
            trade_value = delta * portfolio_value
            trades[asset] = {
                "current_weight": current,
                "target_weight": target,
                "delta_weight": delta,
                "trade_value": trade_value,
                "direction": "buy" if delta > 0 else "sell" if delta < 0 else "hold",
            }
            total_turnover += abs(delta)

        return _sanitize_for_json({
            "tool": "rebalance_analysis",
            "dataset": dataset,
            "portfolio_value": portfolio_value,
            "total_turnover": total_turnover / 2,  # one-way
            "n_trades": sum(1 for t in trades.values() if t["direction"] != "hold"),
            "trades": trades,
        })

    @mcp.tool()
    def black_litterman(
        dataset: str,
        market_weights_json: str = "[]",
        views_json: str = "{}",
    ) -> dict[str, Any]:
        """Run Black-Litterman portfolio optimization with investor views.

        Combines market equilibrium returns with subjective views
        to produce posterior expected returns and optimal weights.

        Parameters:
            dataset: Dataset with multi-asset returns.
            market_weights_json: JSON list of market-cap weights.
                If empty, uses equal weights.
            views_json: JSON dict of asset views (expected returns).
                e.g. '{"AAPL": 0.10, "MSFT": 0.05}'
        """
        import json

        import numpy as np

        from wraquant.opt.portfolio import black_litterman as _bl

        df = ctx.get_dataset(dataset)
        returns = df.select_dtypes(include=[np.number]).dropna()

        views = json.loads(views_json) if views_json and views_json != "{}" else {}

        result = _bl(returns, views=views)

        model_name = f"bl_{dataset}"
        stored = ctx.store_model(
            model_name, result,
            model_type="black_litterman",
            source_dataset=dataset,
        )

        weights = {}
        if hasattr(result, "weights"):
            w = result.weights
            if hasattr(w, "items"):
                weights = {str(k): float(v) for k, v in w.items()}
            elif hasattr(w, "tolist"):
                weights = dict(zip(returns.columns, w.tolist()))
            else:
                weights = dict(zip(returns.columns, list(w)))

        return _sanitize_for_json({
            **stored,
            "method": "black_litterman",
            "views": views,
            "weights": weights,
            "assets": list(returns.columns),
        })

    @mcp.tool()
    def hierarchical_risk_parity(
        dataset: str,
    ) -> dict[str, Any]:
        """Hierarchical Risk Parity (HRP) portfolio optimization.

        Uses hierarchical clustering to build a diversified portfolio
        without matrix inversion. More stable than MVO for
        ill-conditioned covariance matrices.

        Parameters:
            dataset: Dataset with multi-asset returns.
        """
        import numpy as np

        from wraquant.opt.portfolio import hierarchical_risk_parity as _hrp

        df = ctx.get_dataset(dataset)
        returns = df.select_dtypes(include=[np.number]).dropna()

        result = _hrp(returns)

        model_name = f"hrp_{dataset}"
        stored = ctx.store_model(
            model_name, result,
            model_type="hrp",
            source_dataset=dataset,
        )

        weights = {}
        if hasattr(result, "weights"):
            w = result.weights
            if hasattr(w, "items"):
                weights = {str(k): float(v) for k, v in w.items()}
            elif hasattr(w, "tolist"):
                weights = dict(zip(returns.columns, w.tolist()))
            else:
                weights = dict(zip(returns.columns, list(w)))

        return _sanitize_for_json({
            **stored,
            "method": "hrp",
            "weights": weights,
            "assets": list(returns.columns),
        })

    @mcp.tool()
    def min_volatility(
        dataset: str,
    ) -> dict[str, Any]:
        """Minimum variance portfolio optimization.

        Finds the portfolio with the lowest possible volatility.
        The only efficient portfolio that doesn't require expected
        return estimates.

        Parameters:
            dataset: Dataset with multi-asset returns.
        """
        import numpy as np

        from wraquant.opt.portfolio import min_volatility as _min_vol

        df = ctx.get_dataset(dataset)
        returns = df.select_dtypes(include=[np.number]).dropna()

        result = _min_vol(returns)

        model_name = f"minvol_{dataset}"
        stored = ctx.store_model(
            model_name, result,
            model_type="min_volatility",
            source_dataset=dataset,
        )

        weights = {}
        if hasattr(result, "weights"):
            w = result.weights
            if hasattr(w, "items"):
                weights = {str(k): float(v) for k, v in w.items()}
            elif hasattr(w, "tolist"):
                weights = dict(zip(returns.columns, w.tolist()))
            else:
                weights = dict(zip(returns.columns, list(w)))

        return _sanitize_for_json({
            **stored,
            "method": "min_volatility",
            "weights": weights,
            "assets": list(returns.columns),
            "portfolio_volatility": float(result.volatility)
            if hasattr(result, "volatility") else None,
        })

    @mcp.tool()
    def max_sharpe(
        dataset: str,
        risk_free_rate: float = 0.04,
    ) -> dict[str, Any]:
        """Maximum Sharpe ratio (tangency) portfolio optimization.

        Finds the portfolio on the efficient frontier with the
        highest risk-adjusted return.

        Parameters:
            dataset: Dataset with multi-asset returns.
            risk_free_rate: Risk-free rate for Sharpe calculation.
        """
        import numpy as np

        from wraquant.opt.portfolio import max_sharpe as _max_sharpe

        df = ctx.get_dataset(dataset)
        returns = df.select_dtypes(include=[np.number]).dropna()

        result = _max_sharpe(returns, risk_free_rate=risk_free_rate)

        model_name = f"maxsharpe_{dataset}"
        stored = ctx.store_model(
            model_name, result,
            model_type="max_sharpe",
            source_dataset=dataset,
        )

        weights = {}
        if hasattr(result, "weights"):
            w = result.weights
            if hasattr(w, "items"):
                weights = {str(k): float(v) for k, v in w.items()}
            elif hasattr(w, "tolist"):
                weights = dict(zip(returns.columns, w.tolist()))
            else:
                weights = dict(zip(returns.columns, list(w)))

        return _sanitize_for_json({
            **stored,
            "method": "max_sharpe",
            "risk_free_rate": risk_free_rate,
            "weights": weights,
            "assets": list(returns.columns),
            "sharpe_ratio": float(result.sharpe_ratio)
            if hasattr(result, "sharpe_ratio") else None,
            "expected_return": float(result.expected_return)
            if hasattr(result, "expected_return") else None,
            "volatility": float(result.volatility)
            if hasattr(result, "volatility") else None,
        })

    @mcp.tool()
    def risk_budgeting(
        dataset: str,
        target_risk_json: str = "[]",
    ) -> dict[str, Any]:
        """Risk budgeting portfolio: target specific risk contributions per asset.

        Finds weights such that each asset contributes a target
        fraction of total portfolio risk. Generalizes risk parity
        (where all targets are equal).

        Parameters:
            dataset: Dataset with multi-asset returns.
            target_risk_json: JSON list of target risk contributions
                (must sum to 1). If empty, defaults to equal risk (risk parity).
        """
        import json

        import numpy as np

        from wraquant.risk.portfolio_analytics import risk_budgeting as _rb

        df = ctx.get_dataset(dataset)
        returns = df.select_dtypes(include=[np.number]).dropna()

        target = json.loads(target_risk_json) \
            if target_risk_json and target_risk_json != "[]" else []

        cov = returns.cov().values

        if target:
            target_arr = np.array(target)
        else:
            n = returns.shape[1]
            target_arr = np.ones(n) / n

        result = _rb(cov, target_risk=target_arr)

        weights = {}
        if isinstance(result, dict) and "weights" in result:
            w = result["weights"]
            if hasattr(w, "tolist"):
                weights = dict(zip(returns.columns, w.tolist()))
            else:
                weights = dict(zip(returns.columns, list(w)))

        return _sanitize_for_json({
            "tool": "risk_budgeting",
            "dataset": dataset,
            "target_risk": target_arr.tolist(),
            "weights": weights,
            "assets": list(returns.columns),
            "result": {k: v for k, v in result.items()
                       if k != "weights"}
            if isinstance(result, dict) else str(result),
        })
