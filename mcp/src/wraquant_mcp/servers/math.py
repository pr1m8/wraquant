"""Advanced mathematics MCP tools.

Tools: correlation_network, levy_simulate, optimal_stopping.
"""

from __future__ import annotations

from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_math_tools(mcp, ctx: AnalysisContext) -> None:
    """Register advanced math tools on the MCP server."""

    @mcp.tool()
    def correlation_network(
        dataset: str,
        threshold: float = 0.3,
    ) -> dict[str, Any]:
        """Build and analyze a correlation network from asset returns.

        Constructs an adjacency matrix from pairwise correlations,
        computes centrality measures, and identifies clusters.

        Parameters:
            dataset: Dataset with multi-asset returns (one column per asset).
            threshold: Minimum absolute correlation for an edge.
        """
        import numpy as np

        from wraquant.math.network import centrality_measures, correlation_network as _cn

        df = ctx.get_dataset(dataset)
        returns = df.select_dtypes(include=[np.number]).dropna()

        net = _cn(returns, threshold=threshold)

        centrality = centrality_measures(net["adjacency"], net["asset_names"])

        return _sanitize_for_json({
            "tool": "correlation_network",
            "dataset": dataset,
            "n_assets": len(net["asset_names"]),
            "asset_names": net["asset_names"],
            "n_edges": int(np.sum(net["adjacency"]) / 2),
            "threshold": threshold,
            "centrality": centrality,
        })

    @mcp.tool()
    def levy_simulate(
        model: str = "variance_gamma",
        n_steps: int = 252,
        sigma: float = 0.01,
        nu: float = 0.5,
        theta: float = -0.001,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Simulate a Levy process path.

        Generates sample paths from heavy-tailed Levy processes that
        capture the jumps and fat tails observed in real asset returns.

        Parameters:
            model: Levy model ('variance_gamma', 'nig', 'levy_stable').
            n_steps: Number of time steps.
            sigma: Volatility parameter.
            nu: Variance rate (VG) or tail heaviness (NIG).
            theta: Drift / skewness parameter.
            seed: Random seed for reproducibility.
        """
        import pandas as pd

        simulators = {}

        if model == "variance_gamma":
            from wraquant.math.levy import variance_gamma_simulate

            path = variance_gamma_simulate(
                sigma=sigma, nu=nu, theta=theta,
                n_steps=n_steps, seed=seed,
            )
        elif model == "nig":
            from wraquant.math.levy import nig_simulate

            path = nig_simulate(
                alpha=1.0 / nu, beta=theta, delta=sigma,
                n_steps=n_steps, seed=seed,
            )
        elif model == "levy_stable":
            from wraquant.math.levy import levy_stable_simulate

            path = levy_stable_simulate(
                alpha=1.5, beta=theta, scale=sigma,
                n_steps=n_steps, seed=seed,
            )
        else:
            return {"error": f"Unknown model '{model}'. Options: variance_gamma, nig, levy_stable"}

        path_df = pd.DataFrame({"path": path})
        stored = ctx.store_dataset(
            f"levy_{model}", path_df,
            source_op="levy_simulate",
        )

        import numpy as np

        increments = np.diff(path)

        return _sanitize_for_json({
            "tool": "levy_simulate",
            "model": model,
            "n_steps": n_steps,
            "final_value": float(path[-1]),
            "min_value": float(path.min()),
            "max_value": float(path.max()),
            "increment_mean": float(increments.mean()),
            "increment_std": float(increments.std()),
            "increment_skew": float(
                ((increments - increments.mean()) ** 3).mean()
                / increments.std() ** 3
            ) if increments.std() > 0 else 0.0,
            **stored,
        })

    @mcp.tool()
    def optimal_stopping(
        mu: float,
        sigma: float,
        transaction_cost: float = 0.001,
    ) -> dict[str, Any]:
        """Compute optimal entry/exit thresholds for mean-reverting process.

        Uses the analytical solution for an Ornstein-Uhlenbeck process
        to determine optimal trading thresholds balancing expected
        profit against transaction costs.

        Parameters:
            mu: Mean-reversion speed (higher = faster reversion).
            sigma: Process volatility.
            transaction_cost: Round-trip transaction cost per unit.
        """
        from wraquant.math.optimal_stopping import optimal_exit_threshold

        result = optimal_exit_threshold(
            mu=mu,
            sigma=sigma,
            transaction_cost=transaction_cost,
        )

        return _sanitize_for_json({
            "tool": "optimal_stopping",
            "mu": mu,
            "sigma": sigma,
            "transaction_cost": transaction_cost,
            "entry_threshold": result["entry_threshold"],
            "exit_threshold": result["exit_threshold"],
            "expected_profit": result["expected_profit"],
        })
