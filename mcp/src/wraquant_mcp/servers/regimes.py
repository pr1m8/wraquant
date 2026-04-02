"""Regime detection MCP tools (deep module-specific).

Tools: regime_statistics, regime_transition, select_n_states,
rolling_regime_probability, fit_gaussian_hmm, fit_ms_autoregression,
gaussian_mixture_regimes, regime_conditional_moments, regime_scoring,
regime_labels, kalman_filter, kalman_regression.

Note: The basic detect_regimes tool lives in server.py (tier-2).
These tools provide deeper regime analysis capabilities.
"""

from __future__ import annotations

from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_regimes_tools(mcp, ctx: AnalysisContext) -> None:
    """Register regime-detection tools on the MCP server."""

    @mcp.tool()
    def regime_statistics(
        dataset: str,
        column: str = "returns",
        n_regimes: int = 2,
    ) -> dict[str, Any]:
        """Compute per-regime descriptive statistics.

        Fits an HMM then computes mean, volatility, Sharpe, Sortino,
        drawdown, VaR/CVaR, skewness, and kurtosis for each regime.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
            n_regimes: Number of regimes to fit.
        """
        from wraquant.regimes.hmm import fit_hmm, regime_statistics as _regime_stats

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()

        model = fit_hmm(returns, n_states=n_regimes)
        states = model.predict(returns.values.reshape(-1, 1))

        stats_df = _regime_stats(returns, states)

        stored = ctx.store_dataset(
            f"regime_stats_{dataset}", stats_df,
            source_op="regime_statistics", parent=dataset,
        )

        return _sanitize_for_json({
            "tool": "regime_statistics",
            "dataset": dataset,
            "n_regimes": n_regimes,
            "statistics": stats_df.to_dict(orient="index"),
            **stored,
        })

    @mcp.tool()
    def regime_transition(
        dataset: str,
        column: str = "returns",
        n_regimes: int = 2,
    ) -> dict[str, Any]:
        """Analyze regime transition dynamics.

        Returns the empirical and model transition matrices, steady-state
        distribution, average regime durations, and regime visit counts.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
            n_regimes: Number of regimes to fit.
        """
        from wraquant.regimes.hmm import fit_hmm, regime_transition_analysis

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()

        model = fit_hmm(returns, n_states=n_regimes)
        states = model.predict(returns.values.reshape(-1, 1))
        transmat = model.transmat_

        result = regime_transition_analysis(states, transition_matrix=transmat)

        return _sanitize_for_json({
            "tool": "regime_transition",
            "dataset": dataset,
            "n_regimes": n_regimes,
            "transition_matrix": result["transition_matrix"],
            "empirical_transition_matrix": result["empirical_transition_matrix"],
            "steady_state": result["steady_state"],
            "avg_duration": result["avg_duration"],
            "regime_counts": result["regime_counts"],
        })

    @mcp.tool()
    def select_n_states(
        dataset: str,
        column: str = "returns",
        max_states: int = 5,
    ) -> dict[str, Any]:
        """Select optimal number of HMM states using BIC.

        Fits HMMs with 2..max_states and returns BIC scores,
        recommended state count, and per-state model summaries.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
            max_states: Maximum number of states to evaluate.
        """
        from wraquant.regimes.hmm import select_n_states as _select

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()

        result = _select(returns, max_states=max_states)

        return _sanitize_for_json({
            "tool": "select_n_states",
            "dataset": dataset,
            "max_states": max_states,
            "result": result,
        })

    @mcp.tool()
    def rolling_regime_probability(
        dataset: str,
        column: str = "returns",
        n_regimes: int = 2,
        window: int = 120,
    ) -> dict[str, Any]:
        """Compute time-varying regime probabilities using rolling HMM.

        Fits an HMM at each time step using a rolling window to
        produce regime probability time series for real-time monitoring.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
            n_regimes: Number of regimes.
            window: Rolling window size in observations.
        """
        from wraquant.regimes.hmm import rolling_regime_probability as _rolling

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()

        probs = _rolling(returns, n_states=n_regimes, window=window)

        stored = ctx.store_dataset(
            f"regime_probs_{dataset}", probs,
            source_op="rolling_regime_probability", parent=dataset,
        )

        return _sanitize_for_json({
            "tool": "rolling_regime_probability",
            "dataset": dataset,
            "n_regimes": n_regimes,
            "window": window,
            "latest_probabilities": {
                col: float(probs[col].iloc[-1])
                for col in probs.columns
                if not probs[col].isna().iloc[-1]
            },
            **stored,
        })
