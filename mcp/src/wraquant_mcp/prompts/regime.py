"""Regime detection and market state prompt templates."""
from __future__ import annotations
from typing import Any


def register_regime_prompts(mcp: Any) -> None:

    @mcp.prompt()
    def regime_detection(dataset: str = "returns") -> list[dict]:
        """Multi-method regime detection with comparison."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Detect market regimes in {dataset}:

1. detect_regimes with method="hmm", n_regimes=2 — classic HMM.
2. detect_regimes with method="gmm" — compare without temporal structure.
3. detect_regimes with method="changepoint" — structural breaks.
4. Compare: do all methods agree on current regime?
5. Per-regime statistics — Sharpe, vol, max drawdown, duration.
6. Transition matrix — probability of regime switch.
7. Summary: which regime are we in? How long do regimes typically last?
"""}}]

    @mcp.prompt()
    def market_regime_monitor() -> list[dict]:
        """Current market state and regime probability monitoring."""
        return [{"role": "user", "content": {"type": "text", "text": """
Monitor current market regime:

1. Check workspace for recent data. detect_regimes on latest returns.
2. Current regime probability — how confident are we?
3. Expected duration — how long until regime switch?
4. Historical comparison — what happened last time we were in this regime?
5. Correlation regime — are correlations elevated (crisis signal)?
6. VIX regime check if available.
7. Allocation implications: risk-on (full exposure) or risk-off (hedge/reduce)?
"""}}]

    @mcp.prompt()
    def regime_backtest(dataset: str = "returns") -> list[dict]:
        """Backtest regime-conditional strategy vs unconditional."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Regime-conditional backtest on {dataset}:

1. detect_regimes — identify bull (regime 0) and bear (regime 1).
2. Strategy: full exposure in bull regime, reduce/hedge in bear regime.
3. run_backtest with regime-filtered signals.
4. Compare to buy-and-hold (unconditional).
5. backtest_metrics for both — Sharpe improvement? Drawdown reduction?
6. Walk-forward: is the regime detection adding value out-of-sample?
7. Summary: does regime awareness improve risk-adjusted returns?
"""}}]

    @mcp.prompt()
    def changepoint_analysis(dataset: str = "returns") -> list[dict]:
        """Structural break detection and pre/post comparison."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Changepoint analysis on {dataset}:

1. changepoint_detect with PELT method — where are structural breaks?
2. Binary segmentation for hierarchical ordering of breaks.
3. For each major break: compare pre vs post statistics (mean, vol, Sharpe).
4. CUSUM control chart — is there an ongoing shift?
5. stationarity_test pre vs post each break — did the process change?
6. Summary: what caused the breaks? Is the current regime stable?
"""}}]
