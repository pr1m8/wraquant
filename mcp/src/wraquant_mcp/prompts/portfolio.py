"""Portfolio construction and optimization prompt templates."""
from __future__ import annotations
from typing import Any


def register_portfolio_prompts(mcp: Any) -> None:

    @mcp.prompt()
    def portfolio_construction(dataset: str = "multi_asset_returns") -> list[dict]:
        """Full portfolio construction: optimize, decompose, regime-adjust."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Construct an optimal portfolio from {dataset}:

1. compute_returns if needed.
2. correlation_analysis — check for high correlations (>0.8).
3. optimize_portfolio with method="risk_parity" — equal risk contribution.
4. Also try method="max_sharpe" and method="hrp" — compare weights.
5. portfolio_risk — component VaR, diversification ratio for best method.
6. detect_regimes — adjust weights for current regime.
7. comprehensive_tearsheet on the portfolio returns.
8. Summary: recommended weights, risk breakdown, regime adjustment.
"""}}]

    @mcp.prompt()
    def portfolio_rebalance(current_dataset: str = "current_portfolio", target_dataset: str = "target_weights") -> list[dict]:
        """Analyze rebalancing costs and optimal execution."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Rebalance analysis: {current_dataset} → {target_dataset}:

1. Compare current weights vs target weights — drift analysis.
2. Compute turnover (sum of absolute weight changes).
3. Estimate transaction costs (spread + impact + commission).
4. Is the rebalance worth the cost? (compare expected alpha vs cost).
5. Optimal trade schedule — TWAP, VWAP, or IS?
6. regime check — should we delay rebalance if regime is shifting?
"""}}]

    @mcp.prompt()
    def factor_attribution(dataset: str = "portfolio_returns", factors_dataset: str = "factor_returns") -> list[dict]:
        """Factor exposure and risk attribution analysis."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Factor attribution for {dataset}:

1. factor_analysis — regress returns on factors.
2. Report: alpha (skill), factor betas (exposures), R² (explained).
3. Factor risk contribution — which factors drive the risk?
4. Rolling factor exposure — are betas stable or drifting?
5. Regime-conditional factor exposure — do betas change in crisis?
6. Summary: is performance from alpha or factor exposure? Any unwanted exposures?
"""}}]

    @mcp.prompt()
    def portfolio_stress_test(dataset: str = "portfolio_returns") -> list[dict]:
        """Comprehensive portfolio stress testing."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Stress test portfolio {dataset}:

1. stress_test with all 7 crisis scenarios.
2. correlation_stress — correlations go to 1 in crisis.
3. crisis_drawdowns — top 5 worst historical drawdowns.
4. portfolio_risk — which assets contribute most to tail risk?
5. What hedges would have helped in each scenario?
6. Summary: what's the worst-case loss? How to protect against it?
"""}}]

    @mcp.prompt()
    def asset_allocation(datasets: str = "equity,bonds,gold,reits") -> list[dict]:
        """Multi-asset strategic allocation with regime awareness."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Strategic asset allocation across {datasets}:

1. Load/compute returns for each asset class.
2. efficient_frontier — plot risk-return tradeoff.
3. optimize_portfolio with risk_parity — baseline allocation.
4. black_litterman if you have views — tilt from equilibrium.
5. detect_regimes — current environment risk-on or risk-off?
6. Regime-conditional optimal weights vs unconditional.
7. Summary: recommended allocation, expected return/vol, regime adjustment.
"""}}]
