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

    @mcp.prompt()
    def tactical_allocation(dataset: str = "multi_asset_returns") -> list[dict]:
        """Regime-aware tactical allocation: tilt from strategic weights."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Tactical allocation on {dataset}:

1. **Data**: Load {dataset} from workspace. compute_returns for each asset.
   Need at least equity, bonds, and one alternative (gold/commodities).
2. **Strategic baseline**: optimize_portfolio with method="risk_parity" for the
   long-term strategic allocation. This is the neutral starting point.
3. **Regime detection**: detect_regimes with method="hmm", n_regimes=3 on the
   joint asset returns. Identify risk-on, risk-off, and crisis regimes.
   Current regime? Probability? Expected duration?
4. **Regime-conditional optimal weights**: For each regime, optimize_portfolio
   separately using only data from that regime. How do optimal weights differ?
   - Risk-on: overweight equity, underweight bonds
   - Risk-off: overweight bonds/gold, underweight equity
   - Crisis: maximum defensive (bonds, gold, cash)
5. **Tactical tilt**: Blend strategic weights with regime-conditional weights.
   Tilt magnitude = f(regime confidence). If regime probability > 80%, full tilt.
   If 50-80%, partial tilt. If < 50%, stay strategic.
   Compute: tactical_weight = (1 - tilt) × strategic + tilt × regime_optimal
6. **Momentum overlay**: compute_indicator with ROC(63) for each asset.
   Assets with negative 3-month momentum get weight reduced by 50%.
   Redistributed weight goes to cash or bonds.
7. **Risk check**: portfolio_risk on the tactical portfolio. Is concentration
   acceptable? Component VaR — any single asset > 40% of total risk?
   stress_test — does the tactical portfolio survive crisis scenarios?
8. **Backtest the tilt**: run_backtest comparing strategic (static) vs tactical (dynamic).
   backtest_metrics for both. Does tactical add Sharpe? Reduce drawdown?
9. **Summary**: Current tactical weights vs strategic. Regime justification.
   Expected improvement from tactical tilt. Risk budget check.
"""}}]

    @mcp.prompt()
    def risk_parity_deep_dive(dataset: str = "multi_asset_returns") -> list[dict]:
        """Equal risk contribution analysis: deep dive into risk parity construction."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Risk parity deep dive on {dataset}:

1. **Data**: Load {dataset} from workspace. compute_returns for each asset.
2. **Naive risk parity**: optimize_portfolio with method="risk_parity".
   Extract weights. Verify: each asset's risk contribution should be equal
   (1/N of total portfolio risk). Compute marginal and component risk contributions.
3. **Risk contribution verification**: For each asset, compute:
   - Marginal risk contribution: ∂σ_p/∂w_i = (Σw)_i / σ_p
   - Component risk contribution: w_i × (Σw)_i / σ_p
   - Percent risk contribution: should equal 1/N for all assets.
   If any asset deviates > 2% from equal, the optimizer didn't converge properly.
4. **Leveraged risk parity**: Scale weights so portfolio targets 10% annualized vol.
   Leverage ratio = target_vol / unlevered_portfolio_vol. Is leverage reasonable (< 3x)?
5. **Covariance sensitivity**: How do weights change under different covariance estimates?
   - Sample covariance (naive)
   - Shrinkage estimator (Ledoit-Wolf)
   - DCC-GARCH conditional covariance
   Compare weights under each — stable or sensitive to estimation?
6. **Regime-conditional risk parity**: detect_regimes, then compute risk parity weights
   in each regime. In crisis, volatilities spike AND correlations rise — how much do
   weights shift? Does risk parity naturally reduce equity in crisis?
7. **Comparison to alternatives**: Compute and compare side by side:
   - Equal weight (1/N)
   - Risk parity (equal risk contribution)
   - Min variance (optimize_portfolio with method="min_variance")
   - Max Sharpe (optimize_portfolio with method="max_sharpe")
   backtest_metrics for each. Which has best risk-adjusted return? Lowest drawdown?
8. **Transaction cost analysis**: How often does risk parity rebalance require > 5%
   turnover? Estimate annual transaction cost drag at 10bps per trade.
9. **Summary**: Risk parity weights and verification. Leverage ratio needed.
   Robustness to covariance estimation. Comparison to alternatives. Rebalancing cost.
"""}}]

    @mcp.prompt()
    def factor_tilt(dataset: str = "portfolio_returns", factors_dataset: str = "factor_returns") -> list[dict]:
        """Factor tilt analysis: tilt portfolio toward desired factor exposures."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Factor tilt analysis for {dataset} with factors from {factors_dataset}:

1. **Data**: Load {dataset} and {factors_dataset} from workspace.
   Factor data should include standard factors: market, size, value, momentum, quality, low-vol.
   compute_returns if needed.
2. **Current factor exposure**: factor_analysis — regress portfolio returns on factors.
   Report current betas to each factor. Which factors is the portfolio exposed to?
   Is there unintended exposure (e.g., a "quality" portfolio with hidden momentum bet)?
3. **Desired factor profile**: Define target exposures:
   - Value tilt: increase value beta by 0.2
   - Momentum tilt: increase momentum beta by 0.15
   - Low-vol tilt: increase low-vol beta by 0.1
   Compare current vs desired factor betas.
4. **Tilt construction**: For each asset in the portfolio, compute its factor betas.
   Solve for weight adjustments that move portfolio factor betas toward targets
   while minimizing tracking error vs current portfolio.
   optimize_portfolio with factor exposure constraints if available.
5. **Factor risk attribution**: After tilting, decompose portfolio risk into factor
   contributions. What fraction of risk comes from each factor?
   Is the factor tilt adding intended risk or just noise?
6. **Factor timing signal**: detect_regimes on each factor's returns.
   Which factors are in favorable regimes? Tilt toward factors in bull regimes.
   Rolling factor Sharpe (12-month) — which factors have recent momentum?
7. **Backtest the tilt**: run_backtest comparing untilted vs tilted portfolio.
   Does the tilt add alpha? Reduce drawdown? Improve Sharpe?
   walk_forward — is the factor tilt robust out-of-sample?
8. **Factor crowding check**: If many investors tilt similarly, factors get crowded.
   Check recent factor valuations (value spread, momentum drawdown).
   Crowded factors are vulnerable to reversals.
9. **Summary**: Current factor exposures, proposed tilt, expected impact.
   Factor regime status. Backtest results. Crowding risk assessment.
"""}}]

    @mcp.prompt()
    def esg_screen(dataset: str = "portfolio_returns", esg_scores: str = "esg_data") -> list[dict]:
        """ESG-constrained portfolio construction."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
ESG-constrained portfolio analysis for {dataset} with {esg_scores}:

1. **Data**: Load {dataset} (asset returns) and {esg_scores} (ESG ratings per asset).
   ESG data should include E (environmental), S (social), G (governance) sub-scores
   and a composite score. Check workspace_status for available fields.
2. **Current ESG profile**: Compute portfolio-weighted average ESG score using current weights.
   Break down by E, S, G sub-scores. Which assets drag ESG down? Which are leaders?
3. **Exclusion screen**: Remove assets with ESG composite below threshold (e.g., bottom 20%).
   Remove any assets in excluded sectors (tobacco, weapons, coal) if sector labels available.
   How many assets removed? What percent of original portfolio?
4. **Re-optimize after exclusion**: optimize_portfolio with method="max_sharpe" on the
   screened universe. Compare weights and expected Sharpe to unconstrained portfolio.
   ESG cost = Sharpe_unconstrained - Sharpe_ESG. How large is the ESG "tax"?
5. **ESG tilt (best-in-class)**: Instead of excluding, tilt weights toward higher ESG scores.
   Weight multiplier = (ESG_score / median_ESG)^tilt_strength. Re-normalize to sum to 1.
   This keeps all assets but overweights ESG leaders.
6. **Carbon intensity**: If carbon data available, compute portfolio-weighted carbon intensity.
   Compare to benchmark. Set carbon budget and constrain optimization accordingly.
7. **Performance impact**: run_backtest for three versions:
   - Unconstrained (no ESG)
   - Exclusion-based ESG
   - Tilt-based ESG (best-in-class)
   backtest_metrics for each. Does ESG hurt or help performance?
8. **Factor decomposition**: factor_analysis on each version. Is ESG performance
   explained by factor tilts (quality, low-vol) or genuine ESG alpha?
9. **Summary**: Portfolio ESG score improvement. Performance cost (if any).
   Recommended approach (exclusion vs tilt). Factor attribution of ESG effect.
"""}}]
