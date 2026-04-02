"""Portfolio construction and optimization prompt templates."""
from __future__ import annotations
from typing import Any


def register_portfolio_prompts(mcp: Any) -> None:

    @mcp.prompt()
    def portfolio_construction(dataset: str = "multi_asset_returns") -> list[dict]:
        """Full portfolio construction: optimize, decompose, regime-adjust."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
First load the wraquant_system_context prompt for full module context.

Construct an optimal portfolio from {dataset} using a rigorous multi-method approach.
This workflow uses opt/ (26 functions), risk/ (95 functions), regimes/ (38 functions),
and backtest/ (38 functions). The goal is not just to find optimal weights, but to
understand WHY those weights are optimal, how robust they are, and how they should
adapt to the current market regime.

---

## Phase 1: Data Preparation & Universe Assessment

1. **Workspace check**: Run workspace_status. Verify {dataset} exists.
   Need multi-asset return data with at least 3 assets (ideally 5-15).
   Minimum 252 trading days. 500+ preferred for regime detection.
   More than 20 assets may cause optimization instability — consider screening first.

2. **Compute returns**: compute_returns on {dataset} if raw prices.
   Verify all assets have the same date range (inner join on dates).
   Check for survivorship bias — are there any assets that started/stopped mid-sample?

3. **Universe quality check**: For each asset, run a quick risk_metrics.
   - Remove assets with annualized vol > 60% (too volatile for portfolio construction).
   - Remove assets with < 200 days of history (insufficient for covariance estimation).
   - Flag assets with identical or near-identical returns (data error or same underlying).

---

## Phase 2: Correlation & Dependence Analysis

4. **Static correlation matrix**: correlation_analysis across all assets.
   Present the full correlation matrix. Flag notable relationships:
   - Pairs with |corr| > 0.80: Very high — limited diversification benefit.
     Consider removing one from each highly correlated pair.
   - Pairs with |corr| < 0.20: Low correlation — excellent diversification candidates.
   - Pairs with corr < -0.20: Negative correlation — natural hedges.

   **Why this matters for optimization**: Mean-variance optimization is extremely sensitive
   to the correlation matrix. Small estimation errors in correlations cause large weight
   swings. This is the fundamental challenge of portfolio optimization.

5. **Correlation stability**: Are correlations stable or time-varying?
   Compute rolling 60-day correlations for the top 3-5 most correlated pairs.
   If correlations are unstable, sample covariance will mislead the optimizer.
   Use DCC-GARCH or shrinkage to get better estimates.

6. **Covariance estimation methods** — compare three approaches:
   - **Sample covariance**: Naive. Works if T >> N (many more observations than assets).
   - **Ledoit-Wolf shrinkage**: Shrinks sample covariance toward a structured target.
     More stable. Recommended default for N > 5 assets.
   - **DCC-GARCH**: Time-varying. Best for capturing current conditions but noisier.
   Report how different the estimates are. If they agree, the estimate is robust.
   If they disagree, the portfolio is sensitive to estimation — use HRP or risk parity.

---

## Phase 3: Multi-Method Portfolio Optimization (opt/ module)

7. **Method 1 — Risk Parity**: optimize_portfolio with method="risk_parity".
   Equal risk contribution from each asset. Each asset contributes 1/N of total portfolio risk.

   **Verify risk contributions**: After optimization, compute component risk for each asset.
   Each should be approximately 1/N. Deviations > 2% from equal = convergence issue.

   **When to use risk parity**: When you have no views on expected returns (agnostic).
   Risk parity doesn't need return estimates — only the covariance matrix.
   This avoids the biggest source of estimation error (expected returns are nearly
   impossible to estimate reliably). Risk parity is the safest default allocation.

   **Limitations**: Risk parity often overweights low-vol assets (bonds) and underweights
   high-vol assets (equities). This needs leverage to match equity-like returns.
   Compute the leverage ratio needed to target 10% annualized portfolio vol.

8. **Method 2 — Maximum Sharpe**: optimize_portfolio with method="max_sharpe".
   Maximizes expected return per unit of risk. Uses both expected returns AND covariance.

   **Warning**: Max Sharpe is EXTREMELY sensitive to expected return estimates.
   Small changes in expected returns cause large weight swings. Often produces
   concentrated portfolios (one or two assets dominate). This is the optimization's
   way of saying "I'm not confident in my inputs."

   **Sanity checks on max Sharpe weights**:
   - Any asset > 40%? Concentration risk. Consider adding a max-weight constraint (e.g., 25%).
   - Any asset at 0%? The optimizer thinks it's dominated. Is that reasonable?
   - Expected Sharpe > 2.0? Almost certainly overfitting. Out-of-sample will be worse.

9. **Method 3 — Hierarchical Risk Parity (HRP)**: optimize_portfolio with method="hrp".
   Uses hierarchical clustering on the correlation matrix. No matrix inversion needed
   (avoids the instability of mean-variance). Robust to estimation error.

   **When to use HRP**: When you have many assets (> 10) or the correlation matrix is
   unstable. HRP produces more diversified and stable weights than mean-variance.
   It's the best "set and forget" allocation for most portfolios.

10. **Method 4 — Minimum Variance**: optimize_portfolio with method="min_variance".
    Minimizes total portfolio volatility. Ignores expected returns entirely.
    Overweights low-vol, low-correlation assets. Conservative choice.

    **When to use**: In bear markets or when risk reduction is priority over return.
    Min-variance portfolios tend to have lower drawdowns but also lower returns.

11. **Method comparison table**:
    | Method | Asset 1 | Asset 2 | ... | Ann. Return | Ann. Vol | Sharpe | Max DD |
    |--------|---------|---------|-----|-------------|----------|--------|--------|
    | Risk Parity | | | | | | | |
    | Max Sharpe | | | | | | | |
    | HRP | | | | | | | |
    | Min Variance | | | | | | | |
    | Equal Weight | | | | | | | |

    Include equal weight (1/N) as a naive benchmark. If 1/N beats optimized portfolios,
    the optimization is overfitting — the "1/N puzzle" strikes again.

---

## Phase 4: Risk Decomposition of Best Portfolio

12. **Select best method**: Based on the comparison, select the recommended method.
    Criteria: highest out-of-sample Sharpe (from walk-forward), lowest max drawdown,
    most stable weights over time. Usually HRP or risk parity wins.

13. **Portfolio risk analytics** (risk/ module) on the selected portfolio:
    - **Component VaR**: Each asset's contribution to total portfolio VaR.
      Is risk concentrated? Any asset contributing > 30% of portfolio VaR = concentration.
    - **Marginal VaR**: Adding $1 to each asset — which increases risk most?
    - **Diversification ratio**: Sum of standalone vols / portfolio vol.
      > 1.5 = well diversified. < 1.2 = concentrated or highly correlated.
    - **Risk budgeting**: Actual risk allocation vs intended. Any unintended risk tilts?

14. **VaR and stress testing**:
    - var_analysis on the portfolio returns: 95% and 99% VaR and CVaR.
    - stress_test with the top 3 crisis scenarios (GFC, COVID, and the portfolio's
      specific worst-case from Phase 3 analysis).
    - If worst-case loss > 20%: recommend tail hedges (put options, VIX calls, managed futures).

---

## Phase 5: Regime-Aware Adjustment (regimes/ module)

15. **Regime detection**: detect_regimes on the portfolio returns (or a representative
    index like SPY) with method="hmm", n_regimes=2.
    - Current regime: bull (state 0) or bear (state 1)?
    - Confidence: regime probability > 0.8 = high confidence.
    - Expected duration: how long should this regime last?

16. **Regime-conditional optimization**: Re-run optimize_portfolio separately on data
    from each regime only.
    - Bull regime optimal weights: Should overweight risky assets.
    - Bear regime optimal weights: Should overweight defensive assets / cash.
    - How different are regime-conditional weights from unconditional?
      Large differences = the portfolio SHOULD be adjusted for the current regime.

17. **Tactical adjustment**: Blend unconditional and regime-conditional weights.
    If regime confidence > 80%: use regime-conditional weights.
    If regime confidence 50-80%: blend 50/50 with unconditional weights.
    If regime confidence < 50%: stick with unconditional weights.
    This avoids over-reacting to noisy regime signals.

18. **Regime transition warning**: From the transition matrix, compute the probability
    of switching regimes in the next 5 trading days. If > 20%, flag as "regime transition
    risk" — don't make large allocation changes, wait for regime to stabilize.

---

## Phase 6: Backtesting & Validation

19. **Walk-forward backtest**: run_backtest for the selected optimization method.
    Use walk-forward (re-optimize every 63 days using trailing 252 days of data).
    This tests TRUE out-of-sample performance — the only performance that matters.

    Compare walk-forward to static (optimize once on full sample). If static >> walk-forward,
    the optimization is overfitting to in-sample data.

20. **Backtest metrics**: backtest_metrics for the walk-forward portfolio.
    Report: annualized return, vol, Sharpe, Sortino, max drawdown, Calmar,
    hit ratio, recovery factor. Monthly returns table.

21. **Comprehensive tearsheet**: comprehensive_tearsheet on the walk-forward portfolio.
    Include: equity curve, drawdown chart, rolling Sharpe, monthly returns heatmap.

---

## Phase 7: Final Recommendation

22. **Recommended portfolio** — present in a structured format:

    **Allocation**:
    | Asset | Weight | Risk Contribution | Rationale |
    |-------|--------|-------------------|-----------|
    | ... | X% | Y% of total risk | ... |

    **Expected characteristics**:
    - Annualized return: X% (based on walk-forward)
    - Annualized volatility: X%
    - Sharpe ratio: X.XX
    - Max drawdown: X% (historical worst case)
    - 99% daily VaR: X%

    **Regime adjustment** (if applicable):
    - Current regime: [bull/bear] with X% confidence
    - Adjustment: [increase/decrease] equity by X%, [increase/decrease] bonds by X%
    - Revert to strategic weights when regime confidence drops below 60%

    **Rebalancing rule**: Rebalance when any position drifts > 5% from target weight
    or monthly, whichever comes first.

    **Key risks**: Top 3 risks to this allocation and their mitigants.

**Related prompts**: Use risk_parity_deep_dive for deeper risk parity analysis,
factor_tilt to add factor exposures, tactical_allocation for dynamic regime adjustment,
portfolio_stress_test for comprehensive stress testing, asset_allocation for
strategic multi-asset allocation with Black-Litterman views.
"""}}]

    @mcp.prompt()
    def portfolio_rebalance(current_dataset: str = "current_portfolio", target_dataset: str = "target_weights") -> list[dict]:
        """Analyze rebalancing costs and optimal execution."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
First load the wraquant_system_context prompt for full module context.

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
First load the wraquant_system_context prompt for full module context.

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
First load the wraquant_system_context prompt for full module context.

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
First load the wraquant_system_context prompt for full module context.

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
First load the wraquant_system_context prompt for full module context.

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
First load the wraquant_system_context prompt for full module context.

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
First load the wraquant_system_context prompt for full module context.

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
First load the wraquant_system_context prompt for full module context.

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
