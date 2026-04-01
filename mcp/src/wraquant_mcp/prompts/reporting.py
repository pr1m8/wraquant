"""Reporting, monitoring, and news/fundamental prompt templates."""
from __future__ import annotations
from typing import Any


def register_reporting_prompts(mcp: Any) -> None:

    @mcp.prompt()
    def daily_risk_monitor(dataset: str = "portfolio_returns") -> list[dict]:
        """Daily risk monitoring checklist."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Daily risk check for {dataset}:

1. risk_metrics — today's Sharpe, vol, drawdown status.
2. var_analysis — current VaR. Any breaches in last 5 days?
3. detect_regimes — regime probability. Any regime shift signal?
4. correlation_analysis — correlation change vs 20-day average.
5. stress_test — would current portfolio survive GFC/COVID?
6. Summary: GREEN (normal), YELLOW (elevated risk), RED (action needed).
"""}}]

    @mcp.prompt()
    def weekly_portfolio_review(dataset: str = "portfolio_returns") -> list[dict]:
        """Weekly portfolio review with attribution and outlook."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Weekly review for {dataset}:

1. This week's performance — return, vol, Sharpe.
2. factor_attribution — what drove returns this week?
3. portfolio_risk — risk concentration changed?
4. detect_regimes — current regime vs last week.
5. Rebalance signal — drift > threshold? Worth rebalancing?
6. Regime outlook — expected duration, transition probability.
7. Next week's positioning recommendation.
"""}}]

    @mcp.prompt()
    def strategy_tearsheet(dataset: str = "strategy_returns") -> list[dict]:
        """Full strategy performance tearsheet."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Generate tearsheet for {dataset}:

1. comprehensive_tearsheet — all metrics.
2. Monthly returns table.
3. Drawdown analysis — top 5 drawdowns with dates, depth, recovery.
4. Rolling metrics — 12m Sharpe, 6m vol, 3m drawdown.
5. detect_regimes — regime-conditional performance.
6. Distribution analysis — is the return distribution normal? Tail risk?
7. Comparison to benchmark if available.
8. Summary: key strengths and weaknesses.
"""}}]

    @mcp.prompt()
    def research_summary() -> list[dict]:
        """Summarize current workspace: datasets, models, findings."""
        return [{"role": "user", "content": {"type": "text", "text": """
Summarize current research workspace:

1. workspace_status — what datasets and models exist?
2. workspace_history — what operations have been performed?
3. For each fitted model: key metrics (persistence, Sharpe, etc.)
4. For each dataset: shape, date range, what it contains.
5. Any notes in the journal?
6. Summary: what has been done, what are the key findings so far?
"""}}]

    @mcp.prompt()
    def sentiment_analysis(ticker: str = "AAPL") -> list[dict]:
        """News sentiment scoring and impact analysis."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Sentiment analysis for {ticker}:

1. If news data available: compute sentiment scores.
2. Aggregate sentiment over different windows (1d, 5d, 20d).
3. Correlate sentiment with returns — does sentiment predict returns?
4. earnings_impact analysis if earnings data available.
5. News signal: convert sentiment to trading signal.
6. Backtest the sentiment signal.
7. Summary: is sentiment informative? Lead-lag relationship?
"""}}]

    @mcp.prompt()
    def fundamental_screen(universe: str = "sp500_returns") -> list[dict]:
        """Fundamental quality screening and valuation."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Fundamental screen on {universe}:

1. If fundamental data available: compute ratios (P/E, P/B, ROE, D/E).
2. Piotroski F-score for quality ranking (0-9).
3. Rank by composite quality score.
4. Filter: F-score >= 7 (high quality).
5. Among quality stocks: which are undervalued (low P/E, high ROE)?
6. Backtest a quality factor — long high quality, short low quality.
7. Summary: top quality picks, valuation assessment.
"""}}]

    @mcp.prompt()
    def microstructure_analysis(dataset: str = "tick_data") -> list[dict]:
        """Market microstructure and liquidity analysis."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Microstructure analysis on {dataset}:

1. Compute spread metrics — quoted, effective, realized spread.
2. liquidity_analysis — Amihud illiquidity, Kyle's lambda.
3. Toxicity — VPIN, order flow imbalance. Is informed trading present?
4. Market quality — variance ratio, efficiency.
5. Intraday pattern — U-shaped volatility? When is liquidity best?
6. Execution cost estimate — how much would a large order cost?
7. Summary: is the market liquid? Toxic flow present? Best time to trade?
"""}}]

    @mcp.prompt()
    def execution_optimization(dataset: str = "prices", quantity: int = 10000) -> list[dict]:
        """Optimal execution scheduling and cost analysis."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Optimize execution of {quantity} shares using {dataset}:

1. Estimate current market conditions — spread, depth, volatility.
2. Compare schedules: TWAP, VWAP, Implementation Shortfall.
3. Almgren-Chriss optimal trajectory — minimize impact + risk.
4. Expected total cost breakdown: spread + impact + timing risk.
5. Close auction allocation — save portion for closing?
6. Summary: recommended schedule, expected cost, risk.
"""}}]

    @mcp.prompt()
    def bayesian_analysis(dataset: str = "returns") -> list[dict]:
        """Bayesian inference on financial data."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Bayesian analysis on {dataset}:

1. Bayesian Sharpe ratio — posterior distribution with credible interval.
2. Is P(Sharpe > 0) > 95%? If not, insufficient evidence of alpha.
3. Bayesian changepoint detection — posterior on break locations.
4. Bayesian regression — posterior on coefficients with uncertainty.
5. Model comparison — WAIC/LOO across competing models.
6. Summary: what does the Bayesian view add beyond point estimates?
"""}}]

    @mcp.prompt()
    def causal_analysis(treatment_dataset: str = "treated", control_dataset: str = "control") -> list[dict]:
        """Causal inference for policy/event impact."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Causal impact analysis: {treatment_dataset} vs {control_dataset}:

1. Difference-in-differences — parallel trends check.
2. Synthetic control — construct counterfactual.
3. Pre-treatment RMSPE — good fit?
4. Placebo tests — is the effect significant vs random?
5. Granger causality — does X predict Y?
6. Event study — CAR with significance test.
7. Summary: causal effect size, confidence, robustness.
"""}}]

    @mcp.prompt()
    def monthly_report(dataset: str = "portfolio_returns") -> list[dict]:
        """Comprehensive monthly performance report."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Monthly performance report for {dataset}:

1. **Data**: Load {dataset} from workspace. Filter to the most recent complete month.
   Also load full history for context.
2. **Month performance summary**: compute_returns for the month.
   - MTD return (absolute and annualized)
   - MTD volatility (annualized)
   - MTD Sharpe ratio
   - MTD max drawdown (intra-month)
   Compare to benchmark if available.
3. **Context — YTD and inception**: risk_metrics for YTD and since inception.
   Is this month above or below average? Percentile rank vs all historical months.
4. **Monthly returns table**: Build a calendar-style table of monthly returns
   (rows = years, columns = months). Highlight current month. Color code:
   green > 0, red < 0. Show annual totals.
5. **Drawdown status**: Current drawdown from peak. How far from high-water mark?
   If in drawdown: depth, duration so far, comparison to historical drawdowns.
   crisis_drawdowns — is this a top-5 drawdown?
6. **Risk snapshot**: var_analysis at 95% and 99%. Is VaR higher or lower than
   last month? fit_garch — conditional vol trending up or down?
7. **Regime status**: detect_regimes — current regime. Has regime changed this month?
   If regime shifted, note the date and implications.
8. **Attribution**: If factor data available, factor_analysis for the month.
   What drove returns — market, factors, or alpha?
9. **Outlook**: Based on current regime, GARCH vol forecast (10-day), and momentum,
   what is the near-term outlook? Any warning signals?
10. **Summary table**: Present key metrics in a clean summary:
    MTD return, YTD return, Ann. return, Ann. vol, Sharpe, Max DD, Current DD.
"""}}]

    @mcp.prompt()
    def attribution_report(dataset: str = "portfolio_returns", benchmark_dataset: str = "benchmark_returns") -> list[dict]:
        """Brinson attribution: allocation, selection, and interaction effects."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Brinson attribution for {dataset} vs {benchmark_dataset}:

1. **Data**: Load {dataset} (portfolio sector/asset returns and weights) and
   {benchmark_dataset} (benchmark sector/asset returns and weights) from workspace.
   Both need asset-level returns AND weights for the same period.
2. **Total active return**: Portfolio return - benchmark return = active return.
   Is it positive (outperformance) or negative (underperformance)?
3. **Brinson-Fachler decomposition**: For each sector/asset, compute:
   - **Allocation effect**: (w_p - w_b) × (R_b_sector - R_b_total)
     Were you overweight in sectors that outperformed the benchmark?
   - **Selection effect**: w_b × (R_p_sector - R_b_sector)
     Within each sector, did you pick better stocks than the benchmark?
   - **Interaction effect**: (w_p - w_b) × (R_p_sector - R_b_sector)
     Interaction of allocation and selection decisions.
   Sum across sectors: Total active = allocation + selection + interaction.
4. **Attribution by sector**: Present a table with columns:
   Sector | Port Weight | Bench Weight | Port Return | Bench Return | Allocation | Selection | Interaction | Total
   Which sectors contributed most to outperformance/underperformance?
5. **Top contributors / detractors**: Rank sectors by total contribution.
   Top 3 contributors and bottom 3 detractors. Explain why.
6. **Rolling attribution**: Compute 12-month rolling attribution.
   Is outperformance coming consistently from allocation or selection?
   Skill vs luck: consistent alpha from selection = genuine stock-picking skill.
7. **Risk-adjusted attribution**: Scale attribution effects by tracking error.
   Information ratio per attribution component. Is the allocation bet
   adding value per unit of risk?
8. **Summary**: Active return decomposition. Main source of alpha (allocation vs selection).
   Key sector bets and their impact. Is the attribution pattern consistent over time?
"""}}]

    @mcp.prompt()
    def compliance_check(dataset: str = "portfolio_holdings", limits_json: str = '{"max_position":0.10,"max_sector":0.30,"max_drawdown":0.15,"min_liquidity_days":3}') -> list[dict]:
        """Compliance check: verify portfolio against risk limits."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Compliance check for {dataset} with limits: {limits_json}:

1. **Data**: Load {dataset} from workspace. Parse limits JSON.
   Need current holdings with weights, sector classifications, and return history.
2. **Position concentration limits**: Check each position weight against max_position limit.
   Flag any position exceeding the limit. Report:
   - Position name, current weight, limit, breach amount
   - How much must be sold to bring position within limit?
3. **Sector concentration limits**: Aggregate weights by sector.
   Check each sector total against max_sector limit. Flag breaches.
   Report sector weights table with limit status (PASS/BREACH).
4. **Drawdown limit**: Compute current drawdown from high-water mark using risk_metrics.
   Is current drawdown > max_drawdown limit? If approaching limit (within 2%),
   issue WARNING. If breached, issue BREACH — requires immediate action.
5. **Liquidity check**: For each position, estimate liquidation time
   (position_value / (avg_daily_volume × price × 0.10) = days to liquidate at 10% ADV).
   Flag positions where liquidation days > min_liquidity_days limit.
   These positions cannot be exited quickly in a crisis.
6. **VaR limit check**: If VaR limit specified, compute var_analysis at 99%.
   Is 1-day VaR within limit? If not, which positions contribute most to VaR?
   portfolio_risk for component VaR breakdown.
7. **Leverage check**: If leverage limit specified, compute gross exposure
   (sum of |weights|) and net exposure (sum of weights). Both within limits?
8. **Regulatory limits**: Check standard regulatory thresholds:
   - Any single equity position > 5% (disclosure threshold in many jurisdictions)
   - Any single issuer > 10% (concentration limit)
   - Cash/liquid assets > minimum (e.g., 5% for mutual funds)
9. **Breach summary report**: Present a compliance dashboard:
   - GREEN: all limits met
   - YELLOW: approaching limits (within 20% of breach)
   - RED: limit breached — action required
   For each breach: position, current value, limit, recommended trade to cure.
10. **Summary**: Overall compliance status (GREEN/YELLOW/RED).
    Number of breaches by category. Recommended trades to cure all breaches.
    Estimated transaction cost of compliance trades.
"""}}]

    @mcp.prompt()
    def client_report(dataset: str = "portfolio_returns", benchmark_dataset: str = "benchmark_returns") -> list[dict]:
        """Client-facing performance summary with clear, non-technical language."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Client report for {dataset} vs benchmark {benchmark_dataset}:

1. **Data**: Load {dataset} and {benchmark_dataset} from workspace.
   Determine reporting period (MTD, QTD, YTD, 1Y, inception).
2. **Performance overview** (client-friendly language):
   - "Your portfolio returned X% this [period], compared to Y% for the benchmark."
   - "Since inception, your portfolio has returned X% annualized (Y% total)."
   - Use risk_metrics for all periods. Present in a clean table:
     Period | Portfolio | Benchmark | Outperformance
3. **Risk summary** (translated for clients):
   - "The portfolio's risk level (volatility) was X%, meaning in a typical year,
     returns could range from [mean - 2σ] to [mean + 2σ]."
   - "The worst peak-to-trough decline was X%, which occurred from [date] to [date]."
   - "The risk-adjusted return (Sharpe ratio) was X, which is [excellent/good/fair/poor]."
     (Sharpe > 1 = excellent, 0.5-1 = good, 0-0.5 = fair, < 0 = poor)
4. **What drove performance**: Simplified attribution.
   - "The main contributor was [sector/asset], which added X%."
   - "The main detractor was [sector/asset], which cost X%."
   If factor data available, factor_analysis simplified:
   "Performance was driven by [market rally / stock selection / sector allocation]."
5. **Market context**: detect_regimes — describe current market environment in plain terms.
   - "Markets are in a [growth/cautious/stressed] phase."
   - "We [are/are not] adjusting the portfolio for current conditions."
6. **Portfolio positioning**: Current top holdings and sector allocation.
   Any recent changes and rationale (in plain terms).
   "We increased exposure to [X] because [simple reason]."
7. **Outlook** (cautious, compliant language):
   - Based on regime analysis and vol forecast, provide directional guidance.
   - Always include appropriate disclaimers about forward-looking statements.
   - "Based on current conditions, we believe [positioning rationale]."
8. **Risk disclosure**: Standard risk language.
   - Past performance is not indicative of future results.
   - Investment involves risk of loss.
   - Benchmark comparison is for reference only.
9. **Summary page**: One-page summary with:
   - Performance chart (if viz available)
   - Key metrics table (return, vol, Sharpe, max DD)
   - Current allocation pie chart description
   - Market regime status
   - One-sentence outlook
"""}}]
