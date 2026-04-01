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
