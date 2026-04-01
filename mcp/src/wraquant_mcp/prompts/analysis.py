"""Analysis & research prompt templates."""
from __future__ import annotations
from typing import Any


def register_analysis_prompts(mcp: Any) -> None:

    @mcp.prompt()
    def equity_deep_dive(ticker: str = "AAPL") -> list[dict]:
        """Comprehensive single-stock analysis: stats, vol, regimes, TA, risk."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Perform a deep analysis of {ticker}:

1. **Data**: Check workspace_status. If no data, note it needs loading via OpenBB MCP or store_data.
2. **Returns**: compute_returns on price data.
3. **Statistics**: analyze() for comprehensive stats — mean, vol, skew, kurtosis, stationarity.
4. **Distribution**: Check normality. Note fat tails if present.
5. **Volatility**: fit_garch with model="GJR", dist="t". Report persistence (>0.95 = high), half-life.
6. **Regimes**: detect_regimes with 2 states. Current regime? Per-regime Sharpe?
7. **Technical**: compute_indicator for RSI (overbought >70?), MACD (crossover?), Bollinger Bands (squeeze?).
8. **Risk**: risk_metrics — Sharpe, Sortino, max drawdown. Is drawdown recovering?
9. **Summary**: Synthesize — favorable regime? Good risk-adjusted return? Any signals?
"""}}]

    @mcp.prompt()
    def sector_comparison(tickers: str = "XLK,XLF,XLE,XLV") -> list[dict]:
        """Compare multiple sectors/stocks side by side."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Compare these assets: {tickers}

1. Load/check data for each ticker.
2. compute_returns for each.
3. correlation_analysis across all — highly correlated (>0.8) means low diversification.
4. detect_regimes on each — same regime or diverging?
5. risk_metrics side by side — which has best Sharpe? Lowest drawdown?
6. cointegration_test between pairs — any cointegrated for pairs trading?
7. Rank by risk-adjusted return. Recommend overweight/underweight.
"""}}]

    @mcp.prompt()
    def macro_analysis() -> list[dict]:
        """Macro regime and cross-asset correlation analysis."""
        return [{"role": "user", "content": {"type": "text", "text": """
Analyze current macro environment:

1. Check workspace for macro data (SPY, TLT, GLD, VIX, DXY).
2. detect_regimes on SPY — bull or bear market?
3. correlation_analysis across all assets — elevated correlations = risk-off.
4. stress_test with historical crisis scenarios.
5. Yield curve analysis if bond data available.
6. Assessment: risk-on or risk-off? Asset allocation implications?
"""}}]

    @mcp.prompt()
    def earnings_impact(ticker: str = "AAPL", event_date: str = "2024-01-25") -> list[dict]:
        """Event study around earnings announcement."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Analyze {ticker} earnings impact on {event_date}:

1. Load price data: 60 days before, 20 days after the event.
2. compute_returns.
3. Estimate market model (beta vs SPY) on estimation window (-60 to -10 days).
4. Compute abnormal returns in event window (-5 to +10 days).
5. Calculate CAR (Cumulative Abnormal Return).
6. fit_garch pre vs post — did volatility spike?
7. Assessment: positive or negative surprise? How long did impact persist?
"""}}]

    @mcp.prompt()
    def ipo_analysis(ticker: str = "TICKER") -> list[dict]:
        """Post-IPO behavior analysis."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Analyze {ticker} post-IPO:

1. Load price data from IPO date.
2. compute_returns — check for IPO pop and subsequent drift.
3. distribution_fit — heavy-tailed? Skewed?
4. fit_garch — is volatility declining over time (settling)?
5. detect_regimes — has it entered a stable regime?
6. Compare to sector benchmark.
7. Assessment: IPO pop absorbed? Risk profile stabilizing?
"""}}]
