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

    @mcp.prompt()
    def market_breadth(universe: str = "sp500") -> list[dict]:
        """Market breadth analysis: advance/decline, McClellan, percent above MA."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Market breadth analysis for {universe}:

1. **Data**: Check workspace_status for universe constituent data. If not present, note that
   constituent price data for {universe} needs loading via OpenBB MCP or store_data.
2. **Advance/Decline line**: For each day, count advancers vs decliners across all constituents
   using compute_returns. Compute cumulative A/D line. Is the A/D line confirming or
   diverging from the index? Divergence = warning signal.
3. **McClellan Oscillator**: Compute 19-day and 39-day EMA of (advances - declines).
   McClellan = EMA(19) - EMA(39). Positive = breadth thrust, negative = breadth deterioration.
   Use compute_indicator with EMA on the net advances series.
4. **McClellan Summation Index**: Cumulative sum of the McClellan Oscillator.
   Rising = sustained breadth improvement. Crossing zero = regime shift.
5. **Percent above MA**: For each constituent, check if price > SMA(50) and price > SMA(200).
   Compute percentage of universe above each MA. >70% above SMA(50) = healthy breadth,
   <30% = weak breadth even if index is holding up.
6. **New Highs / New Lows**: Count 52-week new highs vs new lows. Ratio > 1 = healthy,
   ratio < 0.5 = deteriorating. Expanding new lows in a rising market = bearish divergence.
7. **Sector breadth**: Break down breadth by sector — which sectors are leading/lagging?
   Use sector_comparison logic on sector-level A/D data.
8. **Summary**: Is breadth confirming the trend? Any divergences? McClellan signal?
   What does breadth say about market sustainability?
"""}}]

    @mcp.prompt()
    def cross_asset_study(datasets: str = "SPY,TLT,GLD,DXY") -> list[dict]:
        """Cross-asset correlation regime and flight-to-quality analysis."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Cross-asset study for {datasets}:

1. **Data**: Check workspace_status for each asset in [{datasets}]. Load/verify all datasets.
   compute_returns for each asset.
2. **Static correlation matrix**: correlation_analysis across all assets. Note key relationships:
   SPY-TLT (negative = normal, positive = stagflation), SPY-GLD (flight to quality),
   SPY-DXY (risk appetite proxy).
3. **Rolling correlation**: Compute 60-day rolling correlations for key pairs.
   Use correlation_analysis with rolling windows. Are correlations stable or shifting?
4. **Correlation regime detection**: detect_regimes on the rolling correlation series itself.
   Identify "normal" vs "crisis" correlation regimes. In crisis: do correlations spike to 1
   (contagion) or do safe havens decorrelate (flight to quality)?
5. **Flight-to-quality signal**: When SPY drawdown > 5%, do TLT and GLD rally?
   Compute conditional correlations: corr(SPY, TLT | SPY < -1σ) vs corr(SPY, TLT | normal).
   If TLT rallies when SPY falls, the hedge is working.
6. **DXY impact**: Strong dollar periods — how do equities and gold respond?
   Compute regime-conditional returns for each asset during DXY appreciation vs depreciation.
7. **Macro regime synthesis**: Combine signals into macro state:
   - Risk-on: SPY up, TLT down, GLD flat, DXY weak
   - Risk-off: SPY down, TLT up, GLD up, DXY strong
   - Stagflation: SPY down, TLT down, GLD up, DXY mixed
8. **Summary**: Current cross-asset regime? Are safe havens working? Hedging implications?
   Asset allocation tilt based on cross-asset signals?
"""}}]

    @mcp.prompt()
    def seasonality_analysis(dataset: str = "returns", column: str = "close") -> list[dict]:
        """Seasonality analysis: day-of-week, month effects, holiday patterns."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Seasonality analysis on {dataset} (column: {column}):

1. **Data**: Load {dataset} from workspace. compute_returns if raw prices.
   Ensure sufficient history (3+ years minimum for seasonal patterns).
2. **Day-of-week effect**: Group returns by weekday (Mon–Fri). Compute mean, median,
   win rate, and Sharpe for each day. Use analyze() on each day's subset.
   Is there a "Monday effect" (negative) or "Friday effect" (positive)?
   Test statistical significance with t-test for each day vs overall mean.
3. **Month-of-year effect**: Group returns by calendar month. Compute same stats.
   "Sell in May"? January effect? September dip? Rank months by average return.
   Compute month-conditional Sharpe ratios. Any month consistently negative?
4. **Turn-of-month effect**: Separate returns into last 3 + first 3 trading days of month
   vs mid-month. Is turn-of-month systematically stronger? (Common equity pattern.)
5. **Holiday patterns**: Identify pre-holiday trading days (day before market holidays).
   Compute average pre-holiday return vs normal days. Historically bullish?
6. **Quarter-end effects**: Group by quarter-end proximity (last 5 days of quarter).
   Window dressing / rebalancing flows — any pattern?
7. **Regime interaction**: detect_regimes first, then check if seasonal patterns hold
   across regimes. Does "sell in May" only work in bull markets?
8. **Backtest seasonal strategy**: Simple rule — only hold in favorable months/days.
   run_backtest and compare to buy-and-hold. Is the seasonal edge tradeable after costs?
9. **Summary**: Which seasonal patterns are statistically significant? Robust across regimes?
   Any actionable patterns after transaction costs?
"""}}]

    @mcp.prompt()
    def liquidity_screen(universe_dataset: str = "universe_prices") -> list[dict]:
        """Liquidity screening: rank assets by Amihud illiquidity, spread, and turnover."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Liquidity screen on {universe_dataset}:

1. **Data**: Load {universe_dataset} from workspace. Needs price and volume data for each asset.
   compute_returns for each asset.
2. **Amihud illiquidity ratio**: For each asset, compute Amihud = mean(|return| / dollar_volume).
   Higher = less liquid. Use liquidity_analysis or compute manually.
   Rank all assets by Amihud ratio (ascending = most liquid first).
3. **Turnover ratio**: If shares outstanding available, compute turnover = volume / shares_out.
   Higher turnover = more liquid. Rank assets. Flag any with turnover < 0.1% (illiquid).
4. **Bid-ask spread proxy**: If tick data unavailable, estimate effective spread using
   Roll's measure: spread ≈ 2 × sqrt(-cov(Δp_t, Δp_{t-1})). Negative autocovariance
   implies spread cost. Rank assets by estimated spread.
5. **Volume stability**: Compute coefficient of variation of daily volume (std/mean).
   Erratic volume = unreliable liquidity. Flag assets with CV > 1.5.
6. **Composite liquidity score**: Combine Amihud (40%), spread proxy (30%), turnover (20%),
   volume stability (10%) into a single score. Normalize each to [0, 1].
   Score > 0.7 = highly liquid, 0.4-0.7 = moderate, < 0.4 = illiquid.
7. **Liquidity-return relationship**: correlation_analysis between liquidity score and
   average return. Is there a liquidity premium (illiquid assets earning more)?
8. **Capacity estimate**: For each asset, estimate max position size that can be traded
   within 1 day without moving price > 1% (using avg daily volume × 0.1).
9. **Summary**: Ranked liquidity table. Which assets pass the liquidity filter?
   Any illiquidity premium detected? Capacity constraints for the universe?
"""}}]
