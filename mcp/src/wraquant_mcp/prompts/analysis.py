"""Analysis & research prompt templates."""
from __future__ import annotations
from typing import Any


def register_analysis_prompts(mcp: Any) -> None:

    @mcp.prompt()
    def equity_deep_dive(ticker: str = "AAPL") -> list[dict]:
        """Comprehensive single-stock analysis: stats, vol, regimes, TA, risk."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
First load the wraquant_system_context prompt for full module context.

Perform a comprehensive deep-dive analysis of {ticker}. This is a multi-phase workflow
that touches stats/, vol/, regimes/, ta/, and risk/ modules. The goal is a complete
picture: statistical properties, volatility dynamics, regime state, technical signals,
and risk profile — synthesized into an actionable assessment.

---

## Phase 1: Data Acquisition & Validation

1. **Check workspace**: Run workspace_status to see if prices_{ticker.lower()} already exists.
   If it does, verify the date range — we need at least 2 years (500+ trading days) for
   GARCH and HMM to converge reliably. 3-5 years is ideal.

2. **Load data**: If no data exists, note that price data for {ticker} needs to be loaded
   via OpenBB MCP (openbb_equity_price_historical) or store_data with OHLCV columns.
   Required columns: open, high, low, close, volume. Adjusted close preferred.

3. **Compute returns**: compute_returns on prices_{ticker.lower()}.
   This produces returns_{ticker.lower()} with log or simple returns.
   Verify: no NaN values, no returns > 50% (likely data error if so).

   **If this fails**: Check that the price dataset has a 'close' column. Use
   query_data("SELECT * FROM prices_{ticker.lower()} LIMIT 5") to inspect columns.

---

## Phase 2: Statistical Profile (stats/ module)

4. **Comprehensive statistics**: analyze() on returns_{ticker.lower()}.
   This computes: mean, median, std, skewness, kurtosis, min, max, Jarque-Bera,
   Shapiro-Wilk, ADF stationarity test, autocorrelation (Ljung-Box).

   **Interpretation guide**:
   - Annualized mean return: multiply daily mean by 252. > 10% is good for equities.
   - Annualized volatility: multiply daily std by sqrt(252). 15-25% is typical for single stocks.
   - Skewness: Negative skew is normal for equities (crashes are bigger than rallies).
     < -0.5 means significant left tail risk.
   - Excess kurtosis: > 3 means fat tails (returns are more extreme than normal distribution
     would predict). Most stocks have excess kurtosis of 3-10. This makes standard VaR
     unreliable — use Cornish-Fisher or GARCH-based VaR instead.
   - Jarque-Bera p-value < 0.05: returns are NOT normally distributed (almost always true).
   - ADF p-value < 0.05: returns are stationary (should be true for log returns).
   - Ljung-Box p-value < 0.05: significant autocorrelation (possible momentum or mean reversion).

5. **Distribution analysis**: distribution_fit on returns_{ticker.lower()}.
   Fit Student-t, skewed-t, and normal distributions. Compare AIC/BIC.
   - Student-t degrees of freedom: < 5 = very fat tails, 5-10 = moderate, > 30 ~ normal.
   - Skewed-t captures both asymmetry and fat tails — usually best fit for equities.
   - Report the best-fit distribution and its parameters.

   **Why this matters**: The distribution determines which risk measures are reliable.
   If tails are fat, standard VaR underestimates risk. If skewed, symmetric measures
   (like standard deviation) understate downside risk.

---

## Phase 3: Volatility Dynamics (vol/ module)

6. **GARCH modeling**: fit_garch on returns_{ticker.lower()} with model="GJR", dist="t".
   GJR-GARCH captures the leverage effect (negative shocks cause more vol than positive).
   Student-t innovations handle fat tails.

   **Report these key metrics**:
   - **Persistence** (alpha + beta + 0.5*gamma): How long vol shocks last.
     > 0.95 = highly persistent (vol regime changes are sticky).
     > 0.99 = near integrated (almost IGARCH — vol shocks are permanent).
   - **Half-life**: = log(0.5) / log(persistence). Days for a vol shock to decay 50%.
     10-30 days is typical. > 60 days = vol is very sticky.
   - **Unconditional volatility**: Long-run average vol (annualized).
   - **Gamma** (leverage coefficient): > 0 confirms leverage effect.
     Typical range: 0.05-0.15 for equities.
   - **Current conditional vol** vs unconditional: Is vol elevated or compressed right now?
     Ratio > 1.5 = vol stress. Ratio < 0.7 = vol compression (breakout may be coming).

7. **Model comparison** (if time permits): Also fit GARCH(1,1) and EGARCH.
   Compare AIC/BIC. GJR usually wins for equities, EGARCH for FX.
   If GARCH(1,1) wins (gamma ~ 0), leverage effect is weak for this stock.

8. **News impact curve**: Compute news_impact_curve from the fitted GARCH.
   This shows how positive vs negative shocks of the same magnitude affect volatility.
   - Symmetric curve = no leverage effect (unusual for equities).
   - Asymmetric with steeper left side = leverage effect (negative news causes more vol).
   - The steepness ratio (left/right slope) quantifies the asymmetry.

   **If GARCH fails to converge**: Try with model="GARCH" (simpler). If still failing,
   the series may be too short (< 250 obs) or have structural breaks. Try a shorter window.

---

## Phase 4: Regime Detection (regimes/ module)

9. **HMM regime detection**: detect_regimes on returns_{ticker.lower()} with method="hmm",
   n_regimes=2. This identifies bull (low-vol) and bear (high-vol) market states.

   **Key outputs to report**:
   - **Current regime**: 0 (low-vol/bull) or 1 (high-vol/bear). By convention,
     regimes are sorted by ascending variance, so state 0 = calm.
   - **Current regime probability**: How confident is the model? > 0.8 = high confidence.
     0.5-0.8 = uncertain / possibly transitioning. < 0.5 = likely in the other regime.
   - **Transition matrix**: [[p_00, p_01], [p_10, p_11]].
     p_00 = probability bull stays bull. p_11 = probability bear stays bear.
     Expected duration of bull = 1/(1-p_00). Expected duration of bear = 1/(1-p_11).
   - **Per-regime statistics**: Mean return, volatility, Sharpe, max drawdown in EACH regime.
     Bull regime Sharpe should be much higher. Bear regime usually has negative Sharpe.

10. **Regime history**: How many regime switches in the sample? Dates of the last 3-5 switches.
    Did they align with known market events (COVID, rate hikes, earnings)?
    Is the current regime young (just switched) or mature (been here a while)?
    Mature regimes have higher probability of continuing. Young regimes are uncertain.

   **If HMM fails**: Try n_regimes=3 if the data is long enough (1000+ obs).
   Or try method="gmm" which ignores temporal structure (faster, less powerful).

---

## Phase 5: Technical Analysis (ta/ module — 265 indicators available)

11. **Momentum signals**: compute_indicator for:
    - **RSI(14)**: Current value. > 70 = overbought (potential sell), < 30 = oversold (potential buy).
      Between 40-60 = neutral. RSI divergence (price makes new high but RSI doesn't) = bearish warning.
    - **MACD(12, 26, 9)**: Current MACD line vs signal line. MACD > signal = bullish.
      MACD histogram: positive and increasing = strengthening momentum.
      Histogram just crossed zero = fresh signal.
    - **ROC(20)**: 20-day rate of change. Positive = positive momentum. Compare to 63-day ROC
      for short vs medium-term momentum alignment. Both positive = trend confirmed.

12. **Trend and volatility indicators**: compute_indicator for:
    - **Bollinger Bands(20, 2)**: Where is price relative to bands?
      At upper band = extended. At lower band = potential support.
      **Bandwidth** (upper - lower) / middle: Squeeze (low bandwidth) = low vol, breakout imminent.
      Typical bandwidth for this stock? Current vs average.
    - **ADX(14)**: > 25 = strong trend (trend-following strategies work). < 20 = no trend
      (mean-reversion strategies work). 20-25 = developing.
    - **Supertrend**: Current direction (above or below price). Flip = trend reversal signal.

13. **Volume analysis** (if volume data available): compute_indicator for:
    - **OBV**: On-Balance Volume trend. Rising OBV + rising price = confirmed uptrend.
      Rising price + falling OBV = bearish divergence (distribution).
    - **CMF(20)**: Chaikin Money Flow. Positive = buying pressure, negative = selling pressure.

   **Signal synthesis**: Count bullish vs bearish signals. 5+ aligned = strong signal.
   Mixed signals = uncertain — wait for confirmation.

---

## Phase 6: Risk Assessment (risk/ module)

14. **Core risk metrics**: risk_metrics on returns_{ticker.lower()}.
    Report: Sharpe, Sortino, Calmar, max drawdown (depth, dates, recovery),
    hit ratio, tail ratio, skewness, kurtosis.

    **Risk scorecard**:
    - Sharpe > 1.0: Excellent risk-adjusted return
    - Sharpe 0.5-1.0: Good
    - Sharpe < 0.5: Poor (might not be worth the risk)
    - Max drawdown > 30%: Very high single-stock risk
    - Sortino > Sharpe: Positive skew (upside vol > downside) — desirable
    - Sortino < Sharpe: Negative skew (more downside surprises) — concerning

15. **Value-at-Risk**: var_analysis at 95% and 99% confidence levels.
    Report both historical and Cornish-Fisher VaR (CF adjusts for skew/kurtosis).
    - If CF VaR is much worse than historical: tails are fatter than history shows.
    - 99% VaR: the "worst 1-in-100 day" loss. For single stocks, typically 3-6%.

16. **Stress testing**: stress_test with at least GFC 2008, COVID 2020, and vol_spike scenarios.
    Report expected loss under each scenario. Which scenario is worst for {ticker}?
    - Sector matters: tech stocks worst in dot-com, financials worst in GFC.
    - Cyclical stocks worst in COVID. Defensive stocks more resilient.

17. **Drawdown analysis**: crisis_drawdowns — top 5 worst drawdowns.
    For the current drawdown (if any): how deep, how long, is it recovering?
    Compare current drawdown to historical average and worst.

---

## Phase 7: Synthesis & Actionable Assessment

18. **Cross-module synthesis** — combine findings into a coherent view:

    **Regime + Volatility**: Is the stock in a low-vol regime with compressed GARCH vol?
    That is a bullish setup (calm markets tend to drift up). Is it in high-vol regime
    with elevated GARCH vol? That is a cautious setup (risk of further decline).

    **Technical + Regime**: Do TA signals align with the regime? Bullish TA in bull regime
    = high conviction. Bullish TA in bear regime = potential bear market rally (lower conviction).

    **Risk + Volatility**: Is current vol > unconditional vol? Risk of further vol expansion.
    Is max drawdown still recovering? Position sizing should be smaller during drawdown recovery.

    **Overall assessment**:
    - Current regime and confidence level
    - Risk-adjusted return quality (Sharpe-based)
    - Volatility state (compressed / normal / elevated)
    - Technical signal direction and strength
    - Key risk (what could go wrong from here?)
    - One-sentence conclusion: favorable/neutral/unfavorable for new positions

**Related prompts**: Use volatility_deep_dive for deeper vol analysis, risk_report for
portfolio-level risk, regime_detection for multi-method regime comparison,
momentum_strategy to build a trading strategy from these signals.
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
