"""Trading strategy prompt templates."""
from __future__ import annotations
from typing import Any


def register_strategy_prompts(mcp: Any) -> None:

    @mcp.prompt()
    def pairs_trading(ticker_a: str = "GLD", ticker_b: str = "GDX") -> list[dict]:
        """Pairs trading: cointegration, spread, signals, backtest."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Pairs trading analysis for {ticker_a} vs {ticker_b}:

1. Load price data for both.
2. cointegration_test — are they cointegrated? What's the p-value?
3. Compute hedge ratio and spread.
4. stationarity_test on the spread — must be stationary.
5. Compute half-life of mean reversion.
6. Generate z-score signals: enter at |z| > 2, exit at |z| < 0.5.
7. run_backtest with the signals.
8. backtest_metrics — Sharpe? Max drawdown? Win rate?
9. detect_regimes on the spread — does mean reversion break in certain regimes?
10. Summary: viable pair? Expected return/risk?
"""}}]

    @mcp.prompt()
    def momentum_strategy(dataset: str = "prices") -> list[dict]:
        """Momentum strategy: signals, regime filter, backtest."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Momentum strategy on {dataset}:

1. compute_indicator: RSI(14), MACD(12,26,9), ROC(20).
2. Signal: go long when RSI < 70 AND MACD histogram > 0 AND ROC > 0.
3. detect_regimes — only trade in bull regime (regime 0).
4. Position sizing: volatility_target at 15% annualized.
5. run_backtest with regime-filtered signals.
6. Compare to buy-and-hold.
7. walk_forward — does it work out-of-sample?
8. Summary: Sharpe improvement? Drawdown reduction? Regime filtering help?
"""}}]

    @mcp.prompt()
    def mean_reversion(dataset: str = "prices") -> list[dict]:
        """Mean reversion strategy: stationarity, OU fit, signals, backtest."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Mean reversion strategy on {dataset}:

1. compute_returns.
2. stationarity_test — is the series mean-reverting?
3. Fit Ornstein-Uhlenbeck: estimate theta (speed), mu (mean), sigma.
4. Half-life of reversion — how fast does it revert?
5. compute_indicator: Bollinger Bands(20, 2) for entry/exit.
6. Signal: buy at lower band, sell at upper band.
7. run_backtest with the signals.
8. detect_regimes — does mean reversion work in all regimes?
9. Summary: is mean reversion present? Profitable? Regime-dependent?
"""}}]

    @mcp.prompt()
    def trend_following(dataset: str = "prices") -> list[dict]:
        """Trend following: MA crossover, ADX filter, PSAR stops."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Trend following strategy on {dataset}:

1. compute_indicator: SMA(50), SMA(200) — golden/death cross.
2. compute_indicator: ADX(14) — trend strength filter (only trade when ADX > 25).
3. compute_indicator: PSAR — trailing stop levels.
4. Signal: long when SMA50 > SMA200 AND ADX > 25, stop at PSAR.
5. Position sizing: risk_parity or volatility_target.
6. run_backtest.
7. Compare to buy-and-hold.
8. Regime analysis — does trend following work better in trending regimes?
9. Summary: captures trends? Avoids whipsaws? Drawdown profile?
"""}}]

    @mcp.prompt()
    def statistical_arbitrage(dataset: str = "universe_returns") -> list[dict]:
        """Stat arb: PCA factors, residual alpha, signals, capacity."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Statistical arbitrage on {dataset}:

1. factor_analysis — extract PCA factors from the universe.
2. Compute residuals (alpha) for each asset.
3. stationarity_test on residuals — must be stationary for stat arb.
4. Z-score the residuals — trade when |z| > 2.
5. Build long-short portfolio from extreme z-scores.
6. run_backtest — does the residual alpha persist?
7. Estimate capacity: how much capital before impact kills the alpha?
8. walk_forward — is it robust out-of-sample?
9. Summary: alpha significant? Capacity adequate? Transaction costs?
"""}}]

    @mcp.prompt()
    def carry_trade(rates_json: str = '{"USD":5.25,"EUR":4.50,"JPY":0.10,"AUD":4.35,"GBP":5.25}') -> list[dict]:
        """FX carry trade: portfolio construction and backtest."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
FX carry trade analysis with rates: {rates_json}:

1. **Data**: Parse the rates JSON. Load FX spot rate data from workspace for each
   currency pair (vs USD). Check workspace_status for available FX series.
   compute_returns on each FX pair.
2. **Carry calculation**: For each currency pair, compute carry = foreign_rate - domestic_rate.
   Positive carry = you earn by holding that currency (funded by domestic).
   Rank currencies by carry: highest carry = long candidates, lowest = short/funding.
3. **Portfolio construction**: Classic carry portfolio:
   - Long top 2 highest-carry currencies (equal weight)
   - Short bottom 2 lowest-carry currencies (equal weight)
   This is a zero-cost (self-funding) portfolio.
4. **Historical carry return decomposition**: Total return = carry (interest differential) +
   spot return (FX appreciation/depreciation). For each currency, what fraction of
   return came from carry vs spot? Carry should dominate in calm markets.
5. **Crash risk analysis**: Carry trades are "picking up pennies in front of a steamroller."
   Compute carry portfolio return distribution — expect negative skew, fat left tail.
   distribution_fit with skewed-t. Compute max drawdown and worst monthly return.
6. **Regime analysis**: detect_regimes on the carry portfolio returns.
   Carry works in risk-on regime, crashes in risk-off. Per-regime Sharpe?
   Duration of crash regime? Recovery time?
7. **VIX/vol hedging**: Compute correlation between carry returns and VIX changes.
   When VIX spikes, carry crashes (risk reversal). Can VIX be used as a hedge?
   Estimate hedge ratio: regress carry returns on VIX changes.
8. **Backtest**: run_backtest with the carry signals. Compare to individual currency
   carry. backtest_metrics — Sharpe, Sortino, max drawdown, hit rate.
9. **Summary**: Current carry ranking. Expected annual carry (in bps).
   Crash risk profile. Regime dependence. Recommended position sizing.
"""}}]

    @mcp.prompt()
    def volatility_selling(dataset: str = "prices") -> list[dict]:
        """Short volatility strategy: put selling, straddle selling, and risk management."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Volatility selling strategy on {dataset}:

1. **Data**: Load {dataset} from workspace. compute_returns.
   If options data available, load implied volatilities. Otherwise, use GARCH conditional vol.
2. **Variance risk premium (VRP)**: Compute VRP = implied_vol² - realized_vol².
   The VRP is positive on average — option sellers earn a premium for bearing crash risk.
   fit_garch to get conditional RV forecast. If IV available, compute VRP time series.
   Average VRP? Current VRP relative to history?
3. **Strategy 1 — Cash-secured put selling**: Simulate selling 1-month ATM puts.
   If no options data: estimate put premium from Black-Scholes using GARCH vol.
   P&L: collect premium if index stays above strike, lose if it drops.
   Monthly return = premium / notional. Hit rate? Average win vs average loss?
4. **Strategy 2 — Short straddle**: Sell ATM call + ATM put. Profit when realized vol
   < implied vol (the VRP). Loss when realized move exceeds premium collected.
   Compute P&L as: premium - |realized_move|. What is the breakeven move?
5. **Tail risk management**: Short vol strategies have unlimited downside.
   Compute worst drawdowns. stress_test with GFC, COVID scenarios.
   How much would the strategy lose in a vol spike?
6. **Delta hedging**: If continuous hedging available, simulate delta-hedged short vol.
   P&L = implied_vol² - realized_vol² (the pure VRP capture).
   Hedging reduces directional risk but not vol-of-vol risk.
7. **Regime filter**: detect_regimes. Only sell vol in low-vol regime?
   In high-vol regime, VRP is largest BUT crash risk is also highest.
   Test: sell vol in all regimes vs only in low/medium vol regimes.
8. **Position sizing**: Kelly criterion for the strategy. Given the negative skew,
   use fractional Kelly (0.25x) to survive tail events.
   What is the max acceptable allocation to short vol?
9. **Backtest**: run_backtest for each strategy variant (put selling, straddle, regime-filtered).
   backtest_metrics — Sharpe, Sortino, max drawdown, worst month.
10. **Summary**: VRP magnitude. Strategy comparison. Regime filtering value.
    Tail risk profile. Recommended approach and sizing.
"""}}]

    @mcp.prompt()
    def market_making(dataset: str = "tick_data") -> list[dict]:
        """Market making strategy: spread capture, inventory management, adverse selection."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Market making analysis on {dataset}:

1. **Data**: Load {dataset} from workspace. Needs high-frequency or daily data with
   bid-ask information (or proxied). compute_returns at appropriate frequency.
2. **Spread estimation**: Compute quoted spread (ask - bid) and effective spread
   (2 × |trade_price - midpoint|). If no bid-ask data, use Roll's estimator:
   effective_spread ≈ 2 × sqrt(-cov(Δp_t, Δp_{t-1})). Average spread in bps?
3. **Gross revenue estimate**: A market maker earns ~0.5 × spread per round trip.
   Estimate daily gross revenue = 0.5 × spread × daily_volume × participation_rate.
   Assume 5% participation rate. Is the gross revenue meaningful?
4. **Adverse selection cost**: Estimate using the realized spread:
   realized_spread = 2 × sign × (trade_price - midpoint_{t+5min}).
   If realized spread < effective spread, the difference is adverse selection cost.
   What fraction of the spread is lost to informed traders?
5. **Inventory risk**: Simulate inventory accumulation with a symmetric quoting strategy.
   Compute max inventory position and inventory holding period.
   fit_garch to estimate conditional vol — inventory risk = inventory × σ_conditional.
   What's the worst-case inventory loss?
6. **Optimal quoting (Avellaneda-Stoikov)**: Compute reservation price adjustment:
   δ = γ × σ² × (T - t) × q, where q = inventory, γ = risk aversion.
   As inventory grows, skew quotes to reduce position.
   Optimal spread = γσ²(T-t) + (2/γ)ln(1 + γ/κ) where κ = order arrival intensity.
7. **Toxicity monitoring**: Compute VPIN (Volume-Synchronized PIN) — probability of
   informed trading. High VPIN = widen spreads or stop quoting.
   Threshold: VPIN > 0.8 = toxic flow, step back.
8. **P&L simulation**: Simulate market making P&L over the dataset period:
   Gross P&L = spread capture - adverse selection - inventory losses.
   Net after costs. Sharpe of the market making strategy?
9. **Summary**: Average spread captured. Adverse selection fraction.
   Inventory risk quantification. VPIN toxicity assessment.
   Net profitability estimate. Recommended spread width and participation.
"""}}]

    @mcp.prompt()
    def sector_rotation(sectors_dataset: str = "sector_returns") -> list[dict]:
        """Sector rotation: momentum + fundamentals-based sector selection."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Sector rotation strategy on {sectors_dataset}:

1. **Data**: Load {sectors_dataset} from workspace. Should include returns for major sectors
   (XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLRE, XLC, XLB).
   compute_returns if raw prices.
2. **Momentum ranking**: For each sector, compute:
   - 1-month return (short-term momentum)
   - 3-month return (medium-term momentum)
   - 12-month return skipping last month (12-1 momentum, avoids reversal)
   Composite momentum = 0.25 × 1M + 0.50 × 3M + 0.25 × 12-1M.
   Rank sectors by composite momentum.
3. **Relative strength**: Compute each sector's return relative to SPY (equal-weighted index).
   Sectors with positive relative strength are outperforming. Track RS trend (rising/falling).
   Use compute_indicator with RSI(14) on relative strength — RS momentum.
4. **Regime overlay**: detect_regimes on SPY or macro data.
   Sector behavior is regime-dependent:
   - Risk-on: overweight cyclicals (XLK, XLI, XLY, XLF)
   - Risk-off: overweight defensives (XLU, XLP, XLV, XLRE)
   - Rate-rising: underweight rate-sensitive (XLRE, XLU)
5. **Fundamental overlay**: If fundamental data available (earnings growth, valuations):
   - Rank sectors by forward earnings growth (prefer accelerating)
   - Rank by relative valuation (prefer cheap vs history)
   - Composite = 0.5 × momentum + 0.3 × fundamentals + 0.2 × regime
6. **Portfolio construction**: Select top 3-4 sectors. Weight by inverse volatility
   (risk parity within selected sectors). Rebalance monthly.
7. **Backtest**: run_backtest with monthly rebalancing. Compare to equal-weight sectors
   and SPY buy-and-hold. backtest_metrics for all three.
   walk_forward — does sector rotation add value out-of-sample?
8. **Turnover analysis**: How many sectors change each month? Average monthly turnover?
   Estimate transaction cost drag. Is the alpha net of costs?
9. **Summary**: Current sector ranking. Top picks and rationale (momentum + regime + fundamentals).
   Backtest results. Turnover and cost. Recommended sector allocation.
"""}}]
