"""Trading strategy prompt templates."""
from __future__ import annotations
from typing import Any


def register_strategy_prompts(mcp: Any) -> None:

    @mcp.prompt()
    def pairs_trading(ticker_a: str = "GLD", ticker_b: str = "GDX") -> list[dict]:
        """Pairs trading: cointegration, spread, signals, backtest."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
First load the wraquant_system_context prompt for full module context.

Comprehensive pairs trading analysis for {ticker_a} vs {ticker_b}. This is a full
statistical arbitrage workflow using stats/ (cointegration, stationarity), regimes/
(regime-conditional behavior), backtest/ (walk-forward validation), and risk/ (position
sizing and drawdown management). The goal is to determine whether this pair is tradeable,
build the trading signals, and validate with a rigorous out-of-sample backtest.

---

## Phase 1: Data Acquisition & Preliminary Check

1. **Workspace check**: Run workspace_status. Look for prices_{ticker_a.lower()} and
   prices_{ticker_b.lower()}. If not present, note that price data needs loading
   via OpenBB MCP or store_data.

2. **Data requirements**: Both assets need at least 3 years (750+ trading days) of history.
   Cointegration tests need long samples for reliability. 5+ years is ideal.
   Both must be price series (not returns) for cointegration analysis.
   Verify dates are aligned (same trading calendar). If one has gaps, inner-join on dates.

3. **Preliminary correlation**: compute_returns on both assets. Then correlation_analysis.
   - Correlation > 0.7: Good candidate. Assets move together (economically linked).
   - Correlation 0.3-0.7: Moderate. Might work if fundamentally related.
   - Correlation < 0.3: Unlikely to be cointegrated. Reconsider the pair.

   **Why correlation alone is insufficient**: High correlation does NOT imply cointegration.
   Two assets can have 0.95 correlation but diverge permanently (not cointegrated).
   Cointegration means the spread is stationary — it MUST mean-revert. Correlation
   only says they move in similar directions, not that the spread reverts.

4. **Economic rationale**: Before statistical tests, ask: WHY should this pair be cointegrated?
   - {ticker_a} and {ticker_b}: What is the economic link? Same sector? Input/output relationship?
     Gold (GLD) and gold miners (GDX) are cointegrated because miner revenue = f(gold price).
   - Without economic rationale, statistical cointegration may be spurious and will break.

---

## Phase 2: Cointegration Testing (stats/ module)

5. **Engle-Granger test**: cointegration_test on the two price series.
   This runs OLS: price_A = alpha + beta * price_B + residual,
   then tests if the residual is stationary (ADF test on residuals).

   **Interpreting results**:
   - **p-value < 0.05**: Cointegrated at 5% significance. Proceed with pairs trading.
   - **p-value 0.05-0.10**: Marginal. The pair might be cointegrated but evidence is weak.
     Consider using a longer sample or Johansen test for confirmation.
   - **p-value > 0.10**: NOT cointegrated. The spread may not mean-revert. STOP HERE
     unless the economic rationale is very strong (maybe the relationship broke recently).

   - **Hedge ratio (beta)**: Units of {ticker_b} to short per unit of {ticker_a} long.
     E.g., beta = 0.5 means short 0.5 shares of {ticker_b} for each share of {ticker_a}.
     This should be roughly stable over time. If it drifts a lot, dynamic hedging is needed.

6. **Johansen test** (more powerful): If Engle-Granger is marginal (p-value 0.05-0.15),
   run Johansen cointegration test. Johansen tests for cointegrating rank (0, 1, or 2).
   - Rank 0: No cointegration (reject the pair).
   - Rank 1: One cointegrating vector (the pair is cointegrated — proceed).
   - Rank 2: Both series are stationary individually (trivial case — just trade each alone).

7. **Rolling cointegration stability**: The critical question — is cointegration STABLE?
   Run cointegration_test on rolling 2-year windows (step forward 63 days at a time).
   Track the p-value over time. If p-value frequently exceeds 0.10, the cointegration
   relationship breaks periodically. This is dangerous — the spread can diverge permanently
   during those periods, causing large losses.

   **If cointegration is unstable**: This pair requires careful regime monitoring.
   You must exit the trade when cointegration weakens (see Phase 4).

---

## Phase 3: Spread Construction & Properties

8. **Spread construction**: Compute the spread: spread_t = price_A_t - beta * price_B_t.
   Store as a dataset. This is the tradeable signal.

9. **Stationarity confirmation**: stationarity_test (ADF) on the spread.
   - ADF p-value < 0.01: Strongly stationary. Excellent for pairs trading.
   - ADF p-value 0.01-0.05: Stationary. Good.
   - ADF p-value > 0.05: NOT stationary. Contradicts cointegration result. Investigate.
     (This can happen with short samples or structural breaks in the relationship.)

10. **Half-life of mean reversion**: Fit an AR(1) to the spread:
    spread_t = c + phi * spread_{t-1} + epsilon.
    Half-life = -log(2) / log(phi) trading days.

    **Interpretation**:
    - Half-life < 10 days: Very fast reversion. Aggressive entry/exit. High-frequency pair.
    - Half-life 10-30 days: Moderate. Standard pairs trading window. Monthly signals.
    - Half-life 30-60 days: Slow. Need patience. Wider stops. Less capital-efficient.
    - Half-life > 60 days: Too slow. Transaction costs and carry costs eat the profit.
      Reconsider unless the spread is very wide.

    **If half-life > 60**: The pair mean-reverts but too slowly. Consider:
    - Using a different lookback window for z-score calculation.
    - Trading only when z-score > 3 (wider entry, bigger expected profit per trade).
    - Or dropping this pair and finding a faster-reverting alternative.

11. **Spread distribution**: analyze() on the spread. Report mean, std, skewness, kurtosis.
    - Is the spread distribution symmetric? Skewed spreads may have directional bias.
    - Fat tails in the spread = risk of extreme divergence (large losses on the trade).
    - distribution_fit with Student-t — how fat are the spread tails?

---

## Phase 4: Signal Generation

12. **Z-score computation**: Compute rolling z-score of the spread.
    z_t = (spread_t - rolling_mean) / rolling_std.
    Use a lookback window of 2x the half-life (e.g., half-life 15 days -> lookback 30 days).

    **Signal rules** (standard):
    - **Enter long spread** (long A, short B) when z < -2.0 (spread is cheap).
    - **Enter short spread** (short A, long B) when z > +2.0 (spread is rich).
    - **Exit** when |z| < 0.5 (spread has reverted to near the mean).
    - **Stop loss** at |z| > 4.0 (spread has diverged too far — cointegration may have broken).

    **Signal tuning considerations**:
    - Entry threshold 2.0 vs 1.5 vs 2.5: Lower threshold = more trades, lower profit per trade.
      Higher threshold = fewer trades, higher profit per trade but may miss opportunities.
    - Exit threshold 0.5 vs 0.0 vs 1.0: Exit at 0 (mean) captures full reversion but risks
      whipsaws. Exit at 0.5 is more conservative. Exit at 1.0 leaves money on the table.
    - Stop loss 4.0 is a safety net. Triggers rarely but limits catastrophic losses.

13. **Dynamic hedge ratio**: The hedge ratio may drift over time. Consider using a rolling
    OLS or Kalman filter to estimate the hedge ratio dynamically.
    - Rolling OLS: Re-estimate beta every 63 days using trailing 252 days.
    - Kalman filter: Continuously updates beta. Better for non-stationary hedge ratios.
    If the hedge ratio drifts > 20% from its mean, the spread definition is changing.
    More frequent rebalancing is needed.

---

## Phase 5: Regime Analysis

14. **Regime detection on the spread**: detect_regimes on the spread series with method="hmm",
    n_regimes=2. This identifies periods where mean reversion works vs breaks.

    **Expected regimes**:
    - Regime 0 (low-vol): Tight spread, fast mean reversion. Pairs trading works well.
      Expected Sharpe in this regime should be high.
    - Regime 1 (high-vol): Wide, volatile spread. Mean reversion may be slow or absent.
      Expected Sharpe in this regime is lower, possibly negative.

    **Current regime**: Which regime is the spread in right now?
    If Regime 1 (stressed): Consider reducing position size or pausing the strategy.

15. **Regime-conditional half-life**: Compute half-life separately in each regime.
    If half-life is much longer in the stressed regime (> 2x normal), mean reversion
    slows dramatically under stress. This is the main risk of pairs trading —
    the trade works until it doesn't, and failures cluster in volatile markets.

16. **Market regime overlay**: detect_regimes on SPY (or broad market).
    Does the spread's behavior change with the market regime?
    Many pairs strategies break during market crises (all correlations spike,
    spread diverges). Test: in market bear regime, does the pairs strategy
    still have positive Sharpe? If not, add a regime filter — only trade in bull regime.

---

## Phase 6: Backtesting & Performance

17. **Walk-forward backtest**: run_backtest with the z-score signals.
    CRITICAL: Use walk-forward validation, NOT in-sample backtest.
    - Estimation window: 252 days (estimate mean, std, hedge ratio).
    - Trading window: 63 days (trade using estimated parameters).
    - Step forward 63 days, re-estimate, repeat.
    This prevents look-ahead bias. In-sample pairs trading results are ALWAYS
    misleading because the parameters were fitted to the same data.

18. **Backtest metrics**: backtest_metrics on the walk-forward results.
    Report:
    | Metric | Value | Interpretation |
    |--------|-------|----------------|
    | Annualized Return | X% | Net of transaction costs? |
    | Annualized Volatility | X% | Should be low for pairs (market-neutral) |
    | Sharpe Ratio | X.XX | > 1.0 for a pairs strategy = very good |
    | Max Drawdown | -X% | For pairs, > 15% drawdown = concerning |
    | Win Rate | X% | > 60% typical for mean reversion |
    | Profit Factor | X.XX | Gross profit / gross loss. > 1.5 is good |
    | Avg Trade Duration | X days | Should be close to half-life |
    | Number of Trades | X | Enough for statistical significance? (> 30 minimum) |
    | Avg Win / Avg Loss | X.XX | > 1.0 = wins are bigger than losses |

19. **Transaction cost sensitivity**: Pairs trading involves 4 legs (buy A, sell B, then
    reverse). Estimate round-trip cost (spread + commission) for both assets.
    Re-run backtest with costs of 5bps, 10bps, 20bps per leg.
    At what cost level does the strategy become unprofitable?
    If the break-even cost is < 10bps, the strategy may not survive real-world execution.

20. **Capacity estimation**: What is the maximum capital this strategy can deploy?
    Capacity = min(ADV_A, ADV_B) * 0.05 * avg_trade_duration / avg_number_of_concurrent_trades.
    If capacity < $1M, the strategy may not be worth the operational cost.

---

## Phase 7: Risk Management & Position Sizing

21. **Position sizing**: Size the position based on the spread's volatility.
    - Target vol approach: position_size = target_portfolio_vol / spread_vol * capital.
    - Example: if target = 10% annual vol and spread vol = 20%, use 50% of capital.
    - Kelly criterion: f* = Sharpe^2 / (spread_vol * sqrt(252)). Use half-Kelly for safety.

22. **Stop loss calibration**: The z = 4 stop loss should be backtested.
    What fraction of trades would have been stopped out? What is the average loss
    on stopped trades? If stops are hit > 5% of the time, the spread is too volatile
    or the cointegration is unreliable.

23. **Maximum concurrent exposure**: If running multiple pairs, total pairs exposure
    should be limited. Pairs strategies are correlated in crises (all spreads diverge
    simultaneously). Set max pairs exposure to 2x single-pair sizing.

---

## Phase 8: Final Assessment

24. **Viability scorecard**:

    | Criterion | Result | Pass/Fail |
    |-----------|--------|-----------|
    | Cointegration p-value < 0.05 | X | PASS/FAIL |
    | Spread ADF stationary | p = X | PASS/FAIL |
    | Half-life < 60 days | X days | PASS/FAIL |
    | Walk-forward Sharpe > 0.5 | X.XX | PASS/FAIL |
    | Profitable after 10bps costs | X% return | PASS/FAIL |
    | Spread regime: currently normal | Regime X | PASS/FAIL |
    | > 30 trades in backtest | X trades | PASS/FAIL |
    | Max drawdown < 15% | -X% | PASS/FAIL |

    **Decision**: If 6+ criteria pass, the pair is viable. 4-5 pass = borderline, proceed
    with caution and smaller sizing. < 4 pass = reject the pair.

25. **Summary**:
    - Is the pair cointegrated? How stable is the relationship?
    - Current spread z-score: is there a trade right now?
    - Expected annual return (walk-forward) and max drawdown
    - Current spread regime: favorable or stressed?
    - Key risk: what could break the cointegration? (Structural change, M&A, sector divergence)
    - One-sentence verdict: trade / monitor / reject

**Related prompts**: Use statistical_arbitrage for a multi-asset PCA-based extension,
mean_reversion for single-asset mean reversion, regime_detection for deeper regime analysis,
cointegration analysis in equity_deep_dive for individual stock assessment.
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
