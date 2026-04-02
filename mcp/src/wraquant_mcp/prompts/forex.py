"""Forex-specific prompt templates."""

from __future__ import annotations

from typing import Any


def register_forex_prompts(mcp: Any) -> None:

    @mcp.prompt()
    def fx_pairs_analysis(
        pair: str = "EURUSD",
        dataset: str = "fx_eurusd",
    ) -> list[dict]:
        """Analyze currency pair dynamics: trend, vol, sessions, microstructure."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Perform a comprehensive analysis of the {pair} currency pair using {dataset}.
This workflow uses forex/ tools alongside stats/, vol/, and ta/ for a complete
FX-specific assessment.

---

## Phase 1: Pair Characterization

1. **Data check**: Run workspace_status to verify {dataset} exists with at least
   1 year of data. FX pairs need 24-hour coverage for session analysis.
   Run compute_returns on {dataset}.

2. **Statistical profile**: Run analyze() on returns_{dataset}.
   FX-specific interpretation:
   - Mean daily return near 0 is normal for major pairs (no risk premium).
   - Annualized vol: 5-8% for majors (EURUSD, USDJPY), 10-15% for EM pairs.
   - Skewness: Negative for carry-funded pairs (crash risk). Positive for funding currencies.
   - Kurtosis: FX typically has excess kurtosis of 2-5 (fat tails from interventions, macro shocks).
   - ADF test: FX returns should be stationary. Levels may show unit root (random walk).

3. **Distribution fit**: Run distribution_fit on returns_{dataset}.
   FX returns are typically better fit by Student-t or skewed-t than normal.
   The degrees of freedom parameter quantifies tail risk -- lower = fatter tails.

---

## Phase 2: Volatility Dynamics

4. **GARCH modeling**: Run fit_garch on returns_{dataset} with model="EGARCH", dist="t".
   EGARCH often fits FX better than GJR because the leverage effect in FX is different
   from equities (both directions of shocks increase vol equally).

   **FX-specific interpretation**:
   - Persistence > 0.98: Very persistent. Vol regimes in FX last weeks/months.
   - If EGARCH gamma is near 0: symmetric response (typical for majors).
   - If gamma < 0: depreciation of base causes more vol (EM currency pattern).
   - Current conditional vol vs unconditional: Is vol elevated (risk event)
     or compressed (quiet period -- breakout imminent)?

5. **Realized vol**: Run risk_metrics on returns_{dataset}.
   Report annualized vol. Compare to implied vol if available.
   RV << IV = "vol risk premium" -- selling options is profitable on average.

---

## Phase 3: Session Analysis

6. **Current session**: Run session_info to identify active trading sessions.
   - **Tokyo session** (00:00-09:00 UTC): JPY and AUD most active. Low vol for EUR/USD.
   - **London session** (08:00-17:00 UTC): Highest liquidity. EUR, GBP most active.
   - **NY session** (13:00-22:00 UTC): USD-centric. Key US data releases.
   - **London-NY overlap** (13:00-17:00 UTC): Best liquidity, tightest spreads.

7. **Session-conditional statistics**: If intraday data available, compute
   returns and volatility for each session separately.
   - Which session has the highest vol? (Usually London open or data releases.)
   - Which session has the best Sharpe? (Momentum in Asian, reversion in NY.)
   - Session-specific VPIN via toxicity_analysis on session-filtered data.

---

## Phase 4: Technical Analysis

8. **Trend assessment**: Run compute_indicator for:
   - **ADX(14)**: > 25 = trending market (momentum strategies work).
     FX trends can persist for months (carry, macro themes).
   - **MACD(12,26,9)**: Current signal. FX MACD works well on daily/4H.
   - **Bollinger Bands(20,2)**: Squeeze detection. FX squeezes often precede
     major moves (central bank decisions, data releases).

9. **Momentum**: Run compute_indicator for:
   - **RSI(14)**: FX RSI extremes (>75, <25) are less reliable than equities.
     Use divergence instead: price makes new high, RSI doesn't = bearish.
   - **ROC(20)**: Rate of change. Compare to ROC(60) for trend alignment.

10. **Support/Resistance**: Key psychological levels for FX pairs (round numbers:
    1.0000, 1.1000, etc.). These act as strong S/R in FX markets.
    Run compute_indicator for pivot points.

---

## Phase 5: Cross-Rate & Regime Analysis

11. **Cross-rate implications**: If analyzing EURUSD, consider the triangular
    relationship with EURGBP and GBPUSD. Use cross_rate to verify consistency.
    Misalignment = potential cross-rate arbitrage opportunity.

12. **Regime detection**: Run detect_regimes on returns_{dataset} with n_regimes=2.
    FX regimes often correspond to:
    - Regime 0 (low vol): Range-bound market. Mean-reversion strategies work.
    - Regime 1 (high vol): Trending or crisis. Momentum/carry strategies work.
    Current regime and probability?

13. **Pip calculator**: Run pip_calculator for a hypothetical trade to understand
    the P&L per pip for this pair at standard lot sizes.

---

## Phase 6: Synthesis

14. **FX pair report**:
    - Current regime (trending/ranging) and confidence.
    - Volatility state (compressed/normal/elevated) vs historical.
    - Active session and liquidity conditions.
    - Technical signal alignment (bullish/bearish/neutral).
    - Key risk: central bank, data releases, geopolitical.
    - Recommended strategy type: momentum, mean-reversion, or flat.

**Related prompts**: Use carry_trade_deep_dive for carry analysis,
fx_risk_management for hedging, currency_strength_analysis for relative value.
""",
                },
            }
        ]

    @mcp.prompt()
    def carry_trade_deep_dive(
        rates_json: str = '{"USD": 0.05, "EUR": 0.04, "JPY": 0.001, "AUD": 0.04, "CHF": 0.015}',
    ) -> list[dict]:
        """Carry trade construction, risk analysis, and crash hedging."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Analyze carry trade opportunities and risks using interest rates: {rates_json}.
This workflow uses forex/ tools to construct, risk-assess, and hedge a carry portfolio.

---

## Phase 1: Carry Portfolio Construction

1. **Interest rate differential**: Run carry_analysis with the provided rates.
   This identifies:
   - **Long (high yield)**: Currencies with highest rates. These are "carry receivers."
   - **Short (low yield)**: Currencies with lowest rates. These are "carry funders."
   - The carry return = interest rate differential (annualized).

   **Interpretation**:
   - JPY, CHF are classic funding currencies (low rates).
   - AUD, NZD, BRL are classic carry currencies (high rates).
   - Carry return of 3-5% per year is typical for G10 carry.

2. **Historical carry performance**: If FX price data available, compute
   total return = spot return + carry (interest differential).
   Use compute_returns on each pair and add the daily carry (annual rate / 252).
   - How much of historical return came from carry vs spot appreciation?
   - Carry trades have Sharpe ratios of 0.5-1.0 but very negative skew.

---

## Phase 2: Risk Assessment

3. **Crash risk**: Carry trades are "picking up pennies in front of a steamroller."
   - Run distribution_fit on the carry portfolio returns.
     Expect: negative skew, high kurtosis. The carry trade crashes 3-4 times per decade.
   - Run risk_metrics: Report max drawdown (typically 15-30% for G10 carry),
     Sortino ratio (should be worse than Sharpe due to negative skew).
   - Run var_analysis at 99% level. The tail risk is the real concern.

4. **Correlation analysis**: Run correlation_analysis on the carry pairs.
   During carry unwinds, all high-yield currencies crash together while
   funding currencies rally together. Correlations spike to near 1.0.
   - Normal-period correlation: Should be moderate (0.3-0.6).
   - Crisis correlation: Check conditional correlation during stress periods.

5. **Regime dependence**: Run detect_regimes on the carry portfolio returns.
   - Bull regime: Positive carry earned, low vol. Sharpe > 1.
   - Bear regime: Carry unwind, high vol, negative returns. Sharpe << 0.
   - Expected duration of each regime? Transition probabilities?

---

## Phase 3: Hedging Strategies

6. **VIX/FX vol hedge**: Carry trades are short volatility. Buying vol protection
   (VIX calls, FX options) can hedge the crash risk.
   - Cost: 1-2% per year for tail protection.
   - Benefit: Truncates the left tail. Improves Sortino dramatically.

7. **Momentum filter**: Only take carry when the spot trend aligns.
   Run compute_indicator RSI(14) and MACD on each carry pair.
   - If carry currency is in a downtrend, skip it (trend > carry).
   - This reduces drawdowns by 30-50% with modest return reduction.

8. **Risk parity carry**: Instead of equal-weight carry, weight by inverse
   volatility. Higher vol pairs get lower weight. This equalizes risk contribution.
   Run risk_metrics on each pair to get volatility for weighting.

---

## Phase 4: Monitoring

9. **Current positioning**:
   - What is the total portfolio carry (weighted interest differential)?
   - What is the portfolio volatility? Sharpe of the carry?
   - Are any carry currencies in a downtrend? (Momentum filter warning.)
   - Is VIX elevated? (Risk-off environment = carry unwind risk.)

10. **Summary**:
    - Carry portfolio composition and expected annual carry.
    - Risk profile: vol, max drawdown, skew, tail VaR.
    - Current regime: safe to carry or unwind risk?
    - Hedging cost vs benefit assessment.
    - Recommendation: full carry, reduced carry, or flat.

**Related prompts**: Use fx_risk_management for hedging,
currency_strength_analysis for relative value signals.
""",
                },
            }
        ]

    @mcp.prompt()
    def fx_risk_management(
        positions_json: str = '{"EUR": 100000, "GBP": 50000, "JPY": -75000}',
    ) -> list[dict]:
        """FX hedging, exposure analysis, and risk decomposition."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Analyze and manage FX risk for positions: {positions_json}.
This uses forex/ tools (fx_risk, pip_calculator) alongside risk/ tools.

---

## Phase 1: Exposure Analysis

1. **Portfolio exposure**: Run fx_risk with the positions and current exchange rates.
   Report:
   - Gross exposure (sum of absolute positions) in base currency.
   - Net exposure (sum of signed positions). Net long or short USD?
   - Largest single-currency exposure as % of total.
   - Concentration risk: Is > 50% in one currency? Diversification?

2. **Position sizing**: Run pip_calculator for each position.
   For each currency:
   - Value per pip at the current lot size.
   - 100-pip adverse move = what dollar loss?
   - Is any position outsized relative to the portfolio?

---

## Phase 2: Risk Quantification

3. **Historical VaR**: If FX return data is available:
   - Compute portfolio returns using the position weights.
   - Run risk_metrics on the portfolio returns.
   - Run var_analysis at 95% and 99% confidence.
   - Report: daily VaR, weekly VaR, monthly VaR in base currency terms.

4. **Correlation risk**: Run correlation_analysis on the FX pairs in the portfolio.
   - Are positions in correlated currencies (EUR and GBP both long)?
     Effective diversification is lower than it appears.
   - Are positions in anti-correlated currencies? Good natural hedge.
   - Compute the portfolio correlation matrix and effective number of bets.

5. **Stress testing**: Estimate portfolio loss under scenarios:
   - USD strengthening: +5% broad USD (risk-off). Impact on EUR, GBP, JPY.
   - EM crisis: +10% USDJPY (yen weakens as safe haven unwinds).
   - European crisis: -5% EURUSD, -3% GBPUSD.
   Run stress_test if available, or compute manually from position sizes.

---

## Phase 3: Hedging Strategy

6. **Natural hedges**: Identify offsetting exposures.
   - Long EUR + short GBP = net exposure to EURGBP. Is this intentional?
   - JPY exposure: is it a hedge for equity risk (JPY rallies in risk-off)?

7. **Hedge ratios**: For each unhedged exposure:
   - Minimum variance hedge ratio from correlation_analysis.
   - Cost of hedging: forward points (carry cost) from interest rate differential.
   - Use carry_analysis to compute the cost of FX forwards for each pair.

8. **Partial hedging**: Full hedge eliminates risk but also return.
   Consider hedging 50-75% of each exposure:
   - 50% hedge: Reduces risk by ~50%, preserves upside.
   - 75% hedge: Near-full protection, low residual risk.
   - Tail hedge: Only hedge the extreme left tail (options, cheaper).

---

## Phase 4: Session & Timing Risk

9. **Session exposure**: Run session_info.
   - During which sessions is the portfolio most at risk?
   - Asian session: JPY exposure is most volatile.
   - London session: EUR, GBP exposure peaks.
   - Overnight gap risk: What happens if a major event occurs during a closed session?

10. **Summary**:
    - Total portfolio risk (VaR, vol, max expected loss).
    - Key exposures and concentration.
    - Correlation-adjusted risk (diversification benefit or penalty).
    - Recommended hedges and their costs.
    - Residual risk after hedging.

**Related prompts**: Use fx_pairs_analysis for individual pair analysis,
carry_trade_deep_dive for carry-specific risk.
""",
                },
            }
        ]

    @mcp.prompt()
    def currency_strength_analysis(
        pairs_dataset: str = "fx_major_pairs",
    ) -> list[dict]:
        """Multi-currency relative strength ranking and rotation signals."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Compute and analyze currency strength rankings using {pairs_dataset}.
This uses forex/ tools (currency_strength, carry_analysis) to identify
the strongest and weakest currencies for directional and relative value trades.

---

## Phase 1: Strength Computation

1. **Currency strength scores**: Run currency_strength on {pairs_dataset} with
   window=20 (short-term) and window=60 (medium-term).
   This decomposes multi-pair data into individual currency strength:
   - Score > 0: Currency is strengthening (appreciating vs peers).
   - Score < 0: Currency is weakening (depreciating vs peers).
   - Rank currencies from strongest to weakest.

2. **Short vs medium-term alignment**:
   - Both positive (20d and 60d): Strong trend, high conviction.
   - 20d positive, 60d negative: Short-term bounce in a downtrend. Lower conviction.
   - 20d negative, 60d positive: Pullback in an uptrend. Potential entry point.
   - Both negative: Weak currency. Sell in all pairs against strong currencies.

---

## Phase 2: Pair Selection

3. **Strongest vs weakest**: The best directional trade is:
   Long (strongest currency) / Short (weakest currency).
   This maximizes the carry AND momentum signal.
   - Identify the top trade: which pair combines the strongest vs weakest?
   - Check the trend of this pair: confirm with ADX and MACD.

4. **Cross-rate opportunities**: Run cross_rate for derived pairs.
   Sometimes the best trade is a cross (e.g., EURJPY, GBPCHF) rather than
   a dollar pair. Cross rates can have cleaner trends.

5. **Carry alignment**: Run carry_analysis with current interest rates.
   - Is the strongest currency also a high-carry currency? (Best case: momentum + carry.)
   - Is the strongest currency a low-carry currency? (Momentum vs carry conflict.
     Short-term momentum may override carry, but be cautious.)

---

## Phase 3: Risk & Rotation

6. **Strength stability**: Compute the rolling standard deviation of strength scores.
   Stable strength = reliable signal. Volatile strength = noisy, lower conviction.
   - Stable strong: Trend-following entry.
   - Volatile strong: Wait for pullback or skip.

7. **Rotation detection**: Compare current strength rankings to 1-month-ago rankings.
   - If the ranking is unchanged: established trends continue.
   - If rankings are shifting: rotation underway. New trades forming, old trades fading.
   - Rapid rotation = choppy market. Reduce position sizes.

8. **Regime context**: Run detect_regimes on a broad FX volatility measure.
   - Low-vol regime: Carry and mean-reversion dominate. Strength matters less.
   - High-vol regime: Momentum and strength dominate. Follow the rankings.

---

## Phase 4: Synthesis

9. **Currency strength dashboard**:

    | Currency | 20d Strength | 60d Strength | Rank | Carry (%) | Signal |
    |----------|-------------|-------------|------|-----------|--------|
    | USD | ... | ... | ... | ... | ... |
    | EUR | ... | ... | ... | ... | ... |
    | GBP | ... | ... | ... | ... | ... |
    | JPY | ... | ... | ... | ... | ... |
    | AUD | ... | ... | ... | ... | ... |
    | CHF | ... | ... | ... | ... | ... |

10. **Recommendations**:
    - Top trade: [Strongest]/[Weakest] pair.
    - Carry-aligned trades: Pairs where momentum and carry agree.
    - Rotation alert: Any currencies changing rank rapidly?
    - Overall FX regime: trending or range-bound?

**Related prompts**: Use fx_pairs_analysis for deep-dive on the top pair,
carry_trade_deep_dive for carry portfolio construction.
""",
                },
            }
        ]
