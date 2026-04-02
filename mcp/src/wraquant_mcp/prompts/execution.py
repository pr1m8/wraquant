"""Execution & microstructure prompt templates."""

from __future__ import annotations

from typing import Any


def register_execution_prompts(mcp: Any) -> None:

    @mcp.prompt()
    def optimal_execution(
        ticker: str = "AAPL",
        total_shares: float = 100000,
        urgency: str = "medium",
    ) -> list[dict]:
        """Full execution optimization: schedule, cost model, Almgren-Chriss trajectory."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Design an optimal execution plan for liquidating {total_shares:,.0f} shares of {ticker}
with {urgency} urgency. This workflow uses the execution/ and microstructure/ tools
to construct, cost, and compare execution strategies.

---

## Phase 1: Pre-Trade Intelligence

1. **Market data**: Run workspace_status to check for prices_{ticker.lower()} data.
   We need OHLCV with at least 60 days of history for volume profiling and
   volatility estimation. If unavailable, load via fetch_yahoo or store_data.

2. **Liquidity assessment**: Run liquidity_metrics on prices_{ticker.lower()}.
   Report:
   - **Amihud illiquidity ratio**: Higher = more price impact per dollar traded.
     Compare to typical large-cap levels (< 1e-6 for liquid names).
   - **Kyle's lambda**: Permanent price impact coefficient. Higher = informed trading
     matters more. Flag if > 2x the cross-sectional median.
   - **Roll spread**: Implied effective spread from serial autocovariance.
     Compare to quoted spread if available.

3. **Spread estimation**: Run corwin_schultz_spread on prices_{ticker.lower()}.
   This gives a daily spread estimate from high-low prices. Average this for the
   cost model. If spread > 20 bps, the stock is semi-liquid -- adjust POV down.

4. **Toxicity check**: Run toxicity_analysis on prices_{ticker.lower()}.
   - **VPIN > 0.7**: High probability of informed trading. Slow down execution
     to avoid adverse selection. Reduce urgency.
   - **VPIN < 0.3**: Low toxicity. Safe to be more aggressive.
   - **Order flow imbalance**: If heavily one-sided, our trades will have more impact.

5. **Intraday patterns**: If intraday data available, run intraday_volatility_pattern.
   Identify the U-shape: high vol at open/close, low at midday. Route volume to
   low-vol periods for lower impact, or to high-vol periods for better fill rates.

---

## Phase 2: Cost Estimation

6. **Pre-trade cost model**: Run execution_cost or expected_cost_model_tool with:
   - quantity = {total_shares}
   - price = latest close from the data
   - spread = Corwin-Schultz mean spread
   - adv = average daily volume from the data
   - volatility = daily return std from compute_returns

   Report the breakdown:
   - **Spread cost**: Fixed cost per share (half-spread). This is the floor.
   - **Market impact**: Square-root model: sigma * sqrt(Q/ADV). Increases with
     order size relative to volume. This is the main variable.
   - **Timing risk**: Opportunity cost of slow execution. Increases with volatility
     and execution duration.
   - **Total expected cost** in bps and dollars.

7. **Slippage estimate**: Run slippage_estimate for a quick sanity check.
   Compare to the detailed cost model. They should be in the same ballpark.
   - Participation rate > 20%: High impact, consider multi-day execution.
   - Participation rate 5-10%: Standard. Single-day execution is feasible.
   - Participation rate < 3%: Low impact. Can be aggressive.

---

## Phase 3: Schedule Construction

8. **VWAP schedule**: Run optimal_schedule with method="vwap". This distributes
   shares proportional to historical volume. Good for minimizing market impact.
   Report the first 5 slices and the max/min slice ratio (concentration).

9. **TWAP schedule**: Run optimal_schedule with method="twap". Uniform distribution.
   Use as a baseline comparison. Lower timing risk but potentially higher impact
   if volume is concentrated.

10. **IS schedule**: Run is_schedule_tool with alpha based on urgency:
    - Low urgency: alpha=0.3 (closer to VWAP, patient)
    - Medium urgency: alpha=0.5 (balanced)
    - High urgency: alpha=0.8 (closer to TWAP, aggressive)

    Report front_loaded_pct -- what fraction executes in the first quarter?
    Higher urgency = more front-loaded.

11. **POV schedule**: Run pov_schedule_tool with:
    - Low urgency: pov_rate=0.05 (5% of volume)
    - Medium urgency: pov_rate=0.10 (10%)
    - High urgency: pov_rate=0.20 (20%)

    Report fill_pct -- will we complete within the session? If < 100%, the order
    needs multiple days or a higher POV rate.

---

## Phase 4: Optimal Trajectory (Academic Models)

12. **Almgren-Chriss**: Run almgren_chriss with:
    - total_shares = {total_shares}
    - n_periods = number of 5-minute intervals in a trading day (78)
    - risk_aversion based on urgency: low=0.0001, medium=0.001, high=0.01

    Report: front_loaded_pct, trajectory shape. Higher risk aversion = more
    front-loaded (pay impact cost now to avoid timing risk later).

13. **Bertsimas-Lo**: Run bertsimas_lo_tool as comparison.
    This uses discrete-period dynamic programming vs AC's continuous approach.
    Compare trajectories -- similar shape validates the execution plan.
    Report expected_cost and cost_variance from BL. Lower variance = more certain cost.

14. **Close auction allocation**: Run close_auction with total_quantity remaining
    after continuous trading. Typical close_volume_pct is 0.15-0.25 for liquid stocks.
    Should we route to MOC? If the stock has high close volume, yes.

---

## Phase 5: Execution Plan Summary

15. **Compare strategies**: Side-by-side table:

    | Strategy | Expected Cost (bps) | Timing Risk | Completion Probability |
    |----------|--------------------|--------------|-----------------------|
    | VWAP     | ...                | Low          | High                  |
    | TWAP     | ...                | Medium       | High                  |
    | IS       | ...                | Medium       | High                  |
    | POV      | ...                | Low          | Depends on rate       |
    | AC       | ...                | Controlled   | High                  |

16. **Recommendation**: Based on urgency level "{urgency}":
    - Which schedule minimizes expected cost?
    - What POV rate keeps us under capacity constraints?
    - Should we split between continuous and close auction?
    - Any toxicity concerns that warrant slowing down?
    - Final execution plan with schedule, POV limits, and cost budget.

**Related prompts**: Use transaction_cost_deep_dive for post-trade TCA,
market_microstructure_deep_dive for deeper liquidity analysis,
dark_pool_analysis for venue selection.
""",
                },
            }
        ]

    @mcp.prompt()
    def market_microstructure_deep_dive(ticker: str = "AAPL") -> list[dict]:
        """Comprehensive microstructure analysis: liquidity, toxicity, spreads, price discovery."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Perform a comprehensive market microstructure analysis of {ticker}. This uses
the full suite of microstructure/ tools to assess liquidity, information asymmetry,
spread dynamics, and market quality.

---

## Phase 1: Liquidity Profile

1. **Core liquidity metrics**: Run liquidity_metrics on prices_{ticker.lower()}.
   Report Amihud illiquidity, Kyle's lambda, and Roll spread.

   **Interpretation framework**:
   - Amihud < 1e-7: Very liquid (mega-cap). Impact is negligible for small orders.
   - Amihud 1e-7 to 1e-5: Liquid. Standard execution algorithms work well.
   - Amihud > 1e-5: Illiquid. Need careful execution. Consider dark pool routing.
   - Kyle's lambda: Permanent impact coefficient. If statistically significant,
     informed trading is detectable. Higher lambda = more adverse selection risk.
   - Roll spread < 5 bps: Tight. Roll spread > 20 bps: Wide, execution costs matter.

2. **Rolling liquidity dynamics**: Run kyle_lambda_rolling with window=60 and
   amihud_rolling with window=20.
   - Is liquidity improving or deteriorating?
   - Any liquidity shocks? Correlate with market events.
   - kyle_lambda significant_pct: How often is the impact coefficient significantly
     different from zero? > 50% = persistent informed trading.

3. **Spread estimation**: Run corwin_schultz_spread for high-low spread estimate.
   Also run roll_spread_tool for the autocovariance-based estimate.
   Compare: they measure different things (CS = daily effective, Roll = bid-ask bounce).
   If CS >> Roll, the spread is dominated by volatility, not market-making.

4. **Effective spread** (if bid/ask data available): Run effective_spread_tool.
   This is the gold standard: 2 * |trade_price - midpoint|.
   Compare effective vs quoted spread. Ratio < 1 = price improvement, > 1 = adverse selection.

---

## Phase 2: Information & Toxicity

5. **Toxicity analysis**: Run toxicity_analysis on prices_{ticker.lower()}.
   - **VPIN**: Volume-synchronized probability of informed trading.
     > 0.6 = elevated toxicity. Market makers widen spreads in response.
     > 0.8 = extreme toxicity. Flash crash risk.
   - **Order flow imbalance**: Sustained one-sided flow = directional pressure.
     Compare to price moves -- does OFI predict returns?

6. **Trade classification** (if bid/ask data available): Run trade_classification_tool.
   Report buy_pct vs sell_pct. > 60% one-sided = significant directional pressure.
   Combine with VPIN for a complete toxicity picture.

7. **Order flow imbalance**: Run order_flow_imbalance if buy/sell volume columns
   exist. Rolling OFI near +1 or -1 = strong directional conviction.
   Mean-reverting OFI = noise traders dominate. Trending OFI = informed flow.

---

## Phase 3: Market Quality

8. **Efficiency metrics**: Run market_quality on prices_{ticker.lower()}.
   - **Variance ratio**: VR(q) = Var(q-period return) / (q * Var(1-period return)).
     VR = 1: random walk (efficient). VR > 1: positive autocorrelation (momentum).
     VR < 1: negative autocorrelation (mean reversion / bid-ask bounce).
   - **Market efficiency ratio**: Higher = more efficient price discovery.

9. **Intraday patterns** (if available): Run intraday_volatility_pattern.
   Report the U-shape or J-shape. Peak hour vs trough hour.
   Compute the peak-to-trough ratio: > 3 = very pronounced intraday pattern.
   Execution timing should avoid peak hours unless seeking liquidity.

10. **Price impact**: Run price_impact to measure the permanent component.
    Mean impact > 0 = trades move prices permanently (informed trading signal).
    Median impact near 0 = mostly temporary impact (noise).

---

## Phase 4: Cross-Venue Analysis

11. **Information share** (if multi-venue data available): Run information_share
    comparing two venue price series. Which venue leads price discovery?
    The venue with > 50% information share is the dominant price-setting venue.

12. **Liquidity commonality**: Run liquidity_commonality vs a market benchmark
    (e.g., SPY). High commonality (R-squared > 0.3) means the stock becomes
    illiquid during market stress -- systematic liquidity risk.
    Low commonality = idiosyncratic liquidity (less concerning for portfolio risk).

13. **Depth analysis** (if order book data available): Run depth_analysis.
    Mean imbalance > 0 = more buying interest on the book. < 0 = selling pressure.
    Compare depth imbalance to subsequent returns -- predictive signal?

---

## Phase 5: Synthesis

14. **Microstructure report card**:

    | Metric | Value | Assessment |
    |--------|-------|------------|
    | Amihud illiquidity | ... | Liquid/Illiquid |
    | Kyle's lambda trend | ... | Stable/Deteriorating |
    | Effective spread | ... | Tight/Wide |
    | VPIN | ... | Low/High toxicity |
    | Variance ratio | ... | Efficient/Inefficient |
    | Liquidity commonality | ... | Systematic/Idiosyncratic |

15. **Implications**:
    - Execution strategy recommendation based on liquidity profile.
    - Optimal execution window based on intraday patterns.
    - Adverse selection risk assessment.
    - Dark pool vs lit market routing recommendation.

**Related prompts**: Use optimal_execution for trade scheduling,
transaction_cost_deep_dive for post-trade analysis.
""",
                },
            }
        ]

    @mcp.prompt()
    def dark_pool_analysis(ticker: str = "AAPL") -> list[dict]:
        """Analyze dark pool impact, routing decisions, and venue selection."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Analyze dark pool dynamics and optimal venue routing for {ticker}. This workflow
combines microstructure metrics with execution cost analysis to determine when
and how to use dark pools effectively.

---

## Phase 1: Lit Market Baseline

1. **Lit market liquidity**: Run liquidity_metrics on prices_{ticker.lower()}.
   Establish the baseline Amihud, Kyle's lambda, and Roll spread for lit markets.

2. **Spread structure**: Run corwin_schultz_spread and roll_spread_tool.
   Wide spreads on lit venues create a stronger case for dark pool routing
   (potential midpoint execution saves half-spread per share).

3. **Toxicity assessment**: Run toxicity_analysis. High VPIN on lit venues
   means market makers are pricing in adverse selection -- dark pools may offer
   better pricing for uninformed orders.

---

## Phase 2: Dark Pool Suitability Assessment

4. **Order characteristics**:
   - Order size relative to ADV: Large orders (> 5% ADV) benefit most from dark pools.
   - Information content: Are we trading on information or rebalancing?
     Informed trades should stay on lit venues for guaranteed execution.
     Uninformed/rebalancing trades are ideal dark pool candidates.
   - Urgency: High urgency = lit market. Low urgency = dark pool acceptable.

5. **Information leakage risk**: Run kyle_lambda_rolling. If lambda is rising,
   there is increasing information asymmetry -- dark pool fills may be toxic
   (filled only when the price moves against you = adverse selection).

6. **Price impact comparison**: Run price_impact on lit market data.
   Estimate dark pool savings as: half-spread savings minus adverse selection cost.
   If permanent impact > 0 consistently, dark pool fills may be adversely selected.

---

## Phase 3: Routing Strategy

7. **Split recommendation**: Based on the analysis:
   - **Pure lit**: VPIN > 0.7, informed order, high urgency
   - **Dark-first**: VPIN < 0.3, uninformed order, wide spread, patient
   - **Hybrid** (typical): Route to dark pools first, sweep to lit for unfilled.
     Target: 30-50% dark pool fill rate for large-cap, 10-20% for mid-cap.

8. **Execution cost comparison**: Run execution_cost for lit-only scenario.
   Estimate dark pool scenario with spread savings:
   - Lit cost: spread + impact + timing risk
   - Dark pool cost: impact + timing risk (no spread) + adverse selection premium
   - Break-even: at what adverse selection rate is dark pool still cheaper?

9. **Market quality check**: Run market_quality. If variance ratio indicates
   efficiency (VR near 1), dark pool price improvement is more reliable.
   If VR far from 1, prices are noisy -- midpoint execution may not be "fair."

---

## Phase 4: Monitoring & Post-Trade

10. **Fill quality monitoring**: After routing, compare:
    - Dark pool fill rate: % of shares filled in dark pools
    - Reversion analysis: Do dark pool fills exhibit adverse selection?
      (Price moves against us after fill = adverse selection)
    - Effective spread: Compare effective spread for dark vs lit fills.

11. **Summary**: Recommended routing table:

    | Scenario | Venue | % of Order | Rationale |
    |----------|-------|-----------|-----------|
    | First pass | Dark pool | 40% | Spread savings, low toxicity |
    | Residual | Lit (VWAP) | 40% | Guaranteed execution |
    | Close | MOC auction | 20% | End-of-day liquidity |

**Related prompts**: Use optimal_execution for detailed scheduling,
market_microstructure_deep_dive for liquidity diagnostics.
""",
                },
            }
        ]

    @mcp.prompt()
    def intraday_trading(ticker: str = "AAPL") -> list[dict]:
        """Intraday strategy construction using microstructure signals."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Build an intraday trading strategy for {ticker} using microstructure signals.
This combines microstructure/ tools with ta/ indicators and execution/ scheduling
to construct a complete intraday workflow.

---

## Phase 1: Intraday Characterization

1. **Volatility pattern**: Run intraday_volatility_pattern on intraday data.
   Map the diurnal cycle. Identify:
   - Opening auction vol spike (9:30-10:00)
   - Midday lull (12:00-14:00)
   - Closing ramp (15:00-16:00)
   - Best windows for mean-reversion vs momentum strategies.

2. **Liquidity map**: Run liquidity_metrics with a short window (5-10).
   How does liquidity vary intraday? Usually worst at open, best at midday.
   Map Kyle's lambda throughout the day -- when is informed trading highest?

3. **Market quality**: Run market_quality on each hour's data.
   Variance ratio by hour: < 1 at midday suggests mean-reversion opportunities.
   > 1 at open suggests momentum continuation.

---

## Phase 2: Signal Construction

4. **Order flow signals**: Run toxicity_analysis on recent intraday data.
   - VPIN crossing above 0.6 = potential large directional move coming.
   - Order flow imbalance: sustained imbalance = momentum signal.
   - Combine with price: OFI leading price = predictive. Lagging = noise.

5. **Technical signals**: Use compute_indicator for intraday indicators:
   - **RSI(14)** on 5-min bars: < 30 or > 70 = intraday reversal signal.
   - **VWAP deviation**: Price far above/below VWAP = mean-reversion target.
   - **Bollinger Bands(20, 2)** on 5-min: Squeeze = breakout setup.

6. **Spread dynamics**: Run corwin_schultz_spread on intraday OHLC.
   Widening spreads = market makers pulling liquidity = caution.
   Narrowing spreads = increasing competition = better execution.

7. **Depth signals** (if available): Run depth_analysis.
   Depth imbalance predicts short-term price direction:
   - Imbalance > 0.3: Buy pressure, expect price increase.
   - Imbalance < -0.3: Sell pressure, expect price decrease.

---

## Phase 3: Strategy Framework

8. **Microstructure mean-reversion**:
   - Entry: OFI reversal + RSI extreme + price at Bollinger Band
   - Exit: Return to VWAP or OFI neutralizes
   - Stop: 2x average spread (to account for execution cost)
   - Timeframe: 5-30 minutes

9. **Information-based momentum**:
   - Entry: VPIN spike + sustained OFI + depth imbalance alignment
   - Exit: VPIN normalization or OFI reversal
   - Stop: Below entry minus 3x spread
   - Timeframe: 15-60 minutes

10. **Execution optimization**: For each signal:
    - Use slippage_estimate to verify the edge exceeds expected cost.
    - If edge < 2x expected slippage, skip the trade.
    - Route via optimal_schedule with method matching the signal:
      Momentum = aggressive (TWAP), Reversion = patient (VWAP).

---

## Phase 4: Risk Controls

11. **Intraday risk limits**:
    - Max position size: Based on Amihud -- never hold more than 5% of ADV.
    - Max loss per trade: 3x average spread.
    - Daily loss limit: 5x single trade max loss.
    - Max concurrent positions: 1-2 for single-stock intraday.

12. **Execution cost budget**: Run execution_cost for typical trade size.
    The strategy must clear this hurdle rate on average.
    Net expected return per trade = signal edge - spread - impact - fees.

**Related prompts**: Use market_microstructure_deep_dive for deeper analysis,
optimal_execution for execution scheduling.
""",
                },
            }
        ]

    @mcp.prompt()
    def transaction_cost_deep_dive(ticker: str = "AAPL") -> list[dict]:
        """Detailed post-trade TCA with cost attribution and benchmark comparison."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Perform a detailed Transaction Cost Analysis (TCA) for {ticker} executions.
This workflow uses execution/ and microstructure/ tools for post-trade attribution
and comparison against standard benchmarks.

---

## Phase 1: Setup & Data

1. **Trade data**: Check workspace for trades_{ticker.lower()} dataset.
   Required columns: price, quantity, timestamp. Each row = one execution fill.
   Also need market data (OHLCV) for the same period in prices_{ticker.lower()}.

2. **Pre-trade baseline**: Run execution_cost with the original order parameters
   to establish the pre-trade cost estimate. This is the benchmark we should
   have achieved or beaten.

---

## Phase 2: Benchmark Comparison

3. **TCA analysis**: Run transaction_cost_analysis with the trades JSON and
   market data dataset. This computes slippage vs:
   - **Arrival price**: Price at decision time. Implementation shortfall =
     (exec price - arrival price) / arrival price.
   - **VWAP benchmark**: Volume-weighted average price of the day.
     Positive VWAP slippage = we paid more than the market.
   - **Close benchmark**: Closing price. Shows timing cost vs waiting.

   Report mean slippage for each benchmark in bps.

4. **Cost attribution**: Break down total cost into components:
   - **Spread cost**: Half-spread at each execution. Fixed, non-negotiable.
   - **Market impact**: Price moved because of our trading. Temporary (reverts)
     vs permanent (doesn't revert). Run price_impact on the execution period.
   - **Timing cost**: Difference between arrival price and execution price due
     to market movement during the execution window. Favorable or adverse?
   - **Opportunity cost**: For unfilled quantity, the cost of NOT executing
     (measured as price move from decision to end of day).

---

## Phase 3: Market Conditions Context

5. **Liquidity during execution**: Run liquidity_metrics on the execution period.
   Was the market more or less liquid than average? Compare Amihud and Kyle's lambda
   to the 20-day average.

6. **Toxicity during execution**: Run toxicity_analysis on the execution period.
   Was VPIN elevated? If so, we may have been adversely selected -- higher cost
   is expected and possibly unavoidable.

7. **Spread conditions**: Run corwin_schultz_spread for the execution day.
   Compare to the 20-day average spread. Wider spreads = higher unavoidable costs.

8. **Volatility context**: Run compute_returns and analyze() on the
   execution period. Was it a high-vol day? Higher vol = higher timing risk cost
   but potentially lower impact if we were patient.

---

## Phase 4: Strategy Assessment

9. **Schedule adherence**: Compare actual execution profile to the planned schedule.
   - If we planned VWAP: did we track volume proportionally?
   - If we planned IS: were we front-loaded appropriately?
   - Deviation from plan = either good (adapted to conditions) or bad (discipline failure).

10. **Venue analysis**: If multi-venue data available, compare costs by venue.
    - Lit vs dark pool fills: which had lower effective spread?
    - Were dark pool fills adversely selected? (Reversion after fill)
    - Venue fill rates: which venues actually provided liquidity?

---

## Phase 5: Summary & Lessons

11. **TCA scorecard**:

    | Metric | Value (bps) | vs. Benchmark | Assessment |
    |--------|------------|---------------|------------|
    | Implementation shortfall | ... | Pre-trade est. | Better/Worse |
    | VWAP slippage | ... | 0 | Positive/Negative |
    | Spread cost | ... | Expected | Normal/Wide |
    | Impact cost | ... | Predicted | Low/High |
    | Timing cost | ... | 0 | Favorable/Adverse |

12. **Recommendations**:
    - Was the execution strategy appropriate for market conditions?
    - Should we have been more/less aggressive?
    - Were dark pools additive or detrimental?
    - Suggested improvements for next similar execution.

**Related prompts**: Use optimal_execution for pre-trade planning,
market_microstructure_deep_dive for venue analysis.
""",
                },
            }
        ]
