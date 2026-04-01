"""Pricing and fixed income prompt templates."""
from __future__ import annotations
from typing import Any


def register_pricing_prompts(mcp: Any) -> None:

    @mcp.prompt()
    def option_pricing(underlying: str = "AAPL", strike: float = 150, expiry: str = "2024-06-21") -> list[dict]:
        """Option pricing: BS, Greeks, vol smile, Heston calibration."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Price a {underlying} option (K={strike}, expiry={expiry}):

1. Get current price from workspace data.
2. price_option with Black-Scholes — baseline price.
3. compute_greeks — delta, gamma, theta, vega, rho.
4. Estimate implied volatility if market price available.
5. Vol smile across strikes — fit SABR model.
6. Heston calibration if multiple strikes/maturities available.
7. Compare BS vs Heston prices — how much does stochastic vol matter?
8. Summary: fair value, key Greeks, vol smile shape.
"""}}]

    @mcp.prompt()
    def yield_curve_analysis(dataset: str = "bond_data") -> list[dict]:
        """Yield curve construction and analysis."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Yield curve analysis from {dataset}:

1. Bootstrap zero curve from bond prices/yields.
2. Interpolate at standard maturities (1y, 2y, 5y, 10y, 30y).
3. Compute forward rates — what does the market expect?
4. Duration and convexity for each maturity.
5. Key rate durations — sensitivity to each point on the curve.
6. Curve shape: normal, flat, or inverted? What does it signal?
7. Rate scenario analysis: parallel shift, steepener, flattener.
8. Summary: curve shape, forward rate expectations, rate risk.
"""}}]

    @mcp.prompt()
    def exotic_pricing() -> list[dict]:
        """Price exotic options via characteristic functions and FBSDEs."""
        return [{"role": "user", "content": {"type": "text", "text": """
Exotic option pricing:

1. Choose model: Heston, Variance Gamma, NIG, or CGMY.
2. Calibrate model to market data (vol surface).
3. Price European via characteristic function (FFT or COS method).
4. Compare across models — which fits the market best?
5. For American exercise: use FBSDE solver (reflected BSDE).
6. For high-dimensional (basket): use deep BSDE solver.
7. Greeks via bump-and-revalue.
8. Summary: model-dependent prices, which model to trust?
"""}}]

    @mcp.prompt()
    def vol_arbitrage() -> list[dict]:
        """Volatility arbitrage: compare implied vs realized and trade the spread."""
        return [{"role": "user", "content": {"type": "text", "text": """
Volatility arbitrage analysis:

1. **Data**: Check workspace_status for options and underlying price data.
   Load implied volatility (IV) series and underlying prices.
   compute_returns on the underlying.
2. **Realized volatility**: Compute realized vol using multiple estimators:
   - Close-to-close (standard) with 21-day window
   - Yang-Zhang (OHLC-based, more efficient) with realized_volatility
   - GARCH conditional vol via fit_garch with model="GJR"
   Compare estimators — they should broadly agree. Yang-Zhang is most efficient.
3. **Variance risk premium (VRP)**: Compute VRP = IV² - RV² (in variance space) or
   IV - RV (in vol space). Plot VRP time series. Is VRP positive on average?
   (It should be — option sellers earn a risk premium.)
   Current VRP: above or below historical mean? Percentile rank?
4. **VRP signal**: Generate trading signal:
   - VRP > historical mean + 1σ: IV is "expensive" → sell vol (short straddle/strangle)
   - VRP < historical mean - 1σ: IV is "cheap" → buy vol (long straddle/strangle)
   - Otherwise: no trade
5. **Term structure signal**: If IV at multiple maturities available:
   - Contango (long-dated IV > short-dated): normal, vol expected to mean-revert up
   - Backwardation (short-dated IV > long-dated): stress, near-term fear elevated
   Trade: sell near-term, buy far-term in backwardation (calendar spread).
6. **Delta hedging simulation**: Simulate delta-hedged short vol position.
   P&L = (IV² - RV²) × vega × time (approximately). Daily delta rebalancing.
   Compute: hedging cost, gamma exposure, P&L distribution.
7. **Risk analysis**: fit_garch to estimate vol-of-vol. When vol-of-vol is high,
   vol arb is riskier. stress_test — what happens in a vol spike (VIX doubles)?
   Maximum loss on the short vol position?
8. **Backtest**: run_backtest with the VRP signal. backtest_metrics.
   Key: Sharpe should be positive but with negative skew (short vol characteristic).
   Compare to naive "always short vol" strategy.
9. **Summary**: Current VRP level and percentile. Signal direction.
   Expected P&L and risk. Backtest results. Regime dependence.
"""}}]

    @mcp.prompt()
    def convertible_bond(bond_dataset: str = "convertible_data") -> list[dict]:
        """Convertible bond pricing: Black-Scholes + credit spread hybrid model."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Convertible bond pricing for {bond_dataset}:

1. **Data**: Load {bond_dataset} from workspace. Needs: face value, coupon rate, maturity,
   conversion ratio, underlying stock price, stock volatility, risk-free rate, credit spread.
   Check workspace_status for available fields.
2. **Straight bond value (bond floor)**: Price the bond ignoring conversion option.
   Discount coupons + face value at risk-free rate + credit spread.
   Bond floor = sum(coupon × e^(-r_credit × t_i)) + face × e^(-r_credit × T).
   This is the minimum value — the bond can't trade below this (ignoring default).
3. **Conversion value**: Conversion value = conversion_ratio × current_stock_price.
   If conversion_value > bond_floor, the convert is "in the money."
   Conversion premium = (convert_price - conversion_value) / conversion_value.
4. **Equity option component**: The conversion right is a call option on the stock.
   Price using Black-Scholes: price_option with S = stock_price × conversion_ratio,
   K = face_value, T = remaining maturity, σ = stock_volatility, r = risk-free rate.
5. **Convertible value**: Convertible ≈ max(bond_floor, conversion_value) + time_value.
   More precisely: convertible = straight_bond_value + equity_option_value.
   The option value accounts for future conversion opportunity.
6. **Credit-adjusted pricing**: The credit spread affects both the bond floor AND the
   option value (through credit-adjusted discounting). Higher credit risk = lower bond
   floor but also lower option value (issuer may default before conversion).
   Re-price with credit spread shifted ±50bps — credit sensitivity.
7. **Greeks**: Compute convertible Greeks:
   - Delta: sensitivity to stock price (0 at deep bond, ~conversion_ratio at deep equity)
   - Gamma: delta sensitivity — peaks near conversion parity
   - Rho: rate sensitivity — convertible has both equity and bond rate exposure
   - Vega: vol sensitivity — higher vol increases option component
8. **Scenario analysis**: Price the convertible under scenarios:
   - Stock +20%: how much does the convert appreciate? (equity participation)
   - Stock -20%: how much downside protection from bond floor?
   - Spread widens 200bps: credit impact on both components
   - Vol increase 10pts: option value increase
9. **Summary**: Fair value estimate. Bond floor. Conversion premium. Key Greeks.
   Is it trading rich or cheap vs model? Equity upside participation vs downside protection.
"""}}]

    @mcp.prompt()
    def structured_product() -> list[dict]:
        """Structured product pricing: price a principal-protected note."""
        return [{"role": "user", "content": {"type": "text", "text": """
Structured product pricing (principal-protected note):

1. **Product definition**: A principal-protected note (PPN) with:
   - 100% principal protection at maturity (investor gets at least face value back)
   - Participation rate in equity upside (e.g., 80% of SPY appreciation)
   - Maturity: 3 years. Face value: $1000.
   The PPN = zero-coupon bond + call option on the underlying.
2. **Zero-coupon bond component**: Price ZCB at the issuer's credit-adjusted discount rate.
   ZCB_value = face_value × e^(-r_credit × T).
   With r_credit = risk_free + credit_spread. If credit spread = 100bps, r_free = 4%:
   ZCB = 1000 × e^(-0.05 × 3). This provides the principal protection.
3. **Option budget**: The remaining value (face_value - ZCB_value) funds the call option.
   Option_budget = face_value - ZCB_value. This determines the participation rate.
4. **Call option pricing**: price_option for a 3-year ATM call on the underlying.
   Use Black-Scholes with current implied vol (or fit_garch for realized vol estimate).
   Call_price = BS(S, K=S, T=3, σ, r). How many calls can the option budget buy?
   Participation_rate = option_budget / call_price.
5. **Issuer economics**: The issuer earns the credit spread on the ZCB (funding advantage)
   plus any vol markup (selling the option at a higher implied vol than fair).
   Fair participation rate vs offered rate — the gap is the issuer's margin.
6. **Scenario analysis**: Compute PPN payoff under scenarios:
   - Underlying +30%: payoff = 1000 + 1000 × participation × 30%
   - Underlying +10%: payoff = 1000 + 1000 × participation × 10%
   - Underlying -20%: payoff = 1000 (principal protected)
   - Underlying -50%: payoff = 1000 (principal protected, but opportunity cost is huge)
   Compare to direct equity investment under each scenario.
7. **Sensitivity analysis**: How does fair participation rate change with:
   - Vol: higher vol → more expensive call → lower participation
   - Rates: higher rates → cheaper ZCB → more option budget → higher participation
   - Credit spread: wider spread → cheaper ZCB → more budget but more issuer default risk
   - Maturity: longer → cheaper ZCB → higher participation but more time risk
8. **Risk analysis**: Credit risk of the issuer — if issuer defaults, principal protection
   is lost. stress_test the issuer's credit. Liquidity risk — PPN is typically illiquid,
   early exit at a discount. Mark-to-market volatility before maturity.
9. **Summary**: Fair participation rate. Issuer margin. Scenario payoffs.
   Is the PPN a good deal for the investor vs direct equity + puts?
   Key risks: issuer credit, liquidity, opportunity cost.
"""}}]
