"""Risk & volatility prompt templates."""
from __future__ import annotations
from typing import Any


def register_risk_prompts(mcp: Any) -> None:

    @mcp.prompt()
    def volatility_deep_dive(dataset: str = "returns_aapl") -> list[dict]:
        """GARCH model selection, forecasting, news impact, realized vs implied."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
First load the wraquant_system_context prompt for full module context.

Deep volatility analysis on {dataset}:

1. realized_volatility with yang_zhang estimator — current annualized vol?
2. fit_garch with GARCH, EGARCH, GJR — compare AIC/BIC via model_selection.
3. Best model: report persistence, half-life, unconditional vol.
4. news_impact_curve — asymmetric response to positive vs negative shocks?
5. forecast_volatility 10 days ahead with confidence intervals.
6. If implied vol available: compute variance_risk_premium (IV² - RV²).
7. Summary: is vol elevated or compressed? Mean-reverting or persistent? Asymmetric?
"""}}]

    @mcp.prompt()
    def risk_report(dataset: str = "portfolio_returns") -> list[dict]:
        """Full portfolio risk report: VaR, stress, crisis, factor decomposition."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
First load the wraquant_system_context prompt for full module context.

Generate a comprehensive institutional-grade risk report for {dataset}. This covers
the full risk/ module: VaR methods, tail risk, stress testing, factor decomposition,
portfolio risk analytics, and drawdown analysis. The report should be suitable for
a risk committee or CIO review — thorough, quantified, and actionable.

---

## Phase 1: Data Validation & Setup

1. **Workspace check**: Run workspace_status. Verify {dataset} exists and has sufficient history.
   Minimum 252 trading days (1 year) for basic risk metrics. 500+ days preferred for VaR
   backtesting and GARCH convergence. 1000+ days for reliable stress testing.

2. **Data quality**: Use query_data to inspect {dataset}.
   - Check for NaN values, gaps, or suspicious returns (|return| > 20% in one day).
   - Verify date alignment if multi-asset.
   - If portfolio returns: are these gross or net of fees? Note assumption.
   - Compute analyze() for a statistical overview before deep-diving into risk.

   **If data issues found**: Clean first. Large gaps may need interpolation.
   Outlier returns may need investigation (stock split? data error? real event?).

---

## Phase 2: Performance & Risk Metrics (risk/ module — 95 functions)

3. **Core risk metrics**: risk_metrics on {dataset}. Report in a structured table:

   | Metric | Value | Interpretation |
   |--------|-------|----------------|
   | Annualized Return | X% | vs benchmark |
   | Annualized Volatility | X% | vs target |
   | Sharpe Ratio | X.XX | > 1.0 excellent, 0.5-1.0 good, < 0.5 poor |
   | Sortino Ratio | X.XX | > Sharpe = positive skew (good) |
   | Calmar Ratio | X.XX | Return / Max DD. > 1.0 is good |
   | Max Drawdown | -X% | Depth of worst peak-to-trough |
   | Current Drawdown | -X% | How far from high-water mark right now |
   | Hit Ratio | X% | Daily win rate. > 53% is good |
   | Tail Ratio | X.XX | Right tail / left tail. > 1 = bigger wins than losses |
   | Omega Ratio | X.XX | Probability-weighted gain/loss. > 1 = positive expectancy |
   | Recovery Factor | X.XX | Total return / max drawdown. > 3 is good |

   **Context**: Compare metrics to the strategy's own history (rolling 12-month Sharpe)
   and to the benchmark. Is current risk-adjusted performance improving or deteriorating?

4. **Downside risk metrics**: Focus on the left tail.
   - Sortino ratio (penalizes only downside vol)
   - Downside deviation vs total std — if downside dev >> total std / sqrt(2), negative skew
   - Semi-variance: variance of negative returns only
   - Maximum consecutive losing days
   - Average losing day magnitude vs average winning day

---

## Phase 3: Value-at-Risk Analysis (5 methods compared)

5. **Historical VaR**: var_analysis with method="historical" at 95% and 99%.
   This is the percentile of the actual return distribution. No distributional assumption.
   - 95% VaR: the loss exceeded only 5% of trading days (1 in 20 days)
   - 99% VaR: the loss exceeded only 1% of trading days (1 in 100 days)
   - Report in both percentage and dollar terms (if portfolio value known)

6. **Parametric (Normal) VaR**: var_analysis with method="parametric".
   Assumes returns ~ Normal(mu, sigma). VaR = mu - z_alpha * sigma.
   - This UNDERESTIMATES risk if returns have fat tails (they usually do).
   - Compare to historical VaR: if parametric is less severe, the distribution has fat tails.

7. **Cornish-Fisher VaR**: tail_risk — Cornish-Fisher expansion adjusts for skewness
   and kurtosis. CF VaR = mu - [z + (z^2-1)*S/6 + (z^3-3z)*K/24 - (2z^3-5z)*S^2/36] * sigma.
   - This is the most reliable single-number VaR for non-normal distributions.
   - Compare CF VaR to normal VaR: the difference is the "fat tail penalty."
   - If CF VaR is 1.5x normal VaR: tails are significantly fatter than normal.

8. **GARCH-based VaR**: fit_garch with model="GJR", then use conditional vol for VaR.
   VaR_t = mu - z_alpha * sigma_t (conditional). This is time-varying — current VaR may be
   very different from average VaR depending on recent vol.
   - Current GARCH VaR vs unconditional VaR: ratio > 1.3 = elevated risk right now.
   - Is GARCH vol trending up or down? Rising = VaR will worsen tomorrow.

9. **CVaR (Expected Shortfall)**: Average loss in the worst 5% (or 1%) of days.
   CVaR is always worse than VaR. The gap between CVaR and VaR indicates tail severity.
   - CVaR / VaR > 1.5: extremely fat-tailed losses (the worst days are MUCH worse than VaR)
   - CVaR is the regulatory standard (Basel III prefers ES over VaR)

   **VaR comparison table**:
   | Method | 95% VaR | 99% VaR | 95% CVaR | 99% CVaR |
   |--------|---------|---------|----------|----------|
   | Historical | | | | |
   | Parametric | | | | |
   | Cornish-Fisher | | | | |
   | GARCH | | | | |

   **Recommendation**: Use Cornish-Fisher for reporting (adjusts for non-normality).
   Use GARCH for real-time monitoring (time-varying). Flag if any method gives VaR
   that is > 2x the parametric method — the normal assumption is dangerously wrong.

---

## Phase 4: Stress Testing (7 built-in crisis scenarios)

10. **Full crisis battery**: stress_test on {dataset} with ALL 7 built-in scenarios:
    - **GFC 2008** (Sep 2008 - Mar 2009): Credit crisis. Equities -50%, credit spreads +400bps.
    - **COVID 2020** (Feb-Mar 2020): Pandemic shock. Fastest -30% in history. V-shaped recovery.
    - **Dot-Com 2000** (Mar 2000 - Oct 2002): Tech bubble burst. Slow grind down. Growth worst.
    - **Rate Hike**: Rising rates scenario. Duration-sensitive assets hit hardest.
    - **Vol Spike**: VIX doubles in a week. Short vol and leveraged strategies destroyed.
    - **Flash Crash**: Sudden liquidity evaporation. Recovery within hours but damage done.
    - **EM Crisis**: Emerging market contagion. Dollar strengthens, EM assets crash.

    **Report for each scenario**:
    - Expected portfolio loss (%)
    - Duration of stress period
    - Recovery time (if strategy survived)
    - Which assets/positions are most vulnerable

11. **Rank scenarios by severity**: Sort all 7 by expected loss (worst first).
    The top 3 scenarios are the portfolio's primary vulnerabilities.
    - Are the worst scenarios plausible in current market conditions?
    - Current regime (from detect_regimes) — which scenarios are more likely right now?

12. **Correlation stress**: What happens if all correlations go to 1.0?
    In crisis, diversification fails. Compute portfolio vol under perfect correlation.
    Diversification ratio in normal vs crisis correlation: how much does it degrade?
    The gap = the "diversification illusion" — the risk you think you've diversified away
    but haven't in a crisis.

---

## Phase 5: Drawdown Analysis

13. **Historical drawdowns**: crisis_drawdowns — extract the top 5 worst drawdowns.
    For each, report:
    - Start date, trough date, recovery date (or "still in drawdown")
    - Depth (max loss from peak)
    - Duration (peak to trough)
    - Recovery time (trough back to previous high)
    - Total underwater period = duration + recovery
    - What caused it? (map to known market events if possible)

14. **Current drawdown status**: Is the portfolio currently in a drawdown?
    - If yes: depth so far, duration, how does it compare to historical drawdowns?
      Is it the worst ever? Top 3? Recovering or still declining?
    - If no: how close to a new drawdown? Days since last peak? Is GARCH vol rising
      (early warning)?

15. **Drawdown risk metrics**:
    - CDaR (Conditional Drawdown at Risk): Average of the worst X% of drawdowns.
      Like CVaR but for drawdowns. More intuitive for investors.
    - DaR (Drawdown at Risk): The drawdown exceeded only X% of the time.
    - Maximum consecutive underwater days: How long could you be stuck below high-water mark?

---

## Phase 6: Factor Risk Decomposition (if factor data available)

16. **Factor analysis**: If factor return data exists (Fama-French, PCA, or custom factors),
    run factor_analysis on {dataset}.
    - **Factor betas**: Exposure to each factor. Which factors drive the portfolio?
    - **R-squared**: Fraction of risk explained by factors. High R^2 = factor-driven.
      Low R^2 = idiosyncratic risk dominates (good for alpha, concerning for risk management).
    - **Alpha**: Residual return not explained by factors. Positive alpha = genuine skill.

17. **Factor risk contribution**: Decompose total portfolio risk into factor contributions.
    - Systematic risk: from factor exposures (market, size, value, momentum, etc.)
    - Idiosyncratic risk: from stock-specific / alpha bets
    - Which single factor contributes most to risk? Is that intentional?
    - Any unintended factor exposures? (e.g., a "quality" strategy with hidden momentum bet)

18. **Rolling factor stability**: Are factor betas stable over time?
    Compute rolling 60-day factor betas. If betas drift significantly, the risk profile
    is changing — may need rebalancing. Flag any beta that changed by > 0.3 in 3 months.

---

## Phase 7: Portfolio Risk Analytics (multi-asset only)

19. **Portfolio risk decomposition** (if multi-asset):
    - **Component VaR**: Each asset's contribution to total portfolio VaR.
      Sum of component VaR = total VaR. Is risk concentrated in 1-2 positions?
    - **Marginal VaR**: If you add $1 to an asset, how much does portfolio VaR change?
      High marginal VaR = adding that asset increases risk significantly.
    - **Incremental VaR**: If you remove an asset entirely, how does portfolio VaR change?
    - **Diversification ratio**: Sum of standalone vols / portfolio vol.
      > 1.5 = good diversification. Close to 1.0 = concentrated / highly correlated.

20. **Risk budgeting check**: Does the actual risk allocation match the intended allocation?
    If using risk parity, each asset should contribute equally. Report deviations.
    If discretionary, is any single position contributing > 40% of total risk?

---

## Phase 8: Report Synthesis

21. **Executive summary** (present this first, even though it's computed last):

    **Risk status**: GREEN / YELLOW / RED
    - GREEN: All metrics within normal ranges, regime stable, no VaR breaches.
    - YELLOW: Some metrics elevated (e.g., vol above average, regime uncertain,
      approaching drawdown limit). Monitor closely.
    - RED: Multiple warning signals (VaR breach, regime shift to bear, drawdown
      exceeding historical norms, stress test showing catastrophic loss).

    **Key numbers** (one table):
    | Metric | Current | Historical Avg | Status |
    |--------|---------|----------------|--------|
    | Annualized Vol | X% | Y% | GREEN/YELLOW/RED |
    | 99% VaR (CF) | X% | Y% | |
    | Max Drawdown | X% | Y% | |
    | Current Drawdown | X% | Limit: Y% | |
    | Regime | Bull/Bear | — | |
    | GARCH Vol | X% | Unconditional: Y% | |

    **Top 3 risks** (ranked by severity):
    1. [Most critical risk with quantification]
    2. [Second risk]
    3. [Third risk]

    **Recommended actions** (if any):
    - Position adjustments to reduce concentration
    - Hedging for top stress scenario
    - Regime-aware allocation shift

**Related prompts**: Use stress_test_battery for deeper stress analysis,
tail_risk_assessment for EVT analysis, correlation_breakdown for DCC dynamics,
portfolio_stress_test for multi-asset stress, var_backtesting to validate VaR accuracy.
"""}}]

    @mcp.prompt()
    def tail_risk_assessment(dataset: str = "returns") -> list[dict]:
        """Extreme value theory and tail dependence analysis."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
First load the wraquant_system_context prompt for full module context.

Tail risk assessment for {dataset}:

1. distribution_fit — fit Student-t, compare to normal. Tail index?
2. tail_risk — CDaR (conditional drawdown at risk), DaR.
3. Cornish-Fisher VaR vs standard VaR — how much does skew/kurtosis matter?
4. If multi-asset: tail_dependence — do assets crash together?
5. stress_test — worst scenarios and their probability.
6. Summary: how fat are the tails? Is standard VaR underestimating risk?
"""}}]

    @mcp.prompt()
    def stress_test_battery(dataset: str = "portfolio_returns") -> list[dict]:
        """Run all stress scenarios and rank by severity."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
First load the wraquant_system_context prompt for full module context.

Stress test battery for {dataset}:

1. stress_test with all built-in scenarios: GFC 2008, COVID 2020, dot-com, rate hike, vol spike, flash crash, EM crisis.
2. Rank scenarios by severity (max loss).
3. correlation_stress — what happens if all correlations go to 1?
4. For each top-3 worst scenario: what's the expected loss? Recovery time?
5. Recommend hedging strategies for the worst scenarios.
"""}}]

    @mcp.prompt()
    def correlation_breakdown(dataset: str = "multi_asset_returns") -> list[dict]:
        """Dynamic correlation and contagion analysis."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
First load the wraquant_system_context prompt for full module context.

Correlation analysis for {dataset}:

1. correlation_analysis — static correlation matrix.
2. fit_garch with DCC model — time-varying correlations.
3. detect_regimes — do correlations spike in crisis regime?
4. contagion_analysis — compare normal vs crisis correlations.
5. diversification_ratio — is the portfolio truly diversified?
6. Summary: are correlations stable or regime-dependent? Contagion risk?
"""}}]

    @mcp.prompt()
    def vol_surface_analysis(dataset: str = "options_data") -> list[dict]:
        """Implied volatility surface and skew analysis."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
First load the wraquant_system_context prompt for full module context.

Volatility surface analysis:

1. If options data available: compute implied vols across strikes and maturities.
2. Fit SABR model — calibrate alpha, rho, nu parameters.
3. Analyze vol skew — steeper skew = more crash fear.
4. Term structure — is vol curve in contango or backwardation?
5. Compare implied vs realized — variance risk premium positive?
6. Summary: what is the market pricing in? Crash protection expensive?
"""}}]

    @mcp.prompt()
    def credit_risk_assessment(dataset: str = "firm_data") -> list[dict]:
        """Credit risk assessment: Merton model, Altman Z-score, default probability."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
First load the wraquant_system_context prompt for full module context.

Credit risk assessment for {dataset}:

1. **Data**: Load {dataset} from workspace. Needs equity price, total assets, total liabilities,
   EBIT, sales, working capital, retained earnings, market cap. Check workspace_status.
2. **Merton structural model**: Treat equity as a call option on firm assets.
   - Estimate asset value (V) and asset volatility (σ_A) from equity price + equity vol
     using the Merton system of equations (iterative solve).
   - Compute distance-to-default: DD = (ln(V/D) + (μ - 0.5σ²_A)T) / (σ_A √T)
     where D = debt face value, T = 1 year horizon.
   - Default probability = N(-DD). DD < 2 = elevated risk. DD < 1 = distressed.
3. **Altman Z-score**: Compute Z = 1.2×(WC/TA) + 1.4×(RE/TA) + 3.3×(EBIT/TA) +
   0.6×(MktCap/TL) + 1.0×(Sales/TA).
   - Z > 2.99 = safe zone, 1.81-2.99 = grey zone, < 1.81 = distress zone.
   - Track Z-score trend — is credit quality improving or deteriorating?
4. **KMV-style EDF**: Map distance-to-default to Expected Default Frequency using
   historical default rates. EDF < 0.5% = investment grade, > 2% = high yield territory.
5. **Credit spread implied**: If bond data available, decompose spread into
   default component (from Merton PD) and liquidity/risk premium residual.
6. **Equity volatility signal**: fit_garch on equity returns. Rising vol = rising default risk.
   Compare GARCH conditional vol to historical average — elevated vol is a warning.
7. **Peer comparison**: If multiple firms, rank by DD and Z-score side by side.
   Which firms are outliers? Any deteriorating trends?
8. **Summary**: Default probability estimate, Z-score zone, DD level. Is credit risk
   increasing or stable? Any early warning signals from equity vol?
"""}}]

    @mcp.prompt()
    def copula_risk(dataset: str = "multi_asset_returns") -> list[dict]:
        """Copula-based risk analysis: tail dependence and crash co-movement."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
First load the wraquant_system_context prompt for full module context.

Copula risk analysis for {dataset}:

1. **Data**: Load {dataset} from workspace. Needs multi-asset return series.
   compute_returns if raw prices. Minimum 2 assets, ideally 5+.
2. **Marginal distributions**: For each asset, distribution_fit — fit Student-t or
   skewed-t to capture fat tails. Estimate degrees of freedom. Lower df = fatter tails.
   Transform to uniform margins using the fitted CDF (probability integral transform).
3. **Gaussian copula fit**: Fit Gaussian copula to the uniform margins.
   Extract correlation matrix. This captures linear dependence but NOT tail dependence.
4. **Student-t copula fit**: Fit Student-t copula — captures symmetric tail dependence.
   Estimate copula degrees of freedom (lower = more tail dependence).
   Compare AIC/BIC vs Gaussian copula — t-copula almost always wins for financial data.
5. **Clayton copula**: Fit Clayton copula for lower tail dependence (crash co-movement).
   Clayton parameter θ > 0 means assets crash together. Higher θ = stronger co-crash.
   This is the key metric for portfolio crash risk.
6. **Tail dependence coefficients**: From each copula, extract:
   - Lower tail dependence λ_L: P(X < q | Y < q) as q → 0. This is crash co-movement.
   - Upper tail dependence λ_U: P(X > q | Y > q) as q → 1. Rally co-movement.
   - Gaussian copula has λ_L = λ_U = 0 (dangerous underestimation of crash risk).
7. **Copula VaR**: Simulate 100,000 scenarios from the best-fit copula.
   Compute portfolio VaR and CVaR from copula simulations.
   Compare to VaR from normal assumption — how much is crash risk underestimated?
8. **Regime-conditional copulas**: detect_regimes first, then fit copulas separately
   in each regime. Does tail dependence increase in bear regimes? (Usually yes.)
9. **Summary**: Which copula fits best? How strong is tail dependence? Is standard VaR
   underestimating crash risk? How much worse is the copula-based worst case?
"""}}]

    @mcp.prompt()
    def liquidity_risk(dataset: str = "returns") -> list[dict]:
        """Liquidity risk analysis: Amihud crisis comparison and spread widening."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
First load the wraquant_system_context prompt for full module context.

Liquidity risk analysis for {dataset}:

1. **Data**: Load {dataset} from workspace. Needs return and volume data.
   Check workspace_status for available fields.
2. **Current Amihud illiquidity**: Compute Amihud ratio = mean(|return| / dollar_volume)
   over trailing 21 days (1 month). Use liquidity_analysis if available.
   Compare to trailing 252-day (1 year) average. Ratio > 1.5x average = liquidity stress.
3. **Historical crisis Amihud levels**: Compute Amihud ratio during known crisis periods:
   - GFC (Sep 2008 – Mar 2009): Amihud spike baseline for extreme stress
   - COVID (Feb 2020 – Apr 2020): rapid liquidity evaporation
   - VIX spike events: liquidity typically worst at vol peaks
   Compare current Amihud to these crisis peaks. Current / GFC_peak = crisis severity %.
4. **Spread widening scenarios**: Estimate effective spread using Roll's measure or
   bid-ask data if available. Model spread widening:
   - Mild stress: spread widens 2x → compute additional transaction cost
   - Severe stress: spread widens 5x → compute cost
   - Crisis: spread widens 10x → compute cost
   For a portfolio of given size, what is the liquidity-adjusted loss?
5. **Volume drought analysis**: Compute rolling 5-day average volume vs 252-day average.
   Identify periods where volume dropped > 50%. What happened to prices during volume droughts?
6. **Liquidation horizon**: For current portfolio, estimate days to liquidate each position
   (assuming max 10% of ADV per day). Flag any position requiring > 5 days to exit.
7. **Liquidity-volatility feedback**: correlation_analysis between Amihud ratio and
   realized volatility. In stress, illiquidity and volatility feed on each other.
   Compute the conditional vol given Amihud > 90th percentile.
8. **Summary**: Current liquidity conditions vs historical norms and crisis peaks.
   Liquidation time estimates. Cost of emergency exit. Stress scenario impacts.
"""}}]

    @mcp.prompt()
    def var_backtesting(dataset: str = "returns", var_dataset: str = "var_forecasts") -> list[dict]:
        """VaR backtesting: compare VaR predictions vs actual realized losses."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
First load the wraquant_system_context prompt for full module context.

VaR backtesting for {dataset} against {var_dataset}:

1. **Data**: Load {dataset} (realized returns) and {var_dataset} (VaR forecasts) from workspace.
   If no VaR forecasts exist, generate them: compute rolling 1-day VaR at 95% and 99%
   using var_analysis with historical, parametric (normal), and Cornish-Fisher methods.
2. **Exception counting (Kupiec test)**: Count VaR breaches (days when loss > VaR forecast).
   At 99% confidence over 250 days, expect ~2.5 exceptions.
   - Green zone: 0-4 exceptions (model acceptable)
   - Yellow zone: 5-9 exceptions (model questionable)
   - Red zone: 10+ exceptions (model rejected)
   Compute Kupiec POF (proportion of failures) test p-value. Reject if p < 0.05.
3. **Independence test (Christoffersen)**: Are VaR breaches clustered or independent?
   Compute Christoffersen's interval forecast test. Clustered breaches = model misses
   volatility dynamics. Use a Markov chain test on breach/no-breach sequence.
4. **Conditional coverage test**: Joint test of correct frequency AND independence
   (Christoffersen's conditional coverage). This is the definitive VaR backtest.
   Report test statistic and p-value.
5. **Method comparison**: If multiple VaR methods available, rank by:
   - Exception rate closest to nominal (1% or 5%)
   - Independence of exceptions (no clustering)
   - Average magnitude of breaches (how bad are the misses?)
   - Cornish-Fisher typically outperforms normal for fat-tailed data.
6. **Breach severity**: When VaR is breached, by how much? Compute average excess loss
   beyond VaR on breach days. If breaches are 3x the VaR, the model badly underestimates
   tail risk even if the count is correct.
7. **Traffic light report**: For each VaR method and confidence level, assign
   Green/Yellow/Red based on Kupiec + Christoffersen + breach severity.
8. **Summary**: Which VaR method passes the backtest? Any clustering of breaches?
   Does the model need recalibration? Recommended VaR approach going forward.
"""}}]
