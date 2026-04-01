"""Risk & volatility prompt templates."""
from __future__ import annotations
from typing import Any


def register_risk_prompts(mcp: Any) -> None:

    @mcp.prompt()
    def volatility_deep_dive(dataset: str = "returns_aapl") -> list[dict]:
        """GARCH model selection, forecasting, news impact, realized vs implied."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
Comprehensive risk report for {dataset}:

1. risk_metrics — Sharpe, Sortino, max drawdown, hit ratio.
2. var_analysis at 95% and 99% confidence — historical and parametric.
3. tail_risk — Cornish-Fisher VaR (adjusts for skew/kurtosis).
4. stress_test — run all 7 built-in crisis scenarios (GFC, COVID, etc.).
5. crisis_drawdowns — top 5 worst drawdowns with dates and recovery.
6. factor_analysis if factor data available — what drives the risk?
7. portfolio_risk — component VaR, diversification ratio.
8. Summary: what's the worst-case scenario? Where is risk concentrated?
"""}}]

    @mcp.prompt()
    def tail_risk_assessment(dataset: str = "returns") -> list[dict]:
        """Extreme value theory and tail dependence analysis."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
