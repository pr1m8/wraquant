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
