"""Regime detection and market state prompt templates."""
from __future__ import annotations
from typing import Any


def register_regime_prompts(mcp: Any) -> None:

    @mcp.prompt()
    def regime_detection(dataset: str = "returns") -> list[dict]:
        """Multi-method regime detection with comparison."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
First load the wraquant_system_context prompt for full module context.

Detect market regimes in {dataset}:

1. detect_regimes with method="hmm", n_regimes=2 — classic HMM.
2. detect_regimes with method="gmm" — compare without temporal structure.
3. detect_regimes with method="changepoint" — structural breaks.
4. Compare: do all methods agree on current regime?
5. Per-regime statistics — Sharpe, vol, max drawdown, duration.
6. Transition matrix — probability of regime switch.
7. Summary: which regime are we in? How long do regimes typically last?
"""}}]

    @mcp.prompt()
    def market_regime_monitor() -> list[dict]:
        """Current market state and regime probability monitoring."""
        return [{"role": "user", "content": {"type": "text", "text": """
First load the wraquant_system_context prompt for full module context.

Monitor current market regime:

1. Check workspace for recent data. detect_regimes on latest returns.
2. Current regime probability — how confident are we?
3. Expected duration — how long until regime switch?
4. Historical comparison — what happened last time we were in this regime?
5. Correlation regime — are correlations elevated (crisis signal)?
6. VIX regime check if available.
7. Allocation implications: risk-on (full exposure) or risk-off (hedge/reduce)?
"""}}]

    @mcp.prompt()
    def regime_backtest(dataset: str = "returns") -> list[dict]:
        """Backtest regime-conditional strategy vs unconditional."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
First load the wraquant_system_context prompt for full module context.

Regime-conditional backtest on {dataset}:

1. detect_regimes — identify bull (regime 0) and bear (regime 1).
2. Strategy: full exposure in bull regime, reduce/hedge in bear regime.
3. run_backtest with regime-filtered signals.
4. Compare to buy-and-hold (unconditional).
5. backtest_metrics for both — Sharpe improvement? Drawdown reduction?
6. Walk-forward: is the regime detection adding value out-of-sample?
7. Summary: does regime awareness improve risk-adjusted returns?
"""}}]

    @mcp.prompt()
    def changepoint_analysis(dataset: str = "returns") -> list[dict]:
        """Structural break detection and pre/post comparison."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
First load the wraquant_system_context prompt for full module context.

Changepoint analysis on {dataset}:

1. changepoint_detect with PELT method — where are structural breaks?
2. Binary segmentation for hierarchical ordering of breaks.
3. For each major break: compare pre vs post statistics (mean, vol, Sharpe).
4. CUSUM control chart — is there an ongoing shift?
5. stationarity_test pre vs post each break — did the process change?
6. Summary: what caused the breaks? Is the current regime stable?
"""}}]

    @mcp.prompt()
    def multi_asset_regime(datasets_json: str = '{"equity":"SPY","bonds":"TLT","gold":"GLD","vol":"VIX"}') -> list[dict]:
        """Multi-asset regime detection: identify regimes across asset classes simultaneously."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
First load the wraquant_system_context prompt for full module context.

Multi-asset regime detection using {datasets_json}:

1. **Data**: Parse the datasets JSON and load each asset from workspace.
   compute_returns for each. Align all series to common date range (inner join on dates).
2. **Joint feature matrix**: Stack returns from all assets into a single feature matrix.
   Each row = one date, each column = one asset's return. Optionally add rolling vol
   (21-day) for each asset as additional features.
3. **Multi-variate HMM**: detect_regimes with method="hmm", n_regimes=3 on the joint matrix.
   Three regimes capture: risk-on (equity up, bonds flat), risk-off (equity down, bonds up),
   and crisis (everything sells off / correlation spike).
4. **Per-regime asset behavior**: For each detected regime, compute mean return, volatility,
   and Sharpe for EVERY asset. Present as a regime × asset table.
   Key question: do bonds hedge equities in all regimes? Does gold work in crisis?
5. **Cross-asset correlation per regime**: correlation_analysis within each regime separately.
   In risk-on, correlations should be low (diversification works).
   In crisis, correlations spike (diversification fails). Quantify the difference.
6. **Regime transition dynamics**: Extract transition matrix. What is the probability of
   moving from risk-on to crisis directly? How long does each regime last on average?
   Current regime duration — are we overdue for a switch?
7. **Compare to single-asset regimes**: detect_regimes on equity alone (2-state).
   Does the multi-asset regime detect crisis earlier or more accurately?
   Multi-asset should capture cross-asset signals that single-asset misses.
8. **Summary**: Current multi-asset regime and confidence. Per-regime asset allocation
   implications. Which assets hedge which regimes? Early warning signals for regime shift.
"""}}]

    @mcp.prompt()
    def volatility_regime(dataset: str = "returns") -> list[dict]:
        """Volatility-based regime classification with GARCH persistence analysis."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
First load the wraquant_system_context prompt for full module context.

Volatility regime analysis for {dataset}:

1. **Data**: Load {dataset} from workspace. compute_returns if raw prices.
2. **Realized vol regimes**: Compute rolling 21-day realized volatility.
   detect_regimes with method="hmm", n_regimes=3 on the vol series itself.
   Label regimes: low-vol (calm), medium-vol (normal), high-vol (stress).
3. **Regime statistics**: For each vol regime, compute:
   - Mean and std of realized vol (how volatile is vol itself?)
   - Mean return and Sharpe (do returns differ across vol regimes?)
   - Average duration (days) and frequency of occurrence
   - Max drawdown experienced in each regime
4. **GARCH analysis per regime**: fit_garch with model="GJR" on the full series.
   Extract conditional volatility series. Compute GARCH persistence (α + β) overall.
   Then compute persistence within each vol regime — is vol more persistent in stress?
5. **GARCH half-life**: Half-life = log(0.5) / log(persistence). In high-vol regime,
   is the half-life longer (vol stays elevated) or shorter (mean-reverts fast)?
   This determines whether vol spikes are buying opportunities or sustained stress.
6. **Vol-of-vol**: Compute rolling std of rolling vol (vol-of-vol). High vol-of-vol =
   uncertainty about uncertainty = regime instability. detect_regimes on vol-of-vol
   for a "meta-regime" signal.
7. **Transition probabilities**: From the HMM, extract transition matrix.
   P(low → high) = crash probability. P(high → low) = recovery probability.
   Current state and duration — what does the model predict next?
8. **Trading implications**: In low-vol regime: sell vol premium (short straddles work).
   In high-vol regime: buy vol protection (hedge). Test these rules with run_backtest.
9. **Summary**: Current vol regime and confidence. GARCH persistence level.
   Expected vol path (mean-reverting or persistent). Regime-aware vol strategy.
"""}}]

    @mcp.prompt()
    def correlation_regime(dataset: str = "multi_asset_returns") -> list[dict]:
        """Correlation regime analysis: DCC and regime-conditional correlations."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
First load the wraquant_system_context prompt for full module context.

Correlation regime analysis for {dataset}:

1. **Data**: Load {dataset} from workspace. Needs multi-asset return series (3+ assets).
   compute_returns if raw prices. Align to common dates.
2. **Static baseline**: correlation_analysis — compute full-sample correlation matrix.
   This is the "average" picture. Note pairs with |corr| > 0.7 (high) and |corr| < 0.2 (low).
3. **DCC-GARCH**: fit_garch with DCC (Dynamic Conditional Correlation) model.
   Extract time-varying correlation series for each pair. Plot key pairs over time.
   When did correlations spike? Do spikes coincide with known crises?
4. **Correlation regimes**: detect_regimes with n_regimes=2 on the average pairwise
   DCC correlation series. Two regimes: "normal correlation" and "crisis correlation."
   - Normal regime: correlations at baseline, diversification works.
   - Crisis regime: correlations spike, diversification fails.
5. **Regime-conditional correlation matrices**: Compute separate correlation matrices
   for each regime. In crisis regime, which pairs see the biggest correlation increase?
   These are the pairs where diversification is an illusion.
6. **Contagion test**: Forbes-Rigobon adjusted correlation test. Raw correlation rises
   mechanically with vol. Test whether correlation increase is genuine contagion
   or just a volatility artifact. Genuine contagion = structural break in dependence.
7. **Diversification ratio per regime**: Compute portfolio diversification ratio
   (σ_weighted / σ_portfolio) in each regime. In crisis, diversification ratio drops
   toward 1.0 (no diversification benefit). Quantify the drop.
8. **Early warning**: Track rolling average DCC correlation. Rising trend = approaching
   crisis regime. Set threshold: if avg DCC > 80th percentile, shift to defensive allocation.
9. **Summary**: Current correlation regime and confidence. How much does diversification
   degrade in crisis? Which pairs are the worst offenders? DCC-based early warning status.
"""}}]
