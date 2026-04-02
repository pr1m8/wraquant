"""Risk analysis workflow using wraquant.

This example demonstrates a comprehensive risk analysis pipeline:

    1. Generate or fetch return data
    2. Compute core risk metrics (Sharpe, Sortino, max drawdown)
    3. Value-at-Risk and Conditional VaR (Expected Shortfall)
    4. GARCH volatility modeling and forecasting
    5. Historical stress testing
    6. Market regime detection via HMM
    7. Performance tearsheet

Uses wraquant.risk, wraquant.vol, and wraquant.regimes -- three modules
that together form the risk management backbone of any quant strategy.

Usage:
    python examples/risk_analysis.py AAPL
    python examples/risk_analysis.py SPY --periods 1000

Requirements:
    pip install wraquant  # core deps only -- no API key needed
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_synthetic_returns(
    ticker: str,
    n: int = 756,
    seed: int = 42,
) -> pd.Series:
    """Generate synthetic daily returns with realistic properties.

    Uses a simple regime-switching model to produce returns with
    volatility clustering, fat tails, and mild negative skewness --
    properties observed in real equity returns.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n, freq="B")

    # Two-regime model: bull (low vol) and bear (high vol)
    returns = np.empty(n)
    regime = 0  # start in bull
    for i in range(n):
        if regime == 0:  # bull
            returns[i] = rng.normal(0.0005, 0.012)
            if rng.random() < 0.02:  # 2% chance of switching to bear
                regime = 1
        else:  # bear
            returns[i] = rng.normal(-0.0008, 0.028)
            if rng.random() < 0.05:  # 5% chance of switching back to bull
                regime = 0

    return pd.Series(returns, index=dates, name=ticker)


def generate_synthetic_prices(returns: pd.Series) -> pd.Series:
    """Convert returns to a price series starting at 100."""
    prices = 100 * (1 + returns).cumprod()
    prices.name = returns.name
    return prices


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def header(title: str) -> None:
    width = 65
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def subheader(title: str) -> None:
    print(f"\n  --- {title} ---")


def pct(value: float) -> str:
    return f"{value:.2%}"


def fmt(value: float, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_risk_analysis(ticker: str, n_periods: int = 756) -> None:
    """Run the full risk analysis pipeline."""

    header(f"RISK ANALYSIS: {ticker}")
    print(f"\n  Generating {n_periods} days of synthetic return data...")

    # Step 0: Generate data
    returns = generate_synthetic_returns(ticker, n=n_periods)
    prices = generate_synthetic_prices(returns)

    print(f"  Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"  Total observations: {len(returns)}")
    print(f"  Cumulative return: {pct((1 + returns).prod() - 1)}")

    # ==================================================================
    # Step 1: Core Risk Metrics
    # ==================================================================
    header("STEP 1: CORE RISK METRICS")

    from wraquant.risk import sharpe_ratio, sortino_ratio, max_drawdown, hit_ratio

    sr = sharpe_ratio(returns)
    so = sortino_ratio(returns)
    md = max_drawdown(prices)
    hr = hit_ratio(returns)

    ann_return = float(((1 + returns).prod() ** (252 / len(returns))) - 1)
    ann_vol = float(returns.std() * np.sqrt(252))

    print(f"\n  Annualized Return:   {pct(ann_return)}")
    print(f"  Annualized Vol:      {pct(ann_vol)}")
    print(f"  Sharpe Ratio:        {fmt(sr, 2)}")
    print(f"  Sortino Ratio:       {fmt(so, 2)}")
    print(f"  Max Drawdown:        {pct(md)}")
    print(f"  Hit Ratio:           {pct(hr)}")

    # Interpretation
    print(f"\n  Interpretation:")
    if sr > 1.0:
        print(f"    Sharpe > 1: Good risk-adjusted performance.")
    elif sr > 0.5:
        print(f"    Sharpe 0.5-1.0: Acceptable for a long-only strategy.")
    else:
        print(f"    Sharpe < 0.5: Poor risk-adjusted returns.")

    if abs(md) > 0.20:
        print(f"    Max drawdown > 20%: Significant peak-to-trough decline.")
    else:
        print(f"    Max drawdown < 20%: Contained drawdown risk.")

    # ==================================================================
    # Step 2: Value-at-Risk and CVaR
    # ==================================================================
    header("STEP 2: VALUE-AT-RISK & EXPECTED SHORTFALL")

    from wraquant.risk import value_at_risk, conditional_var

    # Historical VaR at multiple confidence levels
    var_95_hist = value_at_risk(returns, confidence=0.95, method="historical")
    var_99_hist = value_at_risk(returns, confidence=0.99, method="historical")
    var_95_param = value_at_risk(returns, confidence=0.95, method="parametric")
    var_99_param = value_at_risk(returns, confidence=0.99, method="parametric")

    # CVaR (Expected Shortfall)
    cvar_95 = conditional_var(returns, confidence=0.95)
    cvar_99 = conditional_var(returns, confidence=0.99)

    print(f"\n  {'Method':<15} {'95% VaR':>10} {'99% VaR':>10}")
    print(f"  {'-' * 37}")
    print(f"  {'Historical':<15} {pct(var_95_hist):>10} {pct(var_99_hist):>10}")
    print(f"  {'Parametric':<15} {pct(var_95_param):>10} {pct(var_99_param):>10}")

    print(f"\n  {'Confidence':>12} {'CVaR (ES)':>12}   Meaning")
    print(f"  {'-' * 55}")
    print(f"  {'95%':>12} {pct(cvar_95):>12}   Avg loss on worst 5% of days")
    print(f"  {'99%':>12} {pct(cvar_99):>12}   Avg loss on worst 1% of days")

    # Dollar VaR for a hypothetical portfolio
    portfolio_value = 1_000_000
    print(f"\n  For a ${portfolio_value:,.0f} portfolio:")
    print(f"    95% daily VaR: ${portfolio_value * abs(var_95_hist):,.0f}")
    print(f"    99% daily VaR: ${portfolio_value * abs(var_99_hist):,.0f}")
    print(f"    99% daily CVaR: ${portfolio_value * abs(cvar_99):,.0f}")

    # ==================================================================
    # Step 3: Cornish-Fisher VaR (adjusts for skewness & kurtosis)
    # ==================================================================
    subheader("Cornish-Fisher Adjusted VaR")

    from wraquant.risk import cornish_fisher_var

    cf_var = cornish_fisher_var(returns, confidence=0.99)

    skew = float(returns.skew())
    kurt = float(returns.kurtosis())

    print(f"\n  Return distribution:")
    print(f"    Skewness:    {skew:>8.3f}  {'(negative = left tail heavier)' if skew < 0 else '(positive = right tail heavier)'}")
    print(f"    Kurtosis:    {kurt:>8.3f}  {'(fat tails)' if kurt > 3 else '(thin tails)'}")
    print(f"\n  99% CF VaR:    {pct(cf_var['var'])}")
    print(f"  99% Normal VaR: {pct(var_99_param)}")
    print(f"  CF adjustment:  {pct(abs(cf_var['var']) - abs(var_99_param))} wider")
    print(f"\n  The Cornish-Fisher correction accounts for non-normality.")
    print(f"  Standard VaR underestimates tail risk by ignoring fat tails.")

    # ==================================================================
    # Step 4: GARCH Volatility Modeling
    # ==================================================================
    header("STEP 3: GARCH VOLATILITY MODELING")

    from wraquant.vol import garch_fit, gjr_garch_fit, garch_forecast

    # Scale returns to percentage (arch library convention)
    returns_pct = returns * 100

    # Fit standard GARCH(1,1)
    garch_result = garch_fit(returns_pct, p=1, q=1, dist="normal")

    print(f"\n  GARCH(1,1) Model")
    print(f"    omega (constant):     {garch_result['params']['omega']:.6f}")
    print(f"    alpha (ARCH):         {garch_result['params']['alpha']:.4f}")
    print(f"    beta (GARCH):         {garch_result['params']['beta']:.4f}")
    print(f"    Persistence (a+b):    {garch_result['persistence']:.4f}")
    print(f"    Half-life:            {garch_result['half_life']:.1f} days")
    print(f"    AIC:                  {garch_result['aic']:.2f}")
    print(f"    BIC:                  {garch_result['bic']:.2f}")

    # Fit GJR-GARCH to capture leverage effect
    gjr_result = gjr_garch_fit(returns_pct, p=1, q=1, dist="normal")

    print(f"\n  GJR-GARCH(1,1) Model (captures leverage effect)")
    print(f"    alpha:                {gjr_result['params']['alpha']:.4f}")
    print(f"    gamma (asymmetry):    {gjr_result['params']['gamma']:.4f}")
    print(f"    beta:                 {gjr_result['params']['beta']:.4f}")
    print(f"    Persistence:          {gjr_result['persistence']:.4f}")
    print(f"    AIC:                  {gjr_result['aic']:.2f}")

    # Interpret leverage effect
    gamma = gjr_result["params"]["gamma"]
    if gamma > 0.01:
        print(f"\n  Leverage effect detected (gamma = {gamma:.4f}):")
        print(f"    Negative shocks increase vol more than positive shocks")
        print(f"    of the same magnitude -- asymmetric risk profile.")
    else:
        print(f"\n  Minimal leverage effect (gamma = {gamma:.4f}).")

    # Forecast volatility
    forecast = garch_forecast(garch_result["model"], horizon=10)

    print(f"\n  Volatility Forecast (annualized):")
    for h in [1, 5, 10]:
        if h <= len(forecast["variance_forecast"]):
            daily_var = forecast["variance_forecast"][h - 1]
            ann_vol_fcast = np.sqrt(daily_var * 252) / 100  # convert back from pct
            print(f"    {h:>2}d ahead:  {pct(ann_vol_fcast)}")

    # ==================================================================
    # Step 5: Stress Testing
    # ==================================================================
    header("STEP 4: STRESS TESTING")

    from wraquant.risk import stress_test_returns, historical_stress_test

    # User-defined scenario shocks
    scenarios = {
        "Mild correction": -0.02,
        "Moderate stress": -0.05,
        "Severe crash": -0.10,
        "Flash crash": -0.15,
    }

    stress_results = stress_test_returns(returns, scenarios)

    print(f"\n  Scenario Stress Tests (additive daily return shocks):")
    print(f"  {'Scenario':<20} {'Shock':>8} {'Stressed Mean':>14} {'Stressed VaR':>14}")
    print(f"  {'-' * 58}")

    for scenario_name, result in stress_results["scenarios"].items():
        print(
            f"  {scenario_name:<20} "
            f"{scenarios[scenario_name]:>+7.1%} "
            f"{result['stressed_mean']:>+13.4%} "
            f"{result['stressed_var_95']:>+13.4%}"
        )

    # Historical stress test -- replay known crises
    subheader("Historical Crisis Replay")
    hist_stress = historical_stress_test(returns)

    if hist_stress.get("periods_found"):
        for crisis_name, crisis_data in hist_stress["crises"].items():
            if crisis_data.get("cumulative_return") is not None:
                print(
                    f"  {crisis_name:<20} "
                    f"Return: {pct(crisis_data['cumulative_return']):>8}  "
                    f"Max DD: {pct(crisis_data.get('max_drawdown', 0)):>8}"
                )
    else:
        print("  No historical crisis periods overlap with the data window.")
        print("  (Data starts at 2021 -- GFC/dot-com periods not available.)")
        print("  Using scenario-based stress tests above as alternative.")

    # ==================================================================
    # Step 6: Regime Detection
    # ==================================================================
    header("STEP 5: MARKET REGIME DETECTION")

    from wraquant.regimes import fit_gaussian_hmm, regime_statistics

    # Fit a 2-state Gaussian HMM
    hmm_result = fit_gaussian_hmm(returns.values.reshape(-1, 1), n_states=2)

    print(f"\n  2-State Gaussian HMM:")
    print(f"    Model: Hidden Markov Model with Gaussian emissions")
    print(f"    States: 2 (bull/bear)")

    # Sort regimes by mean return (regime 0 = bull, regime 1 = bear)
    means = hmm_result["means"].flatten()
    stds = np.sqrt(hmm_result["covariances"].flatten())
    labels = hmm_result["states"]

    # Identify which state is bull (higher mean)
    bull_state = int(np.argmax(means))
    bear_state = 1 - bull_state

    print(f"\n    {'Regime':<12} {'Mean':>10} {'Std Dev':>10} {'Ann Vol':>10} {'Time %':>8}")
    print(f"    {'-' * 52}")
    for state, name in [(bull_state, "Bull"), (bear_state, "Bear")]:
        pct_time = np.mean(labels == state)
        ann_v = stds[state] * np.sqrt(252)
        print(
            f"    {name:<12} "
            f"{means[state]:>+9.4f} "
            f"{stds[state]:>10.4f} "
            f"{pct(ann_v):>10} "
            f"{pct(pct_time):>8}"
        )

    # Transition matrix
    trans = hmm_result["transition_matrix"]
    print(f"\n    Transition Matrix:")
    print(f"      {'':>12} {'To Bull':>10} {'To Bear':>10}")
    print(f"      {'From Bull':<12} {trans[bull_state, bull_state]:>9.3f} {trans[bull_state, bear_state]:>9.3f}")
    print(f"      {'From Bear':<12} {trans[bear_state, bull_state]:>9.3f} {trans[bear_state, bear_state]:>9.3f}")

    # Current regime
    current_state = labels[-1]
    current_name = "Bull" if current_state == bull_state else "Bear"
    print(f"\n    Current regime: {current_name}")

    # Expected regime durations
    bull_duration = 1 / (1 - trans[bull_state, bull_state])
    bear_duration = 1 / (1 - trans[bear_state, bear_state])
    print(f"    Expected bull duration: {bull_duration:.0f} days")
    print(f"    Expected bear duration: {bear_duration:.0f} days")

    # Compute regime-conditional Sharpe
    bull_returns = returns.values[labels == bull_state]
    bear_returns = returns.values[labels == bear_state]

    if len(bull_returns) > 10:
        bull_sharpe = (bull_returns.mean() / bull_returns.std()) * np.sqrt(252)
    else:
        bull_sharpe = float("nan")

    if len(bear_returns) > 10:
        bear_sharpe = (bear_returns.mean() / bear_returns.std()) * np.sqrt(252)
    else:
        bear_sharpe = float("nan")

    print(f"\n    Regime-conditional Sharpe:")
    print(f"      Bull: {bull_sharpe:>6.2f}")
    print(f"      Bear: {bear_sharpe:>6.2f}")

    # ==================================================================
    # Step 7: Performance Tearsheet
    # ==================================================================
    header("STEP 6: PERFORMANCE TEARSHEET")

    from wraquant.backtest import performance_summary

    perf = performance_summary(returns)

    print(f"\n  {'Metric':<25} {'Value':>12}")
    print(f"  {'-' * 39}")

    metric_labels = {
        "total_return": ("Total Return", True),
        "annualized_return": ("Annualized Return", True),
        "annualized_vol": ("Annualized Volatility", True),
        "sharpe": ("Sharpe Ratio", False),
        "sortino": ("Sortino Ratio", False),
        "max_drawdown": ("Max Drawdown", True),
        "calmar": ("Calmar Ratio", False),
    }

    for key, (label, is_pct) in metric_labels.items():
        if key in perf:
            val = perf[key]
            val_str = pct(val) if is_pct else fmt(val, 2)
            print(f"  {label:<25} {val_str:>12}")

    # ==================================================================
    # Summary
    # ==================================================================
    header("RISK ANALYSIS SUMMARY")

    print(f"\n  Asset:               {ticker}")
    print(f"  Period:              {returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"  Ann. Return:         {pct(ann_return)}")
    print(f"  Ann. Volatility:     {pct(ann_vol)}")
    print(f"  Sharpe Ratio:        {fmt(sr, 2)}")
    print(f"  99% VaR (hist):      {pct(var_99_hist)}")
    print(f"  99% CVaR:            {pct(cvar_99)}")
    print(f"  GARCH Persistence:   {fmt(garch_result['persistence'])}")
    print(f"  Current Regime:      {current_name}")

    print(f"\n  Key Findings:")
    findings = []
    if sr > 0.5:
        findings.append(f"  + Acceptable risk-adjusted returns (Sharpe {fmt(sr, 2)})")
    else:
        findings.append(f"  - Poor risk-adjusted returns (Sharpe {fmt(sr, 2)})")

    if garch_result["persistence"] > 0.95:
        findings.append(f"  ! High vol persistence ({fmt(garch_result['persistence'])}) -- shocks are long-lived")
    if gamma > 0.01:
        findings.append(f"  ! Leverage effect present -- crashes amplify volatility")
    if current_name == "Bear":
        findings.append(f"  ! Currently in bear regime -- elevated risk")
    else:
        findings.append(f"  + Currently in bull regime -- favorable conditions")

    for f in findings:
        print(f"    {f}")

    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Risk analysis workflow using wraquant.",
    )
    parser.add_argument(
        "ticker",
        nargs="?",
        default="SPY",
        help="Ticker symbol for the analysis (default: SPY)",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=756,
        help="Number of trading days to simulate (default: 756, ~3 years)",
    )
    args = parser.parse_args()

    run_risk_analysis(args.ticker, n_periods=args.periods)
