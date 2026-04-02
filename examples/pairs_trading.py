"""Pairs trading workflow using wraquant.

This example demonstrates a complete pairs trading research pipeline:

    1. Generate synthetic price data for a universe of stocks
    2. Scan for cointegrated pairs using the Engle-Granger test
    3. Compute the spread and hedge ratio for the best pair
    4. Generate z-score-based trading signals
    5. Backtest the pairs strategy
    6. Analyze performance and risk metrics

Pairs trading is a market-neutral strategy that exploits temporary
divergences between cointegrated securities. When two prices move
together in the long run (cointegration), deviations from equilibrium
are expected to mean-revert, creating a statistical edge.

Uses wraquant.stats (cointegration), wraquant.data (transforms),
and wraquant.backtest (performance metrics).

Usage:
    python examples/pairs_trading.py
    python examples/pairs_trading.py --n-assets 10 --seed 123

Requirements:
    pip install wraquant  # core deps only
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_cointegrated_universe(
    n_assets: int = 8,
    n_days: int = 504,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a universe of synthetic stock prices.

    Creates some genuinely cointegrated pairs (sharing a common
    stochastic trend) and some independent series, so the scanner
    has both signal and noise to work with.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days, freq="B")

    tickers = ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON",
               "ZETA", "ETA", "THETA", "IOTA", "KAPPA"][:n_assets]

    prices = {}

    # Create 2 cointegrated pairs by sharing a common random walk

    # Pair 1: ALPHA and BETA share a common trend
    common_trend_1 = np.cumsum(rng.normal(0.0003, 0.015, n_days))
    prices["ALPHA"] = 100 * np.exp(common_trend_1 + rng.normal(0, 0.005, n_days))
    prices["BETA"] = 50 * np.exp(0.6 * common_trend_1 + rng.normal(0, 0.004, n_days))

    # Pair 2: GAMMA and DELTA share a different common trend
    if n_assets >= 4:
        common_trend_2 = np.cumsum(rng.normal(0.0001, 0.012, n_days))
        prices["GAMMA"] = 75 * np.exp(common_trend_2 + rng.normal(0, 0.006, n_days))
        prices["DELTA"] = 120 * np.exp(0.8 * common_trend_2 + rng.normal(0, 0.005, n_days))

    # Remaining tickers: independent random walks (not cointegrated)
    for ticker in tickers:
        if ticker not in prices:
            drift = rng.uniform(-0.0002, 0.0005)
            vol = rng.uniform(0.010, 0.025)
            prices[ticker] = 100 * np.exp(
                np.cumsum(rng.normal(drift, vol, n_days))
            )

    df = pd.DataFrame(prices, index=dates)
    return df[tickers]  # preserve order


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
# Main workflow
# ---------------------------------------------------------------------------

def run_pairs_trading(n_assets: int = 8, seed: int = 42) -> None:
    """Run the pairs trading research pipeline."""

    header("PAIRS TRADING RESEARCH")

    # ==================================================================
    # Step 1: Generate Universe
    # ==================================================================
    header("STEP 1: ASSET UNIVERSE")

    prices = generate_cointegrated_universe(n_assets=n_assets, seed=seed)

    print(f"\n  Universe: {len(prices.columns)} assets")
    print(f"  Period:   {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"  Days:     {len(prices)}")

    print(f"\n  {'Ticker':<10} {'Start':>8} {'End':>8} {'Return':>10} {'Ann Vol':>10}")
    print(f"  {'-' * 48}")
    for col in prices.columns:
        ret = (prices[col].iloc[-1] / prices[col].iloc[0]) - 1
        vol = prices[col].pct_change().std() * np.sqrt(252)
        print(f"  {col:<10} ${prices[col].iloc[0]:>6.2f} ${prices[col].iloc[-1]:>6.2f} {pct(ret):>10} {pct(vol):>10}")

    # ==================================================================
    # Step 2: Scan for Cointegrated Pairs
    # ==================================================================
    header("STEP 2: COINTEGRATION SCAN")

    from wraquant.stats import find_cointegrated_pairs, engle_granger

    print(f"\n  Testing all {n_assets * (n_assets - 1) // 2} pair combinations...")
    print(f"  Method: Engle-Granger two-step test")
    print(f"  Significance level: 5%")

    pairs = find_cointegrated_pairs(prices, significance=0.05)

    if not pairs:
        print("\n  No cointegrated pairs found at 5% significance.")
        print("  Try increasing the sample size or number of assets.")
        return

    print(f"\n  Found {len(pairs)} cointegrated pair(s):\n")
    print(f"  {'Pair':<20} {'p-value':>10} {'Hedge Ratio':>12} {'Significance':>14}")
    print(f"  {'-' * 58}")

    for asset1, asset2, pval, hr in pairs:
        sig_level = "***" if pval < 0.01 else "**" if pval < 0.05 else "*"
        print(f"  {asset1 + '/' + asset2:<20} {pval:>10.6f} {hr:>12.4f} {sig_level:>14}")

    # Select the best pair (lowest p-value)
    best_pair = pairs[0]
    asset1, asset2, best_pval, best_hr = best_pair
    print(f"\n  Best pair: {asset1}/{asset2} (p-value: {best_pval:.6f})")

    # ==================================================================
    # Step 3: Spread Analysis
    # ==================================================================
    header("STEP 3: SPREAD ANALYSIS")

    from wraquant.stats import spread, half_life, zscore_signal, hedge_ratio

    # Compute the OLS hedge ratio
    hr = hedge_ratio(prices[asset1], prices[asset2], method="ols")
    print(f"\n  Hedge ratio ({asset1} = {hr:.4f} * {asset2} + alpha):")
    print(f"    To be dollar-neutral: for every $1 long {asset1},")
    print(f"    short ${abs(hr):.4f} of {asset2}.")

    # Compute the spread
    sprd = spread(prices[asset1], prices[asset2], hedge_ratio=hr)
    hl = half_life(sprd)

    print(f"\n  Spread Statistics:")
    print(f"    Mean:       {sprd.mean():>10.4f}")
    print(f"    Std Dev:    {sprd.std():>10.4f}")
    print(f"    Min:        {sprd.min():>10.4f}")
    print(f"    Max:        {sprd.max():>10.4f}")
    print(f"    Half-life:  {hl:>10.1f} days")

    if hl < 5:
        print(f"\n  Half-life < 5 days: very fast mean reversion.")
        print(f"  Suitable for high-frequency pairs trading.")
    elif hl < 30:
        print(f"\n  Half-life {hl:.0f} days: moderate mean reversion speed.")
        print(f"  Good for daily/weekly rebalancing strategies.")
    elif hl < 120:
        print(f"\n  Half-life {hl:.0f} days: slow mean reversion.")
        print(f"  Requires patience; larger capital at risk.")
    else:
        print(f"\n  Half-life > 120 days: very slow reversion.")
        print(f"  May not be practically tradeable.")

    # Compute the z-score
    zscore = zscore_signal(sprd, window=20)

    subheader("Z-Score Distribution")
    print(f"    Current z-score:    {zscore.dropna().iloc[-1]:>8.3f}")
    print(f"    Mean:               {zscore.dropna().mean():>8.3f}")
    print(f"    Std Dev:            {zscore.dropna().std():>8.3f}")
    print(f"    % above +2:        {pct((zscore.dropna() > 2).mean()):>8}")
    print(f"    % below -2:        {pct((zscore.dropna() < -2).mean()):>8}")

    # ==================================================================
    # Step 4: Generate Trading Signals
    # ==================================================================
    header("STEP 4: TRADING SIGNALS")

    from wraquant.stats import pairs_backtest_signals

    entry_z = 2.0
    exit_z = 0.5

    print(f"\n  Signal Rules:")
    print(f"    LONG spread:   z-score < -{entry_z}")
    print(f"    SHORT spread:  z-score > +{entry_z}")
    print(f"    EXIT:          z-score crosses back inside [{-exit_z}, +{exit_z}]")
    print(f"\n    'Long spread' means: buy {asset1}, sell {asset2}")
    print(f"    'Short spread' means: sell {asset1}, buy {asset2}")

    signals = pairs_backtest_signals(sprd, entry_z=entry_z, exit_z=exit_z)

    n_long = (signals == 1).sum()
    n_short = (signals == -1).sum()
    n_flat = (signals == 0).sum()

    print(f"\n  Signal Distribution:")
    print(f"    Long periods:   {n_long:>5} days ({pct(n_long / len(signals))})")
    print(f"    Short periods:  {n_short:>5} days ({pct(n_short / len(signals))})")
    print(f"    Flat periods:   {n_flat:>5} days ({pct(n_flat / len(signals))})")

    # Count trades (signal changes)
    signal_changes = signals.diff().abs()
    n_trades = int(signal_changes.sum() // 2)  # each round-trip is 2 changes
    print(f"    Round-trip trades: {n_trades}")

    # ==================================================================
    # Step 5: Backtest the Strategy
    # ==================================================================
    header("STEP 5: BACKTEST RESULTS")

    # Compute strategy returns
    # The return of the pairs trade = signal * spread return
    spread_returns = sprd.pct_change().fillna(0)
    strategy_returns = (signals.shift(1) * spread_returns).dropna()
    strategy_returns.name = "pairs_strategy"

    # Also compute buy-and-hold returns for comparison
    asset1_returns = prices[asset1].pct_change().dropna()

    from wraquant.backtest import performance_summary

    strategy_perf = performance_summary(strategy_returns)
    buyhold_perf = performance_summary(asset1_returns.loc[strategy_returns.index])

    print(f"\n  {'Metric':<25} {'Pairs Strategy':>16} {f'Buy & Hold {asset1}':>16}")
    print(f"  {'-' * 59}")

    metrics = [
        ("Total Return", "total_return", True),
        ("Annualized Return", "annualized_return", True),
        ("Annualized Vol", "annualized_vol", True),
        ("Sharpe Ratio", "sharpe", False),
        ("Sortino Ratio", "sortino", False),
        ("Max Drawdown", "max_drawdown", True),
        ("Calmar Ratio", "calmar", False),
    ]

    for label, key, is_pct in metrics:
        s_val = strategy_perf.get(key, 0)
        b_val = buyhold_perf.get(key, 0)
        s_str = pct(s_val) if is_pct else fmt(s_val, 2)
        b_str = pct(b_val) if is_pct else fmt(b_val, 2)
        print(f"  {label:<25} {s_str:>16} {b_str:>16}")

    # Additional strategy-specific metrics
    subheader("Strategy Diagnostics")

    # Win rate
    winning_days = (strategy_returns > 0).sum()
    total_days = (strategy_returns != 0).sum()
    win_rate = winning_days / total_days if total_days > 0 else 0

    # Average win vs average loss
    wins = strategy_returns[strategy_returns > 0]
    losses = strategy_returns[strategy_returns < 0]
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    payoff = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    print(f"    Win Rate:            {pct(win_rate)}")
    print(f"    Average Win:         {pct(avg_win)}")
    print(f"    Average Loss:        {pct(avg_loss)}")
    print(f"    Payoff Ratio:        {fmt(payoff, 2)}x")
    print(f"    Round-trip Trades:   {n_trades}")

    # Time in market
    time_in_market = (signals != 0).mean()
    print(f"    Time in Market:      {pct(time_in_market)}")

    # Equity curve stats
    equity = (1 + strategy_returns).cumprod()
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak

    print(f"\n    Final Equity (start=1):  {equity.iloc[-1]:.4f}")
    print(f"    Worst Drawdown:          {pct(drawdown.min())}")
    dd_duration = 0
    current_dd = 0
    max_dd_dur = 0
    for dd in drawdown:
        if dd < 0:
            current_dd += 1
            max_dd_dur = max(max_dd_dur, current_dd)
        else:
            current_dd = 0
    print(f"    Longest DD Duration:     {max_dd_dur} days")

    # ==================================================================
    # Step 6: Out-of-Sample Considerations
    # ==================================================================
    header("STEP 6: ROBUSTNESS CHECKS")

    print(f"\n  Before deploying a pairs strategy, verify:")
    print(f"\n  1. Rolling Cointegration")
    # Test cointegration on second half of data
    midpoint = len(prices) // 2
    prices_oos = prices.iloc[midpoint:]
    oos_result = engle_granger(prices_oos[asset1], prices_oos[asset2])
    print(f"     Second-half p-value: {oos_result['p_value']:.6f}")
    print(f"     Still cointegrated?  {'YES' if oos_result['is_cointegrated'] else 'NO'}")

    print(f"\n  2. Hedge Ratio Stability")
    hr_first_half = hedge_ratio(prices.iloc[:midpoint][asset1],
                                prices.iloc[:midpoint][asset2])
    hr_second_half = hedge_ratio(prices_oos[asset1], prices_oos[asset2])
    print(f"     First half:  {hr_first_half:.4f}")
    print(f"     Second half: {hr_second_half:.4f}")
    print(f"     Drift:       {abs(hr_second_half - hr_first_half):.4f}")

    print(f"\n  3. Half-Life Consistency")
    sprd_oos = spread(prices_oos[asset1], prices_oos[asset2])
    hl_oos = half_life(sprd_oos)
    print(f"     In-sample:   {hl:.1f} days")
    print(f"     Out-of-sample: {hl_oos:.1f} days")

    print(f"\n  4. Transaction Costs")
    # Estimate round-trip cost impact
    cost_per_trade = 0.001  # 10 bps per side
    total_cost = n_trades * 2 * cost_per_trade  # 2 legs per trade
    ann_cost = total_cost * (252 / len(strategy_returns))
    strat_ann_return = strategy_perf.get("annualized_return", 0)
    net_return = strat_ann_return - ann_cost
    print(f"     Estimated cost per trade: {pct(cost_per_trade)} (per leg)")
    print(f"     Total round-trips:        {n_trades}")
    print(f"     Annual cost drag:         {pct(ann_cost)}")
    print(f"     Gross ann. return:        {pct(strat_ann_return)}")
    print(f"     Net ann. return:          {pct(net_return)}")

    profitable = net_return > 0
    print(f"\n     Profitable after costs?   {'YES' if profitable else 'NO'}")

    # ==================================================================
    # Summary
    # ==================================================================
    header("PAIRS TRADING SUMMARY")

    print(f"\n  Best Pair:           {asset1}/{asset2}")
    print(f"  Cointegration p:     {best_pval:.6f}")
    print(f"  Hedge Ratio:         {best_hr:.4f}")
    print(f"  Half-Life:           {hl:.1f} days")
    print(f"  Entry/Exit Z:        +/-{entry_z:.1f} / +/-{exit_z:.1f}")
    print(f"  Sharpe Ratio:        {strategy_perf.get('sharpe', 0):.2f}")
    print(f"  Max Drawdown:        {pct(strategy_perf.get('max_drawdown', 0))}")
    print(f"  Round-trip Trades:   {n_trades}")
    print(f"  OOS Cointegrated:    {'YES' if oos_result['is_cointegrated'] else 'NO'}")

    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pairs trading workflow using wraquant.",
    )
    parser.add_argument(
        "--n-assets",
        type=int,
        default=8,
        help="Number of assets in the synthetic universe (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    run_pairs_trading(n_assets=args.n_assets, seed=args.seed)
