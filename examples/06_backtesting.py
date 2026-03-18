"""Backtesting with wraquant.

Demonstrates the backtest engine, strategy creation, position sizing,
event tracking, and performance analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 504
dates = pd.date_range("2020-01-01", periods=n, freq="D")
prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n))), index=dates, name="price")
returns = prices.pct_change().dropna()

# --- Backtest engine ---
from wraquant.backtest.engine import Backtest
from wraquant.backtest.strategy import MovingAverageCrossover

print("=== Moving Average Crossover Backtest ===")
strategy = MovingAverageCrossover(fast=20, slow=50)
bt = Backtest(strategy)
result = bt.run(prices)
print(f"  Total return: {result['total_return']:.4f}")
print(f"  Sharpe ratio: {result['sharpe_ratio']:.4f}")
print(f"  Max drawdown: {result['max_drawdown']:.4f}")
print(f"  Total trades: {result['n_trades']}")
print(f"  Win rate: {result['win_rate']:.2%}")

# --- Position sizing ---
from wraquant.backtest.position import PositionSizer

sizer = PositionSizer()

print(f"\n=== Position Sizing ===")
fixed = sizer.fixed_fraction(capital=100_000, risk_per_trade=0.02, stop_distance=0.05)
print(f"  Fixed fraction (2% risk, 5% stop): {fixed:.0f} units")

kelly = sizer.kelly_criterion(win_rate=0.55, win_loss_ratio=1.5)
print(f"  Kelly fraction: {kelly:.4f}")

vol_target = sizer.volatility_targeting(
    returns=returns.values, target_vol=0.15, current_vol=returns.std() * np.sqrt(252),
)
print(f"  Vol-targeting leverage: {vol_target:.4f}")

# --- Event tracking ---
from wraquant.backtest.events import EventTracker

tracker = EventTracker()
tracker.log_trade(date=dates[50], side="buy", price=prices.iloc[50], quantity=100)
tracker.log_trade(date=dates[100], side="sell", price=prices.iloc[100], quantity=100)
tracker.log_signal(date=dates[45], signal="MA_cross_up", value=1.0)
tracker.log_risk_event(date=dates[80], event_type="drawdown", details={"pct": -0.05})

print(f"\n=== Event Log ===")
print(f"  Trades: {len(tracker.trades)}")
print(f"  Signals: {len(tracker.signals)}")
print(f"  Risk events: {len(tracker.risk_events)}")

# --- Tearsheet ---
from wraquant.backtest.tearsheet import generate_tearsheet, monthly_returns_table

print(f"\n=== Tearsheet ===")
tear = generate_tearsheet(returns)
print(f"  Annual return: {tear['annual_return']:.4f}")
print(f"  Annual vol: {tear['annual_volatility']:.4f}")
print(f"  Sharpe: {tear['sharpe_ratio']:.4f}")
print(f"  Sortino: {tear['sortino_ratio']:.4f}")
print(f"  Max DD: {tear['max_drawdown']:.4f}")
print(f"  Calmar: {tear['calmar_ratio']:.4f}")

monthly = monthly_returns_table(returns)
print(f"\n  Monthly returns table shape: {monthly.shape}")
