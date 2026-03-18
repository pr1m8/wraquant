"""Forex analysis with wraquant.

Demonstrates currency pairs, pip calculations, trading sessions,
carry trade analysis, and forex-specific risk.
"""

from __future__ import annotations

import numpy as np

# --- Currency pairs ---
from wraquant.forex.pairs import CurrencyPair, cross_rate, triangular_arbitrage

print("=== Currency Pairs ===")
eurusd = CurrencyPair("EUR", "USD")
print(f"  Pair: {eurusd}")
print(f"  Base: {eurusd.base}, Quote: {eurusd.quote}")

# Cross rate calculation
eurusd_rate = 1.0850
usdjpy_rate = 149.50
eurjpy = cross_rate(eurusd_rate, usdjpy_rate, cross_type="multiply")
print(f"\n  EURUSD: {eurusd_rate}, USDJPY: {usdjpy_rate}")
print(f"  EURJPY (cross): {eurjpy:.2f}")

# Triangular arbitrage
arb = triangular_arbitrage(
    ab_rate=eurusd_rate, bc_rate=usdjpy_rate, ac_rate=162.10,
)
print(f"\n  Triangular arb profit: {arb['profit']:.4f}")
print(f"  Arbitrage exists: {arb['is_arbitrage']}")

# --- Pip calculations ---
from wraquant.forex.analysis import pip_value, lot_size, pip_distance

print(f"\n=== Pip Calculations ===")
pv = pip_value(pair="EURUSD", lot_size=100_000, exchange_rate=eurusd_rate)
print(f"  EURUSD pip value (1 std lot): ${pv:.2f}")

pips = pip_distance(entry=1.0850, exit=1.0920, pair="EURUSD")
print(f"  Pips from 1.0850 to 1.0920: {pips}")

ls = lot_size(
    account_balance=50_000, risk_pct=0.02, stop_loss_pips=30,
    pip_value_per_lot=10.0,
)
print(f"  Lot size (2% risk, 30 pip stop): {ls:.2f}")

# --- Trading sessions ---
from wraquant.forex.session import get_session_times, is_session_overlap

print(f"\n=== Trading Sessions ===")
for session_name in ["london", "new_york", "tokyo", "sydney"]:
    times = get_session_times(session_name)
    print(f"  {session_name.title()}: {times['open']} - {times['close']} UTC")

overlap = is_session_overlap("london", "new_york")
print(f"\n  London/NY overlap: {overlap}")

# --- Carry trade ---
from wraquant.forex.carry import carry_return, carry_trade_pnl

print(f"\n=== Carry Trade ===")
carry = carry_return(
    base_rate=0.05, quote_rate=0.005, holding_period=90,
)
print(f"  90-day carry (5% vs 0.5%): {carry['annualized_carry']:.4f}")
print(f"  90-day carry income: {carry['carry_income']:.4f}")

# --- Forex risk ---
from wraquant.forex.risk import margin_requirement, leverage_ratio, position_risk

print(f"\n=== Forex Risk ===")
margin = margin_requirement(position_size=100_000, leverage=50)
print(f"  Margin required (50:1 leverage): ${margin:,.0f}")

pos_risk = position_risk(
    position_size=100_000, entry_price=1.0850, stop_price=1.0820,
)
print(f"  Position risk: ${pos_risk['risk_amount']:.2f}")
print(f"  Risk in pips: {pos_risk['risk_pips']}")
