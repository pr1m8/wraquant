"""Getting started with wraquant.

Demonstrates core configuration, data types, and basic frame operations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import wraquant as wq

# --- Configuration ---
cfg = wq.get_config()
print(f"Backend: {cfg.backend}")
print(f"Precision: {cfg.precision}")
print(f"Cache dir: {cfg.cache_dir}")

# --- Frame operations ---
from wraquant.frame import ops

dates = pd.date_range("2020-01-01", periods=252, freq="D")
prices = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(42).normal(0.0003, 0.015, 252))), index=dates)

# Compute returns
simple_returns = ops.returns(prices)
log_returns = ops.log_returns(prices)
print(f"\nSimple returns (first 5): {simple_returns.head().values}")
print(f"Log returns (first 5): {log_returns.head().values}")

# Rolling statistics
rolling_vol = ops.rolling(simple_returns, window=20, func="std") * np.sqrt(252)
print(f"\nAnnualized rolling 20-day vol (last value): {rolling_vol.iloc[-1]:.4f}")

# --- Core types ---
print(f"\nAsset classes: {[ac.value for ac in wq.AssetClass]}")
print(f"Frequencies: {[f.value for f in wq.Frequency]}")
print(f"Option types: {[ot.value for ot in wq.OptionType]}")
