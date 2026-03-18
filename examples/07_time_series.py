"""Time series analysis with wraquant.

Demonstrates decomposition, stationarity testing, changepoint detection,
forecasting, and feature extraction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 500
dates = pd.date_range("2020-01-01", periods=n, freq="D")

# Synthetic time series: trend + seasonality + noise
trend = np.linspace(0, 2, n)
seasonal = 0.5 * np.sin(2 * np.pi * np.arange(n) / 30)
noise = rng.normal(0, 0.2, n)
ts = pd.Series(trend + seasonal + noise, index=dates, name="signal")

# --- Decomposition ---
from wraquant.ts.decomposition import stl_decompose, trend_filter

print("=== STL Decomposition ===")
decomp = stl_decompose(ts, period=30)
print(f"  Trend range: [{decomp['trend'].min():.2f}, {decomp['trend'].max():.2f}]")
print(f"  Seasonal amplitude: {decomp['seasonal'].std():.4f}")
print(f"  Residual std: {decomp['residual'].std():.4f}")

# HP filter
hp = trend_filter(ts.values, method="hp", lamb=1600)
print(f"\n  HP trend final value: {hp[-1]:.4f}")

# --- Stationarity ---
from wraquant.ts.stationarity import difference, detrend

diffed = difference(ts.values, order=1)
print(f"\n=== Stationarity ===")
print(f"  Original mean: {ts.values.mean():.4f}")
print(f"  Differenced mean: {diffed.mean():.4f}")

detrended = detrend(ts.values, method="linear")
print(f"  Detrended std: {detrended.std():.4f}")

# --- Changepoint detection ---
from wraquant.ts.changepoint import cusum_detect

# Create data with a level shift
shifted = ts.copy()
shifted.iloc[250:] += 2.0

cp = cusum_detect(shifted.values, threshold=2.0)
print(f"\n=== Changepoint Detection ===")
print(f"  Detected changepoints: {cp['changepoints']}")
print(f"  True changepoint: 250")

# --- Seasonality ---
from wraquant.ts.seasonality import detect_seasonality, seasonal_strength

print(f"\n=== Seasonality ===")
seasons = detect_seasonality(ts.values, max_period=60)
print(f"  Detected period: {seasons['dominant_period']}")
print(f"  Strength: {seasons['strength']:.4f}")

# --- Forecasting (statsmodels ARIMA) ---
from wraquant.ts.forecasting import arima_forecast

print(f"\n=== ARIMA Forecast ===")
# Use the differenced series (stationary)
forecast = arima_forecast(ts.values[-100:], order=(2, 1, 1), horizon=10)
print(f"  Next 10 forecasts: {np.round(forecast['forecast'], 3)}")
