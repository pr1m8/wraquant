"""Statistical analysis with wraquant.

Demonstrates descriptive statistics, hypothesis tests, regression,
distributions, and correlation analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Generate sample data ---
rng = np.random.default_rng(42)
n = 500
dates = pd.date_range("2020-01-01", periods=n, freq="D")
returns = pd.Series(rng.normal(0.0005, 0.02, n), index=dates, name="strategy")
benchmark = pd.Series(rng.normal(0.0003, 0.018, n), index=dates, name="benchmark")

# --- Descriptive statistics ---
from wraquant.stats.descriptive import summary_stats, drawdown_analysis

stats = summary_stats(returns.values)
print("=== Summary Statistics ===")
for key, val in stats.items():
    print(f"  {key}: {val:.6f}")

dd = drawdown_analysis(returns.values)
print(f"\nMax drawdown: {dd['max_drawdown']:.4f}")
print(f"Max drawdown duration: {dd['max_duration']} periods")

# --- Statistical tests ---
from wraquant.stats.tests import adf_test, jarque_bera_test, ljung_box_test

adf = adf_test(returns.values)
print(f"\n=== ADF Test ===")
print(f"  Statistic: {adf['statistic']:.4f}, p-value: {adf['p_value']:.4f}")
print(f"  Stationary: {adf['stationary']}")

jb = jarque_bera_test(returns.values)
print(f"\n=== Jarque-Bera Test ===")
print(f"  Statistic: {jb['statistic']:.4f}, p-value: {jb['p_value']:.4f}")
print(f"  Normal: {jb['normal']}")

# --- Regression ---
from wraquant.stats.regression import ols

X = benchmark.values.reshape(-1, 1)
y = returns.values
result = ols(X, y)
print(f"\n=== OLS Regression (strategy ~ benchmark) ===")
print(f"  Alpha: {result['intercept']:.6f}")
print(f"  Beta: {result['coefficients'][0]:.4f}")
print(f"  R-squared: {result['r_squared']:.4f}")

# --- Correlation ---
from wraquant.stats.correlation import correlation_matrix, shrunk_covariance

data = pd.DataFrame({"A": returns, "B": benchmark, "C": rng.normal(0, 0.02, n)})
corr = correlation_matrix(data.values)
print(f"\n=== Correlation Matrix ===")
print(corr)

shrunk = shrunk_covariance(data.values, shrinkage=0.5)
print(f"\nShrunk covariance (diagonal): {np.diag(shrunk)}")

# --- Distribution fitting ---
from wraquant.stats.distributions import fit_distribution, qqplot_data

fit = fit_distribution(returns.values, distribution="norm")
print(f"\n=== Normal Fit ===")
print(f"  Mean: {fit['params'][0]:.6f}, Std: {fit['params'][1]:.6f}")
print(f"  KS statistic: {fit['ks_statistic']:.4f}")
