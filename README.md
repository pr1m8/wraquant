# wraquant

[![Tests](https://github.com/algebraicwealth/wraquant/actions/workflows/tests.yml/badge.svg)](https://github.com/algebraicwealth/wraquant/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/algebraicwealth/wraquant/branch/main/graph/badge.svg)](https://codecov.io/gh/algebraicwealth/wraquant)
[![Docs](https://readthedocs.org/projects/wraquant/badge/?version=latest)](https://wraquant.readthedocs.io)
[![PyPI](https://img.shields.io/pypi/v/wraquant)](https://pypi.org/project/wraquant/)
[![Python](https://img.shields.io/pypi/pyversions/wraquant)](https://pypi.org/project/wraquant/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**The ultimate quantitative finance toolkit for Python.**

1,000+ functions | 3,200+ tests | 24 modules | 265 TA indicators

## Features

- **Risk Management** -- VaR/CVaR, GARCH-VaR, beta estimation, factor risk, portfolio analytics, tail risk, stress testing, copulas, credit risk
- **Regime Detection** -- Gaussian HMM, Markov-switching, Kalman filter/smoother/UKF, regime scoring, labels, regime-aware portfolios
- **Volatility Modeling** -- Full GARCH family (EGARCH, GJR, FIGARCH, HARCH, APARCH), Hawkes processes, stochastic vol, realized vol estimators
- **Technical Analysis** -- 265 indicators across 19 modules (momentum, overlap, volume, trend, volatility, patterns, cycles, Fibonacci, support/resistance, exotic, and more)
- **Machine Learning** -- LSTM/GRU/Transformer (PyTorch), sklearn pipelines, walk-forward validation, SHAP importance, online regression
- **Derivatives Pricing** -- Black-Scholes, FBSDE solvers, characteristic function pricing (Heston, VG, NIG, CGMY), SABR, rough Bergomi, CIR, Vasicek
- **Portfolio Optimization** -- Mean-variance, risk parity, Black-Litterman, HRP, convex/linear/nonlinear optimization
- **Backtesting** -- Vectorized engine, 15+ performance metrics, walk-forward optimization, comprehensive tearsheets, regime-conditional sizing
- **Time Series** -- Auto-forecasting, SSA/EMD decomposition, ARIMA diagnostics, stochastic processes (OU, jump-diffusion), anomaly detection
- **Statistics** -- Robust stats, advanced distributions, distance correlation, copula selection, factor analysis, cointegration
- **Econometrics** -- Panel data, IV/2SLS, VAR/VECM, event studies, structural breaks
- **Causal Inference** -- Granger causality, IV with diagnostics, event studies, synthetic control, causal forests, mediation, RDD
- **Bayesian** -- Conjugate regression, stochastic vol MCMC, HMC, model comparison (WAIC/LOO), changepoint detection
- **Visualization** -- Interactive Plotly dashboards (portfolio, regime, risk, technical), 3D vol surfaces, correlation networks
- **And more** -- Forex, microstructure, execution algorithms, Levy processes, network analysis, parallel computing

## Quick Start

```bash
pip install wraquant
# Or with optional groups:
pip install wraquant[market-data,viz,risk,ml]
```

```python
import wraquant as wq

# Quick comprehensive analysis
report = wq.analyze(returns)

# Detect market regimes
regimes = wq.detect_regimes(returns, method="hmm", n_regimes=2)

# GARCH volatility forecasting
from wraquant.vol import garch_fit, garch_forecast
model = garch_fit(returns, p=1, q=1, dist="t")
forecast = garch_forecast(returns, horizon=10)

# Portfolio optimization with regime awareness
from wraquant.recipes import portfolio_construction_pipeline
portfolio = portfolio_construction_pipeline(returns_df, regime_aware=True)

# 265 technical indicators
from wraquant.ta import rsi, macd, bollinger_bands
signals = rsi(prices, period=14)
```

## Module Overview

| Module | Functions | Description |
|--------|-----------|-------------|
| `risk` | 95 | Risk management, VaR, beta, factor models, stress testing |
| `stats` | 79 | Statistical analysis, robust stats, distributions, correlation |
| `ta` | 265 | Technical analysis indicators (19 sub-modules) |
| `math` | 55 | Levy processes, networks, optimal stopping |
| `ts` | 51 | Time series forecasting, decomposition, anomaly detection |
| `price` | 50 | Derivatives pricing, FBSDEs, stochastic models |
| `viz` | 46 | Plotly dashboards and interactive charts |
| `ml` | 43 | Machine learning, deep learning, pipelines |
| `regimes` | 38 | Regime detection, scoring, Kalman filters |
| `backtest` | 37 | Backtesting engine, metrics, tearsheets |
| `vol` | 28 | GARCH family, Hawkes, stochastic volatility |
| `bayes` | 28 | Bayesian inference, MCMC, model comparison |

## Documentation

Full API documentation: [wraquant.readthedocs.io](https://wraquant.readthedocs.io)

## Development

```bash
pdm install -G dev
pdm run test          # Run tests
pdm run test-cov      # Tests with coverage
pdm run lint          # Lint with trunk
pdm run fmt           # Format
pdm run changelog     # Generate changelog
pdm run docs          # Build docs
```

## License

MIT
