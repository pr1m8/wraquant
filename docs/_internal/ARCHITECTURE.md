# wraquant Architecture

## Module Boundaries

### Statistical Analysis Stack
- **stats/** — Static statistical tests, distributions, correlation, regression
- **ts/** — Time-dependent analysis: decomposition, forecasting, stationarity, changepoints
- **econometrics/** — Advanced econometric models: panel data, IV/2SLS, VAR/VECM, event studies
- **vol/** — Volatility modeling: GARCH family, realized vol, stochastic vol, Hawkes
  - Canonical home for ALL GARCH models (econometrics/ delegates here)

### Risk & Portfolio Stack
- **risk/** — Portfolio-level risk: VaR, CVaR, copulas, stress testing, credit
- **opt/** — Portfolio optimization: MVO, risk parity, BL, HRP
- **regimes/** — Regime detection: HMM, Markov-switching, Kalman, GMM
  - Feeds into opt/ via regime_aware_portfolio()

### Trading Stack
- **ta/** — Technical analysis indicators (263 functions, 19 sub-modules)
- **backtest/** — Backtesting engine, metrics, position sizing, tearsheets
  - Imports risk metrics from risk/ (single source of truth)
- **execution/** — Execution algorithms: TWAP, VWAP, Almgren-Chriss
- **microstructure/** — Market microstructure: liquidity, toxicity, market quality

### Pricing Stack
- **price/** — Options pricing, fixed income, SDEs, FBSDEs, characteristic functions
- **forex/** — FX-specific analysis

### ML Stack
- **ml/** — Feature engineering, preprocessing, models, deep learning, online learning
  - ml/features wraps ta/ indicators for ML pipelines

### Infrastructure
- **core/** — Config, types, exceptions, decorators
- **data/** — Data fetching, cleaning, validation
- **io/** — ETL, SQL, cloud storage
- **flow/** — Workflow orchestration (Prefect, Pipeline)
- **scale/** — Distributed computing (Dask, Ray)
- **bayes/** — Bayesian inference (PyMC, emcee, blackjax)
- **viz/** — Visualization and dashboards (Plotly)
- **math/** — Advanced math (Levy, networks, optimal stopping)

## Integration Patterns
- backtest/metrics -> risk/metrics (single source of truth for Sharpe, Sortino, max_drawdown)
- ta/volatility -> vol/realized (delegates OHLC volatility estimators)
- econometrics/volatility -> vol/models (delegates GARCH family)
- regimes/ -> opt/ (regime_aware_portfolio bridges detection and optimization)
- ml/features -> ta/ (wraps indicators for ML pipelines)
- regimes/ -> backtest/ (regime_conditional_sizing for position management)

## Parameter Conventions
- Rolling window: `period` (ta/, primary) or `window` (vol/, risk/)
- Return series input: `returns` (dominant, 180+ uses)
- Random seed: `seed` (simple) or `random_state` (sklearn convention)
- Risk-free rate: `risk_free` (dominant in risk/, backtest/)
- Annualization: `periods_per_year=252` (trading days)
