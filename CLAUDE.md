# wraquant — Development Guide

## Overview

Comprehensive quant finance Python package. PDM for package management, Trunk for linting.

## Commands

```bash
pdm install                    # Install core deps
pdm install -G dev             # Install dev deps
pdm install -G market-data     # Install a specific extra group
pdm run test                   # Run tests
pdm run test-cov               # Run tests with coverage
pdm run lint                   # Lint with trunk
pdm run fmt                    # Format with trunk
pdm run changelog              # Generate changelog with git-cliff
```

## Project Structure

```text
src/wraquant/
├── core/        # Config, types, exceptions, logging, decorators
├── _lazy.py     # Lazy import infrastructure
├── _compat.py   # Backend detection (pandas/polars/torch)
├── frame/       # Unified DataFrame/Series abstraction
├── data/        # Data fetching (yfinance, FRED, NASDAQ)
├── ts/          # Time series analysis
├── stats/       # Statistical analysis
├── vol/         # Volatility modeling (GARCH, ARCH, stochastic vol)
├── ta/          # Technical analysis indicators
├── ml/          # Machine learning for finance
├── opt/         # Portfolio & mathematical optimization
├── price/       # Options pricing, fixed income, SDEs
├── regimes/     # Regime detection (HMM, Kalman)
├── risk/        # Risk management (VaR, copulas, EVT)
├── backtest/    # Backtesting engines
├── forex/       # Forex-specific analysis
├── viz/         # Visualization
├── math/        # JAX, symbolic, PDE solvers
├── bayes/       # Bayesian inference
├── io/          # ETL, SQL, cloud storage
├── flow/        # Workflow orchestration
└── scale/       # Distributed computing (dask, ray)
```

## Conventions

- **Commits**: Conventional commits (`feat(module):`, `fix(module):`, `chore:`)
- **Imports**: Lazy imports for all optional deps via `_lazy.py`
- **Decorators**: Use `@requires_extra('group-name')` for optional dep functions
- **Types**: Use type hints everywhere, pydantic for config/validation
- **Logging**: structlog via `core/logging.py`
- **Testing**: pytest + hypothesis, each module has its own test dir
- **Linting**: trunk check (ruff, black, isort, bandit)
- **Docstrings**: Google style with Parameters/Returns/Example sections
- **Exports**: Every module defines `__all__` in `__init__.py`

## Dependency Groups

Core deps always available. Optional groups installed via `pdm install -G <group>`:
market-data, timeseries, cleaning, validation, etl, warehouse, ingestion,
workflow, profiling, dev, optimization, regimes, pde, backtesting, risk,
pricing, stochastic, lp-extra, conic-extra, nlp-extra, causal, quant-math,
bayes, viz, scale
