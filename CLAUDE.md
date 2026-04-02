# wraquant — Development Guide

## Overview

The ultimate quant finance Python package. 27 modules, 3630+ tests, 265 TA indicators, 55-method FMP client, 44 fundamental functions, 218 MCP tools, 12-page Streamlit dashboard. PDM for package management, Trunk for linting.

## Commands

```bash
pdm install                    # Install core deps
pdm install -G dev             # Install dev deps
pdm install -G dashboard       # Install dashboard (streamlit, plotly, option-menu)
pdm install -G market-data     # Install a specific extra group
pdm run test                   # Run tests
pdm run test-cov               # Run tests with coverage
pdm run test-mcp               # Run MCP tests
pdm run lint                   # Lint with trunk
pdm run fmt                    # Format with trunk
pdm run changelog              # Generate changelog with git-cliff
pdm run dashboard              # Launch Streamlit dashboard
pdm run mcp-server             # Launch MCP server
```

## Project Structure

```text
src/wraquant/
├── core/            # Config, types, exceptions, logging, decorators
├── _lazy.py         # Lazy import infrastructure + @requires_extra
├── _compat.py       # Backend detection (pandas/polars/torch)
├── frame/           # Unified DataFrame/Series abstraction
├── data/            # Data fetching (yfinance, FRED, NASDAQ) + cleaning/validation
├── ts/              # Time series (decomposition, forecasting, changepoints, wavelets)
├── stats/           # Statistical analysis (regression, correlation, distributions, cointegration)
├── vol/             # Volatility modeling — DEEP:
│                    #   GARCH/EGARCH/GJR/FIGARCH/HARCH with full diagnostics,
│                    #   DCC multivariate, Hawkes processes, stochastic vol,
│                    #   news impact curves, realized vol estimators
├── ta/              # Technical analysis — 263 indicators across 19 modules:
│                    #   overlap, momentum, volume, trend, volatility, patterns,
│                    #   signals, statistics, cycles, custom, fibonacci,
│                    #   support_resistance, breadth, performance, smoothing,
│                    #   exotic, candles, price_action
├── ml/              # Machine learning — sklearn + torch:
│                    #   features, preprocessing, walk-forward, ensembles,
│                    #   LSTM/GRU/Transformer, SVM, GP, online regression
├── opt/             # Portfolio optimization (MVO, risk parity, BL, HRP)
├── price/           # Options pricing, fixed income, SDEs, Lévy processes
├── regimes/         # Regime detection — DEEP:
│                    #   Gaussian HMM, Markov-switching, GMM regimes,
│                    #   Kalman filter/smoother/UKF, regime-aware portfolios
├── risk/            # Risk management (VaR, copulas, EVT, stress testing, DCC, credit)
├── backtest/        # Backtesting (engine, strategies, position sizing, tearsheets)
├── econometrics/    # Panel data, IV/2SLS, event studies, structural breaks
├── microstructure/  # Market microstructure (liquidity, toxicity, market quality)
├── execution/       # Execution algorithms (TWAP, VWAP, Almgren-Chriss)
├── causal/          # Causal inference (DID, synthetic control, IPW)
├── forex/           # Forex analysis (pairs, sessions, carry, risk)
├── viz/             # Visualization — interactive Plotly dashboards:
│                    #   portfolio/regime/risk/technical dashboards,
│                    #   vol surface, correlation network, tearsheets
├── math/            # Advanced math (Lévy, networks, optimal stopping, PDEs)
├── bayes/           # Bayesian inference (PyMC, emcee, blackjax, NumPyro)
├── experiment/      # Experiment tracking
├── io/              # ETL, SQL, cloud storage
├── flow/            # Workflow orchestration (Prefect, APScheduler, Pipeline)
└── scale/           # Distributed computing (dask, ray, parallel backtest)
```

## Conventions

- **Commits**: Conventional commits (`feat(module):`, `fix(module):`, `chore:`)
- **Imports**: Lazy imports for all optional deps via `_lazy.py`
- **Decorators**: Use `@requires_extra('group-name')` for optional dep functions
- **Types**: Use type hints everywhere, `from __future__ import annotations` in every file
- **Logging**: structlog via `core/logging.py`
- **Testing**: pytest + hypothesis, each module has its own test dir
- **Linting**: trunk check (ruff, black, isort, bandit)
- **Exports**: Every module defines `__all__` in `__init__.py`

## Docstring Standard

Google/Napoleon style. Every public function MUST have:

```python
def function_name(param: type) -> ReturnType:
    """One-line summary of what it does.

    Longer description explaining WHEN to use this function,
    what problem it solves, and how it fits into the workflow.

    Mathematical formulation (if applicable):
    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

    Parameters:
        param: Description with type info and valid values.
            Include guidance on how to choose this parameter.

    Returns:
        Dictionary containing:
        - **key1** (*type*) — What it means and how to interpret it.
        - **key2** (*type*) — What values indicate problems.

    Example:
        >>> from wraquant.module import function_name
        >>> result = function_name(data, param=value)
        >>> print(f"Key metric: {result['key']:.4f}")

    Notes:
        Reference: Author (Year). "Title." *Journal*, vol, pages.

    See Also:
        related_function: When to use that instead.
    """
```

## Module Integration Patterns

- **vol/ ↔ risk/**: ta/volatility delegates to vol/realized for OHLC estimators
- **backtest/ → risk/**: backtest/metrics imports from risk/metrics (single source of truth)
- **regimes/ → opt/**: regime_aware_portfolio bridges regime detection and optimization
- **data/ → all**: data fetching feeds into ts/, stats/, backtest/, viz/

## Dependency Groups

Core deps always available. Optional groups installed via `pdm install -G <group>`:
market-data, timeseries, cleaning, validation, etl, warehouse, ingestion,
workflow, profiling, dev, optimization, regimes, pde, backtesting, risk,
pricing, stochastic, lp-extra, conic-extra, nlp-extra, causal, quant-math,
bayes, viz, scale

## Quality Bar

- **Depth over breadth**: Fewer functions, each production-quality
- **Plan before implementing**: Research packages/papers first
- **Rich documentation**: Every function explains when/why/how to interpret
- **Single source of truth**: No duplicate logic across modules
- **Consistent API**: pd.Series/np.ndarray inputs, dict returns, Napoleon docstrings
