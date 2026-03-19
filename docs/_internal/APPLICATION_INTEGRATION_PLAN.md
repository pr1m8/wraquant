# Application-Based Integration & Hierarchical Organization Plan

**Status:** PLANNING

## Problem

Modules are organized by technical domain (stats/, risk/, ta/) but quants think
in application workflows:
- "I'm researching a momentum strategy"
- "I need to assess my portfolio risk"
- "I'm pricing an exotic option"
- "I want to detect if the market regime changed"

The library should support BOTH:
- Bottom-up: individual functions for power users
- Top-down: application workflows for getting things done

## Proposed Hierarchy

### Tier 1: Application Workflows (top-level)

```python
import wraquant as wq

# Portfolio Management
wq.portfolio.construct(returns_df, method="risk_parity")  # opt + risk + regimes
wq.portfolio.rebalance(current_weights, target_weights)   # execution + cost
wq.portfolio.monitor(portfolio, benchmark)                 # risk + viz

# Strategy Research
wq.research.backtest(strategy_fn, data, params)            # experiment + backtest
wq.research.compare(strategies_dict)                       # backtest + viz
wq.research.report(results)                                # tearsheet + viz

# Risk Assessment
wq.risk_report(portfolio_returns, benchmark)               # risk + vol + stress + viz
wq.var_analysis(returns, methods=["historical", "garch"])   # risk + vol

# Market Analysis
wq.market.regime(returns)                                  # regimes + viz
wq.market.screen(universe, indicators=["rsi", "macd"])     # ta + data
wq.market.correlations(returns_df)                         # stats + viz

# Pricing
wq.price_option(spot, strike, vol, rf, T)                  # price + greeks
wq.yield_curve(maturities, rates)                          # price/curves + viz
```

### Tier 2: Module Workflows (compose.py steps)

```python
# Already built — Workflow + steps
wf = wq.Workflow("research").add(wq.steps.returns()).add(...)
```

### Tier 3: Individual Functions (current)

```python
# Power user — direct module access
from wraquant.risk.metrics import sharpe_ratio
from wraquant.vol.models import garch_fit
```

## What needs to happen

### 1. Create application-level entry points

NOT new modules — just convenience functions in __init__.py or recipes.py
that compose existing modules for common use cases.

### 2. Better cross-module integration

Functions should AUTO-CHAIN:
- garch_fit → automatically available for garch_var
- detect_regimes → automatically available for regime_signal_filter
- ta indicators → automatically available as ml features

Currently these require manual imports and data passing.

### 3. Hierarchical result objects

When you run a portfolio analysis, you should get back ONE object
that contains everything — not 5 separate dicts from 5 modules:

```python
result = wq.portfolio.analyze(returns_df)
result.weights          # from opt/
result.risk            # from risk/
result.regimes         # from regimes/
result.tearsheet       # from backtest/
result.plot()          # from viz/
```

### 4. Better organization of what we have

Some modules should be SUB-modules of others:
- execution/ is really part of the trading/portfolio workflow
- microstructure/ feeds into execution/
- experiment/ is really the research workflow
- flow/ + scale/ are infrastructure, not quant domain

## Priority for implementation

1. Add convenience functions to wraquant namespace (recipes.py already started)
2. Make result objects composable (GARCHResult, RegimeResult already exist)
3. Better top-level API (wq.portfolio, wq.research, wq.market namespaces)
4. Frame redesign with financial types
