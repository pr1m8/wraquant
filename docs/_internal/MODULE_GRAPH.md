# Module Dependency Graph — How Everything Should Link

## Current State: Flat siblings (BAD)

```
wraquant/
├── risk/        ← standalone
├── stats/       ← standalone
├── vol/         ← standalone
├── regimes/     ← standalone
├── ta/          ← standalone
├── ml/          ← standalone
├── price/       ← standalone
├── ...          ← 20 more standalone modules
```

Everything at the same level. No hierarchy. No flow.

## Target State: Layered DAG

```
┌─────────────────────────────────────────────────────────┐
│                 APPLICATION LAYER                        │
│                                                         │
│   wq.portfolio    wq.research    wq.market    wq.price  │
│   (construct,     (backtest,     (regime,     (option,   │
│    rebalance,      compare,       screen,      curve,    │
│    monitor)        report)        analyze)     hedge)    │
└────────┬──────────────┬──────────────┬──────────┬───────┘
         │              │              │          │
┌────────▼──────────────▼──────────────▼──────────▼───────┐
│                 ORCHESTRATION LAYER                      │
│                                                         │
│   recipes.py      compose.py      experiment/           │
│   (pipelines)     (Workflow +     (Lab, grid,           │
│                    steps)          CV, tracking)         │
└────────┬──────────────┬──────────────┬──────────────────┘
         │              │              │
┌────────▼──────────────▼──────────────▼──────────────────┐
│                 ANALYSIS LAYER                           │
│                                                         │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│   │ backtest │  │   viz    │  │ dashboard│             │
│   │ (engine, │  │ (charts, │  │ (streamlit│             │
│   │  metrics,│  │  dashbd) │  │  pages)  │             │
│   │  tear)   │  │          │  │          │             │
│   └────┬─────┘  └────┬─────┘  └──────────┘             │
│        │              │                                  │
└────────┼──────────────┼──────────────────────────────────┘
         │              │
┌────────▼──────────────▼──────────────────────────────────┐
│                 MODELING LAYER                            │
│                                                         │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌───────┐ │
│  │  risk  │ │  vol   │ │regimes │ │  opt   │ │  ml   │ │
│  │ (VaR,  │ │(GARCH, │ │ (HMM,  │ │ (MVO,  │ │(LSTM, │ │
│  │  beta, │ │ Hawkes,│ │ MS-AR, │ │  RP,   │ │ RF,   │ │
│  │  factor│◄┤ stoch) │ │ Kalman)│ │  BL)   │ │ pipe) │ │
│  │  tail) │ │        │ │        │ │        │ │       │ │
│  └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └──┬────┘ │
│      │          │          │          │         │      │
└──────┼──────────┼──────────┼──────────┼─────────┼──────┘
       │          │          │          │         │
┌──────▼──────────▼──────────▼──────────▼─────────▼──────┐
│                 QUANTITATIVE LAYER                       │
│                                                         │
│  ┌────────┐ ┌────────┐ ┌──────────┐ ┌────────┐         │
│  │ stats  │ │  ts    │ │  price   │ │ econom │         │
│  │(regr,  │ │(decomp,│ │(BS, FBSDE│ │(panel, │         │
│  │ corr,  │ │ forec, │ │ char fn, │ │ VAR,   │         │
│  │ distr, │ │ statio,│ │ stoch,   │ │ event) │         │
│  │ robust)│ │ anomal)│ │ curves)  │ │        │         │
│  └───┬────┘ └───┬────┘ └────┬─────┘ └───┬────┘         │
│      │          │           │           │               │
└──────┼──────────┼───────────┼───────────┼───────────────┘
       │          │           │           │
┌──────▼──────────▼───────────▼───────────▼───────────────┐
│                 DOMAIN LAYER                             │
│                                                         │
│  ┌────────┐ ┌────────┐ ┌──────────┐ ┌─────────┐        │
│  │   ta   │ │  forex │ │  micro   │ │execution│        │
│  │(265    │ │(pairs, │ │(liquidity│ │(TWAP,   │        │
│  │ indic) │ │ carry, │ │ toxicity,│ │ VWAP,   │        │
│  │        │ │ session│ │ quality) │ │ optimal)│        │
│  └───┬────┘ └───┬────┘ └────┬─────┘ └────┬────┘        │
│      │          │           │            │              │
└──────┼──────────┼───────────┼────────────┼──────────────┘
       │          │           │            │
┌──────▼──────────▼───────────▼────────────▼──────────────┐
│                 FOUNDATION LAYER                         │
│                                                         │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌───────┐ │
│  │  core  │ │ frame  │ │  data  │ │   io   │ │ bayes │ │
│  │(types, │ │(Price  │ │(fetch, │ │(SQL,   │ │(MCMC, │ │
│  │ coerce,│ │ Series,│ │ clean, │ │ cloud, │ │ conj, │ │
│  │ config,│ │ Return │ │ valid) │ │ files) │ │ model)│ │
│  │ except)│ │ Series)│ │        │ │        │ │       │ │
│  └────────┘ └────────┘ └────────┘ └────────┘ └───────┘ │
│                                                         │
│  ┌────────┐ ┌────────┐ ┌────────┐                       │
│  │  math  │ │  flow  │ │ scale  │                       │
│  │(Levy,  │ │(DAG,   │ │(joblib,│                       │
│  │ network│ │ pipe,  │ │ dask,  │                       │
│  │ optim) │ │ cache) │ │ ray)   │                       │
│  └────────┘ └────────┘ └────────┘                       │
└─────────────────────────────────────────────────────────┘
```

## Data Flow Patterns

### Pattern 1: Strategy Research Pipeline

```
data.fetch_prices("AAPL")
  → frame.PriceSeries (with frequency, metadata)
    → ta.rsi(), ta.macd() (indicators on PriceSeries)
      → ml.features (TA + return + vol features)
        → ml.walk_forward (train/test with purged CV)
          → backtest.engine (signals → returns)
            → risk.metrics (sharpe, drawdown)
              → viz.tearsheet (report)
                → experiment.save (persist results)
```

### Pattern 2: Portfolio Construction

```
data.fetch_prices(["AAPL", "GOOGL", "MSFT"])
  → frame.ReturnFrame (multi-asset returns)
    → regimes.detect (market regime)
      → risk.factor_model (factor decomposition)
        → opt.risk_parity (optimize weights)
          → risk.portfolio_analytics (component VaR)
            → execution.optimal (Almgren-Chriss schedule)
              → microstructure.cost (impact estimate)
                → viz.portfolio_dashboard (report)
```

### Pattern 3: Risk Monitoring

```
portfolio_returns (daily)
  → vol.garch_fit (conditional volatility)
    → risk.garch_var (time-varying VaR)
      → risk.stress (scenario analysis)
        → risk.historical (crisis comparison)
          → regimes.detect (current regime)
            → viz.risk_dashboard (alert if breach)
```

### Pattern 4: Derivatives Pricing

```
market_data (spot, vol surface, rates)
  → price.curves (yield curve bootstrap)
    → price.characteristic (Heston char fn)
      → price.options (FFT pricing)
        → price.greeks (sensitivities)
          → risk.greeks_var (Greeks-based risk)
            → execution.hedge (delta hedge schedule)
```

## Key Integration Points (arrows in the graph)

### MUST EXIST (data flows through these):
1. `data → frame` — fetch returns PriceSeries/OHLCVFrame
2. `frame → ta` — PriceSeries feeds indicators
3. `ta → ml` — indicators become features
4. `ml → backtest` — predictions become signals
5. `backtest → risk` — strategy returns feed risk metrics
6. `risk ← vol` — GARCH vol feeds VaR
7. `regimes → opt` — regime probs adjust weights
8. `regimes → backtest` — regime filters signals
9. `risk → viz` — metrics feed dashboards
10. `all → experiment` — any result can be tracked

### SHOULD EXIST (enrichment):
11. `microstructure → execution` — liquidity adjusts scheduling
12. `price → risk` — Greeks feed risk decomposition
13. `bayes → regimes` — Bayesian regime inference
14. `ts → vol` — GARCH residuals for forecasting
15. `stats → everything` — regression, correlation, tests used everywhere

### NICE TO HAVE (convenience):
16. `forex → risk` — FX exposure in portfolio risk
17. `econometrics → stats` — advanced tests extend basic ones
18. `math → price` — Lévy processes for exotic pricing

## Implementation: What changes

1. **frame/ redesign** — PriceSeries/ReturnSeries that carry metadata
   through the pipeline. When data.fetch returns a PriceSeries,
   ta.rsi() knows the frequency, risk.sharpe() knows periods_per_year.

2. **Result chaining** — GARCHResult has .to_var() method that feeds
   into risk.garch_var. RegimeResult has .filter_signals() that
   feeds into backtest. Results know what they can feed into.

3. **Application namespaces** — wq.portfolio, wq.research, wq.market
   are thin facades that compose the right modules for the use case.

4. **Module __init__.py imports** — Each module's __init__ should import
   the modules it depends on, making the graph explicit in code.
