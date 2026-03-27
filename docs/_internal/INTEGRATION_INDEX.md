# Integration Index — Full 5-Level Map

**Last updated:** 2026-03-19

## Level 1: Coercion (accept any input type)

Every public function should call `coerce_array`/`coerce_series`/`coerce_dataframe`
at entry so it accepts pd.Series, np.ndarray, list, torch.Tensor.

| Module | Files | Funcs | Coercion adopted? |
|--------|-------|-------|-------------------|
| ta/ | 18 | 265 | YES (via _validators.py) |
| risk/metrics | 1 | 10 | YES |
| risk/var | 1 | 5 | YES (garch_var) |
| stats/descriptive | 1 | 8 | YES |
| stats/regression | 1 | 5 | YES |
| stats/correlation | 1 | 9 | YES |
| vol/models | 1 | 16 | YES (via _to_returns_array) |
| vol/realized | 1 | 9 | YES |
| regimes/hmm | 1 | 6 | YES |
| **TOTAL DONE** | **26** | **333** | **~30%** |
| risk/ remaining | 7 | 46 | NO — agents running |
| stats/ remaining | 4 | 22 | NO — agents running |
| ts/ all | 9 | 52 | NO — agents running |
| price/ all | 5 | 27 | NO — agents running |
| opt/ all | 4 | 14 | NO — agents running |
| microstructure/ | 2 | 23 | NO — agents running |
| execution/ | 2 | 13 | NO — agents running |
| forex/ | 3 | 12 | NO — agents running |
| ml/ remaining | 3 | 16 | NO — agents running |
| bayes/ remaining | 1 | 8 | NO — agents running |
| econometrics/ | 3 | 14 | NO — agents running |
| experiment/ | 4 | 10 | NO — agents running |
| math/ | 9 | 55 | NO — agents running |
| backtest/ | 5 | 25 | NO — agents running |

---

## Level 2: Canonical Imports (don't reimplement)

### OLS regression: stats/regression.py::ols is canonical

| File | lstsq calls | Status |
|------|-------------|--------|
| causal/treatment.py | 20 | FIXED → _ols_coefficients |
| bayes/models.py | 1 | FIXED → stats.ols |
| econometrics/diagnostics.py | 1 | FIXED → stats.ols |
| risk/factor.py | 2 | FIXED (earlier agent) |
| risk/beta.py | 1 | FIXED (earlier agent) |
| risk/stress.py | 1 | FIXED (earlier agent) |
| risk/dcc.py | 0 | OK (DCC uses MLE, no OLS lstsq) |
| risk/survival.py | 1 | OK (Newton-Raphson fallback, not OLS) |
| econometrics/panel.py | 2 | FIXED → stats.ols (FE + RE) |
| econometrics/cross_section.py | 1 | FIXED → stats.ols (2SLS 2nd stage) |
| econometrics/event_study.py | 1 | FIXED → stats.ols (market model) |
| math/optimal_stopping.py | 1 | OK (pure math, not standard OLS) |
| math/network.py | 1 | OK (pure math, not standard OLS) |
| regimes/kalman.py | 1 | OK (Kalman is specialized, not OLS) |
| ts/stationarity.py | 1 | FIXED → stats.ols (Phillips-Perron) |
| stats/cointegration.py | 3 | FIXED → stats.ols (engle_granger, spread, hedge_ratio) |
| ml/online.py | 1 | OK (RLS is specialized, not standard OLS) |
| price/fbsde.py | 2 | OK (BSDE regression in MC loop, specialized) |

### Drawdown: risk/metrics.py::max_drawdown is canonical

| File | Status |
|------|--------|
| stats/descriptive.py | FIXED → imports risk.max_drawdown |
| backtest/metrics.py | FIXED → imports risk.max_drawdown |
| backtest/tearsheet.py | FIXED → imports risk.max_drawdown |
| risk/tail.py | OK (different: CDaR needs full drawdown series) |
| ta/performance.py | OK (different: rolling indicator, not scalar) |

### Sharpe: risk/metrics.py::sharpe_ratio is canonical

| File | Status |
|------|--------|
| regimes/hmm.py | FIXED → imports risk.sharpe_ratio |
| ml/evaluation.py | FIXED → imports risk.sharpe_ratio |
| ml/pipeline.py | FIXED → imports risk.sharpe_ratio |
| viz/* | TODO — viz should use risk.sharpe when auto-computing |
| experiment/runner.py | DONE — already imports risk.sharpe_ratio |

### Covariance: stats/correlation.py is canonical

| File | np.cov calls | Should import stats.correlation? |
|------|-------------|--------------------------------|
| risk/beta.py | 2 | MAYBE (simple 2-var cov, inline is fine) |
| risk/metrics.py | 1 | MAYBE (simple cov for beta calc) |
| risk/factor.py | 2 | YES — factor cov matrix |
| risk/copulas.py | 1 | YES — empirical correlation |
| risk/dcc.py | 2 | MAYBE (DCC has its own cov dynamics) |
| bayes/models.py | 2 | YES — posterior covariance |
| regimes/hmm.py | 1 | MAYBE (per-regime cov, specialized) |
| stats/factor_analysis.py | 2 | YES — should use own module |
| experiment/results.py | 1 | YES |
| backtest/tearsheet.py | 1 | YES |
| recipes.py | 1 | YES |
| microstructure/liquidity.py | 1 | MAYBE |

Note: Simple 2-variable `np.cov(x, y)` is fine inline. Multi-variable
covariance matrices should use `stats.correlation` for consistency.

---

## Level 3: Cross-Module Usage (modules USE each other)

### backtest/ should use:
| From | What | Current | Status |
|------|------|---------|--------|
| risk/metrics | sharpe, sortino, max_dd | 2/7 files | PARTIAL |
| risk/var | VaR in tearsheets | 1/7 files (tearsheet.py) | DONE |
| vol/models | GARCH vol for position sizing | 0/7 files | TODO |
| regimes/ | regime-aware strategies | 1/7 files (position.py) | PARTIAL |
| ta/ | indicators for signal generation | 0/7 files | TODO |

### viz/ should use:
| From | What | Current | Status |
|------|------|---------|--------|
| risk/metrics | auto-compute sharpe/dd when plotting | 1/10 files (dashboard.py) | DONE |
| regimes/ | regime overlay | 1/10 files (__init__.py) | MINIMAL |
| vol/ | GARCH vol for vol charts | 0/10 files | TODO |
| stats/ | distribution fit overlay | 1/10 files (charts.py) | DONE |

### opt/ should use:
| From | What | Current | Status |
|------|------|---------|--------|
| risk/portfolio | portfolio_volatility | 1/7 files | PARTIAL |
| risk/portfolio_analytics | component VaR, diversification | 0/7 files | TODO |
| stats/correlation | shrunk covariance for optimization | 0/7 files | TODO |
| regimes/ | regime-conditional weights | 0/7 files | TODO |

### execution/ should use:
| From | What | Current | Status |
|------|------|---------|--------|
| microstructure/ | spread, depth for cost | 2/3 files (algorithms.py, optimal.py) | DONE |
| risk/ | execution risk | 0/3 files | TODO |

### experiment/ should use:
| From | What | Current | Status |
|------|------|---------|--------|
| risk/metrics | experiment metrics | 2/9 files | PARTIAL |
| backtest/ | backtesting | 1/9 files | PARTIAL |
| regimes/ | regime breakdown | 1/9 files | PARTIAL |
| vol/ | vol analysis | 0/9 files | TODO |

### ml/ should use:
| From | What | Current | Status |
|------|------|---------|--------|
| ta/ | indicators as features | 1/9 files (features.py) | PARTIAL |
| stats/ | distribution features | 0/9 files | TODO |
| risk/ | risk-based features | 0/9 files | TODO |
| regimes/ | regime as features | 1/9 files (features.py) | PARTIAL |

---

## Level 4: Result Types (return dataclasses, not dicts)

| Function | Current return | Should return | Status |
|----------|---------------|---------------|--------|
| vol/garch_fit | dict | GARCHResult | TODO |
| vol/egarch_fit | dict | GARCHResult | TODO |
| vol/gjr_garch_fit | dict | GARCHResult | TODO |
| backtest/Backtest.run | dict | BacktestResult | TODO |
| backtest/VectorizedBacktest.run | dict | BacktestResult | TODO |
| ts/auto_forecast | dict | ForecastResult | TODO |
| ts/theta_forecast | dict | ForecastResult | TODO |
| regimes/detect_regimes | RegimeResult | RegimeResult | DONE |

GARCHResult, BacktestResult, ForecastResult exist in core/results.py
with chaining methods (.to_var(), .plot(), .summary()) but aren't
returned by any function yet.

---

## Level 5: Frame Types (PriceSeries/ReturnSeries throughout)

| Module | Should return | Should accept | Status |
|--------|-------------|---------------|--------|
| data/fetch_prices | PriceSeries | N/A | TODO |
| data/fetch_ohlcv | OHLCVFrame | N/A | TODO |
| ta/ indicators | PriceSeries (already Series) | PriceSeries (works via coercion) | PARTIAL |
| risk/metrics | N/A | ReturnSeries (auto periods_per_year) | TODO |
| vol/realized | PriceSeries | PriceSeries | TODO |
| frame/ops | PriceSeries/ReturnSeries | PriceSeries | TODO (dead code) |

PriceSeries and ReturnSeries exist in frame/base.py with:
- frequency auto-detection
- periods_per_year property
- to_returns() / to_prices() conversion
- sharpe(), annualized_vol() on ReturnSeries

But only 2 files outside frame/ currently use them.

---

## Priority for Implementation

1. **Level 2 remaining** — 12 files with lstsq → ols (biggest code quality win)
2. **Level 1 remaining** — 49 files need coercion (agents running)
3. **Level 3** — backtest→risk, viz→risk, opt→stats, execution→micro
4. **Level 4** — vol functions return GARCHResult, backtest returns BacktestResult
5. **Level 5** — data returns PriceSeries, risk accepts ReturnSeries
