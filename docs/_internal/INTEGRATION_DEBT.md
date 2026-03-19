# Integration Debt — Cross-Module Reimplementation Audit

**Date:** 2026-03-19
**Status:** In progress (wiring agent running)

## Problem

74 source files have zero cross-module imports. Functions reimplement logic
that already exists in other wraquant modules. This is a consequence of
parallel worktree development where agents wrote files independently.

## Findings

### OLS Regression (15+ files reimplement `np.linalg.lstsq`)
**Canonical:** `stats/regression.py::ols()`

| File | Calls | Priority |
|------|-------|----------|
| causal/treatment.py | 15 lstsq calls | CRITICAL |
| bayes/models.py | 5 calls | HIGH |
| econometrics/diagnostics.py | 3 calls | MEDIUM |
| risk/factor.py | 2 calls | MEDIUM |
| risk/beta.py | 2 calls | MEDIUM |
| ml/preprocessing.py | 1 call | LOW |
| price/fbsde.py | 1 call | LOW |

### Drawdown (7+ files reimplement `.cummax()` logic)
**Canonical:** `risk/metrics.py::max_drawdown()`

| File | Status |
|------|--------|
| stats/descriptive.py | DELEGATING (fixed) |
| backtest/metrics.py | DELEGATING (fixed) |
| backtest/tearsheet.py | REIMPLEMENTS |
| risk/tail.py | REIMPLEMENTS |
| ta/performance.py | REIMPLEMENTS |
| viz/returns.py | REIMPLEMENTS |

### Sharpe/Sortino (6+ files)
**Canonical:** `risk/metrics.py::sharpe_ratio()`

| File | Status |
|------|--------|
| regimes/hmm.py | inline mean/std*sqrt(252) |
| ml/evaluation.py | inline |
| ml/pipeline.py | inline |
| viz/charts.py | inline |
| viz/dashboard.py | inline |
| viz/interactive.py | inline |

### Correlation/Covariance (16+ files use `np.cov`/`np.corrcoef`)
**Canonical:** `stats/correlation.py`

Major offenders: risk/factor.py, risk/metrics.py, risk/beta.py,
risk/stress.py, risk/copulas.py, risk/dcc.py, bayes/models.py,
stats/factor_analysis.py, opt/utils.py

### VaR via percentile (9+ files)
**Canonical:** `risk/var.py::value_at_risk()`

| File | Status |
|------|--------|
| risk/tail.py | REIMPLEMENTS |
| risk/stress.py | REIMPLEMENTS |
| risk/scenarios.py | REIMPLEMENTS |
| regimes/hmm.py | REIMPLEMENTS |
| backtest/tearsheet.py | REIMPLEMENTS |
| bayes/models.py | REIMPLEMENTS |

## Fix Priority

### Phase 1: OLS consolidation (biggest duplication)
- Replace all 15 lstsq calls in causal/ with stats.ols
- Replace lstsq in bayes/, econometrics/, risk/

### Phase 2: Risk metrics consolidation
- All drawdown → risk/metrics.max_drawdown
- All sharpe → risk/metrics.sharpe_ratio
- All VaR percentile → risk/var.value_at_risk

### Phase 3: Stats consolidation
- All np.cov → stats/correlation.covariance (where appropriate)
- Note: some np.cov calls are fine inline (simple 2x2 cov)

### Phase 4: Cross-module bridges
- vol/ → risk/ (GARCH-informed VaR) — DONE
- regimes/ → backtest/ (regime signals) — DONE
- microstructure/ → execution/ — DONE
- ts/ → vol/ (GARCH residuals) — DONE
- bayes/ → regimes/ — DONE

## Notes

- Don't over-consolidate: `np.cov(x, y)` for a simple 2-variable covariance
  is fine inline. Only consolidate when the full correlation pipeline matters.
- Some "reimplementations" are intentional specializations (e.g., Bayesian VaR
  uses a different formula than frequentist VaR). Document these.
- The compose.py workflow system sits on top — it doesn't fix the underlying
  silo problem, it just provides nice syntax.
