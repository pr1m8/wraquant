# Integration & Consistency Plan

## Status: DRAFT — For discussion before implementation

---

## Phase 1: Cross-Module Integration

### 1.1 vol/ → risk/ (GARCH-informed VaR)
**Current:** `risk.value_at_risk()` uses historical or parametric (normal) VaR.
GARCH conditional vol from `vol.garch_forecast()` is computed separately.

**Proposed:** Add `conditional_vol` parameter to `value_at_risk()`:
```python
# Before:
var = value_at_risk(returns, alpha=0.05, method="parametric")

# After:
garch = garch_fit(returns)
var = value_at_risk(returns, alpha=0.05, method="parametric",
                    conditional_vol=garch['conditional_volatility'])
```
**Question:** Should this live in risk/var.py or be a new `risk.garch_var()` function?

### 1.2 regimes/ → backtest/ (Regime-filtered signals)
**Current:** `backtest.regime_conditional_sizing()` adjusts weights by regime probs.
But no way to GATE signals (only enter trades in certain regimes).

**Proposed:** Add `regime_signal_filter()` to backtest/position.py:
```python
# Only trade when P(bull regime) > 0.6
filtered = regime_signal_filter(signals, regime_probs,
                                 active_regime=0, min_prob=0.6)
```

### 1.3 ts/forecast → backtest/ (Forecast-driven backtesting)
**Current:** Forecasting and backtesting are independent.

**Proposed:** Add `forecast_strategy()` to backtest/strategy.py:
```python
# Backtest a forecasting model
strategy = ForecastStrategy(
    model_fn=lambda data: auto_forecast(data, horizon=5),
    signal_fn=lambda forecast: 1 if forecast['forecast'].mean() > 0 else -1,
)
result = Backtest(strategy).run(prices)
```

### 1.4 ml/ → regimes/ (ML-based regime features)
**Current:** `ml.features.regime_features()` exists but requires pre-computed regimes.

**Proposed:** Add auto-detection: if no regime states provided, auto-run `detect_regimes()`:
```python
features = regime_features(returns)  # auto-detects regimes internally
```

### 1.5 price/greeks → backtest/position (Options position sizing)
**Current:** Greeks computed but not used in position sizing.

**Proposed:** Add `delta_neutral_sizing()` to backtest/position.py

### 1.6 microstructure/ → execution/ (Dynamic impact)
**Current:** Execution cost estimates don't use live microstructure data.

**Proposed:** `execution.cost.dynamic_impact()` accepts spread/depth/VPIN

---

## Phase 2: Consistency Standardization

### 2.1 Parameter Naming Convention (RFC)

**Proposal — keep both `period` and `window`, document which is used where:**

| Context | Parameter | Rationale |
|---------|-----------|-----------|
| TA indicators | `period` | Convention from all TA libraries |
| Rolling stats | `window` | Convention from pandas |
| GARCH models | N/A (internal) | Model order `p`, `q` |
| Forecasting | `horizon` | Forward-looking |
| Annualization | `periods_per_year` | Explicit |

**Alternative:** Standardize everything to `period`. Thoughts?

### 2.2 Return Type Convention

| Output Type | Return As | Example |
|-------------|-----------|---------|
| Single metric | `float` | `sharpe_ratio() → 1.45` |
| Time series | `pd.Series` | `conditional_volatility → Series` |
| Multi-output indicator | `dict[str, pd.Series]` | `macd() → {"macd": ..., "signal": ...}` |
| Model results | `dict[str, Any]` | `garch_fit() → {"params": ..., "aic": ...}` |
| Regime results | `RegimeResult` dataclass | `detect_regimes() → RegimeResult(...)` |
| Comparison table | `pd.DataFrame` | `strategy_comparison() → DataFrame` |

**Question:** Should we create more dataclasses like RegimeResult for other domains?
E.g., `GARCHResult`, `BacktestResult`, `ForecastResult`?

### 2.3 Docstring Level

Three tiers based on function importance:

| Tier | Functions | Docstring Depth |
|------|-----------|----------------|
| **Core** | Top 20% most-used (sharpe, garch_fit, detect_regimes, etc.) | Full Napoleon: math, interpretation, when-to-use, example, refs, See Also |
| **Standard** | Most public functions | Parameters, Returns, Example, one-line "when to use" |
| **Helper** | Internal/niche functions | Parameters, Returns only |

### 2.4 Import Pattern

Every module follows:
```python
# Public API: import from __init__.py
from wraquant.risk import sharpe_ratio

# Cross-module: import specific function
from wraquant.risk.metrics import sharpe_ratio as _sharpe

# Never: import entire module for one function
import wraquant.risk.metrics  # avoid
```

---

## Phase 3: Updated Examples

After integration + consistency, update examples/ to showcase:
- End-to-end workflow: data → regime detection → GARCH vol → VaR → backtest
- Cross-module integration: ta indicators → ml features → regime-aware backtest
- Forecasting pipeline: auto_forecast → ensemble → walk-forward evaluation
- Options workflow: FBSDE pricing → Greeks → delta-hedging backtest

---

## Questions for Discussion

1. **Result dataclasses vs dicts:** Should we add GARCHResult, BacktestResult, ForecastResult like RegimeResult? Pro: IDE autocomplete, type safety. Con: more code to maintain.

2. **`period` vs `window`:** Standardize to one, or keep both with documentation?

3. **Top-level convenience functions:** Should `wraquant` expose `wq.forecast()`, `wq.backtest()`, `wq.detect_regimes()` at the package level?

4. **Experiment lab scope:** What does your ideal research workflow look like? (Discuss before building)
