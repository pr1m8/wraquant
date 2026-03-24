# Next Session Priorities

## The Core Problem

The codebase has 93K LOC and 3500+ tests but modules are DISJOINT.
Written by isolated agents, most files don't import from each other.
Types, coercion, and core utilities exist but aren't adopted.

## Priority 1: Universal coercion + torch overloads

Every public function should:
1. Accept pd.Series, np.ndarray, list, torch.Tensor
2. Use core/_coerce.py at entry
3. Where GPU would help (matrix ops, large GARCH, ML), have torch path

Currently only ~30% of functions use coercion. Need 100%.

Torch overload pattern:
```python
def sharpe_ratio(returns):
    arr = coerce_array(returns)
    # numpy path (default)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    return mean / std * np.sqrt(252)

# If we detect torch input, use torch ops for GPU
```

## Priority 2: Frame types used everywhere

PriceSeries, ReturnSeries, OHLCVFrame should be:
- Returned by data.fetch_prices()
- Accepted by ta/ indicators
- Returned by .to_returns() conversions
- Carrying metadata (frequency, periods_per_year) through pipeline

## Priority 3: Cross-module imports (63 files)

See INTEGRATION_DEBT.md. Key patterns:
- All OLS → stats/regression.ols
- All drawdown → risk/metrics.max_drawdown
- All sharpe → risk/metrics.sharpe_ratio
- All VaR → risk/var.value_at_risk

## Priority 4: Module structure fixes

- scale/ — single __init__.py → split into backends.py, patterns.py
- flow/ — single __init__.py → split into pipeline.py, scheduling.py, utilities.py
- Weak modules that need depth: frame/, flow/

## Priority 5: Result chaining

GARCHResult.to_var() → feeds into risk/var
RegimeResult.to_portfolio() → feeds into opt/
BacktestResult.to_tearsheet() → feeds into viz/
ForecastResult.to_backtest() → feeds into backtest/

## Priority 6: Application layer

wq.portfolio.construct(), wq.research.backtest(), wq.market.regime()
Thin facades composing the integrated modules.

## Approach

Do it MODULE BY MODULE, not all at once:
1. Pick one module (e.g., risk/)
2. Add coercion to all functions
3. Wire all cross-module imports
4. Add torch overloads where relevant
5. Use frame types in signatures
6. Verify tests pass
7. Move to next module

This is 2-3 sessions of focused work, not one giant bomb.
