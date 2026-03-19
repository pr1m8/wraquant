# Type System & Dispatch Architecture Plan

**Status:** PLANNING — Must analyze fully before implementing

## Vision

wraquant needs a proper type system where:
1. Functions auto-handle different input types (pd.Series, np.ndarray, torch.Tensor)
2. Financial data types are first-class (PriceSeries, ReturnSeries, OHLCVFrame)
3. Single dispatch or overloading routes to optimized implementations
4. Time series metadata (frequency, timezone, asset class) flows through pipelines
5. Bottom-up hierarchy: raw arrays → typed series → financial series → portfolio

## Key Design Questions to Analyze

### 1. Single Dispatch vs Overloading
- `functools.singledispatch` for type-based routing?
- `typing.overload` for IDE support?
- Custom `@wraquant_dispatch` decorator?
- How does this interact with `@requires_extra`?

### 2. Type Hierarchy (bottom-up)
```
np.ndarray / torch.Tensor
  └── pd.Series (with index)
       └── TimeSeries (with frequency metadata)
            ├── PriceSeries (non-negative, level data)
            │    ├── EquityPrices
            │    ├── FXRates
            │    └── BondPrices
            ├── ReturnSeries (centered, pct or log)
            │    ├── SimpleReturns
            │    └── LogReturns
            └── VolatilitySeries (non-negative)
  └── pd.DataFrame
       ├── OHLCVFrame (open, high, low, close, volume columns)
       ├── ReturnFrame (multi-asset returns)
       └── FactorFrame (factor returns with names)
```

### 3. Auto-Conversion
Functions should accept multiple types and auto-convert:
```python
# Before: user must know the right type
sharpe = sharpe_ratio(returns.values)  # np.ndarray only

# After: works with anything
sharpe = sharpe_ratio(prices)          # auto-converts to returns
sharpe = sharpe_ratio(returns_series)  # pd.Series fine
sharpe = sharpe_ratio(returns_array)   # np.ndarray fine
sharpe = sharpe_ratio(returns_tensor)  # torch.Tensor → numpy → compute
```

### 4. Torch Tensor Support
- Which functions benefit from torch? (GPU-accelerated matrix ops)
- How to handle optional torch import?
- Should we have `@supports_torch` decorator?
- Dispatch: numpy path vs torch path based on input type

### 5. Rich Annotations
```python
from wraquant.core.types import PriceSeries, ReturnSeries, OHLCVFrame

def sharpe_ratio(
    returns: ReturnSeries | PriceSeries,  # auto-converts prices to returns
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
```

### 6. Time Series Handling
- Auto-detect frequency from DatetimeIndex
- Auto-align series in cross-asset operations
- Handle missing data consistently
- Timezone-aware operations

### 7. Convenience / Auto-detection
- If user passes prices to a function expecting returns, auto-convert
- If user passes daily data to an annualization function, auto-detect periods_per_year
- If user passes DataFrame to a function expecting Series, use first column with warning

## Analysis Tasks (Before Implementation)

### Task 1: Survey existing type patterns
- How do functions currently handle different input types?
- What does the `_validate_series` pattern look like across modules?
- Where do type errors actually happen?

### Task 2: Benchmark dispatch overhead
- Is singledispatch fast enough for hot-path functions (TA indicators)?
- What's the overhead of type checking + conversion?

### Task 3: Survey Python ecosystem
- How does pandas-stubs handle typing?
- How does polars handle dispatch?
- How does scikit-learn handle array-like inputs?
- What does beartype/typeguard add?

### Task 4: Prototype approaches
- Try singledispatch on 3-4 core functions
- Try Protocol-based typing
- Try simple isinstance checks with auto-conversion
- Compare ergonomics and performance

### Task 5: Plan migration
- Which functions to convert first (core/risk/stats)?
- How to maintain backward compatibility?
- How to handle the 265 TA indicators (bulk conversion)?

## Considerations

- **Don't over-engineer**: Simple isinstance + auto-convert may beat fancy dispatch
- **Performance matters**: TA indicators are called millions of times in backtests
- **IDE support matters**: Types should help autocomplete, not hinder it
- **Backward compat**: Can't break existing `sharpe_ratio(np.array([...]))`
- **Gradual adoption**: Convert module by module, not all at once
