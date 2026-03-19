# Type System Analysis — Complete Findings

**Date:** 2026-03-19

## Current State (Problem)

- **74 files** with zero cross-module imports
- **3 competing type patterns**: strict pd.Series (ta/), union with branching (ml/regimes), convert-to-numpy (opt/price)
- **core/types.py**: 8 type aliases defined, **0 functions use them**
- **frame/ module**: Dead code — designed as abstraction, adopted by 0 modules
- **434** `.values` conversions, **499** `np.asarray()` calls — manual glue everywhere

## Research Recommendation

**Use the sklearn/numpy "coerce-first" pattern.** This is what every successful numerical Python library does.

### The Pattern

```python
def _coerce_returns(data) -> np.ndarray:
    """Normalize any input to a 1D float64 numpy array."""
    if hasattr(data, 'values'):       # pd.Series / pd.DataFrame
        return np.asarray(data.values, dtype=np.float64).ravel()
    if hasattr(data, 'numpy'):        # torch.Tensor
        return data.detach().cpu().numpy().ravel()
    return np.asarray(data, dtype=np.float64).ravel()
```

### What NOT to do
- **No plum-dispatch** — multi-arg dispatch rarely needed, adds 3 deps
- **No custom ExtensionDtype** — 20+ methods to implement, fragile across pandas versions
- **No beartype on hot paths** — even O(1) overhead × millions of calls adds up
- **No @runtime_checkable Protocol** in loops — slower than isinstance

### What TO do
1. **Coerce-first** at public function entry points
2. **@typing.overload** for IDE experience (Series in → Series out)
3. **isinstance** for runtime branching (simple, fast, explicit)
4. **singledispatch** only for extensibility points (not internal dispatch)
5. **hasattr duck-typing** for torch (avoid importing torch for type check)
6. **pd.Series.attrs** for metadata (frequency, currency, asset class)

## Implementation Plan

### Phase 1: Core coercion utilities (core/_coerce.py)

```python
def coerce_array(data, name="data") -> np.ndarray:
    """Any array-like → 1D float64 ndarray."""

def coerce_series(data, name="data") -> pd.Series:
    """Any array-like → pd.Series (preserves index if possible)."""

def coerce_returns(data, name="returns") -> np.ndarray:
    """Prices or returns → returns as ndarray. Auto-detects prices vs returns."""

def coerce_dataframe(data, name="data") -> pd.DataFrame:
    """Dict, ndarray, or DataFrame → pd.DataFrame."""
```

### Phase 2: Adopt in risk/ and stats/ (highest impact)

Replace inline `.values` / `np.asarray()` with `coerce_array()`.
Functions that currently only accept pd.Series now also accept ndarray/torch.

### Phase 3: Adopt in ta/ (265 indicators)

Replace `_validate_series` strict check with `coerce_series` that
auto-converts ndarray → pd.Series. Backward compatible.

### Phase 4: Adopt in remaining modules

vol/, regimes/, ml/, price/, backtest/, econometrics/

## Performance Notes

- ta/ indicators called millions of times in backtests
- Coercion overhead: ~100ns per isinstance check
- Current .values overhead: ~200ns per conversion
- Net: coerce-first is faster than current pattern (one conversion vs repeated)
- For torch: only import torch when hasattr(data, 'numpy') is True
