# Example: Portfolio Construction via wraquant-mcp

This document shows the tool calls an AI agent would make to build an
optimized, regime-aware portfolio using wraquant-mcp.

---

## Phase 1: Data Setup

### 1. Check workspace

**Tool call**: `workspace_status()`

**Response**:
```json
{
  "workspace": "default",
  "datasets": ["prices_spy", "prices_tlt", "prices_gld", "prices_eem", "prices_vgk"],
  "models": [],
  "journal_entries": 5
}
```

*Five asset price series are already in the workspace: US equity (SPY),
US bonds (TLT), gold (GLD), emerging markets (EEM), and European equity (VGK).*

### 2. Compute returns for all assets

**Tool call**: `compute_returns("prices_spy")`
**Tool call**: `compute_returns("prices_tlt")`
**Tool call**: `compute_returns("prices_gld")`
**Tool call**: `compute_returns("prices_eem")`
**Tool call**: `compute_returns("prices_vgk")`

**Response** (for SPY):
```json
{
  "dataset_id": "returns_spy",
  "rows": 1257,
  "mean": 0.000542,
  "std": 0.01234,
  "annualized_vol": 0.196
}
```

### 3. Store combined multi-asset returns

**Tool call**: `query_data("SELECT s.returns AS spy, t.returns AS tlt, g.returns AS gld, e.returns AS eem, v.returns AS vgk FROM returns_spy s JOIN returns_tlt t ON s.rowid = t.rowid JOIN returns_gld g ON s.rowid = g.rowid JOIN returns_eem e ON s.rowid = e.rowid JOIN returns_vgk v ON s.rowid = v.rowid")`

The agent stores the aligned returns as a single multi-asset dataset.

**Tool call**: `store_data("multi_asset_returns", {"spy": [...], "tlt": [...], "gld": [...], "eem": [...], "vgk": [...]})`

**Response**:
```json
{
  "dataset_id": "multi_asset_returns",
  "rows": 1257,
  "columns": ["spy", "tlt", "gld", "eem", "vgk"]
}
```

---

## Phase 2: Correlation and Dependence

### 4. Correlation analysis

**Tool call**: `correlation_analysis("multi_asset_returns")`

**Response**:
```json
{
  "tool": "stats.correlation_analysis",
  "correlation_matrix": {
    "spy": {"spy": 1.000, "tlt": -0.352, "gld": 0.048, "eem": 0.724, "vgk": 0.812},
    "tlt": {"spy": -0.352, "tlt": 1.000, "gld": 0.178, "eem": -0.198, "vgk": -0.312},
    "gld": {"spy": 0.048, "tlt": 0.178, "gld": 1.000, "eem": 0.182, "vgk": 0.091},
    "eem": {"spy": 0.724, "tlt": -0.198, "eem": 1.000, "vgk": 0.687},
    "vgk": {"spy": 0.812, "tlt": -0.312, "gld": 0.091, "eem": 0.687, "vgk": 1.000}
  }
}
```

*Key observations:*
- *SPY-VGK correlation 0.81: very high -- limited diversification benefit*
- *SPY-TLT correlation -0.35: negative -- bonds are a good hedge*
- *SPY-GLD correlation 0.05: near zero -- gold is a true diversifier*
- *SPY-EEM correlation 0.72: high -- EM equity moves with US equity*

---

## Phase 3: Multi-Method Optimization

### 5. Risk parity

**Tool call**: `optimize_portfolio("multi_asset_returns", method="risk_parity")`

**Response**:
```json
{
  "tool": "opt.optimize_portfolio",
  "method": "risk_parity",
  "weights": {"spy": 0.143, "tlt": 0.378, "gld": 0.291, "eem": 0.098, "vgk": 0.090},
  "portfolio_vol": 0.062,
  "portfolio_sharpe": 0.91,
  "diversification_ratio": 2.14
}
```

*Risk parity heavily weights TLT and GLD (low vol assets) to equalize
risk contributions. The diversification ratio of 2.14 is excellent.*

### 6. Maximum Sharpe

**Tool call**: `optimize_portfolio("multi_asset_returns", method="max_sharpe")`

**Response**:
```json
{
  "tool": "opt.optimize_portfolio",
  "method": "max_sharpe",
  "weights": {"spy": 0.412, "tlt": 0.321, "gld": 0.187, "eem": 0.000, "vgk": 0.080},
  "portfolio_vol": 0.098,
  "portfolio_sharpe": 1.12,
  "diversification_ratio": 1.67
}
```

*Max Sharpe concentrates in SPY (41%) and drops EEM entirely. Higher
Sharpe (1.12) but also higher vol and lower diversification.*

### 7. Hierarchical Risk Parity (HRP)

**Tool call**: `hierarchical_risk_parity("multi_asset_returns")`

**Response**:
```json
{
  "tool": "opt.hierarchical_risk_parity",
  "weights": {"spy": 0.178, "tlt": 0.342, "gld": 0.267, "eem": 0.112, "vgk": 0.101},
  "portfolio_vol": 0.071,
  "portfolio_sharpe": 0.95,
  "diversification_ratio": 1.98
}
```

*HRP produces more balanced weights than max Sharpe, without the
concentration risk. No matrix inversion needed -- robust to estimation
error.*

### 8. Minimum variance

**Tool call**: `optimize_portfolio("multi_asset_returns", method="min_variance")`

**Response**:
```json
{
  "tool": "opt.optimize_portfolio",
  "method": "min_variance",
  "weights": {"spy": 0.082, "tlt": 0.456, "gld": 0.312, "eem": 0.045, "vgk": 0.105},
  "portfolio_vol": 0.054,
  "portfolio_sharpe": 0.78,
  "diversification_ratio": 2.31
}
```

*Min variance achieves the lowest vol (5.4%) with maximum diversification
(2.31), but overweights bonds and gold.*

---

## Phase 4: Portfolio Risk Decomposition

### 9. Portfolio risk analytics

Using the HRP portfolio (best balance of Sharpe and robustness):

**Tool call**: `portfolio_risk("multi_asset_returns", weights={"spy": 0.178, "tlt": 0.342, "gld": 0.267, "eem": 0.112, "vgk": 0.101})`

**Response**:
```json
{
  "tool": "risk.portfolio_risk",
  "portfolio_vol": 0.071,
  "component_var": {
    "spy": 0.28,
    "tlt": 0.24,
    "gld": 0.18,
    "eem": 0.19,
    "vgk": 0.11
  },
  "diversification_ratio": 1.98,
  "concentration_ratio": 0.28
}
```

*SPY contributes 28% of portfolio VaR despite only 17.8% weight -- its
higher vol means it dominates risk. No single position contributes > 30%,
which is acceptable.*

### 10. VaR analysis

**Tool call**: `var_analysis("portfolio_returns_hrp", confidence=0.99, method="cornish_fisher")`

**Response**:
```json
{
  "tool": "risk.var_analysis",
  "var_95": -0.0098,
  "var_99": -0.0167,
  "cvar_95": -0.0142,
  "cvar_99": -0.0213
}
```

*99% daily VaR of -1.67%. For a $1M portfolio, the worst 1-in-100 day
loss is $16,700.*

### 11. Stress testing

**Tool call**: `stress_test("portfolio_returns_hrp")`

**Response**:
```json
{
  "tool": "risk.stress_test",
  "scenarios": {
    "gfc_2008": {"loss": -0.187},
    "covid_2020": {"loss": -0.142},
    "dot_com": {"loss": -0.098},
    "rate_hike": {"loss": -0.112},
    "vol_spike": {"loss": -0.067},
    "flash_crash": {"loss": -0.031},
    "em_crisis": {"loss": -0.054}
  }
}
```

*Worst case: GFC-like scenario with -18.7% loss. Much better than the
concentrated max Sharpe portfolio. The diversification into TLT and GLD
provides protection.*

---

## Phase 5: Regime-Aware Adjustment

### 12. Regime detection

**Tool call**: `detect_regimes("returns_spy", column="returns", method="hmm", n_regimes=2)`

**Response**:
```json
{
  "model_id": "regime_returns_spy_hmm_2state",
  "current_regime": 0,
  "statistics": {
    "regime_0": {"mean": 0.0008, "std": 0.0082, "sharpe": 1.55, "pct_time": 0.68},
    "regime_1": {"mean": -0.0004, "std": 0.0234, "sharpe": -0.27, "pct_time": 0.32}
  }
}
```

*Currently in bull regime (regime 0) with probability 0.87. The bull
regime has 3x lower vol and a Sharpe of 1.55.*

### 13. Regime-conditional optimization

The agent re-optimizes using only data from each regime, then blends
based on current regime probability.

**Agent reasoning**:
- Current regime: bull (probability 87%)
- Bull-regime weights: overweight SPY/EEM, underweight TLT
- Bear-regime weights: overweight TLT/GLD, underweight equity
- Tactical blend: 87% bull-weights + 13% bear-weights

**Final recommended weights**:
```
SPY: 21.2% (up from 17.8% -- bull tilt)
TLT: 30.1% (down from 34.2% -- less defensive in bull)
GLD: 25.4% (slight reduction)
EEM: 13.1% (up from 11.2% -- risk-on tilt)
VGK: 10.2% (slight increase)
```

---

## Phase 6: Walk-Forward Validation

### 14. Walk-forward backtest

**Tool call**: `walk_forward("multi_asset_returns", method="risk_parity", train_window=252, test_window=63)`

**Response**:
```json
{
  "tool": "backtest.walk_forward",
  "splits": 15,
  "ann_return": 0.068,
  "ann_vol": 0.071,
  "sharpe": 0.96,
  "max_drawdown": -0.112,
  "calmar": 0.61,
  "avg_turnover_per_rebalance": 0.034
}
```

*Walk-forward Sharpe of 0.96 -- very close to the in-sample Sharpe of
0.95. This means the optimization is NOT overfitting. Max drawdown of
11.2% is well within tolerance.*

### 15. Full tearsheet

**Tool call**: `comprehensive_tearsheet("walk_forward_returns")`

**Response**:
```json
{
  "tool": "backtest.comprehensive_tearsheet",
  "ann_return": 0.068,
  "ann_vol": 0.071,
  "sharpe": 0.96,
  "sortino": 1.34,
  "calmar": 0.61,
  "max_drawdown": -0.112,
  "hit_ratio": 0.543,
  "omega": 1.28,
  "recovery_factor": 4.2,
  "monthly_returns_positive": 0.67
}
```

---

## Phase 7: Final Recommendation

The agent synthesizes all findings into a recommendation:

> **Recommended Portfolio: HRP with Regime Tilt**
>
> | Asset | Weight | Risk Contribution | Rationale |
> |-------|--------|-------------------|-----------|
> | SPY | 21.2% | 28% | Core US equity, bull-regime tilt |
> | TLT | 30.1% | 22% | Rates hedge, crisis protection |
> | GLD | 25.4% | 17% | Inflation hedge, zero equity correlation |
> | EEM | 13.1% | 20% | Growth exposure, risk-on tilt |
> | VGK | 10.2% | 13% | International diversification |
>
> **Expected Characteristics**:
> - Annualized return: 6.8% (walk-forward validated)
> - Annualized volatility: 7.1%
> - Sharpe ratio: 0.96
> - Max drawdown: -11.2% (historical)
> - 99% daily VaR: -1.67%
>
> **Regime Adjustment**:
> - Current regime: bull (87% confidence)
> - Adjustment: +3.4% equity, -4.1% bonds vs strategic weights
> - Revert to strategic weights when regime confidence drops below 60%
>
> **Rebalancing Rule**: Rebalance when any position drifts > 5% from
> target or quarterly, whichever comes first. Average turnover per
> rebalance: 3.4%.
>
> **Key Risks**:
> 1. GFC-scenario loss: -18.7% (mitigant: TLT hedge, GLD diversification)
> 2. SPY-VGK correlation spike in crisis (mitigant: GLD provides crisis alpha)
> 3. Rate hike impact on TLT: -11.2% (mitigant: GLD offsets)

---

## Tool Call Summary

| Phase | Tools Used | Count |
|-------|-----------|-------|
| Data setup | `workspace_status`, `compute_returns`, `store_data`, `query_data` | 8 |
| Correlation | `correlation_analysis` | 1 |
| Optimization | `optimize_portfolio` (x3), `hierarchical_risk_parity` | 4 |
| Risk | `portfolio_risk`, `var_analysis`, `stress_test` | 3 |
| Regimes | `detect_regimes` | 1 |
| Validation | `walk_forward`, `comprehensive_tearsheet` | 2 |
| **Total** | | **19** |

The full analysis -- from raw data to a validated, regime-aware portfolio
recommendation -- required 19 tool calls across 7 modules (data, stats,
opt, risk, regimes, backtest).
