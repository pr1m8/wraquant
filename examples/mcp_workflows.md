# wraquant-mcp Workflow Examples

This document shows five example conversations between an AI agent and
wraquant-mcp, demonstrating how the 218 tools compose into powerful
research workflows. Each example shows the tool calls, responses, and
agent reasoning in a realistic research scenario.

---

## Workflow 1: Quick Stock Analysis

**User**: "Give me a quick analysis of NVDA."

### 1.1 Store price data

The agent obtains NVDA price data (from OpenBB MCP, user upload, or
another source) and stores it in the wraquant workspace.

**Tool call**: `store_data("prices_nvda", {"date": [...], "open": [...], "high": [...], "low": [...], "close": [...], "volume": [...]})`

**Response**:
```json
{
  "dataset_id": "prices_nvda",
  "rows": 504,
  "columns": ["date", "open", "high", "low", "close", "volume"]
}
```

### 1.2 Compute returns

**Tool call**: `compute_returns("prices_nvda", column="close", method="simple")`

**Response**:
```json
{
  "dataset_id": "returns_nvda",
  "rows": 503,
  "mean": 0.002341,
  "std": 0.03412,
  "annualized_vol": 0.5417
}
```

*Annualized vol of 54% -- significantly higher than the market. This
is a high-beta growth stock.*

### 1.3 Statistical profile

**Tool call**: `analyze("returns_nvda", column="returns")`

**Response**:
```json
{
  "descriptive": {
    "mean": 0.002341,
    "std": 0.03412,
    "skewness": 0.187,
    "kurtosis": 5.82,
    "min": -0.0987,
    "max": 0.2435
  },
  "normality": {
    "jarque_bera_pvalue": 0.0,
    "is_normal": false
  },
  "risk": {
    "sharpe": 1.09,
    "sortino": 1.68,
    "max_drawdown": -0.312
  }
}
```

### 1.4 Technical snapshot

**Tool call**: `compute_indicator("prices_nvda", indicator="rsi", column="close", period=14)`
**Tool call**: `compute_indicator("prices_nvda", indicator="macd", column="close")`

**Responses**:
```json
{"indicator": "rsi", "summary": {"current": 62.4, "mean": 55.1}}
{"indicator": "macd", "summary": {"macd": 4.82, "signal": 3.21, "histogram": 1.61}}
```

### 1.5 Agent synthesis

> **NVDA Quick Analysis**
>
> | Metric | Value | Interpretation |
> |--------|-------|----------------|
> | Ann. Return | ~59% | Exceptional, driven by AI demand |
> | Ann. Vol | 54.2% | Very high -- position size accordingly |
> | Sharpe | 1.09 | Good risk-adjusted return |
> | Sortino | 1.68 | Upside surprises dominate downside |
> | Max DD | -31.2% | Significant but recovered |
> | RSI(14) | 62.4 | Neutral -- not overbought |
> | MACD | Bullish | MACD above signal line |
>
> **Bottom line**: Strong momentum with good risk-adjusted returns.
> High volatility means position sizing is critical. RSI is neutral,
> MACD is bullish -- technically favorable for continuation.

---

## Workflow 2: Portfolio Construction

**User**: "Build me a diversified portfolio with SPY, TLT, GLD, VWO,
and IEF. Optimize for risk parity."

### 2.1 Compute multi-asset returns

The agent stores prices for all five ETFs, computes returns, and
merges them into a single multi-asset dataset.

**Tool call**: `compute_returns("prices_spy")`
**Tool call**: `compute_returns("prices_tlt")`
**Tool call**: `compute_returns("prices_gld")`
**Tool call**: `compute_returns("prices_vwo")`
**Tool call**: `compute_returns("prices_ief")`

### 2.2 Correlation analysis

**Tool call**: `correlation_analysis("multi_asset_returns")`

**Response**:
```json
{
  "correlation_matrix": {
    "spy": {"spy": 1.000, "tlt": -0.412, "gld": 0.035, "vwo": 0.692, "ief": -0.378},
    "tlt": {"spy": -0.412, "tlt": 1.000, "gld": 0.245, "vwo": -0.231, "ief": 0.978},
    "gld": {"spy": 0.035, "tlt": 0.245, "gld": 1.000, "vwo": 0.167, "ief": 0.218},
    "vwo": {"spy": 0.692, "tlt": -0.231, "gld": 0.167, "vwo": 1.000, "ief": -0.198},
    "ief": {"spy": -0.378, "tlt": 0.978, "gld": 0.218, "vwo": -0.198, "ief": 1.000}
  }
}
```

*Key insight: TLT and IEF are 0.978 correlated -- they are
essentially the same position (long-duration Treasuries). The agent
should flag this and consider removing one.*

### 2.3 Risk parity optimization

**Tool call**: `optimize_portfolio("multi_asset_returns", method="risk_parity")`

**Response**:
```json
{
  "method": "risk_parity",
  "weights": {"spy": 0.121, "tlt": 0.198, "gld": 0.312, "vwo": 0.098, "ief": 0.271},
  "portfolio_vol": 0.058,
  "portfolio_sharpe": 0.87,
  "diversification_ratio": 2.38
}
```

### 2.4 Hierarchical Risk Parity (for robustness comparison)

**Tool call**: `hierarchical_risk_parity("multi_asset_returns")`

**Response**:
```json
{
  "weights": {"spy": 0.142, "tlt": 0.185, "gld": 0.298, "vwo": 0.108, "ief": 0.267},
  "portfolio_vol": 0.061,
  "portfolio_sharpe": 0.89,
  "diversification_ratio": 2.21
}
```

### 2.5 Risk decomposition

**Tool call**: `portfolio_risk("multi_asset_returns", weights={"spy": 0.121, "tlt": 0.198, "gld": 0.312, "vwo": 0.098, "ief": 0.271})`

**Response**:
```json
{
  "component_var": {
    "spy": 0.22,
    "tlt": 0.19,
    "gld": 0.21,
    "vwo": 0.20,
    "ief": 0.18
  },
  "diversification_ratio": 2.38,
  "concentration_ratio": 0.22
}
```

*Risk parity is working: each asset contributes roughly 20% of risk.
No single position dominates. However, TLT + IEF together account for
37% of risk contribution -- the agent should highlight the duration
concentration.*

### 2.6 Agent synthesis

> **Portfolio Recommendation: Risk Parity**
>
> | Asset | Weight | Risk Contribution |
> |-------|--------|-------------------|
> | SPY | 12.1% | 22% |
> | TLT | 19.8% | 19% |
> | GLD | 31.2% | 21% |
> | VWO | 9.8% | 20% |
> | IEF | 27.1% | 18% |
>
> **Warning**: TLT and IEF are 97.8% correlated. Consider replacing
> one with a different asset class (e.g., TIPS, REITs, or commodities)
> to improve true diversification.
>
> **Expected**: Ann. vol 5.8%, Sharpe 0.87, diversification ratio 2.38.

---

## Workflow 3: Regime-Aware Trading

**User**: "Detect market regimes for SPY and tell me whether to be
risk-on or risk-off right now."

### 3.1 Regime detection

**Tool call**: `detect_regimes("returns_spy", column="returns", method="hmm", n_regimes=3)`

**Response**:
```json
{
  "model_id": "regime_returns_spy_hmm_3state",
  "current_regime": 0,
  "statistics": {
    "regime_0": {"mean": 0.0009, "std": 0.0078, "sharpe": 1.83, "pct_time": 0.52},
    "regime_1": {"mean": 0.0003, "std": 0.0142, "sharpe": 0.34, "pct_time": 0.31},
    "regime_2": {"mean": -0.0012, "std": 0.0298, "sharpe": -0.64, "pct_time": 0.17}
  }
}
```

*Three distinct regimes:*
- *Regime 0 (calm bull): 52% of time, Sharpe 1.83, vol 12.4%*
- *Regime 1 (turbulent): 31% of time, Sharpe 0.34, vol 22.5%*
- *Regime 2 (crisis): 17% of time, Sharpe -0.64, vol 47.3%*

### 3.2 Rolling regime probability

**Tool call**: `rolling_regime_probability("returns_spy", model_name="regime_returns_spy_hmm_3state")`

**Response**:
```json
{
  "current_probabilities": {"regime_0": 0.82, "regime_1": 0.15, "regime_2": 0.03},
  "regime_duration": 45,
  "expected_remaining_duration": 28
}
```

### 3.3 Regime-conditional risk

**Tool call**: `var_analysis("returns_spy", column="returns", confidence=0.99)`

**Response**:
```json
{
  "var_99": -0.0312,
  "cvar_99": -0.0456
}
```

### 3.4 Regime-aware portfolio weights

**Tool call**: `regime_aware_portfolio("multi_asset_returns", model_name="regime_returns_spy_hmm_3state")`

**Response**:
```json
{
  "current_regime": 0,
  "regime_weights": {
    "regime_0": {"spy": 0.35, "tlt": 0.15, "gld": 0.20, "vwo": 0.20, "ief": 0.10},
    "regime_1": {"spy": 0.15, "tlt": 0.30, "gld": 0.25, "vwo": 0.10, "ief": 0.20},
    "regime_2": {"spy": 0.05, "tlt": 0.35, "gld": 0.35, "vwo": 0.05, "ief": 0.20}
  },
  "recommended_weights": {"spy": 0.32, "tlt": 0.17, "gld": 0.21, "vwo": 0.19, "ief": 0.11}
}
```

### 3.5 Agent synthesis

> **Regime Analysis: RISK-ON**
>
> We are in **Regime 0 (calm bull)** with 82% probability. This
> regime has been active for 45 days with an expected remaining
> duration of ~28 days.
>
> **Action**: Risk-on positioning.
> - Overweight equities (SPY 32%, VWO 19%)
> - Underweight bonds (TLT 17%)
> - Maintain gold allocation as tail hedge (GLD 21%)
>
> **Trigger to de-risk**: If regime 0 probability drops below 60%
> or regime 2 probability exceeds 15%, switch to defensive weights
> (heavy TLT/GLD, minimal equity).
>
> **Key risk**: A sudden regime 0 -> regime 2 transition (bypassing
> regime 1) is the worst case. The 99% CVaR of -4.56% translates
> to a $45,600 loss per $1M deployed.

---

## Workflow 4: Earnings Analysis

**User**: "Analyze MSFT's earnings quality and financial health.
Is the accounting trustworthy?"

### 4.1 Income analysis

**Tool call**: `income_analysis("MSFT", period="annual")`

**Response**:
```json
{
  "revenue": [211941000000, 198270000000, 168088000000, 143015000000],
  "revenue_growth": [0.069, 0.180, 0.175, 0.178],
  "gross_margin": [0.694, 0.685, 0.688, 0.678],
  "operating_margin": [0.445, 0.422, 0.417, 0.371],
  "net_margin": [0.361, 0.367, 0.363, 0.309],
  "margin_trend": "expanding",
  "revenue_cagr_3y": 0.140
}
```

### 4.2 Earnings quality

**Tool call**: `earnings_quality("MSFT")`

**Response**:
```json
{
  "quality_score": 92,
  "assessment": "high",
  "accruals_ratio": -0.045,
  "cash_conversion": 1.28,
  "details": {
    "accruals_ratio": -0.045,
    "cash_conversion_ratio": 1.28,
    "revenue_quality": 0.97,
    "earnings_persistence": 0.94
  }
}
```

*Excellent earnings quality:*
- *Negative accruals: cash flow exceeds reported earnings*
- *Cash conversion 1.28x: Microsoft generates $1.28 in cash for every
  $1 of reported earnings*
- *Revenue quality 0.97: revenue is backed by actual cash collections*

### 4.3 Financial health score

**Tool call**: `financial_health_score("MSFT")`

**Response**:
```json
{
  "score": 91,
  "grade": "A",
  "components": {
    "profitability": 98,
    "liquidity": 82,
    "solvency": 87,
    "efficiency": 95
  },
  "flags": []
}
```

### 4.4 DuPont decomposition

**Tool call**: `dupont_decomposition("MSFT")`

**Response**:
```json
{
  "roe": 0.389,
  "decomposition_3way": {
    "net_margin": 0.361,
    "asset_turnover": 0.523,
    "equity_multiplier": 2.065
  },
  "decomposition_5way": {
    "tax_burden": 0.832,
    "interest_burden": 0.992,
    "operating_margin": 0.445,
    "asset_turnover": 0.523,
    "equity_multiplier": 2.065
  }
}
```

### 4.5 Agent synthesis

> **MSFT Earnings Quality Assessment: TRUSTWORTHY**
>
> **Earnings Quality Score: 92/100 (High)**
>
> The numbers are highly credible for three reasons:
>
> 1. **Negative accruals (-4.5%)**: Microsoft's cash flow exceeds its
>    reported earnings. This is the opposite of earnings manipulation,
>    where companies inflate earnings through aggressive accruals.
>
> 2. **Cash conversion 1.28x**: For every dollar of reported earnings,
>    Microsoft collects $1.28 in cash. This is the hallmark of a high-
>    quality business with strong customer relationships and pricing power.
>
> 3. **Revenue quality 0.97**: 97% of recognized revenue is backed by
>    actual cash receipts. No channel-stuffing or aggressive revenue
>    recognition concerns.
>
> **DuPont Analysis**: The 38.9% ROE is driven by a 36.1% net margin
> (operating excellence) and moderate leverage (2.1x equity multiplier),
> NOT by financial engineering. The interest burden of 0.992 means
> interest expense barely dents profitability.
>
> **Financial Health: A (91/100)** -- no flags.
>
> **Conclusion**: Microsoft's accounting is conservative and trustworthy.
> The business is high-quality across all dimensions.

---

## Workflow 5: Risk Monitoring Dashboard

**User**: "I hold a portfolio of 60% SPY, 30% TLT, 10% GLD.
Run a full risk check."

### 5.1 Portfolio returns

The agent computes weighted portfolio returns from the stored
individual asset returns.

**Tool call**: `store_data("portfolio_returns", {...})`

### 5.2 Risk metrics

**Tool call**: `risk_metrics("portfolio_returns", column="returns")`

**Response**:
```json
{
  "sharpe": 0.92,
  "sortino": 1.31,
  "max_drawdown": -0.142,
  "annualized_return": 0.074,
  "annualized_vol": 0.081,
  "hit_ratio": 0.551
}
```

### 5.3 VaR analysis

**Tool call**: `var_analysis("portfolio_returns", column="returns", confidence=0.99, method="cornish_fisher")`

**Response**:
```json
{
  "var_95": -0.0112,
  "var_99": -0.0198,
  "cvar_95": -0.0167,
  "cvar_99": -0.0234
}
```

### 5.4 Stress testing

**Tool call**: `stress_test("portfolio_returns")`

**Response**:
```json
{
  "scenarios": {
    "gfc_2008": {"loss": -0.198},
    "covid_2020": {"loss": -0.121},
    "rate_hike_2022": {"loss": -0.178},
    "vol_spike": {"loss": -0.052},
    "flash_crash": {"loss": -0.018}
  }
}
```

### 5.5 GARCH volatility

**Tool call**: `fit_garch("portfolio_returns", model="GJR", p=1, q=1, dist="t")`

**Response**:
```json
{
  "model_id": "garch_portfolio_gjr",
  "metrics": {
    "persistence": 0.953,
    "current_conditional_vol": 0.0072,
    "unconditional_vol": 0.0081,
    "current_vs_unconditional": 0.89
  }
}
```

### 5.6 Regime check

**Tool call**: `detect_regimes("portfolio_returns", method="hmm", n_regimes=2)`

**Response**:
```json
{
  "current_regime": 0,
  "statistics": {
    "regime_0": {"mean": 0.0004, "std": 0.0058, "pct_time": 0.71},
    "regime_1": {"mean": -0.0002, "std": 0.0145, "pct_time": 0.29}
  }
}
```

### 5.7 Agent synthesis

> **RISK DASHBOARD: 60/30/10 Portfolio**
>
> ```
> STATUS: GREEN -- All risk metrics within normal bounds
> ```
>
> | Category | Metric | Value | Status |
> |----------|--------|-------|--------|
> | Return | Ann. Return | 7.4% | OK |
> | Return | Sharpe | 0.92 | Good |
> | Risk | Ann. Vol | 8.1% | Low |
> | Risk | Max DD | -14.2% | Acceptable |
> | Tail | 99% VaR | -1.98% | Within budget |
> | Tail | 99% CVaR | -2.34% | Within budget |
> | Vol | GARCH persist. | 0.953 | Monitor |
> | Vol | Current/Uncon. | 0.89 | Below avg |
> | Regime | Current | Bull (71%) | Favorable |
>
> **Stress Scenarios**:
> - Worst case (GFC replay): -19.8% -- within risk tolerance
> - Rate hike scenario: -17.8% -- main vulnerability (TLT exposure)
> - COVID replay: -12.1% -- manageable
>
> **Alerts**: None.
>
> **Risk Budget**:
> - 99% daily VaR = -1.98% -> for $1M portfolio, max daily loss $19,800
> - Annualized VaR = -1.98% x sqrt(252) = -31.4%
> - Max stress loss (GFC): -19.8% -> $198,000 on $1M
>
> **Recommendations**:
> 1. Portfolio is currently below unconditional vol (0.89x) -- vol
>    compression phase. Consider whether this is calm before a storm.
> 2. Rate sensitivity is the primary risk factor (30% TLT). If rates
>    are expected to rise, consider reducing duration.
> 3. Current regime is favorable (bull, 71% of time). No defensive
>    action needed yet.

---

## Tool Usage Summary

| Workflow | Tools Used | Module Coverage |
|----------|-----------|-----------------|
| Quick Stock Analysis | 4 tools | data, stats, risk, ta |
| Portfolio Construction | 6 tools | data, stats, opt, risk |
| Regime-Aware Trading | 5 tools | data, regimes, risk, opt |
| Earnings Analysis | 4 tools | fundamental (financials, ratios) |
| Risk Monitoring | 6 tools | risk, vol, regimes |

The wraquant-mcp server provides 218 tools across 22 modules. These
five workflows demonstrate how tools compose naturally: the output of
one tool (e.g., `compute_returns`) feeds into another (e.g.,
`fit_garch`), creating a seamless research experience without the
agent needing to write any code.
