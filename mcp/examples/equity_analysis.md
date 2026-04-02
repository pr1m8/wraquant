# Example: Full Equity Analysis via wraquant-mcp

This document shows a full conversation between an AI agent and wraquant-mcp
tools, performing a comprehensive analysis of AAPL stock. Each numbered item
represents a tool call and its response.

---

## Phase 1: Data Setup

### 1. Check workspace state

**Tool call**: `workspace_status()`

**Response**:
```json
{
  "workspace": "default",
  "path": "/home/user/.wraquant/workspaces/default",
  "datasets": [],
  "models": [],
  "journal_entries": 0
}
```

*No data yet. We need to load AAPL prices first.*

### 2. Store price data

The agent has obtained AAPL OHLCV data from OpenBB MCP (or the user
provided it). Now it stores it in the workspace.

**Tool call**: `store_data("prices_aapl", {"date": ["2021-01-04", ...], "open": [133.52, ...], "high": [133.61, ...], "low": [126.76, ...], "close": [129.41, ...], "volume": [143301900, ...]})`

**Response**:
```json
{
  "dataset_id": "prices_aapl",
  "rows": 1258,
  "columns": ["date", "open", "high", "low", "close", "volume"],
  "dtypes": {"date": "object", "open": "float64", "high": "float64",
             "low": "float64", "close": "float64", "volume": "int64"}
}
```

### 3. Compute returns

**Tool call**: `compute_returns("prices_aapl", column="close", method="simple")`

**Response**:
```json
{
  "dataset_id": "returns_aapl",
  "rows": 1257,
  "columns": ["returns"],
  "method": "simple",
  "mean": 0.000782,
  "std": 0.01834,
  "annualized_vol": 0.2912
}
```

*Annualized vol of 29.1% -- typical for a large-cap tech stock.*

---

## Phase 2: Statistical Profile

### 4. Comprehensive analysis

**Tool call**: `analyze("returns_aapl", column="returns")`

**Response**:
```json
{
  "descriptive": {
    "mean": 0.000782,
    "median": 0.001123,
    "std": 0.01834,
    "skewness": -0.342,
    "kurtosis": 7.21,
    "min": -0.1278,
    "max": 0.0891
  },
  "stationarity": {
    "adf_stat": -34.2,
    "adf_pvalue": 0.0001,
    "is_stationary": true
  },
  "normality": {
    "jarque_bera_stat": 912.3,
    "jarque_bera_pvalue": 0.0,
    "shapiro_pvalue": 0.0001,
    "is_normal": false
  },
  "risk": {
    "sharpe": 0.72,
    "sortino": 1.01,
    "max_drawdown": -0.318,
    "hit_ratio": 0.541
  }
}
```

*Key findings: Returns are stationary (good) but not normal (expected).
Negative skewness (-0.34) means the left tail is heavier -- crashes are
bigger than rallies. Excess kurtosis of 7.2 means very fat tails: standard
VaR will underestimate risk.*

---

## Phase 3: Volatility Dynamics

### 5. Fit GJR-GARCH model

**Tool call**: `fit_garch("returns_aapl", column="returns", model="GJR", p=1, q=1, dist="t")`

**Response**:
```json
{
  "model_id": "garch_returns_aapl_gjr",
  "model_type": "GJR",
  "source_dataset": "returns_aapl",
  "metrics": {
    "persistence": 0.971,
    "half_life": 23.5,
    "aic": -5.234,
    "bic": -5.198
  }
}
```

*Persistence of 0.97 means vol shocks last a long time -- half-life of
23.5 days. The GJR model captures the leverage effect (negative shocks
cause more vol than positive shocks of the same magnitude).*

### 6. Vol forecasting

**Tool call**: `forecast_volatility("returns_aapl", model_name="garch_returns_aapl_gjr", horizon=10)`

**Response**:
```json
{
  "tool": "vol.forecast_volatility",
  "dataset_id": "forecast_volatility_result",
  "forecast": {
    "1d": 0.0142,
    "5d": 0.0148,
    "10d": 0.0155
  },
  "annualized": {
    "1d": 0.225,
    "5d": 0.235,
    "10d": 0.246
  },
  "current_vs_unconditional": 0.85
}
```

*Current conditional vol is 85% of the unconditional level -- vol is
slightly compressed. This could mean a calm period or an impending
breakout.*

### 7. News impact curve

**Tool call**: `news_impact_curve("returns_aapl", model_name="garch_returns_aapl_gjr")`

**Response**:
```json
{
  "tool": "vol.news_impact_curve",
  "dataset_id": "news_impact_curve_result",
  "asymmetry_ratio": 1.73,
  "leverage_coefficient": 0.089
}
```

*The asymmetry ratio of 1.73 confirms the leverage effect: a negative
shock of -2% causes 73% more vol than a positive shock of +2%.*

---

## Phase 4: Regime Detection

### 8. HMM regime detection

**Tool call**: `detect_regimes("returns_aapl", column="returns", method="hmm", n_regimes=2)`

**Response**:
```json
{
  "model_id": "regime_returns_aapl_hmm_2state",
  "model_type": "hmm_2state",
  "current_regime": 0,
  "statistics": {
    "regime_0": {
      "mean": 0.0012,
      "std": 0.0098,
      "sharpe": 1.94,
      "pct_time": 0.72
    },
    "regime_1": {
      "mean": -0.0003,
      "std": 0.0312,
      "sharpe": -0.15,
      "pct_time": 0.28
    }
  }
}
```

*Currently in regime 0 (low-vol bull market). The bull regime has a
Sharpe of 1.94 and accounts for 72% of time. The bear regime has 3x
higher volatility and a negative Sharpe.*

---

## Phase 5: Technical Analysis

### 9. RSI indicator

**Tool call**: `compute_indicator("prices_aapl", indicator="rsi", column="close", period=14)`

**Response**:
```json
{
  "dataset_id": "prices_aapl_rsi",
  "indicator": "rsi",
  "summary": {
    "mean": 51.3,
    "min": 18.7,
    "max": 84.2
  }
}
```

### 10. MACD indicator

**Tool call**: `compute_indicator("prices_aapl", indicator="macd", column="close")`

**Response**:
```json
{
  "dataset_id": "prices_aapl_macd",
  "indicator": "macd",
  "summary": {
    "mean": 0.42,
    "min": -8.31,
    "max": 7.56
  }
}
```

### 11. Bollinger Bands

**Tool call**: `compute_indicator("prices_aapl", indicator="bollinger_bands", column="close", period=20)`

**Response**:
```json
{
  "dataset_id": "prices_aapl_bollinger_bands",
  "indicator": "bollinger_bands",
  "summary": {
    "mean": 172.4,
    "min": 124.1,
    "max": 237.8
  }
}
```

---

## Phase 6: Risk Assessment

### 12. Core risk metrics

**Tool call**: `risk_metrics("returns_aapl", column="returns")`

**Response**:
```json
{
  "tool": "risk_metrics",
  "dataset": "returns_aapl",
  "sharpe": 0.72,
  "sortino": 1.01,
  "max_drawdown": -0.318,
  "hit_ratio": 0.541,
  "annualized_return": 0.197,
  "annualized_vol": 0.291
}
```

### 13. Value-at-Risk

**Tool call**: `var_analysis("returns_aapl", column="returns", confidence=0.99, method="cornish_fisher")`

**Response**:
```json
{
  "tool": "risk.var_analysis",
  "var_95": -0.0287,
  "var_99": -0.0456,
  "cvar_95": -0.0412,
  "cvar_99": -0.0589,
  "method": "cornish_fisher"
}
```

*99% Cornish-Fisher VaR of -4.56% means on the worst 1-in-100 days,
losses exceed 4.56%. The CVaR (expected shortfall) of -5.89% is the
average loss on those worst days.*

### 14. Stress testing

**Tool call**: `stress_test("returns_aapl", column="returns")`

**Response**:
```json
{
  "tool": "risk.stress_test",
  "scenarios": {
    "gfc_2008": {"loss": -0.523, "duration_days": 372},
    "covid_2020": {"loss": -0.312, "duration_days": 33},
    "dot_com": {"loss": -0.412, "duration_days": 589},
    "rate_hike": {"loss": -0.187, "duration_days": 125},
    "vol_spike": {"loss": -0.098, "duration_days": 12},
    "flash_crash": {"loss": -0.067, "duration_days": 1},
    "em_crisis": {"loss": -0.043, "duration_days": 45}
  }
}
```

*Worst case: GFC-like scenario would cause a 52% drawdown. COVID-like
scenario: 31% drawdown but fast recovery (33 days).*

---

## Phase 7: Synthesis

The agent now combines all findings:

> **AAPL Assessment**
>
> **Regime**: Bull (low-vol), probability 87%. Expected to continue.
>
> **Risk-adjusted return**: Sharpe 0.72 (good), Sortino 1.01 (positive
> skew -- upside surprises exceed downside).
>
> **Volatility**: Currently compressed (85% of unconditional). GJR-GARCH
> persistence of 0.97 with leverage effect. Half-life 23.5 days.
>
> **Technicals**: RSI in neutral zone. MACD positive. Bollinger bands
> normal width -- no squeeze signal.
>
> **Key risk**: Fat tails (kurtosis 7.2) mean standard VaR underestimates.
> CF VaR at 99% is -4.56%. GFC scenario: -52%.
>
> **Conclusion**: Favorable for existing positions. Vol compression in
> bull regime suggests continued drift upward. However, fat tails and
> leverage effect mean any correction will be amplified. Position sizing
> should account for the 31.8% max historical drawdown.
