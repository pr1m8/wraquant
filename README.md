# wraquant

[![PyPI](https://img.shields.io/pypi/v/wraquant)](https://pypi.org/project/wraquant/)
[![Tests](https://github.com/pr1m8/wraquant/actions/workflows/tests.yml/badge.svg)](https://github.com/pr1m8/wraquant/actions/workflows/tests.yml)
[![Docs](https://readthedocs.org/projects/wraquant/badge/?version=latest)](https://wraquant.readthedocs.io)
[![Python](https://img.shields.io/pypi/pyversions/wraquant)](https://pypi.org/project/wraquant/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**The ultimate quantitative finance toolkit for Python.**

1,097 functions | 3,630+ tests | 27 modules | 265 TA indicators | 100K+ LOC | MCP server for AI agents

---

## Why wraquant?

wraquant is a comprehensive, deeply integrated quant finance library that combines
risk management, regime detection, volatility modeling, derivatives pricing,
backtesting, machine learning, and technical analysis in one cohesive framework.

**wraquant + wraquant-mcp = an AI-native quant research lab.** Point Claude or any
MCP-compatible agent at your data and it has 218 production-grade tools, 327
guided workflow prompts, and a shared DuckDB workspace at its disposal. The agent
can fit GARCH models, detect regimes, optimize portfolios, run backtests, price
derivatives, and generate tearsheets -- all through structured tool calls with
persistent state. No notebooks, no glue code, no context window bloat.

Unlike libraries that wrap a single domain, wraquant's modules work **together**.
The library is organized as a six-layer directed acyclic graph where each module
knows how to feed its outputs into the next:

- Fit a GARCH model --> feed conditional vol into time-varying VaR --> stress test the portfolio --> generate a tearsheet
- Detect market regimes --> adjust portfolio weights by regime probability --> backtest --> compute regime-specific risk metrics
- Build ML features from 265 TA indicators --> walk-forward validate with purged K-fold --> deploy with regime-conditional position sizing
- Bootstrap a yield curve --> price options via characteristic functions --> compute Greeks --> hedge with Almgren-Chriss optimal execution

Every function ships with deep docstrings that explain not just _what_ it does,
but _when_ to use it, _how_ to interpret the output, and _which alternative_ to
consider. Mathematical formulations are included where applicable. References
point to the original papers.

---

## Installation

```bash
pip install wraquant
```

Install with optional dependency groups for specific capabilities:

```bash
# Common combinations
pip install wraquant[market-data,viz,risk]
pip install wraquant[ml,timeseries,backtesting]
pip install wraquant[regimes,optimization,bayes]

# Everything (large install)
pip install wraquant[market-data,timeseries,cleaning,validation,etl,workflow,ml,optimization,regimes,backtesting,risk,pricing,stochastic,causal,bayes,viz,scale,dashboard]
```

---

## Quick Start

### One-liner analysis

```python
import wraquant as wq

# Comprehensive analysis: descriptive stats, risk metrics,
# distribution fit, stationarity test, regime detection, GARCH vol
report = wq.analyze(daily_returns)
print(report["risk"]["sharpe"])
print(report["regime"]["current"])
print(report["volatility"]["persistence"])
```

### Composable workflows

Chain wraquant modules with zero glue code. Each step reads from and writes to
a shared context, automatically wiring outputs to inputs:

```python
from wraquant.compose import Workflow, steps

result = (
    Workflow("full_analysis")
    .add(steps.returns())
    .add(steps.regime_detect(n_regimes=3))
    .add(steps.garch_vol(dist="skewt"))
    .add(steps.var_analysis(confidence=0.99))
    .add(steps.stress_test())
    .add(steps.risk_metrics())
    .add(steps.tearsheet())
    .run(prices)
)

print(result.risk)          # {"sharpe": 1.34, "sortino": 1.87, ...}
print(result.regimes)       # RegimeResult(n_regimes=3, ...)
print(result.var)           # 95th percentile VaR
print(result.stress)        # Stressed metrics per scenario
```

Or use a pre-built workflow:

```python
from wraquant.compose import risk_workflow, portfolio_workflow

# Risk-focused: returns -> VaR -> GARCH VaR -> stress testing
risk_result = risk_workflow().run(prices)

# Multi-asset: returns -> optimize -> risk decomposition -> regimes
portfolio_result = portfolio_workflow().run(multi_asset_prices_df)
print(portfolio_result.weights)
```

### Regime-aware investing

```python
from wraquant.regimes import detect_regimes, regime_statistics
from wraquant.backtest import regime_signal_filter, comprehensive_tearsheet
from wraquant.risk import sharpe_ratio, conditional_var

# 1. Detect bull/bear regimes with a Gaussian HMM
regimes = detect_regimes(returns, method="hmm", n_regimes=2)
print(regimes.statistics)  # Per-regime mean, vol, Sharpe

# 2. Filter signals: only go long when P(bull) > 60%
filtered = regime_signal_filter(
    signals, regimes.probabilities[:, 0], threshold=0.6
)

# 3. Backtest and analyze
strategy_returns = returns * filtered
tearsheet = comprehensive_tearsheet(strategy_returns)
print(f"Sharpe: {sharpe_ratio(strategy_returns):.2f}")
print(f"99% CVaR: {conditional_var(strategy_returns, confidence=0.99):.4f}")
```

### Cross-module pipeline: GARCH --> VaR --> Stress Test

```python
from wraquant.vol import garch_fit, news_impact_curve
from wraquant.risk import garch_var, stress_test_returns, historical_stress_test

# Fit GJR-GARCH with Student-t errors (captures leverage + fat tails)
model = garch_fit(returns * 100, model="GJR", dist="t")
print(f"Persistence: {model['persistence']:.4f}")
print(f"Half-life: {model['half_life']:.1f} days")

# Time-varying VaR using the fitted GARCH conditional volatility
var_result = garch_var(returns, vol_model="GJR", dist="t", alpha=0.01)
print(f"Current 99% VaR: {var_result['current_var']:.4f}")
print(f"Breach rate: {var_result['breach_rate']:.2%}")

# Stress test: what happens under historical crisis scenarios?
stress = historical_stress_test(returns, scenarios=["gfc_2008", "covid_2020"])

# Visualize the leverage effect
nic = news_impact_curve(returns.values, model_type="gjr")
```

---

## Complete Module Reference

### Risk Management -- `wraquant.risk`

**95 exported functions** across 14 sub-modules.

The most comprehensive risk module in any Python quant library. Covers the full
spectrum from simple return-based metrics through tail-risk modeling, credit risk,
copula dependency, and Monte Carlo simulation.

| Sub-module              | Functions | Highlights                                                                                                                                                                        |
| ----------------------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **metrics**             | 10        | Sharpe, Sortino, Information Ratio, Treynor, M-squared, Jensen's alpha, appraisal ratio, capture ratios, hit ratio, max drawdown                                                  |
| **var**                 | 4         | Historical/parametric VaR, CVaR (Expected Shortfall), GARCH-VaR, Greeks-based VaR                                                                                                 |
| **portfolio_analytics** | 7         | Component/marginal/incremental VaR, risk budgeting, concentration ratio, tracking error, active share                                                                             |
| **beta**                | 6         | Rolling, Blume-adjusted, Vasicek-shrinkage, Dimson (illiquidity), conditional (up/down market), EWMA beta                                                                         |
| **factor**              | 4         | Factor risk model, statistical (PCA) factors, Fama-French regression, factor contribution                                                                                         |
| **tail**                | 5         | Cornish-Fisher VaR, ES decomposition, CDaR, DaR, tail ratio analysis                                                                                                              |
| **historical**          | 4         | Crisis drawdowns (top-N with lifecycle), event impact, contagion analysis, drawdown attribution                                                                                   |
| **stress**              | 11        | Return/vol/spot/correlation/liquidity stress, historical crisis replay (GFC, COVID, dot-com), reverse stress test, sensitivity ladder, joint stress, marginal stress contribution |
| **copulas**             | 8         | Gaussian, Student-t, Clayton, Gumbel, Frank copulas + simulation, tail dependence, rank correlation                                                                               |
| **dcc**                 | 4         | DCC-GARCH, rolling DCC, correlation forecasting, conditional covariance                                                                                                           |
| **credit**              | 7         | Merton structural model, Altman Z-score, default probability, credit/CDS spreads, LGD, expected loss                                                                              |
| **survival**            | 8         | Kaplan-Meier, Nelson-Aalen, Cox PH, exponential/Weibull models, hazard rate, log-rank test                                                                                        |
| **monte_carlo**         | 6         | Importance sampling VaR, antithetic variates, stratified sampling, block/stationary bootstrap, filtered historical simulation                                                     |
| **integrations**        | 7         | PyPortfolioOpt, riskfolio-lib, skfolio, copulas/copulae, vine copula, pyextremes EVT                                                                                              |

```python
from wraquant.risk import (
    sharpe_ratio, conditional_var, risk_contribution,
    cornish_fisher_var, crisis_drawdowns, fit_t_copula,
    dcc_garch, merton_model, historical_stress_test,
    importance_sampling_var, kaplan_meier,
)

# Portfolio risk decomposition
contributions = risk_contribution(weights, returns_df)  # Euler decomposition

# Copula-based tail dependence
copula = fit_t_copula(returns_df)
simulated = copula_simulate(copula, n_samples=10000)

# Credit risk: equity as a call on firm assets
pd_result = merton_model(equity=100, debt=80, vol=0.3, rate=0.05, T=1)
```

---

### Volatility Modeling -- `wraquant.vol`

**28 exported functions** spanning realized estimators, the full GARCH family,
stochastic volatility, and implied vol surfaces.

**Realized volatility estimators** (non-parametric, from OHLC data):

| Estimator                     | Efficiency vs Close-to-Close | Best For                                      |
| ----------------------------- | ---------------------------- | --------------------------------------------- |
| `realized_volatility`         | 1x (baseline)                | Simple daily close data                       |
| `parkinson`                   | ~5x                          | Range-based, no drift                         |
| `garman_klass`                | ~8x                          | Most efficient single-day OHLC                |
| `rogers_satchell`             | ~5x                          | Trending markets (handles drift)              |
| `yang_zhang`                  | ~8x                          | **Best general-purpose OHLC estimator**       |
| `bipower_variation`           | Jump-robust                  | Separating jumps from diffusion               |
| `two_scale_realized_variance` | Noise-robust                 | High-frequency data with microstructure noise |
| `realized_kernel`             | Noise-robust                 | Optimal with irregular tick data              |

**GARCH family** (parametric conditional volatility):

| Model           | Key Feature                     | When to Use                                    |
| --------------- | ------------------------------- | ---------------------------------------------- |
| `garch_fit`     | Standard GARCH(p,q)             | Starting point; captures volatility clustering |
| `egarch_fit`    | Exponential GARCH               | Leverage effect without positivity constraints |
| `gjr_garch_fit` | Asymmetric threshold            | Equities: negative returns increase vol more   |
| `figarch_fit`   | Fractionally integrated         | Long-memory vol (FX, commodities)              |
| `harch_fit`     | Heterogeneous horizons          | Mixed participant time horizons                |
| `aparch_fit`    | Asymmetric power                | Flexible power transformation                  |
| `dcc_fit`       | Dynamic Conditional Correlation | Multi-asset time-varying covariance            |

Plus: `garch_forecast`, `garch_rolling_forecast`, `garch_model_selection`,
`realized_garch`, `news_impact_curve`, `volatility_persistence`,
`stochastic_vol_sv`, `hawkes_process`, `gaussian_mixture_vol`,
`vol_surface_svi`, `variance_risk_premium`, `ewma_volatility`,
`jump_test_bns`.

```python
from wraquant.vol import (
    yang_zhang, garch_fit, egarch_fit, dcc_fit,
    news_impact_curve, garch_model_selection,
)

# Best general-purpose OHLC estimator
realized = yang_zhang(high, low, close, open_prices, window=20)

# Fit multiple GARCH variants, pick the best by BIC
best = garch_model_selection(returns, models=["GARCH", "GJR", "EGARCH"])

# Multi-asset dynamic correlations
dcc = dcc_fit(returns_df)  # Time-varying correlation and covariance
```

---

### Regime Detection -- `wraquant.regimes`

**38 exported functions** for identifying, classifying, scoring, and
exploiting market regime shifts.

**Detection methods:**

- `fit_gaussian_hmm` / `fit_hmm` -- Gaussian HMM with multi-restart EM (the workhorse)
- `fit_ms_regression` / `fit_ms_autoregression` -- Hamilton-style Markov-switching models
- `fit_multivariate_hmm` -- Joint regime detection across multiple assets
- `gaussian_mixture_regimes` -- GMM clustering (no sequential dependence)
- `online_changepoint` -- Bayesian online change-point detection (Adams & MacKay 2007)
- `pelt_changepoint` / `binary_segmentation` / `window_changepoint` -- Offline segmentation
- `cusum_control_chart` -- Quality-control-style monitoring

**Kalman filtering:**

- `kalman_filter` -- Forward pass (real-time state estimation)
- `kalman_smoother` -- RTS two-pass (optimal ex-post estimates)
- `kalman_regression` -- Time-varying betas and hedge ratios
- `unscented_kalman` -- Nonlinear state dynamics

**Regime analysis:**

- `regime_statistics` / `regime_conditional_moments` -- Per-regime return/vol/Sharpe
- `regime_transition_analysis` -- Transition probabilities, expected durations
- `regime_separation_score` / `regime_stability_score` -- Model quality metrics
- `compare_regime_methods` -- Head-to-head comparison of detection approaches
- `label_regimes` / `volatility_regime_labels` / `trend_regime_labels` -- Rule-based labels
- `regime_duration_analysis` -- How long does each regime last?

**Portfolio integration:**

- `regime_aware_portfolio` -- Adjust weights by current regime probability
- `rolling_regime_probability` -- Time-varying posterior for blending strategies

**Third-party integrations:** pomegranate (GPU-accelerated HMM), filterpy,
pykalman, dynamax (JAX-based LGSSM), river (streaming drift detection).

```python
from wraquant.regimes import (
    detect_regimes, fit_gaussian_hmm, kalman_regression,
    regime_statistics, regime_aware_portfolio,
    online_changepoint, select_n_states,
)

# Unified interface: method can be "hmm", "gmm", "ms_regression", etc.
result = detect_regimes(returns, method="hmm", n_regimes=3)
print(result.current_regime)        # 0, 1, or 2
print(result.current_probabilities) # [0.05, 0.85, 0.10]
print(result.statistics)            # Per-regime DataFrame

# Track time-varying hedge ratio with Kalman filter
beta_t = kalman_regression(asset_returns, factor_returns)

# Find optimal number of regimes by BIC
best_k = select_n_states(returns, max_states=5)
```

---

### Technical Analysis -- `wraquant.ta`

**265 indicators** across **19 sub-modules**, covering the complete taxonomy
of technical analysis from classical chart overlays to exotic oscillators.

| Sub-module             | Count | Indicators                                                                                                                                                                                                                                                                  |
| ---------------------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **overlap**            | 12    | SMA, EMA, WMA, DEMA, TEMA, KAMA, VWAP, Supertrend, Ichimoku, Bollinger Bands, Keltner Channel, Donchian                                                                                                                                                                     |
| **momentum**           | 22    | RSI, MACD, Stochastic, Williams %R, CCI, ROC, TSI, CMO, DPO, PPO, Awesome Oscillator, Ultimate Oscillator, Stochastic RSI, SMI, Schaff, Squeeze Histogram, PMO, Klinger, Inertia, CoG, Psychological Line                                                                   |
| **trend**              | 18    | ADX, Aroon, PSAR, Vortex, TRIX, Heikin Ashi, ZigZag, McGinley Dynamic, Schaff Trend Cycle, GUPPY MMA, Rainbow MA, Hull MA, Zero-Lag EMA, VIDYA, Tilson T3, Fractal Adaptive MA, Linear Regression + Slope                                                                   |
| **volume**             | 10    | OBV, A/D Line, CMF, MFI, EOM, Force Index, NVI, PVI, VPT, A/D Oscillator                                                                                                                                                                                                    |
| **volatility**         | 18    | ATR, True Range, NATR, Bollinger Width, Keltner Width, Chaikin Volatility, Historical Vol, Mass Index, Ulcer Index, RVI, Acceleration Bands, Std Dev, Variance, + OHLC estimators (Garman-Klass, Parkinson, Rogers-Satchell, Yang-Zhang, Close-to-Close)                    |
| **patterns**           | 37    | Doji, Hammer, Engulfing, Morning/Evening Star, Three White Soldiers, Three Black Crows, Harami, Spinning Top, Marubozu, Piercing, Dark Cloud, Shooting Star, Tweezer Top/Bottom, Abandoned Baby, Kicking, Belt Hold, Rising/Falling Three Methods, and 15 more              |
| **candles**            | 12    | Body size, range, shadow ratios, body-to-range, direction, momentum, gap, inside/outside bar, pin bar                                                                                                                                                                       |
| **price_action**       | 10    | Higher highs/lows, swing high/low, trend bars, gap analysis, range expansion, narrow range (NR4/NR7), wide range bar, key reversal, pivot reversal                                                                                                                          |
| **signals**            | 9     | crossover, crossunder, above, below, rising, falling, highest, lowest, normalize                                                                                                                                                                                            |
| **breadth**            | 10    | Advance/Decline line + ratio, McClellan Oscillator + Summation, Arms Index (TRIN), new highs/lows, percent above MA, high/low index, bullish percent, cumulative volume index                                                                                               |
| **statistics**         | 12    | Z-score, percentile rank, skewness, kurtosis, entropy, Hurst exponent, beta, correlation, R-squared, information coefficient, mean deviation, median                                                                                                                        |
| **cycles**             | 8     | Hilbert Transform (dominant period, trend mode, instantaneous phase), sine wave, Even Better Sinewave, bandpass filter, roofing filter, decycler                                                                                                                            |
| **fibonacci**          | 6     | Retracements, extensions, fans, time zones, pivot points, auto-Fibonacci                                                                                                                                                                                                    |
| **smoothing**          | 12    | ALMA, JMA, Butterworth, Super Smoother, Gaussian, Hann/Hamming window MA, LSMA, SWMA, TRIMA, SinEMA, Kaufman Efficiency Ratio                                                                                                                                               |
| **custom**             | 16    | Squeeze Momentum, Anchored VWAP, Ehlers Fisher, Adaptive RSI, Linear Regression Channel, Market Structure, Volume-Weighted MACD, Pivot Points, Swing Points, Standard Error Bands, R-squared indicator, Polynomial/Raff Regression, Detrended Regression, Relative Strength |
| **exotic**             | 15    | Choppiness Index, Random Walk Index, Polarized Fractal Efficiency, Ergodic Oscillator, Elder Thermometer, KAIRI, Connors TPS, PZO, Pretty Good Oscillator, Market Facilitation Index, Gopalakrishnan Range, Efficiency Ratio, Trend Intensity, DMI, Relative Momentum Index |
| **support_resistance** | 6     | Algorithmic S/R detection, fractal levels, price clustering, round numbers, supply/demand zones, trendline detection                                                                                                                                                        |
| **performance**        | 10    | Relative performance, Mansfield RSI, alpha, tracking error, up/down capture, drawdown, rolling max drawdown, pain index, gain/loss ratio, profit factor                                                                                                                     |

```python
from wraquant.ta import (
    rsi, macd, bollinger_bands, supertrend, adx,
    atr, obv, ichimoku, squeeze_momentum, hurst_exponent,
    find_support_resistance, fibonacci_retracements,
)

# Trend identification
trend_strength = adx(high, low, close, period=14)
bands = bollinger_bands(close, period=20, std_dev=2)
cloud = ichimoku(high, low, close)

# Volatility-normalized position sizing
volatility = atr(high, low, close, period=14)

# Feature engineering for ML
features = {
    "rsi_14": rsi(close, period=14),
    "macd_hist": macd(close)["histogram"],
    "squeeze": squeeze_momentum(high, low, close),
    "hurst": hurst_exponent(close, window=100),
}
```

---

### Machine Learning -- `wraquant.ml`

**44 exported functions** implementing the full ML pipeline for financial
prediction, designed to avoid the pitfalls that make naive ML on financial
data fail (lookahead bias, non-stationarity, overfitting on noise).

**Feature engineering** (11 functions):
`return_features`, `volatility_features`, `technical_features`, `ta_features`,
`rolling_features`, `microstructure_features`, `label_fixed_horizon`,
`label_triple_barrier` (de Prado), `interaction_features`,
`cross_asset_features`, `regime_features`.

**Preprocessing** (5 functions):
`purged_kfold`, `combinatorial_purged_kfold` (de Prado Ch. 12),
`fractional_differentiation` (preserve memory while achieving stationarity),
`denoised_correlation` (Marcenko-Pastur RMT), `detoned_correlation`.

**Models** (5 functions):
`walk_forward_train`, `ensemble_predict`, `feature_importance_mdi`,
`feature_importance_mda` (preferred -- accounts for substitution),
`sequential_feature_selection`.

**Deep learning** (6 functions, requires PyTorch):
`lstm_forecast`, `gru_forecast`, `transformer_forecast`,
`multivariate_lstm_forecast`, `temporal_fusion_transformer` (interpretable
with variable selection), `autoencoder_features` (VAE anomaly detection).

**Advanced sklearn** (6 functions):
`svm_classifier`, `random_forest_importance`, `gradient_boost_forecast`,
`gaussian_process_regression`, `isolation_forest_anomaly`, `pca_factor_model`.

**Online learning** (2 functions):
`online_linear_regression`, `exponential_weighted_regression`.

**Pipeline & evaluation** (7 functions):
`FinancialPipeline` (sklearn Pipeline with chronological splitting),
`walk_forward_backtest`, `feature_importance_shap`, `classification_metrics`,
`financial_metrics`, `learning_curve`, `backtest_predictions`.

**Clustering** (3 functions):
`correlation_clustering`, `regime_clustering`, `optimal_clusters`.

```python
from wraquant.ml import (
    technical_features, label_triple_barrier, purged_kfold,
    walk_forward_train, feature_importance_shap,
    fractional_differentiation, FinancialPipeline,
)

# Triple-barrier labeling (de Prado): which barrier gets hit first?
labels = label_triple_barrier(close, upper=0.02, lower=-0.01, max_holding=10)

# Fractional differentiation: stationarity while preserving memory
stationary_prices = fractional_differentiation(close, d=0.4)

# Walk-forward with purged cross-validation (no lookahead)
results = walk_forward_train(
    model, X, y,
    train_size=252, test_size=21, step_size=21,
)
```

---

### Derivatives Pricing -- `wraquant.price`

**50 exported functions** covering classical pricing, characteristic function
methods, FBSDE solvers, stochastic process simulation, and fixed income.

**Options pricing:**
`black_scholes`, `binomial_tree`, `monte_carlo_option`.

**Greeks:**
`delta`, `gamma`, `theta`, `vega`, `rho`, `all_greeks`.

**Implied volatility:**
`implied_volatility`, `vol_smile`, `vol_surface`.

**Characteristic function pricing** (FFT/COS methods):
`characteristic_function_price` (unified interface),
`heston_characteristic`, `vg_characteristic`, `nig_characteristic`,
`cgmy_characteristic`, `fft_option_price`, `cos_method`,
`vg_european_fft`, `nig_european_fft`.

**FBSDE solvers:**
`fbsde_european` (European derivatives via forward-backward SDEs),
`deep_bsde` (neural network BSDE solver for high-dimensional problems),
`reflected_bsde` (American options with early exercise).

**Stochastic process simulators:**
`geometric_brownian_motion`, `heston`, `jump_diffusion`,
`ornstein_uhlenbeck`, `cir_process`, `simulate_sabr`,
`simulate_rough_bergomi`, `simulate_3_2_model`,
`simulate_cir`, `simulate_vasicek`.

**Fixed income:**
`bond_price`, `bond_yield`, `duration`, `modified_duration`,
`convexity`, `zero_rate`, `bootstrap_zero_curve`,
`interpolate_curve`, `forward_rate`, `discount_factor`.

**Integrations:** QuantLib, FinancePy, rateslib, py-vollib, sdeint.

```python
from wraquant.price import (
    black_scholes, heston_characteristic, characteristic_function_price,
    simulate_sabr, bootstrap_zero_curve, delta, all_greeks,
)

# Classical BS pricing
price = black_scholes(S=100, K=105, T=0.25, r=0.05, sigma=0.2, option_type="call")
greeks = all_greeks(S=100, K=105, T=0.25, r=0.05, sigma=0.2)

# Heston stochastic vol pricing via FFT
char_fn = heston_characteristic(S=100, K=105, T=0.25, r=0.05,
                                 v0=0.04, kappa=2.0, theta=0.04,
                                 sigma=0.3, rho=-0.7)
heston_price = characteristic_function_price(char_fn, K=105, T=0.25, r=0.05)

# Yield curve bootstrapping
curve = bootstrap_zero_curve(tenors=[0.25, 0.5, 1, 2, 5, 10],
                              rates=[0.04, 0.042, 0.045, 0.048, 0.05, 0.052])
```

---

### Time Series -- `wraquant.ts`

**52 exported functions** for decomposition, forecasting, stationarity testing,
anomaly detection, and stochastic process modeling.

**Decomposition** (7 methods):
`seasonal_decompose`, `stl_decompose`, `trend_filter` (HP filter),
`ssa_decompose` (Singular Spectrum Analysis), `emd_decompose` (Empirical Mode),
`wavelet_decompose`, `unobserved_components`.

**Forecasting** (12 functions):
`auto_forecast` (automatic model selection), `theta_forecast`,
`ses_forecast`, `holt_winters_forecast`, `exponential_smoothing`,
`ensemble_forecast`, `rolling_forecast`, `auto_arima`,
`arima_diagnostics`, `arima_model_selection`, `forecast_evaluation`,
`garch_residual_forecast`.

**Stationarity** (8 functions):
`adf_test`, `kpss_test`, `phillips_perron`, `variance_ratio_test`,
`difference`, `fractional_difference`, `detrend`, `optimal_differencing`.

**Stochastic forecasting** (4 functions):
`ornstein_uhlenbeck_forecast`, `jump_diffusion_forecast`,
`regime_switching_forecast`, `var_forecast`.

**Seasonality** (5 functions):
`detect_seasonality`, `fourier_features`, `multi_fourier_features`,
`seasonal_strength`, `multi_seasonal_decompose`.

**Features** (3 functions):
`autocorrelation_features`, `spectral_features`, `complexity_features`.

**Anomaly detection** (3 functions):
`isolation_forest_ts`, `prophet_anomaly`, `grubbs_test_ts`.

**Change-point detection** (2 functions):
`cusum`, `detect_changepoints`.

**Advanced integrations:** tsfresh, stumpy (matrix profile), sktime,
statsforecast, tslearn (DTW, k-means), darts, wavelet transform/denoise.

```python
from wraquant.ts import (
    auto_forecast, ssa_decompose, adf_test,
    ornstein_uhlenbeck_forecast, detect_seasonality,
)

# Automatic model selection and forecasting
forecast = auto_forecast(returns, h=21)  # 21-day ahead

# Singular Spectrum Analysis: extract trend + oscillatory components
components = ssa_decompose(prices, n_components=5)

# Mean-reversion modeling for pairs trading
ou = ornstein_uhlenbeck_forecast(spread, horizon=10)
print(f"Half-life: {ou['half_life']:.1f} days")
```

---

### Statistics -- `wraquant.stats`

**79 exported functions** for robust statistics, advanced correlation,
distribution fitting, regression, diagnostics, factor analysis, and
cointegration.

**Robust statistics** (8 functions):
`mad`, `winsorize`, `trimmed_mean`, `trimmed_std`, `robust_zscore`,
`robust_covariance` (MCD estimator), `huber_mean`, `outlier_detection`.

**Distributions** (11 functions):
`fit_distribution`, `fit_stable_distribution`, `best_fit_distribution`
(ranks 80+ distributions by fit), `kernel_density_estimate`,
`tail_ratio`, `tail_index`, `hurst_exponent`, `qqplot_data`,
`jarque_bera`, `kolmogorov_smirnov`, `anderson_darling`.

**Correlation** (9 functions):
`correlation_matrix`, `shrunk_covariance` (Ledoit-Wolf),
`rolling_correlation`, `partial_correlation`, `distance_correlation`,
`kendall_tau`, `mutual_information`,
`correlation_significance`, `minimum_spanning_tree_correlation`.

**Dependence** (4 functions):
`tail_dependence_coefficient`, `copula_selection`,
`rank_correlation_matrix`, `concordance_index`.

**Regression** (5 functions):
`ols`, `wls`, `rolling_ols`, `fama_macbeth` (cross-sectional),
`newey_west_ols` (HAC standard errors).

**Diagnostics** (9 functions):
`test_normality`, `test_stationarity`, `test_autocorrelation`,
`shapiro_wilk`, `durbin_watson`, `breusch_pagan`, `white_test`,
`chow_test`, `variance_inflation_factor`.

**Cointegration** (8 functions):
`engle_granger`, `johansen`, `half_life`, `spread`, `zscore_signal`,
`hedge_ratio`, `pairs_backtest_signals`, `find_cointegrated_pairs`.

**Factor analysis** (15 functions):
`pca_factors`, `factor_loadings`, `varimax_rotation`,
`factor_mimicking_portfolios`, `risk_factor_decomposition`,
`fama_french_factors`, `fama_french_regression`, `factor_attribution`,
`information_coefficient`, `quantile_analysis`, and more.

**Descriptive** (10 functions):
`summary_stats`, `annualized_return`, `annualized_volatility`,
`max_drawdown`, `calmar_ratio`, `omega_ratio`, `rolling_sharpe`,
`rolling_drawdown`, `return_attribution`, `risk_contribution`.

```python
from wraquant.stats import (
    engle_granger, half_life, find_cointegrated_pairs,
    best_fit_distribution, shrunk_covariance, fama_macbeth,
)

# Pairs trading: find cointegrated pairs across a universe
pairs = find_cointegrated_pairs(prices_df, significance=0.05)
for pair in pairs:
    hl = half_life(pair["spread"])
    print(f"{pair['asset1']}/{pair['asset2']}: half-life={hl:.1f} days")

# Best-fit distribution: ranks 80+ distributions
best = best_fit_distribution(returns)
print(f"Best fit: {best['distribution']} (p={best['p_value']:.4f})")
```

---

### Portfolio Optimization -- `wraquant.opt`

**26 exported functions** for portfolio construction, mathematical programming,
and multi-objective optimization.

**Portfolio optimization:**
`mean_variance`, `min_volatility`, `max_sharpe`, `risk_parity`,
`equal_weight`, `inverse_volatility`, `hierarchical_risk_parity` (HRP),
`black_litterman`.

**Convex optimization** (CVXPY-backed):
`minimize_quadratic`, `solve_qp`, `solve_socp`, `solve_sdp`.

**Linear programming** (PuLP-backed):
`solve_lp`, `solve_milp`, `transportation_problem`.

**Nonlinear optimization** (scipy-backed):
`minimize`, `global_minimize`, `root_find`.

**Multi-objective:**
`pareto_front`, `nsga2` (NSGA-II evolutionary), `epsilon_constraint`.

**Constraint utilities:**
`weight_constraint`, `sum_to_one_constraint`, `sector_constraints`,
`turnover_constraint`, `cardinality_constraint`.

Result types: `OptimizationResult`, `Constraint`, `Objective`.

```python
from wraquant.opt import (
    risk_parity, black_litterman, hierarchical_risk_parity,
    sector_constraints, turnover_constraint,
)

# Risk parity with sector constraints
constraints = sector_constraints(
    assets, sectors, upper={"Tech": 0.3, "Finance": 0.25}
)
result = risk_parity(returns_df, constraints=constraints)
print(result.weights)       # Array of optimal weights
print(result.volatility)    # Portfolio vol

# Black-Litterman with views
bl = black_litterman(
    returns_df, market_caps=caps,
    views={"AAPL": 0.10, "GOOGL": 0.08},  # Expected returns
    view_confidences=[0.6, 0.8],
)
```

---

### Backtesting -- `wraquant.backtest`

**38 exported functions** for strategy simulation, performance measurement,
position sizing, and reporting.

**Engine:**
`Backtest`, `VectorizedBacktest`, `walk_forward_backtest`.

**Strategy:**
`Strategy` base class.

**Performance metrics** (16 functions):
`performance_summary`, `omega_ratio`, `burke_ratio`,
`ulcer_performance_index`, `kappa_ratio`, `tail_ratio`,
`common_sense_ratio`, `rachev_ratio`, `gain_to_pain_ratio`,
`risk_of_ruin`, `kelly_fraction`, `expectancy`, `profit_factor`,
`payoff_ratio`, `recovery_factor`, `system_quality_number` (SQN).

**Position sizing** (7 functions):
`PositionSizer`, `clip_weights`, `invert_signal`,
`rebalance_threshold`, `risk_parity_position`,
`regime_conditional_sizing`, `regime_signal_filter`.

**Event tracking** (5 functions):
`Event`, `EventTracker`, `EventType`,
`detect_drawdown_events`, `detect_regime_changes`.

**Tearsheet** (7 functions):
`comprehensive_tearsheet`, `generate_tearsheet`, `drawdown_table`,
`monthly_returns_table`, `rolling_metrics_table`,
`strategy_comparison`, `trade_analysis`.

**Integrations:** vectorbt, quantstats, empyrical, pyfolio, ffn.

```python
from wraquant.backtest import (
    VectorizedBacktest, comprehensive_tearsheet,
    regime_signal_filter, kelly_fraction,
    system_quality_number, walk_forward_backtest,
)

# Vectorized backtest with regime filter
bt = VectorizedBacktest(signals, returns)
result = bt.run()

# Comprehensive tearsheet: 30+ metrics, drawdown table, monthly returns
tearsheet = comprehensive_tearsheet(strategy_returns)
print(f"SQN: {system_quality_number(strategy_returns):.2f}")
print(f"Kelly: {kelly_fraction(strategy_returns):.2%}")
```

---

### Econometrics -- `wraquant.econometrics`

**34 exported functions** for panel data estimation, instrumental variables,
VAR/VECM time series models, event studies, and regression diagnostics.

**Panel data:** `pooled_ols`, `fixed_effects`, `random_effects`,
`hausman_test`, `between_effects`, `first_difference`.

**Cross-sectional:** `robust_ols`, `quantile_regression`,
`two_stage_least_squares`, `gmm_estimation`, `sargan_test`.

**Time series:** `var_model`, `vecm_model`, `granger_causality`,
`impulse_response`, `variance_decomposition`, `structural_break_test`.

**Diagnostics:** `durbin_watson`, `breusch_godfrey`, `breusch_pagan`,
`white_test`, `jarque_bera`, `ramsey_reset`, `vif`, `condition_number`.

**Event study:** `event_study`, `cumulative_abnormal_return`,
`buy_and_hold_abnormal_return`.

**Volatility:** `garch`, `egarch`, `gjr_garch`, `dcc_garch`, `arch_test`,
`garch_numpy_fallback`.

```python
from wraquant.econometrics import (
    fixed_effects, hausman_test, var_model,
    impulse_response, event_study, two_stage_least_squares,
)

# Panel data: fixed vs random effects
fe = fixed_effects(panel_df, y="returns", x=["size", "btm"], entity="ticker")
re = random_effects(panel_df, y="returns", x=["size", "btm"], entity="ticker")
hausman = hausman_test(fe, re)  # p < 0.05 => use fixed effects

# VAR model and impulse responses
var = var_model(macro_df, lags=4)
irf = impulse_response(var, periods=20)
```

---

### Market Microstructure -- `wraquant.microstructure`

**33 exported functions** for liquidity measurement, order flow toxicity,
and market quality assessment.

**Liquidity** (14 functions):
`amihud_illiquidity`, `amihud_rolling`, `roll_spread`,
`corwin_schultz_spread`, `effective_spread`, `realized_spread`,
`closing_quoted_spread`, `kyle_lambda`, `lambda_kyle_rolling`,
`price_impact`, `spread_decomposition`, `depth_imbalance`,
`liquidity_commonality`, `turnover_ratio`.

**Toxicity** (9 functions):
`pin_model` (Probability of Informed Trading), `adjusted_pin`,
`vpin` (Volume-Synchronized PIN), `bulk_volume_classification`,
`order_flow_imbalance`, `information_share`,
`informed_trading_intensity`, `trade_classification` (Lee-Ready),
`toxicity_index`.

**Market quality** (10 functions):
`quoted_spread`, `relative_spread`, `depth`, `resiliency`,
`variance_ratio` (market efficiency), `market_efficiency_ratio`,
`hasbrouck_information_share`, `gonzalo_granger_component`,
`intraday_volatility_pattern`, `price_impact_regression`.

```python
from wraquant.microstructure import (
    amihud_illiquidity, vpin, corwin_schultz_spread,
    hasbrouck_information_share,
)

# Amihud illiquidity ratio
illiquidity = amihud_illiquidity(returns, volume)

# VPIN: real-time toxicity indicator (Easley, Lopez de Prado, O'Hara)
toxicity = vpin(volume, close, bucket_size=50)

# High-low spread estimator (no bid-ask data needed)
spread = corwin_schultz_spread(high, low)
```

---

### Execution Algorithms -- `wraquant.execution`

**20 exported functions** for trade scheduling, optimal execution, and
transaction cost analysis.

**Scheduling algorithms:**
`twap_schedule`, `vwap_schedule`, `pov_schedule` (Percentage of Volume),
`is_schedule` (Implementation Shortfall), `participation_rate_schedule`,
`adaptive_schedule`, `arrival_price_benchmark`, `close_auction_allocation`,
`implementation_shortfall`.

**Optimal execution:**
`almgren_chriss` (mean-variance optimal with temporary + permanent impact),
`bertsimas_lo` (dynamic programming), `optimal_execution_cost`,
`execution_frontier` (risk-cost trade-off curve).

**Transaction cost analysis:**
`slippage`, `commission_cost`, `total_cost`, `market_impact_model`,
`liquidity_adjusted_cost`, `expected_cost_model`,
`transaction_cost_analysis`.

```python
from wraquant.execution import almgren_chriss, vwap_schedule, total_cost

# Almgren-Chriss optimal execution
schedule = almgren_chriss(
    shares=100_000, T=1.0, sigma=0.02,
    eta=0.01, gamma=0.1, lambda_=1e-6,
)

# VWAP schedule for passive execution
vwap = vwap_schedule(shares=50_000, volume_profile=hist_volume)
```

---

### Causal Inference -- `wraquant.causal`

**15 exported functions** for treatment effect estimation, policy evaluation,
and causal discovery in financial settings.

`propensity_score`, `ipw_ate` (Inverse Probability Weighting),
`matching_ate`, `doubly_robust_ate`,
`regression_discontinuity` / `regression_discontinuity_robust`,
`synthetic_control` / `synthetic_control_weights`,
`diff_in_diff`, `granger_causality`, `instrumental_variable`,
`event_study`, `causal_forest`, `mediation_analysis`,
`bounds_analysis` (partial identification).

Integrations: DoWhy, EconML, DoubleML.

```python
from wraquant.causal import (
    diff_in_diff, synthetic_control, granger_causality,
)

# Did a regulatory change affect trading volume?
did = diff_in_diff(treated, control, pre_period, post_period)
print(f"ATT: {did['att']:.4f} (p={did['p_value']:.4f})")

# Synthetic control for single-unit treatment
sc = synthetic_control(treated_unit, donor_pool, pre_periods=60)
```

---

### Bayesian Inference -- `wraquant.bayes`

**29 exported functions** for Bayesian estimation, MCMC sampling,
model comparison, and Bayesian portfolio construction.

**Models** (15 functions):
`bayesian_regression`, `bayesian_linear_regression`,
`bayesian_sharpe` (posterior distribution of the Sharpe ratio),
`bayesian_portfolio`, `bayesian_portfolio_bl` (Bayesian Black-Litterman),
`bayesian_var`, `bayesian_volatility`,
`bayesian_factor_model`, `bayesian_changepoint`,
`bayesian_cointegration`, `bayesian_regime_inference`,
`credible_interval`, `bayes_factor`, `posterior_predictive`,
`model_comparison` (WAIC/LOO).

**MCMC** (8 functions):
`metropolis_hastings`, `hamiltonian_monte_carlo`, `gibbs_sampler`,
`slice_sampler`, `nuts_diagnostic`, `trace_summary`,
`gelman_rubin`, `convergence_diagnostics`.

**Integrations** (lazy-loaded):
`pymc_regression`, `arviz_summary`, `numpyro_regression`,
`bambi_regression`, `emcee_sample`, `blackjax_nuts`.

```python
from wraquant.bayes import (
    bayesian_sharpe, bayesian_volatility,
    hamiltonian_monte_carlo, model_comparison,
)

# Posterior distribution of the Sharpe ratio
sharpe_posterior = bayesian_sharpe(returns, n_samples=5000)
print(f"Posterior mean: {sharpe_posterior['mean']:.3f}")
print(f"95% credible interval: {sharpe_posterior['ci_95']}")

# Bayesian stochastic volatility via MCMC
vol_posterior = bayesian_volatility(returns, method="nuts")
```

---

### Forex Analysis -- `wraquant.forex`

**23 exported functions** for currency pair analysis, carry trade modeling,
session analytics, and FX risk.

**Pairs:** `CurrencyPair`, `cross_rate`, `major_pairs`,
`correlation_matrix`, `currency_strength`, `volatility_by_session`.

**Analysis:** `pips`, `pip_value`, `pip_distance`, `lot_size`,
`spread_cost`, `position_value`, `risk_reward_ratio`, `margin_call_price`.

**Sessions:** `ForexSession`, `current_session`, `session_overlaps`.

**Carry trade:** `carry_return`, `carry_attractiveness`, `carry_portfolio`,
`interest_rate_differential`, `forward_premium`,
`uncovered_interest_parity`.

**Risk:** `fx_portfolio_risk`.

```python
from wraquant.forex import (
    CurrencyPair, carry_portfolio, volatility_by_session,
)

# Carry trade analysis
portfolio = carry_portfolio(
    pairs=["AUDJPY", "NZDJPY", "USDBRL"],
    rate_differentials=[0.035, 0.04, 0.08],
)

# When is EUR/USD most volatile?
vol = volatility_by_session("EURUSD", returns)
```

---

### Visualization -- `wraquant.viz`

**47 exported functions** for publication-quality charts and interactive
dashboards, all Plotly-powered with a dark theme.

**Multi-panel dashboards:**
`portfolio_dashboard`, `regime_dashboard`, `risk_dashboard`,
`technical_dashboard`.

**Interactive Plotly charts:**
`plotly_returns`, `plotly_drawdown`, `plotly_rolling_stats`,
`plotly_distribution`, `plotly_correlation_heatmap`,
`plotly_efficient_frontier`, `plotly_risk_return_scatter`,
`plotly_regime_overlay`, `plotly_vol_surface`, `plotly_term_structure`,
`plotly_copula_scatter`, `plotly_network_graph`, `plotly_sankey_flow`,
`plotly_treemap`, `plotly_radar`.

**Candlestick / OHLCV:**
`plotly_candlestick`, `plotly_heikin_ashi`, `plotly_market_profile`,
`plotly_renko`.

**Matplotlib charts:**
Returns, drawdowns, distribution, rolling returns, monthly heatmap,
portfolio weights, efficient frontier, risk contribution, correlation matrix,
decomposition, regime overlay, VaR backtest, rolling volatility, tail distribution.

**Rich standalone charts:**
`plot_multi_asset`, `plot_vol_surface`, `plot_distribution_analysis`,
`plot_correlation_network`, `plot_backtest_tearsheet`.

**Auto-detection:** `auto_plot` -- automatically chooses the best
visualization based on data type.

```python
from wraquant.viz import (
    portfolio_dashboard, plotly_vol_surface,
    plot_backtest_tearsheet, auto_plot,
)

# One-line visualization -- auto-detects data type
auto_plot(strategy_returns)  # -> distribution + cumulative returns
auto_plot(returns_df)        # -> correlation heatmap

# Interactive dashboards
portfolio_dashboard(weights, returns_df)
risk_dashboard(returns, var_result, stress_result)
```

---

### Advanced Mathematics -- `wraquant.math`

**22 exported functions** for financial network analysis, Levy processes,
and optimal stopping theory.

**Network analysis** (7 functions):
`correlation_network`, `minimum_spanning_tree`,
`centrality_measures`, `community_detection`,
`systemic_risk_score`, `contagion_simulation`, `granger_network`.

**Levy processes** (9 functions):
`variance_gamma_pdf` / `variance_gamma_simulate` / `fit_variance_gamma`,
`nig_pdf` / `nig_simulate` / `fit_nig`,
`cgmy_simulate`, `levy_stable_simulate`,
`characteristic_function_vg`.

**Optimal stopping** (6 functions):
`longstaff_schwartz` (American option pricing via regression),
`binomial_american`, `optimal_exit_threshold`,
`sequential_probability_ratio` (SPRT for strategy evaluation),
`cusum_stopping`, `secretary_problem_threshold`.

```python
from wraquant.math import (
    correlation_network, systemic_risk_score,
    fit_variance_gamma, longstaff_schwartz,
)

# Financial network analysis
network = correlation_network(returns_df, threshold=0.5)
risk_scores = systemic_risk_score(network)

# Fat-tailed return modeling with Variance-Gamma
vg_params = fit_variance_gamma(returns)
```

---

### Data -- `wraquant.data`

**41 exported functions** for fetching, cleaning, validating, and
transforming financial data from multiple sources.

**Providers:**
`DataProvider`, `ProviderRegistry`, `fetch_prices` (Yahoo Finance),
`fetch_ohlcv`, `fetch_macro` (FRED, NASDAQ Data Link), `list_providers`.

**Cleaning** (8 functions):
`align_series`, `detect_outliers`, `fill_missing`,
`handle_splits_dividends`, `remove_duplicates`, `remove_outliers`,
`resample_ohlcv`, `winsorize`.

**Transforms** (8 functions):
`to_returns`, `to_prices`, `to_excess_returns`,
`normalize_prices`, `rolling_zscore`, `expanding_zscore`,
`percentile_rank`, `rank_transform`.

**Validation** (5 functions):
`validate_returns`, `validate_ohlcv`, `check_completeness`,
`check_staleness`, `data_quality_report`.

**Advanced cleaning** (pyjanitor, fuzzy matching, date parsing):
`janitor_clean_names`, `janitor_remove_empty`, `fuzzy_merge`,
`parse_dates_flexible`, `parse_prices`, `normalize_countries`, `fix_text`.

**Advanced validation** (pandera schemas):
`pandera_validate`, `create_ohlcv_schema`, `create_returns_schema`.

```python
from wraquant.data import (
    fetch_prices, fetch_macro, validate_returns,
    data_quality_report, to_returns,
)

# Fetch and validate
prices = fetch_prices("AAPL", start="2020-01-01")
report = data_quality_report(prices)
returns = to_returns(prices)
validate_returns(returns)  # Raises on invalid data
```

---

### Experiment Lab -- `wraquant.experiment`

A systematic strategy research platform for running, tracking, comparing,
and analyzing strategies across parameter grids and cross-validation folds.

**High-level Lab API:**
`Lab`, `Experiment`, `ExperimentResults`, `ExperimentRunner`,
`ExperimentStore`, `RunResult`.

**Cross-validation splits:**
`walk_forward_splits`, `rolling_splits`, `purged_kfold_splits`,
`combinatorial_purged_splits`.

**Low-level utilities:**
`ParameterGrid`, `grid_search`, `random_search`,
`walk_forward_optimize`, `parameter_sensitivity`,
`parameter_heatmap`, `robustness_check`, `stability_score`.

```python
from wraquant.experiment import Lab, Experiment

lab = Lab("momentum_research")
exp = lab.create("sma_crossover", params={"fast": [5,10,20], "slow": [50,100,200]})
results = exp.run(strategy_fn, prices)
results.summary()      # Ranked parameter combinations
results.best_params()  # Highest Sharpe configuration
```

---

### Compose Workflows -- `wraquant.compose`

A composable workflow system that chains wraquant module steps with
automatic data wiring. Steps read from and write to a shared context,
so users never write manual glue code.

**Classes:** `Workflow`, `WorkflowResult`.

**Pre-built steps:**
`steps.returns()`, `steps.regime_detect()`, `steps.garch_vol()`,
`steps.ta_features()`, `steps.ml_features()`, `steps.risk_metrics()`,
`steps.var_analysis()`, `steps.garch_var()`, `steps.stationarity_test()`,
`steps.forecast()`, `steps.tearsheet()`, `steps.optimize()`,
`steps.backtest_signals()`, `steps.stress_test()`,
`steps.beta_analysis()`, `steps.custom()`.

**Pre-built workflows:**
`quick_analysis_workflow` -- returns -> risk -> stationarity -> regimes -> vol -> tearsheet.
`risk_workflow` -- returns -> risk -> VaR -> GARCH VaR -> stress.
`ml_workflow` -- returns -> features -> backtest signals -> risk -> tearsheet.
`portfolio_workflow` -- returns -> optimize -> risk -> regimes -> tearsheet.

---

### Interactive Dashboard -- `wraquant.dashboard`

An optional Streamlit dashboard for interactive analysis. Six pages:

- **Experiment Browser** -- browse and compare experiment results from the Lab API
- **Strategy Analysis** -- upload returns CSV for comprehensive analysis (metrics, risk, regimes, distribution)
- **Risk Monitor** -- VaR/CVaR, rolling volatility, GARCH VaR, stress testing
- **Regime Viewer** -- interactive regime detection (HMM/GMM/changepoint) with overlay plots
- **Portfolio Optimizer** -- multi-asset optimization (MVO, risk parity) with risk decomposition
- **TA Screener** -- apply 265 technical indicators to OHLCV data with interactive charts

```bash
pip install wraquant[dashboard]
```

```python
from wraquant.dashboard import launch
launch()
# Or: python -m wraquant.dashboard
```

---

### IO & Storage -- `wraquant.io`

**12 exported functions** for file I/O, streaming, and export.

**File I/O:** `read_csv` / `write_csv`, `read_parquet` / `write_parquet`,
`read_hdf` / `write_hdf`, `read_excel` / `write_excel`.

**Streaming:** `WebSocketClient`, `TickBuffer`.

**Export:** `to_tearsheet`, `to_json`, `to_dict`, `format_table`.

---

### Parallel Computing -- `wraquant.scale`

**10 exported functions** for distributing quant workloads across
joblib, Dask, and Ray backends.

**Low-level:** `dask_map`, `ray_map`.

**Quant workflows:**
`parallel_backtest` -- sweep parameter grids in parallel.
`parallel_optimize` -- run portfolio optimization across constraint sets.
`parallel_walk_forward` -- parallelized walk-forward validation.
`parallel_regime_detection` -- detect regimes across multiple assets.
`parallel_monte_carlo` -- split MC simulations across workers.
`parallel_feature_compute` -- compute features per asset in parallel.
`distributed_backtest` -- enhanced backtest with auto-backend selection.
`chunk_apply` -- apply functions to DataFrame chunks in parallel.

```python
from wraquant.scale import parallel_backtest, parallel_monte_carlo

# Sweep 100 parameter combinations across all CPUs
results = parallel_backtest(
    strategy_fn, parameter_grid, prices, backend="joblib"
)

# 100K Monte Carlo simulations split across 8 workers
paths = parallel_monte_carlo(gbm_simulator, n_simulations=100_000, n_workers=8)
```

---

### Workflow Orchestration -- `wraquant.flow`

**10 exported functions** for pipeline construction, DAG execution,
and integration with production orchestrators.

**Built-in (no dependencies):**
`pipeline` / `Pipeline` -- sequential function chaining.
`dag` / `DAG` -- directed acyclic graph with topological execution.
`parallel_pipeline` -- run independent pipelines in threads.
`retry` -- exponential backoff decorator.
`cache_result` -- disk caching with TTL.
`log_step` -- observability decorator.

**External orchestrators:**
`prefect_backtest_flow` -- Prefect-based workflow with retries.
`dagster_pipeline` -- Dagster job definition.
`schedule_data_refresh` -- APScheduler recurring tasks.

```python
from wraquant.flow import pipeline, dag, retry, cache_result

# Sequential pipeline
pipe = pipeline(
    lambda prices: prices.pct_change().dropna(),
    lambda returns: {"sharpe": returns.mean() / returns.std() * 252**0.5},
)
result = pipe.run(prices)

# DAG with dependencies
d = dag({
    "fetch":   (fetch_data, []),
    "clean":   (clean_data, ["fetch"]),
    "analyze": (run_analysis, ["clean"]),
    "report":  (generate_report, ["analyze"]),
})
results = d.run()
```

---

### Core Types -- `wraquant.core` and `wraquant.frame`

**Financial type system** that carries metadata through the entire pipeline:

```python
import wraquant as wq

# Typed financial series with frequency and currency metadata
prices = wq.PriceSeries(data, freq=wq.Frequency.DAILY, currency=wq.Currency.USD)
returns = wq.ReturnSeries(data, return_type=wq.ReturnType.SIMPLE)
ohlcv = wq.OHLCVFrame(df)  # Validates OHLCV structure

# Type coercion: every function accepts array, list, Series, or DataFrame
from wraquant.core import coerce_returns, coerce_series
```

**Enums:** `Frequency`, `AssetClass`, `Currency`, `ReturnType`,
`OptionType`, `OptionStyle`, `OrderSide`, `RegimeState`,
`RiskMeasure`, `VolModel`.

**Result dataclasses:**
`GARCHResult`, `BacktestResult`, `ForecastResult`, `RegimeResult`,
`OptimizationResult` -- structured outputs with attribute access
and method chaining.

**Configuration:**
`WQConfig`, `get_config()`, `reset_config()` -- global settings
for backend selection (pandas/polars), logging, and defaults.

**Exceptions:**
`WQError`, `DataFetchError`, `ValidationError`, `ConfigError`,
`OptimizationError`, `BacktestError`, `PricingError`,
`MissingDependencyError`.

**Decorators:**
`@requires_extra('group-name')` -- graceful missing dependency handling.
`@validate_input` -- automatic type coercion.
`@cache_result` -- memoization.

---

## Architecture

wraquant is organized as a six-layer DAG where higher layers compose
lower layers. Data flows upward through well-defined integration points.

```
+-----------------------------------------------------------+
|                   APPLICATION LAYER                        |
|  wq.analyze    wq.Workflow    wq.recipes                  |
+----------------------------+------------------------------+
                             |
+----------------------------v------------------------------+
|                  ORCHESTRATION LAYER                       |
|  compose.py     experiment/     flow/     scale/          |
+----------------------------+------------------------------+
                             |
+----------------------------v------------------------------+
|                   ANALYSIS LAYER                           |
|  backtest/      viz/        dashboard/                    |
+----------------------------+------------------------------+
                             |
+----------------------------v------------------------------+
|                   MODELING LAYER                           |
|  risk/   vol/   regimes/   opt/   ml/                     |
+----------------------------+------------------------------+
                             |
+----------------------------v------------------------------+
|                 QUANTITATIVE LAYER                         |
|  stats/   ts/   price/   econometrics/                    |
+----------------------------+------------------------------+
                             |
+----------------------------v------------------------------+
|                   DOMAIN LAYER                             |
|  ta/   forex/   microstructure/   execution/              |
+----------------------------+------------------------------+
                             |
+----------------------------v------------------------------+
|                  FOUNDATION LAYER                          |
|  core/   frame/   data/   io/   math/   bayes/           |
+-----------------------------------------------------------+
```

**Key integration points:**

1. `data` --> `frame` -- fetched data becomes typed PriceSeries/OHLCVFrame
2. `frame` --> `ta` -- PriceSeries feeds 265 technical indicators
3. `ta` --> `ml` -- indicators become ML features
4. `ml` --> `backtest` -- predictions become trading signals
5. `backtest` --> `risk` -- strategy returns feed risk metrics (single source of truth)
6. `vol` --> `risk` -- GARCH conditional vol feeds time-varying VaR
7. `regimes` --> `opt` -- regime probabilities adjust portfolio weights
8. `regimes` --> `backtest` -- regime filters gate signal generation
9. `risk` --> `viz` -- metrics and results feed interactive dashboards
10. `all` --> `experiment` -- any result can be tracked and compared

---

## Optional Dependencies

All optional dependencies are organized into installable groups.
Core functionality (numpy, scipy, pandas, statsmodels, pydantic, structlog)
is always available.

| Group          | What it enables           | Key packages                                                                 |
| -------------- | ------------------------- | ---------------------------------------------------------------------------- |
| `market-data`  | Data fetching             | yfinance, fredapi, nasdaq-data-link, exchange-calendars                      |
| `ml`           | Machine learning          | scikit-learn                                                                 |
| `timeseries`   | Advanced time series      | pmdarima, arch, sktime, statsforecast, tsfresh, stumpy, pywavelets, ruptures |
| `cleaning`     | Data cleaning             | pyjanitor, rapidfuzz, dateparser, price-parser                               |
| `validation`   | Schema validation         | pandera, great-expectations, frictionless                                    |
| `etl`          | ETL and databases         | dlt, ibis, sqlalchemy, asyncpg, s3fs, gcsfs                                  |
| `warehouse`    | Data warehouse            | dbt-core, dbt-duckdb, dbt-postgres                                           |
| `ingestion`    | HTTP / websockets         | httpx, aiohttp, websockets, beautifulsoup4                                   |
| `workflow`     | Orchestration             | prefect, dagster, apscheduler                                                |
| `profiling`    | Performance profiling     | pyinstrument, scalene, memory-profiler                                       |
| `optimization` | Mathematical optimization | cvxpy, cvxopt, pulp, pyomo, pymoo, ortools                                   |
| `regimes`      | Regime detection          | hmmlearn, pomegranate, pykalman, filterpy, dynamax, ruptures                 |
| `backtesting`  | Backtest integrations     | vectorbt, quantstats, pyfolio, empyrical, ffn                                |
| `risk`         | Risk integrations         | pyportfolioopt, riskfolio-lib, skfolio, copulas, pyextremes                  |
| `pricing`      | Derivatives pricing       | QuantLib, rateslib, financepy, py-vollib                                     |
| `stochastic`   | SDE simulation            | sdepy, sdeint, torchsde                                                      |
| `pde`          | PDE solvers               | devito, py-pde, FiPy, findiff                                                |
| `causal`       | Causal inference          | dowhy, econml, DoubleML                                                      |
| `bayes`        | Bayesian inference        | pymc, arviz, numpyro, bambi, emcee, blackjax                                 |
| `viz`          | Visualization             | matplotlib, plotly, seaborn, bokeh, altair, holoviews, datashader            |
| `scale`        | Distributed computing     | dask, ray                                                                    |
| `dashboard`    | Interactive dashboard     | streamlit, plotly                                                            |
| `accelerate`   | Performance               | polars, pyarrow, duckdb, numba, bottleneck                                   |
| `symbolic`     | Symbolic math             | sympy, symengine, mpmath, numdifftools                                       |
| `logging`      | Enhanced logging          | loguru, rich, tenacity                                                       |
| `quant-math`   | JAX ecosystem             | jax, jaxlib, equinox, diffrax, optax                                         |
| `lp-extra`     | LP solvers                | highspy, mip, swiglpk                                                        |
| `conic-extra`  | Conic solvers             | ecos                                                                         |
| `nlp-extra`    | NLP solvers               | casadi, nlopt                                                                |
| `dev`          | Development               | pytest, hypothesis, ruff, mypy, pyright, jupyterlab                          |

---

## MCP Server (AI Agent Integration)

**wraquant-mcp** exposes wraquant as an MCP server for Claude, LangChain, and
other AI agents. Instead of writing Python, an agent calls structured tools
to run analysis, with results persisted in a shared DuckDB workspace.

| Metric              | Value                                     |
| ------------------- | ----------------------------------------- |
| Hand-crafted tools  | 218 across 22 module servers              |
| Prompt templates    | 327 (226 per-tool guides + 101 workflows) |
| Tests               | 357 passing, 0 failures                   |
| Tool guide coverage | 100%                                      |
| Error handling      | 100%                                      |

```bash
pip install wraquant-mcp
wraquant-mcp                    # Start stdio server for Claude Desktop
wraquant-mcp --transport http   # Start HTTP server for LangChain
```

Claude Desktop config (`~/.claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "wraquant": {
      "command": "wraquant-mcp",
      "env": { "PYTHONUNBUFFERED": "1" }
    }
  }
}
```

**Example agent conversation:**

```text
User:  Analyze AAPL risk and detect the current market regime
Agent: [calls compute_returns, risk_metrics, detect_regimes, fit_garch]
       AAPL Sharpe: 0.72, Max DD: -32%, Current regime: bull (P=0.87)
       GJR-GARCH persistence: 0.971, leverage effect significant
       VaR(95%): -2.1%, CVaR: -3.4%
```

**Module servers:** risk (15), vol (11), data (17), microstructure (16),
viz (14), math (14), regimes (12), stats (11), ta (11), ts (10),
backtest (10), execution (10), ml (9), price (9), opt (8), causal (7),
bayes (7), econometrics (6), forex (6), experiment (5), fundamental (5),
news (5).

**Prompt categories:** system, analysis, risk, regime, portfolio, strategy,
ML, pricing, reporting, execution, econometrics, forex, data, math, bayes,
plus 226 per-tool usage guides.

Composes with OpenBB MCP (data), DuckDB MCP (SQL), Jupyter MCP (notebooks),
and Alpaca MCP (execution) through a shared DuckDB file.

See [mcp/README.md](mcp/README.md) for full documentation, tool reference,
prompt catalog, and worked examples.

---

## Development

```bash
pdm install -G dev             # Install dev dependencies
pdm run test                   # Run tests
pdm run test-cov               # Tests with coverage
pdm run lint                   # Lint with Trunk
pdm run fmt                    # Format with Trunk
pdm run changelog              # Generate changelog with git-cliff
pdm run docs                   # Build Sphinx documentation
```

Conventions:

- **Commits**: Conventional commits (`feat(module):`, `fix(module):`, `chore:`)
- **Imports**: Lazy imports for all optional deps via `@requires_extra`
- **Types**: Type hints everywhere, `from __future__ import annotations` in every file
- **Testing**: pytest + hypothesis, each module has its own test directory
- **Linting**: Trunk (ruff, black, isort, bandit)
- **Docstrings**: Google/Napoleon style with mathematical formulations and references

---

## Documentation

Full API documentation: [wraquant.readthedocs.io](https://wraquant.readthedocs.io)

---

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like
to change. See the development section above for setup instructions.

---

## License

MIT
