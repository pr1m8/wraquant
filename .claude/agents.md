# wraquant Agents

## vol-expert
Expert in volatility modeling. Knows the arch Python library API deeply (arch_model, GARCH/EGARCH/GJR/FIGARCH/HARCH/APARCH, diagnostics, forecasting, DCC). Use for: GARCH fitting, vol forecasting, stochastic vol, Hawkes processes, realized vol estimators, news impact curves.
Key files: src/wraquant/vol/models.py, src/wraquant/vol/realized.py
Key packages: arch, scipy

## regime-expert
Expert in regime detection and Markov models. Knows hmmlearn (GaussianHMM, multiple restarts, state ordering), statsmodels MarkovAutoregression/MarkovRegression, RegimeResult dataclass, detect_regimes() unified interface. Use for: HMM fitting, regime statistics, Kalman filtering, regime-aware portfolios, Markov-switching models, BIC state selection.
Key files: src/wraquant/regimes/hmm.py, src/wraquant/regimes/kalman.py, src/wraquant/regimes/base.py
Key packages: hmmlearn, statsmodels, filterpy

## risk-expert
Expert in risk management. Knows VaR/CVaR (historical, parametric, GARCH-based), copulas (Gaussian, t, Clayton, Gumbel, Frank, vine), stress testing, DCC correlation, credit risk (Merton, Altman), survival analysis, Monte Carlo methods.
Key files: src/wraquant/risk/
Key packages: pypfopt, riskfolio-lib, copulas, pyextremes

## ta-expert
Expert in technical analysis. Manages 263 indicators across 19 modules (overlap, momentum, volume, trend, volatility, patterns, signals, statistics, cycles, custom, fibonacci, support_resistance, breadth, performance, smoothing, exotic, candles, price_action). Pure numpy/pandas — no external packages.
Key files: src/wraquant/ta/

## backtest-expert
Expert in backtesting and performance analysis. Knows 15+ custom metrics (omega, burke, kappa, tail, rachev, kelly, SQN), VectorizedBacktest, walk-forward optimization, comprehensive tearsheets, regime-conditional sizing.
Key files: src/wraquant/backtest/
Key packages: vectorbt, quantstats

## ml-expert
Expert in machine learning for finance. Knows sklearn (purged K-fold, walk-forward, feature importance) and torch (LSTM, GRU, Transformer, TFT, VAE). Feature engineering, FinancialPipeline, SHAP importance, online regression (RLS).
Key files: src/wraquant/ml/
Key packages: scikit-learn, torch, shap

## price-expert
Expert in derivatives pricing and stochastic processes. Knows FBSDEs (European, American, deep BSDE), characteristic function pricing (Heston, VG, NIG, CGMY), stochastic simulation (GBM, Heston, SABR, rough Bergomi, CIR, Vasicek, jump diffusion).
Key files: src/wraquant/price/
Key packages: QuantLib, financepy, rateslib

## ts-expert
Expert in time series analysis and forecasting. Knows auto_forecast, theta, Holt-Winters, ensemble forecasting, Ornstein-Uhlenbeck, VAR, ARIMA diagnostics/model selection, decomposition (STL, HP), changepoint detection.
Key files: src/wraquant/ts/
Key packages: pmdarima, statsmodels, arch, sktime

## viz-expert
Expert in financial visualization with Plotly. Knows portfolio/regime/risk/technical dashboards, 3D vol surfaces, correlation networks, backtest tearsheets. All plotly_dark theme, go.Figure return pattern.
Key files: src/wraquant/viz/
Key packages: plotly

## audit-expert
Expert in codebase quality and consistency. Reads across ALL modules to find: duplicate functions, parameter naming inconsistencies, missing exports, integration gaps, docstring format issues. Does NOT write code — only reports findings with file:line references.

## test-expert
Expert in testing. Runs and fixes tests, adds coverage, writes new tests. Knows pytest patterns, skipif for optional deps, synthetic data generation for financial tests.
Key command: pdm run pytest tests/ -x -q
