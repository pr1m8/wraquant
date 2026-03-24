# Coercion Adoption Status

## Done (coerce_array/coerce_series at function entry):
- core/_coerce.py — coercion utilities created
- ta/ — all 18 files via _validators.py (265 indicators)
- risk/metrics.py — 10 functions
- stats/descriptive.py — 8 functions
- stats/regression.py — 3 functions (ols, wls, newey_west)
- stats/correlation.py — 6 functions
- vol/models.py — _to_returns_array helper
- vol/realized.py — 8 functions
- regimes/hmm.py — 6 functions

## TODO (need coercion adoption):

### High priority (most-used modules):
- risk/var.py, risk/stress.py, risk/copulas.py, risk/dcc.py
- risk/beta.py, risk/factor.py, risk/portfolio_analytics.py, risk/tail.py
- risk/historical.py, risk/credit.py, risk/survival.py, risk/monte_carlo.py
- stats/tests.py, stats/distributions.py, stats/cointegration.py
- stats/factor_analysis.py, stats/robust.py, stats/dependence.py
- ts/ — all files
- backtest/ — all files
- ml/ — all files

### Medium priority:
- price/ — all files
- econometrics/ — all files
- regimes/ — remaining files (kalman, changepoint, labels, scoring, base)

### Lower priority:
- microstructure/ — all files
- execution/ — all files
- causal/ — all files
- forex/ — all files
- bayes/ — all files
- math/ — all files
