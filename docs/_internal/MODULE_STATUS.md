# Module Status

Last updated: 2026-03-18

| Module | Functions | Tests | Depth | Docs | Integration |
|--------|-----------|-------|-------|------|-------------|
| vol/ | 23 | 100 | Deep (GARCH family, Hawkes, SV) | Rich | -> risk/var, <- ta/vol |
| regimes/ | 20+ | 167 | Deep (HMM, MS-AR, Kalman, GMM) | Rich | -> opt/, -> backtest/ |
| risk/ | 50+ | 100+ | Deep (VaR, copulas, DCC, credit) | Rich | <- backtest/, <- vol/ |
| ta/ | 263 | 830 | Broad (19 sub-modules) | Good | -> ml/features |
| backtest/ | 40+ | 175 | Deep (metrics, engine, tearsheet) | Rich | <- risk/, <- regimes/ |
| ml/ | 30+ | 95 | Deep (torch, sklearn, online) | Rich | <- ta/, -> backtest/ |
| price/ | 30+ | 182 | Deep (FBSDE, char fns, stoch) | Rich | Standalone |
| ts/ | 25+ | 92 | Deep (forecast, stochastic, OU) | Rich | -> backtest/ |
| stats/ | 30+ | 100+ | Good (robust, distributions) | Good | -> risk/, -> ml/ |
| econometrics/ | 25+ | 80+ | Good (panel, VAR, events) | OK | -> vol/ (delegates GARCH) |
| microstructure/ | 15 | 40+ | Good (liquidity, toxicity) | OK | -> execution/ |
| execution/ | 10 | 30+ | Good (TWAP, VWAP, AC) | OK | <- microstructure/ |
| causal/ | 10 | 30+ | Good (DID, IPW, SC) | OK | Standalone |
| forex/ | 15 | 40+ | Good (pairs, carry, risk) | OK | Standalone |
| bayes/ | 10 | 20+ | Good (PyMC, emcee, blackjax) | OK | -> regimes/ |
| viz/ | 20+ | 95 | Deep (dashboards, 3D, network) | Good | <- all modules |
| math/ | 15 | 50+ | Good (Levy, network, stopping) | OK | -> price/ |
| flow/ | 3 | 19 | Stub (Pipeline, Prefect) | OK | Infrastructure |
| scale/ | 3 | 6 | Stub (Dask, Ray, parallel) | OK | Infrastructure |
| io/ | 10+ | 20+ | Good (SQL, cloud, streaming) | OK | Infrastructure |
| data/ | 15+ | 40+ | Good (providers, cleaning) | OK | -> all modules |
| experiment/ | 5 | 10+ | Stub (tracker) | Minimal | Planned redesign |
