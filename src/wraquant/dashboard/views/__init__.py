"""Dashboard page modules.

Each page exposes a ``render()`` function that draws its Streamlit UI.
Pages are imported lazily by ``app.py`` so that only the active page
incurs import costs.

Available pages:
    - overview: Company profile, key metrics, sentiment snapshot
    - fundamental_analysis: Income, balance sheet, cash flow, health score
    - valuation: DCF, Graham Number, relative valuation, margin of safety
    - technical_analysis: Candlestick charts, indicators, signals
    - returns_stats: Deep statistical analysis -- distribution, rolling stats,
      beta & factor, correlation, cointegration
    - time_series: STL decomposition, stationarity tests, seasonality,
      forecasting, anomaly detection, changepoint detection
    - microstructure: Amihud illiquidity, Kyle lambda, VPIN toxicity,
      variance ratio, Corwin-Schultz spread estimation
    - quant_lab: Correlation network, information theory, spectral analysis,
      Monte Carlo path simulation (GBM / Heston)
    - ml_lab: Feature engineering, model training (walk-forward),
      classification results, feature importance
    - portfolio_risk: Multi-asset risk decomposition, component/marginal VaR,
      drawdown analysis, efficient frontier, max-Sharpe optimization
    - risk_regimes: Risk metrics, VaR/CVaR, drawdowns, regime detection
    - news_events: News sentiment, earnings, insider trades
    - screener: Preset and custom stock screens
"""

from __future__ import annotations
