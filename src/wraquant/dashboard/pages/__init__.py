"""Dashboard page modules.

Each page exposes a ``render()`` function that draws its Streamlit UI.
Pages are imported lazily by ``app.py`` so that only the active page
incurs import costs.

Available pages:
    - overview: Company profile, key metrics, sentiment snapshot
    - fundamental_analysis: Income, balance sheet, cash flow, health score
    - valuation: DCF, Graham Number, relative valuation, margin of safety
    - technical_analysis: Candlestick charts, indicators, signals
    - risk_regimes: Risk metrics, VaR/CVaR, drawdowns, regime detection
    - news_events: News sentiment, earnings, insider trades
    - screener: Preset and custom stock screens
"""

from __future__ import annotations
