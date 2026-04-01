"""Trading strategy prompt templates."""
from __future__ import annotations
from typing import Any


def register_strategy_prompts(mcp: Any) -> None:

    @mcp.prompt()
    def pairs_trading(ticker_a: str = "GLD", ticker_b: str = "GDX") -> list[dict]:
        """Pairs trading: cointegration, spread, signals, backtest."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Pairs trading analysis for {ticker_a} vs {ticker_b}:

1. Load price data for both.
2. cointegration_test — are they cointegrated? What's the p-value?
3. Compute hedge ratio and spread.
4. stationarity_test on the spread — must be stationary.
5. Compute half-life of mean reversion.
6. Generate z-score signals: enter at |z| > 2, exit at |z| < 0.5.
7. run_backtest with the signals.
8. backtest_metrics — Sharpe? Max drawdown? Win rate?
9. detect_regimes on the spread — does mean reversion break in certain regimes?
10. Summary: viable pair? Expected return/risk?
"""}}]

    @mcp.prompt()
    def momentum_strategy(dataset: str = "prices") -> list[dict]:
        """Momentum strategy: signals, regime filter, backtest."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Momentum strategy on {dataset}:

1. compute_indicator: RSI(14), MACD(12,26,9), ROC(20).
2. Signal: go long when RSI < 70 AND MACD histogram > 0 AND ROC > 0.
3. detect_regimes — only trade in bull regime (regime 0).
4. Position sizing: volatility_target at 15% annualized.
5. run_backtest with regime-filtered signals.
6. Compare to buy-and-hold.
7. walk_forward — does it work out-of-sample?
8. Summary: Sharpe improvement? Drawdown reduction? Regime filtering help?
"""}}]

    @mcp.prompt()
    def mean_reversion(dataset: str = "prices") -> list[dict]:
        """Mean reversion strategy: stationarity, OU fit, signals, backtest."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Mean reversion strategy on {dataset}:

1. compute_returns.
2. stationarity_test — is the series mean-reverting?
3. Fit Ornstein-Uhlenbeck: estimate theta (speed), mu (mean), sigma.
4. Half-life of reversion — how fast does it revert?
5. compute_indicator: Bollinger Bands(20, 2) for entry/exit.
6. Signal: buy at lower band, sell at upper band.
7. run_backtest with the signals.
8. detect_regimes — does mean reversion work in all regimes?
9. Summary: is mean reversion present? Profitable? Regime-dependent?
"""}}]

    @mcp.prompt()
    def trend_following(dataset: str = "prices") -> list[dict]:
        """Trend following: MA crossover, ADX filter, PSAR stops."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Trend following strategy on {dataset}:

1. compute_indicator: SMA(50), SMA(200) — golden/death cross.
2. compute_indicator: ADX(14) — trend strength filter (only trade when ADX > 25).
3. compute_indicator: PSAR — trailing stop levels.
4. Signal: long when SMA50 > SMA200 AND ADX > 25, stop at PSAR.
5. Position sizing: risk_parity or volatility_target.
6. run_backtest.
7. Compare to buy-and-hold.
8. Regime analysis — does trend following work better in trending regimes?
9. Summary: captures trends? Avoids whipsaws? Drawdown profile?
"""}}]

    @mcp.prompt()
    def statistical_arbitrage(dataset: str = "universe_returns") -> list[dict]:
        """Stat arb: PCA factors, residual alpha, signals, capacity."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Statistical arbitrage on {dataset}:

1. factor_analysis — extract PCA factors from the universe.
2. Compute residuals (alpha) for each asset.
3. stationarity_test on residuals — must be stationary for stat arb.
4. Z-score the residuals — trade when |z| > 2.
5. Build long-short portfolio from extreme z-scores.
6. run_backtest — does the residual alpha persist?
7. Estimate capacity: how much capital before impact kills the alpha?
8. walk_forward — is it robust out-of-sample?
9. Summary: alpha significant? Capacity adequate? Transaction costs?
"""}}]
