"""Machine learning prompt templates."""
from __future__ import annotations
from typing import Any


def register_ml_prompts(mcp: Any) -> None:

    @mcp.prompt()
    def ml_alpha_research(dataset: str = "prices") -> list[dict]:
        """Full ML alpha research pipeline: features → model → backtest."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
ML alpha research on {dataset}:

1. compute_returns.
2. build_features with types=["returns", "volatility", "ta"] — generate feature matrix.
3. Define target: binary (up/down next day) or continuous (next-day return).
4. train_model with gradient_boost — walk-forward with purged CV.
5. feature_importance — which features drive predictions? Remove noise.
6. Retrain with top features only.
7. run_backtest using model predictions as signals.
8. backtest_metrics — Sharpe? Hit rate? Profit factor?
9. detect_regimes — does the model work in all regimes?
10. Summary: is there alpha? Robust out-of-sample? Regime-dependent?
"""}}]

    @mcp.prompt()
    def feature_engineering(dataset: str = "prices") -> list[dict]:
        """Comprehensive feature engineering for ML."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Build feature set from {dataset}:

1. Return features: 1d, 5d, 21d returns, log returns.
2. Volatility features: rolling vol (5, 21, 63 day), GARCH conditional vol.
3. TA features: RSI, MACD histogram, BB %b, ADX, OBV change.
4. Regime features: current regime probability, regime duration.
5. Cross-asset features: rolling correlation, rolling beta (if benchmark available).
6. Interaction features: vol × momentum, regime × RSI.
7. Check for multicollinearity — remove redundant features (VIF > 10).
8. Store final feature matrix for modeling.
"""}}]

    @mcp.prompt()
    def model_comparison(dataset: str = "features") -> list[dict]:
        """Compare multiple ML models with walk-forward validation."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Compare models on {dataset}:

1. Train Random Forest — walk-forward 5 splits.
2. Train Gradient Boosting — same splits.
3. Train SVM — same splits.
4. If torch available: train LSTM — same splits.
5. Compare: Sharpe, hit rate, max drawdown for each.
6. Ensemble: average predictions — does combining improve?
7. feature_importance for best model — what does it rely on?
8. Summary: which model wins? Is ensemble better? Stable across folds?
"""}}]

    @mcp.prompt()
    def hyperparameter_sweep(dataset: str = "features", model: str = "gradient_boost") -> list[dict]:
        """Grid search over model hyperparameters."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Hyperparameter sweep for {model} on {dataset}:

1. Define parameter grid (e.g., n_estimators=[50,100,200], max_depth=[3,5,7]).
2. Walk-forward validation for each combination.
3. Rank by Sharpe ratio (not just accuracy — financial metric matters).
4. Check stability — do nearby parameters give similar results?
5. Overfit check — is in-sample Sharpe >> out-of-sample?
6. Best config with confidence interval on performance.
7. Summary: optimal parameters? Stable or sensitive? Overfitting risk?
"""}}]
