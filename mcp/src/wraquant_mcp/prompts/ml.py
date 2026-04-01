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

    @mcp.prompt()
    def anomaly_detection(dataset: str = "returns") -> list[dict]:
        """Anomaly detection: isolation forest, z-score, and regime-based anomalies."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Anomaly detection on {dataset}:

1. **Data**: Load {dataset} from workspace. compute_returns if raw prices.
   build_features with types=["returns", "volatility", "ta"] to create a rich feature matrix.
2. **Z-score method (univariate)**: For each feature, compute rolling 60-day z-score.
   Flag observations where |z| > 3 as anomalies. Which dates have multiple features
   simultaneously in anomaly territory? Cluster anomaly dates.
3. **Isolation Forest (multivariate)**: Train Isolation Forest on the feature matrix.
   Set contamination=0.02 (expect ~2% anomalies). Extract anomaly scores.
   Rank observations by anomaly score — most anomalous dates?
4. **Regime-based anomalies**: detect_regimes with n_regimes=2. Within each regime,
   compute regime-conditional mean and std for returns. Flag observations that are
   > 3σ from regime-conditional mean. These are "within-regime anomalies" — unusual
   even accounting for the current market state.
5. **Anomaly characterization**: For each detected anomaly, describe:
   - Date and magnitude of return
   - Which features triggered the anomaly?
   - Was it a vol spike, a momentum crash, a liquidity event?
   - Did the anomaly cluster with others (contagion)?
6. **Forward returns after anomalies**: Compute mean return 1, 5, 21 days after each
   anomaly. Do anomalies predict reversals (buying opportunity) or momentum (crisis continuation)?
7. **Trading signal**: Convert anomaly detection into a trading signal:
   - Negative anomaly (crash) with high regime-switch probability → reduce exposure
   - Negative anomaly in stable regime → potential buying opportunity (mean reversion)
   run_backtest with anomaly-based signals.
8. **Summary**: Number of anomalies detected by each method. Agreement between methods?
   Key anomaly dates and characterization. Predictive value of anomalies.
"""}}]

    @mcp.prompt()
    def regime_ml(dataset: str = "returns") -> list[dict]:
        """Regime-enhanced ML: use regime labels and probabilities as features."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Regime-enhanced ML model on {dataset}:

1. **Data**: Load {dataset} from workspace. compute_returns if raw prices.
2. **Regime features**: detect_regimes with method="hmm", n_regimes=2.
   Extract as features:
   - Current regime label (0 or 1) — categorical feature
   - Regime probability (continuous 0-1) — confidence in current regime
   - Regime duration (days in current regime) — longer duration = more stable
   - Transition probability (from transition matrix) — probability of switching
3. **Standard features**: build_features with types=["returns", "volatility", "ta"].
   Combine with regime features into a single feature matrix.
4. **Model WITHOUT regime features**: train_model with gradient_boost using only standard
   features. Walk-forward with 5 splits. Record out-of-sample Sharpe and hit rate.
5. **Model WITH regime features**: train_model with gradient_boost using standard +
   regime features. Same walk-forward splits. Record metrics.
6. **Feature importance comparison**: feature_importance for both models.
   Do regime features rank highly? Which regime feature is most important?
   Does adding regime features change which standard features matter?
7. **Regime-conditional models**: Instead of one model with regime features, train
   separate models for each regime. Model A for bull regime, Model B for bear regime.
   At prediction time, use the model matching the current regime.
   Compare to the single model with regime features — which approach wins?
8. **Regime transition signal**: Build a model to predict regime transitions (target = regime
   switches in next 5 days). This is an early warning system. What features predict transitions?
9. **Backtest all variants**: run_backtest for: (a) no-regime model, (b) regime-feature model,
   (c) regime-conditional models, (d) regime + transition signal.
   backtest_metrics for each. Which approach adds the most value?
10. **Summary**: Does regime information improve ML predictions? By how much (Sharpe delta)?
    Regime features vs regime-conditional models — which is better?
    Regime transition prediction accuracy. Recommended approach.
"""}}]

    @mcp.prompt()
    def ensemble_strategy(dataset: str = "prices") -> list[dict]:
        """Ensemble strategy: combine RF, GBM, and LSTM predictions."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Ensemble ML strategy on {dataset}:

1. **Data**: Load {dataset} from workspace. compute_returns.
   build_features with types=["returns", "volatility", "ta"] — generate feature matrix.
   Define target: next-day return sign (binary classification).
2. **Model 1 — Random Forest**: train_model with random_forest.
   Walk-forward with 5 splits, purged CV (gap between train and test).
   Record predictions and out-of-sample probabilities for each test fold.
3. **Model 2 — Gradient Boosting**: train_model with gradient_boost.
   Same walk-forward splits. Record predictions and probabilities.
   Compare feature_importance between RF and GBM — do they rely on different features?
4. **Model 3 — LSTM (if torch available)**: train_model with lstm.
   Input: 20-day lookback window of features. Same walk-forward splits.
   LSTMs capture sequential patterns that tree models miss.
   If torch unavailable, substitute with SVM as Model 3.
5. **Individual model metrics**: For each model, compute:
   - Hit rate, precision, recall, F1
   - Sharpe ratio of predictions-as-signals
   - Correlation between models' predictions (low correlation = good for ensemble)
6. **Ensemble methods**:
   - Simple average: probability = (p_RF + p_GBM + p_LSTM) / 3
   - Weighted average: weight by inverse out-of-sample error
   - Stacking: train a logistic regression on the 3 models' predictions (meta-learner)
   Compare all three ensemble methods.
7. **Confidence filtering**: Only trade when ensemble probability > 0.6 or < 0.4
   (high-confidence signals). Does filtering improve Sharpe at the cost of fewer trades?
8. **Regime analysis**: detect_regimes. Does the ensemble work in all regimes?
   Which individual model performs best in each regime? Regime-conditional weighting.
9. **Backtest**: run_backtest for best individual model, simple ensemble, stacked ensemble,
   and confidence-filtered ensemble. backtest_metrics for all.
10. **Summary**: Individual model comparison. Ensemble improvement (Sharpe delta).
    Best ensemble method. Confidence filtering value. Regime robustness.
"""}}]

    @mcp.prompt()
    def feature_selection(dataset: str = "features", target: str = "next_day_return") -> list[dict]:
        """Feature selection: RFE, SHAP, and correlation filtering."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Feature selection for {dataset} predicting {target}:

1. **Data**: Load {dataset} from workspace. Should be a feature matrix with many columns.
   Define target variable ({target}). Check for NaN/inf values and handle them.
2. **Correlation filtering (fast pre-screen)**: Compute pairwise correlation matrix among
   all features. For pairs with |corr| > 0.90, drop the feature with lower correlation
   to the target. This removes redundancy without modeling.
   Report: how many features removed? Which pairs were highly correlated?
3. **Variance threshold**: Remove features with near-zero variance (< 0.01 × median variance).
   Constant or near-constant features add nothing. Report removed features.
4. **Univariate importance**: Compute mutual information or F-statistic between each
   feature and target. Rank features. Top 20 features by univariate importance?
5. **RFE (Recursive Feature Elimination)**: train_model with gradient_boost.
   Use RFE to iteratively remove least important features. Track cross-validated
   Sharpe at each step. Find elbow: minimum features that achieve ~95% of best Sharpe.
6. **SHAP values**: Train a final gradient_boost model. Compute SHAP values for each feature.
   - Global importance: mean(|SHAP|) per feature — overall importance ranking
   - Interaction effects: top SHAP interaction pairs — which features work together?
   - Direction: SHAP dependence plots — how does each feature drive predictions?
7. **Stability check**: Run feature_importance across 5 walk-forward splits.
   Do the same features rank highly in each split? If feature importance is unstable,
   the model is fitting noise. Keep only features that rank in top 20 across ALL splits.
8. **Final feature set**: Intersect results from all methods:
   - Passes correlation filter
   - Top 50% by univariate importance
   - Survives RFE to the elbow
   - Top 20 by SHAP
   - Stable across walk-forward splits
   How many features survive? This is the robust feature set.
9. **Validation**: Train model with full features vs selected features. Compare
   out-of-sample Sharpe. Selected features should match or beat full set
   (fewer features = less overfitting).
10. **Summary**: Feature count reduction (original → selected). Key features and their
    interpretation. Agreement between selection methods. Out-of-sample improvement.
"""}}]
