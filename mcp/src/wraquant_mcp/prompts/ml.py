"""Machine learning prompt templates."""
from __future__ import annotations
from typing import Any


def register_ml_prompts(mcp: Any) -> None:

    @mcp.prompt()
    def ml_alpha_research(dataset: str = "prices") -> list[dict]:
        """Full ML alpha research pipeline: features → model → backtest."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
First load the wraquant_system_context prompt for full module context.

Full ML alpha research pipeline on {dataset}. This is the most comprehensive workflow,
touching ml/ (44 functions), ta/ (265 indicators), stats/ (79 functions), regimes/
(38 functions), backtest/ (38 functions), and risk/ (95 functions). The pipeline goes
from raw data to a fully validated, out-of-sample tested trading strategy.

The single most important principle: EVERYTHING must be out-of-sample. In-sample
results are meaningless for financial ML. The data is noisy, the signal-to-noise
ratio is extremely low (typically 0.01-0.05), and overfitting is the default outcome.
Every decision — feature selection, model selection, hyperparameters — must be validated
on data the model has never seen.

---

## Phase 1: Data Preparation & Target Definition

1. **Workspace check**: Run workspace_status. Verify {dataset} exists.
   For ML, we need LONG history: minimum 5 years (1250+ trading days).
   Walk-forward validation splits burn a lot of data for estimation windows.
   10+ years is ideal for financial ML.

2. **Compute returns**: compute_returns on {dataset}.
   Use log returns for ML (more symmetric, better behaved for many models).
   Verify no NaN, no returns > 30% (data error unless IPO/event day).

3. **Target variable definition**: The target is what we are trying to predict.
   Choose ONE of:
   - **Binary classification**: next-day return direction (up = 1, down = 0).
     Easier for models. Hit rate > 52% is tradeable after costs.
   - **5-day forward return sign**: Smooths noise. Easier to predict than 1-day.
   - **Continuous regression**: next-day return magnitude. Harder but more informative.
     Model can output position size proportional to predicted return.

   **Critical**: The target must be computed BEFORE features to avoid look-ahead bias.
   target_t = sign(return_{t+1}). Feature_t uses only data up to time t.
   There must be NO overlap between features and target — this is the #1 source of
   bugs in financial ML. A 1-day gap between feature computation and target is standard.

4. **Data splitting strategy** — DO NOT use random train/test split.
   Financial data is temporal. Use time-series aware splitting:
   - **Walk-forward**: Train on [0, T], test on [T+gap, T+gap+window], step forward.
   - **Purged cross-validation**: Leave a gap between train and test to prevent leakage
     from overlapping features (e.g., 21-day rolling vol computed on day T uses data
     from T-20, which could overlap with the test set).
   - Minimum gap = max lookback window in your features (e.g., if longest feature uses
     63-day window, gap should be >= 63 days).
   - Use 5 walk-forward splits minimum. 10 is better.

---

## Phase 2: Feature Engineering (ml/ + ta/ + stats/ + regimes/ modules)

5. **Return features**: build_features with types=["returns"].
   - 1-day, 5-day, 21-day, 63-day returns (momentum at multiple horizons)
   - 1-day, 5-day, 21-day log returns
   - Return acceleration: 5d return - 21d return (momentum change)
   - These capture short, medium, and long-term price momentum

6. **Volatility features**: build_features with types=["volatility"].
   - Rolling realized vol: 5, 10, 21, 63 day windows
   - GARCH conditional volatility: fit_garch, extract sigma_t series
   - Vol ratio: short-term vol / long-term vol (vol regime proxy)
   - Vol change: today's vol / yesterday's vol (vol momentum)
   - Intraday range: (high - low) / close if OHLC available (Parkinson proxy)

7. **Technical indicator features**: build_features with types=["ta"].
   Use compute_indicator for a curated set (NOT all 265 — that's noise):
   - **Momentum**: RSI(14), RSI(5), MACD histogram, ROC(5), ROC(21), Stochastic %K
   - **Trend**: ADX(14), Aroon oscillator, SMA(10)/SMA(50) ratio
   - **Volume**: OBV rate of change, CMF(20), volume ratio (today / 20d avg)
   - **Volatility**: Bollinger %b, ATR(14), Keltner channel position
   - **Pattern**: Distance from 52-week high, distance from 52-week low

   **Feature count target**: 30-50 features total. More features = more overfitting risk.
   Each feature must have a financial rationale. No "kitchen sink" approach.

8. **Regime features** (optional but often powerful):
   - detect_regimes on the return series -> current regime label and probability
   - Regime duration (days in current regime)
   - Transition probability from HMM transition matrix
   - These capture market state which affects how other features should be interpreted

9. **Cross-asset features** (if benchmark/related data available):
   - Rolling correlation with SPY (market beta proxy)
   - Rolling relative strength (asset return - benchmark return)
   - Market volatility (VIX level or SPY rolling vol)

---

## Phase 3: Feature Preprocessing & Selection

10. **Feature quality check**: For each feature:
    - Missing values: drop features with > 5% NaN. Fill remainder with forward-fill.
    - Stationarity: stationarity_test on each feature. Non-stationary features should
      be differenced or replaced with returns (e.g., use RSI change instead of RSI level
      if RSI is non-stationary, though RSI is bounded so this is rare).
    - Variance: drop features with near-zero variance (< 0.01 * median variance).
    - Extreme values: winsorize at 1st and 99th percentiles to limit outlier influence.

11. **Multicollinearity check**: Compute pairwise correlation among features.
    - Remove one feature from each pair with |corr| > 0.90 (keep the one with
      higher correlation to the target).
    - Compute VIF (Variance Inflation Factor). Drop features with VIF > 10.
    - Target: no feature pair with |corr| > 0.85 after filtering.

    **Why multicollinearity matters**: Tree models (RF, GBM) are somewhat robust to
    correlated features, but feature importance becomes unreliable. Linear models and
    neural networks suffer more directly. Removing redundancy also speeds training.

12. **Univariate screening**: For each feature, compute:
    - Mutual information with the target (non-linear association).
    - Spearman rank correlation with the target.
    - Information coefficient (IC): rank correlation between feature rank and forward return.
      IC > 0.02 is meaningful in finance. IC > 0.05 is excellent.
    - Drop features with IC < 0.01 (no predictive signal detectable).

    Report the top 20 features by IC. These are your strongest signal candidates.

---

## Phase 4: Model Training & Selection

13. **Model 1 — Gradient Boosting (primary)**: train_model with gradient_boost.
    GBM is the workhorse of financial ML: handles non-linearity, missing values,
    feature interactions. Does NOT need feature scaling.

    **Walk-forward setup**:
    - Estimation window: 756 days (3 years)
    - Test window: 63 days (1 quarter)
    - Gap: max(feature lookback windows) days
    - Number of splits: as many as data allows (typically 5-15)
    - Re-train from scratch each split (no warm start — distribution may shift)

    **Hyperparameters** (reasonable defaults):
    - n_estimators: 200-500 (more is NOT better — overfitting)
    - max_depth: 3-5 (shallow trees generalize better for noisy financial data)
    - learning_rate: 0.05-0.1
    - subsample: 0.7-0.8 (bagging regularization)
    - min_child_weight: 50-100 (prevents fitting to small groups)
    - early_stopping_rounds: 20 (stop when validation loss stops improving)

14. **Model 2 — Random Forest (comparison)**: train_model with random_forest.
    RF is more robust to overfitting than GBM but less powerful.
    Same walk-forward splits. Compare directly.

15. **Model 3 — LSTM (if torch available)**: train_model with lstm.
    LSTM captures sequential patterns that tree models miss.
    Input: 20-day lookback window of features (sequence input).
    Same walk-forward splits but LSTM needs more data per split.
    If torch unavailable or data < 2000 days, skip LSTM.

    **If training fails**: LSTM is sensitive to learning rate and architecture.
    Try: lr=0.001, hidden_dim=32, num_layers=1, dropout=0.3.
    If still failing, use SVM as Model 3 instead.

16. **Out-of-sample metrics for each model** (on test splits only):
    | Metric | GBM | RF | LSTM |
    |--------|-----|----|----|
    | Hit rate (accuracy) | | | |
    | Information Coefficient (IC) | | | |
    | Sharpe of predictions-as-signals | | | |
    | Precision (long signals) | | | |
    | Recall (long signals) | | | |
    | Max drawdown of signal strategy | | | |
    | Consistency (% of splits with Sharpe > 0) | | | |

    **Critical check**: If in-sample Sharpe > 3x out-of-sample Sharpe, the model is
    heavily overfit. Reduce complexity (fewer features, shallower trees, more regularization).

---

## Phase 5: Feature Importance & Model Interpretation

17. **Feature importance**: feature_importance on the best model.
    - For GBM: SHAP values (most reliable). Compute mean(|SHAP|) per feature.
    - For RF: permutation importance (more reliable than built-in importance).
    - Rank features by importance. Report top 10 with interpretation.

    **Key questions**:
    - Do the top features make financial sense? If "arbitrary_feature_42" is #1,
      the model may be fitting noise.
    - Is importance concentrated in 1-2 features? If so, the model is fragile.
      Ideal: importance spread across 5-10 features.
    - Do return features dominate? (momentum signal)
    - Do vol features dominate? (vol timing signal)
    - Do TA features dominate? (technical signal)

18. **Feature stability**: Run feature_importance across all walk-forward splits.
    Are the same features important in each split?
    - If the top 5 features are consistent across splits: robust signal.
    - If feature importance changes dramatically between splits: model is fitting noise.
    Compute "importance stability score": fraction of splits where each feature ranks top 10.
    Keep only features with stability score > 0.5 (top 10 in more than half the splits).

19. **Retrain with selected features**: Take the top 15-20 features that are:
    - High IC (> 0.01)
    - Stable across walk-forward splits
    - Not redundant (pairwise |corr| < 0.85)
    - Financially interpretable
    Retrain the best model with ONLY these features. Same walk-forward setup.
    Compare to the full-feature model. The selected-feature model should have
    EQUAL or BETTER out-of-sample Sharpe (fewer features = less overfitting).

---

## Phase 6: Signal Generation & Backtesting

20. **Signal construction**: Convert model predictions to trading signals.
    - For classification: probability > 0.55 -> long, < 0.45 -> short, else flat.
      The 0.55/0.45 thresholds filter for high-confidence predictions.
    - For regression: position size proportional to predicted return (capped at 2x).
    - Apply signal delay: predict at close of day T, trade at open of day T+1.
      This accounts for realistic execution (cannot trade at the same close you used for prediction).

21. **Walk-forward backtest**: run_backtest with the model predictions as signals.
    This must use ONLY out-of-sample predictions (no in-sample predictions in the backtest).

    **Position sizing options**:
    - Equal size: always 100% long or 100% short or flat
    - Vol-targeted: size inversely proportional to predicted vol (volatility_target at 15%)
    - Kelly: size proportional to edge / variance. Use half-Kelly for safety.

22. **Backtest metrics**: backtest_metrics on the walk-forward strategy.
    | Metric | Value | Target | Status |
    |--------|-------|--------|--------|
    | Ann. Return | X% | > 5% (after costs) | PASS/FAIL |
    | Ann. Volatility | X% | 10-20% | |
    | Sharpe Ratio | X.XX | > 0.5 (walk-forward) | PASS/FAIL |
    | Max Drawdown | -X% | < 25% | PASS/FAIL |
    | Hit Rate | X% | > 52% (for daily) | PASS/FAIL |
    | Profit Factor | X.XX | > 1.2 | PASS/FAIL |
    | Avg Win / Avg Loss | X.XX | > 0.8 | PASS/FAIL |
    | # Trades | X | > 100 | PASS/FAIL |
    | % Profitable Splits | X% | > 60% | PASS/FAIL |
    | Max Consecutive Losses | X | < 15 | PASS/FAIL |

    **Walk-forward Sharpe benchmarks**: For daily ML strategies on single stocks:
    - Sharpe > 1.0: Excellent (rare and suspicious — double-check for look-ahead bias)
    - Sharpe 0.5-1.0: Good. Potentially tradeable.
    - Sharpe 0.3-0.5: Marginal. May not survive transaction costs.
    - Sharpe < 0.3: Insufficient alpha. Model is not predicting well enough.

23. **Transaction cost sensitivity**: Re-run backtest with costs of 5, 10, 20, 30 bps.
    At what cost level does Sharpe drop below 0.3? This is the break-even cost.
    For US large-cap equities, realistic costs are 5-15 bps. For small-cap, 20-50 bps.

---

## Phase 7: Regime Robustness

24. **Regime analysis**: detect_regimes on the underlying returns with method="hmm", n_regimes=2.
    Compute strategy performance SEPARATELY in each regime:

    | Regime | Ann. Return | Sharpe | Hit Rate | Max DD | % of Time |
    |--------|-------------|--------|----------|--------|-----------|
    | Bull (0) | | | | | |
    | Bear (1) | | | | | |

    **Common patterns**:
    - Model works in bull, fails in bear: momentum signal that doesn't adapt to regime change.
    - Model works in bear, fails in bull: mean-reversion signal.
    - Model works in both: robust alpha (rare and valuable).
    - Model fails in both: insufficient signal.

25. **Regime-conditional trading**: Consider a regime filter:
    - Trade only in the favorable regime.
    - Or trade in both regimes but with different position sizes (smaller in adverse regime).
    - Or train separate models per regime (see regime_ml prompt for this approach).

    Compare: unfiltered strategy vs regime-filtered strategy.
    Does regime filtering improve Sharpe or reduce drawdown?

---

## Phase 8: Final Assessment & Deployment Readiness

26. **Alpha assessment checklist**:
    | Criterion | Result | Pass/Fail |
    |-----------|--------|-----------|
    | Walk-forward Sharpe > 0.5 | X.XX | |
    | Profitable after 10bps costs | X% ann. | |
    | > 100 out-of-sample trades | X trades | |
    | > 60% of splits profitable | X% | |
    | Top features are interpretable | Yes/No | |
    | Feature importance is stable | Yes/No | |
    | Works in > 1 regime | Yes/No | |
    | In-sample/OOS Sharpe ratio < 2.5 | X.XX | |
    | Max drawdown < 25% | -X% | |

    **If 7+ criteria pass**: Alpha is likely real. Consider paper trading for 3-6 months
    before live deployment.
    **If 4-6 pass**: Marginal. Needs improvement. Try different features, models, or targets.
    **If < 4 pass**: No tradeable alpha found. The signal-to-noise ratio is too low for
    this asset / feature set / model combination. Try a different approach.

27. **Summary**:
    - Best model and its key features (top 5)
    - Walk-forward Sharpe and max drawdown
    - Transaction cost break-even
    - Regime robustness assessment
    - Is the alpha real? Confidence level (high / moderate / low)
    - Recommended next steps (deploy / refine / abandon)
    - One-sentence conclusion

**Related prompts**: Use feature_engineering for deeper feature construction,
model_comparison for systematic model comparison, hyperparameter_sweep for optimization,
ensemble_strategy for combining multiple models, regime_ml for regime-enhanced ML,
feature_selection for rigorous feature filtering.
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
