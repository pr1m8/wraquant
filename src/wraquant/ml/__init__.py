"""Machine learning utilities for quantitative finance.

This module implements the full ML pipeline for financial prediction and
analysis: feature engineering, preprocessing, model training, evaluation,
and online learning -- all designed to avoid the pitfalls that make naive
ML on financial data fail (lookahead bias, non-stationarity, overfitting
on noise).

Pipeline overview
-----------------
A typical financial ML workflow moves through five stages, each
supported by a sub-module here:

1. **Feature engineering** (``features``) -- transform raw market data
   into predictive signals.

   - ``return_features`` -- lagged returns, log returns, and cross-
     sectional return features.
   - ``volatility_features`` -- realized vol, GARCH residuals, vol-of-vol.
   - ``technical_features`` -- wraps ``wraquant.ta`` indicators into a
     feature DataFrame.
   - ``rolling_features`` -- rolling statistics (mean, std, skew, kurt,
     z-score) at multiple windows.
   - ``microstructure_features`` -- bid-ask spread, order imbalance,
     trade intensity.
   - ``label_fixed_horizon`` -- binary or ternary labels based on
     forward returns over a fixed window.
   - ``label_triple_barrier`` -- labels based on the triple-barrier
     method (de Prado): a trade is labeled by which barrier (profit
     target, stop-loss, or time expiry) is hit first. Preferred over
     fixed-horizon labels for realistic strategy evaluation.
   - ``interaction_features`` -- pairwise interaction terms between
     features (products and ratios).
   - ``cross_asset_features`` -- rolling correlation, beta, and relative
     strength between assets.
   - ``regime_features`` -- features derived from regime probabilities
     (current regime, duration, transition probability).

2. **Preprocessing** (``preprocessing``) -- prepare data for training
   without introducing bias.

   - ``purged_kfold`` -- time-series cross-validation with a purge gap
     to prevent information leakage between train and test folds.
   - ``combinatorial_purged_kfold`` -- generates all combinatorial
     train/test splits with purging (de Prado Chapter 12).
   - ``fractional_differentiation`` -- make a price series stationary
     while retaining as much memory as possible (unlike simple
     differencing which destroys long-range dependencies).
   - ``denoised_correlation`` -- apply Marcenko-Pastur random matrix
     theory to shrink noisy eigenvalues of the correlation matrix.
   - ``detoned_correlation`` -- remove the market mode (first
     eigenvector) to expose sector-level structure.

3. **Model training** (``models``, ``advanced``, ``deep``, ``online``,
   ``pipeline``) -- fit models designed for financial prediction.

   - ``walk_forward_train`` -- expanding or rolling window training
     with out-of-sample prediction at each step. The gold standard for
     financial model validation.
   - ``ensemble_predict`` -- blend predictions from multiple models.
   - ``feature_importance_mdi`` / ``feature_importance_mda`` -- Mean
     Decrease Impurity and Mean Decrease Accuracy feature importance
     (de Prado Chapter 8). MDA is preferred as it accounts for
     substitution effects.
   - ``sequential_feature_selection`` -- forward/backward feature
     selection with cross-validation.

   Pipeline utilities:
   - ``FinancialPipeline`` -- sklearn Pipeline wrapper enforcing
     chronological splitting with purged K-fold CV.
   - ``walk_forward_backtest`` -- full walk-forward ML backtest with
     PnL, Sharpe, hit rate, and equity curve.
   - ``feature_importance_shap`` -- SHAP-based feature importance.

   Advanced sklearn wrappers:
   - ``svm_classifier``, ``random_forest_importance``,
     ``gradient_boost_forecast``, ``gaussian_process_regression``,
     ``isolation_forest_anomaly``, ``pca_factor_model``.

   Deep learning (requires PyTorch):
   - ``lstm_forecast``, ``gru_forecast``, ``transformer_forecast`` --
     recurrent and attention-based time-series forecasting.
   - ``multivariate_lstm_forecast`` -- LSTM with multiple input features.
   - ``temporal_fusion_transformer`` -- interpretable forecasting with
     variable selection and attention.
   - ``autoencoder_features`` -- VAE-based latent feature extraction
     and anomaly detection.

   Online / streaming:
   - ``online_linear_regression``, ``exponential_weighted_regression`` --
     models that update incrementally with each new observation.

4. **Clustering** (``clustering``) -- discover structure in returns.

   - ``correlation_clustering`` -- hierarchical clustering of the
     correlation matrix to find asset groups.
   - ``regime_clustering`` -- cluster return features to identify
     market regimes.
   - ``optimal_clusters`` -- determine the optimal number of clusters
     via silhouette score and gap statistic.

5. **Evaluation** (``evaluation``) -- measure performance correctly.

   - ``classification_metrics`` -- accuracy, precision, recall, F1 for
     classification models.
   - ``financial_metrics`` -- Sharpe, Sortino, max drawdown, and hit
     rate of the model's predictions when used as a trading signal.
   - ``learning_curve`` -- diagnose bias/variance trade-off as training
     set size grows.
   - ``backtest_predictions`` -- convert model predictions into a P&L
     series and compute strategy-level metrics.

Common pitfalls
---------------
- **Lookahead bias**: always use ``purged_kfold`` or
  ``walk_forward_train``, never random shuffled CV.
- **Non-stationarity**: apply ``fractional_differentiation`` to price
  levels before using them as features.
- **Overfitting**: financial signal-to-noise ratio is very low; use
  ``feature_importance_mda`` to prune irrelevant features and monitor
  ``learning_curve`` for divergence between train and test error.
- **Label leakage**: ``label_triple_barrier`` barriers must be computed
  on future data only; the purge gap in cross-validation must be at
  least as wide as the label horizon.

References
----------
- de Prado (2018), "Advances in Financial Machine Learning"
- Dixon, Halperin & Bilokon (2020), "Machine Learning in Finance"
"""

from __future__ import annotations

from wraquant.ml.advanced import (
    gaussian_process_regression,
    gradient_boost_forecast,
    isolation_forest_anomaly,
    pca_factor_model,
    random_forest_importance,
    svm_classifier,
)
from wraquant.ml.clustering import (
    correlation_clustering,
    optimal_clusters,
    regime_clustering,
)
from wraquant.ml.deep import (
    autoencoder_features,
    gru_forecast,
    lstm_forecast,
    multivariate_lstm_forecast,
    temporal_fusion_transformer,
    transformer_forecast,
)
from wraquant.ml.evaluation import (
    backtest_predictions,
    classification_metrics,
    financial_metrics,
    learning_curve,
)
from wraquant.ml.features import (
    cross_asset_features,
    interaction_features,
    label_fixed_horizon,
    label_triple_barrier,
    microstructure_features,
    regime_features,
    return_features,
    rolling_features,
    technical_features,
    volatility_features,
)
from wraquant.ml.models import (
    ensemble_predict,
    feature_importance_mda,
    feature_importance_mdi,
    sequential_feature_selection,
    walk_forward_train,
)
from wraquant.ml.online import (
    exponential_weighted_regression,
    online_linear_regression,
)
from wraquant.ml.pipeline import (
    FinancialPipeline,
    feature_importance_shap,
    walk_forward_backtest,
)
from wraquant.ml.preprocessing import (
    combinatorial_purged_kfold,
    denoised_correlation,
    detoned_correlation,
    fractional_differentiation,
    purged_kfold,
)

__all__ = [
    # features
    "rolling_features",
    "return_features",
    "technical_features",
    "volatility_features",
    "microstructure_features",
    "label_fixed_horizon",
    "label_triple_barrier",
    "interaction_features",
    "cross_asset_features",
    "regime_features",
    # preprocessing
    "purged_kfold",
    "combinatorial_purged_kfold",
    "fractional_differentiation",
    "denoised_correlation",
    "detoned_correlation",
    # models
    "walk_forward_train",
    "ensemble_predict",
    "feature_importance_mdi",
    "feature_importance_mda",
    "sequential_feature_selection",
    # pipeline
    "FinancialPipeline",
    "walk_forward_backtest",
    "feature_importance_shap",
    # clustering
    "correlation_clustering",
    "regime_clustering",
    "optimal_clusters",
    # evaluation
    "classification_metrics",
    "financial_metrics",
    "learning_curve",
    "backtest_predictions",
    # deep learning
    "lstm_forecast",
    "transformer_forecast",
    "autoencoder_features",
    "gru_forecast",
    "multivariate_lstm_forecast",
    "temporal_fusion_transformer",
    # advanced sklearn
    "svm_classifier",
    "random_forest_importance",
    "gradient_boost_forecast",
    "gaussian_process_regression",
    "isolation_forest_anomaly",
    "pca_factor_model",
    # online / streaming
    "online_linear_regression",
    "exponential_weighted_regression",
]
