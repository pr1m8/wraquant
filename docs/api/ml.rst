Machine Learning (``wraquant.ml``)
===================================

The ML module implements the full machine learning pipeline for financial
prediction: feature engineering, preprocessing, model training, evaluation,
and online learning -- all designed to avoid the pitfalls that make naive
ML on financial data fail (lookahead bias, non-stationarity, overfitting).

**Pipeline stages:**

1. **Feature engineering** -- return features, TA features, volatility features, triple-barrier labels
2. **Preprocessing** -- purged K-fold CV, fractional differentiation, denoised correlation
3. **Model training** -- walk-forward, ensembles, feature importance (MDA/MDI), sequential selection
4. **Deep learning** -- LSTM, GRU, Transformer, Temporal Fusion Transformer, autoencoders
5. **Evaluation** -- financial metrics (Sharpe, profit factor), learning curves, backtest predictions
6. **Online learning** -- incremental regression for streaming data

Quick Example
-------------

.. code-block:: python

   from wraquant.ml import (
       technical_features, label_triple_barrier,
       purged_kfold, walk_forward_train, financial_metrics,
   )
   from sklearn.ensemble import RandomForestClassifier

   # Engineer features from TA indicators
   features = technical_features(prices, indicators=["rsi", "macd", "adx"])

   # Triple-barrier labels (de Prado method)
   labels = label_triple_barrier(prices, profit_target=0.02, stop_loss=0.01)

   # Align features and labels
   X, y = features.align(labels['label'], join='inner', axis=0)

   # Purged cross-validation (no information leakage)
   splits = purged_kfold(X, y, n_splits=5, purge_gap=10)

   # Walk-forward training (gold standard for financial ML)
   result = walk_forward_train(
       X, y,
       model=RandomForestClassifier(n_estimators=100),
       train_size=504,
       test_size=63,
   )
   print(f"OOS accuracy: {result['mean_accuracy']:.4f}")

   # Evaluate as a trading strategy
   fin = financial_metrics(result['strategy_returns'])
   print(f"Sharpe: {fin['sharpe_ratio']:.4f}")

Feature Importance
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from wraquant.ml import feature_importance_mda

   # MDA: permutation-based importance (preferred over MDI)
   mda = feature_importance_mda(model, X, y, purged_cv=True)
   print(mda['importance'].head(10))

Deep Learning
^^^^^^^^^^^^^

.. code-block:: python

   from wraquant.ml import lstm_forecast, transformer_forecast

   result = lstm_forecast(X, y, hidden_size=64, sequence_length=21, epochs=50)
   print(f"LSTM accuracy: {result['test_accuracy']:.4f}")

.. seealso::

   - :doc:`/tutorials/ml_alpha_research` -- Full ML alpha research tutorial
   - :doc:`ta` -- TA indicators used for feature engineering
   - :doc:`backtest` -- Backtest ML-generated signals

API Reference
-------------

.. automodule:: wraquant.ml
   :members:
   :undoc-members:
   :show-inheritance:

Features
^^^^^^^^

Feature engineering functions for transforming raw market data into
predictive signals.

.. automodule:: wraquant.ml.features
   :members:

Preprocessing
^^^^^^^^^^^^^

Purged CV, fractional differentiation, and correlation matrix denoising.

.. automodule:: wraquant.ml.preprocessing
   :members:

Models
^^^^^^

Walk-forward training, ensembles, and feature importance.

.. automodule:: wraquant.ml.models
   :members:

Deep Learning
^^^^^^^^^^^^^

LSTM, GRU, Transformer, and autoencoder architectures for time-series
forecasting. Requires PyTorch.

.. automodule:: wraquant.ml.deep
   :members:

Advanced Models
^^^^^^^^^^^^^^^

SVM, Random Forest, Gradient Boosting, Gaussian Process, Isolation Forest,
PCA factor models.

.. automodule:: wraquant.ml.advanced
   :members:

Clustering
^^^^^^^^^^

Correlation-based clustering, regime clustering, optimal cluster selection.

.. automodule:: wraquant.ml.clustering
   :members:

Evaluation
^^^^^^^^^^

Classification metrics, financial metrics, learning curves, and backtest
evaluation of predictions.

.. automodule:: wraquant.ml.evaluation
   :members:

Online Learning
^^^^^^^^^^^^^^^

Incrementally updating models for streaming data.

.. automodule:: wraquant.ml.online
   :members:

Pipeline
^^^^^^^^

``FinancialPipeline``, walk-forward backtest, and SHAP integration.

.. automodule:: wraquant.ml.pipeline
   :members:
