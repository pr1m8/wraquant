ML Alpha Research
=================

This tutorial covers the full machine learning pipeline for financial
prediction: feature engineering from technical indicators, proper
cross-validation with purged K-fold, walk-forward training, evaluation
with financial metrics, and experiment tracking.

Financial ML fails more often than it succeeds. The signal-to-noise ratio
is extremely low, and standard ML practices (random shuffled CV, naive
features) introduce lookahead bias and overfit to noise. wraquant is
designed to avoid these pitfalls.


Step 1: Feature Engineering
-----------------------------

Start by transforming raw price data into predictive features. wraquant
provides specialized feature generators that produce financially meaningful
inputs.

.. code-block:: python

   import wraquant as wq
   import pandas as pd
   from wraquant.ml import (
       technical_features, return_features, volatility_features,
       rolling_features, label_triple_barrier,
   )

   prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)["Close"]

   # Technical indicator features (wraps wraquant.ta)
   ta_feats = technical_features(prices, indicators=["rsi", "macd", "bbwidth", "adx"])
   print(f"TA features shape: {ta_feats.shape}")
   print(f"Columns: {ta_feats.columns.tolist()}")

   # Return features: lagged returns at multiple horizons
   ret_feats = return_features(prices, lags=[1, 2, 5, 10, 21])
   print(f"Return features shape: {ret_feats.shape}")

   # Volatility features: realized vol, vol-of-vol, GARCH residuals
   vol_feats = volatility_features(prices, windows=[10, 21, 63])
   print(f"Volatility features shape: {vol_feats.shape}")

   # Rolling statistics: z-score, skew, kurtosis at multiple windows
   roll_feats = rolling_features(prices, windows=[21, 63], stats=["zscore", "skew", "kurt"])
   print(f"Rolling features shape: {roll_feats.shape}")

   # Combine all features
   features = pd.concat([ta_feats, ret_feats, vol_feats, roll_feats], axis=1).dropna()
   print(f"\nTotal features: {features.shape[1]}")


Step 2: Label Construction
----------------------------

Labels define what you are predicting. The triple-barrier method (de Prado)
is preferred over fixed-horizon returns because it accounts for stop-losses
and profit targets.

.. code-block:: python

   # Triple-barrier labels
   labels = label_triple_barrier(
       prices,
       profit_target=0.02,   # 2% profit target
       stop_loss=0.01,       # 1% stop loss
       max_holding=10,       # 10-day maximum holding period
   )
   print(f"Label distribution:\n{labels['label'].value_counts()}")
   # 1 = hit profit target first
   # -1 = hit stop loss first
   # 0 = expired (time barrier hit)

   # Align features and labels
   common_idx = features.index.intersection(labels.index)
   X = features.loc[common_idx]
   y = labels.loc[common_idx, 'label']
   print(f"\nAligned samples: {len(X)}")


Step 3: Preprocessing
-----------------------

Financial data requires special preprocessing to avoid lookahead bias and
handle noisy correlation matrices.

.. code-block:: python

   from wraquant.ml import (
       fractional_differentiation, denoised_correlation,
       detoned_correlation,
   )

   # Fractional differentiation: make features stationary while preserving
   # memory (unlike simple differencing which destroys long-range info)
   X_frac = fractional_differentiation(X, d=0.5)
   print(f"Fractional diff features shape: {X_frac.shape}")

   # Denoise the correlation matrix using Random Matrix Theory
   # (Marcenko-Pastur): shrink noisy eigenvalues to reduce overfitting
   corr_denoised = denoised_correlation(X)
   print(f"Denoised correlation shape: {corr_denoised.shape}")

   # Detone: remove the market mode (first eigenvector) to expose
   # sector-level structure
   corr_detoned = detoned_correlation(X)


Step 4: Purged Cross-Validation
---------------------------------

Standard K-fold CV shuffles data randomly, causing information leakage
between train and test folds. Purged K-fold respects chronological ordering
and inserts a gap to prevent overlap between label windows.

.. code-block:: python

   from wraquant.ml import purged_kfold, combinatorial_purged_kfold
   from sklearn.ensemble import RandomForestClassifier

   # Purged K-fold: time-respecting splits with a purge gap
   splits = purged_kfold(X, y, n_splits=5, purge_gap=10)

   # Train and evaluate on each fold
   scores = []
   for i, (train_idx, test_idx) in enumerate(splits):
       X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
       y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

       model = RandomForestClassifier(n_estimators=100, max_depth=5)
       model.fit(X_train, y_train)
       score = model.score(X_test, y_test)
       scores.append(score)
       print(f"  Fold {i+1}: accuracy={score:.4f}, "
             f"train={len(X_train)}, test={len(X_test)}")

   print(f"\nMean accuracy: {sum(scores)/len(scores):.4f}")

   # Compare with combinatorial purged K-fold for more robust estimates
   combo_splits = combinatorial_purged_kfold(X, y, n_splits=5, purge_gap=10)
   print(f"Combinatorial splits generated: {len(combo_splits)}")


Step 5: Walk-Forward Training
-------------------------------

Walk-forward is the gold standard for financial ML evaluation. It trains
on an expanding or rolling window and predicts out-of-sample at each step,
simulating real-time deployment.

.. code-block:: python

   from wraquant.ml import walk_forward_train

   # Walk-forward with expanding window
   wf_result = walk_forward_train(
       X, y,
       model=RandomForestClassifier(n_estimators=100, max_depth=5),
       train_size=504,        # ~2 years initial training
       test_size=63,          # ~3 months test
       step_size=63,          # advance by one test period
       expanding=True,        # expanding window (vs rolling)
   )

   print(f"Walk-forward results:")
   print(f"  Windows: {wf_result['n_windows']}")
   print(f"  Mean accuracy: {wf_result['mean_accuracy']:.4f}")
   print(f"  Std accuracy:  {wf_result['std_accuracy']:.4f}")

   # The predictions from each test window form a continuous OOS series
   oos_predictions = wf_result['predictions']
   print(f"  OOS predictions: {len(oos_predictions)}")


Step 6: Feature Importance
----------------------------

Understand which features drive predictions. MDA (Mean Decrease Accuracy)
is preferred over MDI because it accounts for substitution effects.

.. code-block:: python

   from wraquant.ml import feature_importance_mda, feature_importance_mdi

   # Mean Decrease Accuracy: permute each feature and measure accuracy drop
   mda = feature_importance_mda(
       model, X, y, n_repeats=10, purged_cv=True, purge_gap=10
   )
   print("Top 10 features by MDA importance:")
   for feat, imp in mda['importance'].head(10).items():
       print(f"  {feat}: {imp:.4f}")

   # MDI for comparison (faster but biased toward high-cardinality features)
   mdi = feature_importance_mdi(model, X)
   print("\nTop 10 features by MDI importance:")
   for feat, imp in mdi['importance'].head(10).items():
       print(f"  {feat}: {imp:.4f}")

   # Prune features with zero or negative MDA importance
   important_features = mda['importance'][mda['importance'] > 0].index.tolist()
   print(f"\nKeeping {len(important_features)} of {X.shape[1]} features")
   X_pruned = X[important_features]


Step 7: Financial Evaluation
------------------------------

Standard ML metrics (accuracy, F1) do not capture what matters for trading.
Evaluate predictions as a trading signal and compute strategy-level metrics.

.. code-block:: python

   from wraquant.ml import financial_metrics, backtest_predictions

   # Convert predictions to a P&L series
   strategy = backtest_predictions(
       predictions=oos_predictions,
       prices=prices,
   )

   # Financial metrics on the resulting strategy
   fin = financial_metrics(strategy['returns'])
   print(f"Strategy Sharpe:  {fin['sharpe_ratio']:.4f}")
   print(f"Strategy Sortino: {fin['sortino_ratio']:.4f}")
   print(f"Max drawdown:     {fin['max_drawdown']:.2%}")
   print(f"Hit rate:         {fin['hit_rate']:.2%}")
   print(f"Profit factor:    {fin['profit_factor']:.4f}")

   # A model can have 60% accuracy but a negative Sharpe if the
   # losing trades are much larger than the winners. Always evaluate
   # with financial metrics, not just classification metrics.


Step 8: SHAP Explanation
--------------------------

Use SHAP values to understand model predictions at the individual
observation level.

.. code-block:: python

   from wraquant.ml import feature_importance_shap

   # SHAP values for the most recent predictions
   shap_result = feature_importance_shap(model, X_pruned.tail(252))
   print("SHAP feature importance (global):")
   for feat, imp in shap_result['mean_shap'].head(10).items():
       print(f"  {feat}: {imp:.4f}")

   # shap_result['shap_values'] contains per-observation SHAP values
   # for interpreting individual predictions.


Step 9: Deep Learning Models
-------------------------------

For sequence-dependent patterns, use LSTM, GRU, or Transformer architectures.
These require the PyTorch extra (``pip install wraquant[ml]``).

.. code-block:: python

   from wraquant.ml import lstm_forecast, transformer_forecast

   # LSTM forecast
   lstm_result = lstm_forecast(
       X_pruned, y,
       hidden_size=64,
       n_layers=2,
       sequence_length=21,    # look back 21 days
       epochs=50,
       train_ratio=0.8,
   )
   print(f"LSTM test accuracy: {lstm_result['test_accuracy']:.4f}")

   # Transformer forecast (attention-based)
   tf_result = transformer_forecast(
       X_pruned, y,
       d_model=64,
       n_heads=4,
       n_layers=2,
       sequence_length=21,
       epochs=50,
       train_ratio=0.8,
   )
   print(f"Transformer test accuracy: {tf_result['test_accuracy']:.4f}")


Next Steps
----------

- :doc:`/tutorials/backtesting_strategies` -- Turn your ML predictions into
  a full backtest with position sizing and tearsheets.
- :doc:`/tutorials/regime_investing` -- Add regime features to your ML model
  or condition predictions on regime state.
- :doc:`/api/ml` -- Full API reference for all ML functions.
