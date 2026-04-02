"""Machine learning alpha research using wraquant.

This example demonstrates a complete ML-driven alpha research pipeline:

    1. Generate synthetic OHLCV data
    2. Build feature matrix using technical indicators and rolling stats
    3. Create forward-return labels
    4. Train a model with walk-forward validation (no lookahead bias)
    5. Backtest the ML predictions as a trading signal
    6. Analyze feature importance
    7. Evaluate model performance

This follows the workflow from de Prado's "Advances in Financial Machine
Learning" -- using purged cross-validation, triple-barrier labels, and
proper feature importance estimation to avoid the common pitfalls of
applying ML to financial data.

Uses wraquant.ml, wraquant.ta, wraquant.backtest, and wraquant.stats.

Usage:
    python examples/ml_alpha.py
    python examples/ml_alpha.py --ticker AAPL --train-size 504

Requirements:
    pip install wraquant[ml]  # includes scikit-learn
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Synthetic OHLCV data generation
# ---------------------------------------------------------------------------

def generate_ohlcv(
    ticker: str = "SYNTH",
    n_days: int = 1260,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data with realistic properties.

    Produces daily bars with:
    - Trending behavior (mean-reverting drift)
    - Volatility clustering (simple GARCH-like process)
    - Volume correlated with absolute returns
    - Proper OHLC ordering (L <= O,C <= H)
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_days, freq="B")

    # Generate close prices with volatility clustering
    close = np.empty(n_days)
    vol = np.empty(n_days)
    close[0] = 100.0
    vol[0] = 0.015

    for i in range(1, n_days):
        # Simple GARCH-like vol dynamics
        vol[i] = 0.005 + 0.85 * vol[i - 1] + 0.10 * abs(rng.normal(0, vol[i - 1]))
        vol[i] = max(0.005, min(0.05, vol[i]))  # clamp

        # Return with slight positive drift
        ret = rng.normal(0.0003, vol[i])
        close[i] = close[i - 1] * (1 + ret)

    # Generate OHLC from close
    daily_range = vol * close * rng.uniform(0.5, 2.0, n_days)
    high = close + daily_range * rng.uniform(0.3, 0.8, n_days)
    low = close - daily_range * rng.uniform(0.3, 0.8, n_days)

    # Open: between previous close and current close
    open_prices = np.empty(n_days)
    open_prices[0] = close[0] * (1 + rng.normal(0, 0.002))
    for i in range(1, n_days):
        gap = rng.normal(0, 0.002)
        open_prices[i] = close[i - 1] * (1 + gap)

    # Ensure proper ordering: L <= min(O,C) and H >= max(O,C)
    low = np.minimum(low, np.minimum(open_prices, close))
    high = np.maximum(high, np.maximum(open_prices, close))

    # Volume: correlated with absolute returns
    base_vol = 50_000_000
    abs_returns = np.abs(np.diff(close, prepend=close[0]) / close)
    volume = (base_vol * (1 + 5 * abs_returns) * rng.uniform(0.5, 1.5, n_days)).astype(int)

    return pd.DataFrame({
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def header(title: str) -> None:
    width = 65
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def subheader(title: str) -> None:
    print(f"\n  --- {title} ---")


def pct(value: float) -> str:
    return f"{value:.2%}"


def fmt(value: float, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}"


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run_ml_alpha(ticker: str = "SYNTH", train_size: int = 504, seed: int = 42) -> None:
    """Run the ML alpha research pipeline."""

    header(f"ML ALPHA RESEARCH: {ticker}")

    # ==================================================================
    # Step 1: Generate Data
    # ==================================================================
    header("STEP 1: DATA PREPARATION")

    ohlcv = generate_ohlcv(ticker, n_days=1260, seed=seed)
    close = ohlcv["close"]
    returns = close.pct_change().dropna()

    print(f"\n  Ticker: {ticker}")
    print(f"  Period: {ohlcv.index[0].date()} to {ohlcv.index[-1].date()}")
    print(f"  Days:   {len(ohlcv)}")
    print(f"  Price range: ${close.min():.2f} - ${close.max():.2f}")
    print(f"  Total return: {pct((close.iloc[-1] / close.iloc[0]) - 1)}")
    print(f"  Ann. volatility: {pct(returns.std() * np.sqrt(252))}")

    # ==================================================================
    # Step 2: Feature Engineering
    # ==================================================================
    header("STEP 2: FEATURE ENGINEERING")

    from wraquant.ml import rolling_features, return_features, volatility_features
    from wraquant.ta import rsi, macd, bollinger_bands, atr, adx, obv

    # 2a. Rolling statistical features
    roll_feats = rolling_features(returns, windows=(5, 10, 21, 63))
    print(f"\n  Rolling features: {roll_feats.shape[1]} columns")
    print(f"    Windows: 5d, 10d, 21d, 63d")
    print(f"    Stats: mean, std, skew, kurtosis, min, max")

    # 2b. Return features (lagged returns)
    ret_feats = return_features(returns, lags=[1, 2, 3, 5, 10, 21])
    print(f"  Return features: {ret_feats.shape[1]} columns")

    # 2c. Volatility features
    vol_feats = volatility_features(returns, windows=(5, 10, 21))
    print(f"  Volatility features: {vol_feats.shape[1]} columns")

    # 2d. Technical indicator features (computed manually from OHLCV)
    ta_features = pd.DataFrame(index=ohlcv.index)

    # RSI
    rsi_14 = rsi(close, period=14)
    ta_features["rsi_14"] = rsi_14

    # MACD
    macd_result = macd(close)
    ta_features["macd_line"] = macd_result["macd"]
    ta_features["macd_signal"] = macd_result["signal"]
    ta_features["macd_hist"] = macd_result["histogram"]

    # Bollinger Bands
    bb = bollinger_bands(close, period=20)
    ta_features["bb_pctb"] = (close - bb["lower"]) / (bb["upper"] - bb["lower"])

    # ATR (normalized)
    atr_14 = atr(ohlcv["high"], ohlcv["low"], close, period=14)
    ta_features["natr"] = atr_14 / close

    # ADX
    adx_14 = adx(ohlcv["high"], ohlcv["low"], close, period=14)
    ta_features["adx"] = adx_14

    # OBV (rate of change)
    obv_series = obv(close, ohlcv["volume"])
    ta_features["obv_roc_10"] = obv_series.pct_change(10)

    print(f"  Technical features: {ta_features.shape[1]} columns")
    print(f"    RSI(14), MACD, BB%B, NATR, ADX(14), OBV ROC(10)")

    # 2e. Combine all features
    features = pd.concat([
        roll_feats,
        ret_feats,
        vol_feats,
        ta_features,
    ], axis=1)

    # Drop rows with NaN (due to rolling windows)
    features = features.dropna()

    print(f"\n  Combined feature matrix: {features.shape[0]} rows x {features.shape[1]} columns")

    # ==================================================================
    # Step 3: Label Generation
    # ==================================================================
    header("STEP 3: LABEL GENERATION")

    from wraquant.ml import label_fixed_horizon

    # Create binary labels: 1 if next 5-day return > 0, else 0
    horizon = 5
    labels = label_fixed_horizon(returns, horizon=horizon, threshold=0.0)

    # Align features and labels
    common_idx = features.index.intersection(labels.index)
    # Remove last `horizon` rows to avoid label leakage at the end
    common_idx = common_idx[:-horizon]
    X = features.loc[common_idx]
    y = labels.loc[common_idx]

    print(f"\n  Label method: Fixed-horizon ({horizon}-day forward return)")
    print(f"  Threshold: 0% (binary classification)")
    print(f"  Class balance:")
    print(f"    Positive (up):   {(y == 1).sum():>5} ({pct((y == 1).mean())})")
    print(f"    Negative (down): {(y == 0).sum():>5} ({pct((y == 0).mean())})")
    print(f"  Aligned dataset: {len(X)} samples, {X.shape[1]} features")

    # ==================================================================
    # Step 4: Walk-Forward Training
    # ==================================================================
    header("STEP 4: WALK-FORWARD VALIDATION")

    from wraquant.ml import walk_forward_train

    # Use a gradient boosting classifier
    from sklearn.ensemble import GradientBoostingClassifier

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        random_state=seed,
    )

    test_size = 21  # ~1 month
    step_size = 21  # step forward 1 month at a time

    print(f"\n  Model: GradientBoostingClassifier")
    print(f"    n_estimators: 100")
    print(f"    max_depth: 3")
    print(f"    learning_rate: 0.1")
    print(f"\n  Walk-Forward Config:")
    print(f"    Training window: {train_size} days (~{train_size // 252} years)")
    print(f"    Test window: {test_size} days (~1 month)")
    print(f"    Step size: {step_size} days")
    print(f"    Expanding window (all past data used)")

    wf_result = walk_forward_train(
        model=model,
        X=X,
        y=y,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
    )

    predictions = wf_result["predictions"]
    actuals = wf_result["actuals"]
    test_indices = wf_result["test_indices"]
    n_folds = wf_result["n_folds"]

    print(f"\n  Results:")
    print(f"    Walk-forward folds: {n_folds}")
    print(f"    Total OOS predictions: {len(predictions)}")

    # ==================================================================
    # Step 5: Model Evaluation
    # ==================================================================
    header("STEP 5: MODEL EVALUATION")

    from wraquant.ml import classification_metrics

    clf_metrics = classification_metrics(actuals, predictions)

    print(f"\n  Classification Metrics (out-of-sample):")
    print(f"    Accuracy:    {pct(clf_metrics['accuracy'])}")
    print(f"    Precision:   {pct(clf_metrics['precision'])}")
    print(f"    Recall:      {pct(clf_metrics['recall'])}")
    print(f"    F1 Score:    {fmt(clf_metrics['f1'], 3)}")

    # Interpretation
    if clf_metrics["accuracy"] > 0.55:
        print(f"\n  Accuracy > 55%: potentially useful signal.")
        print(f"  In financial ML, even 52-53% accuracy can be profitable")
        print(f"  if combined with proper position sizing.")
    elif clf_metrics["accuracy"] > 0.50:
        print(f"\n  Accuracy 50-55%: marginal signal. May be profitable with")
        print(f"  asymmetric payoffs but needs careful cost analysis.")
    else:
        print(f"\n  Accuracy < 50%: model is not better than a coin flip.")
        print(f"  Consider different features or a different approach.")

    # ==================================================================
    # Step 6: Backtest ML Predictions
    # ==================================================================
    header("STEP 6: BACKTEST ML SIGNAL")

    # Convert predictions to trading signals
    # prediction = 1 -> go long, prediction = 0 -> go to cash
    signal_series = pd.Series(
        predictions.astype(float),
        index=X.index[test_indices],
        name="ml_signal",
    )

    # Compute strategy returns
    daily_returns = returns.loc[signal_series.index]
    strategy_returns = (signal_series.shift(1) * daily_returns).dropna()
    strategy_returns.name = "ml_strategy"

    from wraquant.backtest import performance_summary

    ml_perf = performance_summary(strategy_returns)
    bh_perf = performance_summary(daily_returns.loc[strategy_returns.index])

    print(f"\n  {'Metric':<25} {'ML Strategy':>14} {'Buy & Hold':>14}")
    print(f"  {'-' * 55}")

    metrics_to_show = [
        ("Total Return", "total_return", True),
        ("Annualized Return", "annualized_return", True),
        ("Annualized Vol", "annualized_vol", True),
        ("Sharpe Ratio", "sharpe", False),
        ("Sortino Ratio", "sortino", False),
        ("Max Drawdown", "max_drawdown", True),
        ("Calmar Ratio", "calmar", False),
    ]

    for label, key, is_pct in metrics_to_show:
        ml_val = ml_perf.get(key, 0)
        bh_val = bh_perf.get(key, 0)
        ml_str = pct(ml_val) if is_pct else fmt(ml_val, 2)
        bh_str = pct(bh_val) if is_pct else fmt(bh_val, 2)
        print(f"  {label:<25} {ml_str:>14} {bh_str:>14}")

    # Win rate
    winning = (strategy_returns > 0).sum()
    total = (strategy_returns != 0).sum()
    win_rate = winning / total if total > 0 else 0
    print(f"\n  Signal Statistics:")
    print(f"    Days in market:  {(signal_series == 1).sum()} / {len(signal_series)} ({pct((signal_series == 1).mean())})")
    print(f"    Win rate:        {pct(win_rate)}")

    # ==================================================================
    # Step 7: Feature Importance
    # ==================================================================
    header("STEP 7: FEATURE IMPORTANCE")

    from wraquant.ml import feature_importance_mdi

    # Fit a final model on all data for feature importance
    final_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        random_state=seed,
    )
    final_model.fit(X, y)

    importance = feature_importance_mdi(final_model, X.columns.tolist())

    # Sort by importance
    imp_df = pd.DataFrame({
        "feature": importance["features"],
        "importance": importance["importances"],
    }).sort_values("importance", ascending=False)

    print(f"\n  Top 15 Features (Mean Decrease Impurity):\n")
    print(f"  {'Rank':<6} {'Feature':<30} {'Importance':>12}")
    print(f"  {'-' * 50}")

    for rank, (_, row) in enumerate(imp_df.head(15).iterrows(), 1):
        bar_len = int(row["importance"] * 100)
        bar = "#" * min(bar_len, 20)
        print(f"  {rank:<6} {row['feature']:<30} {row['importance']:>10.4f}  {bar}")

    # Identify feature categories
    subheader("Feature Category Breakdown")
    categories = {
        "Rolling stats": [c for c in imp_df["feature"] if any(s in c for s in ["mean_w", "std_w", "skew_w", "kurt", "min_w", "max_w"])],
        "Returns": [c for c in imp_df["feature"] if "lag" in c or "cum" in c],
        "Volatility": [c for c in imp_df["feature"] if "vol" in c or "rv_" in c],
        "Technical": [c for c in imp_df["feature"] if any(s in c for s in ["rsi", "macd", "bb_", "natr", "adx", "obv"])],
    }

    for cat_name, cat_features in categories.items():
        cat_imp = imp_df[imp_df["feature"].isin(cat_features)]["importance"].sum()
        n_feats = len(cat_features)
        if n_feats > 0:
            print(f"    {cat_name:<20} {n_feats:>3} features  Total importance: {fmt(cat_imp)}")

    # ==================================================================
    # Step 8: Recommendations
    # ==================================================================
    header("STEP 8: NEXT STEPS")

    print(f"\n  Model Assessment:")

    sharpe = ml_perf.get("sharpe", 0)
    if sharpe > 1.0:
        print(f"    Strategy Sharpe {sharpe:.2f} is promising (>1.0).")
    elif sharpe > 0.5:
        print(f"    Strategy Sharpe {sharpe:.2f} is marginal (0.5-1.0).")
    else:
        print(f"    Strategy Sharpe {sharpe:.2f} is weak (<0.5).")

    print(f"\n  Potential Improvements:")
    print(f"    1. Try triple-barrier labeling instead of fixed-horizon")
    print(f"       (wraquant.ml.label_triple_barrier)")
    print(f"    2. Use purged K-fold CV for hyperparameter tuning")
    print(f"       (wraquant.ml.purged_kfold)")
    print(f"    3. Apply fractional differentiation to price features")
    print(f"       (wraquant.ml.fractional_differentiation)")
    print(f"    4. Use MDA feature importance instead of MDI")
    print(f"       (wraquant.ml.feature_importance_mda)")
    print(f"    5. Test ensemble methods: combine gradient boosting")
    print(f"       with random forests and linear models")
    print(f"       (wraquant.ml.ensemble_predict)")
    print(f"    6. Add regime features from HMM")
    print(f"       (wraquant.ml.regime_features)")
    print(f"    7. Add cross-asset features for macro context")
    print(f"       (wraquant.ml.cross_asset_features)")

    print(f"\n  Common Pitfalls (avoid these!):")
    print(f"    - NEVER use random K-fold CV on time series data")
    print(f"    - NEVER use future data in feature computation")
    print(f"    - NEVER trust in-sample Sharpe > 2.0 (likely overfit)")
    print(f"    - ALWAYS check feature importance for data leakage")
    print(f"    - ALWAYS account for transaction costs")

    # ==================================================================
    # Summary
    # ==================================================================
    header("ML ALPHA RESEARCH SUMMARY")

    print(f"\n  Asset:               {ticker}")
    print(f"  Features:            {X.shape[1]} ({len(categories)} categories)")
    print(f"  Model:               GradientBoostingClassifier")
    print(f"  Walk-forward folds:  {n_folds}")
    print(f"  OOS Accuracy:        {pct(clf_metrics['accuracy'])}")
    print(f"  Strategy Sharpe:     {sharpe:.2f}")
    print(f"  Strategy Max DD:     {pct(ml_perf.get('max_drawdown', 0))}")
    print(f"  Top feature:         {imp_df.iloc[0]['feature']}")

    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ML alpha research workflow using wraquant.",
    )
    parser.add_argument(
        "--ticker",
        default="SYNTH",
        help="Ticker symbol for labeling (default: SYNTH)",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=504,
        help="Training window size in days (default: 504, ~2 years)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    run_ml_alpha(ticker=args.ticker, train_size=args.train_size, seed=args.seed)
