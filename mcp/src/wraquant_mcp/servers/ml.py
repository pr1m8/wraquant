"""Machine learning MCP tools.

Tools: build_features, train_model, feature_importance, walk_forward_ml.
"""

from __future__ import annotations

from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_ml_tools(mcp, ctx: AnalysisContext) -> None:
    """Register ML-specific tools on the MCP server."""

    @mcp.tool()
    def build_features(
        dataset: str,
        price_column: str = "close",
        feature_sets: list[str] | None = None,
        windows: list[int] | None = None,
        label_method: str = "fixed_horizon",
        label_horizon: int = 5,
    ) -> dict[str, Any]:
        """Build ML features from price data.

        Generates return features, volatility features, rolling
        statistics, and labels for supervised learning.

        Parameters:
            dataset: Dataset with price data.
            price_column: Price column name.
            feature_sets: Feature families to include. Options:
                'returns', 'volatility', 'rolling', 'technical'.
                Defaults to all.
            windows: Rolling windows for features. Defaults to
                [5, 10, 21, 63].
            label_method: Labeling method ('fixed_horizon' or
                'triple_barrier').
            label_horizon: Forward-looking horizon for labels.
        """
        import pandas as pd

        from wraquant.ml.features import (
            label_fixed_horizon,
            label_triple_barrier,
            return_features,
            rolling_features,
            volatility_features,
        )

        df = ctx.get_dataset(dataset)
        prices = df[price_column]

        if windows is None:
            windows = [5, 10, 21, 63]
        if feature_sets is None:
            feature_sets = ["returns", "volatility", "rolling"]

        frames = []

        if "returns" in feature_sets:
            ret_feats = return_features(prices)
            if isinstance(ret_feats, pd.DataFrame):
                frames.append(ret_feats)
            else:
                frames.append(pd.DataFrame(ret_feats))

        if "volatility" in feature_sets:
            vol_feats = volatility_features(prices)
            if isinstance(vol_feats, pd.DataFrame):
                frames.append(vol_feats)
            else:
                frames.append(pd.DataFrame(vol_feats))

        if "rolling" in feature_sets:
            for w in windows:
                roll_feats = rolling_features(prices, window=w)
                if isinstance(roll_feats, pd.DataFrame):
                    frames.append(roll_feats)
                else:
                    frames.append(pd.DataFrame(roll_feats))

        if frames:
            features_df = pd.concat(frames, axis=1)
        else:
            features_df = pd.DataFrame(index=df.index)

        # Add labels
        returns = prices.pct_change()
        if label_method == "triple_barrier":
            labels = label_triple_barrier(prices, horizon=label_horizon)
        else:
            labels = label_fixed_horizon(returns, horizon=label_horizon)

        if isinstance(labels, pd.Series):
            features_df["label"] = labels
        elif isinstance(labels, dict):
            for k, v in labels.items():
                features_df[f"label_{k}"] = v

        features_df = features_df.dropna()

        stored = ctx.store_dataset(
            f"features_{dataset}", features_df,
            source_op="build_features", parent=dataset,
        )

        return _sanitize_for_json({
            "tool": "build_features",
            "n_features": len(features_df.columns) - 1,
            "n_samples": len(features_df),
            "feature_names": list(features_df.columns),
            **stored,
        })

    @mcp.tool()
    def train_model(
        dataset: str,
        label_column: str = "label",
        model_type: str = "random_forest",
        test_size: float = 0.2,
    ) -> dict[str, Any]:
        """Train a machine learning model on feature data.

        Parameters:
            dataset: Feature dataset (from build_features).
            label_column: Label/target column name.
            model_type: Model type. Options: 'random_forest',
                'gradient_boost', 'svm'.
            test_size: Fraction of data for testing (uses
                chronological split, not random).
        """
        import numpy as np

        from wraquant.ml.advanced import (
            gradient_boost_forecast,
            random_forest_importance,
            svm_classifier,
        )

        df = ctx.get_dataset(dataset)

        if label_column not in df.columns:
            return {"error": f"Label column '{label_column}' not found"}

        numeric = df.select_dtypes(include=[np.number])
        feature_cols = [c for c in numeric.columns if c != label_column]
        X = numeric[feature_cols].dropna()
        y = numeric[label_column].loc[X.index]

        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        trainers = {
            "random_forest": lambda: random_forest_importance(X_train, y_train),
            "gradient_boost": lambda: gradient_boost_forecast(X_train, y_train),
            "svm": lambda: svm_classifier(X_train, y_train),
        }

        func = trainers.get(model_type)
        if func is None:
            return {"error": f"Unknown model '{model_type}'. Options: {list(trainers)}"}

        result = func()

        model_name = f"ml_{dataset}_{model_type}"
        metrics = {}
        if isinstance(result, dict):
            metrics = {k: float(v) for k, v in result.items()
                       if isinstance(v, (int, float, np.floating, np.integer))}

        stored = ctx.store_model(
            model_name, result,
            model_type=model_type,
            source_dataset=dataset,
            metrics=metrics,
        )

        return _sanitize_for_json({
            "tool": "train_model",
            "model_type": model_type,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": feature_cols,
            **stored,
        })

    @mcp.tool()
    def feature_importance(
        model_name: str,
        method: str = "mdi",
    ) -> dict[str, Any]:
        """Get feature importance from a trained model.

        Parameters:
            model_name: Name of stored ML model.
            method: Importance method ('mdi' = Mean Decrease Impurity,
                'mda' = Mean Decrease Accuracy).
        """
        model = ctx.get_model(model_name)

        result = {}
        if isinstance(model, dict):
            if "feature_importance" in model:
                result = model["feature_importance"]
            elif "importances" in model:
                result = model["importances"]
        elif hasattr(model, "feature_importances_"):
            result = dict(enumerate(model.feature_importances_.tolist()))

        return _sanitize_for_json({
            "tool": "feature_importance",
            "model": model_name,
            "method": method,
            "importances": result,
        })

    @mcp.tool()
    def walk_forward_ml(
        dataset: str,
        label_column: str = "label",
        model_type: str = "random_forest",
        train_window: int = 252,
        test_window: int = 21,
    ) -> dict[str, Any]:
        """Run walk-forward ML training and prediction.

        Trains on expanding/rolling windows and predicts out-of-sample
        at each step -- the gold standard for financial ML validation.

        Parameters:
            dataset: Feature dataset (from build_features).
            label_column: Label column.
            model_type: Model type.
            train_window: Training window size.
            test_window: Test window size.
        """
        import numpy as np

        from wraquant.ml.models import walk_forward_train

        df = ctx.get_dataset(dataset)

        if label_column not in df.columns:
            return {"error": f"Label column '{label_column}' not found"}

        numeric = df.select_dtypes(include=[np.number])
        feature_cols = [c for c in numeric.columns if c != label_column]
        X = numeric[feature_cols].dropna()
        y = numeric[label_column].loc[X.index]

        result = walk_forward_train(
            X, y,
            train_window=train_window,
            test_window=test_window,
        )

        model_name = f"wfml_{dataset}_{model_type}"
        stored = ctx.store_model(
            model_name, result,
            model_type=f"walk_forward_{model_type}",
            source_dataset=dataset,
        )

        return _sanitize_for_json({
            "tool": "walk_forward_ml",
            "model_type": model_type,
            "train_window": train_window,
            "test_window": test_window,
            "result": result if isinstance(result, dict) else str(result),
            **stored,
        })
