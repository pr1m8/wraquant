"""Machine learning MCP tools.

Tools: build_features, train_model, feature_importance, walk_forward_ml,
pca_factors, isolation_forest, svm_classify, online_regression,
cross_asset_features.
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

    @mcp.tool()
    def pca_factors(
        dataset: str,
        n_factors: int = 3,
    ) -> dict[str, Any]:
        """Extract latent factors from multi-asset returns via PCA.

        The first PC typically captures the market factor, the second
        captures sector rotation or value/growth, etc. Useful for
        dimensionality reduction and factor model construction.

        Parameters:
            dataset: Dataset with multi-asset returns (one column per asset).
            n_factors: Number of principal components to extract.
        """
        import numpy as np
        import pandas as pd

        from wraquant.ml.advanced import pca_factor_model

        df = ctx.get_dataset(dataset)
        returns = df.select_dtypes(include=[np.number]).dropna()

        result = pca_factor_model(returns, n_components=n_factors)

        if isinstance(result, dict) and "factors" in result:
            factor_df = pd.DataFrame(result["factors"])
            stored = ctx.store_dataset(
                f"pca_{dataset}", factor_df,
                source_op="pca_factors", parent=dataset,
            )
        else:
            stored = {}

        model_name = f"pca_{dataset}"
        model_stored = ctx.store_model(
            model_name, result,
            model_type="pca",
            source_dataset=dataset,
        )

        return _sanitize_for_json({
            "tool": "pca_factors",
            "dataset": dataset,
            "n_factors": n_factors,
            "assets": list(returns.columns),
            **model_stored,
            **stored,
            "explained_variance": result.get("explained_variance_ratio")
            if isinstance(result, dict) else None,
        })

    @mcp.tool()
    def isolation_forest(
        dataset: str,
        contamination: float = 0.05,
    ) -> dict[str, Any]:
        """Detect anomalies in return data using Isolation Forest.

        Identifies outlier days (flash crashes, liquidity events)
        by measuring how quickly observations are isolated via
        random partitioning.

        Parameters:
            dataset: Dataset containing returns or features.
            contamination: Expected fraction of anomalies (0-1).
        """
        import numpy as np
        import pandas as pd

        from wraquant.ml.advanced import isolation_forest_anomaly

        df = ctx.get_dataset(dataset)
        numeric = df.select_dtypes(include=[np.number]).dropna()

        result = isolation_forest_anomaly(numeric, contamination=contamination)

        if isinstance(result, dict) and "anomaly_labels" in result:
            anom_df = pd.DataFrame({
                "anomaly": result["anomaly_labels"],
            })
            stored = ctx.store_dataset(
                f"iforest_{dataset}", anom_df,
                source_op="isolation_forest", parent=dataset,
            )
            n_anomalies = int((result["anomaly_labels"] == -1).sum()) \
                if hasattr(result["anomaly_labels"], "sum") else None
        else:
            stored = {}
            n_anomalies = None

        return _sanitize_for_json({
            "tool": "isolation_forest",
            "dataset": dataset,
            "contamination": contamination,
            "n_anomalies": n_anomalies,
            "n_observations": len(numeric),
            **stored,
            "result": {k: v for k, v in result.items()
                       if not hasattr(v, "__len__") or isinstance(v, str)}
            if isinstance(result, dict) else str(result),
        })

    @mcp.tool()
    def svm_classify(
        dataset: str,
        target_col: str = "label",
        feature_cols_json: str = "[]",
    ) -> dict[str, Any]:
        """Train an SVM classifier for market regime or signal classification.

        Uses grid search over kernel and hyperparameters with
        chronological train/test split.

        Parameters:
            dataset: Dataset with features and target.
            target_col: Target/label column name.
            feature_cols_json: JSON list of feature column names.
                If empty, uses all numeric columns except target.
        """
        import json

        import numpy as np

        from wraquant.ml.advanced import svm_classifier

        df = ctx.get_dataset(dataset)

        feature_cols = json.loads(feature_cols_json) \
            if feature_cols_json and feature_cols_json != "[]" else []

        if not feature_cols:
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in numeric if c != target_col]

        X = df[feature_cols].dropna()
        y = df[target_col].loc[X.index]

        # Chronological split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        result = svm_classifier(X_train, y_train, X_test, y_test)

        model_name = f"svm_{dataset}"
        stored = ctx.store_model(
            model_name, result,
            model_type="svm",
            source_dataset=dataset,
        )

        return _sanitize_for_json({
            "tool": "svm_classify",
            "dataset": dataset,
            "target_col": target_col,
            "features": feature_cols,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            **stored,
            "result": {k: v for k, v in result.items()
                       if not hasattr(v, "__len__") or isinstance(v, str)}
            if isinstance(result, dict) else str(result),
        })

    @mcp.tool()
    def online_regression(
        dataset: str,
        y_col: str,
        x_cols_json: str = "[]",
        halflife: int = 60,
    ) -> dict[str, Any]:
        """Online (recursive) regression with time-varying coefficients.

        Uses Recursive Least Squares with exponential forgetting
        to track coefficient evolution over time. Essential for
        detecting factor exposure drift.

        Parameters:
            dataset: Dataset with dependent and independent variables.
            y_col: Dependent variable column.
            x_cols_json: JSON list of independent variable columns.
                If empty, uses all numeric columns except y.
            halflife: Halflife for exponential forgetting (in periods).
                Smaller = faster adaptation. Converted to forgetting factor.
        """
        import json
        import math

        import numpy as np
        import pandas as pd

        from wraquant.ml.online import online_linear_regression

        df = ctx.get_dataset(dataset)

        x_cols = json.loads(x_cols_json) \
            if x_cols_json and x_cols_json != "[]" else []

        if not x_cols:
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            x_cols = [c for c in numeric if c != y_col]

        X = df[x_cols].dropna()
        y = df[y_col].loc[X.index].dropna()

        n = min(len(X), len(y))
        X = X.iloc[:n]
        y = y.iloc[:n]

        # Convert halflife to forgetting factor: lambda = 2^(-1/halflife)
        forgetting = math.pow(2, -1.0 / halflife)

        result = online_linear_regression(X, y, forgetting_factor=forgetting)

        if isinstance(result, dict) and "coefficients" in result:
            coef_df = pd.DataFrame(result["coefficients"])
            stored = ctx.store_dataset(
                f"online_reg_{dataset}", coef_df,
                source_op="online_regression", parent=dataset,
            )
        else:
            stored = {}

        model_name = f"online_reg_{dataset}"
        model_stored = ctx.store_model(
            model_name, result,
            model_type="online_regression",
            source_dataset=dataset,
        )

        return _sanitize_for_json({
            "tool": "online_regression",
            "dataset": dataset,
            "y_col": y_col,
            "x_cols": x_cols,
            "halflife": halflife,
            "forgetting_factor": forgetting,
            **model_stored,
            **stored,
        })

    @mcp.tool()
    def cross_asset_features(
        dataset: str,
        benchmark_dataset: str,
        window: int = 60,
    ) -> dict[str, Any]:
        """Build cross-asset features: rolling correlation, beta, relative vol.

        Creates a feature dataset of inter-asset relationships
        useful for regime detection and portfolio construction.

        Parameters:
            dataset: Dataset with asset returns.
            benchmark_dataset: Dataset with benchmark returns.
            window: Rolling window for feature computation.
        """
        import numpy as np
        import pandas as pd

        df = ctx.get_dataset(dataset)
        bdf = ctx.get_dataset(benchmark_dataset)

        asset_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        bench_cols = bdf.select_dtypes(include=[np.number]).columns.tolist()

        # Use first numeric column from each if single-column
        if len(asset_cols) == 0 or len(bench_cols) == 0:
            return {"error": "Both datasets need numeric columns"}

        asset_ret = df[asset_cols[0]].dropna()
        bench_ret = bdf[bench_cols[0]].dropna()

        n = min(len(asset_ret), len(bench_ret))
        asset_ret = asset_ret.iloc[-n:]
        bench_ret = bench_ret.iloc[-n:]

        features = pd.DataFrame(index=asset_ret.index)

        # Rolling correlation
        features["rolling_corr"] = asset_ret.rolling(window).corr(bench_ret)

        # Rolling beta
        rolling_cov = asset_ret.rolling(window).cov(bench_ret)
        rolling_var = bench_ret.rolling(window).var()
        features["rolling_beta"] = rolling_cov / rolling_var

        # Relative volatility
        features["rel_vol"] = (
            asset_ret.rolling(window).std() / bench_ret.rolling(window).std()
        )

        # Rolling alpha (intercept from beta regression)
        features["rolling_alpha"] = (
            asset_ret.rolling(window).mean()
            - features["rolling_beta"] * bench_ret.rolling(window).mean()
        )

        # Tracking error
        diff = asset_ret - bench_ret
        features["tracking_error"] = diff.rolling(window).std() * np.sqrt(252)

        features = features.dropna()

        stored = ctx.store_dataset(
            f"xasset_{dataset}", features,
            source_op="cross_asset_features", parent=dataset,
        )

        return _sanitize_for_json({
            "tool": "cross_asset_features",
            "dataset": dataset,
            "benchmark_dataset": benchmark_dataset,
            "window": window,
            "n_features": len(features.columns),
            "feature_names": list(features.columns),
            "n_samples": len(features),
            "latest": features.iloc[-1].to_dict() if len(features) > 0 else {},
            **stored,
        })
