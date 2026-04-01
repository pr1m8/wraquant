"""Statistical analysis MCP tools.

Tools: correlation_analysis, distribution_fit, regression,
cointegration_test, stationarity_tests, robust_stats.
"""

from __future__ import annotations

from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_stats_tools(mcp, ctx: AnalysisContext) -> None:
    """Register statistics-specific tools on the MCP server."""

    @mcp.tool()
    def correlation_analysis(
        dataset: str,
        method: str = "pearson",
        shrink: bool = False,
    ) -> dict[str, Any]:
        """Compute correlation matrix with optional shrinkage.

        Requires a multi-column dataset (each column = one series).

        Parameters:
            dataset: Dataset with multiple numeric columns.
            method: Correlation method ('pearson', 'spearman', 'kendall').
            shrink: If True, apply Ledoit-Wolf shrinkage.
        """
        import numpy as np

        from wraquant.stats.correlation import correlation_matrix, shrunk_covariance

        df = ctx.get_dataset(dataset)
        numeric = df.select_dtypes(include=[np.number])

        if shrink:
            result = shrunk_covariance(numeric)
        else:
            result = correlation_matrix(numeric, method=method)

        import pandas as pd

        corr_df = pd.DataFrame(result, columns=numeric.columns, index=numeric.columns)
        stored = ctx.store_dataset(
            f"corr_{dataset}", corr_df,
            source_op="correlation_analysis", parent=dataset,
        )

        return _sanitize_for_json({
            "tool": "correlation_analysis",
            "method": method if not shrink else "ledoit_wolf",
            "shape": list(corr_df.shape),
            "mean_correlation": float(corr_df.values[np.triu_indices_from(
                corr_df.values, k=1
            )].mean()) if corr_df.shape[0] > 1 else None,
            **stored,
        })

    @mcp.tool()
    def distribution_fit(
        dataset: str,
        column: str = "returns",
        distributions: list[str] | None = None,
    ) -> dict[str, Any]:
        """Fit statistical distributions and test goodness-of-fit.

        Parameters:
            dataset: Dataset containing the series.
            column: Column to analyze.
            distributions: List of distributions to fit.
                Defaults to auto-selecting best fit.
        """
        from wraquant.stats.distributions import (
            best_fit_distribution,
            fit_distribution,
            jarque_bera,
        )

        df = ctx.get_dataset(dataset)
        data = df[column].dropna()

        jb = jarque_bera(data)

        if distributions:
            results = {}
            for dist_name in distributions:
                results[dist_name] = fit_distribution(data, distribution=dist_name)
            best = min(results, key=lambda k: results[k].get("aic", float("inf")))
            return _sanitize_for_json({
                "tool": "distribution_fit",
                "dataset": dataset,
                "jarque_bera": jb,
                "fits": results,
                "best": best,
            })
        else:
            best = best_fit_distribution(data)
            return _sanitize_for_json({
                "tool": "distribution_fit",
                "dataset": dataset,
                "jarque_bera": jb,
                "best_fit": best,
            })

    @mcp.tool()
    def regression(
        dataset: str,
        y_column: str,
        x_columns: list[str] | None = None,
        method: str = "ols",
        window: int | None = None,
    ) -> dict[str, Any]:
        """Run regression analysis.

        Parameters:
            dataset: Dataset with dependent and independent variables.
            y_column: Dependent variable column.
            x_columns: Independent variable columns.
                If None, uses all numeric columns except y.
            method: 'ols', 'wls', 'rolling', or 'newey_west'.
            window: Rolling window (required for method='rolling').
        """
        import numpy as np

        from wraquant.stats.regression import newey_west_ols, ols, rolling_ols, wls

        df = ctx.get_dataset(dataset)

        if x_columns is None:
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            x_columns = [c for c in numeric if c != y_column]

        y = df[y_column].dropna()
        X = df[x_columns].dropna()

        n = min(len(y), len(X))
        y = y.iloc[:n]
        X = X.iloc[:n]

        methods = {
            "ols": lambda: ols(y, X),
            "wls": lambda: wls(y, X),
            "newey_west": lambda: newey_west_ols(y, X),
            "rolling": lambda: rolling_ols(y, X, window=window or 60),
        }

        func = methods.get(method)
        if func is None:
            return {"error": f"Unknown method '{method}'. Options: {list(methods)}"}

        result = func()

        model_name = f"reg_{dataset}_{method}"
        stored = ctx.store_model(
            model_name, result,
            model_type=f"regression_{method}",
            source_dataset=dataset,
        )

        return _sanitize_for_json({**stored, "method": method})

    @mcp.tool()
    def cointegration_test(
        dataset: str,
        column_a: str,
        column_b: str,
        method: str = "engle_granger",
    ) -> dict[str, Any]:
        """Test for cointegration between two series.

        Parameters:
            dataset: Dataset containing both series.
            column_a: First series column.
            column_b: Second series column.
            method: 'engle_granger' or 'johansen'.
        """
        from wraquant.stats.cointegration import engle_granger, half_life, hedge_ratio

        df = ctx.get_dataset(dataset)
        a = df[column_a].dropna()
        b = df[column_b].dropna()

        n = min(len(a), len(b))
        a = a.iloc[:n]
        b = b.iloc[:n]

        if method == "johansen":
            from wraquant.stats.cointegration import johansen

            result = johansen(a, b)
        else:
            result = engle_granger(a, b)

        hr = hedge_ratio(a, b)
        hl = half_life(a - hr * b)

        return _sanitize_for_json({
            "tool": "cointegration_test",
            "dataset": dataset,
            "columns": [column_a, column_b],
            "method": method,
            "result": result,
            "hedge_ratio": float(hr),
            "half_life": float(hl),
        })

    @mcp.tool()
    def stationarity_tests(
        dataset: str,
        column: str = "returns",
    ) -> dict[str, Any]:
        """Run comprehensive stationarity tests (ADF, KPSS, PP, VR).

        Parameters:
            dataset: Dataset containing the series.
            column: Column to test.
        """
        from wraquant.ts.stationarity import (
            adf_test,
            kpss_test,
            phillips_perron,
            variance_ratio_test,
        )

        df = ctx.get_dataset(dataset)
        data = df[column].dropna()

        results = {
            "adf": adf_test(data),
            "kpss": kpss_test(data),
            "phillips_perron": phillips_perron(data),
            "variance_ratio": variance_ratio_test(data),
        }

        return _sanitize_for_json({
            "tool": "stationarity_tests",
            "dataset": dataset,
            "column": column,
            "observations": len(data),
            **results,
        })

    @mcp.tool()
    def robust_stats(
        dataset: str,
        column: str = "returns",
    ) -> dict[str, Any]:
        """Compute robust statistics resistant to outliers.

        Returns MAD, trimmed mean, Huber mean, winsorized stats,
        and outlier detection.

        Parameters:
            dataset: Dataset containing the series.
            column: Column to analyze.
        """
        from wraquant.stats.robust import (
            huber_mean,
            mad,
            outlier_detection,
            trimmed_mean,
            winsorize,
        )

        df = ctx.get_dataset(dataset)
        data = df[column].dropna()

        outliers = outlier_detection(data)
        winsorized = winsorize(data)

        return _sanitize_for_json({
            "tool": "robust_stats",
            "dataset": dataset,
            "column": column,
            "mad": float(mad(data)),
            "trimmed_mean": float(trimmed_mean(data)),
            "huber_mean": float(huber_mean(data)),
            "winsorized_mean": float(winsorized.mean()),
            "winsorized_std": float(winsorized.std()),
            "outliers": outliers,
            "n_outliers": int(outliers["n_outliers"])
            if isinstance(outliers, dict) and "n_outliers" in outliers
            else None,
        })
