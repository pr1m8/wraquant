"""Statistical analysis MCP tools.

Tools: correlation_analysis, distribution_fit, regression,
cointegration_test, stationarity_tests, robust_stats,
partial_correlation, distance_correlation, mutual_information,
robust_statistics, kde_estimate, best_fit_distribution.
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
        try:
            import numpy as np

            from wraquant.stats.correlation import correlation_matrix, shrunk_covariance

            df = ctx.get_dataset(dataset)
            numeric = df.select_dtypes(include=[np.number])

            if shrink:
                result = shrunk_covariance(numeric)
            else:
                result = correlation_matrix(numeric, method=method)

            import pandas as pd

            corr_df = pd.DataFrame(
                result, columns=numeric.columns, index=numeric.columns
            )
            stored = ctx.store_dataset(
                f"corr_{dataset}",
                corr_df,
                source_op="correlation_analysis",
                parent=dataset,
            )

            return _sanitize_for_json(
                {
                    "tool": "correlation_analysis",
                    "method": method if not shrink else "ledoit_wolf",
                    "shape": list(corr_df.shape),
                    "mean_correlation": (
                        float(
                            corr_df.values[
                                np.triu_indices_from(corr_df.values, k=1)
                            ].mean()
                        )
                        if corr_df.shape[0] > 1
                        else None
                    ),
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "correlation_analysis"}

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
        try:
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
                return _sanitize_for_json(
                    {
                        "tool": "distribution_fit",
                        "dataset": dataset,
                        "jarque_bera": jb,
                        "fits": results,
                        "best": best,
                    }
                )
            else:
                best = best_fit_distribution(data)
                return _sanitize_for_json(
                    {
                        "tool": "distribution_fit",
                        "dataset": dataset,
                        "jarque_bera": jb,
                        "best_fit": best,
                    }
                )
        except Exception as e:
            return {"error": str(e), "tool": "distribution_fit"}

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
        try:
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
                model_name,
                result,
                model_type=f"regression_{method}",
                source_dataset=dataset,
            )

            return _sanitize_for_json({**stored, "method": method})
        except Exception as e:
            return {"error": str(e), "tool": "regression"}

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
        try:
            from wraquant.stats.cointegration import (
                engle_granger,
                half_life,
                hedge_ratio,
            )

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

            return _sanitize_for_json(
                {
                    "tool": "cointegration_test",
                    "dataset": dataset,
                    "columns": [column_a, column_b],
                    "method": method,
                    "result": result,
                    "hedge_ratio": float(hr),
                    "half_life": float(hl),
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "cointegration_test"}

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
        try:
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

            return _sanitize_for_json(
                {
                    "tool": "stationarity_tests",
                    "dataset": dataset,
                    "column": column,
                    "observations": len(data),
                    **results,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "stationarity_tests"}

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
        try:
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

            return _sanitize_for_json(
                {
                    "tool": "robust_stats",
                    "dataset": dataset,
                    "column": column,
                    "mad": float(mad(data)),
                    "trimmed_mean": float(trimmed_mean(data)),
                    "huber_mean": float(huber_mean(data)),
                    "winsorized_mean": float(winsorized.mean()),
                    "winsorized_std": float(winsorized.std()),
                    "outliers": outliers,
                    "n_outliers": (
                        int(outliers["n_outliers"])
                        if isinstance(outliers, dict) and "n_outliers" in outliers
                        else None
                    ),
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "robust_stats"}

    @mcp.tool()
    def partial_correlation(
        dataset: str,
        columns_json: str = "[]",
    ) -> dict[str, Any]:
        """Compute partial correlation matrix, controlling for other variables.

        Measures the direct linear relationship between each pair
        of variables after removing the effect of all others.
        Essential for distinguishing direct from mediated associations.

        Parameters:
            dataset: Dataset with multiple numeric columns.
            columns_json: JSON list of column names to include.
                If empty, uses all numeric columns.
        """
        try:
            import json

            import numpy as np
            import pandas as pd

            from wraquant.stats.correlation import partial_correlation as _pcorr

            df = ctx.get_dataset(dataset)

            cols = (
                json.loads(columns_json)
                if columns_json and columns_json != "[]"
                else []
            )
            if cols:
                data = df[cols].dropna()
            else:
                data = df.select_dtypes(include=[np.number]).dropna()

            result = _pcorr(data)

            pcorr_df = (
                pd.DataFrame(result, columns=data.columns, index=data.columns)
                if not isinstance(result, pd.DataFrame)
                else result
            )
            stored = ctx.store_dataset(
                f"pcorr_{dataset}",
                pcorr_df,
                source_op="partial_correlation",
                parent=dataset,
            )

            return _sanitize_for_json(
                {
                    "tool": "partial_correlation",
                    "dataset": dataset,
                    "columns": list(data.columns),
                    "shape": list(pcorr_df.shape),
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "partial_correlation"}

    @mcp.tool()
    def distance_correlation(
        dataset: str,
        col_a: str,
        col_b: str,
    ) -> dict[str, Any]:
        """Compute distance correlation between two variables.

        Unlike Pearson, distance correlation captures nonlinear
        dependence and equals zero if and only if the variables
        are independent.

        Parameters:
            dataset: Dataset containing both variables.
            col_a: First variable column.
            col_b: Second variable column.
        """
        try:
            from wraquant.stats.correlation import distance_correlation as _dcorr

            df = ctx.get_dataset(dataset)
            a = df[col_a].dropna()
            b = df[col_b].dropna()

            n = min(len(a), len(b))
            a = a.iloc[:n]
            b = b.iloc[:n]

            result = _dcorr(a, b)

            return _sanitize_for_json(
                {
                    "tool": "distance_correlation",
                    "dataset": dataset,
                    "columns": [col_a, col_b],
                    "distance_correlation": float(result),
                    "observations": n,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "distance_correlation"}

    @mcp.tool()
    def mutual_information(
        dataset: str,
        col_a: str,
        col_b: str,
    ) -> dict[str, Any]:
        """Estimate mutual information between two continuous variables.

        Captures any type of statistical dependence (linear, nonlinear,
        multimodal). Useful for feature selection and detecting hidden
        relationships.

        Parameters:
            dataset: Dataset containing both variables.
            col_a: First variable column.
            col_b: Second variable column.
        """
        try:
            from wraquant.stats.correlation import mutual_information as _mi

            df = ctx.get_dataset(dataset)
            a = df[col_a].dropna()
            b = df[col_b].dropna()

            n = min(len(a), len(b))
            a = a.iloc[:n]
            b = b.iloc[:n]

            result = _mi(a, b)

            return _sanitize_for_json(
                {
                    "tool": "mutual_information",
                    "dataset": dataset,
                    "columns": [col_a, col_b],
                    "mutual_information": float(result),
                    "observations": n,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "mutual_information"}

    @mcp.tool()
    def kde_estimate(
        dataset: str,
        column: str = "returns",
    ) -> dict[str, Any]:
        """Kernel density estimation for non-parametric distribution fitting.

        Estimates the probability density function without assuming
        a parametric form. Useful for visualizing return distributions
        and computing non-parametric VaR.

        Parameters:
            dataset: Dataset containing the series.
            column: Column to estimate density for.
        """
        try:
            import pandas as pd

            from wraquant.stats.distributions import kernel_density_estimate

            df = ctx.get_dataset(dataset)
            data = df[column].dropna()

            result = kernel_density_estimate(data)

            if isinstance(result, dict) and "x" in result and "density" in result:
                kde_df = pd.DataFrame(
                    {
                        "x": result["x"],
                        "density": result["density"],
                    }
                )
                stored = ctx.store_dataset(
                    f"kde_{dataset}_{column}",
                    kde_df,
                    source_op="kde_estimate",
                    parent=dataset,
                )
            else:
                stored = {}

            return _sanitize_for_json(
                {
                    "tool": "kde_estimate",
                    "dataset": dataset,
                    "column": column,
                    "observations": len(data),
                    **stored,
                    "result": (
                        {
                            k: v
                            for k, v in result.items()
                            if not hasattr(v, "__len__") or isinstance(v, str)
                        }
                        if isinstance(result, dict)
                        else str(result)
                    ),
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "kde_estimate"}

    @mcp.tool()
    def best_fit_distribution(
        dataset: str,
        column: str = "returns",
    ) -> dict[str, Any]:
        """Fit multiple distributions and rank by goodness-of-fit.

        Tests normal, t, skewed-t, stable, and other distributions
        against the data using AIC and KS statistics.

        Parameters:
            dataset: Dataset containing the series.
            column: Column to fit distributions to.
        """
        try:
            import pandas as pd

            from wraquant.stats.distributions import best_fit_distribution as _bfd

            df = ctx.get_dataset(dataset)
            data = df[column].dropna()

            result = _bfd(data)

            if isinstance(result, pd.DataFrame):
                stored = ctx.store_dataset(
                    f"bestfit_{dataset}_{column}",
                    result,
                    source_op="best_fit_distribution",
                    parent=dataset,
                )
                ranking = result.to_dict(orient="records")
            else:
                stored = {}
                ranking = result

            return _sanitize_for_json(
                {
                    "tool": "best_fit_distribution",
                    "dataset": dataset,
                    "column": column,
                    "observations": len(data),
                    **stored,
                    "ranking": ranking,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "best_fit_distribution"}
