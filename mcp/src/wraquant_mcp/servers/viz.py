"""Visualization MCP tools.

Tools: plot_returns, plot_regime, plot_correlation, plot_distribution.
Each returns a dict with a base64-encoded PNG image.
"""

from __future__ import annotations

import base64
import io
from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def _fig_to_base64(fig) -> str:
    """Convert a matplotlib or Plotly figure to base64 PNG."""
    buf = io.BytesIO()

    # Try matplotlib first
    try:
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor="#1a1a2e")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except AttributeError:
        pass

    # Try Plotly
    try:
        img_bytes = fig.to_image(format="png", width=1200, height=800)
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception:
        pass

    return ""


def register_viz_tools(mcp, ctx: AnalysisContext) -> None:
    """Register visualization-specific tools on the MCP server."""

    @mcp.tool()
    def plot_returns(
        dataset: str,
        column: str = "returns",
        cumulative: bool = True,
        benchmark_dataset: str | None = None,
        benchmark_column: str = "returns",
    ) -> dict[str, Any]:
        """Plot returns analysis (cumulative returns + drawdowns).

        Returns a base64-encoded PNG image.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
            cumulative: If True, plot cumulative returns.
            benchmark_dataset: Optional benchmark for comparison.
            benchmark_column: Benchmark returns column.
        """
        import matplotlib
        matplotlib.use("Agg")

        from wraquant.viz.returns import plot_cumulative_returns, plot_drawdowns

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()

        benchmark = None
        if benchmark_dataset:
            bdf = ctx.get_dataset(benchmark_dataset)
            benchmark = bdf[benchmark_column].dropna()

        if cumulative:
            fig = plot_cumulative_returns(returns, benchmark=benchmark)
        else:
            fig = plot_drawdowns(returns)

        img = _fig_to_base64(fig)

        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass

        return {
            "tool": "plot_returns",
            "dataset": dataset,
            "type": "cumulative" if cumulative else "drawdown",
            "image_base64": img,
            "format": "png",
        }

    @mcp.tool()
    def plot_regime(
        dataset: str,
        returns_column: str = "returns",
        states_dataset: str | None = None,
        states_column: str = "regime",
    ) -> dict[str, Any]:
        """Plot regime overlay on returns.

        Returns a base64-encoded PNG image showing returns
        colored by regime state.

        Parameters:
            dataset: Dataset containing returns.
            returns_column: Returns column.
            states_dataset: Dataset with regime states
                (from detect_regimes).
            states_column: Regime state column.
        """
        import matplotlib
        matplotlib.use("Agg")

        from wraquant.viz.timeseries import plot_regime_overlay

        df = ctx.get_dataset(dataset)
        returns = df[returns_column].dropna()

        states = None
        if states_dataset:
            sdf = ctx.get_dataset(states_dataset)
            states = sdf[states_column].values

        fig = plot_regime_overlay(returns, states=states)
        img = _fig_to_base64(fig)

        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass

        return {
            "tool": "plot_regime",
            "dataset": dataset,
            "image_base64": img,
            "format": "png",
        }

    @mcp.tool()
    def plot_correlation(
        dataset: str,
        method: str = "pearson",
    ) -> dict[str, Any]:
        """Plot correlation heatmap for multi-asset data.

        Returns a base64-encoded PNG image.

        Parameters:
            dataset: Dataset with multiple numeric columns.
            method: Correlation method ('pearson', 'spearman').
        """
        import matplotlib
        matplotlib.use("Agg")
        import numpy as np

        from wraquant.viz.interactive import plotly_correlation_heatmap

        df = ctx.get_dataset(dataset)
        numeric = df.select_dtypes(include=[np.number])

        fig = plotly_correlation_heatmap(numeric, method=method)
        img = _fig_to_base64(fig)

        return {
            "tool": "plot_correlation",
            "dataset": dataset,
            "method": method,
            "assets": list(numeric.columns),
            "image_base64": img,
            "format": "png",
        }

    @mcp.tool()
    def plot_distribution(
        dataset: str,
        column: str = "returns",
    ) -> dict[str, Any]:
        """Plot return distribution with fitted normal overlay.

        Returns a base64-encoded PNG image showing histogram,
        KDE, and QQ plot.

        Parameters:
            dataset: Dataset containing the series.
            column: Column to plot.
        """
        import matplotlib
        matplotlib.use("Agg")

        from wraquant.viz.charts import plot_distribution_analysis

        df = ctx.get_dataset(dataset)
        data = df[column].dropna()

        fig = plot_distribution_analysis(data)
        img = _fig_to_base64(fig)

        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass

        skew = float(data.skew())
        kurt = float(data.kurtosis())

        return _sanitize_for_json({
            "tool": "plot_distribution",
            "dataset": dataset,
            "column": column,
            "skewness": skew,
            "kurtosis": kurt,
            "image_base64": img,
            "format": "png",
        })
