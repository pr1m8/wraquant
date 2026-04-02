"""Visualization MCP tools.

Tools: plot_returns, plot_regime, plot_correlation, plot_distribution,
plot_equity_curve, plot_drawdown, plot_rolling_metrics, plot_candlestick,
plot_heatmap, plot_vol_surface, plot_tearsheet, portfolio_dashboard,
regime_dashboard, plot_factor_exposure.
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

    # ------------------------------------------------------------------
    # New tools — expanded viz coverage
    # ------------------------------------------------------------------

    @mcp.tool()
    def plot_equity_curve(
        dataset: str,
        column: str = "returns",
    ) -> dict[str, Any]:
        """Plot a cumulative returns equity curve.

        Returns a base64-encoded PNG image of the cumulative wealth
        index derived from a simple return series.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
        """
        from wraquant.viz.interactive import plotly_returns

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()

        fig = plotly_returns(returns)
        img = _fig_to_base64(fig)

        return _sanitize_for_json({
            "tool": "plot_equity_curve",
            "dataset": dataset,
            "column": column,
            "image_base64": img,
            "format": "png",
        })

    @mcp.tool()
    def plot_drawdown(
        dataset: str,
        column: str = "returns",
    ) -> dict[str, Any]:
        """Plot an underwater (drawdown) chart.

        Returns a base64-encoded PNG of the drawdown time series
        with recovery periods highlighted.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
        """
        from wraquant.viz.interactive import plotly_drawdown

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()

        fig = plotly_drawdown(returns)
        img = _fig_to_base64(fig)

        return _sanitize_for_json({
            "tool": "plot_drawdown",
            "dataset": dataset,
            "column": column,
            "image_base64": img,
            "format": "png",
        })

    @mcp.tool()
    def plot_rolling_metrics(
        dataset: str,
        column: str = "returns",
        window: int = 63,
    ) -> dict[str, Any]:
        """Plot rolling Sharpe, volatility, and trend beta.

        Returns a base64-encoded PNG with three vertically-stacked
        subplot panels showing rolling risk-adjusted metrics.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
            window: Rolling window in trading days (default 63).
        """
        from wraquant.viz.interactive import plotly_rolling_stats

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()

        fig = plotly_rolling_stats(returns, window=window)
        img = _fig_to_base64(fig)

        return _sanitize_for_json({
            "tool": "plot_rolling_metrics",
            "dataset": dataset,
            "column": column,
            "window": window,
            "image_base64": img,
            "format": "png",
        })

    @mcp.tool()
    def plot_candlestick(
        dataset: str,
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
    ) -> dict[str, Any]:
        """Plot an interactive OHLCV candlestick chart.

        Returns a base64-encoded PNG candlestick chart with optional
        volume bars.

        Parameters:
            dataset: Dataset containing OHLCV data.
            open_col: Open price column name.
            high_col: High price column name.
            low_col: Low price column name.
            close_col: Close price column name.
            volume_col: Volume column name (ignored if absent).
        """
        import pandas as pd

        from wraquant.viz.candlestick import plotly_candlestick

        df = ctx.get_dataset(dataset)

        # Build a standardised OHLCV DataFrame
        col_map = {
            open_col: "open",
            high_col: "high",
            low_col: "low",
            close_col: "close",
        }
        if volume_col in df.columns:
            col_map[volume_col] = "volume"

        ohlcv = df.rename(columns=col_map)

        fig = plotly_candlestick(ohlcv)
        img = _fig_to_base64(fig)

        return _sanitize_for_json({
            "tool": "plot_candlestick",
            "dataset": dataset,
            "n_bars": len(ohlcv),
            "image_base64": img,
            "format": "png",
        })

    @mcp.tool()
    def plot_heatmap(
        dataset: str,
        method: str = "pearson",
    ) -> dict[str, Any]:
        """Plot a correlation heatmap of multi-asset returns.

        Returns a base64-encoded PNG heatmap coloured by pairwise
        correlation strength.

        Parameters:
            dataset: Dataset with multiple numeric columns.
            method: Correlation method ('pearson' or 'spearman').
        """
        import numpy as np

        from wraquant.viz.interactive import plotly_correlation_heatmap

        df = ctx.get_dataset(dataset)
        numeric = df.select_dtypes(include=[np.number])

        fig = plotly_correlation_heatmap(numeric, method=method)
        img = _fig_to_base64(fig)

        return _sanitize_for_json({
            "tool": "plot_heatmap",
            "dataset": dataset,
            "method": method,
            "n_assets": len(numeric.columns),
            "image_base64": img,
            "format": "png",
        })

    @mcp.tool()
    def plot_vol_surface(
        strikes_json: str,
        maturities_json: str,
        vols_json: str,
    ) -> dict[str, Any]:
        """Plot a 3-D implied volatility surface.

        Returns a base64-encoded PNG of an interactive 3-D surface
        with strike on the x-axis, maturity on the y-axis, and
        implied volatility on the z-axis.

        Parameters:
            strikes_json: JSON array of strike prices.
            maturities_json: JSON array of maturities (years).
            vols_json: JSON 2-D array of implied vols, shape
                (len(maturities), len(strikes)).
        """
        import json

        import numpy as np

        from wraquant.viz.charts import plot_vol_surface as _plot_vol_surface

        strikes = np.array(json.loads(strikes_json), dtype=float)
        maturities = np.array(json.loads(maturities_json), dtype=float)
        vols = np.array(json.loads(vols_json), dtype=float)

        fig = _plot_vol_surface(strikes, maturities, vols)
        img = _fig_to_base64(fig)

        return _sanitize_for_json({
            "tool": "plot_vol_surface",
            "n_strikes": len(strikes),
            "n_maturities": len(maturities),
            "image_base64": img,
            "format": "png",
        })

    @mcp.tool()
    def plot_tearsheet(
        dataset: str,
        column: str = "returns",
    ) -> dict[str, Any]:
        """Plot a full multi-panel backtest tearsheet.

        Returns a base64-encoded PNG with equity curve, drawdown,
        monthly heatmap, return distribution, rolling metrics, and
        summary statistics.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
        """
        from wraquant.viz.charts import plot_backtest_tearsheet

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()

        fig = plot_backtest_tearsheet(returns)
        img = _fig_to_base64(fig)

        return _sanitize_for_json({
            "tool": "plot_tearsheet",
            "dataset": dataset,
            "column": column,
            "image_base64": img,
            "format": "png",
        })

    @mcp.tool()
    def portfolio_dashboard(
        dataset: str,
        column: str = "returns",
        benchmark_dataset: str | None = None,
        benchmark_column: str = "returns",
    ) -> dict[str, Any]:
        """Generate a 6-panel portfolio performance dashboard.

        Returns a base64-encoded PNG with cumulative returns, monthly
        heatmap, drawdown, return distribution, rolling Sharpe/Sortino,
        and summary metrics.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
            benchmark_dataset: Optional benchmark dataset.
            benchmark_column: Benchmark returns column.
        """
        from wraquant.viz.dashboard import portfolio_dashboard as _portfolio_dashboard

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()

        benchmark = None
        if benchmark_dataset:
            bdf = ctx.get_dataset(benchmark_dataset)
            benchmark = bdf[benchmark_column].dropna()

        fig = _portfolio_dashboard(returns, benchmark=benchmark)
        img = _fig_to_base64(fig)

        return _sanitize_for_json({
            "tool": "portfolio_dashboard",
            "dataset": dataset,
            "column": column,
            "image_base64": img,
            "format": "png",
        })

    @mcp.tool()
    def regime_dashboard(
        dataset: str,
        column: str = "returns",
        regime_dataset: str | None = None,
        regime_column: str = "regime",
    ) -> dict[str, Any]:
        """Generate a regime overlay dashboard.

        Returns a base64-encoded PNG with price/returns coloured by
        regime state with background shading.

        Parameters:
            dataset: Dataset containing returns or prices.
            column: Series column name.
            regime_dataset: Dataset with regime labels (from
                detect_regimes or regime_labels).
            regime_column: Column containing integer regime labels.
        """
        import pandas as pd

        from wraquant.viz.advanced import plotly_regime_overlay

        df = ctx.get_dataset(dataset)
        series = df[column].dropna()

        if regime_dataset:
            rdf = ctx.get_dataset(regime_dataset)
            regime_labels = rdf[regime_column]
        else:
            # Default: create simple vol-based regime labels
            from wraquant.regimes.labels import volatility_regime_labels

            vol_labels = volatility_regime_labels(series, n_levels=2)
            label_map = {"low_vol": 0, "high_vol": 1}
            regime_labels = vol_labels.map(label_map).fillna(0).astype(int)

        # Align indices
        common = series.index.intersection(regime_labels.index)
        series = series.loc[common]
        regime_labels = regime_labels.loc[common]

        fig = plotly_regime_overlay(series, pd.Series(regime_labels, index=common))
        img = _fig_to_base64(fig)

        return _sanitize_for_json({
            "tool": "regime_dashboard",
            "dataset": dataset,
            "column": column,
            "image_base64": img,
            "format": "png",
        })

    @mcp.tool()
    def plot_factor_exposure(
        dataset: str,
        factors_dataset: str,
    ) -> dict[str, Any]:
        """Plot factor exposure (beta) bar chart.

        Returns a base64-encoded PNG bar chart showing the estimated
        factor betas from regressing asset returns on factor returns.

        Parameters:
            dataset: Dataset containing asset returns.
            factors_dataset: Dataset containing factor return columns.
        """
        import numpy as np
        import plotly.graph_objects as go

        from wraquant.stats.factor_analysis import factor_exposure

        df = ctx.get_dataset(dataset)
        factors_df = ctx.get_dataset(factors_dataset)

        exposure_df = factor_exposure(df, factors_df)

        # Build a grouped bar chart of betas
        fig = go.Figure()

        if "beta" in exposure_df.columns:
            fig.add_trace(
                go.Bar(
                    x=exposure_df.index.astype(str),
                    y=exposure_df["beta"].values,
                    name="Beta",
                    marker_color=[
                        "#2ca02c" if v >= 0 else "#d62728"
                        for v in exposure_df["beta"].values
                    ],
                )
            )
        else:
            # DataFrame with factor columns as index/columns — plot first row
            betas = exposure_df.iloc[0] if len(exposure_df) > 0 else exposure_df
            fig.add_trace(
                go.Bar(
                    x=[str(c) for c in betas.index],
                    y=betas.values.astype(float),
                    name="Beta",
                    marker_color=[
                        "#2ca02c" if float(v) >= 0 else "#d62728"
                        for v in betas.values
                    ],
                )
            )

        fig.update_layout(
            title="Factor Exposures (Betas)",
            yaxis_title="Beta",
            template="plotly_dark",
            paper_bgcolor="#111111",
            plot_bgcolor="#1e1e1e",
        )

        img = _fig_to_base64(fig)

        return _sanitize_for_json({
            "tool": "plot_factor_exposure",
            "dataset": dataset,
            "factors_dataset": factors_dataset,
            "image_base64": img,
            "format": "png",
        })
