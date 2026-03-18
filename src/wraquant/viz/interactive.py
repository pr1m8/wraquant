"""Core Plotly interactive chart wrappers for financial analysis.

Cumulative returns, drawdowns, rolling statistics, distributions,
correlation heatmaps, efficient frontier, and risk-return scatters ---
all as interactive Plotly figures with hover tooltips and rich formatting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wraquant.core.decorators import requires_extra
from wraquant.viz.themes import COLORS

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import pandas as pd
    import plotly.graph_objects as go

__all__ = [
    "plotly_returns",
    "plotly_drawdown",
    "plotly_rolling_stats",
    "plotly_distribution",
    "plotly_correlation_heatmap",
    "plotly_efficient_frontier",
    "plotly_risk_return_scatter",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLOTLY_TEMPLATE = "plotly_white"

_PALETTE = [
    COLORS["primary"],
    COLORS["secondary"],
    COLORS["positive"],
    COLORS["negative"],
    COLORS["accent"],
    COLORS["info"],
    COLORS["warning"],
]


def _base_layout(**overrides: object) -> dict:
    """Return a base Plotly layout dict with wraquant styling."""
    defaults: dict = dict(
        template=_PLOTLY_TEMPLATE,
        font=dict(family="sans-serif", size=12, color="#333333"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        margin=dict(l=60, r=30, t=50, b=50),
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@requires_extra("viz")
def plotly_returns(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    title: str | None = None,
) -> go.Figure:
    """Interactive cumulative returns chart with hover tooltips.

    Hover shows date, cumulative return, and current drawdown at each point.

    Parameters:
        returns: Simple (non-cumulative) return series.
        benchmark: Optional benchmark return series for comparison.
        title: Chart title.  Defaults to ``"Cumulative Returns"``.

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    import plotly.graph_objects as go

    cum = (1 + returns).cumprod() - 1
    running_max = (1 + returns).cumprod().cummax()
    drawdown = ((1 + returns).cumprod() - running_max) / running_max

    hover_text = [
        f"Date: {d:%Y-%m-%d}<br>Return: {r:.2%}<br>Drawdown: {dd:.2%}"
        for d, r, dd in zip(cum.index, cum.values, drawdown.values, strict=False)
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cum.index,
            y=cum.values,
            mode="lines",
            name=returns.name or "Strategy",
            line=dict(color=COLORS["primary"], width=2),
            text=hover_text,
            hoverinfo="text",
        )
    )

    if benchmark is not None:
        cum_bench = (1 + benchmark).cumprod() - 1
        bench_hover = [
            f"Date: {d:%Y-%m-%d}<br>Return: {r:.2%}"
            for d, r in zip(cum_bench.index, cum_bench.values, strict=False)
        ]
        fig.add_trace(
            go.Scatter(
                x=cum_bench.index,
                y=cum_bench.values,
                mode="lines",
                name=benchmark.name or "Benchmark",
                line=dict(color=COLORS["benchmark"], width=2, dash="dash"),
                text=bench_hover,
                hoverinfo="text",
            )
        )

    fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=0.8)
    fig.update_layout(
        **_base_layout(
            title=title or "Cumulative Returns",
            yaxis_title="Cumulative Return",
            yaxis_tickformat=".0%",
            showlegend=True,
        )
    )
    return fig


@requires_extra("viz")
def plotly_drawdown(
    returns: pd.Series,
) -> go.Figure:
    """Interactive underwater chart with recovery periods highlighted.

    Shades the drawdown area and annotates the deepest drawdown.

    Parameters:
        returns: Simple return series.

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    import numpy as np
    import plotly.graph_objects as go

    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max

    fig = go.Figure()

    # Filled area for drawdown
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill="tozeroy",
            mode="lines",
            name="Drawdown",
            line=dict(color=COLORS["drawdown"], width=1),
            fillcolor="rgba(214, 39, 40, 0.25)",
        )
    )

    # Highlight recovery periods (where drawdown is recovering toward 0)
    recovering = (drawdown < 0) & (drawdown.diff() > 0)
    spans: list[tuple[object, object]] = []
    in_span = False
    start = None
    for idx, val in recovering.items():
        if val and not in_span:
            in_span = True
            start = idx
        elif not val and in_span:
            in_span = False
            spans.append((start, idx))
    if in_span:
        spans.append((start, drawdown.index[-1]))

    for s, e in spans[:10]:  # limit shapes for performance
        fig.add_vrect(
            x0=s, x1=e,
            fillcolor="rgba(44, 160, 44, 0.08)",
            line_width=0,
            layer="below",
        )

    # Annotate max drawdown
    min_idx = drawdown.idxmin()
    min_val = drawdown.min()
    fig.add_annotation(
        x=min_idx,
        y=min_val,
        text=f"Max DD: {min_val:.2%}",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS["negative"],
        font=dict(color=COLORS["negative"], size=11),
    )

    fig.update_layout(
        **_base_layout(
            title="Underwater Plot (Drawdowns)",
            yaxis_title="Drawdown",
            yaxis_tickformat=".1%",
        )
    )
    return fig


@requires_extra("viz")
def plotly_rolling_stats(
    returns: pd.Series,
    window: int = 63,
) -> go.Figure:
    """Rolling Sharpe, volatility, and beta in vertically-stacked subplots.

    Parameters:
        returns: Simple return series.
        window: Rolling window in periods (default 63 ~ 1 quarter).

    Returns:
        A ``plotly.graph_objects.Figure`` with three subplot rows.
    """
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    roll_sharpe = (roll_mean / roll_std) * np.sqrt(252)
    roll_vol = roll_std * np.sqrt(252)
    # Rolling beta relative to own mean (auto-correlation proxy)
    roll_beta = returns.rolling(window).apply(
        lambda x: np.corrcoef(x, np.arange(len(x)))[0, 1],
        raw=True,
    )

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            f"Rolling {window}-Day Sharpe Ratio",
            f"Rolling {window}-Day Annualized Volatility",
            f"Rolling {window}-Day Trend Beta",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=roll_sharpe.index, y=roll_sharpe.values,
            mode="lines", name="Sharpe",
            line=dict(color=COLORS["primary"], width=1.5),
        ),
        row=1, col=1,
    )
    fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=0.6, row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=roll_vol.index, y=roll_vol.values,
            mode="lines", name="Volatility",
            line=dict(color=COLORS["secondary"], width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255, 127, 14, 0.15)",
        ),
        row=2, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=roll_beta.index, y=roll_beta.values,
            mode="lines", name="Trend Beta",
            line=dict(color=COLORS["accent"], width=1.5),
        ),
        row=3, col=1,
    )
    fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=0.6, row=3, col=1)

    fig.update_layout(
        **_base_layout(
            title=f"Rolling Statistics (window={window})",
            height=750,
            showlegend=False,
        )
    )
    fig.update_yaxes(tickformat=".1f", row=1, col=1)
    fig.update_yaxes(tickformat=".1%", row=2, col=1)
    fig.update_yaxes(tickformat=".2f", row=3, col=1)
    return fig


@requires_extra("viz")
def plotly_distribution(
    returns: pd.Series,
    bins: int = 50,
    overlay_normal: bool = True,
) -> go.Figure:
    """Interactive histogram with KDE and optional fitted normal overlay.

    Parameters:
        returns: Simple return series.
        bins: Number of histogram bins.
        overlay_normal: If *True*, overlay a fitted normal distribution PDF.

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    import numpy as np
    import plotly.graph_objects as go

    clean = returns.dropna().values

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=clean,
            nbinsx=bins,
            histnorm="probability density",
            name="Returns",
            marker_color=COLORS["primary"],
            opacity=0.65,
        )
    )

    # KDE
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(clean)
    x_grid = np.linspace(clean.min(), clean.max(), 300)
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=kde(x_grid),
            mode="lines",
            name="KDE",
            line=dict(color=COLORS["accent"], width=2),
        )
    )

    if overlay_normal:
        from scipy.stats import norm

        mu, sigma = clean.mean(), clean.std()
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=norm.pdf(x_grid, mu, sigma),
                mode="lines",
                name="Normal Fit",
                line=dict(color=COLORS["negative"], width=1.5, dash="dash"),
            )
        )

    # Annotate skew / kurtosis
    from scipy.stats import kurtosis, skew

    sk = skew(clean)
    ku = kurtosis(clean)
    fig.add_annotation(
        x=0.98, y=0.95, xref="paper", yref="paper",
        text=f"Skew: {sk:.2f}<br>Kurt: {ku:.2f}",
        showarrow=False,
        font=dict(size=11),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor=COLORS["neutral"],
    )

    fig.update_layout(
        **_base_layout(
            title="Return Distribution",
            xaxis_title="Return",
            yaxis_title="Density",
            bargap=0.02,
        )
    )
    return fig


@requires_extra("viz")
def plotly_correlation_heatmap(
    returns_df: pd.DataFrame,
) -> go.Figure:
    """Interactive correlation matrix heatmap with hierarchical clustering.

    Reorders assets by hierarchical clustering so that correlated groups
    appear together.  Hover shows the pair and correlation value.

    Parameters:
        returns_df: DataFrame of asset returns (columns = assets).

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    import numpy as np
    import plotly.graph_objects as go
    from scipy.cluster.hierarchy import leaves_list, linkage

    corr = returns_df.corr()
    # Hierarchical clustering to reorder
    dist = 1 - corr.values
    np.fill_diagonal(dist, 0)
    # Ensure symmetry
    dist = (dist + dist.T) / 2
    condensed = dist[np.triu_indices_from(dist, k=1)]
    Z = linkage(condensed, method="ward")
    order = leaves_list(Z)

    labels = [corr.columns[i] for i in order]
    ordered = corr.iloc[order, order]

    hover_text = []
    for i, row_label in enumerate(labels):
        row_texts = []
        for j, col_label in enumerate(labels):
            row_texts.append(
                f"{row_label} vs {col_label}<br>Corr: {ordered.iloc[i, j]:.3f}"
            )
        hover_text.append(row_texts)

    fig = go.Figure(
        data=go.Heatmap(
            z=ordered.values,
            x=labels,
            y=labels,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=hover_text,
            hoverinfo="text",
            colorbar=dict(title="Corr"),
        )
    )

    fig.update_layout(
        **_base_layout(
            title="Correlation Matrix (Hierarchically Clustered)",
            width=700,
            height=650,
            xaxis=dict(side="bottom"),
            yaxis=dict(autorange="reversed"),
        )
    )
    return fig


@requires_extra("viz")
def plotly_efficient_frontier(
    expected_returns: npt.NDArray[np.floating],
    cov_matrix: npt.NDArray[np.floating],
    n_portfolios: int = 5000,
) -> go.Figure:
    """Interactive efficient frontier with hover showing portfolio weights.

    Generates random portfolios and plots the risk-return cloud.
    The efficient frontier is highlighted.  Hovering over points reveals
    the weight vector.

    Parameters:
        expected_returns: 1-D array of expected returns per asset.
        cov_matrix: 2-D covariance matrix.
        n_portfolios: Number of random portfolios to simulate.

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    import numpy as np
    import plotly.graph_objects as go

    n_assets = len(expected_returns)
    rng = np.random.default_rng(42)

    port_returns = np.empty(n_portfolios)
    port_vols = np.empty(n_portfolios)
    all_weights = np.empty((n_portfolios, n_assets))

    for i in range(n_portfolios):
        w = rng.dirichlet(np.ones(n_assets))
        all_weights[i] = w
        port_returns[i] = w @ expected_returns
        port_vols[i] = np.sqrt(w @ cov_matrix @ w)

    sharpes = port_returns / port_vols

    # Build hover text with weights
    hover_texts = []
    for i in range(n_portfolios):
        parts = [f"Return: {port_returns[i]:.2%}", f"Vol: {port_vols[i]:.2%}"]
        parts.append(f"Sharpe: {sharpes[i]:.2f}")
        for j in range(n_assets):
            parts.append(f"w{j}: {all_weights[i, j]:.1%}")
        hover_texts.append("<br>".join(parts))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=port_vols,
            y=port_returns,
            mode="markers",
            marker=dict(
                size=4,
                color=sharpes,
                colorscale="Viridis",
                colorbar=dict(title="Sharpe"),
                opacity=0.7,
            ),
            text=hover_texts,
            hoverinfo="text",
            name="Portfolios",
        )
    )

    # Mark max-Sharpe portfolio
    best = int(np.argmax(sharpes))
    fig.add_trace(
        go.Scatter(
            x=[port_vols[best]],
            y=[port_returns[best]],
            mode="markers",
            marker=dict(
                size=14,
                color=COLORS["negative"],
                symbol="star",
                line=dict(width=1, color="white"),
            ),
            name="Max Sharpe",
            text=[hover_texts[best]],
            hoverinfo="text",
        )
    )

    fig.update_layout(
        **_base_layout(
            title="Efficient Frontier (Monte Carlo)",
            xaxis_title="Volatility",
            yaxis_title="Expected Return",
            xaxis_tickformat=".1%",
            yaxis_tickformat=".1%",
        )
    )
    return fig


@requires_extra("viz")
def plotly_risk_return_scatter(
    returns_df: pd.DataFrame,
) -> go.Figure:
    """Risk-return scatter plot with asset labels and clickable points.

    Annualizes both risk and return assuming 252 trading days.

    Parameters:
        returns_df: DataFrame of asset returns (columns = assets).

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    import numpy as np
    import plotly.graph_objects as go

    ann_ret = returns_df.mean() * 252
    ann_vol = returns_df.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ann_vol.values,
            y=ann_ret.values,
            mode="markers+text",
            text=returns_df.columns.tolist(),
            textposition="top center",
            marker=dict(
                size=12,
                color=sharpe.values,
                colorscale="Viridis",
                colorbar=dict(title="Sharpe"),
                line=dict(width=1, color="white"),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Return: %{y:.2%}<br>"
                "Volatility: %{x:.2%}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        **_base_layout(
            title="Risk-Return Scatter",
            xaxis_title="Annualized Volatility",
            yaxis_title="Annualized Return",
            xaxis_tickformat=".1%",
            yaxis_tickformat=".1%",
            showlegend=False,
        )
    )
    return fig
