"""Advanced and unconventional Plotly financial visualizations.

Regime overlays, 3-D volatility surfaces, animated yield curves, copula
scatters, network graphs, Sankey rebalancing flows, treemaps, and radar
charts --- the *wacky* side of quant viz.
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
    "plotly_regime_overlay",
    "plotly_vol_surface",
    "plotly_term_structure",
    "plotly_copula_scatter",
    "plotly_network_graph",
    "plotly_sankey_flow",
    "plotly_treemap",
    "plotly_radar",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLOTLY_TEMPLATE = "plotly_white"

_REGIME_COLORS = [
    "rgba(31, 119, 180, 0.18)",   # blue
    "rgba(255, 127, 14, 0.18)",   # orange
    "rgba(44, 160, 44, 0.18)",    # green
    "rgba(214, 39, 40, 0.18)",    # red
    "rgba(148, 103, 189, 0.18)",  # purple
    "rgba(140, 86, 75, 0.18)",    # brown
    "rgba(227, 119, 194, 0.18)",  # pink
    "rgba(188, 189, 34, 0.18)",   # olive
]

_REGIME_COLORS_SOLID = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#bcbd22",
]


def _base_layout(**overrides: object) -> dict:
    """Return a base Plotly layout dict with wraquant styling."""
    defaults: dict = dict(
        template=_PLOTLY_TEMPLATE,
        font=dict(family="sans-serif", size=12, color="#333333"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        margin=dict(l=60, r=30, t=50, b=50),
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@requires_extra("viz")
def plotly_regime_overlay(
    prices: pd.Series,
    regime_labels: pd.Series,
) -> go.Figure:
    """Price chart with colored background bands for market regimes.

    Parameters:
        prices: Price or level time series.
        regime_labels: Integer series (same index) indicating the regime
            at each observation.

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices.values,
            mode="lines",
            name=prices.name or "Price",
            line=dict(color=COLORS["primary"], width=1.8),
        )
    )

    # Build regime spans
    unique_regimes = sorted(regime_labels.unique())
    prev = regime_labels.iloc[0]
    start = prices.index[0]
    for idx, regime in zip(
        regime_labels.index[1:], regime_labels.iloc[1:], strict=False
    ):
        if regime != prev:
            fig.add_vrect(
                x0=start, x1=idx,
                fillcolor=_REGIME_COLORS[int(prev) % len(_REGIME_COLORS)],
                line_width=0,
                layer="below",
            )
            start = idx
            prev = regime
    # Final span
    fig.add_vrect(
        x0=start, x1=prices.index[-1],
        fillcolor=_REGIME_COLORS[int(prev) % len(_REGIME_COLORS)],
        line_width=0,
        layer="below",
    )

    # Invisible traces for the legend
    for r in unique_regimes:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(
                    size=12,
                    color=_REGIME_COLORS_SOLID[int(r) % len(_REGIME_COLORS_SOLID)],
                ),
                name=f"Regime {r}",
            )
        )

    fig.update_layout(
        **_base_layout(
            title="Price with Regime Overlay",
            yaxis_title="Price",
            showlegend=True,
        )
    )
    return fig


@requires_extra("viz")
def plotly_vol_surface(
    strikes: npt.NDArray[np.floating],
    expiries: npt.NDArray[np.floating],
    implied_vols: npt.NDArray[np.floating],
) -> go.Figure:
    """3-D implied volatility surface.

    Parameters:
        strikes: 1-D array of strike prices.
        expiries: 1-D array of expiries (e.g. days to expiry or years).
        implied_vols: 2-D array of shape ``(len(expiries), len(strikes))``
            containing implied volatilities.

    Returns:
        A ``plotly.graph_objects.Figure`` with a 3-D surface.
    """
    import numpy as np
    import plotly.graph_objects as go

    strike_grid, expiry_grid = np.meshgrid(strikes, expiries)

    fig = go.Figure(
        data=go.Surface(
            x=strike_grid,
            y=expiry_grid,
            z=implied_vols,
            colorscale="Plasma",
            colorbar=dict(title="IV"),
            hovertemplate=(
                "Strike: %{x:.1f}<br>"
                "Expiry: %{y:.2f}<br>"
                "IV: %{z:.2%}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Implied Volatility Surface",
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Expiry",
            zaxis_title="Implied Vol",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
        ),
        template=_PLOTLY_TEMPLATE,
        width=800,
        height=650,
    )
    return fig


@requires_extra("viz")
def plotly_term_structure(
    maturities: npt.NDArray[np.floating],
    yields: npt.NDArray[np.floating],
    dates: list[str] | None = None,
) -> go.Figure:
    """Animated yield curve through time.

    Each row of *yields* is a snapshot of the yield curve at one date.
    The animation steps through dates.

    Parameters:
        maturities: 1-D array of maturities (e.g. years).
        yields: 2-D array of shape ``(n_dates, len(maturities))``.
        dates: Optional list of date labels for the slider.

    Returns:
        A ``plotly.graph_objects.Figure`` with animation frames.
    """
    import numpy as np
    import plotly.graph_objects as go

    n_dates = yields.shape[0]
    if dates is None:
        dates = [f"t={i}" for i in range(n_dates)]

    # Initial frame
    fig = go.Figure(
        data=go.Scatter(
            x=maturities,
            y=yields[0],
            mode="lines+markers",
            line=dict(color=COLORS["primary"], width=2.5),
            marker=dict(size=6),
            name="Yield Curve",
        )
    )

    # Animation frames
    frames = []
    for i in range(n_dates):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=maturities,
                        y=yields[i],
                        mode="lines+markers",
                        line=dict(color=COLORS["primary"], width=2.5),
                        marker=dict(size=6),
                    )
                ],
                name=dates[i],
            )
        )
    fig.frames = frames

    # Slider and buttons
    sliders = [
        dict(
            active=0,
            steps=[
                dict(args=[[d], dict(frame=dict(duration=200, redraw=True),
                                     mode="immediate")],
                     label=d, method="animate")
                for d in dates
            ],
            currentvalue=dict(prefix="Date: "),
        )
    ]
    fig.update_layout(
        **_base_layout(
            title="Yield Curve Term Structure",
            xaxis_title="Maturity (Years)",
            yaxis_title="Yield",
            yaxis_tickformat=".2%",
            sliders=sliders,
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(label="Play",
                             method="animate",
                             args=[None, dict(frame=dict(duration=200),
                                              fromcurrent=True)]),
                        dict(label="Pause",
                             method="animate",
                             args=[[None], dict(frame=dict(duration=0),
                                                mode="immediate")]),
                    ],
                    x=0.05, y=1.12,
                )
            ],
        )
    )
    return fig


@requires_extra("viz")
def plotly_copula_scatter(
    u: npt.NDArray[np.floating],
    v: npt.NDArray[np.floating],
    copula_type: str = "empirical",
) -> go.Figure:
    """Copula scatter plot with marginal histograms.

    Parameters:
        u: 1-D array of uniform marginals for the first variable.
        v: 1-D array of uniform marginals for the second variable.
        copula_type: Label for the copula (used in the title).

    Returns:
        A ``plotly.graph_objects.Figure`` with marginal histograms.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.8, 0.2],
        row_heights=[0.2, 0.8],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
    )

    # Main scatter
    fig.add_trace(
        go.Scatter(
            x=u, y=v,
            mode="markers",
            marker=dict(size=3, color=COLORS["primary"], opacity=0.5),
            name="Copula",
        ),
        row=2, col=1,
    )

    # Top marginal histogram (u)
    fig.add_trace(
        go.Histogram(
            x=u, nbinsx=40,
            marker_color=COLORS["primary"],
            opacity=0.6,
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Right marginal histogram (v)
    fig.add_trace(
        go.Histogram(
            y=v, nbinsy=40,
            marker_color=COLORS["secondary"],
            opacity=0.6,
            showlegend=False,
        ),
        row=2, col=2,
    )

    fig.update_xaxes(range=[0, 1], row=2, col=1)
    fig.update_yaxes(range=[0, 1], row=2, col=1)

    fig.update_layout(
        **_base_layout(
            title=f"Copula Scatter ({copula_type})",
            showlegend=False,
            height=600,
            width=650,
        )
    )
    return fig


@requires_extra("viz")
def plotly_network_graph(
    correlation_matrix: npt.NDArray[np.floating] | pd.DataFrame,
    threshold: float = 0.5,
) -> go.Figure:
    """Asset correlation network graph.

    Draws edges between assets whose absolute correlation exceeds *threshold*.
    Node size is proportional to the number of edges.

    Parameters:
        correlation_matrix: Square correlation matrix (array or DataFrame).
        threshold: Minimum absolute correlation to draw an edge.

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    import math

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    if isinstance(correlation_matrix, pd.DataFrame):
        labels = list(correlation_matrix.columns)
        corr = correlation_matrix.values
    else:
        corr = np.asarray(correlation_matrix)
        labels = [str(i) for i in range(corr.shape[0])]

    n = len(labels)
    # Circular layout
    angles = [2 * math.pi * i / n for i in range(n)]
    x_nodes = [math.cos(a) for a in angles]
    y_nodes = [math.sin(a) for a in angles]

    # Build edges
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_colors: list[str] = []
    degree = [0] * n

    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr[i, j]) >= threshold:
                edge_x.extend([x_nodes[i], x_nodes[j], None])
                edge_y.extend([y_nodes[i], y_nodes[j], None])
                color = COLORS["positive"] if corr[i, j] > 0 else COLORS["negative"]
                edge_colors.append(color)
                degree[i] += 1
                degree[j] += 1

    fig = go.Figure()

    # Draw edges individually to color them
    for k in range(len(edge_colors)):
        fig.add_trace(
            go.Scatter(
                x=edge_x[k * 3:(k + 1) * 3],
                y=edge_y[k * 3:(k + 1) * 3],
                mode="lines",
                line=dict(width=1.5, color=edge_colors[k]),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # Draw nodes
    node_sizes = [max(12, 8 + d * 4) for d in degree]
    fig.add_trace(
        go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode="markers+text",
            marker=dict(
                size=node_sizes,
                color=COLORS["primary"],
                line=dict(width=1.5, color="white"),
            ),
            text=labels,
            textposition="top center",
            textfont=dict(size=11),
            hovertemplate="<b>%{text}</b><br>Connections: %{customdata}<extra></extra>",
            customdata=degree,
            name="Assets",
        )
    )

    fig.update_layout(
        **_base_layout(
            title=f"Correlation Network (|corr| >= {threshold})",
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, scaleanchor="x"),
            width=650,
            height=650,
        )
    )
    return fig


@requires_extra("viz")
def plotly_sankey_flow(
    sectors: list[str],
    allocations_before: list[float],
    allocations_after: list[float],
) -> go.Figure:
    """Sankey diagram showing portfolio rebalancing flows.

    Left side shows *before* weights, right side shows *after* weights.
    Flows connect each sector's allocation change.

    Parameters:
        sectors: Sector / asset names.
        allocations_before: Weights before rebalancing.
        allocations_after: Weights after rebalancing.

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    import plotly.graph_objects as go

    n = len(sectors)

    # Nodes: left side (before) + right side (after)
    node_labels = [f"{s} (before)" for s in sectors] + [
        f"{s} (after)" for s in sectors
    ]
    node_colors = (
        [COLORS["primary"]] * n + [COLORS["secondary"]] * n
    )

    # Flows: each sector connects its before-node to its after-node
    sources = list(range(n))
    targets = list(range(n, 2 * n))
    # Flow value is the minimum of before/after (representing transferred weight)
    values = [min(b, a) for b, a in zip(allocations_before, allocations_after, strict=False)]

    fig = go.Figure(
        data=go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                label=node_labels,
                color=node_colors,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=[
                    "rgba(31, 119, 180, 0.3)" if a >= b
                    else "rgba(214, 39, 40, 0.3)"
                    for b, a in zip(allocations_before, allocations_after, strict=False)
                ],
            ),
        )
    )

    fig.update_layout(
        title="Portfolio Rebalancing Flow",
        template=_PLOTLY_TEMPLATE,
        font=dict(size=11),
        height=500,
    )
    return fig


@requires_extra("viz")
def plotly_treemap(
    weights: list[float],
    sectors: list[str],
    returns: list[float],
) -> go.Figure:
    """Portfolio treemap with tiles sized by weight and colored by return.

    Parameters:
        weights: Portfolio weight per asset/sector.
        sectors: Sector or asset labels.
        returns: Period return per asset/sector (used for color).

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    import plotly.graph_objects as go

    hover_text = [
        f"<b>{s}</b><br>Weight: {w:.1%}<br>Return: {r:.2%}"
        for s, w, r in zip(sectors, weights, returns, strict=False)
    ]

    max_abs = max(abs(r) for r in returns) or 0.01

    fig = go.Figure(
        go.Treemap(
            labels=sectors,
            parents=[""] * len(sectors),
            values=weights,
            marker=dict(
                colors=returns,
                colorscale="RdYlGn",
                cmid=0,
                cmin=-max_abs,
                cmax=max_abs,
                colorbar=dict(title="Return"),
                line=dict(width=2, color="white"),
            ),
            text=hover_text,
            hoverinfo="text",
            textinfo="label+percent parent",
        )
    )

    fig.update_layout(
        title="Portfolio Treemap",
        template=_PLOTLY_TEMPLATE,
        margin=dict(l=10, r=10, t=50, b=10),
        height=550,
    )
    return fig


def _to_fill_color(color: str, alpha: float = 0.15) -> str:
    """Convert a color string to an rgba fill color."""
    if color.startswith("rgb("):
        return color.replace("rgb(", "rgba(").replace(")", f", {alpha})")
    if color.startswith("#") and len(color) == 7:
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    return f"rgba(100, 100, 100, {alpha})"


@requires_extra("viz")
def plotly_radar(
    metrics_dict: dict[str, dict[str, float]],
) -> go.Figure:
    """Radar / spider chart comparing portfolio metrics.

    Parameters:
        metrics_dict: Mapping from portfolio name to a dict of
            ``{metric_name: value}``.  All portfolios must share the same
            metric names.

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    import plotly.graph_objects as go

    palette = [
        COLORS["primary"], COLORS["secondary"], COLORS["positive"],
        COLORS["accent"], COLORS["info"], COLORS["warning"],
    ]

    # All metric names (from first portfolio)
    first_key = next(iter(metrics_dict))
    categories = list(metrics_dict[first_key].keys())

    fig = go.Figure()
    for i, (name, metrics) in enumerate(metrics_dict.items()):
        values = [metrics[c] for c in categories]
        # Close the polygon
        values_closed = values + [values[0]]
        cats_closed = categories + [categories[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=cats_closed,
                fill="toself",
                fillcolor=_to_fill_color(palette[i % len(palette)]),
                line=dict(color=palette[i % len(palette)], width=2),
                name=name,
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, showticklabels=True),
        ),
        title="Portfolio Metrics Comparison",
        template=_PLOTLY_TEMPLATE,
        showlegend=True,
        height=550,
        width=600,
    )
    return fig
