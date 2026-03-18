"""Rich individual Plotly charts for financial analysis.

Standalone chart functions for multi-asset comparison, volatility surfaces,
regime overlays, distribution analysis, correlation networks, and backtest
tearsheets.  Each returns a ``plotly.graph_objects.Figure`` styled with the
``plotly_dark`` template.

Example:
    >>> from wraquant.viz.charts import plot_distribution_analysis
    >>> fig = plot_distribution_analysis(returns)
    >>> fig.show()
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
    "plot_multi_asset",
    "plot_vol_surface",
    "plot_regime_overlay",
    "plot_distribution_analysis",
    "plot_correlation_network",
    "plot_backtest_tearsheet",
]

# ---------------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------------

_TEMPLATE = "plotly_dark"

_PALETTE = [
    COLORS["primary"],
    COLORS["secondary"],
    COLORS["positive"],
    COLORS["negative"],
    COLORS["accent"],
    COLORS["info"],
    COLORS["warning"],
    "#e377c2",
    "#8c564b",
    "#bcbd22",
]


def _dark_layout(**overrides: object) -> dict:
    """Return a base Plotly dark-theme layout dict."""
    defaults: dict = dict(
        template=_TEMPLATE,
        font=dict(family="sans-serif", size=11, color="#e0e0e0"),
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#111111",
        hovermode="x unified",
        margin=dict(l=60, r=30, t=60, b=50),
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# 5. Multi-asset comparison
# ---------------------------------------------------------------------------


@requires_extra("viz")
def plot_multi_asset(
    returns_df: pd.DataFrame,
    rolling_corr_window: int = 63,
    title: str = "Multi-Asset Comparison",
) -> go.Figure:
    """Multi-asset comparison chart with normalized performance and correlations.

    Creates a three-panel figure showing rebased (base-100) performance
    lines, a correlation matrix heatmap, and rolling pairwise correlation
    time series.

    Parameters:
        returns_df: DataFrame of simple daily returns with one column per
            asset.
        rolling_corr_window: Window in trading days for rolling correlation
            (default 63).
        title: Chart title.

    Returns:
        A ``plotly.graph_objects.Figure`` with three subplots.

    Example:
        >>> import pandas as pd, numpy as np
        >>> dates = pd.bdate_range("2020-01-01", periods=504)
        >>> df = pd.DataFrame(np.random.normal(0.0003, 0.015, (504, 3)),
        ...                   index=dates, columns=["SPY", "AGG", "GLD"])
        >>> fig = plot_multi_asset(df)
        >>> fig.show()
    """
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    assets = list(returns_df.columns)
    n_assets = len(assets)

    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.55, 0.45],
        column_widths=[0.55, 0.45],
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
        subplot_titles=(
            "Normalized Performance (Base 100)",
            "Correlation Matrix",
            "Rolling Pairwise Correlation",
            "",
        ),
        specs=[
            [{"type": "xy"}, {"type": "heatmap"}],
            [{"type": "xy", "colspan": 2}, None],
        ],
    )

    # -- Panel 1: Rebased performance --
    for i, asset in enumerate(assets):
        cum = (1 + returns_df[asset]).cumprod() * 100
        fig.add_trace(
            go.Scatter(
                x=cum.index, y=cum.values,
                mode="lines",
                name=asset,
                line=dict(color=_PALETTE[i % len(_PALETTE)], width=1.8),
                hovertemplate=f"<b>{asset}</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: %{{y:.1f}}<extra></extra>",
            ),
            row=1, col=1,
        )
    fig.update_yaxes(title_text="Value (Base 100)", row=1, col=1)

    # -- Panel 2: Correlation matrix --
    corr = returns_df.corr()
    corr_text = [
        [f"{corr.iloc[i, j]:.2f}" for j in range(n_assets)]
        for i in range(n_assets)
    ]
    fig.add_trace(
        go.Heatmap(
            z=corr.values,
            x=assets, y=assets,
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            text=corr_text,
            texttemplate="%{text}",
            hoverinfo="text",
            colorbar=dict(title="Corr", len=0.4, y=0.78, x=1.02),
        ),
        row=1, col=2,
    )
    fig.update_yaxes(autorange="reversed", row=1, col=2)

    # -- Panel 3: Rolling pairwise correlation --
    pair_idx = 0
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            roll_corr = returns_df[assets[i]].rolling(rolling_corr_window).corr(
                returns_df[assets[j]]
            )
            pair_name = f"{assets[i]}/{assets[j]}"
            fig.add_trace(
                go.Scatter(
                    x=roll_corr.index, y=roll_corr.values,
                    mode="lines",
                    name=pair_name,
                    line=dict(
                        color=_PALETTE[pair_idx % len(_PALETTE)],
                        width=1.3,
                    ),
                    showlegend=True,
                ),
                row=2, col=1,
            )
            pair_idx += 1
    fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=0.5, row=2, col=1)
    fig.update_yaxes(
        title_text="Correlation", range=[-1, 1],
        row=2, col=1,
    )

    fig.update_layout(
        **_dark_layout(
            title=dict(text=title, font=dict(size=15)),
            height=850,
            width=1100,
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
        )
    )

    return fig


# ---------------------------------------------------------------------------
# 6. Volatility surface
# ---------------------------------------------------------------------------


@requires_extra("viz")
def plot_vol_surface(
    strikes: npt.NDArray[np.floating],
    maturities: npt.NDArray[np.floating],
    implied_vols: npt.NDArray[np.floating],
    title: str = "Implied Volatility Surface",
) -> go.Figure:
    """3-D implied volatility surface with interactive rotation.

    Renders a Plotly Surface plot with strike on the x-axis, maturity on
    the y-axis, and implied volatility on the z-axis.  A color gradient
    encodes the volatility level.

    Parameters:
        strikes: 1-D array of strike prices.
        maturities: 1-D array of maturities (years or days to expiry).
        implied_vols: 2-D array of shape ``(len(maturities), len(strikes))``
            containing implied volatilities.
        title: Chart title.

    Returns:
        A ``plotly.graph_objects.Figure`` with a 3-D surface.

    Example:
        >>> import numpy as np
        >>> strikes = np.linspace(80, 120, 25)
        >>> mats = np.array([0.1, 0.25, 0.5, 1.0, 2.0])
        >>> iv = 0.2 + 0.05 * np.random.randn(len(mats), len(strikes))
        >>> fig = plot_vol_surface(strikes, mats, iv)
        >>> fig.show()
    """
    import numpy as np
    import plotly.graph_objects as go

    strike_grid, mat_grid = np.meshgrid(strikes, maturities)

    fig = go.Figure(
        data=go.Surface(
            x=strike_grid,
            y=mat_grid,
            z=implied_vols,
            colorscale="Plasma",
            colorbar=dict(title="IV", tickformat=".1%"),
            hovertemplate=(
                "Strike: %{x:.1f}<br>"
                "Maturity: %{y:.2f}<br>"
                "IV: %{z:.2%}<br>"
                "<extra></extra>"
            ),
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True),
            ),
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#e0e0e0")),
        template=_TEMPLATE,
        paper_bgcolor="#111111",
        scene=dict(
            xaxis=dict(title="Strike", backgroundcolor="#1e1e1e", gridcolor="#333"),
            yaxis=dict(title="Maturity", backgroundcolor="#1e1e1e", gridcolor="#333"),
            zaxis=dict(title="Implied Vol", backgroundcolor="#1e1e1e", gridcolor="#333",
                       tickformat=".0%"),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
        ),
        width=850,
        height=700,
    )

    return fig


# ---------------------------------------------------------------------------
# 7. Regime overlay
# ---------------------------------------------------------------------------


@requires_extra("viz")
def plot_regime_overlay(
    prices: pd.Series,
    regime_probs: pd.DataFrame,
    title: str = "Price with Regime Probability Overlay",
) -> go.Figure:
    """Price chart with smooth regime probability shown as background shading.

    Unlike a discrete regime label overlay, this function uses the
    continuous probability of each regime to determine background
    opacity, producing a gradient-like shading effect.

    Parameters:
        prices: Price or level time series.
        regime_probs: DataFrame where each column is the probability of
            a given regime at each time step.  Columns are used as
            regime names.  Probabilities should sum to 1 per row.
        title: Chart title.

    Returns:
        A ``plotly.graph_objects.Figure``.

    Example:
        >>> import pandas as pd, numpy as np
        >>> dates = pd.bdate_range("2020-01-01", periods=252)
        >>> prices = pd.Series(100 * np.exp(np.cumsum(
        ...     np.random.normal(0.0005, 0.02, 252))), index=dates, name="SPY")
        >>> probs = pd.DataFrame({"Bull": 0.7, "Bear": 0.3},
        ...                      index=dates)
        >>> fig = plot_regime_overlay(prices, probs)
        >>> fig.show()
    """
    import plotly.graph_objects as go

    _REGIME_SOLID = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#bcbd22",
    ]

    fig = go.Figure()

    # Price line
    fig.add_trace(
        go.Scatter(
            x=prices.index, y=prices.values,
            mode="lines",
            name=prices.name or "Price",
            line=dict(color=COLORS["primary"], width=2),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
        )
    )

    # Stacked area of probabilities on secondary y-axis (translucent)
    regimes = list(regime_probs.columns)
    for i, regime in enumerate(regimes):
        color = _REGIME_SOLID[i % len(_REGIME_SOLID)]
        # Convert hex to rgba with low opacity
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fill_rgba = f"rgba({r}, {g}, {b}, 0.25)"
        line_rgba = f"rgba({r}, {g}, {b}, 0.5)"

        fig.add_trace(
            go.Scatter(
                x=regime_probs.index,
                y=regime_probs[regime].values,
                mode="lines",
                name=f"P({regime})",
                line=dict(color=line_rgba, width=1),
                fill="tozeroy",
                fillcolor=fill_rgba,
                yaxis="y2",
                stackgroup="probs",
            )
        )

    # Legend markers for each regime
    for i, regime in enumerate(regimes):
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=10, color=_REGIME_SOLID[i % len(_REGIME_SOLID)]),
                name=f"Regime: {regime}",
                showlegend=True,
            )
        )

    fig.update_layout(
        **_dark_layout(
            title=dict(text=title, font=dict(size=15)),
            height=550,
            width=1000,
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
            yaxis=dict(title="Price", side="left"),
            yaxis2=dict(
                title="Probability",
                side="right",
                overlaying="y",
                range=[0, 1],
                tickformat=".0%",
                showgrid=False,
            ),
        )
    )

    return fig


# ---------------------------------------------------------------------------
# 8. Distribution analysis
# ---------------------------------------------------------------------------


@requires_extra("viz")
def plot_distribution_analysis(
    returns: pd.Series,
    bins: int = 60,
    title: str = "Return Distribution Analysis",
) -> go.Figure:
    """Rich distribution analysis with histogram, KDE, normal overlay, QQ plot, and stats.

    Creates a two-panel figure.  The main panel shows a histogram with
    KDE and a fitted normal distribution overlay.  A secondary inset
    displays a QQ plot.  Key statistics (mean, std, skew, kurtosis,
    Jarque-Bera test) are annotated.

    Parameters:
        returns: Simple daily return series.
        bins: Number of histogram bins.
        title: Chart title.

    Returns:
        A ``plotly.graph_objects.Figure``.

    Example:
        >>> import pandas as pd, numpy as np
        >>> rets = pd.Series(np.random.normal(0.0004, 0.015, 504),
        ...                  name="Strategy")
        >>> fig = plot_distribution_analysis(rets)
        >>> fig.show()
    """
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.stats import gaussian_kde, jarque_bera, kurtosis, norm, probplot, skew

    clean = returns.dropna().values

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        horizontal_spacing=0.08,
        subplot_titles=("Distribution (Histogram + KDE + Normal)", "Q-Q Plot"),
    )

    # -- Left panel: Histogram + KDE + Normal --
    fig.add_trace(
        go.Histogram(
            x=clean,
            nbinsx=bins,
            histnorm="probability density",
            name="Returns",
            marker_color=COLORS["primary"],
            opacity=0.65,
        ),
        row=1, col=1,
    )

    x_grid = np.linspace(float(clean.min()), float(clean.max()), 300)

    # KDE
    kde = gaussian_kde(clean)
    fig.add_trace(
        go.Scatter(
            x=x_grid, y=kde(x_grid),
            mode="lines",
            name="KDE",
            line=dict(color=COLORS["accent"], width=2),
        ),
        row=1, col=1,
    )

    # Normal overlay
    mu, sigma = float(clean.mean()), float(clean.std())
    fig.add_trace(
        go.Scatter(
            x=x_grid, y=norm.pdf(x_grid, mu, sigma),
            mode="lines",
            name="Normal Fit",
            line=dict(color=COLORS["negative"], width=1.5, dash="dash"),
        ),
        row=1, col=1,
    )

    fig.update_xaxes(title_text="Return", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=1)

    # -- Right panel: QQ plot --
    (osm, osr), (slope, intercept, r_val) = probplot(clean, dist="norm")
    qq_line_x = np.array([osm.min(), osm.max()])
    qq_line_y = slope * qq_line_x + intercept

    fig.add_trace(
        go.Scatter(
            x=osm, y=osr,
            mode="markers",
            name="Q-Q",
            marker=dict(color=COLORS["primary"], size=3, opacity=0.6),
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=qq_line_x, y=qq_line_y,
            mode="lines",
            name="45-degree",
            line=dict(color=COLORS["negative"], width=1.5, dash="dash"),
        ),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
    fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)

    # -- Statistics annotation --
    sk = float(skew(clean))
    ku = float(kurtosis(clean))
    jb_stat, jb_pval = jarque_bera(clean)
    stats_text = (
        f"<b>Statistics</b><br>"
        f"Mean: {mu:.6f}<br>"
        f"Std: {sigma:.6f}<br>"
        f"Skew: {sk:.3f}<br>"
        f"Kurtosis: {ku:.3f}<br>"
        f"JB stat: {jb_stat:.2f}<br>"
        f"JB p-val: {jb_pval:.4f}"
    )
    fig.add_annotation(
        x=0.63, y=0.95, xref="paper", yref="paper",
        text=stats_text,
        showarrow=False,
        font=dict(size=10, color="#e0e0e0"),
        bgcolor="rgba(30, 30, 30, 0.85)",
        bordercolor=COLORS["neutral"],
        borderwidth=1,
        align="left",
    )

    fig.update_layout(
        **_dark_layout(
            title=dict(text=title, font=dict(size=15)),
            height=480,
            width=1050,
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
            bargap=0.02,
        )
    )

    return fig


# ---------------------------------------------------------------------------
# 9. Correlation network
# ---------------------------------------------------------------------------


@requires_extra("viz")
def plot_correlation_network(
    returns_df: pd.DataFrame,
    threshold: float = 0.3,
    show_mst: bool = False,
    title: str = "Asset Correlation Network",
) -> go.Figure:
    """Interactive correlation network graph.

    Assets are drawn as nodes in a circular layout.  Edges connect pairs
    whose absolute correlation exceeds *threshold*.  Edge thickness
    scales with correlation strength, and edges are colored green
    (positive) or red (negative).

    Optionally overlays the minimum spanning tree (MST) of the
    correlation-derived distance matrix.

    Parameters:
        returns_df: DataFrame of daily returns (columns = assets).
        threshold: Minimum absolute correlation to draw an edge
            (default 0.3).
        show_mst: If *True*, overlay the MST as thicker edges.
        title: Chart title.

    Returns:
        A ``plotly.graph_objects.Figure``.

    Example:
        >>> import pandas as pd, numpy as np
        >>> dates = pd.bdate_range("2020-01-01", periods=252)
        >>> df = pd.DataFrame(np.random.normal(0.0003, 0.015, (252, 6)),
        ...                   index=dates,
        ...                   columns=["A", "B", "C", "D", "E", "F"])
        >>> fig = plot_correlation_network(df, threshold=0.2, show_mst=True)
        >>> fig.show()
    """
    import math

    import numpy as np
    import plotly.graph_objects as go

    corr = returns_df.corr()
    assets = list(corr.columns)
    n = len(assets)

    # Circular layout
    angles = [2 * math.pi * i / n for i in range(n)]
    x_nodes = [math.cos(a) for a in angles]
    y_nodes = [math.sin(a) for a in angles]

    fig = go.Figure()

    # -- MST overlay --
    mst_edges: set[tuple[int, int]] = set()
    if show_mst:
        from scipy.sparse.csgraph import minimum_spanning_tree

        dist = np.sqrt(2 * (1 - corr.values))
        np.fill_diagonal(dist, 0)
        mst_matrix = minimum_spanning_tree(dist).toarray()
        for i in range(n):
            for j in range(n):
                if mst_matrix[i, j] > 0 or mst_matrix[j, i] > 0:
                    mst_edges.add((min(i, j), max(i, j)))

        # Draw MST edges first (thicker, behind)
        for i_node, j_node in mst_edges:
            c = corr.iloc[i_node, j_node]
            edge_color = COLORS["positive"] if c >= 0 else COLORS["negative"]
            fig.add_trace(
                go.Scatter(
                    x=[x_nodes[i_node], x_nodes[j_node], None],
                    y=[y_nodes[i_node], y_nodes[j_node], None],
                    mode="lines",
                    line=dict(width=3.5, color=edge_color),
                    hoverinfo="none",
                    showlegend=False,
                    opacity=0.9,
                )
            )

    # -- Correlation edges --
    degree = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            c = corr.iloc[i, j]
            if abs(c) >= threshold:
                edge_color = COLORS["positive"] if c >= 0 else COLORS["negative"]
                width = 0.5 + 2.5 * abs(c)
                is_mst = (i, j) in mst_edges
                fig.add_trace(
                    go.Scatter(
                        x=[x_nodes[i], x_nodes[j], None],
                        y=[y_nodes[i], y_nodes[j], None],
                        mode="lines",
                        line=dict(
                            width=width if not is_mst else width,
                            color=edge_color,
                        ),
                        hoverinfo="none",
                        showlegend=False,
                        opacity=0.5 if is_mst else 0.6,
                    )
                )
                degree[i] += 1
                degree[j] += 1

    # -- Nodes --
    node_sizes = [max(15, 10 + d * 5) for d in degree]
    fig.add_trace(
        go.Scatter(
            x=x_nodes, y=y_nodes,
            mode="markers+text",
            marker=dict(
                size=node_sizes,
                color=COLORS["primary"],
                line=dict(width=2, color="white"),
            ),
            text=assets,
            textposition="top center",
            textfont=dict(size=12, color="#e0e0e0"),
            hovertemplate="<b>%{text}</b><br>Connections: %{customdata}<extra></extra>",
            customdata=degree,
            name="Assets",
        )
    )

    # Legend entries
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode="lines",
                   line=dict(color=COLORS["positive"], width=2),
                   name="Positive Corr")
    )
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode="lines",
                   line=dict(color=COLORS["negative"], width=2),
                   name="Negative Corr")
    )
    if show_mst:
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode="lines",
                       line=dict(color="white", width=3.5),
                       name="MST")
        )

    fig.update_layout(
        **_dark_layout(
            title=dict(text=title, font=dict(size=15)),
            showlegend=True,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, scaleanchor="x"),
            width=700,
            height=700,
            legend=dict(x=0.85, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
        )
    )

    return fig


# ---------------------------------------------------------------------------
# 10. Backtest tearsheet
# ---------------------------------------------------------------------------


@requires_extra("viz")
def plot_backtest_tearsheet(
    returns: pd.Series,
    trades: pd.DataFrame | None = None,
    benchmark: pd.Series | None = None,
    title: str = "Backtest Tearsheet",
) -> go.Figure:
    """Full backtest analysis tearsheet figure.

    Produces a multi-panel figure with an equity curve and drawdown
    bands, a monthly returns calendar heatmap, a trade analysis scatter
    (if trades are provided), rolling metrics, and a summary statistics
    table.

    Parameters:
        returns: Simple daily return series with a ``DatetimeIndex``.
        trades: Optional DataFrame of trade results with at least a
            ``pnl`` column (profit/loss per trade).  May also include
            ``entry_date`` and ``exit_date`` columns.
        benchmark: Optional benchmark return series for comparison.
        title: Tearsheet title.

    Returns:
        A ``plotly.graph_objects.Figure``.

    Example:
        >>> import pandas as pd, numpy as np
        >>> dates = pd.bdate_range("2020-01-01", periods=504)
        >>> rets = pd.Series(np.random.normal(0.0004, 0.015, 504),
        ...                  index=dates, name="Strategy")
        >>> fig = plot_backtest_tearsheet(rets)
        >>> fig.show()
    """
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    has_trades = trades is not None and len(trades) > 0

    n_rows = 3 + (1 if has_trades else 0)
    row_heights = [0.35, 0.25, 0.30]
    titles_list = [
        "Equity Curve with Drawdown",
        "Monthly Returns",
        "Rolling Metrics (63d)",
    ]
    specs = [
        [{"type": "xy", "colspan": 2}, None],
        [{"type": "heatmap", "colspan": 2}, None],
        [{"type": "xy", "colspan": 2}, None],
    ]

    if has_trades:
        row_heights.insert(3, 0.25)
        titles_list.insert(3, "Trade Analysis")
        specs.insert(3, [{"type": "xy", "colspan": 2}, None])

    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    fig = make_subplots(
        rows=n_rows, cols=2,
        row_heights=row_heights,
        vertical_spacing=0.07,
        horizontal_spacing=0.08,
        subplot_titles=titles_list,
        specs=specs,
    )

    # -- Panel 1: Equity curve with drawdown band --
    cum = (1 + returns).cumprod() - 1
    wealth = (1 + returns).cumprod()
    running_max = wealth.cummax()
    drawdown = (wealth - running_max) / running_max

    fig.add_trace(
        go.Scatter(
            x=cum.index, y=cum.values,
            mode="lines",
            name=returns.name or "Strategy",
            line=dict(color=COLORS["primary"], width=2),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Return: %{y:.2%}<extra></extra>",
        ),
        row=1, col=1,
    )

    if benchmark is not None:
        cum_bench = (1 + benchmark).cumprod() - 1
        fig.add_trace(
            go.Scatter(
                x=cum_bench.index, y=cum_bench.values,
                mode="lines",
                name=benchmark.name or "Benchmark",
                line=dict(color=COLORS["benchmark"], width=2, dash="dash"),
            ),
            row=1, col=1,
        )

    # Drawdown band (secondary y-axis effect via negative fill)
    fig.add_trace(
        go.Scatter(
            x=drawdown.index, y=drawdown.values,
            fill="tozeroy",
            mode="lines",
            name="Drawdown",
            line=dict(color=COLORS["drawdown"], width=0.8),
            fillcolor="rgba(214, 39, 40, 0.25)",
            showlegend=True,
        ),
        row=1, col=1,
    )
    fig.update_yaxes(tickformat=".0%", row=1, col=1)

    # -- Panel 2: Monthly returns heatmap --
    monthly = returns.groupby(
        [returns.index.year, returns.index.month]
    ).apply(lambda x: (1 + x).prod() - 1)
    monthly.index.names = ["year", "month"]
    table = monthly.unstack(level="month")
    month_labels = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    existing_cols = table.columns.tolist()
    col_labels = [month_labels[m - 1] for m in existing_cols]

    vmax = float(np.nanmax(np.abs(table.values))) if table.size > 0 else 0.01
    hm_text = [
        [
            f"{table.index[i]} {col_labels[j]}: {table.iloc[i, j]:.2%}"
            if not np.isnan(table.iloc[i, j]) else ""
            for j in range(table.shape[1])
        ]
        for i in range(table.shape[0])
    ]
    fig.add_trace(
        go.Heatmap(
            z=table.values,
            x=col_labels,
            y=[str(y) for y in table.index],
            colorscale="RdYlGn",
            zmin=-vmax, zmax=vmax,
            text=hm_text,
            hoverinfo="text",
            colorbar=dict(
                title="Return", tickformat=".0%",
                len=0.2, y=0.65, x=1.02,
            ),
        ),
        row=2, col=1,
    )
    fig.update_yaxes(autorange="reversed", row=2, col=1)

    # -- Panel 3: Rolling metrics --
    rolling_window = 63
    roll_mean = returns.rolling(rolling_window).mean()
    roll_std = returns.rolling(rolling_window).std()
    roll_sharpe = (roll_mean / roll_std) * np.sqrt(252)
    roll_vol = roll_std * np.sqrt(252)

    fig.add_trace(
        go.Scatter(
            x=roll_sharpe.index, y=roll_sharpe.values,
            mode="lines", name="Rolling Sharpe",
            line=dict(color=COLORS["primary"], width=1.5),
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=roll_vol.index, y=roll_vol.values,
            mode="lines", name="Rolling Vol",
            line=dict(color=COLORS["secondary"], width=1.5),
        ),
        row=3, col=1,
    )
    fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=0.5, row=3, col=1)

    current_row = 4

    # -- Trade analysis (optional) --
    if has_trades and trades is not None:
        pnl = trades["pnl"].values
        trade_idx = np.arange(len(pnl))
        trade_colors = [
            COLORS["positive"] if p >= 0 else COLORS["negative"]
            for p in pnl
        ]
        fig.add_trace(
            go.Scatter(
                x=trade_idx, y=pnl,
                mode="markers",
                name="Trade PnL",
                marker=dict(color=trade_colors, size=6, opacity=0.7),
                hovertemplate="Trade #%{x}<br>PnL: %{y:.4f}<extra></extra>",
            ),
            row=current_row, col=1,
        )
        fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=0.5,
                      row=current_row, col=1)
        fig.update_xaxes(title_text="Trade #", row=current_row, col=1)
        fig.update_yaxes(title_text="PnL", row=current_row, col=1)
        current_row += 1

    # -- Summary statistics annotation --
    total_ret = float(cum.iloc[-1])
    ann_ret = float((1 + total_ret) ** (252 / len(returns)) - 1)
    ann_vol = float(returns.std() * np.sqrt(252))
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0.0
    max_dd = float(drawdown.min())
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0
    win_rate = float((returns > 0).mean())
    best_day = float(returns.max())
    worst_day = float(returns.min())

    stats_lines = [
        "<b>Summary Statistics</b><br>",
        f"Total Return:    {total_ret:.2%}",
        f"Ann. Return:     {ann_ret:.2%}",
        f"Ann. Volatility: {ann_vol:.2%}",
        f"Sharpe Ratio:    {sharpe:.2f}",
        f"Max Drawdown:    {max_dd:.2%}",
        f"Calmar Ratio:    {calmar:.2f}",
        f"Win Rate:        {win_rate:.1%}",
        f"Best Day:        {best_day:.2%}",
        f"Worst Day:       {worst_day:.2%}",
    ]

    if has_trades and trades is not None:
        n_trades = len(trades)
        wins = int((trades["pnl"] > 0).sum())
        avg_win = float(trades.loc[trades["pnl"] > 0, "pnl"].mean()) if wins > 0 else 0
        avg_loss = float(trades.loc[trades["pnl"] <= 0, "pnl"].mean()) if n_trades - wins > 0 else 0
        stats_lines.extend([
            f"Trades:          {n_trades}",
            f"Trade Win Rate:  {wins / n_trades:.1%}" if n_trades > 0 else "Trade Win Rate:  N/A",
            f"Avg Win:         {avg_win:.4f}",
            f"Avg Loss:        {avg_loss:.4f}",
        ])

    fig.add_annotation(
        x=0.99, y=0.01, xref="paper", yref="paper",
        text="<br>".join(stats_lines),
        showarrow=False,
        font=dict(size=10, color="#e0e0e0", family="monospace"),
        bgcolor="rgba(30, 30, 30, 0.90)",
        bordercolor=COLORS["neutral"],
        borderwidth=1,
        align="left",
        xanchor="right",
        yanchor="bottom",
    )

    fig.update_layout(
        **_dark_layout(
            title=dict(text=title, font=dict(size=16)),
            height=1000 + (200 if has_trades else 0),
            width=1150,
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
        )
    )

    return fig
