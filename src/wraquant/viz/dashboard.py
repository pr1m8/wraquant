"""Multi-panel interactive dashboards for financial analysis.

Comprehensive Plotly dashboards that combine multiple chart types into
single rich figures for portfolio analysis, regime detection, risk
monitoring, and technical trading.

All functions return ``plotly.graph_objects.Figure`` objects styled with
the ``plotly_dark`` template.  Users can call ``.show()`` or save to HTML.

Example:
    >>> from wraquant.viz.dashboard import portfolio_dashboard
    >>> fig = portfolio_dashboard(returns, benchmark=benchmark)
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
    "portfolio_dashboard",
    "regime_dashboard",
    "risk_dashboard",
    "technical_dashboard",
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
]

_REGIME_COLORS_TRANSLUCENT = [
    "rgba(31, 119, 180, 0.20)",
    "rgba(255, 127, 14, 0.20)",
    "rgba(44, 160, 44, 0.20)",
    "rgba(214, 39, 40, 0.20)",
    "rgba(148, 103, 189, 0.20)",
    "rgba(140, 86, 75, 0.20)",
    "rgba(227, 119, 194, 0.20)",
    "rgba(188, 189, 34, 0.20)",
]

_REGIME_COLORS_SOLID = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#bcbd22",
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
# 1. Portfolio Dashboard
# ---------------------------------------------------------------------------


@requires_extra("viz")
def portfolio_dashboard(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    rolling_window: int = 63,
    title: str = "Portfolio Performance Dashboard",
) -> go.Figure:
    """Create a comprehensive multi-panel portfolio performance dashboard.

    Produces a six-panel figure with cumulative returns, drawdowns,
    rolling risk-adjusted ratios, return distributions, a monthly
    returns heatmap, and an annotation box of key performance metrics.

    Parameters:
        returns: Simple (non-cumulative) daily return series with a
            ``DatetimeIndex``.
        benchmark: Optional benchmark return series for comparison.
            Must share the same index or a compatible date range.
        rolling_window: Window in trading days for rolling Sharpe and
            Sortino calculations.  Defaults to 63 (~1 quarter).
        title: Dashboard title displayed at the top.

    Returns:
        A ``plotly.graph_objects.Figure`` with six subplots.

    Example:
        >>> import pandas as pd, numpy as np
        >>> dates = pd.bdate_range("2020-01-01", periods=504)
        >>> rets = pd.Series(np.random.normal(0.0004, 0.015, 504),
        ...                  index=dates, name="Strategy")
        >>> fig = portfolio_dashboard(rets)
        >>> fig.show()
    """
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # ---- Derived series ----
    cum = (1 + returns).cumprod() - 1
    wealth = (1 + returns).cumprod()
    running_max = wealth.cummax()
    drawdown = (wealth - running_max) / running_max

    roll_mean = returns.rolling(rolling_window).mean()
    roll_std = returns.rolling(rolling_window).std()
    roll_sharpe = (roll_mean / roll_std) * np.sqrt(252)
    roll_downside = returns.clip(upper=0).rolling(rolling_window).std()
    roll_sortino = (roll_mean / roll_downside) * np.sqrt(252)

    # ---- Monthly returns table ----
    monthly = returns.groupby(
        [returns.index.year, returns.index.month]
    ).apply(lambda x: (1 + x).prod() - 1)
    monthly.index.names = ["year", "month"]
    table = monthly.unstack(level="month")
    month_labels = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    # Ensure columns cover only existing months
    existing_cols = table.columns.tolist()
    col_labels = [month_labels[m - 1] for m in existing_cols]

    # ---- Key metrics (canonical imports from risk.metrics) ----
    from wraquant.risk.metrics import max_drawdown as _max_drawdown
    from wraquant.risk.metrics import sharpe_ratio as _sharpe_ratio
    from wraquant.risk.metrics import sortino_ratio as _sortino_ratio

    total_ret = float(cum.iloc[-1])
    ann_ret = float((1 + total_ret) ** (252 / len(returns)) - 1)
    ann_vol = float(returns.std() * np.sqrt(252))
    sharpe = _sharpe_ratio(returns)
    max_dd = _max_drawdown(wealth)
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0
    skewness = float(returns.skew())
    kurt = float(returns.kurtosis())

    # ---- Build figure ----
    fig = make_subplots(
        rows=3, cols=2,
        row_heights=[0.35, 0.30, 0.35],
        column_widths=[0.55, 0.45],
        shared_xaxes=False,
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
        subplot_titles=(
            "Cumulative Returns",
            "Monthly Returns Heatmap",
            "Drawdown",
            "Return Distribution",
            f"Rolling Sharpe / Sortino ({rolling_window}d)",
            "",
        ),
        specs=[
            [{"type": "xy"}, {"type": "heatmap"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
    )

    # -- Panel 1: Cumulative Returns (row 1, col 1) --
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
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Return: %{y:.2%}<extra></extra>",
            ),
            row=1, col=1,
        )
    fig.update_yaxes(tickformat=".0%", row=1, col=1)

    # -- Panel 2: Monthly Heatmap (row 1, col 2) --
    vmax = float(np.nanmax(np.abs(table.values))) if table.size > 0 else 0.01
    hover_text = []
    for i in range(table.shape[0]):
        row_texts = []
        for j in range(table.shape[1]):
            val = table.iloc[i, j]
            if np.isnan(val):
                row_texts.append("")
            else:
                row_texts.append(
                    f"{table.index[i]} {col_labels[j]}<br>Return: {val:.2%}"
                )
        hover_text.append(row_texts)

    fig.add_trace(
        go.Heatmap(
            z=table.values,
            x=col_labels,
            y=[str(y) for y in table.index],
            colorscale="RdYlGn",
            zmin=-vmax, zmax=vmax,
            text=hover_text,
            hoverinfo="text",
            colorbar=dict(
                title="Return", tickformat=".0%",
                len=0.25, y=0.88, x=1.02,
            ),
            showscale=True,
        ),
        row=1, col=2,
    )
    fig.update_yaxes(autorange="reversed", row=1, col=2)

    # -- Panel 3: Drawdown (row 2, col 1) --
    fig.add_trace(
        go.Scatter(
            x=drawdown.index, y=drawdown.values,
            fill="tozeroy",
            mode="lines",
            name="Drawdown",
            line=dict(color=COLORS["drawdown"], width=1),
            fillcolor="rgba(214, 39, 40, 0.30)",
            showlegend=False,
            hovertemplate="Date: %{x|%Y-%m-%d}<br>DD: %{y:.2%}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.update_yaxes(tickformat=".0%", row=2, col=1)

    # -- Panel 4: Return Distribution (row 2, col 2) --
    fig.add_trace(
        go.Histogram(
            x=returns.dropna().values,
            nbinsx=50,
            histnorm="probability density",
            name="Returns",
            marker_color=COLORS["primary"],
            opacity=0.7,
            showlegend=False,
        ),
        row=2, col=2,
    )

    # Normal overlay
    clean = returns.dropna().values
    mu, sigma = float(np.mean(clean)), float(np.std(clean))
    x_grid = np.linspace(float(np.min(clean)), float(np.max(clean)), 200)
    from scipy.stats import norm

    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=norm.pdf(x_grid, mu, sigma),
            mode="lines",
            name="Normal",
            line=dict(color=COLORS["negative"], width=1.5, dash="dash"),
            showlegend=False,
        ),
        row=2, col=2,
    )

    # -- Panel 5: Rolling Sharpe / Sortino (row 3, col 1) --
    fig.add_trace(
        go.Scatter(
            x=roll_sharpe.index, y=roll_sharpe.values,
            mode="lines",
            name="Rolling Sharpe",
            line=dict(color=COLORS["primary"], width=1.5),
            showlegend=True,
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=roll_sortino.index, y=roll_sortino.values,
            mode="lines",
            name="Rolling Sortino",
            line=dict(color=COLORS["accent"], width=1.5),
            showlegend=True,
        ),
        row=3, col=1,
    )
    fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=0.6, row=3, col=1)

    # -- Key Metrics Annotation Box (row 3, col 2 area) --
    metrics_text = (
        f"<b>Key Performance Metrics</b><br>"
        f"<br>"
        f"Total Return: {total_ret:.2%}<br>"
        f"Ann. Return: {ann_ret:.2%}<br>"
        f"Ann. Volatility: {ann_vol:.2%}<br>"
        f"Sharpe Ratio: {sharpe:.2f}<br>"
        f"Max Drawdown: {max_dd:.2%}<br>"
        f"Calmar Ratio: {calmar:.2f}<br>"
        f"Skewness: {skewness:.2f}<br>"
        f"Kurtosis: {kurt:.2f}"
    )
    fig.add_annotation(
        x=0.97, y=0.02, xref="paper", yref="paper",
        text=metrics_text,
        showarrow=False,
        font=dict(size=11, color="#e0e0e0", family="monospace"),
        bgcolor="rgba(30, 30, 30, 0.90)",
        bordercolor=COLORS["neutral"],
        borderwidth=1,
        align="left",
        xanchor="right",
        yanchor="bottom",
    )

    # ---- Global layout ----
    fig.update_layout(
        **_dark_layout(
            title=dict(text=title, font=dict(size=16)),
            height=1000,
            width=1200,
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
        )
    )

    return fig


# ---------------------------------------------------------------------------
# 2. Regime Dashboard
# ---------------------------------------------------------------------------


@requires_extra("viz")
def regime_dashboard(
    returns: pd.Series,
    states: pd.Series,
    probabilities: pd.DataFrame | None = None,
    transition_matrix: npt.NDArray[np.floating] | None = None,
    title: str = "Regime Analysis Dashboard",
) -> go.Figure:
    """Create a multi-panel regime analysis dashboard.

    Combines price/returns with regime overlays, probability series,
    per-regime distribution comparisons, and an optional transition
    matrix heatmap.

    Parameters:
        returns: Simple daily return series with a ``DatetimeIndex``.
        states: Integer series (same index as *returns*) indicating the
            detected regime at each observation (e.g. 0, 1, 2).
        probabilities: Optional DataFrame where each column is the
            probability of being in a given regime at each time step.
            Columns should be regime labels or integers.
        transition_matrix: Optional square array of regime transition
            probabilities.  Shape ``(n_regimes, n_regimes)``.
        title: Dashboard title.

    Returns:
        A ``plotly.graph_objects.Figure`` with up to five panels.

    Example:
        >>> import pandas as pd, numpy as np
        >>> dates = pd.bdate_range("2020-01-01", periods=504)
        >>> rets = pd.Series(np.random.normal(0.0004, 0.015, 504), index=dates)
        >>> states = pd.Series(np.random.choice([0, 1], 504), index=dates)
        >>> fig = regime_dashboard(rets, states)
        >>> fig.show()
    """
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    unique_regimes = sorted(states.unique())
    n_regimes = len(unique_regimes)
    has_probs = probabilities is not None
    has_tm = transition_matrix is not None

    n_rows = 2 + (1 if has_probs else 0) + (1 if has_tm else 0)
    row_heights = [0.40, 0.30]
    subplot_titles_list = [
        "Price / Cumulative Returns with Regime Overlay",
        "Per-Regime Return Distributions",
    ]
    specs_list: list[list[dict]] = [
        [{"type": "xy", "colspan": 2}, None],
        [{"type": "xy", "colspan": 2}, None],
    ]

    if has_probs:
        row_heights.insert(1, 0.20)
        subplot_titles_list.insert(1, "Regime Probabilities")
        specs_list.insert(1, [{"type": "xy", "colspan": 2}, None])

    if has_tm:
        row_heights.append(0.25)
        subplot_titles_list.append("Transition Matrix")
        specs_list.append([{"type": "heatmap", "colspan": 2}, None])

    # Normalize row heights
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    fig = make_subplots(
        rows=len(row_heights), cols=2,
        row_heights=row_heights,
        vertical_spacing=0.07,
        horizontal_spacing=0.08,
        subplot_titles=subplot_titles_list,
        specs=specs_list,
    )

    # -- Panel 1: Cumulative returns with regime overlay --
    cum = (1 + returns).cumprod() - 1
    fig.add_trace(
        go.Scatter(
            x=cum.index, y=cum.values,
            mode="lines",
            name="Cumulative Return",
            line=dict(color=COLORS["primary"], width=2),
        ),
        row=1, col=1,
    )

    # Regime background shading
    prev = states.iloc[0]
    start = cum.index[0]
    for idx, regime in zip(states.index[1:], states.iloc[1:], strict=False):
        if regime != prev:
            fig.add_vrect(
                x0=start, x1=idx,
                fillcolor=_REGIME_COLORS_TRANSLUCENT[int(prev) % len(_REGIME_COLORS_TRANSLUCENT)],
                line_width=0, layer="below",
                row=1, col=1,
            )
            start = idx
            prev = regime
    fig.add_vrect(
        x0=start, x1=cum.index[-1],
        fillcolor=_REGIME_COLORS_TRANSLUCENT[int(prev) % len(_REGIME_COLORS_TRANSLUCENT)],
        line_width=0, layer="below",
        row=1, col=1,
    )

    # Legend markers for regimes
    for r in unique_regimes:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(
                    size=10,
                    color=_REGIME_COLORS_SOLID[int(r) % len(_REGIME_COLORS_SOLID)],
                ),
                name=f"Regime {r}",
                showlegend=True,
            ),
            row=1, col=1,
        )
    fig.update_yaxes(tickformat=".0%", row=1, col=1)

    current_row = 2

    # -- Panel 2 (optional): Regime probabilities --
    if has_probs and probabilities is not None:
        for i, col in enumerate(probabilities.columns):
            fig.add_trace(
                go.Scatter(
                    x=probabilities.index,
                    y=probabilities[col].values,
                    mode="lines",
                    name=f"P(Regime {col})",
                    line=dict(
                        color=_REGIME_COLORS_SOLID[i % len(_REGIME_COLORS_SOLID)],
                        width=1.5,
                    ),
                    stackgroup="probs",
                ),
                row=current_row, col=1,
            )
        fig.update_yaxes(range=[0, 1], tickformat=".0%", row=current_row, col=1)
        current_row += 1

    # -- Per-regime return distributions --
    for r in unique_regimes:
        regime_rets = returns[states == r].dropna().values
        if len(regime_rets) > 0:
            fig.add_trace(
                go.Histogram(
                    x=regime_rets,
                    nbinsx=40,
                    histnorm="probability density",
                    name=f"Regime {r}",
                    marker_color=_REGIME_COLORS_SOLID[int(r) % len(_REGIME_COLORS_SOLID)],
                    opacity=0.6,
                ),
                row=current_row, col=1,
            )
    fig.update_layout(barmode="overlay")
    current_row += 1

    # -- Regime statistics annotation --
    stats_lines = ["<b>Regime Statistics</b><br>"]
    stats_lines.append(
        f"{'Regime':<12}{'Ann.Ret':>10}{'Ann.Vol':>10}{'Sharpe':>8}{'Obs':>6}{'%Time':>7}"
    )
    for r in unique_regimes:
        regime_rets = returns[states == r].dropna()
        m = float(regime_rets.mean()) * 252
        v = float(regime_rets.std()) * np.sqrt(252)
        s = f"{m / v:.2f}" if v > 0 else "N/A"
        n_obs = len(regime_rets)
        pct = n_obs / len(returns)
        stats_lines.append(
            f"Regime {r:<5}{m:>9.2%}{v:>10.2%}{s:>8}{n_obs:>6}{pct:>6.1%}"
        )
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

    # -- Transition matrix heatmap (optional) --
    if has_tm and transition_matrix is not None:
        tm_labels = [f"Regime {r}" for r in unique_regimes]
        tm_text = [
            [f"{transition_matrix[i, j]:.2%}" for j in range(n_regimes)]
            for i in range(n_regimes)
        ]
        fig.add_trace(
            go.Heatmap(
                z=transition_matrix,
                x=tm_labels, y=tm_labels,
                colorscale="Blues",
                zmin=0, zmax=1,
                text=tm_text,
                texttemplate="%{text}",
                hoverinfo="text",
                colorbar=dict(
                    title="Prob", tickformat=".0%",
                    len=0.2, y=0.08, x=1.02,
                ),
            ),
            row=current_row, col=1,
        )
        fig.update_yaxes(autorange="reversed", row=current_row, col=1)

    fig.update_layout(
        **_dark_layout(
            title=dict(text=title, font=dict(size=16)),
            height=900 + (150 if has_probs else 0) + (200 if has_tm else 0),
            width=1100,
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
        )
    )

    return fig


# ---------------------------------------------------------------------------
# 3. Risk Dashboard
# ---------------------------------------------------------------------------


@requires_extra("viz")
def risk_dashboard(
    returns: pd.DataFrame,
    var_confidence: float = 0.95,
    rolling_window: int = 63,
    stress_scenarios: dict[str, float] | None = None,
    title: str = "Risk Monitoring Dashboard",
) -> go.Figure:
    """Create a multi-panel risk monitoring dashboard.

    Displays rolling VaR/CVaR with breach markers, an animated
    correlation heatmap snapshot, stress test scenario comparison bars,
    and a risk contribution breakdown.

    Parameters:
        returns: DataFrame of daily returns with one column per asset.
            If a single-column DataFrame or Series is passed, some
            panels (correlation, risk contribution) are simplified.
        var_confidence: Confidence level for VaR/CVaR (default 0.95).
        rolling_window: Window in trading days for rolling risk metrics
            (default 63).
        stress_scenarios: Optional dict mapping scenario names to
            portfolio-level return shocks for bar comparison.  Example:
            ``{"2008 Crisis": -0.38, "COVID Crash": -0.34}``.
        title: Dashboard title.

    Returns:
        A ``plotly.graph_objects.Figure``.

    Example:
        >>> import pandas as pd, numpy as np
        >>> dates = pd.bdate_range("2020-01-01", periods=504)
        >>> df = pd.DataFrame(np.random.normal(0.0003, 0.015, (504, 4)),
        ...                   index=dates, columns=["A", "B", "C", "D"])
        >>> fig = risk_dashboard(df)
        >>> fig.show()
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Normalise input
    if isinstance(returns, pd.Series):
        returns = returns.to_frame(name=returns.name or "Portfolio")

    n_assets = returns.shape[1]
    port_returns = returns.mean(axis=1)  # equal-weight portfolio proxy

    # Rolling VaR / CVaR
    alpha = 1 - var_confidence
    rolling_var = port_returns.rolling(rolling_window).quantile(alpha)
    rolling_cvar = port_returns.rolling(rolling_window).apply(
        lambda x: float(x[x <= np.quantile(x, alpha)].mean())
        if len(x[x <= np.quantile(x, alpha)]) > 0 else float(np.quantile(x, alpha)),
        raw=True,
    )

    has_stress = stress_scenarios is not None and len(stress_scenarios) > 0
    has_multi = n_assets > 1

    n_rows = 2 + (1 if has_stress else 0) + (1 if has_multi else 0)
    row_heights_list = [0.35, 0.30]
    titles_list = [
        f"Rolling VaR / CVaR ({var_confidence:.0%}, {rolling_window}d)",
        "Correlation Heatmap" if has_multi else "Return Distribution",
    ]
    specs = [
        [{"type": "xy", "colspan": 2}, None],
        [{"type": "heatmap" if has_multi else "xy", "colspan": 2}, None],
    ]

    if has_multi:
        row_heights_list.append(0.20)
        titles_list.append("Risk Contribution (Volatility)")
        specs.append([{"type": "xy", "colspan": 2}, None])

    if has_stress:
        row_heights_list.append(0.20)
        titles_list.append("Stress Test Scenarios")
        specs.append([{"type": "xy", "colspan": 2}, None])

    total = sum(row_heights_list)
    row_heights_list = [h / total for h in row_heights_list]

    fig = make_subplots(
        rows=n_rows, cols=2,
        row_heights=row_heights_list,
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        subplot_titles=titles_list,
        specs=specs,
    )

    # -- Panel 1: Rolling VaR / CVaR with breaches --
    fig.add_trace(
        go.Scatter(
            x=port_returns.index, y=port_returns.values,
            mode="lines",
            name="Portfolio Return",
            line=dict(color=COLORS["primary"], width=0.8),
            opacity=0.6,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=rolling_var.index, y=rolling_var.values,
            mode="lines",
            name=f"VaR ({var_confidence:.0%})",
            line=dict(color=COLORS["secondary"], width=1.5, dash="dash"),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=rolling_cvar.index, y=rolling_cvar.values,
            mode="lines",
            name=f"CVaR ({var_confidence:.0%})",
            line=dict(color=COLORS["negative"], width=1.5, dash="dot"),
        ),
        row=1, col=1,
    )

    # Breach markers
    breaches = port_returns[port_returns < rolling_var].dropna()
    if not breaches.empty:
        fig.add_trace(
            go.Scatter(
                x=breaches.index, y=breaches.values,
                mode="markers",
                name=f"Breaches ({len(breaches)})",
                marker=dict(color=COLORS["negative"], size=6, symbol="x"),
            ),
            row=1, col=1,
        )
    fig.update_yaxes(tickformat=".2%", row=1, col=1)

    current_row = 2

    # -- Panel 2: Correlation heatmap or distribution --
    if has_multi:
        corr = returns.corr()
        labels = list(corr.columns)
        corr_text = [
            [f"{corr.iloc[i, j]:.2f}" for j in range(n_assets)]
            for i in range(n_assets)
        ]
        fig.add_trace(
            go.Heatmap(
                z=corr.values,
                x=labels, y=labels,
                colorscale="RdBu_r",
                zmin=-1, zmax=1,
                text=corr_text,
                texttemplate="%{text}",
                hoverinfo="text",
                colorbar=dict(
                    title="Corr", len=0.25, y=0.55, x=1.02,
                ),
            ),
            row=current_row, col=1,
        )
        fig.update_yaxes(autorange="reversed", row=current_row, col=1)
        current_row += 1

        # -- Risk contribution panel --
        vol_contrib = returns.std() * np.sqrt(252)
        total_vol = vol_contrib.sum()
        pct_contrib = vol_contrib / total_vol if total_vol > 0 else vol_contrib

        colors = [_PALETTE[i % len(_PALETTE)] for i in range(n_assets)]
        fig.add_trace(
            go.Bar(
                x=list(pct_contrib.index),
                y=pct_contrib.values,
                marker_color=colors,
                name="Risk Contrib",
                showlegend=False,
                hovertemplate="<b>%{x}</b><br>Contribution: %{y:.1%}<extra></extra>",
            ),
            row=current_row, col=1,
        )
        fig.update_yaxes(tickformat=".0%", row=current_row, col=1)
        current_row += 1
    else:
        fig.add_trace(
            go.Histogram(
                x=port_returns.dropna().values,
                nbinsx=50,
                histnorm="probability density",
                marker_color=COLORS["primary"],
                opacity=0.7,
                name="Returns",
                showlegend=False,
            ),
            row=current_row, col=1,
        )
        current_row += 1

    # -- Stress test scenarios --
    if has_stress and stress_scenarios is not None:
        scenario_names = list(stress_scenarios.keys())
        scenario_values = list(stress_scenarios.values())
        bar_colors = [
            COLORS["positive"] if v >= 0 else COLORS["negative"]
            for v in scenario_values
        ]
        fig.add_trace(
            go.Bar(
                x=scenario_names,
                y=scenario_values,
                marker_color=bar_colors,
                name="Scenarios",
                showlegend=False,
                hovertemplate="<b>%{x}</b><br>Impact: %{y:.2%}<extra></extra>",
            ),
            row=current_row, col=1,
        )
        fig.update_yaxes(tickformat=".0%", row=current_row, col=1)

    fig.update_layout(
        **_dark_layout(
            title=dict(text=title, font=dict(size=16)),
            height=800 + (200 if has_multi else 0) + (200 if has_stress else 0),
            width=1100,
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
        )
    )

    return fig


# ---------------------------------------------------------------------------
# 4. Technical Dashboard
# ---------------------------------------------------------------------------


@requires_extra("viz")
def technical_dashboard(
    ohlcv: pd.DataFrame,
    indicators: list[str] | None = None,
    title: str = "Technical Analysis Dashboard",
) -> go.Figure:
    """Create a multi-panel technical analysis chart with indicators.

    Combines a candlestick chart with volume bars, overlay indicators
    (moving averages, Bollinger Bands), and subplot oscillators (RSI,
    MACD).

    Parameters:
        ohlcv: DataFrame with columns ``open, high, low, close`` and
            optionally ``volume``.  Column names are case-insensitive.
        indicators: List of indicator names to display.  Supported values:

            Overlays (drawn on price chart):
            - ``"sma20"``, ``"sma50"``, ``"sma200"`` -- Simple moving averages
            - ``"ema12"``, ``"ema20"``, ``"ema26"`` -- Exponential moving averages
            - ``"bb"`` -- Bollinger Bands (20-period, 2 std)

            Oscillators (drawn in sub-panels):
            - ``"rsi"`` -- 14-period Relative Strength Index
            - ``"macd"`` -- MACD (12, 26, 9)

            Defaults to ``["sma20", "sma50", "bb", "rsi", "macd"]`` when
            *None*.
        title: Dashboard title.

    Returns:
        A ``plotly.graph_objects.Figure``.

    Example:
        >>> import pandas as pd, numpy as np
        >>> dates = pd.bdate_range("2020-01-01", periods=252)
        >>> close = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, 252)))
        >>> df = pd.DataFrame({
        ...     "open": close * 0.99, "high": close * 1.02,
        ...     "low": close * 0.98, "close": close,
        ...     "volume": np.random.randint(1e6, 1e7, 252),
        ... }, index=dates)
        >>> fig = technical_dashboard(df, indicators=["sma20", "bb", "rsi", "macd"])
        >>> fig.show()
    """
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Normalise column names
    df = ohlcv.copy()
    df.columns = [c.lower() for c in df.columns]
    has_volume = "volume" in df.columns

    if indicators is None:
        indicators = ["sma20", "sma50", "bb", "rsi", "macd"]
    indicators = [ind.lower().strip() for ind in indicators]

    has_rsi = "rsi" in indicators
    has_macd = "macd" in indicators

    # Determine subplot structure
    n_rows = 1  # candlestick always
    row_heights = [0.50]
    if has_volume:
        n_rows += 1
        row_heights.append(0.10)
    if has_rsi:
        n_rows += 1
        row_heights.append(0.15)
    if has_macd:
        n_rows += 1
        row_heights.append(0.20)

    # Normalize
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    # -- Row 1: Candlestick --
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            increasing_line_color=COLORS["positive"],
            decreasing_line_color=COLORS["negative"],
            name="OHLC",
        ),
        row=1, col=1,
    )

    # Overlay indicators
    overlay_colors = [COLORS["secondary"], COLORS["accent"], COLORS["info"], COLORS["warning"]]
    color_idx = 0
    for ind in indicators:
        if ind.startswith("sma"):
            period = int(ind.replace("sma", ""))
            sma = df["close"].rolling(period).mean()
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=sma,
                    mode="lines", name=f"SMA {period}",
                    line=dict(
                        color=overlay_colors[color_idx % len(overlay_colors)],
                        width=1.3,
                    ),
                ),
                row=1, col=1,
            )
            color_idx += 1
        elif ind.startswith("ema"):
            period = int(ind.replace("ema", ""))
            ema = df["close"].ewm(span=period, adjust=False).mean()
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=ema,
                    mode="lines", name=f"EMA {period}",
                    line=dict(
                        color=overlay_colors[color_idx % len(overlay_colors)],
                        width=1.3, dash="dot",
                    ),
                ),
                row=1, col=1,
            )
            color_idx += 1
        elif ind == "bb":
            sma20 = df["close"].rolling(20).mean()
            std20 = df["close"].rolling(20).std()
            upper = sma20 + 2 * std20
            lower = sma20 - 2 * std20
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=upper,
                    mode="lines", name="BB Upper",
                    line=dict(color=COLORS["neutral"], width=1, dash="dash"),
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=lower,
                    mode="lines", name="BB Lower",
                    line=dict(color=COLORS["neutral"], width=1, dash="dash"),
                    fill="tonexty",
                    fillcolor="rgba(127, 127, 127, 0.10)",
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=sma20,
                    mode="lines", name="BB Mid",
                    line=dict(color=COLORS["neutral"], width=0.8),
                ),
                row=1, col=1,
            )

    fig.update_yaxes(title_text="Price", row=1, col=1)

    current_row = 2

    # -- Volume bars --
    if has_volume:
        vol_colors = [
            COLORS["positive"] if c >= o else COLORS["negative"]
            for c, o in zip(df["close"], df["open"], strict=False)
        ]
        fig.add_trace(
            go.Bar(
                x=df.index, y=df["volume"],
                marker_color=vol_colors,
                opacity=0.55,
                name="Volume",
                showlegend=False,
            ),
            row=current_row, col=1,
        )
        fig.update_yaxes(title_text="Volume", row=current_row, col=1)
        current_row += 1

    # -- RSI --
    if has_rsi:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        fig.add_trace(
            go.Scatter(
                x=df.index, y=rsi,
                mode="lines", name="RSI (14)",
                line=dict(color=COLORS["accent"], width=1.5),
            ),
            row=current_row, col=1,
        )
        # Overbought / oversold lines
        fig.add_hline(
            y=70, line_color=COLORS["negative"], line_width=0.8,
            line_dash="dash", row=current_row, col=1,
        )
        fig.add_hline(
            y=30, line_color=COLORS["positive"], line_width=0.8,
            line_dash="dash", row=current_row, col=1,
        )
        fig.add_hrect(
            y0=70, y1=100,
            fillcolor="rgba(214, 39, 40, 0.08)",
            line_width=0, row=current_row, col=1,
        )
        fig.add_hrect(
            y0=0, y1=30,
            fillcolor="rgba(44, 160, 44, 0.08)",
            line_width=0, row=current_row, col=1,
        )
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=current_row, col=1)
        current_row += 1

    # -- MACD --
    if has_macd:
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        hist_colors = [
            COLORS["positive"] if v >= 0 else COLORS["negative"]
            for v in histogram.values
        ]

        fig.add_trace(
            go.Bar(
                x=df.index, y=histogram,
                marker_color=hist_colors,
                opacity=0.5,
                name="MACD Hist",
                showlegend=False,
            ),
            row=current_row, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=macd_line,
                mode="lines", name="MACD",
                line=dict(color=COLORS["primary"], width=1.5),
            ),
            row=current_row, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=signal_line,
                mode="lines", name="Signal",
                line=dict(color=COLORS["secondary"], width=1.5, dash="dash"),
            ),
            row=current_row, col=1,
        )
        fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=0.5, row=current_row, col=1)
        fig.update_yaxes(title_text="MACD", row=current_row, col=1)

    fig.update_layout(
        **_dark_layout(
            title=dict(text=title, font=dict(size=16)),
            height=700 + (100 if has_rsi else 0) + (120 if has_macd else 0),
            width=1100,
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
        )
    )

    return fig
