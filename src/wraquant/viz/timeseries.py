"""Time series visualizations.

Basic time series line plots, regime overlay, and seasonal decomposition
panels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wraquant.core.decorators import requires_extra
from wraquant.viz.themes import COLORS, apply_theme

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_or_create_ax(
    ax: matplotlib.axes.Axes | None,
    figsize: tuple[float, float] = (12, 6),
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Return an existing axes or create a new figure/axes pair."""
    import matplotlib.pyplot as plt

    if ax is not None:
        fig = ax.get_figure()
        return fig, ax
    fig, ax = plt.subplots(figsize=figsize)
    apply_theme(fig, ax)
    return fig, ax


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@requires_extra("viz")
def plot_series(
    data: pd.Series | pd.DataFrame,
    title: str | None = None,
    ylabel: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Plot a basic time series line chart.

    Parameters:
        data: Time series data.  A Series produces a single line; a
            DataFrame plots one line per column.
        title: Plot title.
        ylabel: Y-axis label.
        ax: Matplotlib axes to plot on.  A new figure is created when *None*.

    Returns:
        The matplotlib Axes containing the plot.
    """
    import pandas as pd

    fig, ax = _get_or_create_ax(ax)

    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            ax.plot(data.index, data[col].values, label=col, linewidth=1.2)
        ax.legend()
    else:
        ax.plot(
            data.index,
            data.values,
            color=COLORS["primary"],
            linewidth=1.2,
            label=data.name,
        )
        if data.name:
            ax.legend()

    ax.set_title(title or "")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    return ax


@requires_extra("viz")
def plot_regime_overlay(
    data: pd.Series,
    regimes: pd.Series,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Plot a time series with colored background regions for regimes.

    Parameters:
        data: Time series to plot as a line.
        regimes: Integer or categorical series of the same index indicating
            the regime at each point.
        ax: Matplotlib axes to plot on.  A new figure is created when *None*.

    Returns:
        The matplotlib Axes containing the plot.
    """
    import matplotlib.pyplot as plt

    fig, ax = _get_or_create_ax(ax)

    ax.plot(data.index, data.values, color=COLORS["primary"], linewidth=1.2)

    unique_regimes = sorted(regimes.unique())
    cmap = plt.cm.Pastel1  # type: ignore[attr-defined]
    regime_colors = {r: cmap(i % cmap.N) for i, r in enumerate(unique_regimes)}

    # Shade regime spans
    prev_regime = regimes.iloc[0]
    span_start = data.index[0]
    for idx, regime in zip(regimes.index[1:], regimes.iloc[1:], strict=False):
        if regime != prev_regime:
            ax.axvspan(span_start, idx, alpha=0.25, color=regime_colors[prev_regime])
            span_start = idx
            prev_regime = regime
    # Final span
    ax.axvspan(span_start, data.index[-1], alpha=0.25, color=regime_colors[prev_regime])

    # Legend patches
    import matplotlib.patches as mpatches

    patches = [
        mpatches.Patch(color=regime_colors[r], alpha=0.4, label=f"Regime {r}")
        for r in unique_regimes
    ]
    ax.legend(handles=patches, loc="best")

    ax.set_title("Time Series with Regime Overlay")
    ax.set_xlabel("")
    return ax


@requires_extra("viz")
def plot_decomposition(
    trend: pd.Series,
    seasonal: pd.Series,
    residual: pd.Series,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Plot a 3-panel decomposition (trend, seasonal, residual).

    Parameters:
        trend: Trend component series.
        seasonal: Seasonal component series.
        residual: Residual component series.
        ax: Ignored.  A new 3-panel figure is always created.

    Returns:
        The matplotlib Figure containing the three-panel plot.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for a in axes:
        apply_theme(fig, a)

    components = [
        (trend, "Trend", COLORS["primary"]),
        (seasonal, "Seasonal", COLORS["secondary"]),
        (residual, "Residual", COLORS["accent"]),
    ]

    for a, (series, label, color) in zip(axes, components, strict=False):
        a.plot(series.index, series.values, color=color, linewidth=1.2)
        a.set_ylabel(label)
        a.set_title(label)

    axes[-1].set_xlabel("")
    fig.suptitle("Time Series Decomposition", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig
