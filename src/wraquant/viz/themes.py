"""Consistent theming for wraquant visualizations.

Provides a clean, professional dark-on-white style and a named color
palette used across all plotting functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from wraquant.core.decorators import requires_extra

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

# ---------------------------------------------------------------------------
# Named color palette
# ---------------------------------------------------------------------------

COLORS: dict[str, str] = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "positive": "#2ca02c",
    "negative": "#d62728",
    "neutral": "#7f7f7f",
    "accent": "#9467bd",
    "info": "#17becf",
    "warning": "#bcbd22",
    "drawdown": "#d62728",
    "benchmark": "#ff7f0e",
    "fill": "#1f77b4",
}


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------


@requires_extra("viz")
def set_wraquant_style() -> None:
    """Apply a clean, professional dark-on-white matplotlib style via rcParams.

    This modifies the global matplotlib rcParams so that all subsequent plots
    use the wraquant house style.  Call once at the start of a session or
    notebook.

    Returns:
        None
    """
    import matplotlib.pyplot as plt

    style: dict[str, Any] = {
        # Figure
        "figure.figsize": (12, 6),
        "figure.dpi": 100,
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        # Axes
        "axes.facecolor": "white",
        "axes.edgecolor": "#cccccc",
        "axes.labelcolor": "#333333",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Grid
        "grid.color": "#e0e0e0",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        # Lines
        "lines.linewidth": 1.5,
        "lines.antialiased": True,
        # Ticks
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        # Legend
        "legend.fontsize": 10,
        "legend.framealpha": 0.8,
        "legend.edgecolor": "#cccccc",
        # Font
        "font.family": "sans-serif",
        "font.size": 11,
        # Savefig
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }

    plt.rcParams.update(style)


@requires_extra("viz")
def apply_theme(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
) -> None:
    """Apply the wraquant visual theme to an existing figure and axes.

    Parameters:
        fig: The matplotlib Figure to style.
        ax: The matplotlib Axes to style.

    Returns:
        None
    """
    fig.set_facecolor("white")
    fig.set_edgecolor("white")

    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors="#333333")
    ax.xaxis.label.set_color("#333333")
    ax.yaxis.label.set_color("#333333")
    ax.title.set_color("#333333")
    ax.grid(True, color="#e0e0e0", linewidth=0.5, alpha=0.7)
