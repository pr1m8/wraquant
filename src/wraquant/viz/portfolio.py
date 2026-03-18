"""Portfolio-related visualizations.

Portfolio weight charts, efficient frontier, risk contributions, and
correlation matrix heatmaps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wraquant.core.decorators import requires_extra
from wraquant.viz.themes import COLORS, apply_theme

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import numpy as np
    import numpy.typing as npt
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
def plot_weights(
    weights: pd.Series | npt.NDArray[np.floating],
    names: list[str] | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Plot portfolio weights as a horizontal bar chart.

    Parameters:
        weights: Portfolio weight vector (pandas Series or numpy array).
        names: Asset names.  If *weights* is a Series its index is used
            by default; otherwise sequential integers are used.
        ax: Matplotlib axes to plot on.  A new figure is created when *None*.

    Returns:
        The matplotlib Axes containing the plot.
    """
    import numpy as np
    import pandas as pd

    fig, ax = _get_or_create_ax(ax)

    if isinstance(weights, pd.Series):
        labels = list(weights.index) if names is None else names
        values = weights.values
    else:
        values = np.asarray(weights)
        labels = names if names is not None else [str(i) for i in range(len(values))]

    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in values]
    y_pos = range(len(values))
    ax.barh(y_pos, values, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Weight")
    ax.set_title("Portfolio Weights")
    ax.axvline(0, color=COLORS["neutral"], linewidth=0.8)
    return ax


@requires_extra("viz")
def plot_efficient_frontier(
    returns_range: npt.NDArray[np.floating],
    vol_range: npt.NDArray[np.floating],
    sharpe_range: npt.NDArray[np.floating] | None = None,
    optimal_point: tuple[float, float] | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Plot the efficient frontier as a scatter/line plot.

    Parameters:
        returns_range: Array of expected returns for each portfolio.
        vol_range: Array of volatilities for each portfolio.
        sharpe_range: Optional array of Sharpe ratios used to color the
            scatter points.
        optimal_point: Optional ``(volatility, return)`` tuple marking the
            optimal portfolio.
        ax: Matplotlib axes to plot on.  A new figure is created when *None*.

    Returns:
        The matplotlib Axes containing the plot.
    """

    fig, ax = _get_or_create_ax(ax)

    if sharpe_range is not None:

        sc = ax.scatter(
            vol_range,
            returns_range,
            c=sharpe_range,
            cmap="viridis",
            s=10,
            alpha=0.7,
        )
        fig.colorbar(sc, ax=ax, label="Sharpe Ratio", shrink=0.8)
    else:
        ax.plot(
            vol_range,
            returns_range,
            color=COLORS["primary"],
            linewidth=1.5,
        )

    if optimal_point is not None:
        ax.scatter(
            [optimal_point[0]],
            [optimal_point[1]],
            color=COLORS["negative"],
            marker="*",
            s=200,
            zorder=5,
            label="Optimal",
        )
        ax.legend()

    ax.set_title("Efficient Frontier")
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Expected Return")
    return ax


@requires_extra("viz")
def plot_risk_contribution(
    contributions: pd.Series | npt.NDArray[np.floating],
    names: list[str] | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Plot risk contributions as a stacked bar chart.

    Parameters:
        contributions: Risk contribution per asset (pandas Series or array).
        names: Asset names.  If *contributions* is a Series its index is
            used by default.
        ax: Matplotlib axes to plot on.  A new figure is created when *None*.

    Returns:
        The matplotlib Axes containing the plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    fig, ax = _get_or_create_ax(ax)

    if isinstance(contributions, pd.Series):
        labels = list(contributions.index) if names is None else names
        values = contributions.values.astype(float)
    else:
        values = np.asarray(contributions, dtype=float)
        labels = names if names is not None else [str(i) for i in range(len(values))]

    cmap = plt.cm.tab10  # type: ignore[attr-defined]
    bar_colors = [cmap(i % 10) for i in range(len(values))]

    ax.bar(labels, values, color=bar_colors, edgecolor="white")
    ax.set_title("Risk Contribution by Asset")
    ax.set_ylabel("Risk Contribution")
    ax.set_xlabel("")

    # Rotate labels if there are many assets
    if len(labels) > 6:
        ax.tick_params(axis="x", rotation=45)
    return ax


@requires_extra("viz")
def plot_correlation_matrix(
    corr_matrix: pd.DataFrame | npt.NDArray[np.floating],
    labels: list[str] | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Plot correlation matrix as an annotated heatmap.

    Parameters:
        corr_matrix: Square correlation matrix (DataFrame or 2-D array).
        labels: Axis labels.  If *corr_matrix* is a DataFrame its columns
            are used by default.
        ax: Matplotlib axes to plot on.  A new figure is created when *None*.

    Returns:
        The matplotlib Axes containing the plot.
    """
    import numpy as np
    import pandas as pd

    fig, ax = _get_or_create_ax(ax, figsize=(8, 7))

    if isinstance(corr_matrix, pd.DataFrame):
        tick_labels = list(corr_matrix.columns) if labels is None else labels
        data = corr_matrix.values
    else:
        data = np.asarray(corr_matrix)
        tick_labels = (
            labels if labels is not None else [str(i) for i in range(data.shape[0])]
        )

    im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    n = data.shape[0]
    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(tick_labels)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = data[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(val) > 0.6 else "black",
            )

    ax.set_title("Correlation Matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return ax
