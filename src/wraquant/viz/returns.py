"""Return-related visualizations.

Cumulative returns, drawdowns, return distributions, rolling returns,
and monthly heatmaps.
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
def plot_cumulative_returns(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    title: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Plot cumulative return line chart with optional benchmark overlay.

    Parameters:
        returns: Simple return series (not cumulative).
        benchmark: Optional benchmark return series for comparison.
        title: Plot title.  Defaults to ``"Cumulative Returns"``.
        ax: Matplotlib axes to plot on.  A new figure is created when *None*.

    Returns:
        The matplotlib Axes containing the plot.
    """

    fig, ax = _get_or_create_ax(ax)

    cum = (1 + returns).cumprod() - 1
    ax.plot(
        cum.index, cum.values, color=COLORS["primary"], label=returns.name or "Strategy"
    )

    if benchmark is not None:
        cum_bench = (1 + benchmark).cumprod() - 1
        ax.plot(
            cum_bench.index,
            cum_bench.values,
            color=COLORS["benchmark"],
            label=benchmark.name or "Benchmark",
            linestyle="--",
        )
        ax.legend()

    ax.set_title(title or "Cumulative Returns")
    ax.set_ylabel("Cumulative Return")
    ax.set_xlabel("")
    ax.axhline(0, color=COLORS["neutral"], linewidth=0.8, linestyle="-")
    ax.yaxis.set_major_formatter(_percent_formatter())
    return ax


@requires_extra("viz")
def plot_drawdowns(
    returns: pd.Series,
    top_n: int = 5,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Plot underwater chart showing drawdown periods.

    Parameters:
        returns: Simple return series.
        top_n: Number of largest drawdowns to shade.
        ax: Matplotlib axes to plot on.  A new figure is created when *None*.

    Returns:
        The matplotlib Axes containing the plot.
    """

    fig, ax = _get_or_create_ax(ax)

    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max

    ax.fill_between(
        drawdown.index,
        drawdown.values,
        0,
        color=COLORS["drawdown"],
        alpha=0.35,
        label="Drawdown",
    )
    ax.plot(drawdown.index, drawdown.values, color=COLORS["drawdown"], linewidth=0.8)

    ax.set_title("Underwater Plot (Drawdowns)")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(_percent_formatter())
    ax.legend(loc="lower left")
    return ax


@requires_extra("viz")
def plot_return_distribution(
    returns: pd.Series,
    bins: int = 50,
    fit_normal: bool = True,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Plot histogram of returns with optional normal distribution fit overlay.

    Parameters:
        returns: Simple return series.
        bins: Number of histogram bins.
        fit_normal: If *True*, overlay a fitted normal PDF.
        ax: Matplotlib axes to plot on.  A new figure is created when *None*.

    Returns:
        The matplotlib Axes containing the plot.
    """
    import numpy as np

    fig, ax = _get_or_create_ax(ax)

    ax.hist(
        returns.dropna().values,
        bins=bins,
        density=True,
        color=COLORS["primary"],
        alpha=0.65,
        edgecolor="white",
        label="Returns",
    )

    if fit_normal:
        from scipy.stats import norm

        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 200)
        ax.plot(
            x,
            norm.pdf(x, mu, sigma),
            color=COLORS["negative"],
            linewidth=1.5,
            label="Normal fit",
        )
        ax.legend()

    ax.set_title("Return Distribution")
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    return ax


@requires_extra("viz")
def plot_rolling_returns(
    returns: pd.Series,
    window: int = 252,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Plot rolling annualized returns.

    Parameters:
        returns: Simple return series.
        window: Rolling window in periods (default 252 for 1 year of daily data).
        ax: Matplotlib axes to plot on.  A new figure is created when *None*.

    Returns:
        The matplotlib Axes containing the plot.
    """
    import numpy as np

    fig, ax = _get_or_create_ax(ax)

    rolling_ret = (
        (1 + returns)
        .rolling(window)
        .apply(
            lambda x: np.prod(x) ** (252 / window) - 1,
            raw=True,
        )
    )

    ax.plot(
        rolling_ret.index, rolling_ret.values, color=COLORS["primary"], linewidth=1.2
    )
    ax.axhline(0, color=COLORS["neutral"], linewidth=0.8, linestyle="-")

    ax.set_title(f"Rolling {window}-Day Annualized Return")
    ax.set_ylabel("Annualized Return")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(_percent_formatter())
    return ax


@requires_extra("viz")
def plot_monthly_heatmap(
    returns: pd.Series,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Plot a month-by-year heatmap of returns.

    Parameters:
        returns: Simple return series with a ``DatetimeIndex``.
        ax: Matplotlib axes to plot on.  A new figure is created when *None*.

    Returns:
        The matplotlib Axes containing the plot.
    """
    import numpy as np

    fig, ax = _get_or_create_ax(
        ax, figsize=(12, max(4, len(set(returns.index.year)) * 0.5 + 1))
    )

    # Aggregate to monthly
    monthly = returns.groupby([returns.index.year, returns.index.month]).apply(
        lambda x: (1 + x).prod() - 1,
    )
    monthly.index.names = ["year", "month"]
    table = monthly.unstack(level="month")
    table.columns = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ][: len(table.columns)]

    import matplotlib.pyplot as plt

    cmap = plt.cm.RdYlGn  # type: ignore[attr-defined]
    vmax = max(abs(table.max().max()), abs(table.min().min()))
    im = ax.imshow(
        table.values,
        cmap=cmap,
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
    )

    # Ticks / labels
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels(table.index)

    # Annotate cells
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            val = table.iloc[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.1%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(val) > vmax * 0.6 else "black",
                )

    ax.set_title("Monthly Returns Heatmap")
    fig.colorbar(im, ax=ax, format="%.0f%%", shrink=0.8)
    return ax


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _percent_formatter():  # noqa: ANN202
    """Return a matplotlib FuncFormatter that displays values as percentages."""
    import matplotlib.ticker as mticker

    return mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
