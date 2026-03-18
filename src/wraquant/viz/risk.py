"""Risk visualizations.

VaR backtests, rolling volatility, and tail distribution plots.
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
def plot_var_backtest(
    returns: pd.Series,
    var_level: pd.Series | float,
    confidence: float = 0.95,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Plot returns with a VaR threshold line, highlighting breaches.

    Parameters:
        returns: Simple return series.
        var_level: Value-at-Risk threshold.  Pass a scalar for a constant
            threshold or a Series for a time-varying VaR.
        confidence: VaR confidence level (used only for the title).
        ax: Matplotlib axes to plot on.  A new figure is created when *None*.

    Returns:
        The matplotlib Axes containing the plot.
    """
    import pandas as pd

    fig, ax = _get_or_create_ax(ax)

    ax.plot(
        returns.index,
        returns.values,
        color=COLORS["primary"],
        linewidth=0.8,
        alpha=0.7,
        label="Returns",
    )

    # VaR line
    if isinstance(var_level, (int, float)):
        var_series = pd.Series(var_level, index=returns.index)
    else:
        var_series = var_level

    ax.plot(
        var_series.index,
        var_series.values,
        color=COLORS["negative"],
        linewidth=1.2,
        linestyle="--",
        label=f"VaR ({confidence:.0%})",
    )

    # Highlight breaches
    breaches = returns[returns < var_series]
    if not breaches.empty:
        ax.scatter(
            breaches.index,
            breaches.values,
            color=COLORS["negative"],
            s=18,
            zorder=5,
            label=f"Breaches ({len(breaches)})",
        )

    ax.set_title(f"VaR Backtest ({confidence:.0%} confidence)")
    ax.set_ylabel("Return")
    ax.set_xlabel("")
    ax.legend(loc="lower left")
    ax.axhline(0, color=COLORS["neutral"], linewidth=0.5)
    return ax


@requires_extra("viz")
def plot_rolling_volatility(
    returns: pd.Series,
    window: int = 21,
    annualize: bool = True,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Plot rolling volatility line chart.

    Parameters:
        returns: Simple return series.
        window: Rolling window in periods (default 21 for ~1 month daily).
        annualize: Whether to annualize (multiply by sqrt(252)).
        ax: Matplotlib axes to plot on.  A new figure is created when *None*.

    Returns:
        The matplotlib Axes containing the plot.
    """
    import numpy as np

    fig, ax = _get_or_create_ax(ax)

    vol = returns.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(252)

    ax.plot(vol.index, vol.values, color=COLORS["primary"], linewidth=1.2)
    ax.fill_between(vol.index, 0, vol.values, color=COLORS["primary"], alpha=0.15)

    label = "Annualized " if annualize else ""
    ax.set_title(f"Rolling {window}-Day {label}Volatility")
    ax.set_ylabel("Volatility")
    ax.set_xlabel("")
    return ax


@requires_extra("viz")
def plot_tail_distribution(
    returns: pd.Series,
    threshold_percentile: float = 5,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Plot distribution tails, zoomed into the left tail.

    Parameters:
        returns: Simple return series.
        threshold_percentile: Percentile below which tails are highlighted.
        ax: Matplotlib axes to plot on.  A new figure is created when *None*.

    Returns:
        The matplotlib Axes containing the plot.
    """
    import numpy as np

    fig, ax = _get_or_create_ax(ax)

    clean = returns.dropna()
    threshold = np.percentile(clean.values, threshold_percentile)

    # Full histogram in light colour
    ax.hist(
        clean.values,
        bins=80,
        density=True,
        color=COLORS["primary"],
        alpha=0.35,
        edgecolor="white",
        label="All returns",
    )

    # Tail histogram
    tail = clean[clean <= threshold]
    if not tail.empty:
        ax.hist(
            tail.values,
            bins=30,
            density=True,
            color=COLORS["negative"],
            alpha=0.7,
            edgecolor="white",
            label=f"Left tail (<= {threshold_percentile}th pctile)",
        )

    ax.axvline(
        threshold,
        color=COLORS["negative"],
        linestyle="--",
        linewidth=1.2,
        label=f"{threshold_percentile}th percentile: {threshold:.4f}",
    )

    ax.set_title("Tail Distribution")
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left")
    return ax
