"""Reusable Plotly chart components for the Streamlit dashboard.

Wraps common chart patterns so pages get consistent styling without
repeating boilerplate.  All functions return ``plotly.graph_objects.Figure``
objects with the dark theme applied.

When Plotly is not installed, functions fall back to returning ``None``
so that callers can use ``st.line_chart`` instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


def _try_import_plotly():
    """Return (go, True) if plotly is installed, else (None, False)."""
    try:
        import plotly.graph_objects as go

        return go, True
    except ImportError:
        return None, False


def _dark_layout(**overrides: Any) -> dict[str, Any]:
    """Base dark-theme layout kwargs."""
    defaults: dict[str, Any] = {
        "template": "plotly_dark",
        "font": {"family": "sans-serif", "size": 11, "color": "#e0e0e0"},
        "plot_bgcolor": "#1e1e1e",
        "paper_bgcolor": "#111111",
        "hovermode": "x unified",
        "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
    }
    defaults.update(overrides)
    return defaults


def line_chart(
    series: pd.Series,
    title: str = "",
    yaxis_title: str = "",
) -> Any | None:
    """Create a dark-themed Plotly line chart.

    Parameters:
        series: Data to plot (index = x, values = y).
        title: Chart title.
        yaxis_title: Y-axis label.

    Returns:
        ``plotly.graph_objects.Figure`` or ``None`` if Plotly is missing.
    """
    go, ok = _try_import_plotly()
    if not ok:
        return None

    fig = go.Figure(
        data=[go.Scatter(x=series.index, y=series.values, mode="lines")],
    )
    fig.update_layout(**_dark_layout(title=title, yaxis_title=yaxis_title))
    return fig


def equity_curve_chart(
    returns: pd.Series,
    title: str = "Equity Curve",
) -> Any | None:
    """Create an equity curve from a return series.

    Parameters:
        returns: Simple return series.
        title: Chart title.

    Returns:
        ``plotly.graph_objects.Figure`` or ``None`` if Plotly is missing.
    """
    eq = (1 + returns).cumprod()
    return line_chart(eq, title=title, yaxis_title="Cumulative Return")


def histogram_chart(
    series: pd.Series,
    title: str = "Distribution",
    nbins: int = 50,
) -> Any | None:
    """Create a dark-themed Plotly histogram.

    Parameters:
        series: Data values to bin.
        title: Chart title.
        nbins: Number of histogram bins.

    Returns:
        ``plotly.graph_objects.Figure`` or ``None`` if Plotly is missing.
    """
    go, ok = _try_import_plotly()
    if not ok:
        return None

    fig = go.Figure(data=[go.Histogram(x=series.values, nbinsx=nbins)])
    fig.update_layout(
        **_dark_layout(title=title, xaxis_title="Value", yaxis_title="Count"),
    )
    return fig
