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


# -- Colour palette --------------------------------------------------------

COLORS = {
    "primary": "#6366f1",  # Indigo
    "secondary": "#8b5cf6",  # Violet
    "success": "#22c55e",  # Green
    "danger": "#ef4444",  # Red
    "warning": "#f59e0b",  # Amber
    "info": "#06b6d4",  # Cyan
    "neutral": "#94a3b8",  # Slate
    "bg": "#0f0f14",  # Near-black
    "card_bg": "#16161d",  # Card background
    "surface": "#1e1e28",  # Surface
    "text": "#e2e8f0",  # Light grey
    "text_muted": "#64748b",  # Muted text
    "accent1": "#f472b6",  # Pink
    "accent2": "#38bdf8",  # Sky
    "accent3": "#a78bfa",  # Purple
    "accent4": "#34d399",  # Emerald
}

SERIES_COLORS = [
    COLORS["primary"],
    COLORS["accent2"],
    COLORS["accent4"],
    COLORS["accent1"],
    COLORS["warning"],
    COLORS["accent3"],
    COLORS["info"],
    COLORS["success"],
]


def dark_layout(**overrides: Any) -> dict[str, Any]:
    """Base dark-theme layout kwargs."""
    defaults: dict[str, Any] = {
        "template": "plotly_dark",
        "font": {
            "family": "Inter, system-ui, sans-serif",
            "size": 12,
            "color": COLORS["text"],
        },
        "plot_bgcolor": COLORS["bg"],
        "paper_bgcolor": COLORS["bg"],
        "hovermode": "x unified",
        "margin": {"l": 50, "r": 20, "t": 50, "b": 40},
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 11},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        "xaxis": {
            "gridcolor": "rgba(255,255,255,0.06)",
            "zerolinecolor": "rgba(255,255,255,0.08)",
        },
        "yaxis": {
            "gridcolor": "rgba(255,255,255,0.06)",
            "zerolinecolor": "rgba(255,255,255,0.08)",
        },
    }
    defaults.update(overrides)
    return defaults


def line_chart(
    series: pd.Series,
    title: str = "",
    yaxis_title: str = "",
) -> Any | None:
    """Create a dark-themed Plotly line chart."""
    go, ok = _try_import_plotly()
    if not ok:
        return None

    fig = go.Figure(
        data=[
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                line={"color": COLORS["primary"], "width": 2},
            )
        ],
    )
    fig.update_layout(**dark_layout(title=title, yaxis_title=yaxis_title))
    return fig


def equity_curve_chart(
    returns: pd.Series,
    title: str = "Equity Curve",
) -> Any | None:
    """Create an equity curve from a return series."""
    eq = (1 + returns).cumprod()
    return line_chart(eq, title=title, yaxis_title="Cumulative Return")


def histogram_chart(
    series: pd.Series,
    title: str = "Distribution",
    nbins: int = 50,
) -> Any | None:
    """Create a dark-themed Plotly histogram."""
    go, ok = _try_import_plotly()
    if not ok:
        return None

    fig = go.Figure(
        data=[
            go.Histogram(
                x=series.values,
                nbinsx=nbins,
                marker_color=COLORS["primary"],
            )
        ]
    )
    fig.update_layout(
        **dark_layout(title=title, xaxis_title="Value", yaxis_title="Count"),
    )
    return fig
