"""Metric display cards for the Streamlit dashboard.

Provides ``metric_card`` and ``metrics_row`` helpers that render
consistent metric displays across all dashboard pages.
"""

from __future__ import annotations


def metric_card(
    label: str,
    value: str,
    delta: str | None = None,
    delta_color: str = "normal",
) -> None:
    """Display a single metric using ``st.metric``."""
    import streamlit as st

    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def metrics_row(
    metrics: dict[str, str],
    deltas: dict[str, str] | None = None,
) -> None:
    """Display a row of metrics in equally-spaced columns."""
    import streamlit as st

    deltas = deltas or {}
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items(), strict=False):
        delta = deltas.get(label)
        col.metric(label=label, value=value, delta=delta)


def fmt_number(
    value: float, prefix: str = "", suffix: str = "", decimals: int = 2
) -> str:
    """Format a number with optional prefix/suffix and smart abbreviation."""
    if abs(value) >= 1e12:
        return f"{prefix}{value / 1e12:.{decimals}f}T{suffix}"
    if abs(value) >= 1e9:
        return f"{prefix}{value / 1e9:.{decimals}f}B{suffix}"
    if abs(value) >= 1e6:
        return f"{prefix}{value / 1e6:.{decimals}f}M{suffix}"
    if abs(value) >= 1e3:
        return f"{prefix}{value / 1e3:.{decimals}f}K{suffix}"
    return f"{prefix}{value:.{decimals}f}{suffix}"


def fmt_pct(value: float, decimals: int = 1) -> str:
    """Format a decimal as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


def fmt_currency(value: float, decimals: int = 2) -> str:
    """Format a number as USD currency with smart abbreviation."""
    return fmt_number(value, prefix="$", decimals=decimals)
