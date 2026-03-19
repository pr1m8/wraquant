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
    """Display a single metric using ``st.metric``.

    Parameters:
        label: Metric name shown above the value.
        value: Formatted value string.
        delta: Optional delta string (e.g. "+2.3%").
        delta_color: Streamlit delta_color option
            ("normal", "inverse", "off").
    """
    import streamlit as st

    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def metrics_row(
    metrics: dict[str, str],
    deltas: dict[str, str] | None = None,
) -> None:
    """Display a row of metrics in equally-spaced columns.

    Parameters:
        metrics: Mapping of label -> formatted value string.
        deltas: Optional mapping of label -> delta string.
            Keys must match ``metrics`` keys.
    """
    import streamlit as st

    deltas = deltas or {}
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items(), strict=False):
        delta = deltas.get(label)
        col.metric(label=label, value=value, delta=delta)
