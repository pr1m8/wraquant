"""Common sidebar elements for the Streamlit dashboard.

Provides reusable sidebar widgets (date range pickers, file uploaders)
used across multiple dashboard pages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def date_range_selector(
    key_prefix: str = "date",
) -> tuple["pd.Timestamp | None", "pd.Timestamp | None"]:
    """Render start/end date inputs in the sidebar.

    Parameters:
        key_prefix: Unique key prefix for the Streamlit widgets
            (prevents key collisions across pages).

    Returns:
        Tuple of (start_date, end_date) as ``pd.Timestamp`` or
        ``None`` if not set.
    """
    import streamlit as st

    start = st.sidebar.date_input(
        "Start date", value=None, key=f"{key_prefix}_start",
    )
    end = st.sidebar.date_input(
        "End date", value=None, key=f"{key_prefix}_end",
    )

    import pandas as pd

    return (
        pd.Timestamp(start) if start else None,
        pd.Timestamp(end) if end else None,
    )


def file_uploader_returns(
    label: str = "Upload returns CSV",
    key: str = "returns_upload",
) -> "pd.DataFrame | None":
    """Sidebar file uploader that reads a returns CSV.

    Parameters:
        label: Upload widget label.
        key: Unique Streamlit widget key.

    Returns:
        Parsed DataFrame with DatetimeIndex, or ``None`` if nothing
        uploaded.
    """
    import streamlit as st

    upload = st.sidebar.file_uploader(label, type=["csv"], key=key)
    if upload is not None:
        import pandas as pd

        return pd.read_csv(upload, index_col=0, parse_dates=True)
    return None
