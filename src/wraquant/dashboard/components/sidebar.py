"""Common sidebar elements for the Streamlit dashboard.

Provides reusable sidebar widgets (date range pickers, file uploaders,
ticker input) used across multiple dashboard pages.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def check_api_key() -> bool:
    """Check whether FMP_API_KEY is set and display a message if not.

    Returns:
        True if the key is available, False otherwise.
    """
    import streamlit as st

    if not os.environ.get("FMP_API_KEY"):
        st.error(
            "**FMP API key not found.**  \n"
            "Set the `FMP_API_KEY` environment variable before launching:\n"
            "```bash\n"
            "export FMP_API_KEY='your_key_here'\n"
            "streamlit run src/wraquant/dashboard/app.py\n"
            "```\n"
            "Get a free key at [financialmodelingprep.com](https://financialmodelingprep.com/developer/docs/)",
        )
        return False
    return True


def ticker_input() -> str:
    """Render a ticker symbol input in the sidebar and persist it in session state.

    Returns:
        The current ticker symbol (uppercase, stripped).
    """
    import streamlit as st

    if "ticker" not in st.session_state:
        st.session_state["ticker"] = "AAPL"

    ticker = (
        st.sidebar.text_input(
            "Ticker Symbol",
            value=st.session_state["ticker"],
            key="ticker_input_widget",
            placeholder="e.g. AAPL",
        )
        .strip()
        .upper()
    )

    if ticker:
        st.session_state["ticker"] = ticker

    return st.session_state["ticker"]


def date_range_selector(
    key_prefix: str = "date",
) -> tuple["pd.Timestamp | None", "pd.Timestamp | None"]:
    """Render start/end date inputs in the sidebar."""
    import streamlit as st

    start = st.sidebar.date_input(
        "Start date",
        value=None,
        key=f"{key_prefix}_start",
    )
    end = st.sidebar.date_input(
        "End date",
        value=None,
        key=f"{key_prefix}_end",
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
    """Sidebar file uploader that reads a returns CSV."""
    import streamlit as st

    upload = st.sidebar.file_uploader(label, type=["csv"], key=key)
    if upload is not None:
        import pandas as pd

        return pd.read_csv(upload, index_col=0, parse_dates=True)
    return None
