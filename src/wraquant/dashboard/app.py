"""wraquant Dashboard -- Interactive Quantitative Finance Analysis.

Main Streamlit application that wires together the page modules
under ``wraquant.dashboard.pages``.  Each page is a self-contained
render function that owns its own layout and state.

Launch:
    streamlit run src/wraquant/dashboard/app.py
    python -m wraquant.dashboard
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="wraquant Dashboard",
    page_icon="\U0001f4ca",  # bar chart emoji
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for dark, polished look
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background-color: #0f0f14;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #16161d;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
    }
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: 600;
    }
    /* Remove default padding on main block */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
    }
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 8px;
    }
    /* Divider color */
    hr {
        border-color: rgba(255,255,255,0.06) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.markdown("## wraquant")
st.sidebar.caption("The ultimate quant finance toolkit")
st.sidebar.divider()

PAGES = [
    "Home",
    "Fundamental Analysis",
    "Valuation",
    "Technical Analysis",
    "Risk & Regimes",
    "Portfolio Risk",
    "VaR & Stress Testing",
    "Volatility Modeling",
    "Regime Analysis",
    "Backtest Lab",
    "News & Events",
    "Screener",
]

page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.divider()

# Global ticker input (available on all pages via session state)
from wraquant.dashboard.components.sidebar import ticker_input  # noqa: E402

ticker = ticker_input()

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------

if page == "Home":
    from wraquant.dashboard.pages.overview import render

    render()

elif page == "Fundamental Analysis":
    from wraquant.dashboard.pages.fundamental_analysis import render

    render()

elif page == "Valuation":
    from wraquant.dashboard.pages.valuation import render

    render()

elif page == "Technical Analysis":
    from wraquant.dashboard.pages.technical_analysis import render

    render()

elif page == "Risk & Regimes":
    from wraquant.dashboard.pages.risk_regimes import render

    render()

elif page == "Portfolio Risk":
    from wraquant.dashboard.pages.portfolio_risk import render

    render()

elif page == "VaR & Stress Testing":
    from wraquant.dashboard.pages.var_analysis import render

    render()

elif page == "Volatility Modeling":
    from wraquant.dashboard.pages.volatility import render

    render()

elif page == "Regime Analysis":
    from wraquant.dashboard.pages.regime_analysis import render

    render()

elif page == "Backtest Lab":
    from wraquant.dashboard.pages.backtest_lab import render

    render()

elif page == "News & Events":
    from wraquant.dashboard.pages.news_events import render

    render()

elif page == "Screener":
    from wraquant.dashboard.pages.screener import render

    render()
