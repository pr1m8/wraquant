"""wraquant Dashboard -- Interactive Quantitative Finance Analysis.

Main Streamlit application that wires together the page modules
under ``wraquant.dashboard.pages``.  Each page is a self-contained
render function that owns its own layout and state.
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
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("wraquant")
st.sidebar.caption("The ultimate quant finance toolkit")

page = st.sidebar.selectbox(
    "Navigate",
    [
        "Home",
        "Experiment Browser",
        "Strategy Analysis",
        "Risk Monitor",
        "Regime Viewer",
        "Portfolio Optimizer",
        "TA Screener",
    ],
)

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------

if page == "Home":
    st.title("wraquant Dashboard")
    st.markdown("*Interactive quantitative finance analysis*")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Functions", "1,085")
    col2.metric("Tests", "3,493")
    col3.metric("Modules", "26")
    col4.metric("TA Indicators", "265")

    st.markdown("---")
    st.markdown("### Quick Analysis")
    st.markdown(
        "Upload a CSV with a returns column on the **Strategy Analysis** page, "
        "or explore experiments you have already run via the "
        "**Experiment Browser**."
    )

    st.markdown("### Available Pages")
    st.markdown(
        "- **Experiment Browser** -- Browse and compare experiment results.\n"
        "- **Strategy Analysis** -- Upload returns and get a full report.\n"
        "- **Risk Monitor** -- VaR, rolling metrics, stress tests.\n"
        "- **Regime Viewer** -- Detect and visualize market regimes.\n"
        "- **Portfolio Optimizer** -- Interactive portfolio construction.\n"
        "- **TA Screener** -- Apply 265 technical indicators to price data.\n"
    )

    st.markdown("### Quick Start (code)")
    st.code(
        """\
import wraquant as wq
report = wq.analyze(returns)
print(report["risk"]["sharpe"])
""",
        language="python",
    )

elif page == "Experiment Browser":
    from wraquant.dashboard.pages.experiment_browser import render
    render()

elif page == "Strategy Analysis":
    from wraquant.dashboard.pages.strategy_analysis import render
    render()

elif page == "Risk Monitor":
    from wraquant.dashboard.pages.risk_monitor import render
    render()

elif page == "Regime Viewer":
    from wraquant.dashboard.pages.regime_viewer import render
    render()

elif page == "Portfolio Optimizer":
    from wraquant.dashboard.pages.portfolio_optimizer import render
    render()

elif page == "TA Screener":
    from wraquant.dashboard.pages.ta_screener import render
    render()
