"""wraquant Dashboard v2 -- Interactive Quantitative Finance Analysis.

Complete rewrite with streamlit-option-menu sidebar navigation,
grouped page sections, and polished dark theme.

Launch:
    streamlit run src/wraquant/dashboard/app.py
    python -m wraquant.dashboard
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Auto-load .env file if it exists (before any other imports)
# ---------------------------------------------------------------------------
_env_file = Path(__file__).resolve().parents[3] / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

import streamlit as st  # noqa: E402

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="wraquant Dashboard",
    page_icon="\U0001f4ca",
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
    /* Option menu tweaks */
    .nav-link {
        font-size: 0.9rem !important;
    }
    .nav-link-selected {
        background-color: #6366f1 !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar navigation with streamlit-option-menu
# ---------------------------------------------------------------------------

with st.sidebar:
    try:
        from streamlit_option_menu import option_menu

        selected = option_menu(
            "wraquant",
            [
                "Overview",
                "---",
                "Fundamental",
                "Valuation",
                "Screener",
                "---",
                "Technical Analysis",
                "Returns & Stats",
                "---",
                "Risk Dashboard",
                "VaR & Stress Test",
                "Portfolio",
                "---",
                "Volatility",
                "Regimes",
                "Forecasting",
                "---",
                "News & Events",
                "FX Analysis",
            ],
            icons=[
                "house",
                None,
                "building",
                "calculator",
                "search",
                None,
                "graph-up",
                "bar-chart",
                None,
                "shield",
                "exclamation-triangle",
                "pie-chart",
                None,
                "activity",
                "layers",
                "graph-up-arrow",
                None,
                "newspaper",
                "currency-exchange",
            ],
            menu_icon="graph-up",
            default_index=0,
            styles={
                "container": {"padding": "4px", "background-color": "#16161d"},
                "icon": {"color": "#94a3b8", "font-size": "14px"},
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "left",
                    "margin": "2px 0",
                    "color": "#e2e8f0",
                    "--hover-color": "#1e1e28",
                },
                "nav-link-selected": {
                    "background-color": "#6366f1",
                    "color": "white",
                },
                "separator": {
                    "border-color": "rgba(255,255,255,0.06)",
                },
            },
        )
    except ImportError:
        st.markdown("## wraquant")
        st.caption("Install `streamlit-option-menu` for enhanced nav")
        PAGES = [
            "Overview",
            "Fundamental",
            "Valuation",
            "Screener",
            "Technical Analysis",
            "Returns & Stats",
            "Risk Dashboard",
            "VaR & Stress Test",
            "Portfolio",
            "Volatility",
            "Regimes",
            "Forecasting",
            "News & Events",
            "FX Analysis",
        ]
        selected = st.radio("Navigate", PAGES, label_visibility="collapsed")

# ---------------------------------------------------------------------------
# Top bar with ticker input (NOT in sidebar)
# ---------------------------------------------------------------------------

_top_left, _top_right = st.columns([3, 1])
with _top_right:
    _ticker = st.text_input(
        "Ticker",
        value=st.session_state.get("ticker", "AAPL"),
        key="_ticker_input",
        label_visibility="collapsed",
        placeholder="Enter ticker...",
    )
    if _ticker:
        st.session_state["ticker"] = _ticker.upper().strip()

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------

_PAGE_MAP = {
    "Overview": "wraquant.dashboard.views.overview",
    "Fundamental": "wraquant.dashboard.views.fundamental_analysis",
    "Valuation": "wraquant.dashboard.views.valuation",
    "Screener": "wraquant.dashboard.views.screener",
    "Technical Analysis": "wraquant.dashboard.views.technical_analysis",
    "Returns & Stats": "wraquant.dashboard.views.returns_stats",
    "Risk Dashboard": "wraquant.dashboard.views.risk_regimes",
    "VaR & Stress Test": "wraquant.dashboard.views.var_analysis",
    "Portfolio": "wraquant.dashboard.views.portfolio_risk",
    "Volatility": "wraquant.dashboard.views.volatility",
    "Regimes": "wraquant.dashboard.views.regime_analysis",
    "Forecasting": "wraquant.dashboard.views.forecasting",
    "News & Events": "wraquant.dashboard.views.news_events",
    "FX Analysis": "wraquant.dashboard.views.fx_analysis",
}

if selected and selected != "---":
    module_path = _PAGE_MAP.get(selected)
    if module_path:
        import importlib

        try:
            mod = importlib.import_module(module_path)
            mod.render()
        except ImportError as exc:
            st.error(f"Could not load page **{selected}**: {exc}")
        except Exception as exc:
            st.error(f"Error rendering **{selected}**: {exc}")
            import traceback

            st.code(traceback.format_exc(), language="python")
