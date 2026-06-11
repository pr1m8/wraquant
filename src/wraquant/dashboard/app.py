"""wraquant Dashboard v2 -- Interactive Quantitative Finance Analysis.

Complete rewrite with streamlit-option-menu sidebar navigation,
grouped page sections, and polished dark theme.

Launch:
    streamlit run src/wraquant/dashboard/app.py
    python -m wraquant.dashboard
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Auto-load .env file if it exists (before any other imports)
# ---------------------------------------------------------------------------
_env_file = Path(__file__).resolve().parents[3] / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            os.environ.setdefault(_key.strip(), _val.strip())

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
_CSS = (
    "<style>"
    ".stApp{background-color:#0f0f14}"
    '[data-testid="stSidebar"]{background-color:#16161d;border-right:1px solid rgba(255,255,255,0.06)}'
    '[data-testid="stSidebar"] .stMarkdown h1,'
    '[data-testid="stSidebar"] .stMarkdown h2,'
    '[data-testid="stSidebar"] .stMarkdown h3{color:#e2e8f0}'
    '[data-testid="stMetricValue"]{font-size:1.4rem;font-weight:600}'
    ".block-container{padding-top:2rem;padding-bottom:1rem}"
    '.stTabs [data-baseweb="tab-list"]{gap:8px}'
    '.stTabs [data-baseweb="tab"]{border-radius:6px 6px 0 0;padding:8px 20px}'
    ".stDataFrame{border-radius:8px}"
    "hr{border-color:rgba(255,255,255,0.06)!important}"
    ".nav-link{font-size:0.9rem!important}"
    ".nav-link-selected{background-color:#6366f1!important;font-weight:600!important}"
    "</style>"
)
st.markdown(_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
_ITEMS = [
    "Overview", "---",
    "Fundamental", "Valuation", "Screener", "---",
    "Technical Analysis", "Returns & Stats", "Time Series", "---",
    "Risk Dashboard", "VaR & Stress Test", "Portfolio", "---",
    "Volatility", "Regimes", "Forecasting", "---",
    "Backtest Lab", "ML Lab", "---",
    "Microstructure", "Quant Lab", "---",
    "Pricing & Options", "Causal & Bayes", "---",
    "News & Events", "FX Analysis",
]
_ICONS = [
    "house", None,
    "building", "calculator", "search", None,
    "graph-up", "bar-chart", "clock-history", None,
    "shield", "exclamation-triangle", "pie-chart", None,
    "activity", "layers", "graph-up-arrow", None,
    "play-circle", "robot", None,
    "cpu", "mortarboard", None,
    "cash-coin", "diagram-3", None,
    "newspaper", "currency-exchange",
]
_STYLES = {
    "container": {"padding": "4px", "background-color": "#16161d"},
    "icon": {"color": "#94a3b8", "font-size": "14px"},
    "nav-link": {
        "font-size": "14px", "text-align": "left",
        "margin": "2px 0", "color": "#e2e8f0", "--hover-color": "#1e1e28",
    },
    "nav-link-selected": {"background-color": "#6366f1", "color": "white"},
    "separator": {"border-color": "rgba(255,255,255,0.06)"},
}

with st.sidebar:
    try:
        from streamlit_option_menu import option_menu  # noqa: E402
        selected = option_menu(
            "wraquant", _ITEMS, icons=_ICONS,
            menu_icon="graph-up", default_index=0, styles=_STYLES,
        )
    except ImportError:
        st.markdown("## wraquant")
        st.caption("Install `streamlit-option-menu` for enhanced nav")
        selected = st.radio(
            "Navigate",
            [p for p in _ITEMS if p != "---"],
            label_visibility="collapsed",
        )

    st.divider()

# ---------------------------------------------------------------------------
# Page routing  (wraquant.dashboard.views.*)
# ---------------------------------------------------------------------------
_MAP = {
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
    "Time Series": "wraquant.dashboard.views.time_series",
    "Backtest Lab": "wraquant.dashboard.views.backtest_lab",
    "Microstructure": "wraquant.dashboard.views.microstructure",
    "Quant Lab": "wraquant.dashboard.views.quant_lab",
    "ML Lab": "wraquant.dashboard.views.ml_lab",
    "Pricing & Options": "wraquant.dashboard.views.pricing",
    "Causal & Bayes": "wraquant.dashboard.views.causal_bayes",
}

# ---------------------------------------------------------------------------
# Ticker input — top header bar, NOT sidebar
# ---------------------------------------------------------------------------

_hdr_l, _hdr_r = st.columns([3, 1])
with _hdr_r:
    _tk = st.text_input(
        "Symbol", value=st.session_state.get("ticker", "AAPL"),
        key="_global_ticker", label_visibility="collapsed", placeholder="Ticker...",
    )
    if _tk:
        st.session_state["ticker"] = _tk.upper().strip()

if selected and selected != "---":
    _path = _MAP.get(selected)
    if _path:
        try:
            importlib.import_module(_path).render()
        except ImportError as _e:
            st.error(f"Could not load page **{selected}**: {_e}")
        except Exception as _e:
            st.error(f"Error rendering **{selected}**: {_e}")
            import traceback
            st.code(traceback.format_exc(), language="python")
