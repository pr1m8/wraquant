"""Screener page -- preset screens, custom criteria, results table.

Provides preset screens (Value, Growth, Quality, Piotroski, Magic Formula)
and a custom criteria builder.  Results are displayed in a sortable table
with the ability to click into any stock.
"""

from __future__ import annotations

import streamlit as st


@st.cache_data(ttl=300, show_spinner=False)
def _run_value_screen(
    max_pe: float, min_div: float, max_de: float, min_cap: int, limit: int
) -> "pd.DataFrame":
    from wraquant.fundamental.screening import value_screen

    return value_screen(
        max_pe=max_pe,
        min_dividend_yield=min_div,
        max_debt_equity=max_de,
        min_market_cap=min_cap,
        limit=limit,
    )


@st.cache_data(ttl=300, show_spinner=False)
def _run_growth_screen(
    min_rev_growth: float, min_cap: int, limit: int
) -> "pd.DataFrame":
    from wraquant.fundamental.screening import growth_screen

    return growth_screen(
        min_revenue_growth=min_rev_growth,
        min_market_cap=min_cap,
        limit=limit,
    )


@st.cache_data(ttl=300, show_spinner=False)
def _run_quality_screen(min_roe: float, max_de: float, limit: int) -> "pd.DataFrame":
    from wraquant.fundamental.screening import quality_screen

    return quality_screen(min_roe=min_roe, max_de=max_de, limit=limit)


@st.cache_data(ttl=300, show_spinner=False)
def _run_piotroski_screen(min_score: int, limit: int) -> "pd.DataFrame":
    from wraquant.fundamental.screening import piotroski_screen

    return piotroski_screen(min_score=min_score, limit=limit)


@st.cache_data(ttl=300, show_spinner=False)
def _run_magic_formula(top_n: int) -> "pd.DataFrame":
    from wraquant.fundamental.screening import magic_formula_screen

    return magic_formula_screen(top_n=top_n)


@st.cache_data(ttl=300, show_spinner=False)
def _run_custom_screen(
    min_cap: int,
    max_cap: int | None,
    min_price: float,
    max_price: float | None,
    sector: str,
    country: str,
    min_volume: int,
    min_dividend: float,
    limit: int,
) -> "pd.DataFrame":
    from wraquant.fundamental.screening import custom_screen

    criteria: dict = {"limit": limit}
    if min_cap > 0:
        criteria["min_market_cap"] = min_cap
    if max_cap and max_cap > 0:
        criteria["max_market_cap"] = max_cap
    if min_price > 0:
        criteria["min_price"] = min_price
    if max_price and max_price > 0:
        criteria["max_price"] = max_price
    if sector and sector != "Any":
        criteria["sector"] = sector
    if country and country != "Any":
        criteria["country"] = country
    if min_volume > 0:
        criteria["min_volume"] = min_volume
    if min_dividend > 0:
        criteria["min_dividend_yield"] = min_dividend

    return custom_screen(criteria=criteria)


def _display_results(df: "pd.DataFrame", title: str) -> None:
    """Display screening results with formatting."""
    import pandas as pd

    from wraquant.dashboard.components.metrics import fmt_currency

    if df.empty:
        st.info(f"No stocks matched the {title} criteria.")
        return

    st.markdown(f"**Found {len(df)} stocks**")

    # Select columns to display
    preferred_cols = [
        "symbol",
        "companyName",
        "price",
        "marketCap",
        "pe",
        "peRatio",
        "beta",
        "lastAnnualDividend",
        "sector",
        "industry",
        "country",
        "volume",
        "exchange",
        "revenue_growth",
        "f_score",
        "earnings_yield",
        "roic",
    ]
    available_cols = [c for c in preferred_cols if c in df.columns]
    if not available_cols:
        available_cols = list(df.columns[:10])

    display_df = df[available_cols].copy()

    # Format market cap
    if "marketCap" in display_df.columns:
        display_df["marketCap"] = display_df["marketCap"].apply(
            lambda x: fmt_currency(x) if pd.notna(x) and x > 0 else "N/A"
        )

    # Format price
    if "price" in display_df.columns:
        display_df["price"] = display_df["price"].apply(
            lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A"
        )

    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        height=min(len(display_df) * 35 + 38, 600),
    )

    # Click to drill in
    st.markdown("---")
    st.markdown(
        "**Drill into a stock:** select a symbol and switch to another page for deep analysis."
    )
    if "symbol" in df.columns:
        symbols = df["symbol"].tolist()
        selected = st.selectbox(
            "Select symbol to analyze",
            symbols,
            key=f"screener_drill_{title}",
        )
        if st.button(f"Analyze {selected}", key=f"screener_btn_{title}"):
            st.session_state["ticker"] = selected
            st.rerun()


def render() -> None:
    """Render the Screener page."""
    from wraquant.dashboard.components.charts import COLORS
    from wraquant.dashboard.components.sidebar import check_api_key

    st.markdown("# Stock Screener")
    st.markdown(
        "Pre-built screens and custom criteria to find investment opportunities."
    )

    if not check_api_key():
        return

    tab_presets, tab_custom = st.tabs(["Preset Screens", "Custom Screen"])

    # =====================================================================
    # TAB 1: Preset Screens
    # =====================================================================
    with tab_presets:
        # Screen selector
        screen_type = st.selectbox(
            "Select Strategy",
            [
                "Value (Graham)",
                "Growth",
                "Quality (Buffett)",
                "Piotroski F-Score",
                "Magic Formula (Greenblatt)",
            ],
            key="screen_type",
        )

        limit = st.slider("Max Results", 10, 100, 30, key="preset_limit")

        # -- Value Screen --
        if screen_type == "Value (Graham)":
            st.markdown(
                '<div style="background:#16161d; border-radius:10px; padding:1rem; '
                'border:1px solid rgba(255,255,255,0.06); margin-bottom:1rem;">'
                '<p style="font-weight:600; margin:0 0 4px 0;">Value Screen</p>'
                '<p style="color:#94a3b8; font-size:0.85rem; margin:0;">'
                "Classic Ben Graham: low P/E, decent dividend yield, manageable debt. "
                "Identifies stocks trading below their intrinsic value.</p>"
                "</div>",
                unsafe_allow_html=True,
            )

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                max_pe = st.number_input(
                    "Max P/E", 1.0, 50.0, 20.0, step=1.0, key="vs_pe"
                )
            with col2:
                min_div = (
                    st.number_input(
                        "Min Div Yield (%)", 0.0, 10.0, 2.0, step=0.5, key="vs_div"
                    )
                    / 100
                )
            with col3:
                max_de = st.number_input(
                    "Max D/E", 0.1, 5.0, 1.5, step=0.1, key="vs_de"
                )
            with col4:
                min_cap = st.number_input(
                    "Min Mkt Cap ($B)", 0.1, 100.0, 1.0, step=0.5, key="vs_cap"
                )
                min_cap_int = int(min_cap * 1e9)

            if st.button("Run Value Screen", type="primary", key="btn_value"):
                with st.spinner("Screening..."):
                    try:
                        results = _run_value_screen(
                            max_pe, min_div, max_de, min_cap_int, limit
                        )
                        _display_results(results, "Value")
                    except Exception as exc:
                        st.error(f"Screen failed: {exc}")

        # -- Growth Screen --
        elif screen_type == "Growth":
            st.markdown(
                '<div style="background:#16161d; border-radius:10px; padding:1rem; '
                'border:1px solid rgba(255,255,255,0.06); margin-bottom:1rem;">'
                '<p style="font-weight:600; margin:0 0 4px 0;">Growth Screen</p>'
                '<p style="color:#94a3b8; font-size:0.85rem; margin:0;">'
                "High revenue growth companies with positive momentum. "
                "Best for bull markets and sector rotation strategies.</p>"
                "</div>",
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)
            with col1:
                min_rev_growth = (
                    st.number_input(
                        "Min Rev Growth (%)",
                        5.0,
                        100.0,
                        15.0,
                        step=5.0,
                        key="gs_growth",
                    )
                    / 100
                )
            with col2:
                min_cap = st.number_input(
                    "Min Mkt Cap ($B)", 0.1, 100.0, 0.5, step=0.5, key="gs_cap"
                )
                min_cap_int = int(min_cap * 1e9)

            if st.button("Run Growth Screen", type="primary", key="btn_growth"):
                with st.spinner("Screening (this may take a minute)..."):
                    try:
                        results = _run_growth_screen(min_rev_growth, min_cap_int, limit)
                        _display_results(results, "Growth")
                    except Exception as exc:
                        st.error(f"Screen failed: {exc}")

        # -- Quality Screen --
        elif screen_type == "Quality (Buffett)":
            st.markdown(
                '<div style="background:#16161d; border-radius:10px; padding:1rem; '
                'border:1px solid rgba(255,255,255,0.06); margin-bottom:1rem;">'
                '<p style="font-weight:600; margin:0 0 4px 0;">Quality Screen</p>'
                '<p style="color:#94a3b8; font-size:0.85rem; margin:0;">'
                "Buffett-style: high ROE, low leverage, durable competitive advantages. "
                "Quality factors have delivered consistent risk-adjusted returns.</p>"
                "</div>",
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)
            with col1:
                min_roe = (
                    st.number_input(
                        "Min ROE (%)", 5.0, 50.0, 15.0, step=1.0, key="qs_roe"
                    )
                    / 100
                )
            with col2:
                max_de = st.number_input(
                    "Max D/E", 0.1, 5.0, 1.0, step=0.1, key="qs_de"
                )

            if st.button("Run Quality Screen", type="primary", key="btn_quality"):
                with st.spinner("Screening..."):
                    try:
                        results = _run_quality_screen(min_roe, max_de, limit)
                        _display_results(results, "Quality")
                    except Exception as exc:
                        st.error(f"Screen failed: {exc}")

        # -- Piotroski Screen --
        elif screen_type == "Piotroski F-Score":
            st.markdown(
                '<div style="background:#16161d; border-radius:10px; padding:1rem; '
                'border:1px solid rgba(255,255,255,0.06); margin-bottom:1rem;">'
                '<p style="font-weight:600; margin:0 0 4px 0;">Piotroski F-Score Screen</p>'
                '<p style="color:#94a3b8; font-size:0.85rem; margin:0;">'
                "Academic screen using the 9-point F-Score (Piotroski, 2000). "
                "Separates financially healthy value stocks from value traps.</p>"
                "</div>",
                unsafe_allow_html=True,
            )

            min_fscore = st.slider("Min F-Score", 1, 9, 7, key="ps_fscore")

            if st.button("Run Piotroski Screen", type="primary", key="btn_piotroski"):
                with st.spinner("Screening (this may take a few minutes)..."):
                    try:
                        results = _run_piotroski_screen(min_fscore, limit)
                        _display_results(results, "Piotroski")
                    except Exception as exc:
                        st.error(f"Screen failed: {exc}")

        # -- Magic Formula --
        elif screen_type == "Magic Formula (Greenblatt)":
            st.markdown(
                '<div style="background:#16161d; border-radius:10px; padding:1rem; '
                'border:1px solid rgba(255,255,255,0.06); margin-bottom:1rem;">'
                '<p style="font-weight:600; margin:0 0 4px 0;">Magic Formula Screen</p>'
                '<p style="color:#94a3b8; font-size:0.85rem; margin:0;">'
                "Greenblatt's Magic Formula: rank by ROIC + earnings yield, buy top-ranked. "
                "Combines quality (high ROIC) and value (high earnings yield).</p>"
                "</div>",
                unsafe_allow_html=True,
            )

            if st.button("Run Magic Formula", type="primary", key="btn_magic"):
                with st.spinner("Screening (this may take a few minutes)..."):
                    try:
                        results = _run_magic_formula(limit)
                        _display_results(results, "Magic Formula")
                    except Exception as exc:
                        st.error(f"Screen failed: {exc}")

        st.divider()
        st.markdown(
            '<div style="background:#16161d; border-radius:10px; padding:1rem; '
            'border:1px solid rgba(255,255,255,0.06);">'
            '<p style="color:#94a3b8; font-size:0.85rem; margin:0;">'
            "<strong>Tip:</strong> After running a screen, select any symbol from the results "
            'and click "Analyze" to switch it into the global ticker for deep analysis '
            "on the Fundamental, Valuation, and other pages.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    # =====================================================================
    # TAB 2: Custom Screen
    # =====================================================================
    with tab_custom:
        st.markdown("### Custom Criteria Builder")
        st.markdown(
            "Build your own screen by combining market cap, price, sector, and other filters."
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            min_cap = st.number_input(
                "Min Market Cap ($B)", 0.0, 1000.0, 1.0, step=0.5, key="cs_min_cap"
            )
            max_cap = st.number_input(
                "Max Market Cap ($B)",
                0.0,
                5000.0,
                0.0,
                step=10.0,
                key="cs_max_cap",
                help="0 = no limit",
            )
            min_price = st.number_input(
                "Min Price ($)", 0.0, 10000.0, 1.0, step=1.0, key="cs_min_price"
            )
            max_price = st.number_input(
                "Max Price ($)",
                0.0,
                100000.0,
                0.0,
                step=10.0,
                key="cs_max_price",
                help="0 = no limit",
            )

        with col2:
            sector = st.selectbox(
                "Sector",
                [
                    "Any",
                    "Technology",
                    "Healthcare",
                    "Financial Services",
                    "Consumer Cyclical",
                    "Consumer Defensive",
                    "Industrials",
                    "Energy",
                    "Basic Materials",
                    "Real Estate",
                    "Utilities",
                    "Communication Services",
                ],
                key="cs_sector",
            )
            country = st.selectbox(
                "Country",
                ["Any", "US", "GB", "CA", "DE", "FR", "JP", "CN", "IN", "AU"],
                key="cs_country",
            )

        with col3:
            min_volume = st.number_input(
                "Min Avg Volume", 0, 100_000_000, 100_000, step=100_000, key="cs_volume"
            )
            min_dividend = (
                st.number_input(
                    "Min Div Yield (%)", 0.0, 20.0, 0.0, step=0.5, key="cs_div"
                )
                / 100
            )
            custom_limit = st.slider("Max Results", 10, 200, 50, key="cs_limit")

        if st.button("Run Custom Screen", type="primary", key="btn_custom"):
            with st.spinner("Screening..."):
                try:
                    results = _run_custom_screen(
                        min_cap=int(min_cap * 1e9),
                        max_cap=int(max_cap * 1e9) if max_cap > 0 else None,
                        min_price=min_price,
                        max_price=max_price if max_price > 0 else None,
                        sector=sector,
                        country=country,
                        min_volume=min_volume,
                        min_dividend=min_dividend,
                        limit=custom_limit,
                    )
                    _display_results(results, "Custom")
                except Exception as exc:
                    st.error(f"Screen failed: {exc}")
