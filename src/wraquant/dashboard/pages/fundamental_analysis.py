"""Fundamental Analysis page -- financial statements, health score, DuPont.

Displays income statement trends, balance sheet composition, cash flow
analysis, financial health score gauge, DuPont decomposition waterfall,
earnings quality assessment, and ratio comparison tables.
"""

from __future__ import annotations

import streamlit as st


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_income(ticker: str, period: str) -> dict:
    from wraquant.fundamental.financials import income_analysis

    return income_analysis(ticker, period=period)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_balance(ticker: str, period: str) -> dict:
    from wraquant.fundamental.financials import balance_sheet_analysis

    return balance_sheet_analysis(ticker, period=period)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_cashflow(ticker: str, period: str) -> dict:
    from wraquant.fundamental.financials import cash_flow_analysis

    return cash_flow_analysis(ticker, period=period)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_health(ticker: str) -> dict:
    from wraquant.fundamental.financials import financial_health_score

    return financial_health_score(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_earnings_quality(ticker: str) -> dict:
    from wraquant.fundamental.financials import earnings_quality

    return earnings_quality(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_dupont(ticker: str) -> dict:
    from wraquant.fundamental.ratios import dupont_decomposition

    return dupont_decomposition(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_comprehensive_ratios(ticker: str) -> dict:
    from wraquant.fundamental.ratios import comprehensive_ratios

    return comprehensive_ratios(ticker)


def _make_trend_chart(dates: list, series_dict: dict, title: str, yaxis_fmt: str = ""):
    """Create a multi-line trend chart."""
    try:
        import plotly.graph_objects as go

        from wraquant.dashboard.components.charts import SERIES_COLORS, dark_layout
    except ImportError:
        return None

    fig = go.Figure()
    # Reverse so dates go chronologically left-to-right
    x = list(reversed(dates))
    for i, (name, values) in enumerate(series_dict.items()):
        y = list(reversed(values))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=name,
                line={"color": SERIES_COLORS[i % len(SERIES_COLORS)], "width": 2},
                marker={"size": 5},
            )
        )

    layout_kwargs = dark_layout(title=title)
    if yaxis_fmt == "pct":
        layout_kwargs["yaxis"] = {**layout_kwargs.get("yaxis", {}), "tickformat": ".1%"}
    elif yaxis_fmt == "dollar":
        layout_kwargs["yaxis"] = {**layout_kwargs.get("yaxis", {}), "tickprefix": "$"}
    fig.update_layout(**layout_kwargs)
    return fig


def _make_bar_chart(dates: list, series_dict: dict, title: str, stacked: bool = False):
    """Create a bar chart (optionally stacked)."""
    try:
        import plotly.graph_objects as go

        from wraquant.dashboard.components.charts import SERIES_COLORS, dark_layout
    except ImportError:
        return None

    fig = go.Figure()
    x = list(reversed(dates))
    for i, (name, values) in enumerate(series_dict.items()):
        y = list(reversed(values))
        fig.add_trace(
            go.Bar(
                x=x,
                y=y,
                name=name,
                marker_color=SERIES_COLORS[i % len(SERIES_COLORS)],
            )
        )

    layout_kwargs = dark_layout(title=title)
    if stacked:
        layout_kwargs["barmode"] = "stack"
    else:
        layout_kwargs["barmode"] = "group"
    fig.update_layout(**layout_kwargs)
    return fig


def _make_gauge(value: float, title: str, grade: str = ""):
    """Create a gauge chart for the health score."""
    try:
        import plotly.graph_objects as go

        from wraquant.dashboard.components.charts import COLORS, dark_layout
    except ImportError:
        return None

    # Color based on score
    if value >= 80:
        bar_color = COLORS["success"]
    elif value >= 60:
        bar_color = COLORS["info"]
    elif value >= 40:
        bar_color = COLORS["warning"]
    else:
        bar_color = COLORS["danger"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=value,
            number={"suffix": f"/100", "font": {"size": 40}},
            title={"text": f"{title} ({grade})", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748b"},
                "bar": {"color": bar_color, "thickness": 0.7},
                "bgcolor": "#1e1e28",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 20], "color": "rgba(239,68,68,0.15)"},
                    {"range": [20, 40], "color": "rgba(245,158,11,0.15)"},
                    {"range": [40, 60], "color": "rgba(234,179,8,0.15)"},
                    {"range": [60, 80], "color": "rgba(6,182,212,0.15)"},
                    {"range": [80, 100], "color": "rgba(34,197,94,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#e2e8f0", "width": 2},
                    "thickness": 0.8,
                    "value": value,
                },
            },
        )
    )
    layout = dark_layout(height=300)
    layout["margin"] = {"l": 30, "r": 30, "t": 60, "b": 20}
    fig.update_layout(**layout)
    return fig


def _make_waterfall(labels: list, values: list, title: str):
    """Create a waterfall chart for DuPont decomposition."""
    try:
        import plotly.graph_objects as go

        from wraquant.dashboard.components.charts import COLORS, dark_layout
    except ImportError:
        return None

    measures = ["relative"] * len(labels)
    measures[-1] = "total"

    fig = go.Figure(
        go.Waterfall(
            x=labels,
            y=values,
            measure=measures,
            connector={
                "line": {"color": COLORS["text_muted"], "width": 1, "dash": "dot"}
            },
            increasing={"marker": {"color": COLORS["success"]}},
            decreasing={"marker": {"color": COLORS["danger"]}},
            totals={"marker": {"color": COLORS["primary"]}},
            textposition="outside",
            text=[f"{v:.2%}" if abs(v) < 10 else f"{v:.1f}" for v in values],
        )
    )
    fig.update_layout(**dark_layout(title=title, height=400))
    return fig


def render() -> None:
    """Render the Fundamental Analysis page."""
    from wraquant.dashboard.components.metrics import fmt_currency, fmt_pct
    from wraquant.dashboard.components.sidebar import check_api_key

    ticker = st.session_state.get("ticker", "AAPL")
    st.markdown(f"# Fundamental Analysis: **{ticker}**")

    if not check_api_key():
        return

    period = st.radio("Period", ["annual", "quarter"], horizontal=True, key="fa_period")

    # -- Tabs for major sections -------------------------------------------

    tab_income, tab_balance, tab_cashflow, tab_health, tab_ratios = st.tabs(
        [
            "Income Statement",
            "Balance Sheet",
            "Cash Flow",
            "Health Score",
            "Ratios & DuPont",
        ]
    )

    # =====================================================================
    # TAB 1: Income Statement
    # =====================================================================
    with tab_income:
        with st.spinner("Loading income statement..."):
            try:
                inc = _fetch_income(ticker, period)
            except Exception as exc:
                st.error(f"Error: {exc}")
                inc = None

        if inc and inc.get("dates"):
            dates = inc["dates"]

            # Summary metrics
            c1, c2, c3, c4 = st.columns(4)
            rev = inc["revenue"]
            c1.metric("Revenue (Latest)", fmt_currency(rev[0]) if rev else "N/A")
            c2.metric("Rev CAGR (3Y)", fmt_pct(inc.get("revenue_cagr_3y", 0)))
            c3.metric(
                "Op Margin (Latest)",
                (
                    fmt_pct(inc["operating_margin"][0])
                    if inc["operating_margin"]
                    else "N/A"
                ),
            )
            c4.metric("Margin Trend", inc.get("margin_trend", "N/A").capitalize())

            # Revenue trend
            fig = _make_bar_chart(
                dates,
                {"Revenue": rev},
                "Revenue Trend",
            )
            if fig:
                fig.update_layout(yaxis_tickprefix="$")
                st.plotly_chart(fig, use_container_width=True)

            # Margin trends
            fig = _make_trend_chart(
                dates,
                {
                    "Gross Margin": inc["gross_margin"],
                    "Operating Margin": inc["operating_margin"],
                    "Net Margin": inc["net_margin"],
                    "EBITDA Margin": inc["ebitda_margin"],
                },
                "Margin Trends",
                yaxis_fmt="pct",
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # EPS trend
            if inc.get("eps"):
                fig = _make_bar_chart(dates, {"EPS": inc["eps"]}, "Earnings Per Share")
                if fig:
                    fig.update_layout(yaxis_tickprefix="$")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No income data available.")

    # =====================================================================
    # TAB 2: Balance Sheet
    # =====================================================================
    with tab_balance:
        with st.spinner("Loading balance sheet..."):
            try:
                bs = _fetch_balance(ticker, period)
            except Exception as exc:
                st.error(f"Error: {exc}")
                bs = None

        if bs and bs.get("dates"):
            dates = bs["dates"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Total Assets",
                fmt_currency(bs["total_assets"][0]) if bs["total_assets"] else "N/A",
            )
            c2.metric(
                "Total Debt",
                fmt_currency(bs["total_debt"][0]) if bs["total_debt"] else "N/A",
            )
            c3.metric(
                "Net Debt", fmt_currency(bs["net_debt"][0]) if bs["net_debt"] else "N/A"
            )
            c4.metric("Leverage Trend", bs.get("leverage_trend", "N/A").capitalize())

            # Stacked composition
            fig = _make_bar_chart(
                dates,
                {
                    "Equity": bs["total_equity"],
                    "Total Debt": bs["total_debt"],
                    "Cash": bs["cash"],
                },
                "Capital Structure",
                stacked=True,
            )
            if fig:
                fig.update_layout(yaxis_tickprefix="$")
                st.plotly_chart(fig, use_container_width=True)

            # Leverage ratios
            fig = _make_trend_chart(
                dates,
                {
                    "D/E Ratio": bs["debt_to_equity"],
                    "Current Ratio": bs["current_ratio"],
                },
                "Leverage & Liquidity Ratios",
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # Book value
            fig = _make_trend_chart(
                dates,
                {
                    "BVPS": bs["book_value_per_share"],
                    "Tangible BVPS": bs["tangible_bvps"],
                },
                "Book Value Per Share",
                yaxis_fmt="dollar",
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No balance sheet data available.")

    # =====================================================================
    # TAB 3: Cash Flow
    # =====================================================================
    with tab_cashflow:
        with st.spinner("Loading cash flow..."):
            try:
                cf = _fetch_cashflow(ticker, period)
            except Exception as exc:
                st.error(f"Error: {exc}")
                cf = None

        if cf and cf.get("dates"):
            dates = cf["dates"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "FCF (Latest)",
                (
                    fmt_currency(cf["free_cash_flow"][0])
                    if cf["free_cash_flow"]
                    else "N/A"
                ),
            )
            c2.metric(
                "FCF Margin",
                fmt_pct(cf["fcf_margin"][0]) if cf["fcf_margin"] else "N/A",
            )
            c3.metric("FCF Yield", fmt_pct(cf.get("fcf_yield", 0)))
            c4.metric(
                "Cash Conv.",
                f"{cf['cash_conversion'][0]:.2f}x" if cf["cash_conversion"] else "N/A",
            )

            # FCF trend
            fig = _make_bar_chart(
                dates,
                {
                    "Operating CF": cf["operating_cash_flow"],
                    "CapEx": cf["capital_expenditures"],
                    "Free CF": cf["free_cash_flow"],
                },
                "Cash Flow Trend",
            )
            if fig:
                fig.update_layout(yaxis_tickprefix="$")
                st.plotly_chart(fig, use_container_width=True)

            # FCF margin trend
            fig = _make_trend_chart(
                dates,
                {
                    "FCF Margin": cf["fcf_margin"],
                    "CapEx/Revenue": cf["capex_to_revenue"],
                },
                "Cash Flow Efficiency",
                yaxis_fmt="pct",
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # Shareholder returns
            if cf.get("dividends_paid") and any(d > 0 for d in cf["dividends_paid"]):
                fig = _make_bar_chart(
                    dates,
                    {
                        "Dividends": cf["dividends_paid"],
                        "Buybacks": cf["buybacks"],
                    },
                    "Shareholder Returns",
                    stacked=True,
                )
                if fig:
                    fig.update_layout(yaxis_tickprefix="$")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cash flow data available.")

    # =====================================================================
    # TAB 4: Health Score
    # =====================================================================
    with tab_health:
        with st.spinner("Computing health score..."):
            try:
                health = _fetch_health(ticker)
            except Exception as exc:
                st.error(f"Error: {exc}")
                health = None

        if health:
            col_gauge, col_detail = st.columns([1, 1])

            with col_gauge:
                fig = _make_gauge(
                    health.get("total_score", 0),
                    "Financial Health",
                    health.get("grade", ""),
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                st.markdown(
                    f"**Category:** {health.get('category', 'N/A').capitalize()}"
                )
                if health.get("piotroski_f_score") is not None:
                    st.markdown(
                        f"**Piotroski F-Score:** {health['piotroski_f_score']}/9"
                    )

            with col_detail:
                st.markdown("#### Sub-Scores")
                sub_scores = {
                    "Profitability": (health.get("profitability_score", 0), 30),
                    "Liquidity": (health.get("liquidity_score", 0), 15),
                    "Leverage": (health.get("leverage_score", 0), 20),
                    "Efficiency": (health.get("efficiency_score", 0), 15),
                    "Cash Flow": (health.get("cash_flow_score", 0), 20),
                }

                for name, (score, max_score) in sub_scores.items():
                    pct = score / max_score if max_score > 0 else 0
                    bar_color = (
                        "#22c55e"
                        if pct >= 0.7
                        else ("#f59e0b" if pct >= 0.4 else "#ef4444")
                    )
                    st.markdown(
                        f'<div style="margin-bottom:8px;">'
                        f'<div style="display:flex; justify-content:space-between; margin-bottom:2px;">'
                        f'<span style="font-size:0.9rem;">{name}</span>'
                        f'<span style="font-size:0.9rem; color:#94a3b8;">{score:.1f}/{max_score}</span>'
                        f"</div>"
                        f'<div style="background:#1e1e28; border-radius:4px; height:8px;">'
                        f'<div style="background:{bar_color}; border-radius:4px; height:8px; '
                        f'width:{pct*100:.0f}%;"></div>'
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )

                strengths = health.get("strengths", [])
                weaknesses = health.get("weaknesses", [])

                if strengths:
                    st.markdown("#### Strengths")
                    for s in strengths:
                        st.markdown(f"- {s}")
                if weaknesses:
                    st.markdown("#### Weaknesses")
                    for w in weaknesses:
                        st.markdown(f"- {w}")

            # Earnings quality
            st.divider()
            st.markdown("### Earnings Quality")
            with st.spinner("Assessing earnings quality..."):
                try:
                    eq = _fetch_earnings_quality(ticker)
                except Exception:
                    eq = None

            if eq:
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("Accruals Ratio", f"{eq.get('accruals_ratio', 0):.2%}")
                q2.metric(
                    "Cash Conversion", f"{eq.get('cash_conversion_ratio', 0):.2f}x"
                )
                q3.metric("Quality Grade", eq.get("quality_grade", "N/A"))
                q4.metric(
                    "Earnings Persistence", f"{eq.get('earnings_persistence', 0):.2f}"
                )

                fcf_ni = eq.get("fcf_to_net_income", 0)
                if fcf_ni:
                    st.metric("FCF / Net Income", f"{fcf_ni:.2f}x")

                flags = eq.get("red_flags", [])
                if flags:
                    st.warning("**Red Flags:** " + ", ".join(flags))
        else:
            st.info("Health score data unavailable.")

    # =====================================================================
    # TAB 5: Ratios & DuPont
    # =====================================================================
    with tab_ratios:
        col_ratios, col_dupont = st.columns(2)

        with col_ratios:
            st.markdown("### Comprehensive Ratios")
            with st.spinner("Computing ratios..."):
                try:
                    ratios = _fetch_comprehensive_ratios(ticker)
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    ratios = None

            if ratios:
                import pandas as pd

                # Build ratio tables by category
                for category in [
                    "profitability",
                    "liquidity",
                    "leverage",
                    "efficiency",
                    "valuation",
                    "growth",
                ]:
                    cat_data = ratios.get(category, {})
                    if cat_data and isinstance(cat_data, dict):
                        # Filter out 'period' key
                        display = {
                            k: v
                            for k, v in cat_data.items()
                            if k != "period" and isinstance(v, (int, float))
                        }
                        if display:
                            st.markdown(f"**{category.capitalize()}**")
                            df = pd.DataFrame(
                                [
                                    (k.replace("_", " ").title(), f"{v:.4f}")
                                    for k, v in display.items()
                                ],
                                columns=["Metric", "Value"],
                            )
                            st.dataframe(df, hide_index=True, use_container_width=True)

        with col_dupont:
            st.markdown("### DuPont Decomposition")
            with st.spinner("Computing DuPont..."):
                try:
                    dupont = _fetch_dupont(ticker)
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    dupont = None

            if dupont:
                d1, d2, d3 = st.columns(3)
                d1.metric("Net Margin", fmt_pct(dupont.get("net_margin", 0)))
                d2.metric("Asset Turnover", f"{dupont.get('asset_turnover', 0):.2f}x")
                d3.metric(
                    "Equity Multiplier", f"{dupont.get('equity_multiplier', 0):.2f}x"
                )

                roe = dupont.get("roe", 0)
                st.metric("ROE (DuPont)", fmt_pct(roe))

                # Waterfall: show how the three components combine into ROE
                net_margin = dupont.get("net_margin", 0)
                asset_turn = dupont.get("asset_turnover", 0)
                eq_mult = dupont.get("equity_multiplier", 0)

                fig = _make_waterfall(
                    ["Net Margin", "x Asset Turn.", "x Eq. Mult.", "= ROE"],
                    [net_margin, asset_turn, eq_mult, roe],
                    "DuPont ROE Decomposition",
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                # 5-way DuPont if available
                if dupont.get("tax_burden") is not None:
                    st.markdown("#### 5-Way Decomposition")
                    st.markdown(
                        f"- **Tax Burden** (NI/EBT): {dupont.get('tax_burden', 0):.4f}\n"
                        f"- **Interest Burden** (EBT/EBIT): {dupont.get('interest_burden', 0):.4f}\n"
                        f"- **EBIT Margin** (EBIT/Rev): {dupont.get('ebit_margin', 0):.4f}\n"
                        f"- **Asset Turnover** (Rev/Assets): {dupont.get('asset_turnover', 0):.4f}\n"
                        f"- **Equity Multiplier** (Assets/Equity): {dupont.get('equity_multiplier', 0):.4f}\n"
                    )
