"""Valuation page -- DCF, Graham Number, relative valuation, margin of safety.

Interactive valuation models with adjustable inputs via sliders.
"""

from __future__ import annotations

import streamlit as st


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_dcf(
    ticker: str, growth: float, discount: float, terminal: float, years: int
) -> dict:
    from wraquant.fundamental.valuation import dcf_valuation

    return dcf_valuation(
        ticker,
        growth_rate=growth,
        discount_rate=discount,
        terminal_growth=terminal,
        projection_years=years,
    )


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_graham(ticker: str) -> dict:
    from wraquant.fundamental.valuation import graham_number

    return graham_number(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_relative(ticker: str, peers: tuple) -> dict:
    from wraquant.fundamental.valuation import relative_valuation

    return relative_valuation(ticker, peers=list(peers) if peers else None)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_lynch(ticker: str) -> dict:
    from wraquant.fundamental.valuation import peter_lynch_value

    return peter_lynch_value(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_ddm(ticker: str) -> dict:
    from wraquant.fundamental.valuation import dividend_discount_model

    return dividend_discount_model(ticker)


def render() -> None:
    """Render the Valuation page."""
    from wraquant.dashboard.components.charts import COLORS, SERIES_COLORS, dark_layout
    from wraquant.dashboard.components.metrics import fmt_currency, fmt_pct
    from wraquant.dashboard.components.sidebar import check_api_key

    ticker = st.session_state.get("ticker", "AAPL")
    st.markdown(f"# Valuation: **{ticker}**")

    if not check_api_key():
        return

    tab_dcf, tab_graham, tab_relative, tab_mos = st.tabs(
        [
            "DCF Model",
            "Graham & Lynch",
            "Relative Valuation",
            "Margin of Safety",
        ]
    )

    # =====================================================================
    # TAB 1: DCF Model
    # =====================================================================
    with tab_dcf:
        st.markdown("### Discounted Cash Flow Valuation")
        st.markdown("Adjust the sliders to see how assumptions affect intrinsic value.")

        col_inputs, col_results = st.columns([1, 2])

        with col_inputs:
            growth_rate = (
                st.slider(
                    "FCF Growth Rate (%)",
                    min_value=-10.0,
                    max_value=30.0,
                    value=8.0,
                    step=0.5,
                    key="dcf_growth",
                )
                / 100.0
            )

            discount_rate = (
                st.slider(
                    "Discount Rate / WACC (%)",
                    min_value=5.0,
                    max_value=20.0,
                    value=10.0,
                    step=0.5,
                    key="dcf_discount",
                )
                / 100.0
            )

            terminal_growth = (
                st.slider(
                    "Terminal Growth Rate (%)",
                    min_value=0.0,
                    max_value=5.0,
                    value=2.5,
                    step=0.25,
                    key="dcf_terminal",
                )
                / 100.0
            )

            projection_years = st.slider(
                "Projection Years",
                min_value=3,
                max_value=10,
                value=5,
                key="dcf_years",
            )

            if discount_rate <= terminal_growth:
                st.error("Discount rate must exceed terminal growth rate.")
                return

        with col_results:
            with st.spinner("Computing DCF..."):
                try:
                    dcf = _fetch_dcf(
                        ticker,
                        growth_rate,
                        discount_rate,
                        terminal_growth,
                        projection_years,
                    )
                except Exception as exc:
                    st.error(f"DCF Error: {exc}")
                    dcf = None

            if dcf:
                intrinsic = dcf.get("intrinsic_value_per_share", 0)
                price = dcf.get("current_price", 0)
                mos = dcf.get("margin_of_safety", 0)
                upside = dcf.get("upside_potential", 0)
                fmp_dcf = dcf.get("fmp_dcf", 0)

                # Verdict color
                if mos > 0.20:
                    verdict, vcolor = "Undervalued", COLORS["success"]
                elif mos > 0:
                    verdict, vcolor = "Slightly Undervalued", "#86efac"
                elif mos > -0.15:
                    verdict, vcolor = "Fairly Valued", COLORS["warning"]
                else:
                    verdict, vcolor = "Overvalued", COLORS["danger"]

                # Metrics row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Intrinsic Value", f"${intrinsic:,.2f}")
                m2.metric("Current Price", f"${price:,.2f}")
                m3.metric("Margin of Safety", fmt_pct(mos))
                m4.metric("Upside", fmt_pct(upside))

                st.markdown(
                    f'<div style="text-align:center; padding:0.8rem; background:#16161d; '
                    f'border-radius:10px; border:1px solid rgba(255,255,255,0.06); margin:0.5rem 0;">'
                    f'<span style="color:{vcolor}; font-size:1.3rem; font-weight:600;">{verdict}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Projected FCF bar chart
                projected_fcf = dcf.get("projected_fcf", [])
                if projected_fcf:
                    try:
                        import plotly.graph_objects as go

                        years_labels = [
                            f"Year {i+1}" for i in range(len(projected_fcf))
                        ]
                        fig = go.Figure(
                            go.Bar(
                                x=years_labels,
                                y=projected_fcf,
                                marker_color=SERIES_COLORS[0],
                                text=[fmt_currency(v) for v in projected_fcf],
                                textposition="outside",
                            )
                        )
                        fig.update_layout(
                            **dark_layout(
                                title="Projected Free Cash Flows",
                                yaxis_title="FCF ($)",
                                height=350,
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        pass

                # Value composition
                pv_cf = dcf.get("pv_cash_flows", 0)
                pv_term = dcf.get("pv_terminal", 0)
                terminal_pct = dcf.get("terminal_pct", 0)

                st.markdown("#### Value Composition")
                vc1, vc2, vc3 = st.columns(3)
                vc1.metric("PV of Cash Flows", fmt_currency(pv_cf))
                vc2.metric("PV of Terminal Value", fmt_currency(pv_term))
                vc3.metric("Terminal % of Total", fmt_pct(terminal_pct))

                if terminal_pct > 0.75:
                    st.warning(
                        "Terminal value accounts for >75% of total value. "
                        "The valuation is highly sensitive to terminal assumptions."
                    )

                if fmp_dcf > 0:
                    st.info(
                        f"FMP's own DCF estimate: **${fmp_dcf:,.2f}** (for comparison)"
                    )

    # =====================================================================
    # TAB 2: Graham Number & Peter Lynch
    # =====================================================================
    with tab_graham:
        col_graham, col_lynch = st.columns(2)

        with col_graham:
            st.markdown("### Graham Number")
            st.caption("Conservative intrinsic value = sqrt(22.5 x EPS x BVPS)")

            with st.spinner("Computing Graham Number..."):
                try:
                    gn = _fetch_graham(ticker)
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    gn = None

            if gn:
                graham_val = gn.get("graham_number", 0)
                price = gn.get("current_price", 0)
                mos = gn.get("margin_of_safety", 0)

                st.metric(
                    "Graham Number", f"${graham_val:,.2f}" if graham_val > 0 else "N/A"
                )
                st.metric("Current Price", f"${price:,.2f}")
                st.metric("Margin of Safety", fmt_pct(mos))
                st.metric("EPS", f"${gn.get('eps', 0):.2f}")
                st.metric("Book Value/Share", f"${gn.get('bvps', 0):.2f}")

                if graham_val > 0:
                    # Price vs Graham chart
                    try:
                        import plotly.graph_objects as go

                        fig = go.Figure()
                        fig.add_trace(
                            go.Bar(
                                x=["Current Price", "Graham Number"],
                                y=[price, graham_val],
                                marker_color=[
                                    (
                                        COLORS["danger"]
                                        if price > graham_val
                                        else COLORS["success"]
                                    ),
                                    COLORS["primary"],
                                ],
                                text=[f"${price:,.2f}", f"${graham_val:,.2f}"],
                                textposition="outside",
                            )
                        )
                        fig.update_layout(
                            **dark_layout(
                                title="Price vs Graham Number",
                                height=300,
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        pass

        with col_lynch:
            st.markdown("### Peter Lynch Fair Value")
            st.caption("PEG-based: stock is fairly valued when P/E = EPS growth rate")

            with st.spinner("Computing Lynch value..."):
                try:
                    plv = _fetch_lynch(ticker)
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    plv = None

            if plv:
                fair_val = plv.get("fair_value", 0)
                price = plv.get("current_price", 0)
                peg = plv.get("peg_ratio", 0)
                category = plv.get("lynch_category", "N/A")

                st.metric(
                    "Lynch Fair Value", f"${fair_val:,.2f}" if fair_val > 0 else "N/A"
                )
                st.metric("Current Price", f"${price:,.2f}")
                st.metric("PEG Ratio", f"{peg:.2f}" if peg > 0 else "N/A")
                st.metric("P/E Ratio", f"{plv.get('pe_ratio', 0):.1f}")
                st.metric("EPS Growth", fmt_pct(plv.get("eps_growth_rate", 0)))

                # Category badge
                cat_colors = {
                    "undervalued": COLORS["success"],
                    "fairly valued": COLORS["warning"],
                    "overvalued": COLORS["danger"],
                }
                cat_color = cat_colors.get(category, COLORS["neutral"])
                st.markdown(
                    f'<div style="text-align:center; padding:0.5rem; background:#16161d; '
                    f'border-radius:8px; border:1px solid {cat_color}40;">'
                    f'<span style="color:{cat_color}; font-weight:600;">'
                    f"{category.upper()}</span></div>",
                    unsafe_allow_html=True,
                )

        # DDM
        st.divider()
        st.markdown("### Dividend Discount Model (Gordon Growth)")
        with st.spinner("Computing DDM..."):
            try:
                ddm = _fetch_ddm(ticker)
            except Exception:
                ddm = None

        if ddm and ddm.get("fair_value", 0) > 0:
            dd1, dd2, dd3, dd4 = st.columns(4)
            dd1.metric("DDM Fair Value", f"${ddm.get('fair_value', 0):,.2f}")
            dd2.metric("Current Dividend", f"${ddm.get('current_dividend', 0):.2f}")
            dd3.metric("Dividend Growth", fmt_pct(ddm.get("dividend_growth_rate", 0)))
            dd4.metric("Margin of Safety", fmt_pct(ddm.get("margin_of_safety", 0)))
        else:
            st.info(
                "DDM not applicable (company may not pay dividends or insufficient data)."
            )

    # =====================================================================
    # TAB 3: Relative Valuation
    # =====================================================================
    with tab_relative:
        st.markdown("### Relative Valuation vs Peers")
        peers_input = st.text_input(
            "Peer Symbols (comma-separated)",
            value="",
            placeholder="e.g. MSFT, GOOG, META",
            key="rel_peers",
        )
        peers = (
            tuple(s.strip().upper() for s in peers_input.split(",") if s.strip())
            if peers_input.strip()
            else ()
        )

        with st.spinner("Computing relative valuation..."):
            try:
                rv = _fetch_relative(ticker, peers)
            except Exception as exc:
                st.error(f"Error: {exc}")
                rv = None

        if rv:
            multiples = rv.get("multiples", {})
            peer_medians = rv.get("peer_medians", {})
            premium_disc = rv.get("premium_discount", {})
            verdict = rv.get("verdict", "")

            # Verdict
            v_colors = {
                "undervalued": COLORS["success"],
                "fairly valued": COLORS["warning"],
                "overvalued": COLORS["danger"],
            }
            v_color = v_colors.get(verdict, COLORS["neutral"])
            st.markdown(
                f'<div style="text-align:center; padding:0.8rem; background:#16161d; '
                f'border-radius:10px; border:1px solid rgba(255,255,255,0.06); margin-bottom:1rem;">'
                f'<span style="color:{v_color}; font-size:1.3rem; font-weight:600;">'
                f"{verdict.upper()}</span> vs peers</div>",
                unsafe_allow_html=True,
            )

            # Multiples comparison bar chart
            try:
                import plotly.graph_objects as go

                metrics = ["pe_ratio", "pb_ratio", "ps_ratio", "ev_to_ebitda"]
                labels = ["P/E", "P/B", "P/S", "EV/EBITDA"]

                target_vals = [multiples.get(m, 0) for m in metrics]
                median_vals = [peer_medians.get(m, 0) for m in metrics]

                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=labels,
                        y=target_vals,
                        name=ticker,
                        marker_color=COLORS["primary"],
                    )
                )
                fig.add_trace(
                    go.Bar(
                        x=labels,
                        y=median_vals,
                        name="Peer Median",
                        marker_color=COLORS["neutral"],
                    )
                )
                fig.update_layout(
                    **dark_layout(
                        title=f"{ticker} vs Peer Median Multiples",
                        barmode="group",
                        height=400,
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                pass

            # Premium/Discount table
            if premium_disc:
                import pandas as pd

                pd_data = []
                for metric, pct in premium_disc.items():
                    label = metric.replace("_", " ").upper()
                    pd_data.append(
                        {
                            "Multiple": label,
                            f"{ticker}": f"{multiples.get(metric, 0):.2f}",
                            "Peer Median": f"{peer_medians.get(metric, 0):.2f}",
                            "Premium/Discount": f"{pct:+.1%}",
                        }
                    )
                df = pd.DataFrame(pd_data)
                st.dataframe(df, hide_index=True, use_container_width=True)

            # Peer details
            peers_data = rv.get("peers_data", [])
            if peers_data:
                st.markdown("#### Individual Peer Multiples")
                import pandas as pd

                peer_rows = []
                for p in peers_data:
                    peer_rows.append(
                        {
                            "Symbol": p.get("symbol", ""),
                            "P/E": f"{p.get('pe_ratio', 0):.1f}",
                            "P/B": f"{p.get('pb_ratio', 0):.2f}",
                            "P/S": f"{p.get('ps_ratio', 0):.2f}",
                            "EV/EBITDA": f"{p.get('ev_to_ebitda', 0):.1f}",
                        }
                    )
                st.dataframe(
                    pd.DataFrame(peer_rows), hide_index=True, use_container_width=True
                )

    # =====================================================================
    # TAB 4: Margin of Safety Summary
    # =====================================================================
    with tab_mos:
        st.markdown("### Margin of Safety Summary")
        st.markdown(
            "Compares current price against multiple intrinsic value estimates "
            "to quantify how much downside protection you have."
        )

        mos_data = []

        # Collect all valuation estimates
        try:
            dcf_default = _fetch_dcf(ticker, 0.08, 0.10, 0.025, 5)
            if dcf_default:
                mos_data.append(
                    {
                        "Model": "DCF (8% growth, 10% WACC)",
                        "Intrinsic Value": dcf_default.get(
                            "intrinsic_value_per_share", 0
                        ),
                        "Margin of Safety": dcf_default.get("margin_of_safety", 0),
                    }
                )
        except Exception:
            pass

        try:
            gn = _fetch_graham(ticker)
            if gn and gn.get("graham_number", 0) > 0:
                mos_data.append(
                    {
                        "Model": "Graham Number",
                        "Intrinsic Value": gn.get("graham_number", 0),
                        "Margin of Safety": gn.get("margin_of_safety", 0),
                    }
                )
        except Exception:
            pass

        try:
            plv = _fetch_lynch(ticker)
            if plv and plv.get("fair_value", 0) > 0:
                mos_data.append(
                    {
                        "Model": "Peter Lynch",
                        "Intrinsic Value": plv.get("fair_value", 0),
                        "Margin of Safety": plv.get("margin_of_safety", 0),
                    }
                )
        except Exception:
            pass

        try:
            ddm = _fetch_ddm(ticker)
            if ddm and ddm.get("fair_value", 0) > 0:
                mos_data.append(
                    {
                        "Model": "DDM (Gordon Growth)",
                        "Intrinsic Value": ddm.get("fair_value", 0),
                        "Margin of Safety": ddm.get("margin_of_safety", 0),
                    }
                )
        except Exception:
            pass

        if mos_data:
            import pandas as pd

            # Summary chart
            try:
                import plotly.graph_objects as go

                current_price = mos_data[0].get("Intrinsic Value", 0) * (
                    1 - mos_data[0].get("Margin of Safety", 0)
                )

                models = [d["Model"] for d in mos_data]
                values = [d["Intrinsic Value"] for d in mos_data]

                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=models,
                        y=values,
                        name="Intrinsic Value",
                        marker_color=[
                            (
                                COLORS["success"]
                                if d["Margin of Safety"] > 0
                                else COLORS["danger"]
                            )
                            for d in mos_data
                        ],
                        text=[f"${v:,.2f}" for v in values],
                        textposition="outside",
                    )
                )
                # Add price line
                fig.add_hline(
                    y=current_price,
                    line_dash="dash",
                    line_color=COLORS["warning"],
                    annotation_text=f"Price: ${current_price:,.2f}",
                    annotation_position="top right",
                )
                fig.update_layout(
                    **dark_layout(
                        title="Intrinsic Value Estimates vs Current Price",
                        height=400,
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                pass

            # Table
            df = pd.DataFrame(mos_data)
            df["Intrinsic Value"] = df["Intrinsic Value"].apply(lambda x: f"${x:,.2f}")
            df["Margin of Safety"] = df["Margin of Safety"].apply(lambda x: f"{x:+.1%}")
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.info("Unable to compute any valuation estimates.")
