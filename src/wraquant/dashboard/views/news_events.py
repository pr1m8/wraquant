"""News & Events page -- sentiment, earnings, insider trading, dividends.

Displays recent news with sentiment coloring, sentiment timeline,
earnings surprise history, insider trading activity, and upcoming
earnings date.
"""

from __future__ import annotations

import streamlit as st


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_sentiment(ticker: str) -> dict:
    from wraquant.news.sentiment import news_sentiment

    return news_sentiment(ticker, limit=50)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_earnings_history(ticker: str) -> dict:
    from wraquant.news.events import earnings_history

    return earnings_history(ticker, limit=20)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_upcoming(ticker: str) -> dict:
    from wraquant.news.events import upcoming_earnings

    return upcoming_earnings(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_insider(ticker: str) -> dict:
    from wraquant.news.events import insider_activity

    return insider_activity(ticker, limit=50)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_institutional(ticker: str) -> dict:
    from wraquant.news.events import institutional_ownership

    return institutional_ownership(ticker)


def render() -> None:
    """Render the News & Events page."""
    import pandas as pd

    from wraquant.dashboard.components.charts import COLORS, SERIES_COLORS, dark_layout
    from wraquant.dashboard.components.metrics import fmt_currency, fmt_pct
    from wraquant.dashboard.components.sidebar import check_api_key

    ticker = st.session_state.get("ticker", "AAPL")
    st.markdown(f"# News & Events: **{ticker}**")

    if not check_api_key():
        return

    tab_news, tab_earnings, tab_insider, tab_inst = st.tabs(
        [
            "News Sentiment",
            "Earnings History",
            "Insider Activity",
            "Institutional Ownership",
        ]
    )

    # =====================================================================
    # TAB 1: News Sentiment
    # =====================================================================
    with tab_news:
        with st.spinner("Analyzing news sentiment..."):
            try:
                sentiment = _fetch_sentiment(ticker)
            except Exception as exc:
                st.error(f"Error: {exc}")
                sentiment = None

        if sentiment:
            agg = sentiment.get("aggregate", {})
            weighted = agg.get("weighted_mean", 0)
            trend = sentiment.get("trend", "stable")
            bullish_pct = agg.get("bullish_pct", 0)
            bearish_pct = agg.get("bearish_pct", 0)
            neutral_pct = agg.get("neutral_pct", 0)
            article_count = sentiment.get("article_count", 0)
            engine = sentiment.get("engine", "keyword")

            # Sentiment header
            if weighted > 0.15:
                sent_label, sent_color = "BULLISH", COLORS["success"]
            elif weighted > 0.05:
                sent_label, sent_color = "SLIGHTLY BULLISH", "#86efac"
            elif weighted < -0.15:
                sent_label, sent_color = "BEARISH", COLORS["danger"]
            elif weighted < -0.05:
                sent_label, sent_color = "SLIGHTLY BEARISH", "#fca5a5"
            else:
                sent_label, sent_color = "NEUTRAL", COLORS["neutral"]

            s1, s2, s3, s4 = st.columns(4)
            s1.markdown(
                f'<div style="text-align:center; padding:1rem; background:#16161d; '
                f'border-radius:12px; border:1px solid {sent_color}40;">'
                f'<p style="color:#94a3b8; font-size:0.8rem; margin:0 0 4px 0;">Weighted Sentiment</p>'
                f'<p style="color:{sent_color}; font-size:2rem; font-weight:700; margin:0;">'
                f"{weighted:+.3f}</p>"
                f'<p style="color:{sent_color}; font-size:0.9rem; margin:4px 0 0 0;">{sent_label}</p>'
                f"</div>",
                unsafe_allow_html=True,
            )
            s2.metric("Trend", trend.capitalize())
            s3.metric("Articles", f"{article_count}")
            s4.metric("Engine", engine.capitalize())

            st.divider()

            # Sentiment breakdown bar
            st.markdown("### Sentiment Breakdown")
            b1, b2, b3 = st.columns(3)
            b1.metric("Bullish", fmt_pct(bullish_pct))
            b2.metric("Neutral", fmt_pct(neutral_pct))
            b3.metric("Bearish", fmt_pct(bearish_pct))

            # Sentiment breakdown visualization
            try:
                import plotly.graph_objects as go

                fig = go.Figure(
                    go.Bar(
                        x=[bullish_pct, neutral_pct, bearish_pct],
                        y=["Bullish", "Neutral", "Bearish"],
                        orientation="h",
                        marker_color=[
                            COLORS["success"],
                            COLORS["neutral"],
                            COLORS["danger"],
                        ],
                        text=[
                            fmt_pct(bullish_pct),
                            fmt_pct(neutral_pct),
                            fmt_pct(bearish_pct),
                        ],
                        textposition="outside",
                    )
                )
                fig.update_layout(
                    **dark_layout(
                        title="Sentiment Distribution",
                        xaxis_title="Percentage",
                        xaxis_tickformat=".0%",
                        height=200,
                        margin={"l": 80, "r": 60, "t": 50, "b": 40},
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                pass

            # Sentiment timeline from articles
            articles = sentiment.get("articles", [])
            if articles:
                st.divider()
                st.markdown("### Sentiment Timeline")

                art_df = pd.DataFrame(articles)
                if "date" in art_df.columns and "sentiment" in art_df.columns:
                    art_df["date"] = pd.to_datetime(art_df["date"], errors="coerce")
                    art_df = art_df.dropna(subset=["date"]).sort_values("date")

                    try:
                        import plotly.graph_objects as go

                        fig = go.Figure()
                        colors = [
                            (
                                COLORS["success"]
                                if s > 0.05
                                else (
                                    COLORS["danger"] if s < -0.05 else COLORS["neutral"]
                                )
                            )
                            for s in art_df["sentiment"]
                        ]
                        fig.add_trace(
                            go.Scatter(
                                x=art_df["date"],
                                y=art_df["sentiment"],
                                mode="markers",
                                marker={"color": colors, "size": 8, "opacity": 0.7},
                                text=art_df.get("title", ""),
                                hovertemplate="%{text}<br>Score: %{y:.3f}<extra></extra>",
                            )
                        )
                        # Add zero line
                        fig.add_hline(
                            y=0,
                            line_color=COLORS["text_muted"],
                            line_dash="dash",
                            opacity=0.5,
                        )
                        fig.update_layout(
                            **dark_layout(
                                title="Individual Article Sentiment Over Time",
                                yaxis_title="Sentiment Score",
                                height=350,
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        pass

                # Recent headlines
                st.divider()
                st.markdown("### Recent Headlines")
                for art in articles[:15]:
                    title = art.get("title", "")
                    source = art.get("source", "")
                    date_str = str(art.get("date", ""))[:10]
                    score = art.get("sentiment", 0)
                    url = art.get("url", "")

                    if score > 0.05:
                        badge_color = COLORS["success"]
                        badge_text = "+"
                    elif score < -0.05:
                        badge_color = COLORS["danger"]
                        badge_text = "-"
                    else:
                        badge_color = COLORS["neutral"]
                        badge_text = "~"

                    title_link = f"[{title}]({url})" if url else title
                    st.markdown(
                        f'<div style="padding:0.5rem 0; border-bottom:1px solid rgba(255,255,255,0.06);">'
                        f'<span style="display:inline-block; width:24px; height:24px; '
                        f"line-height:24px; text-align:center; border-radius:50%; "
                        f"background:{badge_color}30; color:{badge_color}; font-weight:700; "
                        f'font-size:0.8rem; margin-right:8px;">{badge_text}</span>'
                        f'<span style="color:#94a3b8; font-size:0.8rem;">{date_str} | {source}</span>'
                        f'<br/><span style="font-size:0.95rem;">{title}</span>'
                        f'<span style="color:#64748b; font-size:0.8rem; float:right;">{score:+.3f}</span>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No sentiment data available.")

    # =====================================================================
    # TAB 2: Earnings History
    # =====================================================================
    with tab_earnings:
        col_upcoming, col_history = st.columns([1, 2])

        with col_upcoming:
            st.markdown("### Upcoming Earnings")
            with st.spinner("Checking upcoming earnings..."):
                try:
                    upcoming = _fetch_upcoming(ticker)
                except Exception:
                    upcoming = None

            if upcoming and upcoming.get("next_date"):
                st.metric("Next Date", upcoming["next_date"])
                days_until = upcoming.get("days_until")
                if days_until is not None:
                    st.metric("Days Until", f"{days_until}")
                eps_est = upcoming.get("eps_estimate")
                if eps_est is not None:
                    st.metric("EPS Estimate", f"${eps_est:.2f}")
                rev_est = upcoming.get("revenue_estimate")
                if rev_est is not None:
                    st.metric("Rev Estimate", fmt_currency(rev_est))
                time_val = upcoming.get("time")
                if time_val:
                    time_label = {
                        "bmo": "Before Market Open",
                        "amc": "After Market Close",
                    }.get(time_val, time_val)
                    st.metric("Timing", time_label)
            else:
                st.info("No upcoming earnings scheduled.")

        with col_history:
            st.markdown("### Earnings Track Record")
            with st.spinner("Loading earnings history..."):
                try:
                    hist = _fetch_earnings_history(ticker)
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    hist = None

            if hist:
                h1, h2, h3, h4 = st.columns(4)
                h1.metric("Beat Rate", fmt_pct(hist.get("beat_rate", 0)))
                h2.metric("Avg Surprise", f"{hist.get('avg_surprise', 0):+.2%}")
                streak = hist.get("streak", {})
                h3.metric(
                    "Current Streak",
                    f"{streak.get('length', 0)} {streak.get('type', 'N/A')}s",
                )
                h4.metric(
                    "PEAD Signal",
                    hist.get("pead_signal", "N/A").replace("_", " ").title(),
                )

                # Earnings surprise bar chart
                surprises = hist.get("surprises")
                if (
                    surprises is not None
                    and not surprises.empty
                    and "surprise_pct" in surprises.columns
                ):
                    try:
                        import plotly.graph_objects as go

                        surprise_data = surprises.head(16).copy()
                        surprise_data = surprise_data.iloc[::-1]  # Chronological order

                        colors = [
                            COLORS["success"] if b else COLORS["danger"]
                            for b in surprise_data.get(
                                "beat", [False] * len(surprise_data)
                            )
                        ]

                        fig = go.Figure(
                            go.Bar(
                                x=surprise_data.get("date", range(len(surprise_data))),
                                y=surprise_data["surprise_pct"],
                                marker_color=colors,
                                text=[
                                    f"{v:+.1f}%" for v in surprise_data["surprise_pct"]
                                ],
                                textposition="outside",
                            )
                        )
                        fig.add_hline(
                            y=0, line_color=COLORS["text_muted"], line_dash="solid"
                        )
                        fig.update_layout(
                            **dark_layout(
                                title="Earnings Surprise History (%)",
                                yaxis_title="Surprise %",
                                height=350,
                            )
                        )
                        fig.update_xaxes(tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        pass

                    # Surprise table
                    display_cols = [
                        c
                        for c in ["date", "actual", "estimate", "surprise_pct", "beat"]
                        if c in surprises.columns
                    ]
                    if display_cols:
                        st.dataframe(
                            surprises[display_cols].head(12),
                            hide_index=True,
                            use_container_width=True,
                        )

    # =====================================================================
    # TAB 3: Insider Activity
    # =====================================================================
    with tab_insider:
        with st.spinner("Loading insider activity..."):
            try:
                insider = _fetch_insider(ticker)
            except Exception as exc:
                st.error(f"Error: {exc}")
                insider = None

        if insider:
            signal = insider.get("signal", "neutral")
            sig_colors = {
                "bullish": COLORS["success"],
                "bearish": COLORS["danger"],
                "neutral": COLORS["neutral"],
            }
            sig_color = sig_colors.get(signal, COLORS["neutral"])

            i1, i2, i3, i4, i5 = st.columns(5)
            i1.markdown(
                f'<div style="text-align:center; padding:0.6rem; background:#16161d; '
                f'border-radius:10px; border:1px solid {sig_color}40;">'
                f'<p style="color:#94a3b8; font-size:0.75rem; margin:0;">Signal</p>'
                f'<p style="color:{sig_color}; font-size:1.3rem; font-weight:700; margin:0;">'
                f"{signal.upper()}</p></div>",
                unsafe_allow_html=True,
            )
            i2.metric("Total Txns", insider.get("total_transactions", 0))
            i3.metric("Buys", insider.get("buy_count", 0))
            i4.metric("Sells", insider.get("sell_count", 0))
            bsr = insider.get("buy_sell_ratio", 0)
            i5.metric(
                "Buy/Sell Ratio", f"{bsr:.2f}" if bsr < float("inf") else "All Buys"
            )

            net_val = insider.get("net_value", 0)
            st.metric("Net Value", fmt_currency(net_val))

            # Notable trades
            notable = insider.get("notable_trades", [])
            if notable:
                st.markdown("#### Notable Trades (>$1M)")
                notable_df = pd.DataFrame(notable)
                display_cols = [
                    c
                    for c in [
                        "date",
                        "insider",
                        "transaction_type",
                        "shares",
                        "price",
                        "value",
                    ]
                    if c in notable_df.columns
                ]
                if display_cols:
                    st.dataframe(
                        notable_df[display_cols],
                        hide_index=True,
                        use_container_width=True,
                    )

            # All transactions
            txns = insider.get("transactions")
            if txns is not None and not txns.empty:
                with st.expander("All Insider Transactions"):
                    st.dataframe(
                        txns.head(50), hide_index=True, use_container_width=True
                    )
        else:
            st.info("No insider activity data available.")

    # =====================================================================
    # TAB 4: Institutional Ownership
    # =====================================================================
    with tab_inst:
        with st.spinner("Loading institutional ownership..."):
            try:
                inst = _fetch_institutional(ticker)
            except Exception as exc:
                st.error(f"Error: {exc}")
                inst = None

        if inst:
            io1, io2, io3, io4 = st.columns(4)
            io1.metric("Total Holders", inst.get("total_institutional_holders", 0))
            io2.metric("Shares Held", f"{inst.get('total_shares_held', 0):,}")
            io3.metric("Top Holder", inst.get("top_holder", "N/A"))
            io4.metric("Net Change", inst.get("net_change", "N/A").capitalize())

            conc = inst.get("concentration", 0)
            if conc > 0:
                st.metric("Concentration (HHI)", f"{conc:.4f}")

            holders = inst.get("holders")
            if holders is not None and not holders.empty:
                st.markdown("### Top Institutional Holders")
                st.dataframe(
                    holders.head(20), hide_index=True, use_container_width=True
                )

                # Pie chart of top holders
                try:
                    import plotly.graph_objects as go

                    top = holders.head(10)
                    if "holder" in top.columns and "shares" in top.columns:
                        fig = go.Figure(
                            go.Pie(
                                labels=top["holder"],
                                values=top["shares"],
                                hole=0.4,
                                marker_colors=SERIES_COLORS,
                                textposition="inside",
                                textinfo="percent+label",
                            )
                        )
                        fig.update_layout(
                            **dark_layout(
                                title="Top 10 Institutional Holders",
                                height=400,
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    pass
        else:
            st.info("No institutional ownership data available.")
