"""Home / Overview page -- company snapshot, sparklines, key metrics.

Displays a company profile card, key financial metrics, sparkline
charts for price and volume, sentiment indicator, and analyst ratings.
"""

from __future__ import annotations

import streamlit as st


def _get_fmp_client():
    """Create an FMPClient instance."""
    from wraquant.data.providers.fmp import FMPClient

    return FMPClient()


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_profile(ticker: str) -> dict:
    try:
        client = _get_fmp_client()
        return client.company_profile(ticker)
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_quote(ticker: str) -> dict:
    try:
        client = _get_fmp_client()
        return client.quote(ticker)
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_rating(ticker: str) -> dict:
    try:
        client = _get_fmp_client()
        return client.rating(ticker)
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_historical(ticker: str) -> "pd.DataFrame":
    """Fetch 6 months of daily price data for sparklines."""
    from datetime import datetime, timedelta

    import pandas as pd

    try:
        client = _get_fmp_client()
        end = datetime.now()
        start = end - timedelta(days=180)
        df = client.historical_price(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="daily",
        )
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
            elif not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except (ValueError, TypeError):
                    pass
            return df
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_sentiment_quick(ticker: str) -> dict:
    try:
        from wraquant.news.sentiment import news_sentiment

        return news_sentiment(ticker, limit=20)
    except Exception:
        return {}


def render() -> None:
    """Render the Home / Overview page."""
    from wraquant.dashboard.components.charts import COLORS, dark_layout
    from wraquant.dashboard.components.metrics import fmt_currency, fmt_number, fmt_pct
    from wraquant.dashboard.components.sidebar import check_api_key

    ticker = st.session_state.get("ticker", "AAPL")

    st.markdown("# wraquant Dashboard")
    st.caption("Interactive quantitative finance analysis")

    if not check_api_key():
        return

    # -- Fetch data --------------------------------------------------------

    with st.spinner(f"Loading {ticker} data..."):
        profile = _fetch_profile(ticker)
        quote = _fetch_quote(ticker)
        hist_df = _fetch_historical(ticker)

    if not profile and not quote:
        st.warning(
            f"No data found for **{ticker}**. Check the symbol and try again."
        )
        return

    # -- Company header ----------------------------------------------------

    col_name, col_price = st.columns([3, 1])
    with col_name:
        company_name = profile.get("companyName", ticker) if profile else ticker
        sector = profile.get("sector", "") if profile else ""
        industry = profile.get("industry", "") if profile else ""
        exchange = profile.get("exchangeShortName", "") if profile else ""
        st.markdown(f"## {company_name}")
        st.caption(f"{exchange}  |  {sector}  |  {industry}")

    with col_price:
        price = quote.get("price", 0) if quote else 0
        change_pct = quote.get("changesPercentage", 0) if quote else 0
        delta_color = "normal" if change_pct >= 0 else "inverse"
        st.metric(
            label="Price",
            value=f"${price:,.2f}" if price else "N/A",
            delta=f"{change_pct:+.2f}%" if change_pct else None,
            delta_color=delta_color,
        )

    st.divider()

    # -- Key metrics row ---------------------------------------------------

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    mkt_cap = (
        quote.get("marketCap", profile.get("mktCap", 0))
        if quote
        else profile.get("mktCap", 0)
        if profile
        else 0
    )
    pe = quote.get("pe", 0) if quote else 0
    eps = quote.get("eps", 0) if quote else 0
    div_yield = profile.get("lastDiv", 0) if profile else 0
    beta = profile.get("beta", 0) if profile else 0
    year_high = quote.get("yearHigh", 0) if quote else 0
    year_low = quote.get("yearLow", 0) if quote else 0

    m1.metric("Market Cap", fmt_currency(mkt_cap) if mkt_cap else "N/A")
    m2.metric("P/E Ratio", f"{pe:.1f}" if pe else "N/A")
    m3.metric("EPS", f"${eps:.2f}" if eps else "N/A")
    m4.metric(
        "Div Yield",
        fmt_pct(div_yield / price, 1) if price and div_yield else "N/A",
    )
    m5.metric("Beta", f"{beta:.2f}" if beta else "N/A")
    m6.metric(
        "52W Range",
        f"${year_low:,.0f} - ${year_high:,.0f}" if year_high else "N/A",
    )

    st.divider()

    # -- Sparkline charts --------------------------------------------------

    if hist_df is not None and not hist_df.empty:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            hist_df.columns = [c.lower() for c in hist_df.columns]
            close = hist_df.get("close")
            volume = hist_df.get("volume")
            dates = hist_df.index

            if close is not None and len(close) > 5:
                spark_col1, spark_col2 = st.columns(2)

                with spark_col1:
                    price_color = (
                        COLORS["success"]
                        if float(close.iloc[-1]) >= float(close.iloc[0])
                        else COLORS["danger"]
                    )
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=close,
                            mode="lines",
                            line={"color": price_color, "width": 2},
                            fill="tozeroy",
                            fillcolor=price_color.replace(")", ",0.1)").replace(
                                "rgb", "rgba"
                            )
                            if "rgb" in price_color
                            else f"rgba({int(price_color[1:3], 16)},{int(price_color[3:5], 16)},{int(price_color[5:7], 16)},0.1)",
                            showlegend=False,
                        )
                    )
                    fig.update_layout(
                        **dark_layout(
                            title="6-Month Price",
                            height=200,
                            margin={"l": 40, "r": 10, "t": 40, "b": 30},
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with spark_col2:
                    if volume is not None and volume.sum() > 0:
                        fig2 = go.Figure()
                        fig2.add_trace(
                            go.Bar(
                                x=dates,
                                y=volume,
                                marker_color=COLORS["primary"],
                                opacity=0.6,
                                showlegend=False,
                            )
                        )
                        fig2.update_layout(
                            **dark_layout(
                                title="6-Month Volume",
                                height=200,
                                margin={"l": 40, "r": 10, "t": 40, "b": 30},
                            )
                        )
                        st.plotly_chart(fig2, use_container_width=True)
        except Exception:
            pass

    st.divider()

    # -- Two-column layout: profile + sentiment ----------------------------

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### Company Profile")
        description = profile.get("description", "") if profile else ""
        if description:
            if len(description) > 600:
                description = description[:600] + "..."
            st.markdown(description)

        details_col1, details_col2 = st.columns(2)
        with details_col1:
            st.markdown(
                f"**CEO:** {profile.get('ceo', 'N/A')}" if profile else ""
            )
            st.markdown(
                f"**Employees:** {fmt_number(profile.get('fullTimeEmployees', 0), decimals=0)}"
                if profile
                else ""
            )
            st.markdown(
                f"**Country:** {profile.get('country', 'N/A')}"
                if profile
                else ""
            )
        with details_col2:
            st.markdown(
                f"**IPO Date:** {profile.get('ipoDate', 'N/A')}"
                if profile
                else ""
            )
            website = profile.get("website", "") if profile else ""
            if website:
                st.markdown(f"**Website:** [{website}]({website})")
            vol_avg = quote.get("avgVolume", 0) if quote else 0
            if vol_avg:
                st.markdown(f"**Avg Volume:** {fmt_number(vol_avg, decimals=1)}")

    with col_right:
        st.markdown("### Quick Sentiment")
        try:
            sentiment = _fetch_sentiment_quick(ticker)
            if sentiment:
                agg = sentiment.get("aggregate", {})
                weighted = agg.get("weighted_mean", 0)
                trend = sentiment.get("trend", "stable")
                bullish_pct = agg.get("bullish_pct", 0)
                bearish_pct = agg.get("bearish_pct", 0)
                article_count = sentiment.get("article_count", 0)

                if weighted > 0.15:
                    sent_label, sent_color = "Bullish", COLORS["success"]
                elif weighted > 0.05:
                    sent_label, sent_color = "Slightly Bullish", "#86efac"
                elif weighted < -0.15:
                    sent_label, sent_color = "Bearish", COLORS["danger"]
                elif weighted < -0.05:
                    sent_label, sent_color = "Slightly Bearish", "#fca5a5"
                else:
                    sent_label, sent_color = "Neutral", COLORS["neutral"]

                st.markdown(
                    f'<div style="text-align:center; padding: 1rem; '
                    f"background: {COLORS['card_bg']}; border-radius: 12px; "
                    f'border: 1px solid rgba(255,255,255,0.06);">'
                    f'<p style="font-size: 2.5rem; font-weight: 700; color: {sent_color}; '
                    f'margin: 0;">{weighted:+.3f}</p>'
                    f'<p style="font-size: 1.1rem; color: {sent_color}; margin: 4px 0;">'
                    f"{sent_label}</p>"
                    f'<p style="color: {COLORS["text_muted"]}; font-size: 0.85rem; margin: 0;">'
                    f"Based on {article_count} articles</p></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Trend:** {trend.capitalize()}")
                st.markdown(
                    f"**Bullish:** {fmt_pct(bullish_pct)} | "
                    f"**Bearish:** {fmt_pct(bearish_pct)}"
                )
            else:
                st.info("Sentiment data unavailable.")
        except Exception:
            st.info("Sentiment data unavailable.")

    st.divider()

    # -- FMP Rating --------------------------------------------------------

    try:
        rating = _fetch_rating(ticker)
        if rating:
            st.markdown("### Analyst Rating")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Rating", rating.get("rating", "N/A"))
            r2.metric("Score", f"{rating.get('ratingScore', 'N/A')}/5")
            r3.metric(
                "Recommendation",
                rating.get("ratingRecommendation", "N/A"),
            )
            dcf_rec = rating.get("ratingDetailsDCFRecommendation", "")
            r4.metric("DCF Signal", dcf_rec if dcf_rec else "N/A")
    except Exception:
        pass

    # -- Quick navigation --------------------------------------------------

    st.divider()
    st.markdown("### Explore")
    nav_items = [
        (
            "Fundamental Analysis",
            "Income, balance sheet, cash flow trends, health score, DuPont decomposition",
        ),
        (
            "Valuation",
            "DCF model, Graham Number, relative valuation, margin of safety",
        ),
        (
            "Technical Analysis",
            "Candlestick charts, 263 indicators, support/resistance, signals",
        ),
        (
            "Risk Dashboard",
            "VaR, CVaR, drawdowns, regime detection, rolling volatility",
        ),
        (
            "Volatility Modeling",
            "GARCH, EGARCH, GJR, realized vol, news impact curves",
        ),
        (
            "Portfolio",
            "Multi-asset risk, correlation, efficient frontier, drawdown analysis",
        ),
    ]
    cols = st.columns(3)
    for i, (title, desc) in enumerate(nav_items):
        with cols[i % 3]:
            st.markdown(
                f'<div style="background:{COLORS["card_bg"]}; border-radius:12px; '
                f"padding:1.2rem; border: 1px solid rgba(255,255,255,0.06); "
                f'margin-bottom: 0.8rem;">'
                f'<p style="font-weight:600; font-size:1rem; margin-bottom:4px; '
                f'color:{COLORS["text"]};">{title}</p>'
                f'<p style="color:{COLORS["text_muted"]}; font-size:0.85rem; margin:0;">'
                f"{desc}</p></div>",
                unsafe_allow_html=True,
            )
