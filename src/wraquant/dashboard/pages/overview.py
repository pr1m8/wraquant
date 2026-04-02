"""Home / Overview page -- company snapshot and key metrics.

Displays a company profile card, key financial metrics, quick
sentiment indicator, and a summary of wraquant capabilities.
"""

from __future__ import annotations

import streamlit as st


def _get_fmp_client():
    """Create and cache an FMPClient instance."""
    from wraquant.data.providers.fmp import FMPClient

    return FMPClient()


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_profile(ticker: str) -> dict:
    client = _get_fmp_client()
    return client.company_profile(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_quote(ticker: str) -> dict:
    client = _get_fmp_client()
    return client.quote(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_rating(ticker: str) -> dict:
    client = _get_fmp_client()
    return client.rating(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_sentiment_quick(ticker: str) -> dict:
    from wraquant.news.sentiment import news_sentiment

    return news_sentiment(ticker, limit=20)


def render() -> None:
    """Render the Home / Overview page."""
    from wraquant.dashboard.components.metrics import fmt_currency, fmt_number, fmt_pct
    from wraquant.dashboard.components.sidebar import check_api_key

    ticker = st.session_state.get("ticker", "AAPL")

    st.markdown(
        f"# wraquant Dashboard\n" f"*Interactive quantitative finance analysis*"
    )

    if not check_api_key():
        return

    # -- Fetch data --------------------------------------------------------

    with st.spinner(f"Loading {ticker} data..."):
        try:
            profile = _fetch_profile(ticker)
            quote = _fetch_quote(ticker)
        except Exception as exc:
            st.error(f"Failed to fetch data for **{ticker}**: {exc}")
            return

    if not profile or not quote:
        st.warning(f"No data found for **{ticker}**. Check the symbol and try again.")
        return

    # -- Company header ----------------------------------------------------

    col_name, col_price = st.columns([3, 1])
    with col_name:
        company_name = profile.get("companyName", ticker)
        sector = profile.get("sector", "")
        industry = profile.get("industry", "")
        exchange = profile.get("exchangeShortName", "")
        st.markdown(f"## {company_name}")
        st.caption(f"{exchange}  |  {sector}  |  {industry}")

    with col_price:
        price = quote.get("price", 0)
        change_pct = quote.get("changesPercentage", 0)
        delta_color = "normal" if change_pct >= 0 else "inverse"
        st.metric(
            label="Price",
            value=f"${price:,.2f}",
            delta=f"{change_pct:+.2f}%",
            delta_color=delta_color,
        )

    st.divider()

    # -- Key metrics row ---------------------------------------------------

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    mkt_cap = quote.get("marketCap", profile.get("mktCap", 0))
    pe = quote.get("pe", 0)
    eps = quote.get("eps", 0)
    div_yield = profile.get("lastDiv", 0)
    beta = profile.get("beta", 0)
    year_high = quote.get("yearHigh", 0)
    year_low = quote.get("yearLow", 0)

    m1.metric("Market Cap", fmt_currency(mkt_cap))
    m2.metric("P/E Ratio", f"{pe:.1f}" if pe else "N/A")
    m3.metric("EPS", f"${eps:.2f}" if eps else "N/A")
    m4.metric(
        "Div Yield", fmt_pct(div_yield / price, 1) if price and div_yield else "N/A"
    )
    m5.metric("Beta", f"{beta:.2f}" if beta else "N/A")
    m6.metric(
        "52W Range", f"${year_low:,.0f} - ${year_high:,.0f}" if year_high else "N/A"
    )

    st.divider()

    # -- Two-column layout: profile + sentiment ----------------------------

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### Company Profile")
        description = profile.get("description", "")
        if description:
            # Truncate long descriptions
            if len(description) > 600:
                description = description[:600] + "..."
            st.markdown(description)

        # Additional details
        details_col1, details_col2 = st.columns(2)
        with details_col1:
            st.markdown(f"**CEO:** {profile.get('ceo', 'N/A')}")
            st.markdown(
                f"**Employees:** {fmt_number(profile.get('fullTimeEmployees', 0), decimals=0)}"
            )
            st.markdown(f"**Country:** {profile.get('country', 'N/A')}")
        with details_col2:
            st.markdown(f"**IPO Date:** {profile.get('ipoDate', 'N/A')}")
            website = profile.get("website", "")
            if website:
                st.markdown(f"**Website:** [{website}]({website})")
            vol_avg = quote.get("avgVolume", 0)
            st.markdown(f"**Avg Volume:** {fmt_number(vol_avg, decimals=1)}")

    with col_right:
        st.markdown("### Quick Sentiment")
        try:
            sentiment = _fetch_sentiment_quick(ticker)
            agg = sentiment.get("aggregate", {})
            weighted = agg.get("weighted_mean", 0)
            trend = sentiment.get("trend", "stable")
            bullish_pct = agg.get("bullish_pct", 0)
            bearish_pct = agg.get("bearish_pct", 0)
            article_count = sentiment.get("article_count", 0)

            # Sentiment gauge color
            if weighted > 0.15:
                sent_label, sent_color = "Bullish", "#22c55e"
            elif weighted > 0.05:
                sent_label, sent_color = "Slightly Bullish", "#86efac"
            elif weighted < -0.15:
                sent_label, sent_color = "Bearish", "#ef4444"
            elif weighted < -0.05:
                sent_label, sent_color = "Slightly Bearish", "#fca5a5"
            else:
                sent_label, sent_color = "Neutral", "#94a3b8"

            st.markdown(
                f'<div style="text-align:center; padding: 1rem; '
                f'background: #16161d; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);">'
                f'<p style="font-size: 2.5rem; font-weight: 700; color: {sent_color}; '
                f'margin: 0;">{weighted:+.3f}</p>'
                f'<p style="font-size: 1.1rem; color: {sent_color}; margin: 4px 0;">{sent_label}</p>'
                f'<p style="color: #64748b; font-size: 0.85rem; margin: 0;">Based on {article_count} articles</p>'
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown(f"**Trend:** {trend.capitalize()}")
            st.markdown(
                f"**Bullish:** {fmt_pct(bullish_pct)} | **Bearish:** {fmt_pct(bearish_pct)}"
            )

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
            r3.metric("Recommendation", rating.get("ratingRecommendation", "N/A"))
            dcf_rec = rating.get("ratingDetailsDCFRecommendation", "")
            r4.metric("DCF Signal", dcf_rec if dcf_rec else "N/A")
    except Exception:
        pass

    # -- Quick navigation --------------------------------------------------

    st.divider()
    st.markdown("### Explore")
    nav1, nav2, nav3 = st.columns(3)
    with nav1:
        st.markdown(
            '<div style="background:#16161d; border-radius:12px; padding:1.2rem; '
            'border: 1px solid rgba(255,255,255,0.06);">'
            '<p style="font-weight:600; font-size:1rem; margin-bottom:4px;">Fundamental Analysis</p>'
            '<p style="color:#64748b; font-size:0.85rem; margin:0;">'
            "Income, balance sheet, cash flow trends, health score, DuPont decomposition</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with nav2:
        st.markdown(
            '<div style="background:#16161d; border-radius:12px; padding:1.2rem; '
            'border: 1px solid rgba(255,255,255,0.06);">'
            '<p style="font-weight:600; font-size:1rem; margin-bottom:4px;">Valuation</p>'
            '<p style="color:#64748b; font-size:0.85rem; margin:0;">'
            "DCF model, Graham Number, relative valuation, margin of safety</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with nav3:
        st.markdown(
            '<div style="background:#16161d; border-radius:12px; padding:1.2rem; '
            'border: 1px solid rgba(255,255,255,0.06);">'
            '<p style="font-weight:600; font-size:1rem; margin-bottom:4px;">Screener</p>'
            '<p style="color:#64748b; font-size:0.85rem; margin:0;">'
            "Value, growth, quality, Piotroski, Magic Formula preset screens</p>"
            "</div>",
            unsafe_allow_html=True,
        )
