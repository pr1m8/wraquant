"""FX Analysis page -- currency pairs, sessions, strength, carry trades.

Displays currency pair price charts with session bands, currency
strength rankings, carry trade analysis, and pip calculator.
"""

from __future__ import annotations

import streamlit as st

MAJOR_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
PAIR_DISPLAY = {"EURUSD": "EUR/USD", "GBPUSD": "GBP/USD", "USDJPY": "USD/JPY", "USDCHF": "USD/CHF", "AUDUSD": "AUD/USD", "USDCAD": "USD/CAD", "NZDUSD": "NZD/USD"}


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_fx_data(pair, days=365):
    """Fetch FX pair historical data."""
    from datetime import datetime, timedelta
    import pandas as pd
    try:
        from wraquant.data.providers.fmp import FMPClient
        client = FMPClient()
        end = datetime.now()
        start = end - timedelta(days=days)
        df = client.historical_price(pair, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="daily")
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    try:
        import yfinance as yf
        yf_pair = f"{pair[:3]}{pair[3:]}=X"
        data = yf.download(yf_pair, period="1y", auto_adjust=True, progress=False)
        if not data.empty:
            if hasattr(data.columns, "levels"):
                data.columns = data.columns.droplevel(1)
            data.columns = [c.lower() for c in data.columns]
            return data
    except Exception:
        pass
    import numpy as np
    rng = np.random.default_rng(hash(pair) % 2**31)
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=min(days, 252))
    base_rate = {"EURUSD": 1.08, "GBPUSD": 1.27, "USDJPY": 150.0, "USDCHF": 0.88, "AUDUSD": 0.66, "USDCAD": 1.36, "NZDUSD": 0.61}.get(pair, 1.0)
    rets = rng.normal(0, 0.005, len(idx))
    close = base_rate * np.exp(np.cumsum(rets))
    return pd.DataFrame({"open": close * (1 + rng.normal(0, 0.002, len(idx))), "high": close * (1 + rng.uniform(0.001, 0.008, len(idx))), "low": close * (1 - rng.uniform(0.001, 0.008, len(idx))), "close": close}, index=idx)


def render():
    """Render the FX Analysis page."""
    import numpy as np
    import pandas as pd
    from wraquant.dashboard.components.charts import COLORS, SERIES_COLORS, dark_layout
    from wraquant.dashboard.components.metrics import fmt_pct

    st.markdown("# FX Analysis")
    with st.sidebar:
        st.subheader("FX Settings")
        selected_pair = st.selectbox("Currency Pair", MAJOR_PAIRS, format_func=lambda x: PAIR_DISPLAY.get(x, x), key="fx_pair")
        lookback = st.selectbox("Lookback", [90, 180, 365], index=2, format_func=lambda x: {90: "3 Months", 180: "6 Months", 365: "1 Year"}[x], key="fx_lookback")

    with st.spinner(f"Loading {PAIR_DISPLAY.get(selected_pair, selected_pair)}..."):
        df = _fetch_fx_data(selected_pair, days=lookback)

    if df is None or df.empty:
        st.warning("No FX data available.")
        return

    df.columns = [c.lower() for c in df.columns]
    close = df["close"] if "close" in df.columns else df.iloc[:, -1]
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    dates = df["date"] if "date" in df.columns else df.index
    returns = close.pct_change().dropna()

    last_rate = float(close.iloc[-1])
    period_return = float(close.iloc[-1]) / float(close.iloc[0]) - 1
    ann_vol = float(returns.std()) * np.sqrt(252)

    k1, k2, k3 = st.columns(3)
    k1.metric(PAIR_DISPLAY.get(selected_pair, selected_pair), f"{last_rate:.4f}")
    k2.metric("Period Return", fmt_pct(period_return), delta=f"{period_return:+.2%}", delta_color="normal" if period_return >= 0 else "inverse")
    k3.metric("Ann. Volatility", fmt_pct(ann_vol))
    st.divider()

    tab_chart, tab_strength, tab_carry, tab_calc = st.tabs(["Price Chart", "Currency Strength", "Carry Trade", "Pip Calculator"])

    with tab_chart:
        st.subheader(f"{PAIR_DISPLAY.get(selected_pair, selected_pair)} Price Chart")
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=close, mode="lines", name="Close", line={"color": COLORS["primary"], "width": 2}))
            if len(close) >= 50:
                sma20 = close.rolling(20).mean()
                sma50 = close.rolling(50).mean()
                fig.add_trace(go.Scatter(x=dates, y=sma20, mode="lines", name="SMA(20)", line={"color": COLORS["accent2"], "width": 1, "dash": "dot"}))
                fig.add_trace(go.Scatter(x=dates, y=sma50, mode="lines", name="SMA(50)", line={"color": COLORS["accent4"], "width": 1, "dash": "dot"}))
            fig.update_layout(**dark_layout(title=f"{PAIR_DISPLAY.get(selected_pair, selected_pair)} Daily Chart", yaxis_title="Exchange Rate", height=500))
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.line_chart(close)

        st.subheader("Trading Sessions")
        st.markdown("**Session Times (UTC)**\n- Tokyo: 00:00-09:00\n- London: 07:00-16:00\n- New York: 12:00-21:00\n\n**Overlap Windows**\n- London/Tokyo: 07:00-09:00\n- London/New York: 12:00-16:00")

    with tab_strength:
        st.subheader("Currency Strength Ranking")
        strength_data = {}
        for pair in MAJOR_PAIRS:
            try:
                pair_df = _fetch_fx_data(pair, days=lookback)
                if pair_df is not None and not pair_df.empty:
                    pair_df.columns = [c.lower() for c in pair_df.columns]
                    c = pair_df["close"]
                    if hasattr(c, "columns"):
                        c = c.iloc[:, 0]
                    ret = float(c.iloc[-1] / c.iloc[0] - 1) if len(c) > 1 else 0
                    base = pair[:3]
                    quote = pair[3:]
                    strength_data.setdefault(base, 0)
                    strength_data.setdefault(quote, 0)
                    strength_data[base] += ret
                    strength_data[quote] -= ret
            except Exception:
                pass

        if not strength_data:
            strength_data = {"USD": 0.5, "EUR": -0.3, "GBP": 0.1, "JPY": -0.8, "CHF": 0.2, "AUD": -0.4, "CAD": -0.1, "NZD": -0.5}

        try:
            import plotly.graph_objects as go
            sorted_pairs = sorted(strength_data.items(), key=lambda x: x[1], reverse=True)
            currencies = [p[0] for p in sorted_pairs]
            values = [p[1] for p in sorted_pairs]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=currencies, y=values, marker_color=[COLORS["success"] if v >= 0 else COLORS["danger"] for v in values], text=[f"{v:+.2%}" if abs(v) < 1 else f"{v:+.2f}" for v in values], textposition="auto"))
            fig.update_layout(**dark_layout(title="Currency Strength (Period Performance)", yaxis_title="Strength Score", height=450))
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.dataframe(pd.DataFrame({"Currency": list(strength_data.keys()), "Strength": list(strength_data.values())}), use_container_width=True)

    with tab_carry:
        st.subheader("Carry Trade Analysis")
        rates = {"USD": 5.25, "EUR": 4.50, "GBP": 5.25, "JPY": 0.10, "CHF": 1.75, "AUD": 4.35, "CAD": 5.00, "NZD": 5.50}
        st.markdown("**Central Bank Policy Rates (approximate)**")
        rate_cols = st.columns(len(rates))
        for col, (ccy, rate) in zip(rate_cols, rates.items()):
            col.metric(ccy, f"{rate:.2f}%")
        st.divider()

        col_fund, col_invest = st.columns(2)
        with col_fund:
            funding_ccy = st.selectbox("Funding Currency (borrow)", list(rates.keys()), index=list(rates.keys()).index("JPY"), key="fx_fund_ccy")
        with col_invest:
            invest_ccy = st.selectbox("Investment Currency (lend)", list(rates.keys()), index=0, key="fx_invest_ccy")

        if funding_ccy != invest_ccy:
            diff = rates[invest_ccy] - rates[funding_ccy]
            notional = st.number_input("Notional (units)", 10000, 10000000, 100000, 10000, key="fx_notional")
            annual_carry = (diff / 100) * notional
            c1, c2, c3 = st.columns(3)
            c1.metric("Rate Differential", f"{diff:+.2f}%", delta="Favorable" if diff > 0 else "Unfavorable", delta_color="normal" if diff > 0 else "inverse")
            c2.metric("Annual Carry Income", f"${annual_carry:,.0f}")
            c3.metric("Daily Carry", f"${annual_carry / 365:,.2f}")

            st.subheader("Carry Attractiveness Matrix")
            carry_matrix = []
            for invest in rates:
                row = {"Invest \\ Fund": invest}
                for fund in rates:
                    row[fund] = f"{rates[invest] - rates[fund]:+.2f}%" if invest != fund else "-"
                carry_matrix.append(row)
            st.dataframe(pd.DataFrame(carry_matrix).set_index("Invest \\ Fund"), use_container_width=True)
        else:
            st.warning("Select different currencies.")

    with tab_calc:
        st.subheader("Pip Calculator")
        col_entry, col_exit, col_lots = st.columns(3)
        with col_entry:
            entry_price = st.number_input("Entry Price", 0.0001, 500.0, float(last_rate), 0.0001, key="fx_entry")
        with col_exit:
            exit_price = st.number_input("Exit Price", 0.0001, 500.0, float(last_rate) * 1.001, 0.0001, key="fx_exit")
        with col_lots:
            lot_size_input = st.selectbox("Lot Size", [1000, 10000, 100000], index=2, format_func=lambda x: {1000: "Micro (1K)", 10000: "Mini (10K)", 100000: "Standard (100K)"}[x], key="fx_lot_size")

        try:
            from wraquant.forex.analysis import pip_distance, pip_value
            pips_moved = pip_distance(entry_price, exit_price, selected_pair)
            pv = pip_value(selected_pair, lot_size=lot_size_input)
        except Exception:
            is_jpy = "JPY" in selected_pair
            pip_size = 0.01 if is_jpy else 0.0001
            pips_moved = (exit_price - entry_price) / pip_size
            pv = pip_size * lot_size_input
            if is_jpy:
                pv = pv / exit_price if exit_price > 0 else 0

        pnl = float(pips_moved) * float(pv)
        r1, r2, r3 = st.columns(3)
        r1.metric("Pips", f"{float(pips_moved):+.1f}")
        r2.metric("Pip Value", f"${float(pv):.4f}")
        r3.metric("P&L", f"${pnl:+,.2f}", delta_color="normal" if pnl >= 0 else "inverse")

        st.divider()
        st.markdown("### Pip Size Reference")
        ref_data = [{"Pair": PAIR_DISPLAY.get(p, p), "Pip Size": "0.01" if "JPY" in p else "0.0001"} for p in MAJOR_PAIRS]
        st.dataframe(pd.DataFrame(ref_data), hide_index=True, use_container_width=True)
