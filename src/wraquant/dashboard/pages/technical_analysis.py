"""Technical Analysis page -- candlestick charts, indicators, signals.

Displays interactive candlestick charts with technical indicator
overlays (SMA, EMA, Bollinger Bands), volume bars, RSI and MACD
subplots, support/resistance levels, and a TA summary signal.
"""

from __future__ import annotations

import streamlit as st


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_price_data(ticker: str, days: int = 365) -> "pd.DataFrame":
    """Fetch historical price data via FMP."""
    from datetime import datetime, timedelta

    import pandas as pd

    from wraquant.data.providers.fmp import FMPClient

    client = FMPClient()
    end = datetime.now()
    start = end - timedelta(days=days)
    df = client.historical_price(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="daily",
    )
    return df


def _compute_indicator(close, high, low, volume, name: str, period: int):
    """Compute a single TA indicator, returning result or None."""
    import importlib

    indicator_map = {
        "SMA": ("wraquant.ta.overlap", "sma", "close"),
        "EMA": ("wraquant.ta.overlap", "ema", "close"),
        "Bollinger Bands": ("wraquant.ta.overlap", "bollinger_bands", "close"),
        "RSI": ("wraquant.ta.momentum", "rsi", "close"),
        "MACD": ("wraquant.ta.momentum", "macd", "close"),
        "Stochastic": ("wraquant.ta.momentum", "stochastic", "hlc"),
        "ADX": ("wraquant.ta.trend", "adx", "hlc"),
        "ATR": ("wraquant.ta.volatility", "atr", "hlc"),
        "OBV": ("wraquant.ta.volume", "obv", "cv"),
        "MFI": ("wraquant.ta.volume", "mfi", "hlcv"),
        "CCI": ("wraquant.ta.momentum", "cci", "hlc"),
        "Williams %R": ("wraquant.ta.momentum", "williams_r", "hlc"),
        "ROC": ("wraquant.ta.momentum", "roc", "close"),
        "TRIX": ("wraquant.ta.trend", "trix", "close"),
        "CMF": ("wraquant.ta.volume", "cmf", "hlcv"),
    }

    if name not in indicator_map:
        return None

    module_path, func_name, input_type = indicator_map[name]
    try:
        mod = importlib.import_module(module_path)
        func = getattr(mod, func_name)

        if input_type == "close":
            return func(close, period=period)
        elif input_type == "hlc":
            return func(high, low, close, period=period)
        elif input_type == "cv":
            try:
                return func(close, volume, period=period)
            except TypeError:
                return func(close, volume)
        elif input_type == "hlcv":
            try:
                return func(high, low, close, volume, period=period)
            except TypeError:
                return func(high, low, close, volume)
    except TypeError:
        # Retry without period
        try:
            if input_type == "close":
                return func(close)
            elif input_type == "hlc":
                return func(high, low, close)
            elif input_type == "cv":
                return func(close, volume)
            elif input_type == "hlcv":
                return func(high, low, close, volume)
        except Exception:
            return None
    except Exception:
        return None
    return None


def _generate_ta_summary(close, high, low, volume, period: int = 14) -> dict:
    """Generate a simple bullish/bearish/neutral TA summary."""
    signals = {"bullish": 0, "bearish": 0, "neutral": 0}

    try:
        # RSI
        rsi = _compute_indicator(close, high, low, volume, "RSI", period)
        if rsi is not None:
            last_rsi = rsi.dropna().iloc[-1] if len(rsi.dropna()) > 0 else 50
            if last_rsi < 30:
                signals["bullish"] += 1  # Oversold
            elif last_rsi > 70:
                signals["bearish"] += 1  # Overbought
            else:
                signals["neutral"] += 1
    except Exception:
        pass

    try:
        # SMA crossover
        sma_20 = _compute_indicator(close, high, low, volume, "SMA", 20)
        sma_50 = _compute_indicator(close, high, low, volume, "SMA", 50)
        if sma_20 is not None and sma_50 is not None:
            last_20 = sma_20.dropna().iloc[-1]
            last_50 = sma_50.dropna().iloc[-1]
            if last_20 > last_50:
                signals["bullish"] += 1
            else:
                signals["bearish"] += 1
    except Exception:
        pass

    try:
        # Price vs SMA 200
        sma_200 = _compute_indicator(close, high, low, volume, "SMA", 200)
        if sma_200 is not None:
            last_price = close.iloc[-1]
            last_200 = sma_200.dropna().iloc[-1]
            if last_price > last_200:
                signals["bullish"] += 1
            else:
                signals["bearish"] += 1
    except Exception:
        pass

    try:
        # MACD signal
        macd = _compute_indicator(close, high, low, volume, "MACD", 12)
        if macd is not None and isinstance(macd, dict):
            macd_line = macd.get("macd", macd.get("MACD"))
            signal_line = macd.get("signal", macd.get("Signal"))
            if macd_line is not None and signal_line is not None:
                last_macd = macd_line.dropna().iloc[-1]
                last_signal = signal_line.dropna().iloc[-1]
                if last_macd > last_signal:
                    signals["bullish"] += 1
                else:
                    signals["bearish"] += 1
    except Exception:
        pass

    total = sum(signals.values())
    if total == 0:
        return {"signal": "Neutral", "bullish": 0, "bearish": 0, "neutral": 0}

    if signals["bullish"] > signals["bearish"]:
        overall = "Bullish"
    elif signals["bearish"] > signals["bullish"]:
        overall = "Bearish"
    else:
        overall = "Neutral"

    return {
        "signal": overall,
        "bullish": signals["bullish"],
        "bearish": signals["bearish"],
        "neutral": signals["neutral"],
    }


def render() -> None:
    """Render the Technical Analysis page."""
    import pandas as pd

    from wraquant.dashboard.components.charts import COLORS, SERIES_COLORS, dark_layout
    from wraquant.dashboard.components.sidebar import check_api_key

    ticker = st.session_state.get("ticker", "AAPL")
    st.markdown(f"# Technical Analysis: **{ticker}**")

    if not check_api_key():
        return

    # Controls
    col_range, col_period = st.columns([1, 1])
    with col_range:
        lookback = st.selectbox(
            "Lookback Period",
            [90, 180, 365, 730],
            index=2,
            format_func=lambda x: {
                90: "3 Months",
                180: "6 Months",
                365: "1 Year",
                730: "2 Years",
            }[x],
            key="ta_lookback",
        )
    with col_period:
        indicator_period = st.slider("Indicator Period", 5, 50, 14, key="ta_ind_period")

    # Fetch data
    with st.spinner(f"Loading {ticker} price data..."):
        try:
            df = _fetch_price_data(ticker, days=lookback)
        except Exception as exc:
            st.error(f"Failed to fetch price data: {exc}")
            return

    if df.empty or "close" not in df.columns:
        st.warning("No price data available.")
        return

    # Ensure proper column names
    df.columns = [c.lower() for c in df.columns]
    close = df["close"]
    high = df.get("high", close)
    low = df.get("low", close)
    volume = df.get("volume", pd.Series(dtype=float))
    dates = df["date"] if "date" in df.columns else df.index

    # -- TA Summary --------------------------------------------------------
    st.divider()
    summary = _generate_ta_summary(close, high, low, volume, indicator_period)
    sig_colors = {
        "Bullish": COLORS["success"],
        "Bearish": COLORS["danger"],
        "Neutral": COLORS["warning"],
    }
    sig_color = sig_colors.get(summary["signal"], COLORS["neutral"])

    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(
        f'<div style="text-align:center; padding:0.6rem; background:#16161d; '
        f'border-radius:10px; border:1px solid {sig_color}40;">'
        f'<p style="color:#94a3b8; font-size:0.8rem; margin:0;">Overall Signal</p>'
        f'<p style="color:{sig_color}; font-size:1.5rem; font-weight:700; margin:0;">'
        f'{summary["signal"]}</p></div>',
        unsafe_allow_html=True,
    )
    s2.metric("Bullish Signals", summary["bullish"])
    s3.metric("Bearish Signals", summary["bearish"])
    s4.metric("Neutral Signals", summary["neutral"])

    st.divider()

    # -- Indicator selection -----------------------------------------------
    overlay_indicators = st.multiselect(
        "Price Overlays",
        ["SMA", "EMA", "Bollinger Bands"],
        default=["SMA", "EMA"],
        key="ta_overlays",
    )

    subplot_indicators = st.multiselect(
        "Subplot Indicators",
        ["RSI", "MACD", "Stochastic", "ADX", "ATR", "CCI", "Williams %R", "ROC"],
        default=["RSI", "MACD"],
        key="ta_subplots",
    )

    # -- Candlestick chart with overlays -----------------------------------
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Determine number of rows
        n_subplots = len(subplot_indicators) + 1  # +1 for volume
        total_rows = 1 + n_subplots
        row_heights = [0.45] + [0.55 / n_subplots] * n_subplots

        subplot_titles = ["", "Volume"] + subplot_indicators
        fig = make_subplots(
            rows=total_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            subplot_titles=subplot_titles,
        )

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=dates,
                open=df.get("open", close),
                high=high,
                low=low,
                close=close,
                name="OHLC",
                increasing_line_color=COLORS["success"],
                decreasing_line_color=COLORS["danger"],
                increasing_fillcolor=COLORS["success"],
                decreasing_fillcolor=COLORS["danger"],
            ),
            row=1,
            col=1,
        )

        # Overlays on the price chart
        for i, name in enumerate(overlay_indicators):
            color = SERIES_COLORS[(i + 2) % len(SERIES_COLORS)]

            if name == "Bollinger Bands":
                result = _compute_indicator(
                    close, high, low, volume, name, indicator_period
                )
                if result is not None and isinstance(result, dict):
                    upper = result.get("upper", result.get("upper_band"))
                    middle = result.get("middle", result.get("middle_band"))
                    lower = result.get("lower", result.get("lower_band"))
                    if upper is not None:
                        fig.add_trace(
                            go.Scatter(
                                x=dates,
                                y=upper,
                                mode="lines",
                                name="BB Upper",
                                line={
                                    "color": COLORS["accent3"],
                                    "width": 1,
                                    "dash": "dot",
                                },
                            ),
                            row=1,
                            col=1,
                        )
                    if middle is not None:
                        fig.add_trace(
                            go.Scatter(
                                x=dates,
                                y=middle,
                                mode="lines",
                                name="BB Middle",
                                line={"color": COLORS["accent3"], "width": 1},
                            ),
                            row=1,
                            col=1,
                        )
                    if lower is not None:
                        fig.add_trace(
                            go.Scatter(
                                x=dates,
                                y=lower,
                                mode="lines",
                                name="BB Lower",
                                line={
                                    "color": COLORS["accent3"],
                                    "width": 1,
                                    "dash": "dot",
                                },
                                fill="tonexty",
                                fillcolor="rgba(167,139,250,0.08)",
                            ),
                            row=1,
                            col=1,
                        )
            else:
                for p in ([20, 50] if name == "SMA" else [indicator_period]):
                    result = _compute_indicator(close, high, low, volume, name, p)
                    if result is not None:
                        data = result if isinstance(result, pd.Series) else None
                        if data is not None:
                            fig.add_trace(
                                go.Scatter(
                                    x=dates,
                                    y=data,
                                    mode="lines",
                                    name=f"{name}({p})",
                                    line={"color": color, "width": 1.5},
                                ),
                                row=1,
                                col=1,
                            )
                    if name == "SMA":
                        # Use a different color for 50
                        color = SERIES_COLORS[(i + 3) % len(SERIES_COLORS)]

        # Volume bars
        if not volume.empty and volume.sum() > 0:
            vol_colors = [
                COLORS["success"] if c >= o else COLORS["danger"]
                for c, o in zip(close, df.get("open", close))
            ]
            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=volume,
                    name="Volume",
                    marker_color=vol_colors,
                    opacity=0.5,
                ),
                row=2,
                col=1,
            )

        # Subplot indicators
        for idx, name in enumerate(subplot_indicators):
            row_num = idx + 3  # rows 3, 4, 5, ...
            result = _compute_indicator(
                close, high, low, volume, name, indicator_period
            )
            if result is None:
                continue

            if isinstance(result, dict):
                for j, (key, series) in enumerate(result.items()):
                    if isinstance(series, pd.Series):
                        fig.add_trace(
                            go.Scatter(
                                x=dates,
                                y=series,
                                mode="lines",
                                name=f"{name} {key}",
                                line={
                                    "color": SERIES_COLORS[j % len(SERIES_COLORS)],
                                    "width": 1.5,
                                },
                            ),
                            row=row_num,
                            col=1,
                        )
            elif isinstance(result, pd.Series):
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=result,
                        mode="lines",
                        name=name,
                        line={"color": SERIES_COLORS[0], "width": 1.5},
                    ),
                    row=row_num,
                    col=1,
                )

                # Add reference lines for RSI
                if name == "RSI":
                    fig.add_hline(
                        y=70,
                        line_dash="dash",
                        line_color=COLORS["danger"],
                        opacity=0.5,
                        row=row_num,
                        col=1,
                    )
                    fig.add_hline(
                        y=30,
                        line_dash="dash",
                        line_color=COLORS["success"],
                        opacity=0.5,
                        row=row_num,
                        col=1,
                    )
                elif name == "CCI":
                    fig.add_hline(
                        y=100,
                        line_dash="dash",
                        line_color=COLORS["danger"],
                        opacity=0.5,
                        row=row_num,
                        col=1,
                    )
                    fig.add_hline(
                        y=-100,
                        line_dash="dash",
                        line_color=COLORS["success"],
                        opacity=0.5,
                        row=row_num,
                        col=1,
                    )

        # Layout
        layout = dark_layout(
            title=f"{ticker} Technical Analysis",
            height=250 + 200 * n_subplots,
            showlegend=True,
            xaxis_rangeslider_visible=False,
        )
        fig.update_layout(**layout)

        # Update all axes to dark theme
        for i in range(1, total_rows + 1):
            fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", row=i, col=1)
            fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", row=i, col=1)

        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.warning(
            "Plotly is required for interactive charts. Install with: `pip install plotly`"
        )
        st.line_chart(close)

    # -- Support / Resistance levels (simple pivots) -----------------------
    st.divider()
    st.markdown("### Support & Resistance Levels")
    try:
        last_close = float(close.iloc[-1])
        last_high = float(high.max())
        last_low = float(low.min())
        pivot = (last_high + last_low + last_close) / 3
        r1 = 2 * pivot - last_low
        r2 = pivot + (last_high - last_low)
        s1 = 2 * pivot - last_high
        s2 = pivot - (last_high - last_low)

        lc1, lc2, lc3, lc4, lc5 = st.columns(5)
        lc1.metric("S2", f"${s2:,.2f}")
        lc2.metric("S1", f"${s1:,.2f}")
        lc3.metric("Pivot", f"${pivot:,.2f}")
        lc4.metric("R1", f"${r1:,.2f}")
        lc5.metric("R2", f"${r2:,.2f}")
    except Exception:
        st.info("Could not compute pivot levels.")

    # -- Price stats -------------------------------------------------------
    st.divider()
    st.markdown("### Price Statistics")
    ps1, ps2, ps3, ps4, ps5, ps6 = st.columns(6)
    ps1.metric("Last Close", f"${close.iloc[-1]:,.2f}")
    ps2.metric("Period High", f"${high.max():,.2f}")
    ps3.metric("Period Low", f"${low.min():,.2f}")
    returns = close.pct_change().dropna()
    ps4.metric("Period Return", f"{((close.iloc[-1] / close.iloc[0]) - 1) * 100:+.1f}%")
    ps5.metric("Daily Vol", f"{returns.std() * 100:.2f}%")
    ps6.metric("Ann. Vol", f"{returns.std() * (252**0.5) * 100:.1f}%")
