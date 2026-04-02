"""Risk & Regimes page -- risk metrics, VaR/CVaR, drawdowns, regime detection.

Displays risk metrics table, VaR/CVaR gauge, drawdown chart, regime
detection visualization, and rolling volatility chart.
"""

from __future__ import annotations

import streamlit as st


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_price_data(ticker: str, days: int = 730) -> "pd.DataFrame":
    from datetime import datetime, timedelta

    import pandas as pd

    try:
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
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    try:
        import yfinance as yf

        period_map = {365: "1y", 730: "2y", 1095: "3y", 1825: "5y"}
        period = period_map.get(days, "2y")
        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if not data.empty:
            if hasattr(data.columns, "levels"):
                data.columns = data.columns.droplevel(1)
            data.columns = [c.lower() for c in data.columns]
            return data
    except Exception:
        pass

    import numpy as np

    rng = np.random.default_rng(42)
    n = min(days, 504)
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    rets = rng.normal(0.0004, 0.015, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(
        {"close": close}, index=idx,
    )


def _compute_risk_metrics(returns: "pd.Series") -> dict:
    """Compute risk metrics from a returns series."""
    import numpy as np

    returns_clean = returns.dropna()
    n = len(returns_clean)

    if n < 10:
        return {}

    mean_ret = float(returns_clean.mean())
    std_ret = float(returns_clean.std())
    ann_ret = mean_ret * 252
    ann_vol = std_ret * np.sqrt(252)

    # Sharpe (assume 0% risk-free for simplicity)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    # Sortino
    downside = returns_clean[returns_clean < 0]
    downside_std = (
        float(downside.std()) * np.sqrt(252) if len(downside) > 1 else ann_vol
    )
    sortino = ann_ret / downside_std if downside_std > 0 else 0.0

    # Max drawdown
    cum = (1 + returns_clean).cumprod()
    running_max = cum.cummax()
    drawdowns = (cum - running_max) / running_max
    max_dd = float(drawdowns.min())

    # Calmar
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    # VaR and CVaR (historical, 95%)
    var_95 = float(np.percentile(returns_clean, 5))
    cvar_95 = (
        float(returns_clean[returns_clean <= var_95].mean())
        if len(returns_clean[returns_clean <= var_95]) > 0
        else var_95
    )

    # VaR 99%
    var_99 = float(np.percentile(returns_clean, 1))

    # Skewness and Kurtosis
    try:
        from scipy import stats as sp_stats

        skew = float(sp_stats.skew(returns_clean))
        kurt = float(sp_stats.kurtosis(returns_clean))
    except ImportError:
        # Fallback without scipy
        n = len(returns_clean)
        m3 = float(((returns_clean - returns_clean.mean()) ** 3).mean())
        m2 = float(((returns_clean - returns_clean.mean()) ** 2).mean())
        skew = m3 / (m2**1.5) if m2 > 0 else 0.0
        kurt = (
            float(((returns_clean - returns_clean.mean()) ** 4).mean()) / (m2**2) - 3.0
            if m2 > 0
            else 0.0
        )

    # Win rate
    win_rate = float((returns_clean > 0).mean())

    # Best/worst days
    best_day = float(returns_clean.max())
    worst_day = float(returns_clean.min())

    return {
        "ann_return": ann_ret,
        "ann_volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "var_99": var_99,
        "skewness": skew,
        "kurtosis": kurt,
        "win_rate": win_rate,
        "best_day": best_day,
        "worst_day": worst_day,
        "n_observations": n,
    }


def render() -> None:
    """Render the Risk & Regimes page."""
    import numpy as np
    import pandas as pd

    from wraquant.dashboard.components.charts import COLORS, SERIES_COLORS, dark_layout
    from wraquant.dashboard.components.metrics import fmt_pct

    ticker = st.session_state.get("ticker", "AAPL")
    st.markdown(f"# Risk & Regimes: **{ticker}**")

    lookback = st.selectbox(
        "Lookback",
        [365, 730, 1095, 1825],
        index=1,
        format_func=lambda x: {
            365: "1 Year",
            730: "2 Years",
            1095: "3 Years",
            1825: "5 Years",
        }[x],
        key="risk_lookback",
    )

    with st.spinner(f"Loading {ticker} data..."):
        try:
            df = _fetch_price_data(ticker, days=lookback)
        except Exception as exc:
            st.error(f"Failed to fetch data: {exc}")
            return

    if df is None or df.empty:
        st.warning("No data available.")
        return

    df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except (ValueError, TypeError):
            pass
    close = df["close"]
    returns = close.pct_change().dropna()

    tab_risk, tab_dd, tab_regime, tab_vol = st.tabs(
        [
            "Risk Metrics",
            "Drawdown Analysis",
            "Regime Detection",
            "Rolling Volatility",
        ]
    )

    # =====================================================================
    # TAB 1: Risk Metrics
    # =====================================================================
    with tab_risk:
        try:
            metrics = _compute_risk_metrics(returns)
        except Exception as exc:
            st.error(f"Error computing metrics: {exc}")
            metrics = {}

        if metrics:
            # Key metrics row
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Ann. Return", fmt_pct(metrics["ann_return"]))
            m2.metric("Ann. Volatility", fmt_pct(metrics["ann_volatility"]))
            m3.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
            m4.metric("Sortino Ratio", f"{metrics['sortino']:.2f}")
            m5.metric("Max Drawdown", fmt_pct(metrics["max_drawdown"]))

            st.divider()

            # VaR / CVaR gauges
            st.markdown("### Value at Risk")
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("VaR (95%)", fmt_pct(metrics["var_95"]))
            v2.metric("CVaR (95%)", fmt_pct(metrics["cvar_95"]))
            v3.metric("VaR (99%)", fmt_pct(metrics["var_99"]))
            v4.metric("Calmar Ratio", f"{metrics['calmar']:.2f}")

            # VaR visualization
            try:
                import plotly.graph_objects as go

                fig = go.Figure()
                fig.add_trace(
                    go.Histogram(
                        x=returns.values,
                        nbinsx=80,
                        name="Daily Returns",
                        marker_color=COLORS["primary"],
                        opacity=0.7,
                    )
                )

                # VaR lines
                fig.add_vline(
                    x=metrics["var_95"],
                    line_color=COLORS["warning"],
                    line_dash="dash",
                    annotation_text=f"VaR 95%: {metrics['var_95']:.2%}",
                )
                fig.add_vline(
                    x=metrics["var_99"],
                    line_color=COLORS["danger"],
                    line_dash="dash",
                    annotation_text=f"VaR 99%: {metrics['var_99']:.2%}",
                )

                fig.update_layout(
                    **dark_layout(
                        title="Return Distribution with VaR Thresholds",
                        xaxis_title="Daily Return",
                        yaxis_title="Frequency",
                        height=350,
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                pass

            st.divider()

            # Additional stats
            st.markdown("### Distribution Statistics")
            d1, d2, d3, d4, d5 = st.columns(5)
            d1.metric("Skewness", f"{metrics['skewness']:.3f}")
            d2.metric("Kurtosis", f"{metrics['kurtosis']:.3f}")
            d3.metric("Win Rate", fmt_pct(metrics["win_rate"]))
            d4.metric("Best Day", fmt_pct(metrics["best_day"]))
            d5.metric("Worst Day", fmt_pct(metrics["worst_day"]))

            # Full risk table
            st.markdown("### All Risk Metrics")
            risk_table = pd.DataFrame(
                [
                    {"Metric": k.replace("_", " ").title(), "Value": f"{v:.4f}"}
                    for k, v in metrics.items()
                ]
            )
            st.dataframe(risk_table, hide_index=True, use_container_width=True)

    # =====================================================================
    # TAB 2: Drawdown Analysis
    # =====================================================================
    with tab_dd:
        cum = (1 + returns).cumprod()
        running_max = cum.cummax()
        drawdowns = (cum - running_max) / running_max

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                row_heights=[0.5, 0.5],
                subplot_titles=["Cumulative Return", "Drawdown"],
            )

            # Equity curve
            fig.add_trace(
                go.Scatter(
                    x=cum.index,
                    y=cum.values,
                    mode="lines",
                    name="Equity Curve",
                    line={"color": COLORS["primary"], "width": 2},
                    fill="tozeroy",
                    fillcolor="rgba(99,102,241,0.1)",
                ),
                row=1,
                col=1,
            )

            # Drawdown
            fig.add_trace(
                go.Scatter(
                    x=dd_dates,
                    y=drawdowns.values,
                    mode="lines",
                    name="Drawdown",
                    line={"color": COLORS["danger"], "width": 1.5},
                    fill="tozeroy",
                    fillcolor="rgba(239,68,68,0.15)",
                ),
                row=2,
                col=1,
            )

            fig.update_layout(
                **dark_layout(
                    title=f"{ticker} Drawdown Analysis",
                    height=600,
                )
            )
            fig.update_yaxes(tickformat=".1%", row=2, col=1)
            for i in range(1, 3):
                fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", row=i, col=1)
                fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", row=i, col=1)

            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.line_chart(drawdowns)

        # Top drawdowns table
        st.markdown("### Worst Drawdowns")
        try:
            # Find top 5 drawdown periods
            dd_series = drawdowns.copy()
            top_dds = []
            for _ in range(5):
                if dd_series.empty or dd_series.min() >= -0.001:
                    break
                trough_idx = dd_series.idxmin()
                trough_val = float(dd_series.loc[trough_idx])

                # Find start (last time drawdown was 0 before trough)
                prior = dd_series.loc[:trough_idx]
                zero_mask = prior >= -0.001
                if zero_mask.any():
                    start_idx = prior[zero_mask].index[-1]
                else:
                    start_idx = prior.index[0]

                # Find recovery (first time after trough where dd >= 0)
                post = dd_series.loc[trough_idx:]
                recovered = post[post >= -0.001]
                recovery_idx = recovered.index[0] if not recovered.empty else None

                top_dds.append(
                    {
                        "Start": (
                            str(start_idx)[:10]
                            if hasattr(start_idx, "__str__")
                            else str(start_idx)
                        ),
                        "Trough": (
                            str(trough_idx)[:10]
                            if hasattr(trough_idx, "__str__")
                            else str(trough_idx)
                        ),
                        "Recovery": (
                            str(recovery_idx)[:10]
                            if recovery_idx is not None
                            else "Ongoing"
                        ),
                        "Depth": f"{trough_val:.2%}",
                    }
                )

                # Mask out this drawdown period
                dd_series.loc[start_idx:trough_idx] = 0.0

            if top_dds:
                st.dataframe(
                    pd.DataFrame(top_dds), hide_index=True, use_container_width=True
                )
        except Exception:
            pass

    # =====================================================================
    # TAB 3: Regime Detection
    # =====================================================================
    with tab_regime:
        st.markdown("### Market Regime Detection")
        st.markdown(
            "Identifies market regimes using rolling volatility and return characteristics. "
            "Three regimes: **Low Vol** (calm/trending), **Normal**, and **High Vol** (crisis/mean-reversion)."
        )

        # Simple regime detection via rolling volatility quantiles
        window = st.slider("Rolling Window (days)", 10, 60, 21, key="regime_window")
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()

        if not rolling_vol.empty:
            # Define regimes by volatility quantiles
            q33 = float(rolling_vol.quantile(0.33))
            q66 = float(rolling_vol.quantile(0.66))

            regimes = pd.Series("Normal", index=rolling_vol.index)
            regimes[rolling_vol <= q33] = "Low Vol"
            regimes[rolling_vol >= q66] = "High Vol"

            regime_colors = {
                "Low Vol": COLORS["success"],
                "Normal": COLORS["warning"],
                "High Vol": COLORS["danger"],
            }

            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots

                fig = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    row_heights=[0.55, 0.45],
                    subplot_titles=["Price with Regime Coloring", "Rolling Volatility"],
                )

                # Price chart colored by regime
                regime_dates = rolling_vol.index
                price_aligned = close.reindex(regime_dates)

                for regime_name, color in regime_colors.items():
                    mask = regimes == regime_name
                    if mask.any():
                        price_segment = price_aligned.copy()
                        price_segment[~mask] = np.nan
                        fig.add_trace(
                            go.Scatter(
                                x=regime_dates,
                                y=price_segment,
                                mode="lines",
                                name=regime_name,
                                line={"color": color, "width": 2},
                            ),
                            row=1,
                            col=1,
                        )

                # Rolling vol
                fig.add_trace(
                    go.Scatter(
                        x=regime_dates,
                        y=rolling_vol,
                        mode="lines",
                        name="Annualized Vol",
                        line={"color": COLORS["accent2"], "width": 1.5},
                        fill="tozeroy",
                        fillcolor="rgba(56,189,248,0.1)",
                    ),
                    row=2,
                    col=1,
                )

                # Thresholds
                fig.add_hline(
                    y=q33,
                    line_dash="dot",
                    line_color=COLORS["success"],
                    opacity=0.5,
                    row=2,
                    col=1,
                )
                fig.add_hline(
                    y=q66,
                    line_dash="dot",
                    line_color=COLORS["danger"],
                    opacity=0.5,
                    row=2,
                    col=1,
                )

                fig.update_layout(
                    **dark_layout(
                        title=f"{ticker} Market Regimes",
                        height=600,
                    )
                )
                fig.update_yaxes(tickformat=".0%", row=2, col=1)
                for i in range(1, 3):
                    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", row=i, col=1)
                    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", row=i, col=1)

                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.line_chart(rolling_vol)

            # Regime statistics
            st.markdown("### Regime Statistics")
            regime_stats = []
            for regime_name in ["Low Vol", "Normal", "High Vol"]:
                mask = regimes == regime_name
                if mask.any():
                    r = returns.reindex(rolling_vol.index)[mask]
                    v = rolling_vol[mask]
                    regime_stats.append(
                        {
                            "Regime": regime_name,
                            "% of Time": f"{mask.mean():.1%}",
                            "Avg Daily Return": f"{r.mean():.4f}",
                            "Ann. Return": f"{r.mean() * 252:.2%}",
                            "Avg Vol": f"{v.mean():.2%}",
                            "Sharpe": (
                                f"{(r.mean() * 252) / (r.std() * np.sqrt(252)):.2f}"
                                if r.std() > 0
                                else "N/A"
                            ),
                        }
                    )
            if regime_stats:
                st.dataframe(
                    pd.DataFrame(regime_stats),
                    hide_index=True,
                    use_container_width=True,
                )

    # =====================================================================
    # TAB 4: Rolling Volatility
    # =====================================================================
    with tab_vol:
        st.markdown("### Rolling Volatility Analysis")
        vol_windows = st.multiselect(
            "Windows (days)",
            [5, 10, 21, 63, 126, 252],
            default=[21, 63],
            key="vol_windows",
        )

        if vol_windows:
            try:
                import plotly.graph_objects as go

                fig = go.Figure()
                for i, w in enumerate(sorted(vol_windows)):
                    rv = returns.rolling(window=w).std() * np.sqrt(252)
                    rv = rv.dropna()
                    fig.add_trace(
                        go.Scatter(
                            x=rv.index,
                            y=rv,
                            mode="lines",
                            name=f"{w}-day Vol",
                            line={
                                "color": SERIES_COLORS[i % len(SERIES_COLORS)],
                                "width": 1.5,
                            },
                        )
                    )

                fig.update_layout(
                    **dark_layout(
                        title=f"{ticker} Rolling Annualized Volatility",
                        yaxis_title="Annualized Volatility",
                        yaxis_tickformat=".0%",
                        height=450,
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                for w in sorted(vol_windows):
                    rv = returns.rolling(window=w).std() * np.sqrt(252)
                    st.line_chart(rv.dropna())

        # Vol summary
        st.divider()
        st.markdown("### Volatility Summary")
        vol_data = []
        for w in [5, 10, 21, 63, 126, 252]:
            rv = returns.rolling(window=w).std() * np.sqrt(252)
            rv = rv.dropna()
            if not rv.empty:
                vol_data.append(
                    {
                        "Window": f"{w} days",
                        "Current Vol": f"{rv.iloc[-1]:.2%}",
                        "Average Vol": f"{rv.mean():.2%}",
                        "Min Vol": f"{rv.min():.2%}",
                        "Max Vol": f"{rv.max():.2%}",
                        "Percentile": f"{(rv <= rv.iloc[-1]).mean():.0%}",
                    }
                )
        if vol_data:
            st.dataframe(
                pd.DataFrame(vol_data), hide_index=True, use_container_width=True
            )

        # Vol cone
        st.divider()
        st.markdown("### Volatility Term Structure")
        try:
            import plotly.graph_objects as go

            windows = [5, 10, 21, 42, 63, 126, 252]
            current_vols = []
            avg_vols = []
            p25 = []
            p75 = []
            valid_windows = []
            for w in windows:
                rv = returns.rolling(window=w).std() * np.sqrt(252)
                rv = rv.dropna()
                if not rv.empty:
                    current_vols.append(float(rv.iloc[-1]))
                    avg_vols.append(float(rv.mean()))
                    p25.append(float(rv.quantile(0.25)))
                    p75.append(float(rv.quantile(0.75)))
                    valid_windows.append(w)

            if valid_windows:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=valid_windows,
                        y=current_vols,
                        mode="lines+markers",
                        name="Current",
                        line={"color": COLORS["primary"], "width": 2},
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=valid_windows,
                        y=avg_vols,
                        mode="lines+markers",
                        name="Average",
                        line={"color": COLORS["neutral"], "width": 1.5, "dash": "dash"},
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=valid_windows,
                        y=p75,
                        mode="lines",
                        name="75th pctl",
                        line={"color": COLORS["warning"], "width": 1, "dash": "dot"},
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=valid_windows,
                        y=p25,
                        mode="lines",
                        name="25th pctl",
                        line={"color": COLORS["info"], "width": 1, "dash": "dot"},
                        fill="tonexty",
                        fillcolor="rgba(6,182,212,0.08)",
                    )
                )
                fig.update_layout(
                    **dark_layout(
                        title="Volatility Term Structure",
                        xaxis_title="Window (days)",
                        yaxis_title="Annualized Volatility",
                        yaxis_tickformat=".0%",
                        height=400,
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            pass
