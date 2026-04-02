"""Portfolio Risk Dashboard -- COMPLETE REWRITE.

Multi-asset portfolio analysis with risk decomposition, correlation,
drawdown analysis, and optimization. Uses wraquant.risk and wraquant.opt
for computations with Plotly for interactive charting.
"""

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_prices(tickers: list, period: str = "2y"):
    """Fetch adjusted close prices for multiple tickers."""
    import pandas as pd

    try:
        import yfinance as yf

        data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
        if "Close" in data.columns or (
            hasattr(data.columns, "levels")
            and "Close" in data.columns.get_level_values(0)
        ):
            prices = (
                data["Close"]
                if len(tickers) > 1
                else data[["Close"]].rename(columns={"Close": tickers[0]})
            )
        else:
            prices = data
        return prices.dropna()
    except Exception:
        pass

    try:
        from wraquant.data.providers.fmp import FMPClient

        client = FMPClient()
        from datetime import datetime, timedelta

        end = datetime.now()
        years = int(period.replace("y", ""))
        start = end - timedelta(days=years * 365)
        frames = {}
        for t in tickers:
            try:
                df = client.historical_price(
                    t,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    interval="daily",
                )
                if df is not None and not df.empty:
                    df.columns = [c.lower() for c in df.columns]
                    frames[t] = df["close"]
            except Exception:
                pass
        if frames:
            return pd.DataFrame(frames).dropna()
    except Exception:
        pass

    return None


def _synthetic_returns(tickers: list, n: int = 504):
    """Generate synthetic multi-asset returns for demo mode."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    k = len(tickers)
    A = rng.standard_normal((k, k))
    cov = (A @ A.T) / k * 0.0002
    np.fill_diagonal(cov, np.abs(np.diag(cov)) + 0.0001)
    means = rng.uniform(0.0002, 0.0008, k)
    rets = rng.multivariate_normal(means, cov, size=n)
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    return pd.DataFrame(rets, index=idx, columns=tickers)


def render() -> None:
    """Render the Portfolio Risk Dashboard page."""
    import numpy as np
    import pandas as pd

    from wraquant.dashboard.components.charts import COLORS, SERIES_COLORS, dark_layout
    from wraquant.dashboard.components.metrics import fmt_pct

    st.markdown("# Portfolio Risk Dashboard")

    # -- Input: tickers and weights --------------------------------------------

    ticker = st.session_state.get("ticker", "AAPL")

    tickers_input = st.text_input(
        "Portfolio tickers (comma-separated)",
        value=f"{ticker}, MSFT, GOOGL, BND, GLD",
        key="port_tickers_input",
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if len(tickers) < 2:
        st.warning("Enter at least 2 tickers.")
        return

    # Weight sliders
    st.markdown("**Portfolio Weights** (auto-normalized)")
    weight_cols = st.columns(min(len(tickers), 6))
    raw_weights = {}
    default_w = round(1.0 / len(tickers), 2)
    for i, t in enumerate(tickers):
        col = weight_cols[i % len(weight_cols)]
        raw_weights[t] = col.slider(t, 0.0, 1.0, default_w, 0.01, key=f"pw_{t}")

    total_w = sum(raw_weights.values())
    if total_w == 0:
        st.error("Total weight cannot be zero.")
        return
    weights = {t: w / total_w for t, w in raw_weights.items()}
    st.caption(f"Weights sum to {total_w:.2f}, normalized to 1.0")

    # Controls
    ctrl1, ctrl2 = st.columns(2)
    with ctrl1:
        period = st.selectbox("Lookback", ["1y", "2y", "3y", "5y"], index=1, key="port_period")
    with ctrl2:
        confidence = st.slider("VaR Confidence", 0.90, 0.99, 0.95, 0.01, key="port_conf")

    w_arr = np.array([weights[t] for t in tickers])

    # -- Fetch data ------------------------------------------------------------

    with st.spinner("Fetching price data..."):
        prices = _fetch_prices(tickers, period=period)

    if prices is not None and not prices.empty:
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.droplevel(0)
        matched = [t for t in tickers if t in prices.columns]
        if len(matched) < 2:
            st.info("Live data unavailable -- using synthetic returns for demo.")
            returns = _synthetic_returns(tickers)
        else:
            prices = prices[matched]
            returns = prices.pct_change().dropna()
            tickers = matched
            w_arr = np.array([weights.get(t, 1.0 / len(tickers)) for t in tickers])
            w_arr = w_arr / w_arr.sum()
    else:
        st.info("Live data unavailable -- using synthetic returns for demo.")
        returns = _synthetic_returns(tickers)

    port_returns = returns.values @ w_arr
    port_ret_series = pd.Series(port_returns, index=returns.index, name="Portfolio")
    cov_matrix = returns.cov().values
    ann_cov = cov_matrix * 252

    # -- Overview metrics ------------------------------------------------------

    try:
        from wraquant.risk.metrics import (
            hit_ratio,
            max_drawdown,
            sharpe_ratio,
            sortino_ratio,
        )

        sr = float(sharpe_ratio(port_ret_series))
        so = float(sortino_ratio(port_ret_series))
        cum = (1 + port_ret_series).cumprod()
        mdd = float(max_drawdown(cum))
        hr = float(hit_ratio(port_ret_series))
    except Exception:
        sr = float(port_ret_series.mean() / port_ret_series.std() * np.sqrt(252)) if port_ret_series.std() > 0 else 0.0
        so = sr
        cum = (1 + port_ret_series).cumprod()
        mdd = float((cum / cum.cummax() - 1).min())
        hr = float((port_ret_series > 0).mean())

    ann_vol = float(port_ret_series.std() * np.sqrt(252))
    ann_ret = float((1 + port_ret_series).prod() ** (252 / max(len(port_ret_series), 1)) - 1)

    var_95 = float(np.percentile(port_returns, (1 - confidence) * 100))
    tail = port_returns[port_returns <= var_95]
    cvar_95 = float(tail.mean()) if len(tail) > 0 else var_95

    st.divider()

    tab_overview, tab_alloc, tab_risk, tab_corr, tab_dd, tab_opt = st.tabs([
        "Overview",
        "Allocation",
        "Risk Decomposition",
        "Correlation",
        "Drawdown",
        "Optimization",
    ])

    # ---- Overview Tab ----
    with tab_overview:
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Ann. Return", fmt_pct(ann_ret))
        m2.metric("Ann. Vol", fmt_pct(ann_vol))
        m3.metric("Sharpe", f"{sr:.2f}")
        m4.metric("Max DD", fmt_pct(mdd))
        m5.metric(f"VaR ({confidence:.0%})", fmt_pct(var_95))
        m6.metric(f"CVaR ({confidence:.0%})", fmt_pct(cvar_95))

        m7, m8, m9, m10 = st.columns(4)
        m7.metric("Sortino", f"{so:.2f}")
        m8.metric("Hit Rate", fmt_pct(hr))
        calmar = ann_ret / abs(mdd) if mdd != 0 else 0.0
        m9.metric("Calmar", f"{calmar:.2f}")
        m10.metric("Observations", f"{len(port_ret_series):,}")

        # Equity curve
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=cum.index, y=cum.values, mode="lines",
                    line={"color": COLORS["primary"], "width": 2},
                    name="Portfolio",
                )
            )
            fig.update_layout(
                **dark_layout(title="Portfolio Equity Curve", yaxis_title="Growth of $1", height=400)
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.line_chart(cum)

    # ---- Allocation Tab ----
    with tab_alloc:
        col_pie, col_table = st.columns([1, 1])

        with col_pie:
            st.subheader("Weight Allocation")
            try:
                import plotly.graph_objects as go

                fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=tickers,
                            values=w_arr,
                            hole=0.4,
                            marker={"colors": SERIES_COLORS[:len(tickers)]},
                            textinfo="label+percent",
                        )
                    ]
                )
                fig.update_layout(**dark_layout(title="Portfolio Weights", height=400))
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                for t, w in zip(tickers, w_arr, strict=False):
                    st.markdown(f"**{t}:** {w:.1%}")

        with col_table:
            st.subheader("Individual Asset Metrics")
            asset_data = []
            for i, t in enumerate(tickers):
                asset_ret = returns[t]
                a_vol = float(asset_ret.std() * np.sqrt(252))
                a_ret = float((1 + asset_ret).prod() ** (252 / max(len(asset_ret), 1)) - 1)
                a_sr = a_ret / a_vol if a_vol > 0 else 0.0
                a_cum = (1 + asset_ret).cumprod()
                a_dd = float((a_cum / a_cum.cummax() - 1).min())
                asset_data.append({
                    "Asset": t,
                    "Weight": f"{w_arr[i]:.1%}",
                    "Ann. Return": f"{a_ret:.1%}",
                    "Ann. Vol": f"{a_vol:.1%}",
                    "Sharpe": f"{a_sr:.2f}",
                    "Max DD": f"{a_dd:.1%}",
                })
            st.dataframe(
                pd.DataFrame(asset_data).set_index("Asset"),
                use_container_width=True,
            )

    # ---- Risk Decomposition Tab ----
    with tab_risk:
        try:
            import plotly.graph_objects as go

            # Component VaR
            try:
                from wraquant.risk.portfolio_analytics import component_var, marginal_var

                comp = component_var(w_arr, returns, alpha=1 - confidence)
                comp_values = comp.values if hasattr(comp, "values") else np.array(comp)
                marg = marginal_var(w_arr, returns, alpha=1 - confidence)
                marg_values = marg.values if hasattr(marg, "values") else np.array(marg)
            except Exception:
                marginal = cov_matrix @ w_arr
                comp_values = np.abs(w_arr * marginal)
                comp_values = comp_values / comp_values.sum()
                marg_values = marginal / np.sqrt(w_arr @ cov_matrix @ w_arr) if np.sqrt(w_arr @ cov_matrix @ w_arr) > 0 else marginal

            # Risk contribution
            try:
                from wraquant.risk.portfolio import risk_contribution

                rc = risk_contribution(w_arr, cov_matrix)
            except Exception:
                marginal = cov_matrix @ w_arr
                port_vol_raw = np.sqrt(w_arr @ cov_matrix @ w_arr)
                rc = (w_arr * (cov_matrix @ w_arr)) / port_vol_raw if port_vol_raw > 0 else w_arr
                rc = rc / rc.sum()

            col_waterfall, col_marginal = st.columns(2)

            with col_waterfall:
                st.subheader("Component VaR (Waterfall)")
                fig = go.Figure(
                    go.Waterfall(
                        name="Component VaR",
                        x=tickers,
                        y=np.abs(comp_values),
                        textposition="auto",
                        text=[f"{v:.4f}" for v in np.abs(comp_values)],
                        connector={"line": {"color": COLORS["neutral"]}},
                        increasing={"marker": {"color": COLORS["danger"]}},
                        decreasing={"marker": {"color": COLORS["success"]}},
                        totals={"marker": {"color": COLORS["primary"]}},
                    )
                )
                fig.update_layout(
                    **dark_layout(title="Component VaR Waterfall", yaxis_title="VaR Contribution", height=400)
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_marginal:
                st.subheader("Marginal VaR")
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=tickers, y=np.abs(marg_values),
                        marker_color=[
                            COLORS["danger"] if v > np.mean(np.abs(marg_values)) else COLORS["primary"]
                            for v in np.abs(marg_values)
                        ],
                        text=[f"{v:.6f}" for v in np.abs(marg_values)],
                        textposition="auto",
                    )
                )
                fig.update_layout(
                    **dark_layout(title="Marginal VaR by Asset", yaxis_title="Marginal VaR", height=400)
                )
                st.plotly_chart(fig, use_container_width=True)

            # Risk contribution bar chart
            st.subheader("Risk Contribution")
            col_rc, col_rctable = st.columns([2, 1])

            with col_rc:
                colors = [
                    COLORS["danger"] if v > 1.5 / len(tickers) else COLORS["primary"]
                    for v in rc
                ]
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=tickers, y=rc, marker_color=colors,
                        text=[f"{v:.1%}" for v in rc],
                        textposition="auto",
                    )
                )
                fig.add_hline(
                    y=1.0 / len(tickers), line_dash="dash",
                    line_color=COLORS["neutral"],
                    annotation_text="Equal contribution",
                )
                fig.update_layout(
                    **dark_layout(title="Risk Contribution by Asset", yaxis_title="Fraction", height=350)
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_rctable:
                rc_df = pd.DataFrame({
                    "Asset": tickers,
                    "Weight": [f"{w:.1%}" for w in w_arr],
                    "Risk Contrib.": [f"{r:.1%}" for r in rc],
                    "RC / Weight": [f"{r/w:.2f}" if w > 0 else "N/A" for r, w in zip(rc, w_arr, strict=False)],
                }).set_index("Asset")
                st.dataframe(rc_df, use_container_width=True)

        except ImportError:
            st.warning("Plotly required for interactive charts.")

    # ---- Correlation Tab ----
    with tab_corr:
        st.subheader("Correlation Heatmap")
        try:
            import plotly.graph_objects as go

            corr = returns.corr()
            fig = go.Figure(
                data=go.Heatmap(
                    z=corr.values, x=corr.columns.tolist(),
                    y=corr.index.tolist(), colorscale="RdBu_r",
                    zmin=-1, zmax=1,
                    text=corr.values.round(3), texttemplate="%{text}",
                    textfont={"size": 11},
                )
            )
            fig.update_layout(**dark_layout(title="Return Correlation Matrix", height=500))
            st.plotly_chart(fig, use_container_width=True)

            # Diversification ratio
            try:
                from wraquant.risk.portfolio import diversification_ratio

                div_ratio = float(diversification_ratio(w_arr, ann_cov))
            except Exception:
                ind_vols = np.sqrt(np.diag(ann_cov))
                div_ratio = float(w_arr @ ind_vols) / ann_vol if ann_vol > 0 else 1.0

            st.metric(
                "Diversification Ratio",
                f"{div_ratio:.2f}",
                help="Ratio of weighted avg vol to portfolio vol. Higher = better diversified.",
            )

            # Rolling correlation between top 2 holdings
            st.subheader("Rolling Pairwise Correlation")
            if len(tickers) >= 2:
                pair_a, pair_b = tickers[0], tickers[1]
                rc_window = st.select_slider(
                    "Rolling Window", options=[20, 60, 120, 252], value=60,
                    key="port_rc_window",
                )
                roll_corr = returns[pair_a].rolling(rc_window).corr(returns[pair_b]).dropna()
                fig_rc = go.Figure()
                fig_rc.add_trace(
                    go.Scatter(
                        x=roll_corr.index, y=roll_corr.values, mode="lines",
                        line={"color": COLORS["primary"], "width": 2},
                        name=f"Corr({pair_a}, {pair_b})",
                    )
                )
                fig_rc.update_layout(
                    **dark_layout(
                        title=f"Rolling {rc_window}d Correlation: {pair_a} vs {pair_b}",
                        yaxis_title="Correlation", height=350,
                    )
                )
                st.plotly_chart(fig_rc, use_container_width=True)

        except ImportError:
            st.dataframe(
                returns.corr().style.format("{:.3f}"),
                use_container_width=True,
            )

    # ---- Drawdown Tab ----
    with tab_dd:
        st.subheader("Drawdown Chart")
        try:
            import plotly.graph_objects as go

            cum = (1 + port_ret_series).cumprod()
            dd = cum / cum.cummax() - 1

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=dd.index, y=dd.values,
                    fill="tozeroy",
                    line={"color": COLORS["danger"], "width": 1},
                    fillcolor="rgba(239, 68, 68, 0.3)",
                    name="Drawdown",
                )
            )
            fig.update_layout(
                **dark_layout(title="Portfolio Drawdown", yaxis_title="Drawdown", height=400)
            )
            fig.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig, use_container_width=True)

            # Top 5 drawdown events
            st.subheader("Top 5 Drawdown Events")
            try:
                from wraquant.risk.historical import crisis_drawdowns

                crises = crisis_drawdowns(port_ret_series, top_n=5)
                if isinstance(crises, dict) and "drawdowns" in crises:
                    dd_list = crises["drawdowns"]
                    if isinstance(dd_list, list) and dd_list:
                        dd_rows = []
                        for d in dd_list[:5]:
                            dd_rows.append({
                                "Depth": f"{d.get('depth', d.get('max_drawdown', 0)):.1%}",
                                "Start": str(d.get("start", "")),
                                "Trough": str(d.get("trough", d.get("valley", ""))),
                                "End": str(d.get("end", d.get("recovery", ""))),
                                "Duration": d.get("duration", d.get("days", "N/A")),
                            })
                        st.dataframe(pd.DataFrame(dd_rows), hide_index=True, use_container_width=True)
            except Exception:
                # Fallback: find top drawdowns manually
                dd_vals = dd.values
                in_dd = False
                dd_events = []
                start_idx = 0

                for i in range(len(dd_vals)):
                    if dd_vals[i] < -0.001 and not in_dd:
                        in_dd = True
                        start_idx = i
                    elif dd_vals[i] >= -0.001 and in_dd:
                        in_dd = False
                        trough_idx = start_idx + np.argmin(dd_vals[start_idx:i])
                        dd_events.append({
                            "Depth": f"{dd_vals[trough_idx]:.1%}",
                            "Start": str(dd.index[start_idx].date()),
                            "Trough": str(dd.index[trough_idx].date()),
                            "End": str(dd.index[i].date()),
                            "Duration": f"{i - start_idx} days",
                        })

                if dd_events:
                    dd_events.sort(key=lambda x: float(x["Depth"].rstrip("%")))
                    st.dataframe(
                        pd.DataFrame(dd_events[:5]),
                        hide_index=True, use_container_width=True,
                    )

            # Recovery analysis
            st.subheader("Recovery Analysis")
            underwater = dd < -0.01
            n_underwater = int(underwater.sum())
            pct_underwater = n_underwater / len(dd) if len(dd) > 0 else 0
            current_dd = float(dd.iloc[-1])

            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Current Drawdown", f"{current_dd:.1%}")
            rc2.metric("Days Underwater (>1%)", f"{n_underwater}")
            rc3.metric("% Time Underwater", f"{pct_underwater:.1%}")

        except ImportError:
            eq = (1 + port_ret_series).cumprod()
            dd = eq / eq.cummax() - 1
            st.area_chart(dd)

    # ---- Optimization Tab ----
    with tab_opt:
        st.subheader("Portfolio Optimization")

        col_current, col_optimal = st.columns(2)

        # Current portfolio metrics
        with col_current:
            st.markdown("**Current Portfolio**")
            curr_metrics = pd.DataFrame({
                "Asset": tickers,
                "Weight": [f"{w:.1%}" for w in w_arr],
            })
            st.dataframe(curr_metrics, hide_index=True, use_container_width=True)
            st.metric("Sharpe", f"{sr:.2f}")
            st.metric("Vol", f"{ann_vol:.1%}")

        # Optimal portfolio (max Sharpe)
        with col_optimal:
            st.markdown("**Max Sharpe Optimal**")

            opt_weights = None
            try:
                from wraquant.opt.portfolio import max_sharpe

                opt_result = max_sharpe(returns)
                if hasattr(opt_result, "weights"):
                    opt_weights = opt_result.weights
                elif isinstance(opt_result, dict):
                    opt_weights = opt_result.get("weights", None)
                elif isinstance(opt_result, np.ndarray):
                    opt_weights = opt_result
            except Exception:
                pass

            if opt_weights is None:
                # Monte Carlo fallback: find max Sharpe via simulation
                rng = np.random.default_rng(0)
                n_sim = 5000
                n_assets = len(tickers)
                mu = returns.mean().values * 252
                sigma = ann_cov
                best_sharpe = -np.inf
                best_w = w_arr.copy()

                for _ in range(n_sim):
                    rw = rng.dirichlet(np.ones(n_assets))
                    p_ret = float(rw @ mu)
                    p_vol = float(np.sqrt(rw @ sigma @ rw))
                    p_sharpe = p_ret / p_vol if p_vol > 0 else 0.0
                    if p_sharpe > best_sharpe:
                        best_sharpe = p_sharpe
                        best_w = rw
                opt_weights = best_w

            opt_arr = np.array(opt_weights).flatten()
            if len(opt_arr) == len(tickers):
                opt_df = pd.DataFrame({
                    "Asset": tickers,
                    "Weight": [f"{w:.1%}" for w in opt_arr],
                })
                st.dataframe(opt_df, hide_index=True, use_container_width=True)

                opt_ret = float(opt_arr @ returns.mean().values * 252)
                opt_vol = float(np.sqrt(opt_arr @ ann_cov @ opt_arr))
                opt_sharpe = opt_ret / opt_vol if opt_vol > 0 else 0.0
                st.metric("Sharpe", f"{opt_sharpe:.2f}")
                st.metric("Vol", f"{opt_vol:.1%}")

        # Weight comparison chart
        st.divider()
        st.subheader("Current vs Optimal Weights")
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    name="Current", x=tickers, y=w_arr,
                    marker_color=COLORS["primary"],
                    text=[f"{w:.1%}" for w in w_arr],
                    textposition="auto",
                )
            )
            if opt_weights is not None and len(opt_arr) == len(tickers):
                fig.add_trace(
                    go.Bar(
                        name="Optimal (Max Sharpe)", x=tickers, y=opt_arr,
                        marker_color=COLORS["accent2"],
                        text=[f"{w:.1%}" for w in opt_arr],
                        textposition="auto",
                    )
                )
            fig.update_layout(
                **dark_layout(title="Weight Comparison", yaxis_title="Weight", barmode="group", height=400)
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            pass

        # Efficient frontier
        st.subheader("Efficient Frontier (Monte Carlo)")
        try:
            import plotly.graph_objects as go

            rng = np.random.default_rng(0)
            n_portfolios = 3000
            n_assets = len(tickers)
            mu = returns.mean().values * 252
            sigma = ann_cov

            sim_ret_arr = np.zeros(n_portfolios)
            sim_vols = np.zeros(n_portfolios)
            sim_sharpes = np.zeros(n_portfolios)

            for i in range(n_portfolios):
                rw = rng.dirichlet(np.ones(n_assets))
                p_ret = float(rw @ mu)
                p_vol = float(np.sqrt(rw @ sigma @ rw))
                sim_ret_arr[i] = p_ret
                sim_vols[i] = p_vol
                sim_sharpes[i] = p_ret / p_vol if p_vol > 0 else 0.0

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=sim_vols, y=sim_ret_arr, mode="markers",
                    marker={
                        "color": sim_sharpes, "colorscale": "Viridis",
                        "size": 3, "colorbar": {"title": "Sharpe"},
                    },
                    name="Random Portfolios",
                    hovertemplate="Vol: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>",
                )
            )
            # Current portfolio
            fig.add_trace(
                go.Scatter(
                    x=[ann_vol], y=[ann_ret], mode="markers",
                    marker={"color": COLORS["danger"], "size": 14, "symbol": "star"},
                    name="Current Portfolio",
                )
            )
            # Optimal
            if opt_weights is not None and len(opt_arr) == len(tickers):
                fig.add_trace(
                    go.Scatter(
                        x=[opt_vol], y=[opt_ret], mode="markers",
                        marker={"color": COLORS["success"], "size": 14, "symbol": "diamond"},
                        name="Max Sharpe",
                    )
                )

            fig.update_layout(
                **dark_layout(
                    title="Efficient Frontier",
                    xaxis_title="Annualized Volatility",
                    yaxis_title="Annualized Return",
                    height=500,
                )
            )
            fig.update_xaxes(tickformat=".0%")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            st.info("Plotly required for efficient frontier plot.")
