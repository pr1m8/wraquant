"""Portfolio Risk Dashboard -- multi-asset risk analysis.

Displays portfolio risk metrics, component VaR decomposition,
correlation heatmaps, risk contribution charts, drawdown analysis,
and an efficient frontier visualization. Uses ``wraquant.risk`` for
computations and Plotly for interactive charts.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_prices(tickers: list[str], period: str = "2y") -> "pd.DataFrame":
    """Fetch adjusted close prices for multiple tickers."""
    import yfinance as yf

    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if "Close" in data.columns or (
        hasattr(data.columns, "levels") and "Close" in data.columns.get_level_values(0)
    ):
        prices = (
            data["Close"]
            if len(tickers) > 1
            else data[["Close"]].rename(columns={"Close": tickers[0]})
        )
    else:
        prices = data
    return prices.dropna()


def _synthetic_returns(tickers: list[str], n: int = 504) -> "pd.DataFrame":
    """Generate synthetic multi-asset returns for demo mode."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    n_assets = len(tickers)
    # Create a random correlation structure
    A = rng.standard_normal((n_assets, n_assets))
    cov = (A @ A.T) / n_assets * 0.0002
    np.fill_diagonal(cov, np.abs(np.diag(cov)) + 0.0001)
    means = rng.uniform(0.0002, 0.0008, n_assets)
    rets = rng.multivariate_normal(means, cov, size=n)
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    return pd.DataFrame(rets, index=idx, columns=tickers)


def render() -> None:
    """Render the Portfolio Risk Dashboard page."""
    import numpy as np
    import pandas as pd

    st.header("Portfolio Risk Dashboard")

    # -- Ticker and weight inputs ------------------------------------------

    ticker = st.session_state.get("ticker", "AAPL")

    with st.sidebar:
        st.subheader("Portfolio Setup")
        tickers_input = st.text_input(
            "Tickers (comma-separated)",
            value=f"{ticker}, MSFT, GOOGL, BND, GLD",
            key="port_risk_tickers",
        )
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        if len(tickers) < 2:
            st.warning("Enter at least 2 tickers.")
            return

        period = st.selectbox(
            "Lookback", ["1y", "2y", "3y", "5y"], index=1, key="port_risk_period"
        )
        confidence = st.slider(
            "VaR Confidence", 0.90, 0.99, 0.95, 0.01, key="port_risk_conf"
        )

        st.markdown("**Portfolio Weights**")
        raw_weights = {}
        for t in tickers:
            raw_weights[t] = st.slider(
                t, 0.0, 1.0, round(1.0 / len(tickers), 2), 0.01, key=f"w_{t}"
            )

        total_w = sum(raw_weights.values())
        if total_w == 0:
            st.error("Total weight cannot be zero.")
            return
        weights = {t: w / total_w for t, w in raw_weights.items()}
        st.caption(f"Weights normalized (sum={total_w:.2f})")

    w_arr = np.array([weights[t] for t in tickers])

    # -- Fetch data --------------------------------------------------------

    with st.spinner("Fetching price data..."):
        try:
            prices = _fetch_prices(tickers, period=period)
            if isinstance(prices.columns, pd.MultiIndex):
                prices.columns = prices.columns.droplevel(0)
            # Keep only columns that matched
            matched = [t for t in tickers if t in prices.columns]
            if len(matched) < 2:
                raise ValueError("Fewer than 2 tickers returned data")
            prices = prices[matched]
            returns = prices.pct_change().dropna()
            tickers = matched
            w_arr = np.array([weights.get(t, 1.0 / len(tickers)) for t in tickers])
            w_arr = w_arr / w_arr.sum()
        except Exception:
            st.info("Live data unavailable -- using synthetic returns for demo.")
            returns = _synthetic_returns(tickers)

    port_returns = returns.values @ w_arr
    port_ret_series = pd.Series(port_returns, index=returns.index, name="Portfolio")
    cov_matrix = returns.cov().values
    ann_cov = cov_matrix * 252

    # -- KPI metrics -------------------------------------------------------

    try:
        from wraquant.risk.metrics import (
            hit_ratio,
            max_drawdown,
            sharpe_ratio,
            sortino_ratio,
        )

        sr = sharpe_ratio(port_ret_series)
        so = sortino_ratio(port_ret_series)
        cum = (1 + port_ret_series).cumprod()
        mdd = max_drawdown(cum)
        hr = hit_ratio(port_ret_series)
    except Exception:
        sr = (
            float(port_ret_series.mean() / port_ret_series.std() * np.sqrt(252))
            if port_ret_series.std() > 0
            else 0.0
        )
        so = sr  # fallback
        cum = (1 + port_ret_series).cumprod()
        mdd = float((cum / cum.cummax() - 1).min())
        hr = float((port_ret_series > 0).mean())

    ann_vol = float(port_ret_series.std() * np.sqrt(252))
    ann_ret = float((1 + port_ret_series).prod() ** (252 / len(port_ret_series)) - 1)
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0.0

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Sharpe", f"{sr:.2f}", delta_color="normal")
    m2.metric("Sortino", f"{so:.2f}")
    m3.metric("Calmar", f"{calmar:.2f}")
    m4.metric("Max DD", f"{mdd:.1%}", delta_color="inverse")
    m5.metric("Ann. Vol", f"{ann_vol:.1%}")
    m6.metric("Hit Ratio", f"{hr:.1%}")

    st.divider()

    # -- Tabs --------------------------------------------------------------

    tab_decomp, tab_corr, tab_frontier, tab_dd = st.tabs(
        ["Risk Decomposition", "Correlation", "Efficient Frontier", "Drawdown"],
    )

    # ---- Risk Decomposition ----
    with tab_decomp:
        try:
            import plotly.graph_objects as go

            from wraquant.dashboard.components.charts import COLORS, dark_layout

            col_pie, col_bar = st.columns(2)

            # Component VaR
            try:
                from wraquant.risk.portfolio_analytics import component_var

                comp = component_var(w_arr, returns, alpha=1 - confidence)
                comp_values = comp.values if hasattr(comp, "values") else comp
            except Exception:
                # Fallback: proportional to weighted variance
                marginal = cov_matrix @ w_arr
                comp_values = np.abs(w_arr * marginal)
                comp_values = comp_values / comp_values.sum()

            with col_pie:
                st.subheader("Component VaR")
                fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=tickers,
                            values=np.abs(comp_values),
                            hole=0.4,
                            marker={
                                "colors": [
                                    COLORS["primary"],
                                    COLORS["accent2"],
                                    COLORS["accent4"],
                                    COLORS["accent1"],
                                    COLORS["warning"],
                                    COLORS["accent3"],
                                    COLORS["info"],
                                    COLORS["success"],
                                ][: len(tickers)]
                            },
                        )
                    ]
                )
                fig.update_layout(**dark_layout(title="Component VaR Contribution"))
                st.plotly_chart(fig, use_container_width=True)

            # Risk contribution
            try:
                from wraquant.risk.portfolio import risk_contribution

                rc = risk_contribution(w_arr, cov_matrix)
            except Exception:
                marginal = cov_matrix @ w_arr
                port_vol_raw = np.sqrt(w_arr @ cov_matrix @ w_arr)
                rc = (
                    (w_arr * (cov_matrix @ w_arr)) / port_vol_raw
                    if port_vol_raw > 0
                    else w_arr
                )
                rc = rc / rc.sum()

            with col_bar:
                st.subheader("Risk Contribution")
                colors = [
                    COLORS["danger"] if v > 1.5 / len(tickers) else COLORS["primary"]
                    for v in rc
                ]
                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=tickers,
                            y=rc,
                            marker_color=colors,
                            text=[f"{v:.1%}" for v in rc],
                            textposition="auto",
                        )
                    ]
                )
                fig.update_layout(
                    **dark_layout(
                        title="Risk Contribution by Asset", yaxis_title="Fraction"
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

            # Risk metrics table
            st.subheader("Per-Asset Risk Metrics")
            asset_data = []
            for i, t in enumerate(tickers):
                asset_ret = returns[t]
                a_vol = float(asset_ret.std() * np.sqrt(252))
                a_ret = float((1 + asset_ret).prod() ** (252 / len(asset_ret)) - 1)
                a_sr = a_ret / a_vol if a_vol > 0 else 0.0
                asset_data.append(
                    {
                        "Asset": t,
                        "Weight": f"{w_arr[i]:.1%}",
                        "Ann. Return": f"{a_ret:.1%}",
                        "Ann. Vol": f"{a_vol:.1%}",
                        "Sharpe": f"{a_sr:.2f}",
                        "Risk Contrib.": f"{rc[i]:.1%}",
                    }
                )
            st.dataframe(
                pd.DataFrame(asset_data).set_index("Asset"), use_container_width=True
            )

        except ImportError:
            st.warning("Plotly required for interactive charts. `pip install plotly`")

    # ---- Correlation ----
    with tab_corr:
        st.subheader("Correlation Heatmap")
        try:
            import plotly.graph_objects as go

            from wraquant.dashboard.components.charts import COLORS, dark_layout

            corr = returns.corr()
            fig = go.Figure(
                data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns.tolist(),
                    y=corr.index.tolist(),
                    colorscale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    text=corr.values.round(3),
                    texttemplate="%{text}",
                    textfont={"size": 11},
                )
            )
            fig.update_layout(
                **dark_layout(title="Return Correlation Matrix", height=500)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Diversification ratio
            try:
                from wraquant.risk.portfolio import diversification_ratio

                div_ratio = diversification_ratio(w_arr, ann_cov)
            except Exception:
                ind_vols = np.sqrt(np.diag(ann_cov))
                div_ratio = float(w_arr @ ind_vols) / ann_vol if ann_vol > 0 else 1.0

            st.metric(
                "Diversification Ratio",
                f"{div_ratio:.2f}",
                help="Ratio of weighted avg vol to portfolio vol. Higher = better diversified.",
            )

        except ImportError:
            st.dataframe(
                returns.corr().style.format("{:.3f}"), use_container_width=True
            )

    # ---- Efficient Frontier ----
    with tab_frontier:
        st.subheader("Efficient Frontier (Monte Carlo)")
        try:
            import plotly.graph_objects as go

            from wraquant.dashboard.components.charts import COLORS, dark_layout

            rng = np.random.default_rng(0)
            n_portfolios = 2000
            n_assets = len(tickers)
            mu = returns.mean().values * 252
            sigma = ann_cov

            sim_returns_arr = np.zeros(n_portfolios)
            sim_vols = np.zeros(n_portfolios)
            sim_sharpes = np.zeros(n_portfolios)

            for i in range(n_portfolios):
                rw = rng.dirichlet(np.ones(n_assets))
                p_ret = float(rw @ mu)
                p_vol = float(np.sqrt(rw @ sigma @ rw))
                sim_returns_arr[i] = p_ret
                sim_vols[i] = p_vol
                sim_sharpes[i] = p_ret / p_vol if p_vol > 0 else 0.0

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=sim_vols,
                    y=sim_returns_arr,
                    mode="markers",
                    marker={
                        "color": sim_sharpes,
                        "colorscale": "Viridis",
                        "size": 3,
                        "colorbar": {"title": "Sharpe"},
                    },
                    name="Random Portfolios",
                    hovertemplate="Vol: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[ann_vol],
                    y=[ann_ret],
                    mode="markers",
                    marker={"color": COLORS["danger"], "size": 14, "symbol": "star"},
                    name="Current Portfolio",
                )
            )
            fig.update_layout(
                **dark_layout(
                    title="Efficient Frontier (Monte Carlo)",
                    xaxis_title="Annualized Volatility",
                    yaxis_title="Annualized Return",
                )
            )
            fig.update_xaxes(tickformat=".0%")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            st.info("Plotly required for efficient frontier plot.")

    # ---- Drawdown ----
    with tab_dd:
        st.subheader("Drawdown Chart")
        try:
            import plotly.graph_objects as go

            from wraquant.dashboard.components.charts import COLORS, dark_layout

            cum = (1 + port_ret_series).cumprod()
            dd = cum / cum.cummax() - 1

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=dd.index,
                    y=dd.values,
                    fill="tozeroy",
                    line={"color": COLORS["danger"], "width": 1},
                    fillcolor="rgba(239, 68, 68, 0.3)",
                    name="Drawdown",
                )
            )

            # Highlight recovery periods: where drawdown is recovering (dd increasing toward 0)
            # Mark periods where dd < -0.05
            severe_mask = dd < -0.05
            if severe_mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=dd.index[severe_mask],
                        y=dd.values[severe_mask],
                        mode="markers",
                        marker={"color": COLORS["warning"], "size": 2},
                        name="Severe (>5%)",
                        showlegend=True,
                    )
                )

            fig.update_layout(
                **dark_layout(
                    title="Portfolio Drawdown",
                    yaxis_title="Drawdown",
                    height=400,
                )
            )
            fig.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig, use_container_width=True)

            # Drawdown statistics
            try:
                from wraquant.risk.historical import crisis_drawdowns

                crises = crisis_drawdowns(port_ret_series, top_n=5)
                if isinstance(crises, dict) and "drawdowns" in crises:
                    st.subheader("Top 5 Drawdown Events")
                    dd_list = crises["drawdowns"]
                    if isinstance(dd_list, list) and dd_list:
                        dd_rows = []
                        for d in dd_list[:5]:
                            dd_rows.append(
                                {
                                    "Depth": f"{d.get('depth', d.get('max_drawdown', 0)):.1%}",
                                    "Start": str(d.get("start", "")),
                                    "Trough": str(d.get("trough", d.get("valley", ""))),
                                    "End": str(d.get("end", d.get("recovery", ""))),
                                    "Duration (days)": d.get(
                                        "duration", d.get("days", "N/A")
                                    ),
                                }
                            )
                        st.dataframe(pd.DataFrame(dd_rows), use_container_width=True)
            except Exception:
                pass

            # Equity curve alongside
            st.subheader("Equity Curve")
            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=cum.index,
                    y=cum.values,
                    mode="lines",
                    line={"color": COLORS["primary"], "width": 2},
                    name="Portfolio",
                )
            )
            fig2.update_layout(
                **dark_layout(title="Cumulative Return", yaxis_title="Growth of $1")
            )
            st.plotly_chart(fig2, use_container_width=True)

        except ImportError:
            eq = (1 + port_ret_series).cumprod()
            dd = eq / eq.cummax() - 1
            st.area_chart(dd)
