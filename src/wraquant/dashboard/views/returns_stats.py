"""Returns & Statistics page -- deep statistical analysis.

Summary stats, distribution fitting, rolling statistics, beta & factor
analysis, correlation analysis, and cointegration testing. Uses
wraquant.stats, wraquant.risk, and wraquant.ts for computations with
Plotly for interactive charting.
"""

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_returns(ticker: str, days: int = 730):
    """Fetch close prices and compute returns. Returns (close, returns)."""
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
            return close, close.pct_change().dropna()
    except Exception:
        pass

    try:
        import yfinance as yf

        data = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
        if not data.empty:
            close = data["Close"]
            if hasattr(close, "columns"):
                close = close.iloc[:, 0]
            return close, close.pct_change().dropna()
    except Exception:
        pass

    import numpy as np

    rng = np.random.default_rng(42)
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=504)
    rets = pd.Series(rng.normal(0.0004, 0.015, 504), index=idx, name="returns")
    close = (1 + rets).cumprod() * 100
    return close, rets


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_benchmark(ticker: str, days: int = 730):
    """Fetch benchmark returns for beta / correlation analysis."""
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
            df.columns = [c.lower() for c in df.columns]
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
            elif not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except (ValueError, TypeError):
                    pass
            return df["close"].pct_change().dropna()
    except Exception:
        pass

    try:
        import yfinance as yf

        data = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
        if not data.empty:
            close = data["Close"]
            if hasattr(close, "columns"):
                close = close.iloc[:, 0]
            return close.pct_change().dropna()
    except Exception:
        pass

    import numpy as np

    rng = np.random.default_rng(99)
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=504)
    return pd.Series(rng.normal(0.0003, 0.012, 504), index=idx, name="benchmark")


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_multi_returns(tickers: list, days: int = 730):
    """Fetch returns for multiple tickers."""
    import pandas as pd

    try:
        import yfinance as yf

        data = yf.download(tickers, period="2y", auto_adjust=True, progress=False)
        if not data.empty:
            prices = data["Close"] if "Close" in data.columns else data
            return prices.pct_change().dropna()
    except Exception:
        pass

    import numpy as np

    rng = np.random.default_rng(42)
    n = min(504, days)
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    return pd.DataFrame(
        rng.normal(0.0004, 0.015, (n, len(tickers))), index=idx, columns=tickers
    )


def render() -> None:
    """Render the Returns & Statistics page."""
    import numpy as np
    import pandas as pd

    from wraquant.dashboard.components.charts import COLORS, SERIES_COLORS, dark_layout
    from wraquant.dashboard.components.metrics import fmt_pct

    ticker = st.session_state.get("ticker", "AAPL")
    st.markdown(f"# Returns & Statistics: **{ticker}**")

    lookback = st.selectbox(
        "Lookback",
        [365, 730, 1095, 1825],
        index=1,
        format_func=lambda x: {365: "1Y", 730: "2Y", 1095: "3Y", 1825: "5Y"}[x],
        key="rs_lookback",
    )

    with st.spinner(f"Loading {ticker} returns..."):
        close, returns = _fetch_returns(ticker, days=lookback)

    if returns is None or len(returns) < 20:
        st.warning("Insufficient data for statistical analysis.")
        return

    st.caption(f"{len(returns)} daily observations")

    # -- Summary KPIs ----------------------------------------------------------

    ann_ret = float(returns.mean() * 252)
    ann_vol = float(returns.std() * np.sqrt(252))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    sortino_val = 0.0
    try:
        from wraquant.risk.metrics import sortino_ratio

        sortino_val = float(sortino_ratio(returns))
    except Exception:
        downside = returns[returns < 0]
        d_std = float(downside.std() * np.sqrt(252)) if len(downside) > 0 else ann_vol
        sortino_val = ann_ret / d_std if d_std > 0 else 0.0

    cum = (1 + returns).cumprod()
    max_dd = float((cum / cum.cummax() - 1).min())
    hit = float((returns > 0).mean())

    try:
        from scipy import stats as sp_stats

        skew = float(sp_stats.skew(returns))
        kurt = float(sp_stats.kurtosis(returns))
    except ImportError:
        skew = float(returns.skew())
        kurt = float(returns.kurtosis())

    var_95 = float(np.percentile(returns, 5))
    tail = returns[returns <= var_95]
    cvar_95 = float(tail.mean()) if len(tail) > 0 else var_95

    hurst_val = 0.5
    try:
        from wraquant.stats.distributions import hurst_exponent

        h_res = hurst_exponent(returns)
        hurst_val = float(h_res.get("hurst", h_res) if isinstance(h_res, dict) else h_res)
    except Exception:
        pass

    cols = st.columns(11)
    labels = [
        ("Mean", fmt_pct(ann_ret)),
        ("Vol", fmt_pct(ann_vol)),
        ("Skew", f"{skew:.3f}"),
        ("Kurt", f"{kurt:.3f}"),
        ("VaR 95%", fmt_pct(var_95)),
        ("CVaR 95%", fmt_pct(cvar_95)),
        ("Sharpe", f"{sharpe:.2f}"),
        ("Sortino", f"{sortino_val:.2f}"),
        ("Max DD", fmt_pct(max_dd)),
        ("Hit Rate", fmt_pct(hit)),
        ("Hurst", f"{hurst_val:.3f}"),
    ]
    for c, (lbl, val) in zip(cols, labels, strict=False):
        c.metric(lbl, val)

    st.divider()

    # -- Tabs ------------------------------------------------------------------

    (
        tab_summary,
        tab_dist,
        tab_rolling,
        tab_beta,
        tab_corr,
        tab_coint,
    ) = st.tabs([
        "Summary Stats",
        "Distribution",
        "Rolling Stats",
        "Beta & Factor",
        "Correlation",
        "Cointegration",
    ])

    # ---- Summary Stats Tab ----
    with tab_summary:
        st.subheader("Summary Statistics")
        summary = {
            "Mean (daily)": f"{float(returns.mean()):.6f}",
            "Mean (annualized)": f"{ann_ret:.2%}",
            "Std Dev (daily)": f"{float(returns.std()):.6f}",
            "Std Dev (annualized)": f"{ann_vol:.2%}",
            "Skewness": f"{skew:.4f}",
            "Excess Kurtosis": f"{kurt:.4f}",
            "Sharpe Ratio": f"{sharpe:.3f}",
            "Sortino Ratio": f"{sortino_val:.3f}",
            "Calmar Ratio": f"{ann_ret / abs(max_dd):.3f}" if max_dd != 0 else "N/A",
            "Max Drawdown": f"{max_dd:.2%}",
            "VaR (95%)": f"{var_95:.4%}",
            "CVaR (95%)": f"{cvar_95:.4%}",
            "Hit Rate": f"{hit:.1%}",
            "Hurst Exponent": f"{hurst_val:.4f}",
            "Min Return": f"{float(returns.min()):.4%}",
            "Max Return": f"{float(returns.max()):.4%}",
            "Median Return": f"{float(returns.median()):.6f}",
            "Observations": f"{len(returns):,}",
            "First Date": str(returns.index[0].date()),
            "Last Date": str(returns.index[-1].date()),
        }

        c1, c2 = st.columns(2)
        items = list(summary.items())
        mid = (len(items) + 1) // 2
        with c1:
            for k, v in items[:mid]:
                st.markdown(f"**{k}:** {v}")
        with c2:
            for k, v in items[mid:]:
                st.markdown(f"**{k}:** {v}")

        st.divider()
        summary_df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in summary.items()])
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

    # ---- Distribution Tab ----
    with tab_dist:
        try:
            import plotly.graph_objects as go

            col_hist, col_qq = st.columns(2)

            mu_r = float(returns.mean())
            sigma_r = float(returns.std())

            with col_hist:
                st.subheader("Return Distribution")
                fig = go.Figure()
                fig.add_trace(
                    go.Histogram(
                        x=returns.values,
                        nbinsx=80,
                        name="Returns",
                        marker_color=COLORS["primary"],
                        opacity=0.7,
                        histnorm="probability density",
                    )
                )

                x_range = np.linspace(float(returns.min()), float(returns.max()), 200)
                try:
                    from scipy.stats import norm, t as t_dist

                    normal_pdf = norm.pdf(x_range, mu_r, sigma_r)
                    df_t, loc_t, scale_t = t_dist.fit(returns)
                    t_pdf = t_dist.pdf(x_range, df_t, loc_t, scale_t)
                except ImportError:
                    normal_pdf = (
                        1
                        / (sigma_r * np.sqrt(2 * np.pi))
                        * np.exp(-0.5 * ((x_range - mu_r) / sigma_r) ** 2)
                    )
                    t_pdf = None

                fig.add_trace(
                    go.Scatter(
                        x=x_range, y=normal_pdf, mode="lines",
                        name="Normal Fit",
                        line={"color": COLORS["danger"], "width": 2, "dash": "dash"},
                    )
                )
                if t_pdf is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=x_range, y=t_pdf, mode="lines",
                            name=f"t-dist (df={df_t:.1f})",
                            line={"color": COLORS["accent4"], "width": 2, "dash": "dot"},
                        )
                    )

                fig.add_vline(
                    x=var_95, line_dash="dash", line_color=COLORS["warning"],
                    annotation_text=f"VaR 95%: {var_95:.2%}",
                )
                fig.update_layout(
                    **dark_layout(
                        title="Return Distribution with Normal & t Overlay",
                        xaxis_title="Daily Return", yaxis_title="Density",
                        height=450,
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_qq:
                st.subheader("Q-Q Plot")
                try:
                    from scipy import stats as sp_stats

                    sorted_rets = np.sort(returns.values)
                    n = len(sorted_rets)
                    theoretical = sp_stats.norm.ppf(np.linspace(0.001, 0.999, n))

                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=theoretical, y=sorted_rets, mode="markers",
                            marker={"color": COLORS["primary"], "size": 3, "opacity": 0.6},
                            name="Returns",
                        )
                    )
                    min_t, max_t = float(theoretical.min()), float(theoretical.max())
                    fig.add_trace(
                        go.Scatter(
                            x=[min_t, max_t],
                            y=[min_t * sigma_r + mu_r, max_t * sigma_r + mu_r],
                            mode="lines",
                            line={"color": COLORS["danger"], "dash": "dash"},
                            name="Normal Line",
                        )
                    )
                    fig.update_layout(
                        **dark_layout(
                            title="Q-Q Plot vs Normal",
                            xaxis_title="Theoretical Quantiles",
                            yaxis_title="Sample Quantiles",
                            height=450,
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.info("scipy required for Q-Q plot.")

            # Jarque-Bera test result
            st.subheader("Normality Test")
            try:
                from wraquant.stats.distributions import jarque_bera

                jb = jarque_bera(returns)
                jb_stat = jb.get("statistic", jb.get("jb_statistic", 0))
                jb_p = jb.get("p_value", 0)
            except Exception:
                try:
                    from scipy.stats import jarque_bera as jb_scipy

                    jb_stat, jb_p = jb_scipy(returns.dropna())
                except Exception:
                    jb_stat, jb_p = None, None

            if jb_stat is not None:
                jc1, jc2, jc3 = st.columns(3)
                jc1.metric("JB Statistic", f"{jb_stat:.2f}")
                jc2.metric("p-value", f"{jb_p:.6f}")
                jc3.metric(
                    "Result",
                    "Normal" if jb_p >= 0.05 else "Non-normal",
                )

            # Best-fit distribution
            st.subheader("Best-Fit Distribution")
            try:
                from wraquant.stats.distributions import best_fit_distribution

                best = best_fit_distribution(returns)
                if isinstance(best, dict):
                    st.dataframe(
                        pd.DataFrame([best]),
                        use_container_width=True,
                    )
                else:
                    st.write(best)
            except Exception:
                try:
                    from scipy import stats as sp_stats

                    fits = {}
                    for name, dist in [
                        ("Normal", sp_stats.norm),
                        ("Student-t", sp_stats.t),
                        ("Laplace", sp_stats.laplace),
                    ]:
                        try:
                            params = dist.fit(returns)
                            ks = sp_stats.kstest(returns, name.lower().replace("-", ""), args=params)
                            fits[name] = {
                                "KS Stat": f"{ks.statistic:.4f}",
                                "p-value": f"{ks.pvalue:.4f}",
                            }
                        except Exception:
                            pass
                    if fits:
                        st.dataframe(pd.DataFrame(fits).T, use_container_width=True)
                except ImportError:
                    st.info("scipy required for distribution fitting.")

        except ImportError:
            st.warning("Plotly required for charts. `pip install plotly`")

    # ---- Rolling Stats Tab ----
    with tab_rolling:
        st.subheader("Rolling Statistics")
        window = st.select_slider(
            "Rolling Window (days)",
            options=[20, 60, 120, 252],
            value=60,
            key="rs_roll_window",
        )
        benchmark_ticker = st.text_input(
            "Benchmark for rolling beta/Sharpe", value="SPY", key="rs_bench"
        )

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            rolling_mean = returns.rolling(window).mean() * 252
            rolling_vol = returns.rolling(window).std() * np.sqrt(252)
            rolling_skew = returns.rolling(window).skew()
            rolling_kurt = returns.rolling(window).kurt()

            # Rolling Sharpe
            r_sharpe = rolling_mean / rolling_vol

            fig = make_subplots(
                rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                row_heights=[0.2] * 5,
                subplot_titles=[
                    f"Rolling {window}d Mean (Ann.)",
                    f"Rolling {window}d Vol (Ann.)",
                    f"Rolling {window}d Skewness",
                    f"Rolling {window}d Kurtosis",
                    f"Rolling {window}d Sharpe",
                ],
            )

            series_data = [
                (rolling_mean, COLORS["primary"]),
                (rolling_vol, COLORS["accent2"]),
                (rolling_skew, COLORS["accent4"]),
                (rolling_kurt, COLORS["accent1"]),
                (r_sharpe, COLORS["warning"]),
            ]

            for i, (s, color) in enumerate(series_data, 1):
                s_clean = s.dropna()
                fig.add_trace(
                    go.Scatter(
                        x=s_clean.index, y=s_clean.values, mode="lines",
                        line={"color": color, "width": 1.5}, showlegend=False,
                    ),
                    row=i, col=1,
                )
                fig.add_hline(
                    y=0, line_dash="dot", line_color=COLORS["neutral"],
                    opacity=0.3, row=i, col=1,
                )

            fig.update_layout(**dark_layout(title=f"Rolling Statistics ({window}d)", height=900))
            for r in range(1, 6):
                fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", row=r, col=1)
                fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", row=r, col=1)
            st.plotly_chart(fig, use_container_width=True)

            # Rolling beta vs benchmark
            st.subheader(f"Rolling {window}d Beta vs {benchmark_ticker}")
            with st.spinner(f"Loading {benchmark_ticker}..."):
                bench_ret = _fetch_benchmark(benchmark_ticker, days=lookback)
            if bench_ret is not None and len(bench_ret) > window:
                aligned = pd.DataFrame(
                    {"asset": returns, "bench": bench_ret}
                ).dropna()
                if len(aligned) > window:
                    r_beta = aligned["asset"].rolling(window).cov(aligned["bench"]) / aligned[
                        "bench"
                    ].rolling(window).var()
                    r_beta = r_beta.dropna()
                    fig_b = go.Figure()
                    fig_b.add_trace(
                        go.Scatter(
                            x=r_beta.index, y=r_beta.values, mode="lines",
                            line={"color": COLORS["accent2"], "width": 2},
                            name="Rolling Beta",
                        )
                    )
                    fig_b.add_hline(y=1, line_dash="dash", line_color=COLORS["neutral"])
                    fig_b.update_layout(
                        **dark_layout(
                            title=f"Rolling {window}d Beta vs {benchmark_ticker}",
                            yaxis_title="Beta", height=350,
                        )
                    )
                    st.plotly_chart(fig_b, use_container_width=True)

        except ImportError:
            st.warning("Plotly required for rolling statistics charts.")

    # ---- Beta & Factor Tab ----
    with tab_beta:
        st.subheader("Beta Analysis")
        bench_ticker_beta = st.text_input(
            "Benchmark", value="SPY", key="rs_beta_bench"
        )

        with st.spinner(f"Computing beta vs {bench_ticker_beta}..."):
            bench_ret = _fetch_benchmark(bench_ticker_beta, days=lookback)

        if bench_ret is None or len(bench_ret) < 60:
            st.warning("Could not load benchmark data for beta analysis.")
        else:
            aligned = pd.DataFrame({"asset": returns, "bench": bench_ret}).dropna()
            if len(aligned) < 60:
                st.warning("Insufficient overlapping data for beta analysis.")
            else:
                asset_r = aligned["asset"]
                bench_r = aligned["bench"]
                ols_beta = float(np.cov(asset_r, bench_r)[0, 1] / np.var(bench_r))
                ols_alpha = float(asset_r.mean() - ols_beta * bench_r.mean())

                # Up/down beta
                up_mask = bench_r > 0
                dn_mask = bench_r < 0
                up_beta = (
                    float(
                        np.cov(asset_r[up_mask], bench_r[up_mask])[0, 1]
                        / np.var(bench_r[up_mask])
                    )
                    if up_mask.sum() > 10
                    else ols_beta
                )
                dn_beta = (
                    float(
                        np.cov(asset_r[dn_mask], bench_r[dn_mask])[0, 1]
                        / np.var(bench_r[dn_mask])
                    )
                    if dn_mask.sum() > 10
                    else ols_beta
                )

                # Blume-adjusted beta
                blume_beta = 0.33 + 0.67 * ols_beta

                # R-squared
                ss_res = float(np.sum((asset_r - ols_alpha - ols_beta * bench_r) ** 2))
                ss_tot = float(np.sum((asset_r - asset_r.mean()) ** 2))
                r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

                try:
                    from wraquant.risk.beta import (
                        blume_adjusted_beta,
                        conditional_beta,
                        rolling_beta as wrq_rolling_beta,
                    )

                    blume_beta = float(blume_adjusted_beta(asset_r, bench_r))
                    cond = conditional_beta(asset_r, bench_r)
                    if isinstance(cond, dict):
                        up_beta = float(cond.get("up_beta", up_beta))
                        dn_beta = float(cond.get("down_beta", dn_beta))
                except Exception:
                    pass

                bc1, bc2, bc3, bc4, bc5 = st.columns(5)
                bc1.metric("OLS Beta", f"{ols_beta:.3f}")
                bc2.metric("Up Beta", f"{up_beta:.3f}")
                bc3.metric("Down Beta", f"{dn_beta:.3f}")
                bc4.metric("Blume-Adj Beta", f"{blume_beta:.3f}")
                bc5.metric("R-squared", f"{r_squared:.3f}")

                # Rolling beta chart
                try:
                    import plotly.graph_objects as go

                    r_beta = asset_r.rolling(60).cov(bench_r) / bench_r.rolling(60).var()
                    r_beta = r_beta.dropna()
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=r_beta.index, y=r_beta.values, mode="lines",
                            line={"color": COLORS["primary"], "width": 2}, name="60d Beta",
                        )
                    )
                    fig.add_hline(y=ols_beta, line_dash="dash", line_color=COLORS["danger"],
                                  annotation_text=f"Full-period: {ols_beta:.2f}")
                    fig.add_hline(y=1.0, line_dash="dot", line_color=COLORS["neutral"])
                    fig.update_layout(
                        **dark_layout(
                            title=f"Rolling 60d Beta vs {bench_ticker_beta}",
                            yaxis_title="Beta", height=400,
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    pass

                # Factor exposure
                st.divider()
                st.subheader("Factor Exposure (Fama-French)")
                try:
                    from wraquant.stats.factor import fama_french_regression

                    ff = fama_french_regression(asset_r)
                    if isinstance(ff, dict):
                        ff_rows = []
                        for key in ["alpha", "mkt_rf", "smb", "hml"]:
                            if key in ff:
                                ff_rows.append({
                                    "Factor": key.upper(),
                                    "Coefficient": f"{ff[key]:.4f}",
                                    "t-stat": f"{ff.get(f'{key}_tstat', 'N/A')}",
                                    "p-value": f"{ff.get(f'{key}_pvalue', 'N/A')}",
                                })
                        if "coefficients" in ff:
                            for k, v in ff["coefficients"].items():
                                ff_rows.append({
                                    "Factor": k,
                                    "Coefficient": f"{v:.4f}",
                                    "t-stat": "N/A",
                                    "p-value": "N/A",
                                })
                        if ff_rows:
                            st.dataframe(
                                pd.DataFrame(ff_rows).set_index("Factor"),
                                use_container_width=True,
                            )
                        r2_ff = ff.get("r_squared", ff.get("r2", "N/A"))
                        st.metric("Factor R-squared", f"{r2_ff}")
                except Exception as e:
                    st.info(f"Fama-French data unavailable: {e}")

    # ---- Correlation Tab ----
    with tab_corr:
        st.subheader("Multi-Asset Correlation")
        other_tickers = st.text_input(
            "Additional tickers (comma-separated)",
            value="SPY, QQQ, TLT, GLD, IWM",
            key="rs_corr_tickers",
        )

        if other_tickers:
            all_tickers = [ticker] + [
                t.strip().upper() for t in other_tickers.split(",") if t.strip()
            ]

            with st.spinner("Fetching multi-asset data..."):
                multi_returns = _fetch_multi_returns(all_tickers, days=lookback)

            if multi_returns is not None and not multi_returns.empty:
                try:
                    import plotly.graph_objects as go

                    # Correlation matrix heatmap
                    st.subheader("Correlation Matrix")
                    corr = multi_returns.corr()
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

                    # Rolling correlation vs benchmark
                    st.subheader(f"Rolling Correlation vs {all_tickers[1] if len(all_tickers) > 1 else 'SPY'}")
                    bench_col = all_tickers[1] if len(all_tickers) > 1 else all_tickers[0]
                    if bench_col in multi_returns.columns and ticker in multi_returns.columns:
                        rc_window = st.select_slider(
                            "Rolling Window", options=[20, 60, 120, 252], value=60,
                            key="rs_rc_window",
                        )
                        roll_corr = multi_returns[ticker].rolling(rc_window).corr(
                            multi_returns[bench_col]
                        )
                        fig_rc = go.Figure()
                        fig_rc.add_trace(
                            go.Scatter(
                                x=roll_corr.dropna().index,
                                y=roll_corr.dropna().values,
                                mode="lines",
                                line={"color": COLORS["primary"], "width": 2},
                                name=f"Corr({ticker}, {bench_col})",
                            )
                        )
                        fig_rc.update_layout(
                            **dark_layout(
                                title=f"Rolling {rc_window}d Correlation: {ticker} vs {bench_col}",
                                yaxis_title="Correlation", height=350,
                            )
                        )
                        st.plotly_chart(fig_rc, use_container_width=True)

                    # Distance correlation
                    st.subheader("Distance Correlation")
                    try:
                        from wraquant.stats.correlation import distance_correlation

                        dcorr_results = {}
                        for col in multi_returns.columns:
                            if col != ticker and ticker in multi_returns.columns:
                                try:
                                    dc = distance_correlation(
                                        multi_returns[ticker], multi_returns[col]
                                    )
                                    dcorr_results[col] = float(
                                        dc.get("distance_correlation", dc) if isinstance(dc, dict) else dc
                                    )
                                except Exception:
                                    pass
                        if dcorr_results:
                            dc_df = pd.DataFrame(
                                [{"Asset": k, "Distance Corr": f"{v:.4f}"} for k, v in dcorr_results.items()]
                            )
                            st.dataframe(dc_df, hide_index=True, use_container_width=True)
                    except Exception:
                        st.info("Distance correlation requires wraquant.stats.correlation.")

                    # Partial correlation
                    st.subheader("Partial Correlation")
                    try:
                        from wraquant.stats.correlation import partial_correlation

                        pcorr = partial_correlation(multi_returns)
                        if isinstance(pcorr, pd.DataFrame):
                            fig_pc = go.Figure(
                                data=go.Heatmap(
                                    z=pcorr.values, x=pcorr.columns.tolist(),
                                    y=pcorr.index.tolist(), colorscale="RdBu_r",
                                    zmin=-1, zmax=1,
                                    text=pcorr.values.round(3), texttemplate="%{text}",
                                    textfont={"size": 11},
                                )
                            )
                            fig_pc.update_layout(
                                **dark_layout(title="Partial Correlation Matrix", height=500)
                            )
                            st.plotly_chart(fig_pc, use_container_width=True)
                        elif isinstance(pcorr, dict) and "matrix" in pcorr:
                            mat = pcorr["matrix"]
                            if isinstance(mat, pd.DataFrame):
                                st.dataframe(mat.style.format("{:.3f}"), use_container_width=True)
                    except Exception:
                        st.info("Partial correlation requires wraquant.stats.correlation.")

                except ImportError:
                    st.dataframe(
                        multi_returns.corr().style.format("{:.3f}"),
                        use_container_width=True,
                    )

    # ---- Cointegration Tab ----
    with tab_coint:
        st.subheader("Cointegration Analysis")
        st.markdown(
            "Test two assets for cointegration using the Engle-Granger method."
        )

        coint_ticker = st.text_input(
            "Second ticker for cointegration test",
            value="MSFT",
            key="rs_coint_ticker",
        )

        if coint_ticker:
            with st.spinner(f"Testing cointegration: {ticker} vs {coint_ticker}..."):
                close_b, returns_b = _fetch_returns(coint_ticker.strip().upper(), days=lookback)

            if close_b is not None and len(close_b) > 60:
                aligned_prices = pd.DataFrame(
                    {ticker: close, coint_ticker.upper(): close_b}
                ).dropna()

                if len(aligned_prices) < 60:
                    st.warning("Insufficient overlapping price data.")
                else:
                    price_a = aligned_prices[ticker]
                    price_b = aligned_prices[coint_ticker.upper()]

                    try:
                        from wraquant.stats.cointegration import (
                            engle_granger,
                            half_life,
                            hedge_ratio,
                            spread,
                            zscore_signal,
                        )

                        eg = engle_granger(price_a, price_b)
                        hr = hedge_ratio(price_a, price_b)
                        sp = spread(price_a, price_b)
                        hl = half_life(sp)
                        zs = zscore_signal(sp)

                        eg_stat = eg.get("test_statistic", eg.get("adf_statistic", 0))
                        eg_p = eg.get("p_value", 0)
                        eg_coint = eg.get("cointegrated", eg_p < 0.05)
                        hr_val = hr.get("hedge_ratio", hr) if isinstance(hr, dict) else hr
                        hl_val = hl.get("half_life", hl) if isinstance(hl, dict) else hl

                        ec1, ec2, ec3, ec4 = st.columns(4)
                        ec1.metric("EG Statistic", f"{eg_stat:.4f}")
                        ec2.metric("p-value", f"{eg_p:.4f}")
                        ec3.metric(
                            "Cointegrated",
                            "Yes" if eg_coint else "No",
                        )
                        ec4.metric("Hedge Ratio", f"{float(hr_val):.4f}")

                        ec5, ec6 = st.columns(2)
                        ec5.metric("Half-Life (days)", f"{float(hl_val):.1f}")
                        zs_current = float(zs.iloc[-1]) if hasattr(zs, "iloc") else 0.0
                        ec6.metric("Current Z-Score", f"{zs_current:.2f}")

                        # Spread chart
                        try:
                            import plotly.graph_objects as go
                            from plotly.subplots import make_subplots

                            fig = make_subplots(
                                rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.08,
                                subplot_titles=["Spread", "Z-Score"],
                                row_heights=[0.5, 0.5],
                            )

                            sp_vals = sp if isinstance(sp, pd.Series) else pd.Series(sp)
                            fig.add_trace(
                                go.Scatter(
                                    x=sp_vals.index, y=sp_vals.values,
                                    mode="lines",
                                    line={"color": COLORS["primary"], "width": 1.5},
                                    name="Spread",
                                ),
                                row=1, col=1,
                            )
                            fig.add_hline(
                                y=float(sp_vals.mean()), line_dash="dash",
                                line_color=COLORS["neutral"], row=1, col=1,
                            )

                            zs_vals = zs if isinstance(zs, pd.Series) else pd.Series(zs)
                            fig.add_trace(
                                go.Scatter(
                                    x=zs_vals.index, y=zs_vals.values,
                                    mode="lines",
                                    line={"color": COLORS["accent2"], "width": 1.5},
                                    name="Z-Score",
                                ),
                                row=2, col=1,
                            )
                            for lvl, color in [
                                (2.0, COLORS["danger"]),
                                (-2.0, COLORS["danger"]),
                                (1.0, COLORS["warning"]),
                                (-1.0, COLORS["warning"]),
                            ]:
                                fig.add_hline(
                                    y=lvl, line_dash="dot", line_color=color,
                                    opacity=0.5, row=2, col=1,
                                )
                            fig.add_hline(
                                y=0, line_dash="dash", line_color=COLORS["neutral"],
                                row=2, col=1,
                            )

                            fig.update_layout(
                                **dark_layout(
                                    title=f"Spread & Z-Score: {ticker} vs {coint_ticker.upper()}",
                                    height=600,
                                )
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        except ImportError:
                            pass

                    except Exception as e:
                        st.warning(f"Could not run cointegration test: {e}")

                        # Fallback: simple OLS-based spread
                        try:
                            from scipy import stats as sp_stats

                            slope, intercept, r_val, p_val, _ = sp_stats.linregress(
                                price_b.values, price_a.values
                            )
                            sp_fallback = price_a - slope * price_b - intercept
                            st.metric("OLS Hedge Ratio", f"{slope:.4f}")
                            st.line_chart(sp_fallback)
                        except Exception:
                            st.info("scipy required for fallback cointegration.")
            else:
                st.warning(f"Could not load data for {coint_ticker}.")
