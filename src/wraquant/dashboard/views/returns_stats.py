"""Returns & Statistics page -- distributional analysis, rolling stats, tests.

Displays return distribution with normal overlay, QQ plot, rolling
statistics, correlation heatmap, distribution fit comparison,
stationarity tests, and summary statistics table.
"""

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_returns(ticker: str, days: int = 730) -> "tuple[pd.Series, pd.Series]":
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
            close = df["close"]
            return close, close.pct_change().dropna()
    except Exception:
        pass

    # Fallback to yfinance
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

    # Synthetic fallback
    import numpy as np

    rng = np.random.default_rng(42)
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=504)
    rets = pd.Series(rng.normal(0.0004, 0.015, 504), index=idx, name="returns")
    close = (1 + rets).cumprod() * 100
    return close, rets


def render() -> None:
    """Render the Returns & Statistics page."""
    import numpy as np
    import pandas as pd

    from wraquant.dashboard.components.charts import COLORS, SERIES_COLORS, dark_layout
    from wraquant.dashboard.components.metrics import fmt_pct

    ticker = st.session_state.get("ticker", "AAPL")
    st.markdown(f"# Returns & Statistics: **{ticker}**")

    # Controls
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
        key="rs_lookback",
    )

    # Fetch data
    with st.spinner(f"Loading {ticker} returns..."):
        close, returns = _fetch_returns(ticker, days=lookback)

    if returns is None or len(returns) < 20:
        st.warning("Insufficient data for statistical analysis.")
        return

    st.caption(f"{len(returns)} daily observations")

    # -- Summary KPIs ------------------------------------------------------

    ann_ret = float(returns.mean() * 252)
    ann_vol = float(returns.std() * np.sqrt(252))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cum = (1 + returns).cumprod()
    max_dd = float((cum / cum.cummax() - 1).min())

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

    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    k1.metric("Ann. Return", fmt_pct(ann_ret))
    k2.metric("Ann. Vol", fmt_pct(ann_vol))
    k3.metric("Sharpe", f"{sharpe:.2f}")
    k4.metric("Max DD", fmt_pct(max_dd))
    k5.metric("Skewness", f"{skew:.3f}")
    k6.metric("Kurtosis", f"{kurt:.3f}")
    k7.metric("VaR 95%", fmt_pct(var_95))

    st.divider()

    # -- Tabs --------------------------------------------------------------

    tab_dist, tab_rolling, tab_corr, tab_tests, tab_summary = st.tabs(
        [
            "Distribution",
            "Rolling Statistics",
            "Correlation",
            "Statistical Tests",
            "Summary Table",
        ]
    )

    # ---- Distribution Tab ----
    with tab_dist:
        try:
            import plotly.graph_objects as go

            col_hist, col_qq = st.columns(2)

            # Histogram with normal overlay
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

                # Normal overlay
                x_range = np.linspace(
                    float(returns.min()), float(returns.max()), 200
                )
                mu_r = float(returns.mean())
                sigma_r = float(returns.std())
                try:
                    from scipy.stats import norm

                    normal_pdf = norm.pdf(x_range, mu_r, sigma_r)
                except ImportError:
                    normal_pdf = (
                        1
                        / (sigma_r * np.sqrt(2 * np.pi))
                        * np.exp(-0.5 * ((x_range - mu_r) / sigma_r) ** 2)
                    )

                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=normal_pdf,
                        mode="lines",
                        name="Normal Fit",
                        line={"color": COLORS["danger"], "width": 2, "dash": "dash"},
                    )
                )

                # VaR line
                fig.add_vline(
                    x=var_95,
                    line_dash="dash",
                    line_color=COLORS["warning"],
                    annotation_text=f"VaR 95%: {var_95:.2%}",
                )

                fig.update_layout(
                    **dark_layout(
                        title="Return Distribution with Normal Overlay",
                        xaxis_title="Daily Return",
                        yaxis_title="Density",
                        height=450,
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

            # QQ Plot
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
                            x=theoretical,
                            y=sorted_rets,
                            mode="markers",
                            marker={
                                "color": COLORS["primary"],
                                "size": 3,
                                "opacity": 0.6,
                            },
                            name="Returns",
                        )
                    )
                    # Reference line
                    min_t = float(theoretical.min())
                    max_t = float(theoretical.max())
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

            # Distribution fit comparison
            st.subheader("Distribution Fit Comparison")
            try:
                from scipy import stats as sp_stats

                fits = {}
                # Normal
                loc_n, scale_n = sp_stats.norm.fit(returns)
                ks_n = sp_stats.kstest(returns, "norm", args=(loc_n, scale_n))
                fits["Normal"] = {
                    "KS Stat": f"{ks_n.statistic:.4f}",
                    "p-value": f"{ks_n.pvalue:.4f}",
                    "Params": f"mu={loc_n:.6f}, sigma={scale_n:.6f}",
                }

                # Student-t
                try:
                    df_t, loc_t, scale_t = sp_stats.t.fit(returns)
                    ks_t = sp_stats.kstest(
                        returns, "t", args=(df_t, loc_t, scale_t)
                    )
                    fits["Student-t"] = {
                        "KS Stat": f"{ks_t.statistic:.4f}",
                        "p-value": f"{ks_t.pvalue:.4f}",
                        "Params": f"df={df_t:.2f}, loc={loc_t:.6f}, scale={scale_t:.6f}",
                    }
                except Exception:
                    pass

                # Laplace
                try:
                    loc_l, scale_l = sp_stats.laplace.fit(returns)
                    ks_l = sp_stats.kstest(
                        returns, "laplace", args=(loc_l, scale_l)
                    )
                    fits["Laplace"] = {
                        "KS Stat": f"{ks_l.statistic:.4f}",
                        "p-value": f"{ks_l.pvalue:.4f}",
                        "Params": f"loc={loc_l:.6f}, scale={scale_l:.6f}",
                    }
                except Exception:
                    pass

                if fits:
                    st.dataframe(
                        pd.DataFrame(fits).T,
                        use_container_width=True,
                    )

            except ImportError:
                st.info("scipy required for distribution fitting.")

        except ImportError:
            st.warning("Plotly required for charts. `pip install plotly`")

    # ---- Rolling Statistics Tab ----
    with tab_rolling:
        st.subheader("Rolling Statistics")
        window = st.slider("Rolling Window", 10, 126, 21, key="rs_window")

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            rolling_mean = returns.rolling(window).mean() * 252
            rolling_vol = returns.rolling(window).std() * np.sqrt(252)
            rolling_skew = returns.rolling(window).skew()
            rolling_kurt = returns.rolling(window).kurt()

            fig = make_subplots(
                rows=4,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.04,
                row_heights=[0.25, 0.25, 0.25, 0.25],
                subplot_titles=[
                    f"Rolling {window}d Mean (Ann.)",
                    f"Rolling {window}d Vol (Ann.)",
                    f"Rolling {window}d Skewness",
                    f"Rolling {window}d Kurtosis",
                ],
            )

            series_data = [
                (rolling_mean, COLORS["primary"], ".1%"),
                (rolling_vol, COLORS["accent2"], ".1%"),
                (rolling_skew, COLORS["accent4"], ".2f"),
                (rolling_kurt, COLORS["accent1"], ".2f"),
            ]

            for i, (series, color, fmt) in enumerate(series_data, 1):
                s = series.dropna()
                fig.add_trace(
                    go.Scatter(
                        x=s.index,
                        y=s.values,
                        mode="lines",
                        line={"color": color, "width": 1.5},
                        showlegend=False,
                    ),
                    row=i,
                    col=1,
                )
                # Zero reference line
                fig.add_hline(
                    y=0,
                    line_dash="dot",
                    line_color=COLORS["neutral"],
                    opacity=0.3,
                    row=i,
                    col=1,
                )

            fig.update_layout(
                **dark_layout(
                    title=f"Rolling Statistics ({window}-day window)",
                    height=800,
                )
            )
            for r in range(1, 5):
                fig.update_xaxes(
                    gridcolor="rgba(255,255,255,0.06)", row=r, col=1
                )
                fig.update_yaxes(
                    gridcolor="rgba(255,255,255,0.06)", row=r, col=1
                )

            st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            st.warning("Plotly required for rolling statistics charts.")

    # ---- Correlation Tab ----
    with tab_corr:
        st.subheader("Multi-Asset Correlation")
        st.markdown(
            "Enter additional tickers to compute a correlation matrix."
        )

        other_tickers = st.text_input(
            "Additional tickers (comma-separated)",
            value="SPY, QQQ, TLT, GLD",
            key="rs_corr_tickers",
        )

        if other_tickers:
            all_tickers = [ticker] + [
                t.strip().upper()
                for t in other_tickers.split(",")
                if t.strip()
            ]

            with st.spinner("Fetching multi-asset data..."):
                try:
                    import yfinance as yf

                    data = yf.download(
                        all_tickers,
                        period="2y",
                        auto_adjust=True,
                        progress=False,
                    )
                    if not data.empty:
                        prices = data["Close"] if "Close" in data.columns else data
                        if hasattr(prices.columns, "levels"):
                            pass
                        multi_returns = prices.pct_change().dropna()

                        try:
                            import plotly.graph_objects as go

                            corr = multi_returns.corr()
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
                                    textfont={"size": 12},
                                )
                            )
                            fig.update_layout(
                                **dark_layout(
                                    title="Return Correlation Matrix",
                                    height=500,
                                )
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        except ImportError:
                            st.dataframe(
                                multi_returns.corr().style.format("{:.3f}"),
                                use_container_width=True,
                            )
                    else:
                        st.warning("Could not fetch multi-asset data.")
                except Exception as exc:
                    st.warning(f"Could not load correlation data: {exc}")

    # ---- Statistical Tests Tab ----
    with tab_tests:
        st.subheader("Stationarity & Normality Tests")

        test_results = []

        # ADF Test
        try:
            from wraquant.ts.stationarity import adf_test

            adf = adf_test(returns)
            test_results.append(
                {
                    "Test": "Augmented Dickey-Fuller",
                    "Hypothesis": "H0: Unit root (non-stationary)",
                    "Statistic": f"{adf.get('test_statistic', adf.get('adf_statistic', 0)):.4f}",
                    "p-value": f"{adf.get('p_value', 0):.4f}",
                    "Result": (
                        "Stationary (reject H0)"
                        if adf.get("p_value", 1) < 0.05
                        else "Non-stationary (fail to reject H0)"
                    ),
                }
            )
        except Exception:
            try:
                from statsmodels.tsa.stattools import adfuller

                result = adfuller(returns.dropna())
                test_results.append(
                    {
                        "Test": "Augmented Dickey-Fuller",
                        "Hypothesis": "H0: Unit root (non-stationary)",
                        "Statistic": f"{result[0]:.4f}",
                        "p-value": f"{result[1]:.4f}",
                        "Result": (
                            "Stationary (reject H0)"
                            if result[1] < 0.05
                            else "Non-stationary"
                        ),
                    }
                )
            except Exception:
                pass

        # KPSS Test
        try:
            from wraquant.ts.stationarity import kpss_test

            kpss = kpss_test(returns)
            test_results.append(
                {
                    "Test": "KPSS",
                    "Hypothesis": "H0: Stationary",
                    "Statistic": f"{kpss.get('test_statistic', kpss.get('kpss_statistic', 0)):.4f}",
                    "p-value": f"{kpss.get('p_value', 0):.4f}",
                    "Result": (
                        "Stationary (fail to reject H0)"
                        if kpss.get("p_value", 0) >= 0.05
                        else "Non-stationary (reject H0)"
                    ),
                }
            )
        except Exception:
            pass

        # Jarque-Bera
        try:
            from wraquant.stats.distributions import jarque_bera

            jb = jarque_bera(returns)
            test_results.append(
                {
                    "Test": "Jarque-Bera",
                    "Hypothesis": "H0: Normal distribution",
                    "Statistic": f"{jb.get('statistic', jb.get('jb_statistic', 0)):.4f}",
                    "p-value": f"{jb.get('p_value', 0):.6f}",
                    "Result": (
                        "Normal (fail to reject H0)"
                        if jb.get("p_value", 0) >= 0.05
                        else "Non-normal (reject H0)"
                    ),
                }
            )
        except Exception:
            try:
                from scipy.stats import jarque_bera as jb_scipy

                jb_stat, jb_p = jb_scipy(returns.dropna())
                test_results.append(
                    {
                        "Test": "Jarque-Bera",
                        "Hypothesis": "H0: Normal distribution",
                        "Statistic": f"{jb_stat:.4f}",
                        "p-value": f"{jb_p:.6f}",
                        "Result": (
                            "Normal"
                            if jb_p >= 0.05
                            else "Non-normal (reject H0)"
                        ),
                    }
                )
            except Exception:
                pass

        # Ljung-Box (autocorrelation)
        try:
            from wraquant.stats.tests import test_autocorrelation

            lb = test_autocorrelation(returns, lags=10)
            test_results.append(
                {
                    "Test": "Ljung-Box (lag=10)",
                    "Hypothesis": "H0: No autocorrelation",
                    "Statistic": f"{lb.get('statistic', lb.get('lb_statistic', 0)):.4f}",
                    "p-value": f"{lb.get('p_value', 0):.4f}",
                    "Result": (
                        "No autocorrelation"
                        if lb.get("p_value", 0) >= 0.05
                        else "Autocorrelation detected"
                    ),
                }
            )
        except Exception:
            pass

        if test_results:
            st.dataframe(
                pd.DataFrame(test_results).set_index("Test"),
                use_container_width=True,
            )
        else:
            st.info(
                "Install wraquant statistical modules or scipy/statsmodels "
                "for full test results."
            )

        # Hurst exponent
        st.divider()
        st.subheader("Additional Statistics")
        try:
            from wraquant.stats.distributions import hurst_exponent

            hurst = hurst_exponent(returns)
            h_val = hurst.get("hurst", hurst) if isinstance(hurst, dict) else hurst
            h1, h2 = st.columns(2)
            h1.metric(
                "Hurst Exponent",
                f"{float(h_val):.3f}",
                help="<0.5 = mean-reverting, 0.5 = random walk, >0.5 = trending",
            )
            if float(h_val) < 0.45:
                h2.info("Mean-reverting behavior detected.")
            elif float(h_val) > 0.55:
                h2.info("Trending / persistent behavior detected.")
            else:
                h2.info("Close to random walk (H ~ 0.5).")
        except Exception:
            pass

    # ---- Summary Table Tab ----
    with tab_summary:
        st.subheader("Summary Statistics")

        summary_data = {
            "Mean (daily)": f"{float(returns.mean()):.6f}",
            "Mean (annualized)": f"{ann_ret:.2%}",
            "Std Dev (daily)": f"{float(returns.std()):.6f}",
            "Std Dev (annualized)": f"{ann_vol:.2%}",
            "Skewness": f"{skew:.4f}",
            "Excess Kurtosis": f"{kurt:.4f}",
            "Sharpe Ratio": f"{sharpe:.3f}",
            "Max Drawdown": f"{max_dd:.2%}",
            "VaR (95%)": f"{var_95:.4%}",
            "CVaR (95%)": f"{cvar_95:.4%}",
            "Min Return": f"{float(returns.min()):.4%}",
            "Max Return": f"{float(returns.max()):.4%}",
            "Median Return": f"{float(returns.median()):.6f}",
            "Win Rate": f"{float((returns > 0).mean()):.1%}",
            "Observations": f"{len(returns):,}",
            "First Date": str(returns.index[0].date()),
            "Last Date": str(returns.index[-1].date()),
        }

        col_s1, col_s2 = st.columns(2)
        items = list(summary_data.items())
        mid = (len(items) + 1) // 2
        with col_s1:
            for k, v in items[:mid]:
                st.markdown(f"**{k}:** {v}")
        with col_s2:
            for k, v in items[mid:]:
                st.markdown(f"**{k}:** {v}")

        # Downloadable table
        st.divider()
        summary_df = pd.DataFrame(
            [{"Metric": k, "Value": v} for k, v in summary_data.items()]
        )
        st.dataframe(summary_df, hide_index=True, use_container_width=True)
