"""Time Series Lab -- decomposition, stationarity, seasonality, forecasting, anomaly, changepoint.

Deep time-series analysis page using wraquant.ts for all computations.
"""

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_series(ticker: str, days: int = 1095):
    """Fetch close prices and returns. Returns (close, returns)."""
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

    try:
        import yfinance as yf

        data = yf.download(ticker, period="3y", auto_adjust=True, progress=False)
        if not data.empty:
            close = data["Close"]
            if hasattr(close, "columns"):
                close = close.iloc[:, 0]
            return close, close.pct_change().dropna()
    except Exception:
        pass

    import numpy as np

    rng = np.random.default_rng(42)
    n = min(756, days)
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    rets = pd.Series(rng.normal(0.0004, 0.015, n), index=idx, name="returns")
    close = (1 + rets).cumprod() * 100
    return close, rets


def render() -> None:
    """Render the Time Series Lab page."""
    import numpy as np
    import pandas as pd

    from wraquant.dashboard.components.charts import COLORS, SERIES_COLORS, dark_layout

    ticker = st.session_state.get("ticker", "AAPL")
    st.markdown(f"# Time Series Lab: **{ticker}**")

    with st.spinner(f"Loading {ticker} data..."):
        close, returns = _fetch_series(ticker)

    if returns is None or len(returns) < 60:
        st.warning("Insufficient data for time series analysis.")
        return

    st.caption(f"{len(close)} price observations, {len(returns)} returns")

    tab_decomp, tab_station, tab_season, tab_forecast, tab_anomaly, tab_change = st.tabs([
        "Decomposition",
        "Stationarity",
        "Seasonality",
        "Forecasting",
        "Anomaly Detection",
        "Changepoints",
    ])

    # ---- Decomposition Tab ----
    with tab_decomp:
        st.subheader("STL Decomposition")
        period = st.select_slider(
            "Seasonal Period", options=[5, 21, 63, 126, 252], value=21,
            key="ts_decomp_period",
        )
        st.caption(
            "5 = weekly, 21 = monthly, 63 = quarterly, 126 = semi-annual, 252 = annual"
        )

        try:
            from wraquant.ts.decomposition import stl_decompose

            decomp = stl_decompose(close, period=period)

            trend = decomp.get("trend", None) if isinstance(decomp, dict) else getattr(decomp, "trend", None)
            seasonal = decomp.get("seasonal", None) if isinstance(decomp, dict) else getattr(decomp, "seasonal", None)
            residual = decomp.get("residual", decomp.get("resid", None)) if isinstance(decomp, dict) else getattr(decomp, "resid", None)

            if trend is not None:
                try:
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots

                    fig = make_subplots(
                        rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04,
                        subplot_titles=["Observed", "Trend", "Seasonal", "Residual"],
                        row_heights=[0.25] * 4,
                    )

                    components = [
                        (close, COLORS["primary"], "Observed"),
                        (trend, COLORS["accent2"], "Trend"),
                        (seasonal, COLORS["accent4"], "Seasonal"),
                        (residual, COLORS["accent1"], "Residual"),
                    ]

                    for i, (s, color, name) in enumerate(components, 1):
                        if s is not None:
                            s_series = pd.Series(s) if not isinstance(s, pd.Series) else s
                            fig.add_trace(
                                go.Scatter(
                                    x=s_series.index if hasattr(s_series, "index") else list(range(len(s_series))),
                                    y=s_series.values,
                                    mode="lines",
                                    line={"color": color, "width": 1.5},
                                    name=name,
                                    showlegend=False,
                                ),
                                row=i, col=1,
                            )

                    fig.update_layout(**dark_layout(title="STL Decomposition", height=800))
                    for r in range(1, 5):
                        fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", row=r, col=1)
                        fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", row=r, col=1)
                    st.plotly_chart(fig, use_container_width=True)

                except ImportError:
                    st.line_chart(pd.DataFrame({"Trend": trend, "Seasonal": seasonal}))
            else:
                st.warning("Decomposition did not return expected components.")

        except Exception as e:
            st.warning(f"Could not run STL decomposition: {e}")

            # Fallback using statsmodels
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose as sm_decompose

                decomp = sm_decompose(close.dropna(), model="additive", period=period)
                st.line_chart(
                    pd.DataFrame({
                        "Trend": decomp.trend,
                        "Seasonal": decomp.seasonal,
                        "Residual": decomp.resid,
                    }).dropna()
                )
            except Exception:
                st.info("Install wraquant or statsmodels for decomposition.")

    # ---- Stationarity Tab ----
    with tab_station:
        st.subheader("Stationarity Tests")

        test_results = []

        # ADF
        try:
            from wraquant.ts.stationarity import adf_test

            adf = adf_test(returns)
            test_results.append({
                "Test": "Augmented Dickey-Fuller",
                "Hypothesis": "H0: Unit root (non-stationary)",
                "Statistic": f"{adf.get('test_statistic', adf.get('adf_statistic', 0)):.4f}",
                "p-value": f"{adf.get('p_value', 0):.4f}",
                "Result": "Stationary" if adf.get("p_value", 1) < 0.05 else "Non-stationary",
            })
        except Exception:
            try:
                from statsmodels.tsa.stattools import adfuller

                r = adfuller(returns.dropna())
                test_results.append({
                    "Test": "Augmented Dickey-Fuller",
                    "Hypothesis": "H0: Unit root",
                    "Statistic": f"{r[0]:.4f}",
                    "p-value": f"{r[1]:.4f}",
                    "Result": "Stationary" if r[1] < 0.05 else "Non-stationary",
                })
            except Exception:
                pass

        # KPSS
        try:
            from wraquant.ts.stationarity import kpss_test

            kpss = kpss_test(returns)
            test_results.append({
                "Test": "KPSS",
                "Hypothesis": "H0: Stationary",
                "Statistic": f"{kpss.get('test_statistic', kpss.get('kpss_statistic', 0)):.4f}",
                "p-value": f"{kpss.get('p_value', 0):.4f}",
                "Result": "Stationary" if kpss.get("p_value", 0) >= 0.05 else "Non-stationary",
            })
        except Exception:
            pass

        # Phillips-Perron
        try:
            from wraquant.ts.stationarity import phillips_perron

            pp = phillips_perron(returns)
            test_results.append({
                "Test": "Phillips-Perron",
                "Hypothesis": "H0: Unit root",
                "Statistic": f"{pp.get('test_statistic', pp.get('pp_statistic', 0)):.4f}",
                "p-value": f"{pp.get('p_value', 0):.4f}",
                "Result": "Stationary" if pp.get("p_value", 1) < 0.05 else "Non-stationary",
            })
        except Exception:
            pass

        if test_results:
            st.dataframe(
                pd.DataFrame(test_results).set_index("Test"),
                use_container_width=True,
            )
        else:
            st.info("No stationarity tests available. Install wraquant or statsmodels.")

        # Optimal differencing
        st.divider()
        st.subheader("Optimal Differencing")
        try:
            from wraquant.ts.stationarity import optimal_differencing

            opt = optimal_differencing(close)
            if isinstance(opt, dict):
                od1, od2 = st.columns(2)
                od1.metric("Recommended d", f"{opt.get('d', opt.get('order', 'N/A'))}")
                od2.metric("ADF p-value after diff", f"{opt.get('p_value', 'N/A')}")
            else:
                st.metric("Recommended d", f"{opt}")
        except Exception as e:
            st.info(f"Optimal differencing unavailable: {e}")

        # Variance ratio
        st.divider()
        st.subheader("Variance Ratio Test")
        try:
            from wraquant.ts.stationarity import variance_ratio_test

            vr = variance_ratio_test(returns)
            if isinstance(vr, dict):
                vr1, vr2, vr3 = st.columns(3)
                vr1.metric("Variance Ratio", f"{vr.get('variance_ratio', 'N/A')}")
                vr2.metric("z-statistic", f"{vr.get('z_statistic', vr.get('statistic', 'N/A'))}")
                vr3.metric(
                    "Random Walk?",
                    "Yes" if vr.get("p_value", 1) >= 0.05 else "No",
                )
        except Exception:
            pass

    # ---- Seasonality Tab ----
    with tab_season:
        st.subheader("Calendar Effects")

        # Day-of-week
        if hasattr(returns.index, "dayofweek"):
            dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
            dow_means = returns.groupby(returns.index.dayofweek).mean()
            dow_means.index = [dow_names[i] for i in dow_means.index if i < 5]

            try:
                import plotly.graph_objects as go

                fig = go.Figure()
                colors = [COLORS["success"] if v > 0 else COLORS["danger"] for v in dow_means.values]
                fig.add_trace(
                    go.Bar(
                        x=dow_means.index, y=dow_means.values * 252,
                        marker_color=colors,
                        text=[f"{v*252:.2%}" for v in dow_means.values],
                        textposition="auto",
                    )
                )
                fig.update_layout(
                    **dark_layout(
                        title="Day-of-Week Effect (Annualized Mean Return)",
                        yaxis_title="Ann. Return", height=350,
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(dow_means)

        # Month-of-year
        if hasattr(returns.index, "month"):
            month_names = [
                "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
            ]
            month_means = returns.groupby(returns.index.month).mean()
            month_means.index = [month_names[i - 1] for i in month_means.index]

            try:
                import plotly.graph_objects as go

                fig = go.Figure()
                colors = [COLORS["success"] if v > 0 else COLORS["danger"] for v in month_means.values]
                fig.add_trace(
                    go.Bar(
                        x=month_means.index, y=month_means.values * 252,
                        marker_color=colors,
                        text=[f"{v*252:.2%}" for v in month_means.values],
                        textposition="auto",
                    )
                )
                fig.update_layout(
                    **dark_layout(
                        title="Month-of-Year Effect (Annualized Mean Return)",
                        yaxis_title="Ann. Return", height=350,
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(month_means)

        # Seasonal strength
        st.divider()
        st.subheader("Seasonal Strength")
        try:
            from wraquant.ts.seasonality import seasonal_strength

            ss = seasonal_strength(close)
            if isinstance(ss, dict):
                ss1, ss2 = st.columns(2)
                ss1.metric("Seasonal Strength", f"{ss.get('strength', ss.get('seasonal_strength', 'N/A'))}")
                ss2.metric("Trend Strength", f"{ss.get('trend_strength', 'N/A')}")
            else:
                st.metric("Seasonal Strength", f"{ss}")
        except Exception as e:
            st.info(f"Seasonal strength unavailable: {e}")

        try:
            from wraquant.ts.seasonality import detect_seasonality

            detected = detect_seasonality(returns)
            if isinstance(detected, dict):
                st.metric(
                    "Detected Period",
                    f"{detected.get('period', detected.get('dominant_period', 'N/A'))} days",
                )
            elif detected is not None:
                st.metric("Detected Period", f"{detected} days")
        except Exception:
            pass

    # ---- Forecasting Tab ----
    with tab_forecast:
        st.subheader("Forecasting")
        horizon = st.slider("Forecast Horizon (days)", 5, 60, 20, key="ts_horizon")

        try:
            from wraquant.ts.forecasting import auto_forecast

            with st.spinner("Running auto_forecast..."):
                fc = auto_forecast(returns, h=horizon)

            if isinstance(fc, dict):
                forecast_vals = fc.get("forecast", fc.get("predictions", None))
                lower = fc.get("lower", fc.get("conf_lower", None))
                upper = fc.get("upper", fc.get("conf_upper", None))
                model_name = fc.get("model", fc.get("best_model", "auto"))

                st.info(f"Best model: **{model_name}**")

                # Metrics
                metrics = fc.get("metrics", {})
                if metrics:
                    mc = st.columns(min(len(metrics), 4))
                    for i, (k, v) in enumerate(list(metrics.items())[:4]):
                        mc[i].metric(k.upper(), f"{v:.4f}" if isinstance(v, float) else str(v))

                if forecast_vals is not None:
                    try:
                        import plotly.graph_objects as go

                        last_date = returns.index[-1]
                        fc_idx = pd.bdate_range(start=last_date, periods=horizon + 1)[1:]
                        fc_series = pd.Series(
                            forecast_vals[:horizon] if hasattr(forecast_vals, "__len__") else [forecast_vals] * horizon,
                            index=fc_idx,
                        )

                        fig = go.Figure()
                        # Historical
                        hist_tail = returns.iloc[-120:]
                        fig.add_trace(
                            go.Scatter(
                                x=hist_tail.index, y=hist_tail.values,
                                mode="lines",
                                line={"color": COLORS["primary"], "width": 1.5},
                                name="Historical",
                            )
                        )
                        # Forecast
                        fig.add_trace(
                            go.Scatter(
                                x=fc_series.index, y=fc_series.values,
                                mode="lines",
                                line={"color": COLORS["accent2"], "width": 2, "dash": "dash"},
                                name="Forecast",
                            )
                        )
                        # Confidence bands
                        if lower is not None and upper is not None:
                            lower_s = pd.Series(
                                lower[:horizon] if hasattr(lower, "__len__") else [lower] * horizon,
                                index=fc_idx,
                            )
                            upper_s = pd.Series(
                                upper[:horizon] if hasattr(upper, "__len__") else [upper] * horizon,
                                index=fc_idx,
                            )
                            fig.add_trace(
                                go.Scatter(
                                    x=fc_idx, y=upper_s.values,
                                    mode="lines", line={"width": 0},
                                    showlegend=False,
                                )
                            )
                            fig.add_trace(
                                go.Scatter(
                                    x=fc_idx, y=lower_s.values,
                                    mode="lines", line={"width": 0},
                                    fill="tonexty",
                                    fillcolor="rgba(99,102,241,0.2)",
                                    name="95% CI",
                                )
                            )

                        fig.update_layout(
                            **dark_layout(
                                title=f"Return Forecast ({horizon}d ahead)",
                                yaxis_title="Return", height=450,
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    except ImportError:
                        st.line_chart(fc_series)

                # Model comparison table
                comparison = fc.get("comparison", fc.get("model_comparison", None))
                if comparison is not None:
                    st.subheader("Model Comparison")
                    if isinstance(comparison, pd.DataFrame):
                        st.dataframe(comparison, use_container_width=True)
                    elif isinstance(comparison, dict):
                        st.dataframe(pd.DataFrame(comparison), use_container_width=True)
                    elif isinstance(comparison, list):
                        st.dataframe(pd.DataFrame(comparison), use_container_width=True)

            else:
                st.warning("Forecast returned unexpected format.")

        except Exception as e:
            st.warning(f"Could not run auto_forecast: {e}")
            st.info("Showing simple exponential smoothing fallback.")
            try:
                alpha = 0.3
                fc_val = float(returns.iloc[-1])
                forecast_vals = []
                for _ in range(horizon):
                    fc_val = alpha * float(returns.iloc[-1]) + (1 - alpha) * fc_val
                    forecast_vals.append(fc_val)
                last_date = returns.index[-1]
                fc_idx = pd.bdate_range(start=last_date, periods=horizon + 1)[1:]
                st.line_chart(pd.Series(forecast_vals, index=fc_idx, name="Forecast"))
            except Exception:
                pass

    # ---- Anomaly Detection Tab ----
    with tab_anomaly:
        st.subheader("Anomaly Detection")

        method = st.selectbox(
            "Detection Method",
            ["Z-Score", "IQR", "Isolation Forest"],
            key="ts_anomaly_method",
        )
        threshold = st.slider(
            "Z-Score / IQR Threshold", 1.5, 4.0, 3.0, 0.1, key="ts_anomaly_thresh"
        )

        if method == "Z-Score":
            z = (returns - returns.mean()) / returns.std()
            anomalies = returns[z.abs() > threshold]
        elif method == "IQR":
            q1, q3 = returns.quantile(0.25), returns.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            anomalies = returns[(returns < lower) | (returns > upper)]
        else:
            # Isolation Forest
            try:
                from wraquant.ts.anomaly import isolation_forest_ts

                iso = isolation_forest_ts(returns)
                if isinstance(iso, dict):
                    labels = iso.get("labels", iso.get("anomalies", None))
                    if labels is not None:
                        if isinstance(labels, pd.Series):
                            anomalies = returns[labels == -1]
                        else:
                            mask = np.array(labels) == -1
                            anomalies = returns.iloc[mask[:len(returns)]]
                    else:
                        anomalies = pd.Series(dtype=float)
                else:
                    anomalies = pd.Series(dtype=float)
            except Exception:
                try:
                    from sklearn.ensemble import IsolationForest

                    iso = IsolationForest(contamination=0.05, random_state=42)
                    preds = iso.fit_predict(returns.values.reshape(-1, 1))
                    anomalies = returns[preds == -1]
                except Exception:
                    z = (returns - returns.mean()) / returns.std()
                    anomalies = returns[z.abs() > 3.0]

        st.metric("Anomalies Detected", f"{len(anomalies)} ({len(anomalies)/len(returns)*100:.1f}%)")

        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=returns.index, y=returns.values, mode="lines",
                    line={"color": COLORS["primary"], "width": 1},
                    name="Returns",
                )
            )
            if len(anomalies) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=anomalies.index, y=anomalies.values,
                        mode="markers",
                        marker={"color": COLORS["danger"], "size": 8, "symbol": "x"},
                        name=f"Anomalies ({method})",
                    )
                )
            fig.update_layout(
                **dark_layout(
                    title=f"Anomaly Detection ({method})",
                    yaxis_title="Return", height=450,
                )
            )
            st.plotly_chart(fig, use_container_width=True)

            # Anomaly table
            if len(anomalies) > 0:
                st.subheader("Anomaly Events")
                anom_df = pd.DataFrame({
                    "Date": anomalies.index.strftime("%Y-%m-%d"),
                    "Return": [f"{v:.4%}" for v in anomalies.values],
                    "Z-Score": [
                        f"{(v - returns.mean()) / returns.std():.2f}"
                        for v in anomalies.values
                    ],
                })
                st.dataframe(anom_df.head(20), hide_index=True, use_container_width=True)

        except ImportError:
            st.line_chart(returns)
            if len(anomalies) > 0:
                st.write(f"Anomaly dates: {list(anomalies.index[:10])}")

    # ---- Changepoint Tab ----
    with tab_change:
        st.subheader("Changepoint Detection")

        try:
            from wraquant.ts.changepoint import detect_changepoints

            with st.spinner("Detecting changepoints..."):
                cp = detect_changepoints(returns)

            if isinstance(cp, dict):
                cps = cp.get("changepoints", cp.get("breakpoints", []))
            elif isinstance(cp, (list, np.ndarray)):
                cps = list(cp)
            else:
                cps = []

            st.metric("Changepoints Found", len(cps))

            try:
                import plotly.graph_objects as go

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=close.index, y=close.values, mode="lines",
                        line={"color": COLORS["primary"], "width": 1.5},
                        name="Price",
                    )
                )

                for cp_idx in cps:
                    if isinstance(cp_idx, (int, np.integer)):
                        if cp_idx < len(close):
                            cp_date = close.index[cp_idx]
                        else:
                            continue
                    else:
                        cp_date = cp_idx
                    fig.add_vline(
                        x=cp_date, line_dash="dash",
                        line_color=COLORS["danger"], opacity=0.7,
                    )

                fig.update_layout(
                    **dark_layout(
                        title="Price with Detected Changepoints",
                        yaxis_title="Price", height=450,
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

                # Pre/post statistics for each changepoint
                if cps:
                    st.subheader("Pre/Post Change Statistics")
                    cp_stats = []
                    for i, cp_idx in enumerate(cps[:10]):
                        if isinstance(cp_idx, (int, np.integer)):
                            idx_val = int(cp_idx)
                        else:
                            try:
                                idx_val = returns.index.get_loc(cp_idx)
                            except Exception:
                                continue

                        if idx_val < 20 or idx_val >= len(returns) - 5:
                            continue

                        pre = returns.iloc[max(0, idx_val - 60):idx_val]
                        post = returns.iloc[idx_val:min(len(returns), idx_val + 60)]
                        cp_date_str = str(returns.index[idx_val].date()) if idx_val < len(returns) else "?"

                        cp_stats.append({
                            "Changepoint": cp_date_str,
                            "Pre Mean (ann)": f"{float(pre.mean() * 252):.2%}",
                            "Post Mean (ann)": f"{float(post.mean() * 252):.2%}",
                            "Pre Vol (ann)": f"{float(pre.std() * np.sqrt(252)):.2%}",
                            "Post Vol (ann)": f"{float(post.std() * np.sqrt(252)):.2%}",
                        })

                    if cp_stats:
                        st.dataframe(
                            pd.DataFrame(cp_stats),
                            hide_index=True, use_container_width=True,
                        )

            except ImportError:
                st.line_chart(close)
                st.write(f"Changepoints at indices: {cps[:10]}")

        except Exception as e:
            st.warning(f"Could not detect changepoints: {e}")

            # CUSUM fallback
            try:
                from wraquant.ts.changepoint import cusum

                cs = cusum(returns)
                if isinstance(cs, dict):
                    st.write(cs)
                else:
                    st.write(f"CUSUM result: {cs}")
            except Exception:
                st.info("Changepoint detection requires wraquant.ts.changepoint.")
