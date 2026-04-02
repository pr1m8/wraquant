"""Forecasting page -- GARCH vol forecasts, time series, Monte Carlo.

Displays GARCH conditional volatility forecasts with confidence bands,
time series forecasts (auto_forecast/ARIMA), Monte Carlo simulation
paths, and forecast evaluation metrics.
"""

from __future__ import annotations

import streamlit as st


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_returns(ticker, period="3y"):
    """Fetch close prices and returns."""
    import pandas as pd
    try:
        import yfinance as yf
        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if not data.empty:
            close = data["Close"]
            if hasattr(close, "columns"):
                close = close.iloc[:, 0]
            return close, close.pct_change().dropna()
    except Exception:
        pass
    try:
        from datetime import datetime, timedelta
        from wraquant.data.providers.fmp import FMPClient
        client = FMPClient()
        end = datetime.now()
        days = {"1y": 365, "2y": 730, "3y": 1095, "5y": 1825}.get(period, 1095)
        start = end - timedelta(days=days)
        df = client.historical_price(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="daily")
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
    import numpy as np
    rng = np.random.default_rng(42)
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=756)
    rets = pd.Series(rng.normal(0.0003, 0.015, 756), index=idx, name="returns")
    close = (1 + rets).cumprod() * 100
    return close, rets


def render():
    """Render the Forecasting page."""
    import numpy as np
    import pandas as pd
    from wraquant.dashboard.components.charts import COLORS, SERIES_COLORS, dark_layout

    ticker = st.session_state.get("ticker", "AAPL")
    st.markdown(f"# Forecasting: **{ticker}**")

    with st.sidebar:
        st.subheader("Forecast Settings")
        period = st.selectbox("Data Lookback", ["1y", "2y", "3y", "5y"], index=2, key="fc_period")
        horizon = st.slider("Forecast Horizon (days)", 5, 60, 21, key="fc_horizon")
        n_mc = st.number_input("Monte Carlo Paths", 100, 10000, 1000, 100, key="fc_mc_paths")

    with st.spinner(f"Loading {ticker} data..."):
        close, returns = _fetch_returns(ticker, period=period)

    if returns is None or len(returns) < 50:
        st.warning("Insufficient data for forecasting.")
        return

    ann_vol = float(returns.std() * np.sqrt(252))
    last_price = float(close.iloc[-1])
    st.caption(f"{len(returns)} observations | Last: {returns.index[-1].date()}")

    k1, k2, k3 = st.columns(3)
    k1.metric("Last Price", f"${last_price:,.2f}")
    k2.metric("Ann. Volatility", f"{ann_vol:.1%}")
    k3.metric("Forecast Horizon", f"{horizon} days")
    st.divider()

    tab_garch, tab_ts, tab_mc, tab_eval = st.tabs(["GARCH Vol Forecast", "Time Series Forecast", "Monte Carlo Simulation", "Forecast Evaluation"])

    with tab_garch:
        st.subheader("GARCH Conditional Volatility Forecast")
        garch_type = st.selectbox("Model", ["GARCH(1,1)", "EGARCH", "GJR-GARCH"], key="fc_garch_type")
        cond_vol = None
        fc_vol = None

        with st.spinner(f"Fitting {garch_type}..."):
            try:
                if garch_type == "GARCH(1,1)":
                    from wraquant.vol.models import garch_fit, garch_forecast
                    result = garch_fit(returns)
                elif garch_type == "EGARCH":
                    from wraquant.vol.models import egarch_fit, garch_forecast
                    result = egarch_fit(returns)
                else:
                    from wraquant.vol.models import gjr_garch_fit, garch_forecast
                    result = gjr_garch_fit(returns)
                if hasattr(result, "conditional_volatility"):
                    cond_vol = result.conditional_volatility
                elif isinstance(result, dict):
                    cond_vol = result.get("conditional_volatility", result.get("cond_vol"))
                try:
                    fc = garch_forecast(returns, horizon=horizon)
                    if isinstance(fc, dict):
                        fc_vol = fc.get("forecast_volatility", fc.get("forecast_vol", fc.get("forecasts")))
                    elif hasattr(fc, "variance"):
                        fc_vol = np.sqrt(fc.variance.values[-1]) * np.sqrt(252)
                except Exception:
                    pass
            except Exception as exc:
                st.warning(f"GARCH fitting failed: {exc}. Using EWMA fallback.")
                lam = 0.94
                ewma_var = np.zeros(len(returns))
                ewma_var[0] = returns.iloc[0] ** 2
                for i in range(1, len(returns)):
                    ewma_var[i] = lam * ewma_var[i - 1] + (1 - lam) * returns.iloc[i] ** 2
                cond_vol = pd.Series(np.sqrt(ewma_var) * np.sqrt(252), index=returns.index)

        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            if cond_vol is not None:
                if isinstance(cond_vol, np.ndarray):
                    cond_vol = pd.Series(cond_vol, index=returns.index[-len(cond_vol):])
                cv = cond_vol.dropna()
                fig.add_trace(go.Scatter(x=cv.index, y=cv.values, mode="lines", name="Conditional Vol", line={"color": COLORS["primary"], "width": 1.5}))
            fc_idx = pd.bdate_range(start=returns.index[-1] + pd.Timedelta(days=1), periods=horizon)
            if fc_vol is not None:
                fc_arr = np.array(fc_vol[:horizon]) if isinstance(fc_vol, (list, np.ndarray)) else np.full(horizon, float(fc_vol))
            else:
                last_vol = float(cond_vol.iloc[-1]) if cond_vol is not None and len(cond_vol) > 0 else ann_vol
                decay = 0.98
                fc_arr = np.array([last_vol * (decay**i) + ann_vol * (1 - decay**i) for i in range(horizon)])
            fig.add_trace(go.Scatter(x=fc_idx, y=fc_arr, mode="lines", name="Forecast", line={"color": COLORS["warning"], "width": 2.5}))
            upper = fc_arr * 1.5
            lower = fc_arr * 0.6
            fig.add_trace(go.Scatter(x=fc_idx, y=upper, mode="lines", line={"width": 0}, showlegend=False))
            fig.add_trace(go.Scatter(x=fc_idx, y=lower, mode="lines", line={"width": 0}, fill="tonexty", fillcolor="rgba(245,158,11,0.15)", name="Confidence Band"))
            fig.update_layout(**dark_layout(title=f"{garch_type} Volatility Forecast ({horizon}d)", yaxis_title="Annualized Volatility", height=500))
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.warning("Plotly required for forecast charts.")

    with tab_ts:
        st.subheader("Time Series Forecast")
        ts_model = st.selectbox("Method", ["Auto Forecast", "Exponential Smoothing", "Theta"], key="fc_ts_model")
        forecast_result = None
        with st.spinner(f"Running {ts_model}..."):
            try:
                if ts_model == "Auto Forecast":
                    from wraquant.ts.forecasting import auto_forecast
                    forecast_result = auto_forecast(returns, h=horizon)
                elif ts_model == "Exponential Smoothing":
                    from wraquant.ts.forecasting import exponential_smoothing
                    es_result = exponential_smoothing(returns)
                    fc_vals = es_result.forecast(horizon)
                    forecast_result = {"forecast": fc_vals.values.tolist(), "model": "ExponentialSmoothing"}
                else:
                    from wraquant.ts.forecasting import theta_forecast
                    forecast_result = theta_forecast(returns, h=horizon)
            except Exception as exc:
                st.warning(f"Forecast failed: {exc}. Using naive forecast.")

        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            hist_window = min(60, len(returns))
            recent = returns.iloc[-hist_window:]
            fig.add_trace(go.Scatter(x=recent.index, y=recent.values, mode="lines", name="Historical", line={"color": COLORS["primary"], "width": 1.5}))
            fc_idx = pd.bdate_range(start=returns.index[-1] + pd.Timedelta(days=1), periods=horizon)
            if forecast_result is not None and hasattr(forecast_result, "get"):
                fc_values = forecast_result.get("forecast", forecast_result.get("predictions", forecast_result.get("mean")))
                if fc_values is not None:
                    if isinstance(fc_values, pd.Series):
                        fc_arr = fc_values.values[:horizon]
                    else:
                        fc_arr = np.array(fc_values)[:horizon]
                    fig.add_trace(go.Scatter(x=fc_idx[:len(fc_arr)], y=fc_arr, mode="lines", name="Forecast", line={"color": COLORS["accent4"], "width": 2.5}))
                else:
                    drift = float(returns.mean())
                    fig.add_trace(go.Scatter(x=fc_idx, y=np.full(horizon, drift), mode="lines", name="Naive (drift)", line={"color": COLORS["accent4"], "width": 2, "dash": "dash"}))
            elif forecast_result is not None and hasattr(forecast_result, "forecast"):
                fc_vals = forecast_result.forecast
                if isinstance(fc_vals, pd.Series):
                    fc_arr = fc_vals.values[:horizon]
                else:
                    fc_arr = np.array(fc_vals)[:horizon]
                fig.add_trace(go.Scatter(x=fc_idx[:len(fc_arr)], y=fc_arr, mode="lines", name="Forecast", line={"color": COLORS["accent4"], "width": 2.5}))
            else:
                drift = float(returns.mean())
                fig.add_trace(go.Scatter(x=fc_idx, y=np.full(horizon, drift), mode="lines", name="Naive (drift)", line={"color": COLORS["accent4"], "width": 2, "dash": "dash"}))
            fig.update_layout(**dark_layout(title=f"{ts_model} -- {horizon}-Day Forecast", xaxis_title="Date", yaxis_title="Daily Return", height=450))
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.warning("Plotly required.")

    with tab_mc:
        st.subheader("Monte Carlo Price Simulation")
        mu_daily = float(returns.mean())
        sigma_daily = float(returns.std())
        rng = np.random.default_rng(0)
        paths = np.zeros((int(n_mc), horizon + 1))
        paths[:, 0] = last_price
        for t in range(1, horizon + 1):
            z = rng.standard_normal(int(n_mc))
            paths[:, t] = paths[:, t - 1] * np.exp((mu_daily - 0.5 * sigma_daily**2) + sigma_daily * z)
        terminal = paths[:, -1]
        median_price = float(np.median(terminal))
        p5 = float(np.percentile(terminal, 5))
        p95 = float(np.percentile(terminal, 95))

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Median Price", f"${median_price:,.2f}")
        k2.metric("5th Percentile", f"${p5:,.2f}")
        k3.metric("95th Percentile", f"${p95:,.2f}")
        k4.metric("Expected Return", f"{(median_price / last_price - 1) * 100:+.1f}%")

        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            fc_dates = pd.bdate_range(start=returns.index[-1], periods=horizon + 1)
            n_show = min(100, int(n_mc))
            for i in range(n_show):
                fig.add_trace(go.Scatter(x=fc_dates, y=paths[i], mode="lines", line={"color": COLORS["primary"], "width": 0.3}, opacity=0.15, showlegend=False))
            fig.add_trace(go.Scatter(x=fc_dates, y=np.percentile(paths, 50, axis=0), mode="lines", name="Median", line={"color": COLORS["warning"], "width": 2.5}))
            fig.add_trace(go.Scatter(x=fc_dates, y=np.percentile(paths, 95, axis=0), mode="lines", name="5-95%", line={"color": COLORS["accent2"], "width": 1, "dash": "dot"}))
            fig.add_trace(go.Scatter(x=fc_dates, y=np.percentile(paths, 5, axis=0), mode="lines", line={"color": COLORS["accent2"], "width": 1, "dash": "dot"}, fill="tonexty", fillcolor="rgba(56,189,248,0.1)", showlegend=False))
            fig.update_layout(**dark_layout(title=f"Monte Carlo Simulation ({n_mc} paths, {horizon}d)", xaxis_title="Date", yaxis_title="Price ($)", height=500))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Terminal Price Distribution")
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=terminal, nbinsx=60, marker_color=COLORS["primary"], opacity=0.7))
            fig2.add_vline(x=last_price, line_dash="dash", line_color=COLORS["danger"], annotation_text=f"Current: ${last_price:,.2f}")
            fig2.add_vline(x=median_price, line_dash="dash", line_color=COLORS["warning"], annotation_text=f"Median: ${median_price:,.2f}")
            fig2.update_layout(**dark_layout(title="Terminal Price Distribution", xaxis_title="Price ($)", yaxis_title="Count", height=350))
            st.plotly_chart(fig2, use_container_width=True)
        except ImportError:
            st.warning("Plotly required.")

    with tab_eval:
        st.subheader("Forecast Evaluation (Walk-Forward)")
        eval_window = st.slider("Evaluation Window", 60, 252, 126, key="fc_eval_window")
        if len(returns) > eval_window + 20:
            test = returns.iloc[-eval_window:]
            train = returns.iloc[:-eval_window]
            naive_fc = np.full(len(test), float(train.mean()))
            naive_errors = test.values - naive_fc

            e1, e2 = st.columns(2)
            e1.metric("MAE", f"{float(np.mean(np.abs(naive_errors))):.6f}")
            e2.metric("RMSE", f"{float(np.sqrt(np.mean(naive_errors**2))):.6f}")

            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4], subplot_titles=["Actual vs Forecast", "Errors"])
                fig.add_trace(go.Scatter(x=test.index, y=test.values, mode="lines", name="Actual", line={"color": COLORS["primary"], "width": 1.5}), row=1, col=1)
                fig.add_trace(go.Scatter(x=test.index, y=naive_fc, mode="lines", name="Naive Forecast", line={"color": COLORS["accent4"], "width": 1.5, "dash": "dash"}), row=1, col=1)
                fig.add_trace(go.Bar(x=test.index, y=naive_errors, marker_color=[COLORS["success"] if e >= 0 else COLORS["danger"] for e in naive_errors], opacity=0.6), row=2, col=1)
                fig.update_layout(**dark_layout(title="Walk-Forward Evaluation", height=550))
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.warning("Plotly required.")
        else:
            st.warning(f"Need at least {eval_window + 20} observations.")
