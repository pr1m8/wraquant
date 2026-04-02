"""Volatility Modeling page -- realized vol, GARCH, and forecasting.

Displays realized volatility estimators (close-to-close, Parkinson,
Yang-Zhang), GARCH model fitting with diagnostics, conditional vol
forecasts, news impact curves, and EWMA vs GARCH comparisons.
Uses ``wraquant.vol`` for computations.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_ohlcv(ticker: str, period: str = "3y") -> "pd.DataFrame":
    """Fetch OHLCV data for a single ticker."""
    import yfinance as yf

    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No data for {ticker}")
    # Flatten multi-level columns if present
    if hasattr(data.columns, "levels"):
        data.columns = (
            data.columns.droplevel(1) if len(data.columns.levels) > 1 else data.columns
        )
    return data


def _synthetic_ohlcv(n: int = 756) -> "pd.DataFrame":
    """Generate synthetic OHLCV for demo."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    high = close * (1 + rng.uniform(0.001, 0.025, n))
    low = close * (1 - rng.uniform(0.001, 0.025, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(1_000_000, 50_000_000, n)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=idx,
    )


def render() -> None:
    """Render the Volatility Modeling page."""
    import numpy as np
    import pandas as pd

    st.header("Volatility Modeling")

    ticker = st.session_state.get("ticker", "AAPL")

    with st.sidebar:
        st.subheader("Vol Settings")
        period = st.selectbox(
            "Lookback", ["1y", "2y", "3y", "5y"], index=2, key="vol_period"
        )
        vol_window = st.slider("Rolling Window", 10, 63, 21, key="vol_window")
        garch_type = st.selectbox(
            "GARCH Variant", ["GARCH(1,1)", "EGARCH", "GJR-GARCH"], key="garch_type"
        )

    # -- Fetch data --------------------------------------------------------

    with st.spinner(f"Loading {ticker} OHLCV data..."):
        try:
            ohlcv = _fetch_ohlcv(ticker, period=period)
        except Exception:
            st.info(f"Live data unavailable for {ticker} -- using synthetic data.")
            ohlcv = _synthetic_ohlcv()

    close = ohlcv["Close"]
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    returns = close.pct_change().dropna()
    high = ohlcv["High"]
    low = ohlcv["Low"]
    open_ = ohlcv["Open"]
    if hasattr(high, "columns"):
        high = high.iloc[:, 0]
    if hasattr(low, "columns"):
        low = low.iloc[:, 0]
    if hasattr(open_, "columns"):
        open_ = open_.iloc[:, 0]

    ann_vol = float(returns.std() * np.sqrt(252))
    current_vol_20d = float(returns.tail(20).std() * np.sqrt(252))

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Full-Sample Vol", f"{ann_vol:.1%}")
    k2.metric(
        "20-Day Vol",
        f"{current_vol_20d:.1%}",
        delta=f"{(current_vol_20d - ann_vol) / ann_vol:.0%} vs avg",
        delta_color="inverse" if current_vol_20d > ann_vol else "normal",
    )
    k3.metric("Observations", f"{len(returns):,}")
    k4.metric("Latest Close", f"${float(close.iloc[-1]):,.2f}")

    st.divider()

    # -- Tabs --------------------------------------------------------------

    tab_realized, tab_garch, tab_nic, tab_ewma = st.tabs(
        ["Realized Volatility", "GARCH Modeling", "News Impact Curve", "EWMA vs GARCH"],
    )

    # ---- Realized Volatility ----
    with tab_realized:
        st.subheader("Realized Volatility Estimators")

        # Compute estimators
        try:
            from wraquant.vol.realized import parkinson, realized_volatility, yang_zhang

            rv_cc = realized_volatility(returns, window=vol_window)
            rv_park = parkinson(high, low, window=vol_window)
            rv_yz = yang_zhang(open_, high, low, close, window=vol_window)
        except Exception:
            # Fallback
            rv_cc = returns.rolling(vol_window).std() * np.sqrt(252)
            log_hl = np.log(high / low)
            rv_park = np.sqrt(
                (1 / (4 * np.log(2))) * (log_hl**2).rolling(vol_window).mean()
            ) * np.sqrt(252)
            rv_yz = rv_cc  # Simplified fallback

        estimator_choice = st.multiselect(
            "Show Estimators",
            ["Close-to-Close", "Parkinson", "Yang-Zhang"],
            default=["Close-to-Close", "Parkinson", "Yang-Zhang"],
            key="vol_estimators",
        )

        try:
            import plotly.graph_objects as go

            from wraquant.dashboard.components.charts import (
                COLORS,
                SERIES_COLORS,
                dark_layout,
            )

            fig = go.Figure()
            estimator_map = {
                "Close-to-Close": (rv_cc, SERIES_COLORS[0]),
                "Parkinson": (rv_park, SERIES_COLORS[1]),
                "Yang-Zhang": (rv_yz, SERIES_COLORS[2]),
            }
            for name in estimator_choice:
                series, color = estimator_map[name]
                s = series.dropna()
                fig.add_trace(
                    go.Scatter(
                        x=s.index,
                        y=s.values,
                        mode="lines",
                        name=name,
                        line={"color": color, "width": 1.5},
                    )
                )

            fig.update_layout(
                **dark_layout(
                    title=f"Rolling {vol_window}-Day Realized Volatility (Annualized)",
                    yaxis_title="Volatility",
                    height=450,
                )
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            vol_df = pd.DataFrame(
                {
                    "Close-to-Close": rv_cc,
                    "Parkinson": rv_park,
                    "Yang-Zhang": rv_yz,
                }
            ).dropna()
            st.line_chart(vol_df[estimator_choice])

        # Latest values
        st.markdown("### Current Estimates")
        cols = st.columns(3)
        for i, (name, series) in enumerate(
            [
                ("Close-to-Close", rv_cc),
                ("Parkinson", rv_park),
                ("Yang-Zhang", rv_yz),
            ]
        ):
            val = float(series.dropna().iloc[-1]) if not series.dropna().empty else 0.0
            cols[i].metric(name, f"{val:.1%}")

    # ---- GARCH Modeling ----
    with tab_garch:
        st.subheader(f"{garch_type} Model")

        garch_result = None
        cond_vol = None

        with st.spinner("Fitting GARCH model..."):
            try:
                if garch_type == "GARCH(1,1)":
                    from wraquant.vol.models import garch_fit

                    garch_result = garch_fit(returns)
                elif garch_type == "EGARCH":
                    from wraquant.vol.models import egarch_fit

                    garch_result = egarch_fit(returns)
                else:
                    from wraquant.vol.models import gjr_garch_fit

                    garch_result = gjr_garch_fit(returns)

                # Extract info from result
                if hasattr(garch_result, "conditional_volatility"):
                    cond_vol = garch_result.conditional_volatility
                elif isinstance(garch_result, dict):
                    cond_vol = garch_result.get("conditional_volatility")
                    if cond_vol is None:
                        cond_vol = garch_result.get("cond_vol")
            except Exception as exc:
                st.warning(f"GARCH fitting failed: {exc}. Using EWMA fallback.")
                # EWMA fallback
                lam = 0.94
                ewma_var = np.zeros(len(returns))
                ewma_var[0] = returns.iloc[0] ** 2
                for i in range(1, len(returns)):
                    ewma_var[i] = (
                        lam * ewma_var[i - 1] + (1 - lam) * returns.iloc[i] ** 2
                    )
                cond_vol = pd.Series(
                    np.sqrt(ewma_var) * np.sqrt(252), index=returns.index
                )

        # Display parameters
        if garch_result is not None:
            st.markdown("### Model Parameters")
            params = {}
            if hasattr(garch_result, "params"):
                p = garch_result.params
                params = dict(p) if hasattr(p, "items") else {"params": str(p)}
            elif isinstance(garch_result, dict):
                params = {
                    k: v
                    for k, v in garch_result.items()
                    if k
                    in (
                        "omega",
                        "alpha",
                        "beta",
                        "gamma",
                        "persistence",
                        "half_life",
                        "log_likelihood",
                        "aic",
                        "bic",
                    )
                }

            if params:
                pc1, pc2 = st.columns(2)
                items = list(params.items())
                mid = (len(items) + 1) // 2
                with pc1:
                    for k, v in items[:mid]:
                        if isinstance(v, float):
                            st.metric(k, f"{v:.6f}")
                        else:
                            st.metric(k, str(v))
                with pc2:
                    for k, v in items[mid:]:
                        if isinstance(v, float):
                            st.metric(k, f"{v:.6f}")
                        else:
                            st.metric(k, str(v))

        # Conditional volatility plot
        if cond_vol is not None:
            st.markdown("### Conditional Volatility")
            try:
                import plotly.graph_objects as go

                from wraquant.dashboard.components.charts import COLORS, dark_layout

                if isinstance(cond_vol, np.ndarray):
                    cond_vol = pd.Series(
                        cond_vol, index=returns.index[-len(cond_vol) :]
                    )

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=cond_vol.index,
                        y=cond_vol.values,
                        mode="lines",
                        name="Conditional Vol",
                        line={"color": COLORS["primary"], "width": 1.5},
                    )
                )

                # Add confidence bands (approximate)
                upper = cond_vol * 1.5
                lower = cond_vol * 0.5
                fig.add_trace(
                    go.Scatter(
                        x=cond_vol.index,
                        y=upper.values,
                        mode="lines",
                        line={"width": 0},
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=cond_vol.index,
                        y=lower.values,
                        mode="lines",
                        line={"width": 0},
                        fill="tonexty",
                        fillcolor="rgba(99, 102, 241, 0.15)",
                        name="Confidence Band",
                    )
                )

                fig.update_layout(
                    **dark_layout(
                        title=f"{garch_type} Conditional Volatility",
                        yaxis_title="Volatility",
                        height=450,
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                if isinstance(cond_vol, pd.Series):
                    st.line_chart(cond_vol)

        # Forecast
        st.markdown("### Volatility Forecast")
        forecast_horizon = st.slider(
            "Forecast Horizon (days)", 5, 60, 21, key="vol_forecast_h"
        )
        try:
            from wraquant.vol.models import garch_forecast

            fc = garch_forecast(returns, horizon=forecast_horizon)
            if isinstance(fc, dict):
                fc_vol = fc.get("forecast_volatility", fc.get("forecast_vol", fc.get("forecasts")))
            elif hasattr(fc, "variance"):
                fc_vol = np.sqrt(fc.variance.values[-1]) * np.sqrt(252)
            else:
                fc_vol = None

            if fc_vol is not None:
                if isinstance(fc_vol, (list, np.ndarray)):
                    fc_series = pd.Series(
                        fc_vol[:forecast_horizon],
                        index=pd.bdate_range(
                            start=returns.index[-1] + pd.Timedelta(days=1),
                            periods=min(len(fc_vol), forecast_horizon),
                        ),
                    )
                    st.line_chart(fc_series)
                else:
                    st.metric("Forecast Vol (next day)", f"{float(fc_vol):.1%}")
        except Exception:
            # Simple persistence forecast
            last_vol = (
                float(cond_vol.iloc[-1])
                if cond_vol is not None and len(cond_vol) > 0
                else current_vol_20d
            )
            decay = 0.98
            fc_vals = [
                last_vol * (decay**i) + ann_vol * (1 - decay**i)
                for i in range(forecast_horizon)
            ]
            fc_idx = pd.bdate_range(
                start=returns.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon
            )
            st.caption("Persistence-based forecast (GARCH unavailable)")
            st.line_chart(pd.Series(fc_vals, index=fc_idx, name="Forecast Vol"))

    # ---- News Impact Curve ----
    with tab_nic:
        st.subheader("News Impact Curve")
        st.markdown(
            "The news impact curve shows how volatility responds to return shocks of different "
            "magnitudes. Asymmetric models (EGARCH, GJR) show a steeper response to negative shocks "
            "(the leverage effect)."
        )

        try:
            from wraquant.vol.models import news_impact_curve

            nic = news_impact_curve(returns)

            if isinstance(nic, dict):
                shocks = nic.get("shocks", nic.get("x"))
                impact = nic.get(
                    "impact", nic.get("y", nic.get("conditional_variance"))
                )
            elif hasattr(nic, "shocks"):
                shocks = nic.shocks
                impact = nic.impact
            else:
                raise ValueError("Unexpected NIC result format")

            if shocks is not None and impact is not None:
                import plotly.graph_objects as go

                from wraquant.dashboard.components.charts import COLORS, dark_layout

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=(
                            np.array(shocks)
                            if not isinstance(shocks, np.ndarray)
                            else shocks
                        ),
                        y=(
                            np.array(impact)
                            if not isinstance(impact, np.ndarray)
                            else impact
                        ),
                        mode="lines",
                        line={"color": COLORS["accent4"], "width": 2.5},
                        name="News Impact",
                    )
                )
                fig.add_vline(x=0, line_dash="dash", line_color=COLORS["neutral"])
                fig.update_layout(
                    **dark_layout(
                        title="News Impact Curve (Volatility Response to Shocks)",
                        xaxis_title="Return Shock",
                        yaxis_title="Conditional Variance",
                        height=450,
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                raise ValueError("No data")

        except Exception:
            # Simulate an asymmetric NIC
            st.caption("Simulated news impact curve (GARCH model unavailable)")
            shocks = np.linspace(-0.05, 0.05, 200)
            sigma_est = returns.std() if len(returns) > 0 else 0.02
            omega = sigma_est**2 * 0.05
            alpha = 0.08
            beta = 0.88
            gamma = 0.04
            _unused_var = omega / (1 - alpha - beta - gamma / 2)
            impact = (
                omega + alpha * shocks**2 + gamma * np.where(shocks < 0, shocks**2, 0)
            )

            try:
                import plotly.graph_objects as go

                from wraquant.dashboard.components.charts import COLORS, dark_layout

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=shocks,
                        y=impact,
                        mode="lines",
                        line={"color": COLORS["accent4"], "width": 2.5},
                        name="GJR-GARCH NIC",
                    )
                )
                fig.add_vline(x=0, line_dash="dash", line_color=COLORS["neutral"])
                fig.update_layout(
                    **dark_layout(
                        title="Simulated News Impact Curve (GJR-GARCH)",
                        xaxis_title="Return Shock",
                        yaxis_title="Conditional Variance",
                        height=450,
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.line_chart(pd.Series(impact, index=shocks))

        # Vol-of-vol / clustering
        st.markdown("### Volatility Clustering")
        rolling_vol = returns.rolling(vol_window).std() * np.sqrt(252)
        vol_of_vol = rolling_vol.rolling(vol_window).std()

        try:
            import plotly.graph_objects as go

            from wraquant.dashboard.components.charts import COLORS, dark_layout

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=vol_of_vol.dropna().index,
                    y=vol_of_vol.dropna().values,
                    mode="lines",
                    name="Vol-of-Vol",
                    line={"color": COLORS["accent1"], "width": 1.5},
                )
            )
            fig.update_layout(
                **dark_layout(
                    title="Volatility of Volatility (Vol Clustering Indicator)",
                    yaxis_title="Vol-of-Vol",
                    height=350,
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.line_chart(vol_of_vol.dropna())

    # ---- EWMA vs GARCH ----
    with tab_ewma:
        st.subheader("EWMA vs GARCH Comparison")

        ewma_lambda = st.slider(
            "EWMA Lambda", 0.80, 0.99, 0.94, 0.01, key="ewma_lambda"
        )

        # EWMA
        try:
            from wraquant.vol.models import ewma_volatility

            ewma_vol = ewma_volatility(returns, lam=ewma_lambda)
        except Exception:
            ewma_var = np.zeros(len(returns))
            ewma_var[0] = returns.iloc[0] ** 2
            for i in range(1, len(returns)):
                ewma_var[i] = (
                    ewma_lambda * ewma_var[i - 1]
                    + (1 - ewma_lambda) * returns.iloc[i] ** 2
                )
            ewma_vol = pd.Series(np.sqrt(ewma_var) * np.sqrt(252), index=returns.index)

        # GARCH conditional vol (reuse from above if available)
        garch_vol = cond_vol if cond_vol is not None else rv_cc

        try:
            import plotly.graph_objects as go

            from wraquant.dashboard.components.charts import COLORS, dark_layout

            fig = go.Figure()

            ewma_s = (
                ewma_vol.dropna()
                if isinstance(ewma_vol, pd.Series)
                else pd.Series(ewma_vol, index=returns.index).dropna()
            )
            fig.add_trace(
                go.Scatter(
                    x=ewma_s.index,
                    y=ewma_s.values,
                    mode="lines",
                    name=f"EWMA (lambda={ewma_lambda})",
                    line={"color": COLORS["accent2"], "width": 1.5},
                )
            )

            if isinstance(garch_vol, pd.Series):
                gv = garch_vol.dropna()
                fig.add_trace(
                    go.Scatter(
                        x=gv.index,
                        y=gv.values,
                        mode="lines",
                        name=f"{garch_type}",
                        line={"color": COLORS["danger"], "width": 1.5},
                    )
                )

            fig.update_layout(
                **dark_layout(
                    title="EWMA vs GARCH Conditional Volatility",
                    yaxis_title="Volatility",
                    height=450,
                )
            )
            st.plotly_chart(fig, use_container_width=True)

            # Comparison metrics
            st.markdown("### Model Comparison")
            if isinstance(garch_vol, pd.Series) and isinstance(ewma_vol, pd.Series):
                common_idx = garch_vol.dropna().index.intersection(
                    ewma_vol.dropna().index
                )
                if len(common_idx) > 0:
                    diff = ewma_vol.loc[common_idx] - garch_vol.loc[common_idx]
                    c1, c2, c3 = st.columns(3)
                    c1.metric(
                        "Mean EWMA Vol", f"{float(ewma_vol.loc[common_idx].mean()):.1%}"
                    )
                    c2.metric(
                        f"Mean {garch_type} Vol",
                        f"{float(garch_vol.loc[common_idx].mean()):.1%}",
                    )
                    c3.metric(
                        "Avg Difference",
                        f"{float(diff.mean()):.2%}",
                        help="Positive = EWMA higher than GARCH",
                    )

        except ImportError:
            st.line_chart(
                pd.DataFrame(
                    {
                        "EWMA": ewma_vol,
                        "GARCH": (
                            garch_vol if isinstance(garch_vol, pd.Series) else None
                        ),
                    }
                ).dropna()
            )

        # Variance risk premium placeholder
        st.markdown("### Variance Risk Premium")
        st.info(
            "The Variance Risk Premium (VRP) measures the difference between implied volatility "
            "(from options) and realized volatility. A positive VRP means options are 'expensive' "
            "relative to actual moves. Options data integration coming soon."
        )
