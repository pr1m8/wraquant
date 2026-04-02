"""Market Microstructure page -- liquidity, toxicity, market quality, spreads.

Deep microstructure analytics using wraquant.microstructure with
Plotly for interactive charting and daily-frequency OHLCV proxies.
"""

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_ohlcv(ticker: str, days: int = 730):
    """Fetch OHLCV data. Returns DataFrame with ohlcv columns."""
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
            return df
    except Exception:
        pass

    try:
        import yfinance as yf

        data = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
        if not data.empty:
            data.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in data.columns]
            return data
    except Exception:
        pass

    import numpy as np

    rng = np.random.default_rng(42)
    n = min(504, days)
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    high = close * (1 + rng.uniform(0.001, 0.025, n))
    low = close * (1 - rng.uniform(0.001, 0.025, n))
    opn = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(500_000, 10_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def render() -> None:
    """Render the Market Microstructure page."""
    import numpy as np
    import pandas as pd

    from wraquant.dashboard.components.charts import COLORS, dark_layout

    ticker = st.session_state.get("ticker", "AAPL")
    st.markdown(f"# Market Microstructure: **{ticker}**")

    with st.spinner(f"Loading {ticker} OHLCV data..."):
        df = _fetch_ohlcv(ticker)

    if df is None or len(df) < 60:
        st.warning("Insufficient data for microstructure analysis.")
        return

    # Normalize column names
    col_map = {}
    for c in df.columns:
        cl = c.lower() if isinstance(c, str) else str(c).lower()
        col_map[c] = cl
    df = df.rename(columns=col_map)

    close = df.get("close")
    high = df.get("high")
    low = df.get("low")
    volume = df.get("volume")
    returns = close.pct_change().dropna() if close is not None else pd.Series(dtype=float)

    if close is None or len(close) < 60:
        st.warning("Missing close price data.")
        return

    st.caption(f"{len(df)} daily observations")

    tab_liq, tab_tox, tab_quality, tab_spread = st.tabs([
        "Liquidity",
        "Toxicity",
        "Market Quality",
        "Spread Analysis",
    ])

    # ---- Liquidity Tab ----
    with tab_liq:
        st.subheader("Liquidity Metrics")

        liquidity_metrics = {}

        # Amihud Illiquidity
        try:
            from wraquant.microstructure.liquidity import amihud_illiquidity

            amihud = amihud_illiquidity(returns, volume)
            if isinstance(amihud, dict):
                amihud_val = amihud.get("amihud", amihud.get("illiquidity", None))
                amihud_series = amihud.get("rolling", amihud.get("series", None))
            elif isinstance(amihud, pd.Series):
                amihud_val = float(amihud.mean())
                amihud_series = amihud
            else:
                amihud_val = float(amihud) if amihud is not None else None
                amihud_series = None
            if amihud_val is not None:
                liquidity_metrics["Amihud Illiquidity"] = amihud_val
        except Exception:
            if volume is not None and len(returns) > 0:
                amihud_raw = (returns.abs() / volume).replace([np.inf, -np.inf], np.nan)
                amihud_val = float(amihud_raw.mean())
                amihud_series = amihud_raw.rolling(21).mean()
                liquidity_metrics["Amihud Illiquidity"] = amihud_val
            else:
                amihud_val = None
                amihud_series = None

        # Kyle's Lambda
        try:
            from wraquant.microstructure.liquidity import kyle_lambda

            kyle = kyle_lambda(returns, volume)
            if isinstance(kyle, dict):
                kyle_val = kyle.get("lambda", kyle.get("kyle_lambda", None))
            else:
                kyle_val = float(kyle) if kyle is not None else None
            if kyle_val is not None:
                liquidity_metrics["Kyle's Lambda"] = kyle_val
        except Exception:
            if volume is not None and len(returns) > 10:
                try:
                    slope = np.polyfit(
                        np.sign(returns.values[1:]) * np.sqrt(volume.values[1:]),
                        returns.values[1:],
                        1,
                    )[0]
                    kyle_val = abs(slope)
                    liquidity_metrics["Kyle's Lambda"] = kyle_val
                except Exception:
                    kyle_val = None
            else:
                kyle_val = None

        # Roll Spread
        try:
            from wraquant.microstructure.liquidity import roll_spread

            roll = roll_spread(returns)
            if isinstance(roll, dict):
                roll_val = roll.get("spread", roll.get("roll_spread", None))
            else:
                roll_val = float(roll) if roll is not None else None
            if roll_val is not None:
                liquidity_metrics["Roll Spread"] = roll_val
        except Exception:
            cov_lag = float(returns.iloc[:-1].values @ returns.iloc[1:].values) / (len(returns) - 1)
            if cov_lag < 0:
                roll_val = 2 * np.sqrt(abs(cov_lag))
                liquidity_metrics["Roll Spread"] = roll_val
            else:
                roll_val = None

        # Effective Spread
        try:
            from wraquant.microstructure.liquidity import effective_spread

            eff = effective_spread(close, volume)
            if isinstance(eff, dict):
                eff_val = eff.get("effective_spread", eff.get("spread", None))
            else:
                eff_val = float(eff) if eff is not None else None
            if eff_val is not None:
                liquidity_metrics["Effective Spread"] = eff_val
        except Exception:
            pass

        # Display gauges
        if liquidity_metrics:
            gauge_cols = st.columns(len(liquidity_metrics))
            for col, (name, val) in zip(gauge_cols, liquidity_metrics.items(), strict=False):
                col.metric(name, f"{val:.6f}")

            # Amihud trend chart
            if amihud_series is not None:
                try:
                    import plotly.graph_objects as go

                    s = amihud_series.dropna()
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=s.index, y=s.values, mode="lines",
                            line={"color": COLORS["primary"], "width": 1.5},
                            name="Amihud (rolling 21d)",
                        )
                    )
                    fig.update_layout(
                        **dark_layout(
                            title="Amihud Illiquidity (Rolling 21-day)",
                            yaxis_title="Illiquidity", height=350,
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.line_chart(amihud_series.dropna())

            # Kyle lambda rolling
            try:
                from wraquant.microstructure.liquidity import lambda_kyle_rolling

                kyle_roll = lambda_kyle_rolling(returns, volume, window=60)
                if isinstance(kyle_roll, pd.Series):
                    try:
                        import plotly.graph_objects as go

                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=kyle_roll.dropna().index, y=kyle_roll.dropna().values,
                                mode="lines",
                                line={"color": COLORS["accent2"], "width": 1.5},
                                name="Kyle Lambda (60d)",
                            )
                        )
                        fig.update_layout(
                            **dark_layout(
                                title="Kyle's Lambda (Rolling 60-day)",
                                yaxis_title="Lambda", height=350,
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        st.line_chart(kyle_roll.dropna())
            except Exception:
                pass
        else:
            st.info("No liquidity metrics could be computed.")

        # Turnover
        try:
            from wraquant.microstructure.liquidity import turnover_ratio

            to = turnover_ratio(volume, close)
            if isinstance(to, (pd.Series, dict)):
                st.subheader("Turnover Ratio")
                if isinstance(to, pd.Series):
                    st.line_chart(to.rolling(21).mean().dropna())
                else:
                    st.metric("Avg Turnover", f"{to.get('turnover', 'N/A')}")
        except Exception:
            pass

    # ---- Toxicity Tab ----
    with tab_tox:
        st.subheader("Order Flow Toxicity")

        # VPIN estimate
        try:
            from wraquant.microstructure.toxicity import vpin

            with st.spinner("Computing VPIN..."):
                # Align lengths (returns is 1 shorter than volume)
                n = min(len(returns), len(volume))
                vpin_result = vpin(returns.iloc[-n:], volume.iloc[-n:])

            if isinstance(vpin_result, dict):
                vpin_val = vpin_result.get("vpin", vpin_result.get("mean_vpin", None))
                vpin_series = vpin_result.get("series", vpin_result.get("vpin_series", None))
            elif isinstance(vpin_result, pd.Series):
                vpin_val = float(vpin_result.mean())
                vpin_series = vpin_result
            else:
                vpin_val = float(vpin_result) if vpin_result is not None else None
                vpin_series = None

            if vpin_val is not None:
                vc1, vc2 = st.columns(2)
                vc1.metric("VPIN (avg)", f"{vpin_val:.4f}")
                level = "Low" if vpin_val < 0.3 else "Moderate" if vpin_val < 0.6 else "High"
                vc2.metric("Toxicity Level", level)

            if vpin_series is not None and isinstance(vpin_series, pd.Series):
                try:
                    import plotly.graph_objects as go

                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=vpin_series.index, y=vpin_series.values,
                            mode="lines",
                            line={"color": COLORS["warning"], "width": 1.5},
                            name="VPIN",
                        )
                    )
                    fig.add_hline(y=0.5, line_dash="dash", line_color=COLORS["danger"],
                                  annotation_text="High Toxicity Threshold")
                    fig.update_layout(
                        **dark_layout(title="VPIN Time Series", yaxis_title="VPIN", height=400)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.line_chart(vpin_series)

        except Exception as e:
            st.info(f"VPIN unavailable: {e}")

        # Order flow imbalance
        st.divider()
        st.subheader("Order Flow Imbalance")
        try:
            from wraquant.microstructure.toxicity import order_flow_imbalance

            ofi = order_flow_imbalance(returns, volume)

            if isinstance(ofi, pd.Series):
                ofi_series = ofi
            elif isinstance(ofi, dict):
                ofi_series = ofi.get("imbalance", ofi.get("ofi", None))
                if isinstance(ofi_series, pd.Series):
                    pass
                else:
                    ofi_series = None
            else:
                ofi_series = None

            if ofi_series is not None:
                try:
                    import plotly.graph_objects as go

                    roll_ofi = ofi_series.rolling(21).mean().dropna()
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=roll_ofi.index, y=roll_ofi.values,
                            mode="lines",
                            line={"color": COLORS["accent4"], "width": 1.5},
                            name="OFI (21d avg)",
                        )
                    )
                    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"])
                    fig.update_layout(
                        **dark_layout(
                            title="Order Flow Imbalance (Rolling 21d)",
                            yaxis_title="Imbalance", height=350,
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.line_chart(ofi_series.rolling(21).mean().dropna())
        except Exception as e:
            st.info(f"Order flow imbalance unavailable: {e}")
            # Proxy: sign of return * volume
            if volume is not None:
                ofi_proxy = np.sign(returns) * volume
                ofi_roll = ofi_proxy.rolling(21).mean().dropna()
                st.line_chart(ofi_roll)
                st.caption("Proxy: sign(return) x volume")

        # Bulk volume classification
        try:
            from wraquant.microstructure.toxicity import bulk_volume_classification

            bvc = bulk_volume_classification(close, volume)
            if isinstance(bvc, (pd.Series, dict)):
                st.subheader("Bulk Volume Classification")
                if isinstance(bvc, pd.Series):
                    st.line_chart(bvc.rolling(21).mean().dropna())
                elif isinstance(bvc, dict) and "buy_volume_pct" in bvc:
                    st.metric("Buy Volume %", f"{bvc['buy_volume_pct']:.1%}")
        except Exception:
            pass

    # ---- Market Quality Tab ----
    with tab_quality:
        st.subheader("Market Quality & Efficiency")

        quality_metrics = {}

        # Variance ratio
        try:
            from wraquant.microstructure.market_quality import variance_ratio

            vr = variance_ratio(returns)
            if isinstance(vr, dict):
                vr_val = vr.get("variance_ratio", vr.get("ratio", None))
                vr_stat = vr.get("z_statistic", vr.get("statistic", None))
                vr_p = vr.get("p_value", None)
            else:
                vr_val = float(vr) if vr is not None else None
                vr_stat = None
                vr_p = None

            if vr_val is not None:
                quality_metrics["Variance Ratio"] = vr_val
        except Exception:
            if len(returns) > 20:
                var_1 = float(returns.var())
                var_5 = float(returns.rolling(5).sum().var() / 5)
                vr_val = var_5 / var_1 if var_1 > 0 else 1.0
                quality_metrics["Variance Ratio"] = vr_val
                vr_stat = None
                vr_p = None
            else:
                vr_val = None

        # Market efficiency ratio
        try:
            from wraquant.microstructure.market_quality import market_efficiency_ratio

            mer = market_efficiency_ratio(returns)
            if isinstance(mer, dict):
                mer_val = mer.get("efficiency_ratio", mer.get("ratio", None))
            else:
                mer_val = float(mer) if mer is not None else None
            if mer_val is not None:
                quality_metrics["Efficiency Ratio"] = mer_val
        except Exception:
            pass

        # Autocorrelation (lag-1)
        acf1 = float(returns.autocorr(lag=1)) if len(returns) > 10 else 0.0
        quality_metrics["Autocorrelation (lag-1)"] = acf1

        if quality_metrics:
            qcols = st.columns(len(quality_metrics))
            for col, (name, val) in zip(qcols, quality_metrics.items(), strict=False):
                col.metric(name, f"{val:.4f}")

        if vr_val is not None:
            st.markdown(
                f"**Interpretation:** Variance ratio = {vr_val:.4f}. "
                f"{'Close to 1 suggests random walk (efficient market).' if abs(vr_val - 1) < 0.15 else 'Deviation from 1 suggests predictability.'}"
            )

        # Autocorrelation chart
        st.divider()
        st.subheader("Return Autocorrelation")
        try:
            import plotly.graph_objects as go

            max_lag = min(30, len(returns) // 5)
            acf_vals = [float(returns.autocorr(lag=i)) for i in range(1, max_lag + 1)]

            # Confidence band (approximate)
            conf = 1.96 / np.sqrt(len(returns))

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=list(range(1, max_lag + 1)), y=acf_vals,
                    marker_color=[
                        COLORS["danger"] if abs(v) > conf else COLORS["primary"]
                        for v in acf_vals
                    ],
                    name="ACF",
                )
            )
            fig.add_hline(y=conf, line_dash="dash", line_color=COLORS["neutral"], opacity=0.5)
            fig.add_hline(y=-conf, line_dash="dash", line_color=COLORS["neutral"], opacity=0.5)
            fig.add_hline(y=0, line_color=COLORS["neutral"], opacity=0.3)

            fig.update_layout(
                **dark_layout(
                    title="Autocorrelation Function",
                    xaxis_title="Lag", yaxis_title="ACF", height=350,
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            pass

        # Squared returns autocorrelation (volatility clustering)
        st.subheader("Squared Returns Autocorrelation (Volatility Clustering)")
        try:
            import plotly.graph_objects as go

            sq_returns = returns ** 2
            sq_acf = [float(sq_returns.autocorr(lag=i)) for i in range(1, max_lag + 1)]
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=list(range(1, max_lag + 1)), y=sq_acf,
                    marker_color=COLORS["accent2"],
                    name="ACF(r^2)",
                )
            )
            fig.add_hline(y=conf, line_dash="dash", line_color=COLORS["neutral"], opacity=0.5)
            fig.add_hline(y=-conf, line_dash="dash", line_color=COLORS["neutral"], opacity=0.5)
            fig.update_layout(
                **dark_layout(
                    title="ACF of Squared Returns",
                    xaxis_title="Lag", yaxis_title="ACF", height=350,
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            pass

    # ---- Spread Analysis Tab ----
    with tab_spread:
        st.subheader("Spread Estimation")

        if high is None or low is None:
            st.warning("High/Low prices required for spread estimation.")
        else:
            # Corwin-Schultz spread
            try:
                from wraquant.microstructure.liquidity import corwin_schultz_spread

                cs = corwin_schultz_spread(high, low)
                if isinstance(cs, dict):
                    cs_val = cs.get("spread", cs.get("mean_spread", None))
                    cs_series = cs.get("series", cs.get("spread_series", None))
                elif isinstance(cs, pd.Series):
                    cs_val = float(cs.mean())
                    cs_series = cs
                else:
                    cs_val = float(cs) if cs is not None else None
                    cs_series = None
            except Exception:
                # Fallback: Corwin-Schultz approximation
                try:
                    beta = (np.log(high / low) ** 2).rolling(2).sum()
                    gamma = np.log(
                        high.rolling(2).max() / low.rolling(2).min()
                    ) ** 2
                    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (
                        3 - 2 * np.sqrt(2)
                    ) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
                    cs_series = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
                    cs_series = cs_series.clip(lower=0)
                    cs_val = float(cs_series.mean())
                except Exception:
                    cs_val = None
                    cs_series = None

            if cs_val is not None:
                sc1, sc2 = st.columns(2)
                sc1.metric("Corwin-Schultz Spread (avg)", f"{cs_val:.6f}")
                sc2.metric("Spread (bps)", f"{cs_val * 10000:.1f} bps")

                if cs_series is not None and isinstance(cs_series, pd.Series):
                    try:
                        import plotly.graph_objects as go

                        cs_smooth = cs_series.rolling(21).mean().dropna()
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=cs_smooth.index, y=cs_smooth.values,
                                mode="lines",
                                line={"color": COLORS["primary"], "width": 1.5},
                                name="CS Spread (21d avg)",
                            )
                        )
                        fig.update_layout(
                            **dark_layout(
                                title="Corwin-Schultz Spread (Rolling 21-day)",
                                yaxis_title="Spread", height=400,
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        st.line_chart(cs_series.rolling(21).mean().dropna())

            # Roll spread over time
            st.divider()
            st.subheader("Roll Spread (Rolling)")
            try:
                roll_window = 60
                roll_spreads = []
                for i in range(roll_window, len(returns)):
                    window_rets = returns.iloc[i - roll_window:i]
                    cov = float(window_rets.iloc[:-1].values @ window_rets.iloc[1:].values) / (roll_window - 1)
                    if cov < 0:
                        roll_spreads.append(2 * np.sqrt(abs(cov)))
                    else:
                        roll_spreads.append(0.0)

                roll_spread_series = pd.Series(
                    roll_spreads, index=returns.index[roll_window:]
                )

                try:
                    import plotly.graph_objects as go

                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=roll_spread_series.index, y=roll_spread_series.values,
                            mode="lines",
                            line={"color": COLORS["accent4"], "width": 1.5},
                            name="Roll Spread (60d)",
                        )
                    )
                    fig.update_layout(
                        **dark_layout(
                            title="Roll Spread (Rolling 60-day Window)",
                            yaxis_title="Implied Spread", height=350,
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.line_chart(roll_spread_series)

            except Exception as e:
                st.info(f"Roll spread computation failed: {e}")

            # High-low spread estimator comparison
            st.divider()
            st.subheader("Spread Estimator Comparison")
            comparison = {}
            if cs_val is not None:
                comparison["Corwin-Schultz"] = f"{cs_val:.6f}"
            try:
                comparison["Roll Spread"] = f"{roll_val:.6f}" if roll_val is not None else "N/A"
            except Exception:
                pass

            try:
                from wraquant.microstructure.liquidity import effective_spread as eff_sp

                eff = eff_sp(close, volume)
                eff_v = eff.get("effective_spread", eff) if isinstance(eff, dict) else eff
                comparison["Effective Spread"] = f"{float(eff_v):.6f}"
            except Exception:
                pass

            if comparison:
                st.dataframe(
                    pd.DataFrame([comparison]).T.rename(columns={0: "Value"}),
                    use_container_width=True,
                )
