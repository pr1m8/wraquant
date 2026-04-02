"""VaR & Stress Testing page -- Value-at-Risk methods and stress scenarios.

Compares Historical, Parametric, Cornish-Fisher, and Monte Carlo VaR
side by side, runs VaR backtesting with breach analysis, computes
Expected Shortfall, and applies historical and custom stress tests.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_returns(ticker: str, period: str = "3y") -> "pd.Series":
    """Fetch daily returns for a single ticker."""
    import yfinance as yf

    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No data for {ticker}")
    close = data["Close"]
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    return close.pct_change().dropna()


def _synthetic_returns(n: int = 756) -> "pd.Series":
    """Generate synthetic return series for demo."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    rets = rng.normal(0.0004, 0.012, n)
    # Add a few tail events
    rets[100] = -0.065
    rets[350] = -0.078
    rets[500] = -0.052
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    return pd.Series(rets, index=idx, name="returns")


def render() -> None:
    """Render the VaR & Stress Testing page."""
    import numpy as np
    import pandas as pd

    st.header("VaR & Stress Testing")

    ticker = st.session_state.get("ticker", "SPY")

    with st.sidebar:
        st.subheader("VaR Settings")
        period = st.selectbox(
            "Lookback", ["1y", "2y", "3y", "5y"], index=2, key="var_period"
        )
        confidence = st.slider(
            "Confidence Level", 0.90, 0.99, 0.95, 0.01, key="var_conf"
        )
        n_mc = st.number_input(
            "Monte Carlo Sims", 1000, 50000, 10000, 1000, key="var_mc_sims"
        )

    # -- Fetch data --------------------------------------------------------

    with st.spinner(f"Loading {ticker} returns..."):
        try:
            returns = _fetch_returns(ticker, period=period)
        except Exception:
            st.info(f"Live data unavailable for {ticker} -- using synthetic returns.")
            returns = _synthetic_returns()

    st.caption(
        f"{len(returns)} daily observations | {returns.index[0].date()} to {returns.index[-1].date()}"
    )

    # -- Compute VaR methods -----------------------------------------------

    alpha = 1 - confidence

    # Historical VaR
    hist_var = float(-np.percentile(returns, alpha * 100))

    # Parametric VaR (Gaussian)
    from scipy import stats as sp_stats

    mu = float(returns.mean())
    sigma = float(returns.std())
    param_var = float(-(mu + sigma * sp_stats.norm.ppf(alpha)))

    # Cornish-Fisher VaR
    try:
        from wraquant.risk.tail import cornish_fisher_var

        cf_result = cornish_fisher_var(returns, alpha=alpha)
        cf_var = cf_result["cf_var"]
        cf_skew = cf_result.get("skewness", float(returns.skew()))
        cf_kurt = cf_result.get("excess_kurtosis", float(returns.kurtosis()))
    except Exception:
        skew = float(returns.skew())
        kurt = float(returns.kurtosis())
        z = sp_stats.norm.ppf(alpha)
        z_cf = (
            z
            + (z**2 - 1) * skew / 6
            + (z**3 - 3 * z) * kurt / 24
            - (2 * z**3 - 5 * z) * skew**2 / 36
        )
        cf_var = float(-(mu + sigma * z_cf))
        cf_skew = skew
        cf_kurt = kurt

    # Monte Carlo VaR
    rng = np.random.default_rng(0)
    mc_sims = rng.normal(mu, sigma, int(n_mc))
    mc_var = float(-np.percentile(mc_sims, alpha * 100))

    # CVaR (Expected Shortfall)
    tail_hist = returns[returns <= -hist_var]
    hist_cvar = float(-tail_hist.mean()) if len(tail_hist) > 0 else hist_var

    tail_param = mc_sims[mc_sims <= -param_var]
    param_cvar = float(-tail_param.mean()) if len(tail_param) > 0 else param_var

    tail_mc = mc_sims[mc_sims <= -mc_var]
    mc_cvar = float(-tail_mc.mean()) if len(tail_mc) > 0 else mc_var

    try:
        from wraquant.risk.var import conditional_var

        hist_cvar_wq = conditional_var(returns, confidence=confidence)
        hist_cvar = hist_cvar_wq if isinstance(hist_cvar_wq, float) else hist_cvar
    except Exception:
        pass

    # -- Tabs --------------------------------------------------------------

    tab_var, tab_backtest, tab_stress, tab_tail = st.tabs(
        ["VaR Comparison", "VaR Backtest", "Stress Testing", "Tail Risk"],
    )

    # ---- VaR Comparison ----
    with tab_var:
        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Historical VaR", f"{hist_var:.2%}")
        k2.metric("Parametric VaR", f"{param_var:.2%}")
        k3.metric("Cornish-Fisher VaR", f"{cf_var:.2%}")
        k4.metric("Monte Carlo VaR", f"{mc_var:.2%}")

        try:
            import plotly.graph_objects as go

            from wraquant.dashboard.components.charts import COLORS, dark_layout

            col_bar, col_cvar = st.columns(2)

            # VaR comparison bar chart
            with col_bar:
                methods = ["Historical", "Parametric", "Cornish-Fisher", "Monte Carlo"]
                var_values = [hist_var, param_var, cf_var, mc_var]
                bar_colors = [
                    COLORS["primary"],
                    COLORS["accent2"],
                    COLORS["accent4"],
                    COLORS["accent1"],
                ]

                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=methods,
                            y=[v * 100 for v in var_values],
                            marker_color=bar_colors,
                            text=[f"{v:.2%}" for v in var_values],
                            textposition="auto",
                        )
                    ]
                )
                fig.update_layout(
                    **dark_layout(
                        title=f"VaR Comparison ({confidence:.0%} Confidence)",
                        yaxis_title="VaR (%)",
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

            # CVaR comparison
            with col_cvar:
                cvar_values = [hist_cvar, param_cvar, cf_var * 1.15, mc_cvar]
                fig2 = go.Figure(
                    data=[
                        go.Bar(
                            x=methods,
                            y=[v * 100 for v in cvar_values],
                            marker_color=[COLORS["danger"]] * 4,
                            text=[f"{v:.2%}" for v in cvar_values],
                            textposition="auto",
                        )
                    ]
                )
                fig2.update_layout(
                    **dark_layout(
                        title="CVaR (Expected Shortfall) Comparison",
                        yaxis_title="CVaR (%)",
                    )
                )
                st.plotly_chart(fig2, use_container_width=True)

        except ImportError:
            st.write("Historical VaR:", f"{hist_var:.2%}")
            st.write("Parametric VaR:", f"{param_var:.2%}")
            st.write("Cornish-Fisher VaR:", f"{cf_var:.2%}")
            st.write("Monte Carlo VaR:", f"{mc_var:.2%}")

    # ---- VaR Backtest ----
    with tab_backtest:
        st.subheader("VaR Backtesting")

        window = st.slider("Rolling VaR Window", 60, 252, 126, key="var_bt_window")

        # Rolling historical VaR
        rolling_var = returns.rolling(window).quantile(alpha).shift(1) * -1
        rolling_var = rolling_var.dropna()
        aligned_returns = returns.loc[rolling_var.index]

        # Breach detection
        breaches = aligned_returns < -rolling_var
        breach_count = int(breaches.sum())
        expected_breaches = int(len(aligned_returns) * alpha)
        breach_rate = (
            breach_count / len(aligned_returns) if len(aligned_returns) > 0 else 0
        )

        b1, b2, b3 = st.columns(3)
        b1.metric("Observed Breaches", str(breach_count))
        b2.metric("Expected Breaches", str(expected_breaches))
        breach_delta = breach_count - expected_breaches
        b3.metric(
            "Breach Ratio",
            f"{breach_rate:.1%}",
            delta=f"{breach_delta:+d} vs expected",
            delta_color="inverse" if breach_delta > 0 else "normal",
        )

        # Kupiec test interpretation
        if breach_count > expected_breaches * 1.5:
            st.error(
                "VaR model **underestimates** risk (too many breaches). Consider using Cornish-Fisher or a fatter-tailed model."
            )
        elif breach_count < expected_breaches * 0.5:
            st.warning(
                "VaR model may be **too conservative** (too few breaches). Capital may be over-allocated."
            )
        else:
            st.success(
                "VaR model appears **well-calibrated**. Breach count is within expected range."
            )

        try:
            import plotly.graph_objects as go

            from wraquant.dashboard.components.charts import COLORS, dark_layout

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=aligned_returns.index,
                    y=aligned_returns.values,
                    mode="lines",
                    name="Daily Returns",
                    line={"color": COLORS["primary"], "width": 1},
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=rolling_var.index,
                    y=-rolling_var.values,
                    mode="lines",
                    name=f"VaR ({confidence:.0%})",
                    line={"color": COLORS["danger"], "width": 2, "dash": "dash"},
                )
            )

            # Mark breaches
            breach_idx = aligned_returns.index[breaches]
            breach_vals = aligned_returns.loc[breach_idx]
            fig.add_trace(
                go.Scatter(
                    x=breach_idx,
                    y=breach_vals.values,
                    mode="markers",
                    name="Breaches",
                    marker={"color": COLORS["warning"], "size": 6, "symbol": "x"},
                )
            )

            fig.update_layout(
                **dark_layout(
                    title="Returns vs Rolling VaR",
                    yaxis_title="Return",
                    height=500,
                )
            )
            fig.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            st.line_chart(
                pd.DataFrame(
                    {
                        "Returns": aligned_returns,
                        "VaR": -rolling_var,
                    }
                )
            )

    # ---- Stress Testing ----
    with tab_stress:
        st.subheader("Stress Test Scenarios")

        # Historical stress test
        try:
            from wraquant.risk.stress import historical_stress_test

            hist_stress = historical_stress_test(returns)

            if "crisis_results" in hist_stress and hist_stress["crisis_results"]:
                st.markdown("### Historical Crisis Replay")
                rows = []
                for name, result in hist_stress["crisis_results"].items():
                    cum_ret = result.get("cumulative_return", 0)
                    max_dd = result.get("max_drawdown", 0)
                    n_days = result.get("n_days", 0)

                    if cum_ret < -0.2:
                        severity = "HIGH"
                    elif cum_ret < -0.05:
                        severity = "MEDIUM"
                    else:
                        severity = "LOW"

                    rows.append(
                        {
                            "Crisis": name.replace("_", " ").title(),
                            "Cum. Return": f"{cum_ret:.1%}",
                            "Max Drawdown": f"{max_dd:.1%}",
                            "Days": n_days,
                            "Severity": severity,
                        }
                    )
                if rows:
                    df_stress = pd.DataFrame(rows)
                    st.dataframe(
                        df_stress.set_index("Crisis"), use_container_width=True
                    )
        except Exception:
            st.info(
                "Historical stress test requires date-indexed data covering crisis periods."
            )

        # Custom scenario stress test
        st.markdown("### Custom Stress Scenarios")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            mild_shock = (
                st.number_input(
                    "Mild Shock (%)", -20.0, 0.0, -2.0, 0.5, key="mild_shock"
                )
                / 100
            )
        with col_s2:
            moderate_shock = (
                st.number_input(
                    "Moderate Shock (%)", -40.0, 0.0, -5.0, 0.5, key="mod_shock"
                )
                / 100
            )
        with col_s3:
            severe_shock = (
                st.number_input(
                    "Severe Shock (%)", -60.0, 0.0, -15.0, 1.0, key="sev_shock"
                )
                / 100
            )

        scenarios = {
            "Mild": mild_shock,
            "Moderate": moderate_shock,
            "Severe": severe_shock,
        }

        try:
            from wraquant.risk.stress import stress_test_returns

            stress_result = stress_test_returns(returns, scenarios)
            scenario_results = stress_result.get("scenario_results", {})
        except Exception:
            # Fallback computation
            scenario_results = {}
            for name, shock in scenarios.items():
                stressed = returns.values + shock
                var_95 = float(np.percentile(stressed, 5))
                tail = stressed[stressed <= var_95]
                scenario_results[name] = {
                    "stressed_mean": float(np.mean(stressed)),
                    "stressed_var_95": var_95,
                    "stressed_cvar_95": (
                        float(np.mean(tail)) if len(tail) > 0 else var_95
                    ),
                }

        try:
            import plotly.graph_objects as go

            from wraquant.dashboard.components.charts import COLORS, dark_layout

            names = list(scenario_results.keys())
            stressed_vars = [
                abs(scenario_results[n].get("stressed_var_95", 0)) for n in names
            ]
            stressed_cvars = [
                abs(scenario_results[n].get("stressed_cvar_95", 0)) for n in names
            ]

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=names,
                    y=[v * 100 for v in stressed_vars],
                    name="Stressed VaR",
                    marker_color=COLORS["warning"],
                )
            )
            fig.add_trace(
                go.Bar(
                    x=names,
                    y=[v * 100 for v in stressed_cvars],
                    name="Stressed CVaR",
                    marker_color=COLORS["danger"],
                )
            )
            fig.update_layout(
                **dark_layout(
                    title="Stress Test: VaR & CVaR Under Scenarios",
                    yaxis_title="Loss (%)",
                    barmode="group",
                )
            )
            st.plotly_chart(fig, use_container_width=True)

            # Traffic-light table
            st.markdown("### Scenario Impact Summary")
            stress_rows = []
            for name, res in scenario_results.items():
                var_val = abs(res.get("stressed_var_95", 0))
                cvar_val = abs(res.get("stressed_cvar_95", 0))
                if cvar_val > 0.10:
                    color = "background-color: rgba(239, 68, 68, 0.3)"
                elif cvar_val > 0.04:
                    color = "background-color: rgba(245, 158, 11, 0.3)"
                else:
                    color = "background-color: rgba(34, 197, 94, 0.3)"
                stress_rows.append(
                    {
                        "Scenario": name,
                        "Shock": f"{scenarios[name]:.1%}",
                        "Stressed VaR": f"{var_val:.2%}",
                        "Stressed CVaR": f"{cvar_val:.2%}",
                        "Stressed Mean": f"{res.get('stressed_mean', 0):.4%}",
                    }
                )
            st.dataframe(
                pd.DataFrame(stress_rows).set_index("Scenario"),
                use_container_width=True,
            )

        except ImportError:
            for name, res in scenario_results.items():
                st.write(
                    f"**{name}**: VaR={abs(res.get('stressed_var_95', 0)):.2%}, "
                    f"CVaR={abs(res.get('stressed_cvar_95', 0)):.2%}"
                )

    # ---- Tail Risk ----
    with tab_tail:
        st.subheader("Tail Risk Analysis")

        col_qq, col_dist = st.columns(2)

        with col_qq:
            st.markdown("### Q-Q Plot (Normal)")
            try:
                import plotly.graph_objects as go

                from wraquant.dashboard.components.charts import COLORS, dark_layout

                sorted_rets = np.sort(returns.values)
                n = len(sorted_rets)
                theoretical = sp_stats.norm.ppf(np.linspace(0.001, 0.999, n))

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=theoretical,
                        y=sorted_rets,
                        mode="markers",
                        marker={"color": COLORS["primary"], "size": 3},
                        name="Returns",
                    )
                )
                # Reference line
                min_val = min(theoretical.min(), sorted_rets.min())
                max_val = max(theoretical.max(), sorted_rets.max())
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val * sigma + mu, max_val * sigma + mu],
                        mode="lines",
                        line={"color": COLORS["danger"], "dash": "dash"},
                        name="Normal Line",
                    )
                )
                fig.update_layout(
                    **dark_layout(
                        title="Q-Q Plot vs Normal Distribution",
                        xaxis_title="Theoretical Quantiles",
                        yaxis_title="Sample Quantiles",
                        height=450,
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

            except ImportError:
                st.info("Plotly required for Q-Q plot.")

        with col_dist:
            st.markdown("### Return Distribution")
            try:
                import plotly.graph_objects as go

                from wraquant.dashboard.components.charts import COLORS, dark_layout

                fig = go.Figure()
                fig.add_trace(
                    go.Histogram(
                        x=returns.values,
                        nbinsx=80,
                        marker_color=COLORS["primary"],
                        opacity=0.7,
                        name="Returns",
                    )
                )
                # Add VaR lines
                fig.add_vline(
                    x=-hist_var,
                    line_dash="dash",
                    line_color=COLORS["danger"],
                    annotation_text=f"VaR {confidence:.0%}",
                    annotation_position="top left",
                )
                fig.add_vline(
                    x=-hist_cvar,
                    line_dash="dash",
                    line_color=COLORS["warning"],
                    annotation_text="CVaR",
                    annotation_position="top left",
                )

                fig.update_layout(
                    **dark_layout(
                        title="Return Distribution with VaR/CVaR",
                        xaxis_title="Return",
                        yaxis_title="Count",
                        height=450,
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

            except ImportError:
                st.info("Plotly required for distribution plot.")

        # Tail statistics
        st.markdown("### Tail Statistics")
        t1, t2, t3, t4 = st.columns(4)
        t1.metric(
            "Skewness",
            f"{cf_skew:.3f}",
            help="Negative = left-skewed (more downside risk)",
        )
        t2.metric(
            "Excess Kurtosis",
            f"{cf_kurt:.3f}",
            help="Positive = fat tails (more extreme events)",
        )

        # Tail ratio
        p95 = float(np.percentile(returns, 95))
        p5 = float(np.percentile(returns, 5))
        tail_ratio_val = abs(p95 / p5) if p5 != 0 else 0.0
        t3.metric(
            "Tail Ratio (95/5)",
            f"{tail_ratio_val:.2f}",
            help=">1 means right tail is larger; <1 means left tail dominates",
        )

        # Hill tail index (simplified)
        sorted_losses = np.sort(-returns.values)
        k = max(int(len(sorted_losses) * 0.05), 5)
        top_k = sorted_losses[:k]
        if top_k[-1] > 0:
            hill_est = 1.0 / (np.mean(np.log(top_k / top_k[-1])) + 1e-10)
        else:
            hill_est = 0.0
        t4.metric(
            "Hill Tail Index",
            f"{hill_est:.2f}",
            help="Lower = fatter tails. Gaussian=inf, Student-t=degrees of freedom",
        )
