"""Strategy Backtesting Lab -- interactive strategy evaluation.

Supports momentum, mean reversion, buy & hold, and regime-filtered
strategies with configurable parameters. Displays equity curves,
performance metrics, monthly return heatmaps, drawdown charts,
rolling Sharpe, and trade statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_prices(ticker: str, period: str = "5y") -> "pd.Series":
    """Fetch adjusted close prices."""
    import yfinance as yf

    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No data for {ticker}")
    close = data["Close"]
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    return close


def _synthetic_prices(n: int = 1260) -> "pd.Series":
    """Generate synthetic prices for demo."""
    import pandas as pd

    rng = np.random.default_rng(42)
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    rets = rng.normal(0.0004, 0.013, n)
    prices = 100 * np.exp(np.cumsum(rets))
    return pd.Series(prices, index=idx, name="Close")


# ---------------------------------------------------------------------------
# Strategy signal generators
# ---------------------------------------------------------------------------


def _momentum_signal(
    prices: "pd.Series", lookback: int, threshold: float
) -> "pd.Series":
    """Momentum strategy: long when lookback return > threshold."""
    import pandas as pd

    mom = prices.pct_change(lookback)
    signal = pd.Series(0.0, index=prices.index)
    signal[mom > threshold] = 1.0
    signal[mom < -threshold] = -1.0
    return signal.shift(1).fillna(0)


def _mean_reversion_signal(
    prices: "pd.Series", lookback: int, z_threshold: float
) -> "pd.Series":
    """Mean reversion: short when z-score > threshold, long when z-score < -threshold."""
    import pandas as pd

    ma = prices.rolling(lookback).mean()
    std = prices.rolling(lookback).std()
    z = (prices - ma) / std.replace(0, float("nan"))
    signal = pd.Series(0.0, index=prices.index)
    signal[z < -z_threshold] = 1.0
    signal[z > z_threshold] = -1.0
    return signal.shift(1).fillna(0)


def _buy_hold_signal(prices: "pd.Series") -> "pd.Series":
    """Buy & Hold: always long."""
    import pandas as pd

    return pd.Series(1.0, index=prices.index)


def _regime_filtered_signal(
    prices: "pd.Series", returns: "pd.Series", lookback: int
) -> "pd.Series":
    """Regime-filtered momentum: long only in low-vol regime."""
    import pandas as pd

    # Rolling vol regime
    rolling_vol = returns.rolling(63).std()
    vol_median = rolling_vol.expanding().median()
    low_vol = rolling_vol <= vol_median

    # Momentum signal
    mom = prices.pct_change(lookback)
    signal = pd.Series(0.0, index=prices.index)
    signal[(mom > 0) & low_vol] = 1.0
    return signal.shift(1).fillna(0)


def render() -> None:
    """Render the Strategy Backtesting Lab page."""
    import pandas as pd

    st.header("Strategy Backtesting Lab")

    ticker = st.session_state.get("ticker", "SPY")

    # -- Sidebar controls --------------------------------------------------

    with st.sidebar:
        st.subheader("Backtest Settings")
        period = st.selectbox(
            "Data Period", ["2y", "3y", "5y", "10y"], index=2, key="bt_period"
        )
        strategy = st.selectbox(
            "Strategy",
            ["Momentum", "Mean Reversion", "Buy & Hold", "Regime-Filtered Momentum"],
            key="bt_strategy",
        )

        st.markdown("**Strategy Parameters**")
        if strategy == "Momentum":
            lookback = st.slider("Lookback (days)", 5, 252, 63, key="bt_mom_lb")
            threshold = st.slider("Threshold", 0.0, 0.20, 0.02, 0.005, key="bt_mom_th")
        elif strategy == "Mean Reversion":
            lookback = st.slider("Lookback (days)", 10, 252, 42, key="bt_mr_lb")
            z_threshold = st.slider(
                "Z-Score Threshold", 0.5, 3.0, 1.5, 0.1, key="bt_mr_z"
            )
        elif strategy == "Regime-Filtered Momentum":
            lookback = st.slider("Momentum Lookback", 10, 252, 63, key="bt_rf_lb")
        # Buy & Hold has no parameters

        include_costs = st.checkbox(
            "Include Transaction Costs", value=True, key="bt_costs"
        )
        cost_bps = (
            st.slider("Cost (bps/trade)", 1, 50, 10, key="bt_cost_bps")
            if include_costs
            else 0
        )
        benchmark_ticker = st.text_input("Benchmark", value="SPY", key="bt_bench")

    # -- Fetch data --------------------------------------------------------

    with st.spinner(f"Loading {ticker} prices..."):
        try:
            prices = _fetch_prices(ticker, period=period)
        except Exception:
            st.info(f"Live data unavailable for {ticker} -- using synthetic data.")
            prices = _synthetic_prices()

    returns = prices.pct_change().dropna()
    prices = prices.loc[returns.index]

    # Benchmark
    benchmark_returns = None
    if benchmark_ticker.strip().upper() != ticker:
        try:
            bench_prices = _fetch_prices(
                benchmark_ticker.strip().upper(), period=period
            )
            benchmark_returns = bench_prices.pct_change().dropna()
            common_idx = returns.index.intersection(benchmark_returns.index)
            returns = returns.loc[common_idx]
            prices = prices.loc[common_idx]
            benchmark_returns = benchmark_returns.loc[common_idx]
        except Exception:
            benchmark_returns = None
    else:
        benchmark_returns = returns.copy()

    # -- Generate signal ---------------------------------------------------

    if strategy == "Momentum":
        signal = _momentum_signal(prices, lookback, threshold)
    elif strategy == "Mean Reversion":
        signal = _mean_reversion_signal(prices, lookback, z_threshold)
    elif strategy == "Buy & Hold":
        signal = _buy_hold_signal(prices)
    else:
        signal = _regime_filtered_signal(prices, returns, lookback)

    signal = signal.loc[returns.index]

    # -- Compute strategy returns ------------------------------------------

    strat_returns = signal * returns

    # Transaction costs
    if include_costs and cost_bps > 0:
        trades = signal.diff().abs()
        cost_per_trade = cost_bps / 10000
        strat_returns = strat_returns - trades * cost_per_trade

    strat_returns = strat_returns.dropna()
    strat_equity = (1 + strat_returns).cumprod()

    if benchmark_returns is not None:
        bench_equity = (1 + benchmark_returns.loc[strat_returns.index]).cumprod()
    else:
        bench_equity = (1 + returns.loc[strat_returns.index]).cumprod()
        benchmark_returns = returns

    # -- Performance summary -----------------------------------------------

    try:
        from wraquant.backtest.metrics import performance_summary

        perf = performance_summary(strat_returns)
    except Exception:
        total_return = float((1 + strat_returns).prod() - 1)
        n = len(strat_returns)
        ann_factor = 252 / n if n > 0 else 1
        ann_ret = float((1 + total_return) ** ann_factor - 1)
        ann_vol = float(strat_returns.std() * np.sqrt(252))
        sr = ann_ret / ann_vol if ann_vol > 0 else 0
        cum = (1 + strat_returns).cumprod()
        mdd = float((cum / cum.cummax() - 1).min())
        wins = int((strat_returns > 0).sum())
        wr = wins / n if n > 0 else 0
        gains = float(strat_returns[strat_returns > 0].sum())
        losses_val = float(abs(strat_returns[strat_returns < 0].sum()))
        pf = gains / losses_val if losses_val > 0 else float("inf")
        perf = {
            "total_return": total_return,
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sr,
            "sortino_ratio": sr,  # fallback
            "max_drawdown": mdd,
            "calmar_ratio": ann_ret / abs(mdd) if mdd != 0 else 0,
            "win_rate": wr,
            "profit_factor": pf,
            "n_periods": n,
        }

    # KPI row
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Return", f"{perf['total_return']:.1%}")
    k2.metric("Sharpe", f"{perf['sharpe_ratio']:.2f}")
    k3.metric("Max Drawdown", f"{perf['max_drawdown']:.1%}", delta_color="inverse")
    k4.metric("Win Rate", f"{perf['win_rate']:.0%}")
    k5.metric("Profit Factor", f"{perf['profit_factor']:.2f}")
    k6.metric("Calmar", f"{perf['calmar_ratio']:.2f}")

    st.divider()

    # -- Tabs --------------------------------------------------------------

    tab_eq, tab_metrics, tab_monthly, tab_dd, tab_trades = st.tabs(
        [
            "Equity Curve",
            "Metrics & Rolling",
            "Monthly Returns",
            "Drawdowns",
            "Trade Statistics",
        ],
    )

    # ---- Equity Curve ----
    with tab_eq:
        st.subheader("Equity Curve: Strategy vs Benchmark")
        try:
            import plotly.graph_objects as go

            from wraquant.dashboard.components.charts import COLORS, dark_layout

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=strat_equity.index,
                    y=strat_equity.values,
                    mode="lines",
                    name=f"{strategy}",
                    line={"color": COLORS["primary"], "width": 2},
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=bench_equity.index,
                    y=bench_equity.values,
                    mode="lines",
                    name=f"Benchmark ({benchmark_ticker})",
                    line={"color": COLORS["neutral"], "width": 1.5, "dash": "dash"},
                )
            )

            fig.update_layout(
                **dark_layout(
                    title=f"{strategy} vs {benchmark_ticker}",
                    yaxis_title="Growth of $1",
                    height=450,
                )
            )
            st.plotly_chart(fig, use_container_width=True)

            # Signal chart
            st.subheader("Position Signal")
            sig_aligned = signal.loc[strat_returns.index]
            fig_sig = go.Figure()
            fig_sig.add_trace(
                go.Scatter(
                    x=sig_aligned.index,
                    y=sig_aligned.values,
                    mode="lines",
                    name="Signal",
                    line={"color": COLORS["accent4"], "width": 1},
                    fill="tozeroy",
                    fillcolor="rgba(52, 211, 153, 0.2)",
                )
            )
            fig_sig.update_layout(
                **dark_layout(
                    title="Position Over Time (-1=Short, 0=Flat, 1=Long)",
                    yaxis_title="Position",
                    height=250,
                )
            )
            st.plotly_chart(fig_sig, use_container_width=True)

        except ImportError:
            chart_df = pd.DataFrame(
                {"Strategy": strat_equity, "Benchmark": bench_equity}
            )
            st.line_chart(chart_df)

    # ---- Metrics & Rolling ----
    with tab_metrics:
        col_perf, col_roll = st.columns(2)

        with col_perf:
            st.subheader("Performance Summary")
            rows = [
                ("Total Return", f"{perf['total_return']:.1%}"),
                ("Annualized Return", f"{perf['annualized_return']:.1%}"),
                ("Annualized Volatility", f"{perf['annualized_volatility']:.1%}"),
                ("Sharpe Ratio", f"{perf['sharpe_ratio']:.2f}"),
                ("Sortino Ratio", f"{perf['sortino_ratio']:.2f}"),
                ("Max Drawdown", f"{perf['max_drawdown']:.1%}"),
                ("Calmar Ratio", f"{perf['calmar_ratio']:.2f}"),
                ("Win Rate", f"{perf['win_rate']:.0%}"),
                ("Profit Factor", f"{perf['profit_factor']:.2f}"),
                ("# Trading Days", str(perf["n_periods"])),
            ]

            # Additional metrics
            try:
                from wraquant.backtest.metrics import kelly_fraction, omega_ratio

                omega = omega_ratio(strat_returns)
                kelly = kelly_fraction(strat_returns)
                rows.append(("Omega Ratio", f"{omega:.2f}"))
                rows.append(("Kelly Fraction", f"{kelly:.1%}"))
            except Exception:
                pass

            perf_df = pd.DataFrame(rows, columns=["Metric", "Value"]).set_index(
                "Metric"
            )
            st.dataframe(perf_df, use_container_width=True)

        with col_roll:
            st.subheader("Rolling Sharpe Ratio")
            roll_window = st.slider("Window", 21, 252, 63, key="bt_roll_w")
            roll_mean = strat_returns.rolling(roll_window).mean() * 252
            roll_std = strat_returns.rolling(roll_window).std() * np.sqrt(252)
            roll_sharpe = (roll_mean / roll_std).dropna()

            try:
                import plotly.graph_objects as go

                from wraquant.dashboard.components.charts import COLORS, dark_layout

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=roll_sharpe.index,
                        y=roll_sharpe.values,
                        mode="lines",
                        name="Rolling Sharpe",
                        line={"color": COLORS["primary"], "width": 1.5},
                    )
                )
                fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"])
                fig.add_hline(
                    y=1,
                    line_dash="dot",
                    line_color=COLORS["success"],
                    annotation_text="Sharpe=1",
                    annotation_position="top right",
                )
                fig.update_layout(
                    **dark_layout(
                        title=f"Rolling {roll_window}-Day Sharpe Ratio",
                        yaxis_title="Sharpe",
                        height=400,
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.line_chart(roll_sharpe)

    # ---- Monthly Returns ----
    with tab_monthly:
        st.subheader("Monthly Returns Heatmap")

        try:
            from wraquant.backtest.tearsheet import monthly_returns_table

            monthly = monthly_returns_table(strat_returns)
        except Exception:
            # Fallback
            monthly_rets = (1 + strat_returns).groupby(
                [strat_returns.index.year, strat_returns.index.month]
            ).prod() - 1
            monthly_rets.index.names = ["year", "month"]
            monthly = monthly_rets.unstack(level="month")
            if isinstance(monthly.columns, pd.MultiIndex):
                monthly.columns = monthly.columns.droplevel(0)

        if not monthly.empty:
            month_names = {
                1: "Jan",
                2: "Feb",
                3: "Mar",
                4: "Apr",
                5: "May",
                6: "Jun",
                7: "Jul",
                8: "Aug",
                9: "Sep",
                10: "Oct",
                11: "Nov",
                12: "Dec",
            }
            monthly.columns = [month_names.get(c, str(c)) for c in monthly.columns]

            try:
                import plotly.graph_objects as go

                from wraquant.dashboard.components.charts import dark_layout

                fig = go.Figure(
                    data=go.Heatmap(
                        z=monthly.values * 100,
                        x=monthly.columns.tolist(),
                        y=[str(y) for y in monthly.index],
                        colorscale=[[0, "#ef4444"], [0.5, "#1e1e28"], [1, "#22c55e"]],
                        zmid=0,
                        text=monthly.applymap(
                            lambda v: f"{v:.1%}" if pd.notna(v) else ""
                        ).values,
                        texttemplate="%{text}",
                        textfont={"size": 10},
                        colorbar={"title": "Return (%)"},
                    )
                )
                fig.update_layout(
                    **dark_layout(
                        title="Monthly Returns (%)",
                        height=max(300, len(monthly) * 35 + 100),
                    )
                )
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.dataframe(monthly.style.format("{:.1%}"), use_container_width=True)

            # Annual returns
            st.subheader("Annual Returns")
            annual = monthly.sum(axis=1)
            try:
                import plotly.graph_objects as go

                from wraquant.dashboard.components.charts import COLORS, dark_layout

                bar_colors = [
                    COLORS["success"] if v >= 0 else COLORS["danger"]
                    for v in annual.values
                ]
                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=[str(y) for y in annual.index],
                            y=annual.values * 100,
                            marker_color=bar_colors,
                            text=[f"{v:.1%}" for v in annual.values],
                            textposition="auto",
                        )
                    ]
                )
                fig.update_layout(
                    **dark_layout(
                        title="Annual Returns",
                        yaxis_title="Return (%)",
                        height=350,
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(annual)

    # ---- Drawdowns ----
    with tab_dd:
        st.subheader("Drawdown Analysis")
        cum = (1 + strat_returns).cumprod()
        dd = cum / cum.cummax() - 1

        try:
            import plotly.graph_objects as go

            from wraquant.dashboard.components.charts import COLORS, dark_layout

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
            fig.update_layout(
                **dark_layout(
                    title="Strategy Drawdown",
                    yaxis_title="Drawdown",
                    height=400,
                )
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            st.area_chart(dd)

        # Top drawdowns table
        st.subheader("Top 5 Drawdowns")
        # Find drawdown episodes
        dd < 0
        dd_episodes = []
        i = 0
        dd_vals = dd.values
        while i < len(dd_vals):
            if dd_vals[i] < -0.001:
                start = i
                trough = i
                min_dd = dd_vals[i]
                while i < len(dd_vals) and dd_vals[i] < -0.001:
                    if dd_vals[i] < min_dd:
                        min_dd = dd_vals[i]
                        trough = i
                    i += 1
                end = i - 1
                dd_episodes.append(
                    {
                        "Start": str(dd.index[start].date()),
                        "Trough": str(dd.index[trough].date()),
                        "Recovery": str(dd.index[end].date()),
                        "Depth": f"{min_dd:.1%}",
                        "Duration": f"{end - start + 1} days",
                    }
                )
            else:
                i += 1

        dd_episodes.sort(key=lambda x: float(x["Depth"].rstrip("%")) / 100)
        if dd_episodes:
            st.dataframe(pd.DataFrame(dd_episodes[:5]), use_container_width=True)

    # ---- Trade Statistics ----
    with tab_trades:
        st.subheader("Trade Statistics")

        sig_aligned = signal.loc[strat_returns.index]

        # Count trades (position changes)
        position_changes = sig_aligned.diff().abs()
        n_trades = int((position_changes > 0).sum())
        total_days = len(strat_returns)

        # Winning/losing days
        winning_days = strat_returns[strat_returns > 0]
        losing_days = strat_returns[strat_returns < 0]

        avg_win = float(winning_days.mean()) if len(winning_days) > 0 else 0
        avg_loss = float(losing_days.mean()) if len(losing_days) > 0 else 0
        payoff = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        # Trade-level P&L (approximate)
        strat_returns[sig_aligned > 0]
        strat_returns[sig_aligned < 0]
        flat_days = int((sig_aligned == 0).sum())

        col_t1, col_t2 = st.columns(2)

        with col_t1:
            st.markdown("### Trading Activity")
            activity = [
                ("Total Days", total_days),
                ("Position Changes", n_trades),
                (
                    "Trades per Month",
                    f"{n_trades / (total_days / 21):.1f}" if total_days > 0 else "0",
                ),
                ("Long Days", int((sig_aligned > 0).sum())),
                ("Short Days", int((sig_aligned < 0).sum())),
                ("Flat Days", flat_days),
                (
                    "Invested %",
                    f"{(1 - flat_days / total_days):.0%}" if total_days > 0 else "0%",
                ),
            ]
            st.dataframe(
                pd.DataFrame(activity, columns=["Metric", "Value"]).set_index("Metric"),
                use_container_width=True,
            )

        with col_t2:
            st.markdown("### Win/Loss Analysis")
            wl_data = [
                ("Winning Days", len(winning_days)),
                ("Losing Days", len(losing_days)),
                ("Win Rate", f"{perf['win_rate']:.0%}"),
                ("Avg Win", f"{avg_win:.3%}"),
                ("Avg Loss", f"{avg_loss:.3%}"),
                ("Payoff Ratio", f"{payoff:.2f}"),
                ("Best Day", f"{float(strat_returns.max()):.2%}"),
                ("Worst Day", f"{float(strat_returns.min()):.2%}"),
                ("Profit Factor", f"{perf['profit_factor']:.2f}"),
            ]

            # Additional metrics
            try:
                from wraquant.backtest.metrics import expectancy, system_quality_number

                exp = expectancy(strat_returns)
                sqn = system_quality_number(strat_returns)
                wl_data.append(("Expectancy", f"{exp:.4f}"))
                wl_data.append(("SQN", f"{sqn:.2f}"))
            except Exception:
                exp = perf["win_rate"] * avg_win + (1 - perf["win_rate"]) * avg_loss
                wl_data.append(("Expectancy", f"{exp:.4f}"))

            st.dataframe(
                pd.DataFrame(wl_data, columns=["Metric", "Value"]).set_index("Metric"),
                use_container_width=True,
            )

        # Return distribution
        st.markdown("### Strategy Return Distribution")
        try:
            import plotly.graph_objects as go

            from wraquant.dashboard.components.charts import COLORS, dark_layout

            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=strat_returns.values,
                    nbinsx=80,
                    marker_color=COLORS["primary"],
                    opacity=0.7,
                    name="Strategy Returns",
                )
            )
            if benchmark_returns is not None:
                bench_aligned = benchmark_returns.loc[strat_returns.index]
                fig.add_trace(
                    go.Histogram(
                        x=bench_aligned.values,
                        nbinsx=80,
                        marker_color=COLORS["neutral"],
                        opacity=0.4,
                        name="Benchmark Returns",
                    )
                )
            fig.update_layout(
                **dark_layout(
                    title="Return Distribution: Strategy vs Benchmark",
                    xaxis_title="Return",
                    yaxis_title="Count",
                    barmode="overlay",
                    height=400,
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.bar_chart(pd.cut(strat_returns, 50).value_counts().sort_index())
