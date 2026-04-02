"""Regime Detection & Analysis page -- HMM, statistics, transitions.

Detects market regimes using Hidden Markov Models, displays regime-
colored price charts, per-regime statistics, transition matrices,
rolling probabilities, regime-conditional drawdowns, and duration
analysis. Delegates to ``wraquant.regimes`` for computations.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_returns(ticker: str, period: str = "5y") -> "pd.Series":
    """Fetch daily returns for a single ticker."""
    import yfinance as yf

    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No data for {ticker}")
    close = data["Close"]
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    return close.pct_change().dropna()


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_prices(ticker: str, period: str = "5y") -> "pd.Series":
    """Fetch close prices."""
    import yfinance as yf

    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No data for {ticker}")
    close = data["Close"]
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    return close


def _synthetic_data(n: int = 1260) -> "tuple[pd.Series, pd.Series]":
    """Generate synthetic price/return data with regime structure."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n)

    # Simulate 2-regime returns
    states = np.zeros(n, dtype=int)
    state = 0
    for i in range(1, n):
        if state == 0:
            state = 1 if rng.random() < 0.02 else 0
        else:
            state = 0 if rng.random() < 0.05 else 1
        states[i] = state

    rets = np.where(
        states == 0, rng.normal(0.0005, 0.01, n), rng.normal(-0.001, 0.025, n)
    )

    prices = 100 * np.exp(np.cumsum(rets))
    return pd.Series(prices, index=idx, name="Close"), pd.Series(
        rets, index=idx, name="returns"
    )


def render() -> None:
    """Render the Regime Detection & Analysis page."""
    import numpy as np
    import pandas as pd

    st.header("Regime Detection & Analysis")

    ticker = st.session_state.get("ticker", "SPY")

    with st.sidebar:
        st.subheader("Regime Settings")
        period = st.selectbox(
            "Lookback", ["2y", "3y", "5y", "10y"], index=2, key="regime_period"
        )
        n_states = st.slider("Number of States", 2, 4, 2, key="regime_n_states")
        method = st.selectbox(
            "Method", ["Gaussian HMM", "GMM", "Label-Based"], key="regime_method"
        )

    # -- Fetch data --------------------------------------------------------

    with st.spinner(f"Loading {ticker} data..."):
        try:
            returns = _fetch_returns(ticker, period=period)
            prices = _fetch_prices(ticker, period=period)
        except Exception:
            st.info(f"Live data unavailable for {ticker} -- using synthetic data.")
            prices, returns = _synthetic_data()

    st.caption(
        f"{len(returns)} observations | {returns.index[0].date()} to {returns.index[-1].date()}"
    )

    # -- Fit regime model --------------------------------------------------

    hmm_result = None
    states = None
    state_probs = None
    transition_matrix = None

    with st.spinner("Fitting regime model..."):
        try:
            if method == "Gaussian HMM":
                from wraquant.regimes.hmm import fit_gaussian_hmm

                hmm_result = fit_gaussian_hmm(returns, n_states=n_states)
            elif method == "GMM":
                from wraquant.regimes.hmm import gaussian_mixture_regimes

                hmm_result = gaussian_mixture_regimes(
                    returns.values, n_components=n_states
                )
            else:
                from wraquant.regimes.labels import label_regimes

                hmm_result = label_regimes(returns)

            if isinstance(hmm_result, dict):
                states = hmm_result.get("states")
                state_probs = hmm_result.get(
                    "state_probs", hmm_result.get("probabilities")
                )
                transition_matrix = hmm_result.get("transition_matrix")
                means = hmm_result.get("means")
                covs = hmm_result.get("covariances", hmm_result.get("variances"))
            elif hasattr(hmm_result, "states"):
                states = hmm_result.states
                state_probs = getattr(
                    hmm_result,
                    "state_probs",
                    getattr(hmm_result, "probabilities", None),
                )
                transition_matrix = getattr(hmm_result, "transition_matrix", None)
                means = getattr(hmm_result, "means", None)
                covs = getattr(hmm_result, "covariances", None)
            else:
                raise ValueError("Unexpected result format")

        except Exception as exc:
            st.warning(
                f"Regime detection failed: {exc}. Using volatility-based fallback."
            )
            # Fallback: simple vol-based regime labeling
            rolling_vol = returns.rolling(63).std()
            vol_median = rolling_vol.median()
            states = np.where(rolling_vol > vol_median * 1.2, 1, 0).astype(int)
            states = states[-len(returns) :]
            state_probs = None
            transition_matrix = None
            means = None
            covs = None

    if states is None:
        st.error("Could not detect regimes.")
        return

    # Align states with returns
    n_obs = min(len(states), len(returns))
    states = np.array(states[-n_obs:])
    returns_aligned = returns.iloc[-n_obs:]
    prices_aligned = prices.iloc[-n_obs:] if len(prices) >= n_obs else prices

    actual_n_states = len(np.unique(states))

    # State colors
    STATE_COLORS = ["#22c55e", "#ef4444", "#f59e0b", "#6366f1"]
    STATE_LABELS = ["Low Vol", "High Vol", "Transition", "Extreme"]

    # -- Current regime indicator ------------------------------------------

    current_regime = int(states[-1])
    current_label = (
        STATE_LABELS[current_regime]
        if current_regime < len(STATE_LABELS)
        else f"State {current_regime}"
    )
    current_color = (
        STATE_COLORS[current_regime]
        if current_regime < len(STATE_COLORS)
        else "#94a3b8"
    )

    if state_probs is not None and len(state_probs) > 0:
        current_prob = (
            state_probs[-1]
            if state_probs.ndim == 1
            else state_probs[-1, current_regime]
        )
    else:
        current_prob = None

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.markdown(
        f'<div style="text-align:center; padding:0.5rem; background:{current_color}22; '
        f'border-radius:8px; border:2px solid {current_color};">'
        f'<p style="font-size:0.8rem; color:#94a3b8; margin:0;">Current Regime</p>'
        f'<p style="font-size:1.5rem; font-weight:700; color:{current_color}; margin:0;">'
        f"{current_label}</p></div>",
        unsafe_allow_html=True,
    )
    if current_prob is not None:
        kpi2.metric("Confidence", f"{float(current_prob):.0%}")
    else:
        kpi2.metric("Confidence", "N/A")
    kpi3.metric("States Detected", str(actual_n_states))
    kpi4.metric("Observations", f"{n_obs:,}")

    st.divider()

    # -- Tabs --------------------------------------------------------------

    tab_chart, tab_stats, tab_trans, tab_dd, tab_dur = st.tabs(
        [
            "Price & Regimes",
            "Regime Statistics",
            "Transition Matrix",
            "Regime Drawdowns",
            "Duration Analysis",
        ],
    )

    # ---- Price chart with regime bands ----
    with tab_chart:
        st.subheader("Price Chart with Regime Overlay")
        try:
            import plotly.graph_objects as go

            from wraquant.dashboard.components.charts import COLORS, dark_layout

            fig = go.Figure()

            # Price line
            fig.add_trace(
                go.Scatter(
                    x=prices_aligned.index,
                    y=prices_aligned.values,
                    mode="lines",
                    name="Price",
                    line={"color": COLORS["text"], "width": 1.5},
                )
            )

            # Regime bands as background shapes
            i = 0
            while i < len(states):
                regime = states[i]
                start_idx = i
                while i < len(states) and states[i] == regime:
                    i += 1
                end_idx = i - 1

                color = (
                    STATE_COLORS[regime] if regime < len(STATE_COLORS) else "#94a3b8"
                )
                fig.add_vrect(
                    x0=returns_aligned.index[start_idx],
                    x1=returns_aligned.index[end_idx],
                    fillcolor=color,
                    opacity=0.15,
                    line_width=0,
                )

            # Legend entries for regimes
            for s in range(actual_n_states):
                color = STATE_COLORS[s] if s < len(STATE_COLORS) else "#94a3b8"
                label = STATE_LABELS[s] if s < len(STATE_LABELS) else f"State {s}"
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker={"color": color, "size": 10, "symbol": "square"},
                        name=label,
                    )
                )

            fig.update_layout(
                **dark_layout(
                    title=f"{ticker} Price with Regime Overlay",
                    yaxis_title="Price",
                    height=500,
                )
            )
            st.plotly_chart(fig, use_container_width=True)

            # Rolling regime probability
            if state_probs is not None and state_probs.ndim == 2:
                st.subheader("Rolling Regime Probabilities")
                prob_fig = go.Figure()
                probs_aligned = state_probs[-n_obs:]
                for s in range(min(actual_n_states, probs_aligned.shape[1])):
                    color = STATE_COLORS[s] if s < len(STATE_COLORS) else "#94a3b8"
                    label = STATE_LABELS[s] if s < len(STATE_LABELS) else f"State {s}"
                    prob_fig.add_trace(
                        go.Scatter(
                            x=returns_aligned.index,
                            y=probs_aligned[:, s],
                            mode="lines",
                            name=label,
                            line={"color": color, "width": 1.5},
                            stackgroup="one" if actual_n_states <= 3 else None,
                        )
                    )
                prob_fig.update_layout(
                    **dark_layout(
                        title="Regime Probabilities Over Time",
                        yaxis_title="Probability",
                        height=350,
                    )
                )
                prob_fig.update_yaxes(range=[0, 1])
                st.plotly_chart(prob_fig, use_container_width=True)

        except ImportError:
            st.line_chart(prices_aligned)

    # ---- Regime Statistics ----
    with tab_stats:
        st.subheader("Per-Regime Statistics")

        try:
            from wraquant.regimes.hmm import regime_statistics

            reg_stats = regime_statistics(returns_aligned.values, states)
        except Exception:
            reg_stats = None

        rows = []
        for s in range(actual_n_states):
            mask = states == s
            r_s = returns_aligned.values[mask]
            label = STATE_LABELS[s] if s < len(STATE_LABELS) else f"State {s}"

            if len(r_s) > 1:
                mean_r = float(np.mean(r_s)) * 252
                vol_r = float(np.std(r_s)) * np.sqrt(252)
                sharpe_r = mean_r / vol_r if vol_r > 0 else 0
                min_r = float(np.min(r_s))
                max_r = float(np.max(r_s))
                n_days = int(np.sum(mask))
                pct = n_days / len(states)
            else:
                mean_r = vol_r = sharpe_r = min_r = max_r = 0.0
                n_days = int(np.sum(mask))
                pct = n_days / len(states) if len(states) > 0 else 0

            if isinstance(reg_stats, dict) and reg_stats:
                stats_s = reg_stats.get(s, {})
                mean_r = stats_s.get("annualized_mean", mean_r)
                vol_r = stats_s.get("annualized_vol", vol_r)
                sharpe_r = stats_s.get("sharpe", sharpe_r)

            rows.append(
                {
                    "Regime": label,
                    "Ann. Return": f"{mean_r:.1%}",
                    "Ann. Vol": f"{vol_r:.1%}",
                    "Sharpe": f"{sharpe_r:.2f}",
                    "Min Daily": f"{min_r:.2%}",
                    "Max Daily": f"{max_r:.2%}",
                    "Days": n_days,
                    "% Time": f"{pct:.0%}",
                }
            )

        st.dataframe(pd.DataFrame(rows).set_index("Regime"), use_container_width=True)

        # HMM emission parameters
        if means is not None:
            st.markdown("### Emission Parameters")
            em_cols = st.columns(actual_n_states)
            for s in range(actual_n_states):
                label = STATE_LABELS[s] if s < len(STATE_LABELS) else f"State {s}"
                with em_cols[s]:
                    st.markdown(f"**{label}**")
                    m = (
                        means[s]
                        if isinstance(means, (list, np.ndarray)) and s < len(means)
                        else "N/A"
                    )
                    st.metric(
                        "Mean",
                        (
                            f"{float(m) * 252:.1%}"
                            if isinstance(m, (int, float, np.floating))
                            else str(m)
                        ),
                    )
                    if covs is not None and s < len(covs):
                        c = covs[s]
                        vol = (
                            float(np.sqrt(c)) * np.sqrt(252)
                            if isinstance(c, (int, float, np.floating))
                            else 0
                        )
                        st.metric("Vol", f"{vol:.1%}")

    # ---- Transition Matrix ----
    with tab_trans:
        st.subheader("Transition Probability Matrix")

        if transition_matrix is not None:
            trans = np.array(transition_matrix)
            labels = [
                STATE_LABELS[s] if s < len(STATE_LABELS) else f"State {s}"
                for s in range(trans.shape[0])
            ]

            try:
                import plotly.graph_objects as go

                from wraquant.dashboard.components.charts import dark_layout

                fig = go.Figure(
                    data=go.Heatmap(
                        z=trans,
                        x=[f"To: {l}" for l in labels],
                        y=[f"From: {l}" for l in labels],
                        colorscale="Blues",
                        text=trans.round(4),
                        texttemplate="%{text:.4f}",
                        textfont={"size": 13},
                        zmin=0,
                        zmax=1,
                    )
                )
                fig.update_layout(
                    **dark_layout(
                        title="Regime Transition Probabilities",
                        height=400,
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                trans_df = pd.DataFrame(
                    trans,
                    index=[f"From: {l}" for l in labels],
                    columns=[f"To: {l}" for l in labels],
                )
                st.dataframe(trans_df.style.format("{:.4f}"), use_container_width=True)

            # Expected durations
            st.markdown("### Expected Duration per Regime")
            dur_cols = st.columns(actual_n_states)
            for s in range(min(actual_n_states, trans.shape[0])):
                label = STATE_LABELS[s] if s < len(STATE_LABELS) else f"State {s}"
                self_prob = trans[s, s] if s < trans.shape[0] else 0
                exp_dur = 1 / (1 - self_prob) if self_prob < 1 else float("inf")
                with dur_cols[s]:
                    color = STATE_COLORS[s] if s < len(STATE_COLORS) else "#94a3b8"
                    st.metric(label, f"{exp_dur:.0f} days")
        else:
            # Compute empirical transition matrix
            st.info(
                "Transition matrix not available from model. Computing empirical transitions."
            )
            trans = np.zeros((actual_n_states, actual_n_states))
            for i in range(len(states) - 1):
                trans[states[i], states[i + 1]] += 1
            row_sums = trans.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            trans = trans / row_sums

            labels = [
                STATE_LABELS[s] if s < len(STATE_LABELS) else f"State {s}"
                for s in range(actual_n_states)
            ]
            trans_df = pd.DataFrame(
                trans,
                index=[f"From: {l}" for l in labels],
                columns=[f"To: {l}" for l in labels],
            )
            st.dataframe(trans_df.style.format("{:.4f}"), use_container_width=True)

    # ---- Regime-Conditional Drawdowns ----
    with tab_dd:
        st.subheader("Regime-Conditional Drawdown Analysis")

        cum = (1 + returns_aligned).cumprod()
        dd = cum / cum.cummax() - 1

        try:
            import plotly.graph_objects as go

            from wraquant.dashboard.components.charts import COLORS, dark_layout

            fig = go.Figure()

            for s in range(actual_n_states):
                mask = states == s
                label = STATE_LABELS[s] if s < len(STATE_LABELS) else f"State {s}"
                color = STATE_COLORS[s] if s < len(STATE_COLORS) else "#94a3b8"

                dd_regime = dd.copy()
                dd_regime[~mask] = np.nan

                fig.add_trace(
                    go.Scatter(
                        x=dd_regime.index,
                        y=dd_regime.values,
                        mode="lines",
                        name=f"DD in {label}",
                        line={"color": color, "width": 1.5},
                        connectgaps=False,
                    )
                )

            fig.update_layout(
                **dark_layout(
                    title="Drawdown by Regime",
                    yaxis_title="Drawdown",
                    height=450,
                )
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            st.area_chart(dd)

        # Max drawdown per regime
        st.markdown("### Max Drawdown per Regime")
        dd_cols = st.columns(actual_n_states)
        for s in range(actual_n_states):
            mask = states == s
            label = STATE_LABELS[s] if s < len(STATE_LABELS) else f"State {s}"
            dd_vals = dd.values[mask]
            max_dd_regime = float(np.min(dd_vals)) if len(dd_vals) > 0 else 0.0
            with dd_cols[s]:
                st.metric(label, f"{max_dd_regime:.1%}", delta_color="inverse")

    # ---- Duration Analysis ----
    with tab_dur:
        st.subheader("Regime Duration Histogram")

        # Compute regime durations
        durations = {s: [] for s in range(actual_n_states)}
        current_state = states[0]
        current_dur = 1
        for i in range(1, len(states)):
            if states[i] == current_state:
                current_dur += 1
            else:
                durations[current_state].append(current_dur)
                current_state = states[i]
                current_dur = 1
        durations[current_state].append(current_dur)

        try:
            import plotly.graph_objects as go

            from wraquant.dashboard.components.charts import dark_layout

            fig = go.Figure()
            for s in range(actual_n_states):
                if durations[s]:
                    label = STATE_LABELS[s] if s < len(STATE_LABELS) else f"State {s}"
                    color = STATE_COLORS[s] if s < len(STATE_COLORS) else "#94a3b8"
                    fig.add_trace(
                        go.Histogram(
                            x=durations[s],
                            name=label,
                            marker_color=color,
                            opacity=0.7,
                            nbinsx=30,
                        )
                    )

            fig.update_layout(
                **dark_layout(
                    title="Distribution of Regime Durations",
                    xaxis_title="Duration (days)",
                    yaxis_title="Count",
                    barmode="overlay",
                    height=400,
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            for s in range(actual_n_states):
                if durations[s]:
                    label = STATE_LABELS[s] if s < len(STATE_LABELS) else f"State {s}"
                    st.write(
                        f"**{label}**: mean={np.mean(durations[s]):.0f} days, "
                        f"max={np.max(durations[s])} days"
                    )

        # Duration statistics table
        st.markdown("### Duration Statistics")
        dur_rows = []
        for s in range(actual_n_states):
            d = durations[s]
            label = STATE_LABELS[s] if s < len(STATE_LABELS) else f"State {s}"
            if d:
                dur_rows.append(
                    {
                        "Regime": label,
                        "Episodes": len(d),
                        "Mean Duration": f"{np.mean(d):.0f} days",
                        "Median Duration": f"{np.median(d):.0f} days",
                        "Max Duration": f"{np.max(d)} days",
                        "Min Duration": f"{np.min(d)} days",
                        "Std Duration": f"{np.std(d):.1f} days",
                    }
                )
        if dur_rows:
            st.dataframe(
                pd.DataFrame(dur_rows).set_index("Regime"), use_container_width=True
            )
