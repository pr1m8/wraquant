"""Quantitative Research Lab -- network analysis, information theory, spectral, Monte Carlo.

Advanced quantitative tools using wraquant.math, wraquant.stats, and
numpy/scipy for deeper research-grade analysis.
"""

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


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
    # Correlated returns
    k = len(tickers)
    A = rng.standard_normal((k, k))
    cov = (A @ A.T) / k * 0.0002
    import numpy as np

    np.fill_diagonal(cov, np.abs(np.diag(cov)) + 0.0001)
    means = rng.uniform(0.0002, 0.0008, k)
    rets = rng.multivariate_normal(means, cov, size=n)
    return pd.DataFrame(rets, index=idx, columns=tickers)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_single_returns(ticker: str, days: int = 730):
    """Fetch single-asset returns."""
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

    rng = np.random.default_rng(42)
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=504)
    return pd.Series(rng.normal(0.0004, 0.015, 504), index=idx, name="returns")


def render() -> None:
    """Render the Quantitative Research Lab page."""
    import numpy as np
    import pandas as pd

    from wraquant.dashboard.components.charts import COLORS, SERIES_COLORS, dark_layout

    ticker = st.session_state.get("ticker", "AAPL")
    st.markdown("# Quantitative Research Lab")

    tab_network, tab_info, tab_spectral, tab_mc = st.tabs([
        "Network Analysis",
        "Information Theory",
        "Spectral Analysis",
        "Monte Carlo Simulation",
    ])

    # ---- Network Analysis Tab ----
    with tab_network:
        st.subheader("Correlation Network")
        net_tickers_input = st.text_input(
            "Tickers (comma-separated, 5-20 recommended)",
            value="AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM, GS, XOM, PFE, JNJ",
            key="ql_net_tickers",
        )
        corr_threshold = st.slider(
            "Correlation threshold for edges", 0.1, 0.9, 0.3, 0.05,
            key="ql_net_thresh",
        )

        net_tickers = [t.strip().upper() for t in net_tickers_input.split(",") if t.strip()]
        if len(net_tickers) < 3:
            st.warning("Enter at least 3 tickers.")
        else:
            with st.spinner("Fetching data for network..."):
                multi_ret = _fetch_multi_returns(net_tickers)

            if multi_ret is not None and not multi_ret.empty:
                corr = multi_ret.corr()

                # Try wraquant correlation_network
                try:
                    from wraquant.math.network import (
                        centrality_measures,
                        correlation_network,
                        minimum_spanning_tree,
                    )

                    G = correlation_network(multi_ret, threshold=corr_threshold)
                    centrality = centrality_measures(G)
                    mst = minimum_spanning_tree(corr)
                except Exception:
                    G = None
                    centrality = None
                    mst = None

                try:
                    import plotly.graph_objects as go

                    # Build network visualization from correlation matrix
                    n_nodes = len(corr.columns)
                    tickers_list = corr.columns.tolist()

                    # Layout: circular
                    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
                    x_pos = np.cos(angles)
                    y_pos = np.sin(angles)

                    # Edges
                    edge_x, edge_y = [], []
                    edge_weights = []
                    for i in range(n_nodes):
                        for j in range(i + 1, n_nodes):
                            if abs(corr.iloc[i, j]) > corr_threshold:
                                edge_x.extend([x_pos[i], x_pos[j], None])
                                edge_y.extend([y_pos[i], y_pos[j], None])
                                edge_weights.append(corr.iloc[i, j])

                    fig = go.Figure()

                    # Draw edges
                    if edge_x:
                        fig.add_trace(
                            go.Scatter(
                                x=edge_x, y=edge_y, mode="lines",
                                line={"color": COLORS["neutral"], "width": 0.8},
                                opacity=0.4, hoverinfo="none", showlegend=False,
                            )
                        )

                    # Draw nodes
                    node_colors = []
                    for t in tickers_list:
                        if t == ticker:
                            node_colors.append(COLORS["danger"])
                        else:
                            node_colors.append(COLORS["primary"])

                    node_sizes = [20] * n_nodes
                    if centrality is not None:
                        if isinstance(centrality, dict):
                            degree = centrality.get("degree", centrality.get("degree_centrality", {}))
                            if isinstance(degree, dict):
                                for i, t in enumerate(tickers_list):
                                    node_sizes[i] = 10 + 30 * degree.get(t, 0.5)

                    fig.add_trace(
                        go.Scatter(
                            x=x_pos, y=y_pos, mode="markers+text",
                            marker={"color": node_colors, "size": node_sizes, "line": {"width": 1, "color": COLORS["text"]}},
                            text=tickers_list, textposition="top center",
                            textfont={"size": 11, "color": COLORS["text"]},
                            hovertemplate="%{text}<extra></extra>",
                            showlegend=False,
                        )
                    )

                    fig.update_layout(
                        **dark_layout(
                            title=f"Correlation Network (threshold={corr_threshold})",
                            height=550,
                            showlegend=False,
                        )
                    )
                    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
                    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
                    st.plotly_chart(fig, use_container_width=True)

                    # Centrality ranking
                    st.subheader("Centrality Ranking")
                    if centrality is not None and isinstance(centrality, dict):
                        cent_data = []
                        degree_c = centrality.get("degree", centrality.get("degree_centrality", {}))
                        between_c = centrality.get("betweenness", centrality.get("betweenness_centrality", {}))
                        close_c = centrality.get("closeness", centrality.get("closeness_centrality", {}))
                        eigen_c = centrality.get("eigenvector", centrality.get("eigenvector_centrality", {}))

                        for t in tickers_list:
                            row = {"Ticker": t}
                            if isinstance(degree_c, dict):
                                row["Degree"] = f"{degree_c.get(t, 0):.4f}"
                            if isinstance(between_c, dict):
                                row["Betweenness"] = f"{between_c.get(t, 0):.4f}"
                            if isinstance(close_c, dict):
                                row["Closeness"] = f"{close_c.get(t, 0):.4f}"
                            if isinstance(eigen_c, dict):
                                row["Eigenvector"] = f"{eigen_c.get(t, 0):.4f}"
                            cent_data.append(row)

                        if cent_data:
                            st.dataframe(
                                pd.DataFrame(cent_data).set_index("Ticker"),
                                use_container_width=True,
                            )
                    else:
                        # Fallback: degree based on correlation count
                        degree_count = {}
                        for t in tickers_list:
                            idx_i = tickers_list.index(t)
                            degree_count[t] = sum(
                                1 for j in range(n_nodes)
                                if j != idx_i and abs(corr.iloc[idx_i, j]) > corr_threshold
                            )
                        deg_df = pd.DataFrame([
                            {"Ticker": t, "Degree": d, "Normalized": f"{d / max(n_nodes-1, 1):.3f}"}
                            for t, d in sorted(degree_count.items(), key=lambda x: -x[1])
                        ])
                        st.dataframe(deg_df, hide_index=True, use_container_width=True)

                    # MST visualization
                    st.subheader("Minimum Spanning Tree")
                    if mst is not None:
                        st.info("MST computed. Use the network chart above for visualization context.")
                        if hasattr(mst, "edges"):
                            edges = list(mst.edges(data=True))
                            mst_data = [
                                {"From": u, "To": v, "Weight": f"{d.get('weight', 0):.4f}"}
                                for u, v, d in edges[:20]
                            ]
                            if mst_data:
                                st.dataframe(pd.DataFrame(mst_data), hide_index=True, use_container_width=True)

                except ImportError:
                    st.dataframe(corr.style.format("{:.3f}"), use_container_width=True)

    # ---- Information Theory Tab ----
    with tab_info:
        st.subheader("Information Theory")

        it_tickers_input = st.text_input(
            "Tickers (comma-separated)",
            value=f"{ticker}, MSFT, GOOGL, SPY, QQQ",
            key="ql_it_tickers",
        )
        it_tickers = [t.strip().upper() for t in it_tickers_input.split(",") if t.strip()]

        if len(it_tickers) < 2:
            st.warning("Enter at least 2 tickers.")
        else:
            with st.spinner("Computing information metrics..."):
                it_ret = _fetch_multi_returns(it_tickers)

            if it_ret is not None and not it_ret.empty:
                # Shannon entropy
                st.subheader("Shannon Entropy")
                entropy_vals = {}
                for col in it_ret.columns:
                    try:
                        hist, _ = np.histogram(it_ret[col].dropna(), bins=50, density=True)
                        hist = hist[hist > 0]
                        dx = (it_ret[col].max() - it_ret[col].min()) / 50
                        entropy_vals[col] = float(-np.sum(hist * dx * np.log(hist * dx + 1e-12)))
                    except Exception:
                        pass

                if entropy_vals:
                    try:
                        import plotly.graph_objects as go

                        ticks = list(entropy_vals.keys())
                        vals = list(entropy_vals.values())
                        fig = go.Figure()
                        fig.add_trace(
                            go.Bar(
                                x=ticks, y=vals,
                                marker_color=COLORS["primary"],
                                text=[f"{v:.3f}" for v in vals],
                                textposition="auto",
                            )
                        )
                        fig.update_layout(
                            **dark_layout(
                                title="Shannon Entropy by Asset",
                                yaxis_title="Entropy (nats)", height=350,
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        st.dataframe(
                            pd.DataFrame([entropy_vals]).T.rename(columns={0: "Entropy"}),
                            use_container_width=True,
                        )

                # Mutual information heatmap
                st.subheader("Mutual Information")
                try:
                    from wraquant.stats.correlation import mutual_information

                    n_assets = len(it_ret.columns)
                    mi_matrix = np.zeros((n_assets, n_assets))
                    cols = it_ret.columns.tolist()
                    for i in range(n_assets):
                        for j in range(i, n_assets):
                            if i == j:
                                mi_matrix[i, j] = entropy_vals.get(cols[i], 0)
                            else:
                                try:
                                    mi = mutual_information(it_ret.iloc[:, i], it_ret.iloc[:, j])
                                    mi_val = float(mi.get("mutual_information", mi) if isinstance(mi, dict) else mi)
                                    mi_matrix[i, j] = mi_val
                                    mi_matrix[j, i] = mi_val
                                except Exception:
                                    pass

                    try:
                        import plotly.graph_objects as go

                        fig = go.Figure(
                            data=go.Heatmap(
                                z=mi_matrix, x=cols, y=cols,
                                colorscale="Viridis",
                                text=np.round(mi_matrix, 4),
                                texttemplate="%{text}",
                                textfont={"size": 10},
                            )
                        )
                        fig.update_layout(
                            **dark_layout(title="Mutual Information Heatmap", height=500)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        st.dataframe(
                            pd.DataFrame(mi_matrix, index=cols, columns=cols).style.format("{:.4f}"),
                            use_container_width=True,
                        )
                except Exception as e:
                    st.info(f"Mutual information computation unavailable: {e}")

                    # Fallback: histogram-based MI
                    try:
                        from sklearn.metrics import mutual_info_score

                        n_assets = len(it_ret.columns)
                        cols = it_ret.columns.tolist()
                        mi_matrix = np.zeros((n_assets, n_assets))
                        for i in range(n_assets):
                            for j in range(i, n_assets):
                                x_d = pd.qcut(it_ret.iloc[:, i], 20, labels=False, duplicates="drop")
                                y_d = pd.qcut(it_ret.iloc[:, j], 20, labels=False, duplicates="drop")
                                mi_val = mutual_info_score(x_d, y_d)
                                mi_matrix[i, j] = mi_val
                                mi_matrix[j, i] = mi_val
                        st.dataframe(
                            pd.DataFrame(mi_matrix, index=cols, columns=cols).style.format("{:.4f}"),
                            use_container_width=True,
                        )
                    except Exception:
                        st.info("sklearn required for fallback MI computation.")

    # ---- Spectral Analysis Tab ----
    with tab_spectral:
        st.subheader("Spectral Analysis (FFT)")

        with st.spinner(f"Loading {ticker} returns..."):
            returns = _fetch_single_returns(ticker)

        if returns is not None and len(returns) > 60:
            # FFT
            n = len(returns)
            fft_vals = np.fft.rfft(returns.values - returns.mean())
            power = np.abs(fft_vals) ** 2
            freqs = np.fft.rfftfreq(n, d=1.0)  # d=1 trading day

            # Convert to periods
            periods = np.zeros_like(freqs)
            periods[1:] = 1.0 / freqs[1:]
            periods[0] = np.inf

            try:
                import plotly.graph_objects as go

                # Power spectrum vs frequency
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=freqs[1:n//2], y=power[1:n//2],
                        mode="lines",
                        line={"color": COLORS["primary"], "width": 1.5},
                        name="Power",
                    )
                )
                fig.update_layout(
                    **dark_layout(
                        title="FFT Power Spectrum",
                        xaxis_title="Frequency (cycles/day)",
                        yaxis_title="Power",
                        height=400,
                    )
                )
                fig.update_yaxes(type="log")
                st.plotly_chart(fig, use_container_width=True)

                # Power vs period
                valid = (periods > 2) & (periods < 260)
                fig2 = go.Figure()
                fig2.add_trace(
                    go.Scatter(
                        x=periods[valid], y=power[valid],
                        mode="lines",
                        line={"color": COLORS["accent2"], "width": 1.5},
                        name="Power",
                    )
                )
                fig2.update_layout(
                    **dark_layout(
                        title="Power Spectrum vs Period",
                        xaxis_title="Period (trading days)",
                        yaxis_title="Power",
                        height=400,
                    )
                )
                fig2.update_yaxes(type="log")
                st.plotly_chart(fig2, use_container_width=True)
            except ImportError:
                pass

            # Dominant frequencies table
            st.subheader("Dominant Frequencies / Periods")
            # Skip DC component
            top_k = 10
            power_no_dc = power[1:n//2].copy()
            freqs_no_dc = freqs[1:n//2].copy()
            top_indices = np.argsort(power_no_dc)[-top_k:][::-1]

            dom_data = []
            for idx in top_indices:
                f = freqs_no_dc[idx]
                p = 1.0 / f if f > 0 else np.inf
                dom_data.append({
                    "Frequency": f"{f:.6f}",
                    "Period (days)": f"{p:.1f}",
                    "Period (weeks)": f"{p/5:.1f}",
                    "Power": f"{power_no_dc[idx]:.2e}",
                })
            st.dataframe(pd.DataFrame(dom_data), hide_index=True, use_container_width=True)

            # Spectral features from wraquant
            try:
                from wraquant.ts.features import spectral_features

                sf = spectral_features(returns)
                if isinstance(sf, dict):
                    st.subheader("Spectral Features")
                    sf_df = pd.DataFrame([{k: f"{v:.6f}" if isinstance(v, float) else str(v) for k, v in sf.items()}])
                    st.dataframe(sf_df, use_container_width=True)
            except Exception:
                pass

        else:
            st.warning("Insufficient data for spectral analysis.")

    # ---- Monte Carlo Tab ----
    with tab_mc:
        st.subheader("Monte Carlo Path Simulation")

        with st.spinner(f"Loading {ticker} returns..."):
            returns_mc = _fetch_single_returns(ticker)

        if returns_mc is None or len(returns_mc) < 60:
            st.warning("Insufficient data for Monte Carlo simulation.")
        else:
            mc_col1, mc_col2 = st.columns(2)
            with mc_col1:
                model = st.selectbox(
                    "Model",
                    ["Geometric Brownian Motion", "Heston Stochastic Vol"],
                    key="ql_mc_model",
                )
                n_paths = st.slider("Number of Paths", 100, 5000, 1000, 100, key="ql_mc_paths")
            with mc_col2:
                horizon_days = st.slider("Horizon (days)", 21, 504, 252, key="ql_mc_horizon")
                s0 = st.number_input(
                    "Initial Price",
                    value=100.0, min_value=1.0,
                    key="ql_mc_s0",
                )

            mu = float(returns_mc.mean() * 252)
            sigma = float(returns_mc.std() * np.sqrt(252))
            dt = 1.0 / 252

            st.caption(f"Estimated params: mu={mu:.4f}, sigma={sigma:.4f}")

            with st.spinner("Running simulation..."):
                rng = np.random.default_rng(42)

                if model == "Geometric Brownian Motion":
                    Z = rng.standard_normal((n_paths, horizon_days))
                    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
                    paths = s0 * np.exp(np.cumsum(log_returns, axis=1))
                    paths = np.column_stack([np.full(n_paths, s0), paths])

                else:  # Heston
                    kappa = st.slider("Kappa (mean reversion)", 0.5, 10.0, 2.0, 0.5, key="ql_hest_k")
                    theta = st.slider("Theta (long-run var)", 0.01, 0.5, sigma ** 2, 0.01, key="ql_hest_t")
                    xi = st.slider("Xi (vol of vol)", 0.1, 1.5, 0.5, 0.1, key="ql_hest_xi")
                    rho = st.slider("Rho (correlation)", -0.99, 0.0, -0.7, 0.01, key="ql_hest_rho")

                    paths = np.zeros((n_paths, horizon_days + 1))
                    paths[:, 0] = s0
                    v = np.full(n_paths, sigma ** 2)

                    for t_step in range(horizon_days):
                        z1 = rng.standard_normal(n_paths)
                        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n_paths)
                        v_pos = np.maximum(v, 0)
                        paths[:, t_step + 1] = paths[:, t_step] * np.exp(
                            (mu - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * z1
                        )
                        v = v + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos * dt) * z2
                        v = np.maximum(v, 0)

            try:
                import plotly.graph_objects as go

                fig = go.Figure()

                # Plot a subset of paths
                n_show = min(100, n_paths)
                for i in range(n_show):
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(horizon_days + 1)),
                            y=paths[i],
                            mode="lines",
                            line={"width": 0.5, "color": COLORS["primary"]},
                            opacity=0.15,
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

                # Percentiles
                p5 = np.percentile(paths, 5, axis=0)
                p25 = np.percentile(paths, 25, axis=0)
                p50 = np.percentile(paths, 50, axis=0)
                p75 = np.percentile(paths, 75, axis=0)
                p95 = np.percentile(paths, 95, axis=0)

                x_axis = list(range(horizon_days + 1))
                fig.add_trace(
                    go.Scatter(
                        x=x_axis, y=p50, mode="lines",
                        line={"color": COLORS["accent2"], "width": 2.5},
                        name="Median",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_axis, y=p5, mode="lines",
                        line={"color": COLORS["danger"], "width": 1.5, "dash": "dash"},
                        name="5th percentile",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_axis, y=p95, mode="lines",
                        line={"color": COLORS["success"], "width": 1.5, "dash": "dash"},
                        name="95th percentile",
                    )
                )

                fig.update_layout(
                    **dark_layout(
                        title=f"{model}: {n_paths} Simulated Paths",
                        xaxis_title="Trading Days",
                        yaxis_title="Price",
                        height=500,
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.line_chart(pd.DataFrame(paths[:20].T))

            # Terminal distribution
            st.subheader("Terminal Price Distribution")
            terminal = paths[:, -1]

            try:
                import plotly.graph_objects as go

                fig = go.Figure()
                fig.add_trace(
                    go.Histogram(
                        x=terminal, nbinsx=80,
                        marker_color=COLORS["primary"], opacity=0.7,
                        name="Terminal Price",
                    )
                )
                fig.add_vline(
                    x=s0, line_dash="dash", line_color=COLORS["warning"],
                    annotation_text=f"S0={s0:.0f}",
                )
                fig.update_layout(
                    **dark_layout(
                        title=f"Terminal Price Distribution ({horizon_days}d)",
                        xaxis_title="Price", yaxis_title="Count",
                        height=400,
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(pd.Series(terminal).value_counts(bins=50).sort_index())

            # Summary stats
            st.subheader("Simulation Summary")
            mc_stats = {
                "Mean Terminal": f"${float(terminal.mean()):.2f}",
                "Median Terminal": f"${float(np.median(terminal)):.2f}",
                "5th Percentile": f"${float(np.percentile(terminal, 5)):.2f}",
                "95th Percentile": f"${float(np.percentile(terminal, 95)):.2f}",
                "Prob(gain)": f"{float((terminal > s0).mean()):.1%}",
                "Expected Return": f"{float(terminal.mean() / s0 - 1):.2%}",
                "Ann. Vol (realized)": f"{sigma:.2%}",
            }
            mc_cols = st.columns(len(mc_stats))
            for c, (k, v) in zip(mc_cols, mc_stats.items(), strict=False):
                c.metric(k, v)
