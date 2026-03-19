"""Portfolio Optimizer page -- interactive portfolio construction.

Upload multi-asset returns, choose an optimization method
(MVO, risk parity, Black-Litterman, HRP), and view optimal weights,
risk decomposition, and diversification metrics.

Delegates to ``wraquant.opt`` for optimization and
``wraquant.risk.portfolio_analytics`` for risk decomposition.
"""

from __future__ import annotations


def render() -> None:
    """Render the Portfolio Optimizer page."""
    import streamlit as st

    st.header("Portfolio Optimizer")

    upload = st.file_uploader(
        "Upload multi-asset returns CSV (columns = assets)",
        type=["csv"],
        key="portfolio_upload",
    )

    if upload is not None:
        import numpy as np
        import pandas as pd

        df = pd.read_csv(upload, index_col=0, parse_dates=True)
        n_assets = df.shape[1]
        st.write(
            f"Loaded **{len(df)}** observations for **{n_assets}** assets: "
            f"{', '.join(df.columns[:10])}"
            + ("..." if n_assets > 10 else ""),
        )

        if n_assets < 2:
            st.warning("Need at least 2 assets for portfolio optimization.")
            return

        # --- sidebar controls ---
        method = st.sidebar.selectbox(
            "Optimization method",
            ["risk_parity", "mean_variance"],
            help="risk_parity = equal risk contribution, "
            "mean_variance = Markowitz MVO (max Sharpe).",
        )
        regime_aware = st.sidebar.checkbox("Regime-aware adjustment", value=False)

        tab_weights, tab_risk, tab_frontier = st.tabs(
            ["Weights", "Risk Decomposition", "Analysis"],
        )

        try:
            from wraquant.recipes import portfolio_construction_pipeline

            result = portfolio_construction_pipeline(
                df,
                method=method,
                regime_aware=regime_aware,
            )

            weights = result["weights"]

            # ---- Weights ----
            with tab_weights:
                st.subheader("Optimal Weights")

                weights_series = pd.Series(weights)
                st.bar_chart(weights_series)

                st.dataframe(
                    pd.DataFrame(
                        {"Asset": weights_series.index, "Weight": weights_series.values},
                    ).set_index("Asset"),
                    use_container_width=True,
                )

                if result["regime_adjusted"]:
                    st.info("Weights were adjusted for the current regime.")

            # ---- Risk Decomposition ----
            with tab_risk:
                st.subheader("Component VaR")
                comp_var = result["component_var"]
                if isinstance(comp_var, (pd.Series, pd.DataFrame)):
                    st.bar_chart(comp_var)
                else:
                    st.write(comp_var)

                from wraquant.dashboard.components.metrics import metrics_row

                metrics_row(
                    {
                        "Diversification Ratio": f"{result['diversification_ratio']:.4f}",
                    },
                )

                st.subheader("Rolling Betas (vs first asset)")
                betas = result["betas"]
                st.dataframe(
                    pd.DataFrame(
                        {"Asset": list(betas.keys()), "Beta": list(betas.values())},
                    ).set_index("Asset"),
                    use_container_width=True,
                )

            # ---- Analysis ----
            with tab_frontier:
                st.subheader("Portfolio Performance")

                # Compute portfolio returns using the weights
                port_returns = df.dot(pd.Series(weights))
                eq = (1 + port_returns).cumprod()
                st.line_chart(eq)

                ann_ret = float((1 + port_returns).prod() ** (252 / len(port_returns)) - 1)
                ann_vol = float(port_returns.std() * np.sqrt(252))
                sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

                metrics_row(
                    {
                        "Ann. Return": f"{ann_ret:.2%}",
                        "Ann. Volatility": f"{ann_vol:.2%}",
                        "Sharpe Ratio": f"{sharpe:.2f}",
                    },
                )

                # Correlation heatmap
                st.subheader("Correlation Matrix")
                corr = df.corr()
                try:
                    from wraquant.viz.interactive import plotly_correlation_heatmap

                    fig = plotly_correlation_heatmap(df)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:  # noqa: BLE001
                    st.dataframe(
                        corr.style.format("{:.3f}"),
                        use_container_width=True,
                    )

        except Exception as exc:  # noqa: BLE001
            st.error(f"Optimization failed: {exc}")
    else:
        st.info(
            "Upload a CSV with multi-asset returns "
            "(columns = assets, rows = daily returns).",
        )
        st.code(
            """\
# Example: build a multi-asset returns CSV
import wraquant as wq
from wraquant.recipes import portfolio_construction_pipeline

result = portfolio_construction_pipeline(returns_df, method="risk_parity")
print(result["weights"])
""",
            language="python",
        )
