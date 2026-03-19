"""Risk Monitor page -- interactive risk dashboard.

Displays VaR, rolling risk metrics, and stress-test results for an
uploaded return series.  Uses ``wraquant.risk`` for computations and
``wraquant.viz.dashboard.risk_dashboard`` for Plotly figures.
"""

from __future__ import annotations


def render() -> None:
    """Render the Risk Monitor page."""
    import streamlit as st

    st.header("Risk Monitor")

    upload = st.file_uploader("Upload returns CSV", type=["csv"], key="risk_upload")

    if upload is not None:
        import numpy as np
        import pandas as pd

        df = pd.read_csv(upload, index_col=0, parse_dates=True)
        returns = df.iloc[:, 0]
        st.write(
            f"Loaded **{len(returns)}** observations from column **{returns.name}**.",
        )

        # --- sidebar parameters ---
        var_alpha = st.sidebar.slider("VaR confidence", 0.90, 0.99, 0.95, 0.01)
        rolling_window = st.sidebar.slider("Rolling window", 21, 252, 63)

        tab_var, tab_rolling, tab_stress = st.tabs(
            ["VaR / CVaR", "Rolling Metrics", "Stress Testing"],
        )

        # ---- VaR / CVaR ----
        with tab_var:
            st.subheader("Value at Risk")
            try:
                from wraquant.risk.var import historical_var, parametric_var

                hist_var = historical_var(returns, alpha=1 - var_alpha)
                param_var = parametric_var(returns, alpha=1 - var_alpha)

                from wraquant.dashboard.components.metrics import metrics_row

                metrics_row(
                    {
                        "Historical VaR": f"{hist_var['var']:.4f}",
                        "Historical CVaR": f"{hist_var['cvar']:.4f}",
                        "Parametric VaR": f"{param_var['var']:.4f}",
                        "Parametric CVaR": f"{param_var['cvar']:.4f}",
                    },
                )
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not compute VaR: {exc}")

            # GARCH VaR pipeline (if enough data)
            if len(returns) >= 200:
                st.subheader("GARCH VaR")
                try:
                    from wraquant.recipes import garch_risk_pipeline

                    garch = garch_risk_pipeline(returns, var_alpha=1 - var_alpha)
                    diag = garch["diagnostics"]
                    metrics_row(
                        {
                            "Persistence": f"{diag['persistence']:.4f}",
                            "Half-life": f"{diag['half_life']:.1f}",
                            "Current Vol": f"{diag['current_vol']:.4f}",
                            "Breach Rate": f"{diag['breach_rate']:.2%}",
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    st.info(f"GARCH VaR unavailable: {exc}")

        # ---- Rolling Metrics ----
        with tab_rolling:
            st.subheader("Rolling Volatility")
            rolling_vol = returns.rolling(rolling_window).std() * np.sqrt(252)
            st.line_chart(rolling_vol.dropna())

            st.subheader("Rolling Sharpe Ratio")
            try:
                from wraquant.stats.descriptive import rolling_sharpe

                rolling_sr = rolling_sharpe(returns, window=rolling_window)
                st.line_chart(rolling_sr.dropna())
            except Exception:  # noqa: BLE001
                # Fallback manual calculation
                roll_mean = returns.rolling(rolling_window).mean() * 252
                roll_std = returns.rolling(rolling_window).std() * np.sqrt(252)
                sr = (roll_mean / roll_std).dropna()
                st.line_chart(sr)

            st.subheader("Drawdown")
            eq = (1 + returns).cumprod()
            dd = eq / eq.cummax() - 1
            st.area_chart(dd)

        # ---- Stress Testing ----
        with tab_stress:
            st.subheader("Stress Scenarios")
            st.markdown(
                "Apply historical stress multipliers to the return series "
                "to see portfolio impact under adverse conditions.",
            )

            shock_pct = st.slider("Shock magnitude (%)", 1, 50, 10)
            shock = shock_pct / 100

            stressed = returns - shock / len(returns)
            eq_base = (1 + returns).cumprod()
            eq_stress = (1 + stressed).cumprod()

            chart_df = pd.DataFrame(
                {"Base": eq_base.values, "Stressed": eq_stress.values},
                index=returns.index,
            )
            st.line_chart(chart_df)

            from wraquant.dashboard.components.metrics import metrics_row

            base_dd = float((eq_base / eq_base.cummax() - 1).min())
            stress_dd = float((eq_stress / eq_stress.cummax() - 1).min())
            metrics_row(
                {
                    "Base Max DD": f"{base_dd:.2%}",
                    "Stressed Max DD": f"{stress_dd:.2%}",
                    "DD Increase": f"{stress_dd - base_dd:.2%}",
                },
            )

        # ---- Plotly risk dashboard (if viz available) ----
        st.markdown("---")
        st.subheader("Full Risk Dashboard")
        try:
            from wraquant.viz.dashboard import risk_dashboard

            fig = risk_dashboard(returns)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.info(f"Interactive risk dashboard unavailable: {exc}")
    else:
        st.info("Upload a CSV with a returns column to monitor risk.")
