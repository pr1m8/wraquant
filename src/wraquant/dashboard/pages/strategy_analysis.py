"""Strategy Analysis page -- deep-dive into a single return series.

The user uploads a CSV containing a return column.  The page runs
``wraquant.recipes.analyze`` and displays the results across multiple
tabs: overview metrics, risk breakdown, regime detection, and return
distribution.
"""

from __future__ import annotations


def render() -> None:
    """Render the Strategy Analysis page."""
    import streamlit as st

    st.header("Strategy Analysis")

    upload = st.file_uploader("Upload returns CSV", type=["csv"])

    if upload is not None:
        import numpy as np
        import pandas as pd

        df = pd.read_csv(upload, index_col=0, parse_dates=True)
        returns = df.iloc[:, 0]

        st.write(f"Loaded **{len(returns)}** observations from column **{returns.name}**.")

        # Run wq.analyze()
        from wraquant.recipes import analyze

        report = analyze(returns)

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Overview", "Risk", "Regimes", "Distribution"],
        )

        # ---- Overview ----
        with tab1:
            desc = report["descriptive"]
            risk = report["risk"]

            from wraquant.dashboard.components.metrics import metrics_row

            metrics_row(
                {
                    "Ann. Return": f"{desc.get('mean', 0) * 252:.2%}",
                    "Ann. Vol": f"{desc.get('std', 0) * np.sqrt(252):.2%}",
                    "Sharpe": f"{risk['sharpe']:.2f}",
                    "Max DD": f"{risk['max_drawdown']:.2%}",
                },
            )

            st.subheader("Equity Curve")
            eq = (1 + returns).cumprod()
            st.line_chart(eq)

        # ---- Risk ----
        with tab2:
            st.subheader("Risk Metrics")
            for k, v in report["risk"].items():
                st.write(f"**{k}:** {v:.4f}")

            st.subheader("Descriptive Statistics")
            for k, v in report["descriptive"].items():
                st.write(f"**{k}:** {v}")

            if "volatility" in report:
                st.subheader("GARCH Volatility")
                for k, v in report["volatility"].items():
                    st.write(f"**{k}:** {v:.4f}")

        # ---- Regimes ----
        with tab3:
            if "regime" in report:
                st.subheader("Regime Detection")
                st.write(f"Current regime: **{report['regime']['current']}**")
                st.write(f"Probabilities: {report['regime']['probabilities']}")
                st.write(
                    f"Number of regimes: {report['regime']['n_regimes']}",
                )
            else:
                st.info(
                    "Not enough data for regime detection "
                    "(need 100+ observations).",
                )

        # ---- Distribution ----
        with tab4:
            st.subheader("Return Distribution")
            try:
                import plotly.graph_objects as go

                fig = go.Figure(
                    data=[go.Histogram(x=returns.values, nbinsx=50)],
                )
                fig.update_layout(
                    template="plotly_dark",
                    xaxis_title="Return",
                    yaxis_title="Count",
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(
                    pd.cut(returns, 50).value_counts().sort_index(),
                )

            if "distribution" in report:
                st.subheader("Distribution Fit")
                for k, v in report["distribution"].items():
                    st.write(f"**{k}:** {v}")
    else:
        st.info("Upload a CSV with a returns column to begin analysis.")
        st.code(
            """\
# Example: generate a CSV from wraquant
import wraquant as wq
import pandas as pd

prices = pd.Series(...)  # your price data
returns = prices.pct_change().dropna()
returns.to_csv("returns.csv")
""",
            language="python",
        )
