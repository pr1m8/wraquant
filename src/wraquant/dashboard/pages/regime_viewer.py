"""Regime Viewer page -- interactive regime detection and analysis.

Lets the user choose a detection method (HMM, GMM, changepoint),
the number of regimes, and displays regime overlay plots with
per-regime statistics.  Delegates to ``wraquant.regimes`` for
computations and ``wraquant.viz.dashboard.regime_dashboard`` for
visualisation.
"""

from __future__ import annotations


def render() -> None:
    """Render the Regime Viewer page."""
    import streamlit as st

    st.header("Regime Viewer")

    upload = st.file_uploader(
        "Upload returns CSV", type=["csv"], key="regime_upload",
    )

    if upload is not None:
        import pandas as pd

        df = pd.read_csv(upload, index_col=0, parse_dates=True)
        returns = df.iloc[:, 0]
        st.write(
            f"Loaded **{len(returns)}** observations from column **{returns.name}**.",
        )

        # --- sidebar controls ---
        method = st.sidebar.selectbox(
            "Detection method",
            ["hmm", "gmm", "changepoint"],
            help="HMM = Gaussian Hidden Markov Model, "
            "GMM = Gaussian Mixture Model, "
            "changepoint = structural break detection.",
        )
        n_regimes = st.sidebar.slider("Number of regimes", 2, 5, 2)

        if len(returns) < 50:
            st.warning("Need at least 50 observations for regime detection.")
            return

        # --- run detection ---
        try:
            from wraquant.regimes.base import detect_regimes

            result = detect_regimes(
                returns.values, method=method, n_regimes=n_regimes,
            )

            # summary metrics
            from wraquant.dashboard.components.metrics import metrics_row

            metrics_row(
                {
                    "Method": method.upper(),
                    "Regimes": str(result.n_regimes),
                    "Current Regime": str(result.current_regime),
                },
            )

            # regime statistics table
            if result.statistics is not None:
                st.subheader("Regime Statistics")
                st.dataframe(result.statistics, use_container_width=True)

            # transition matrix
            st.subheader("Transition Matrix")
            trans_df = pd.DataFrame(
                result.transition_matrix,
                index=[f"From {i}" for i in range(result.n_regimes)],
                columns=[f"To {j}" for j in range(result.n_regimes)],
            )
            st.dataframe(trans_df.style.format("{:.4f}"), use_container_width=True)

            # regime probabilities over time
            st.subheader("Regime Probabilities")
            prob_df = pd.DataFrame(
                result.probabilities,
                index=returns.index[-len(result.probabilities) :],
                columns=[f"Regime {k}" for k in range(result.n_regimes)],
            )
            st.area_chart(prob_df)

            # regime-coloured returns
            st.subheader("Returns by Regime")
            states_series = pd.Series(
                result.states,
                index=returns.index[-len(result.states) :],
                name="regime",
            )
            chart_df = pd.DataFrame(
                {"return": returns.iloc[-len(result.states) :].values},
                index=states_series.index,
            )
            chart_df["regime"] = states_series.values
            st.line_chart(chart_df["return"])

            # interactive dashboard from viz module
            st.markdown("---")
            st.subheader("Full Regime Dashboard")
            try:
                from wraquant.viz.dashboard import regime_dashboard

                fig = regime_dashboard(result)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:  # noqa: BLE001
                st.info(f"Interactive regime dashboard unavailable: {exc}")

        except ImportError as exc:
            st.error(
                f"Missing dependency for regime detection: {exc}. "
                f"Install with: pdm install -G regimes",
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Regime detection failed: {exc}")
    else:
        st.info("Upload a CSV with a returns column for regime detection.")
        st.code(
            """\
# Or use detect_regimes directly:
from wraquant.regimes.base import detect_regimes
result = detect_regimes(returns.values, method="hmm", n_regimes=2)
print(result.current_regime)
""",
            language="python",
        )
