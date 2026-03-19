"""Experiment Browser page -- browse and compare experiment results.

Connects to the ``wraquant.experiment.lab.Lab`` API to list, load,
and visualise saved experiments.  Supports parameter heatmaps, equity
curve overlays, and stability analysis.
"""

from __future__ import annotations


def render() -> None:
    """Render the Experiment Browser page."""
    import streamlit as st

    st.header("Experiment Browser")

    lab_dir = st.text_input(
        "Experiments directory",
        value="./experiments/",
        help="Path to the directory where Lab experiments are stored.",
    )

    try:
        from wraquant.experiment.lab import Lab

        lab = Lab("browser", storage_dir=lab_dir)
        experiments = lab.list_experiments()

        if experiments is not None and len(experiments) > 0:
            st.dataframe(experiments, use_container_width=True)

            # --- drill-down into a single experiment ---
            selected = st.selectbox("Select experiment", experiments.index)
            if selected is not None:
                results = lab.load(selected)

                # summary table
                summary = results.summary()
                st.subheader("Parameter Grid Results")
                st.dataframe(summary, use_container_width=True)

                # best config + stability side-by-side
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Best Configuration")
                    best = results.best()
                    if isinstance(best, dict):
                        for k, v in best.items():
                            st.write(f"**{k}:** {v}")
                    else:
                        st.write(best)

                with col2:
                    st.subheader("Stability Analysis")
                    try:
                        stability = results.stability()
                        st.dataframe(stability)
                    except Exception:  # noqa: BLE001
                        st.info("Stability analysis unavailable.")

                # parameter heatmap
                params = (
                    list(results.runs[0].params.keys())
                    if results.runs
                    else []
                )
                if len(params) >= 2:
                    st.subheader("Parameter Heatmap")
                    px_col, py_col = st.columns(2)
                    p1 = px_col.selectbox("X axis", params, index=0)
                    p2 = py_col.selectbox(
                        "Y axis", params, index=min(1, len(params) - 1),
                    )
                    metric = st.selectbox(
                        "Metric",
                        ["sharpe", "sortino", "max_drawdown", "total_return"],
                    )
                    try:
                        fig = results.plot_parameter_heatmap(p1, p2, metric=metric)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"Could not render heatmap: {exc}")

                # equity curves
                st.subheader("Top Strategy Equity Curves")
                n_top = st.slider("Show top N", 1, 10, 5)
                try:
                    fig = results.plot_equity_curves(top_n=n_top)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"Could not render equity curves: {exc}")
        else:
            st.info("No experiments found. Run some experiments first!")
            st.code(
                """\
from wraquant.experiment import Lab

lab = Lab("my_research")
exp = lab.create(
    "test_strategy",
    strategy_fn=my_strategy,
    params={"period": [7, 14, 21]},
    data=prices,
)
results = exp.run(cv="walk_forward")
""",
                language="python",
            )
    except ImportError:
        st.error(
            "Could not import wraquant.experiment. "
            "Make sure wraquant is installed."
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error loading experiments: {exc}")
