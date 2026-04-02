"""Regime detection MCP tools (deep module-specific).

Tools: regime_statistics, regime_transition, select_n_states,
rolling_regime_probability, fit_gaussian_hmm, fit_ms_autoregression,
gaussian_mixture_regimes, regime_conditional_moments, regime_scoring,
regime_labels, kalman_filter, kalman_regression.

Note: The basic detect_regimes tool lives in server.py (tier-2).
These tools provide deeper regime analysis capabilities.
"""

from __future__ import annotations

from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_regimes_tools(mcp, ctx: AnalysisContext) -> None:
    """Register regime-detection tools on the MCP server."""

    @mcp.tool()
    def regime_statistics(
        dataset: str,
        column: str = "returns",
        n_regimes: int = 2,
    ) -> dict[str, Any]:
        """Compute per-regime descriptive statistics.

        Fits an HMM then computes mean, volatility, Sharpe, Sortino,
        drawdown, VaR/CVaR, skewness, and kurtosis for each regime.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
            n_regimes: Number of regimes to fit.
        """
        try:
            from wraquant.regimes.hmm import fit_hmm
            from wraquant.regimes.hmm import regime_statistics as _regime_stats

            df = ctx.get_dataset(dataset)
            returns = df[column].dropna()

            model = fit_hmm(returns, n_states=n_regimes)
            states = model.predict(returns.values.reshape(-1, 1))

            stats_df = _regime_stats(returns, states)

            stored = ctx.store_dataset(
                f"regime_stats_{dataset}",
                stats_df,
                source_op="regime_statistics",
                parent=dataset,
            )

            return _sanitize_for_json(
                {
                    "tool": "regime_statistics",
                    "dataset": dataset,
                    "n_regimes": n_regimes,
                    "statistics": stats_df.to_dict(orient="index"),
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "regime_statistics"}

    @mcp.tool()
    def regime_transition(
        dataset: str,
        column: str = "returns",
        n_regimes: int = 2,
    ) -> dict[str, Any]:
        """Analyze regime transition dynamics.

        Returns the empirical and model transition matrices, steady-state
        distribution, average regime durations, and regime visit counts.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
            n_regimes: Number of regimes to fit.
        """
        try:
            from wraquant.regimes.hmm import fit_hmm, regime_transition_analysis

            df = ctx.get_dataset(dataset)
            returns = df[column].dropna()

            model = fit_hmm(returns, n_states=n_regimes)
            states = model.predict(returns.values.reshape(-1, 1))
            transmat = model.transmat_

            result = regime_transition_analysis(states, transition_matrix=transmat)

            return _sanitize_for_json(
                {
                    "tool": "regime_transition",
                    "dataset": dataset,
                    "n_regimes": n_regimes,
                    "transition_matrix": result["transition_matrix"],
                    "empirical_transition_matrix": result[
                        "empirical_transition_matrix"
                    ],
                    "steady_state": result["steady_state"],
                    "avg_duration": result["avg_duration"],
                    "regime_counts": result["regime_counts"],
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "regime_transition"}

    @mcp.tool()
    def select_n_states(
        dataset: str,
        column: str = "returns",
        max_states: int = 5,
    ) -> dict[str, Any]:
        """Select optimal number of HMM states using BIC.

        Fits HMMs with 2..max_states and returns BIC scores,
        recommended state count, and per-state model summaries.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
            max_states: Maximum number of states to evaluate.
        """
        try:
            from wraquant.regimes.hmm import select_n_states as _select

            df = ctx.get_dataset(dataset)
            returns = df[column].dropna()

            result = _select(returns, max_states=max_states)

            return _sanitize_for_json(
                {
                    "tool": "select_n_states",
                    "dataset": dataset,
                    "max_states": max_states,
                    "result": result,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "select_n_states"}

    @mcp.tool()
    def rolling_regime_probability(
        dataset: str,
        column: str = "returns",
        n_regimes: int = 2,
        window: int = 120,
    ) -> dict[str, Any]:
        """Compute time-varying regime probabilities using rolling HMM.

        Fits an HMM at each time step using a rolling window to
        produce regime probability time series for real-time monitoring.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
            n_regimes: Number of regimes.
            window: Rolling window size in observations.
        """
        try:
            from wraquant.regimes.hmm import rolling_regime_probability as _rolling

            df = ctx.get_dataset(dataset)
            returns = df[column].dropna()

            probs = _rolling(returns, n_states=n_regimes, window=window)

            stored = ctx.store_dataset(
                f"regime_probs_{dataset}",
                probs,
                source_op="rolling_regime_probability",
                parent=dataset,
            )

            return _sanitize_for_json(
                {
                    "tool": "rolling_regime_probability",
                    "dataset": dataset,
                    "n_regimes": n_regimes,
                    "window": window,
                    "latest_probabilities": {
                        col: float(probs[col].iloc[-1])
                        for col in probs.columns
                        if not probs[col].isna().iloc[-1]
                    },
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "rolling_regime_probability"}

    # ------------------------------------------------------------------
    # New tools — expanded regime coverage
    # ------------------------------------------------------------------

    @mcp.tool()
    def fit_gaussian_hmm(
        dataset: str,
        column: str = "returns",
        n_states: int = 2,
        n_init: int = 10,
    ) -> dict[str, Any]:
        """Fit a Gaussian Hidden Markov Model directly.

        Returns the Viterbi state sequence, smoothed probabilities,
        transition matrix, per-state means and variances, and model
        selection criteria (AIC/BIC).

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
            n_states: Number of hidden states (2 = bull/bear).
            n_init: Number of EM random restarts.
        """
        try:
            import pandas as pd

            from wraquant.regimes.hmm import fit_gaussian_hmm as _fit_hmm

            df = ctx.get_dataset(dataset)
            returns = df[column].dropna()

            result = _fit_hmm(returns, n_states=n_states, n_init=n_init)

            states_df = pd.DataFrame(
                {
                    "state": result["states"],
                },
                index=returns.index[: len(result["states"])],
            )

            stored = ctx.store_dataset(
                f"hmm_states_{dataset}",
                states_df,
                source_op="fit_gaussian_hmm",
                parent=dataset,
            )

            return _sanitize_for_json(
                {
                    "tool": "fit_gaussian_hmm",
                    "dataset": dataset,
                    "n_states": n_states,
                    "means": result["means"].tolist(),
                    "variances": result["variances"].tolist(),
                    "transition_matrix": result["transition_matrix"].tolist(),
                    "current_state": int(result["states"][-1]),
                    "aic": result.get("aic"),
                    "bic": result.get("bic"),
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "fit_gaussian_hmm"}

    @mcp.tool()
    def fit_ms_autoregression(
        dataset: str,
        column: str = "returns",
        k_regimes: int = 2,
        order: int = 1,
    ) -> dict[str, Any]:
        """Fit a Markov-switching autoregression (MS-AR) model.

        Extends HMM with autoregressive lags whose coefficients can
        switch across regimes. Suited for series with serial correlation
        within each regime (GDP growth, interest rates, momentum).

        Parameters:
            dataset: Dataset containing the series.
            column: Column to model.
            k_regimes: Number of Markov-switching regimes.
            order: Autoregressive lag order.
        """
        try:
            import pandas as pd

            from wraquant.regimes.hmm import fit_ms_autoregression as _fit_msar

            df = ctx.get_dataset(dataset)
            series = df[column].dropna()

            result = _fit_msar(series, k_regimes=k_regimes, order=order)

            states_df = pd.DataFrame(
                {
                    "state": result["states"],
                },
                index=series.index[: len(result["states"])],
            )

            stored = ctx.store_dataset(
                f"msar_states_{dataset}",
                states_df,
                source_op="fit_ms_autoregression",
                parent=dataset,
            )

            return _sanitize_for_json(
                {
                    "tool": "fit_ms_autoregression",
                    "dataset": dataset,
                    "k_regimes": k_regimes,
                    "order": order,
                    "transition_matrix": result["transition_matrix"].tolist(),
                    "expected_durations": result["expected_durations"].tolist(),
                    "regime_params": result.get("regime_params"),
                    "aic": result.get("aic"),
                    "bic": result.get("bic"),
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "fit_ms_autoregression"}

    @mcp.tool()
    def gaussian_mixture_regimes(
        dataset: str,
        column: str = "returns",
        n_regimes: int = 2,
    ) -> dict[str, Any]:
        """Classify returns into regimes using a Gaussian Mixture Model.

        Quick classification without temporal structure. Good for
        labeling return distributions when you do not care about
        transition dynamics.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
            n_regimes: Number of mixture components.
        """
        try:
            import pandas as pd

            from wraquant.regimes.hmm import gaussian_mixture_regimes as _gmm

            df = ctx.get_dataset(dataset)
            returns = df[column].dropna()

            result = _gmm(returns, n_components=n_regimes)

            states_df = pd.DataFrame(
                {
                    "state": result["states"],
                },
                index=returns.index[: len(result["states"])],
            )

            stored = ctx.store_dataset(
                f"gmm_states_{dataset}",
                states_df,
                source_op="gaussian_mixture_regimes",
                parent=dataset,
            )

            return _sanitize_for_json(
                {
                    "tool": "gaussian_mixture_regimes",
                    "dataset": dataset,
                    "n_regimes": n_regimes,
                    "means": result["means"].tolist(),
                    "covariances": result["covariances"].tolist(),
                    "weights": result["weights"].tolist(),
                    "aic": result.get("aic"),
                    "bic": result.get("bic"),
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "gaussian_mixture_regimes"}

    @mcp.tool()
    def regime_conditional_moments(
        dataset: str,
        column: str = "returns",
        n_regimes: int = 2,
    ) -> dict[str, Any]:
        """Compute per-regime mean vector and covariance matrices.

        Fits an HMM then extracts the sample moments for each regime.
        These moments are building blocks for regime-aware portfolio
        construction.

        Parameters:
            dataset: Dataset containing multi-asset returns.
            column: Ignored for multi-column datasets; included for
                single-column datasets where an HMM is fitted first.
            n_regimes: Number of regimes to fit.
        """
        try:
            import numpy as np

            from wraquant.regimes.hmm import fit_gaussian_hmm as _fit_hmm
            from wraquant.regimes.hmm import regime_conditional_moments as _cond_moments

            df = ctx.get_dataset(dataset)
            numeric = df.select_dtypes(include=[np.number]).dropna()

            if numeric.shape[1] == 1:
                # Single column — fit HMM on it
                returns = numeric.iloc[:, 0]
                hmm_result = _fit_hmm(returns, n_states=n_regimes)
                states = hmm_result["states"]
            else:
                # Multi-column — fit HMM on first column, compute moments on all
                returns = numeric.iloc[:, 0]
                hmm_result = _fit_hmm(returns, n_states=n_regimes)
                states = hmm_result["states"]

            moments = _cond_moments(numeric.iloc[: len(states)], states)

            # Flatten for JSON
            moments_json = {}
            for regime_id, m in moments.items():
                moments_json[str(regime_id)] = {
                    "mean": m["mean"].tolist(),
                    "cov": m["cov"].tolist(),
                    "corr": m["corr"].tolist(),
                }

            return _sanitize_for_json(
                {
                    "tool": "regime_conditional_moments",
                    "dataset": dataset,
                    "n_regimes": n_regimes,
                    "n_assets": numeric.shape[1],
                    "moments": moments_json,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "regime_conditional_moments"}

    @mcp.tool()
    def regime_scoring(
        dataset: str,
        column: str = "returns",
        n_regimes: int = 2,
    ) -> dict[str, Any]:
        """Score regime quality: stability, separation, predictability.

        Fits an HMM and then evaluates the resulting regime
        classification on three orthogonal quality dimensions.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
            n_regimes: Number of regimes to fit.
        """
        try:
            from wraquant.regimes.hmm import fit_gaussian_hmm as _fit_hmm
            from wraquant.regimes.scoring import (
                regime_predictability,
                regime_separation_score,
                regime_stability_score,
            )

            df = ctx.get_dataset(dataset)
            returns = df[column].dropna()

            hmm_result = _fit_hmm(returns, n_states=n_regimes)
            states = hmm_result["states"]

            stability = regime_stability_score(
                states,
                transition_matrix=hmm_result["transition_matrix"],
            )
            separation = regime_separation_score(returns.values[: len(states)], states)
            predictability = regime_predictability(states)

            return _sanitize_for_json(
                {
                    "tool": "regime_scoring",
                    "dataset": dataset,
                    "n_regimes": n_regimes,
                    "stability": stability,
                    "separation": separation,
                    "predictability": predictability,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "regime_scoring"}

    @mcp.tool()
    def regime_labels(
        dataset: str,
        column: str = "returns",
        method: str = "volatility",
    ) -> dict[str, Any]:
        """Generate rule-based regime labels without model fitting.

        Assigns regime labels using volatility quantiles, trend
        direction, or a composite of both. No hidden states or EM
        — just raw statistical classification.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
            method: Labeling method — 'volatility', 'trend', or
                'composite'.
        """
        try:
            import pandas as pd

            from wraquant.regimes.labels import (
                composite_regime_labels,
                trend_regime_labels,
                volatility_regime_labels,
            )

            df = ctx.get_dataset(dataset)
            returns = df[column].dropna()

            if method == "volatility":
                labels = volatility_regime_labels(returns)
            elif method == "trend":
                labels = trend_regime_labels(returns)
            elif method == "composite":
                labels = composite_regime_labels(returns)
            else:
                msg = f"Unknown method '{method}'. Use 'volatility', 'trend', or 'composite'."
                raise ValueError(msg)

            labels_df = pd.DataFrame({"regime": labels})
            stored = ctx.store_dataset(
                f"regime_labels_{dataset}_{method}",
                labels_df,
                source_op="regime_labels",
                parent=dataset,
            )

            counts = labels.dropna().value_counts().to_dict()

            return _sanitize_for_json(
                {
                    "tool": "regime_labels",
                    "dataset": dataset,
                    "method": method,
                    "label_counts": {str(k): int(v) for k, v in counts.items()},
                    "current_label": (
                        str(labels.dropna().iloc[-1])
                        if len(labels.dropna()) > 0
                        else None
                    ),
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "regime_labels"}

    @mcp.tool()
    def kalman_filter(
        dataset: str,
        column: str = "close",
    ) -> dict[str, Any]:
        """Apply a Kalman filter for noise reduction on a time series.

        Uses a simple local-level model (random walk + noise) to
        produce smoothed state estimates from noisy observations.

        Parameters:
            dataset: Dataset containing the series.
            column: Column to filter.
        """
        try:
            import numpy as np
            import pandas as pd

            from wraquant.regimes.kalman import kalman_filter as _kalman

            df = ctx.get_dataset(dataset)
            series = df[column].dropna()
            obs = series.values.reshape(-1, 1)

            # Local-level model: state = random walk, observation = state + noise
            F = np.array([[1.0]])
            H = np.array([[1.0]])
            obs_var = float(np.var(np.diff(series.values)))
            Q = np.array([[obs_var * 0.01]])  # process noise
            R = np.array([[obs_var]])  # observation noise
            x0 = np.array([series.values[0]])
            P0 = np.array([[obs_var]])

            result = _kalman(obs, F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)

            filtered_df = pd.DataFrame(
                {
                    "observed": series.values,
                    "filtered": result["filtered_states"].flatten(),
                },
                index=series.index,
            )

            stored = ctx.store_dataset(
                f"kalman_{dataset}_{column}",
                filtered_df,
                source_op="kalman_filter",
                parent=dataset,
            )

            return _sanitize_for_json(
                {
                    "tool": "kalman_filter",
                    "dataset": dataset,
                    "column": column,
                    "observations": len(series),
                    "log_likelihood": result.get("log_likelihood"),
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "kalman_filter"}

    @mcp.tool()
    def kalman_regression(
        dataset: str,
        y_column: str,
        x_columns_json: str = "[]",
        window: int = 60,
    ) -> dict[str, Any]:
        """Estimate time-varying regression coefficients via Kalman filter.

        Models coefficients as a random walk, producing smoothed
        time-varying betas. Essential for dynamic hedge ratios,
        time-varying CAPM betas, and factor exposure monitoring.

        Parameters:
            dataset: Dataset with dependent and independent variables.
            y_column: Dependent variable column.
            x_columns_json: JSON list of independent variable column
                names. If empty, uses all numeric columns except y.
            window: Not used directly (Kalman uses all data), kept
                for API consistency. Controls state_noise scaling.
        """
        try:
            import json

            import numpy as np
            import pandas as pd

            from wraquant.regimes.kalman import kalman_regression as _kalman_reg

            df = ctx.get_dataset(dataset)

            x_columns = (
                json.loads(x_columns_json)
                if x_columns_json and x_columns_json != "[]"
                else []
            )
            if not x_columns:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                x_columns = [c for c in numeric_cols if c != y_column]

            y = df[y_column].dropna()
            X = df[x_columns].dropna()

            n = min(len(y), len(X))
            y = y.iloc[:n]
            X = X.iloc[:n]

            # Scale state noise by window
            state_noise = 1e-4 * (60 / max(window, 1))

            result = _kalman_reg(y.values, X.values, state_noise=state_noise)

            coeff_df = pd.DataFrame(
                result["coefficients"],
                columns=x_columns,
                index=y.index[: len(result["coefficients"])],
            )

            stored = ctx.store_dataset(
                f"kalman_reg_{dataset}",
                coeff_df,
                source_op="kalman_regression",
                parent=dataset,
            )

            # Latest coefficients
            latest = {col: float(coeff_df[col].iloc[-1]) for col in coeff_df.columns}

            return _sanitize_for_json(
                {
                    "tool": "kalman_regression",
                    "dataset": dataset,
                    "y_column": y_column,
                    "x_columns": x_columns,
                    "latest_coefficients": latest,
                    "log_likelihood": result.get("log_likelihood"),
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "kalman_regression"}
