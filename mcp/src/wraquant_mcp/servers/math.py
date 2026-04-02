"""Advanced math MCP tools.

Tools: correlation_network, systemic_risk, levy_simulate,
optimal_stopping, hawkes_fit, spectral_analysis,
minimum_spanning_tree, community_detection, contagion_simulation,
variance_gamma_simulate, nig_simulate, longstaff_schwartz,
cusum_detect, entropy_analysis.
"""

from __future__ import annotations

from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_math_tools(mcp, ctx: AnalysisContext) -> None:
    """Register math-specific tools on the MCP server."""

    @mcp.tool()
    def correlation_network(
        dataset: str,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Build a correlation network from multi-asset returns.

        Creates an adjacency matrix where edges connect assets with
        absolute correlation above the threshold.

        Parameters:
            dataset: Dataset with multi-asset returns (one column per asset).
            threshold: Minimum absolute correlation to create an edge.
        """
        try:
            import numpy as np

            from wraquant.math.network import (
                centrality_measures,
            )
            from wraquant.math.network import correlation_network as _corr_net

            df = ctx.get_dataset(dataset)
            returns = df.select_dtypes(include=[np.number]).dropna()

            net = _corr_net(returns, threshold=threshold)
            centrality = centrality_measures(net["adjacency"])

            return _sanitize_for_json(
                {
                    "tool": "correlation_network",
                    "dataset": dataset,
                    "threshold": threshold,
                    "n_assets": len(net.get("asset_names", net.get("labels", []))),
                    "labels": net.get("asset_names", net.get("labels", [])),
                    "n_edges": int((net["adjacency"] != 0).sum() // 2),
                    "centrality": {
                        label: float(centrality["degree"][i])
                        for i, label in enumerate(
                            net.get("asset_names", net.get("labels", []))
                        )
                    },
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "correlation_network"}

    @mcp.tool()
    def systemic_risk(
        dataset: str,
    ) -> dict[str, Any]:
        """Compute systemic risk scores for each asset in a portfolio.

        Uses Marginal Expected Shortfall (MES) to measure each asset's
        contribution to systemic risk.

        Parameters:
            dataset: Dataset with multi-asset returns (one column per asset).
        """
        try:
            import numpy as np

            from wraquant.math.network import systemic_risk_score

            df = ctx.get_dataset(dataset)
            returns = df.select_dtypes(include=[np.number]).dropna()

            scores = systemic_risk_score(returns)

            scores_df = scores.to_frame("systemic_risk_score")
            stored = ctx.store_dataset(
                f"systemic_risk_{dataset}",
                scores_df,
                source_op="systemic_risk",
                parent=dataset,
            )

            return _sanitize_for_json(
                {
                    "tool": "systemic_risk",
                    "dataset": dataset,
                    "n_assets": len(scores),
                    "scores": {str(k): float(v) for k, v in scores.items()},
                    "most_systemic": str(scores.idxmax()) if len(scores) > 0 else None,
                    "least_systemic": str(scores.idxmin()) if len(scores) > 0 else None,
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "systemic_risk"}

    @mcp.tool()
    def levy_simulate(
        model: str = "variance_gamma",
        n_steps: int = 1000,
        sigma: float = 0.2,
        nu: float = 0.5,
        theta: float = -0.1,
        alpha_levy: float = 1.5,
        beta_levy: float = 0.0,
        mu_nig: float = 0.0,
        delta_nig: float = 1.0,
        alpha_nig: float = 1.0,
        beta_nig: float = 0.0,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Simulate a Levy process path.

        Supports Variance Gamma, Normal Inverse Gaussian, CGMY, and
        stable Levy models for fat-tailed asset return simulation.

        Parameters:
            model: Levy model — 'variance_gamma', 'nig', 'cgmy', or 'stable'.
            n_steps: Number of simulation steps.
            sigma: Volatility parameter (VG, CGMY).
            nu: Variance rate of the Gamma subordinator (VG).
            theta: Drift of the VG process.
            alpha_levy: Stability index for stable process (0 < alpha <= 2).
            beta_levy: Skewness for stable process (-1 <= beta <= 1).
            mu_nig: Location parameter for NIG.
            delta_nig: Scale parameter for NIG.
            alpha_nig: Tail heaviness for NIG.
            beta_nig: Asymmetry for NIG.
            seed: Random seed for reproducibility.
        """
        try:
            import numpy as np
            import pandas as pd

            if model == "variance_gamma":
                from wraquant.math.levy import variance_gamma_simulate

                path = variance_gamma_simulate(
                    sigma=sigma,
                    nu=nu,
                    theta=theta,
                    n_steps=n_steps,
                    seed=seed,
                )
            elif model == "nig":
                from wraquant.math.levy import nig_simulate

                path = nig_simulate(
                    alpha=alpha_nig,
                    beta=beta_nig,
                    mu=mu_nig,
                    delta=delta_nig,
                    n_steps=n_steps,
                    seed=seed,
                )
            elif model == "cgmy":
                from wraquant.math.levy import cgmy_simulate

                # C, G, M from sigma/nu/theta mapping
                path = cgmy_simulate(
                    C=sigma,
                    G=abs(theta) + 1.0,
                    M=abs(theta) + 1.0 + nu,
                    Y=0.5,
                    n_steps=n_steps,
                    seed=seed,
                )
            elif model == "stable":
                from wraquant.math.levy import levy_stable_simulate

                path = levy_stable_simulate(
                    alpha=alpha_levy,
                    beta=beta_levy,
                    n_steps=n_steps,
                    seed=seed,
                )
            else:
                msg = f"Unknown model '{model}'. Use 'variance_gamma', 'nig', 'cgmy', or 'stable'."
                raise ValueError(msg)

            path_df = pd.DataFrame({"path": path})
            stored = ctx.store_dataset(
                f"levy_{model}",
                path_df,
                source_op="levy_simulate",
            )

            increments = np.diff(path)

            return _sanitize_for_json(
                {
                    "tool": "levy_simulate",
                    "model": model,
                    "n_steps": n_steps,
                    "final_value": float(path[-1]),
                    "max_value": float(path.max()),
                    "min_value": float(path.min()),
                    "increment_stats": {
                        "mean": float(increments.mean()),
                        "std": float(increments.std()),
                        "skew": (
                            float(
                                ((increments - increments.mean()) ** 3).mean()
                                / (increments.std() ** 3)
                            )
                            if increments.std() > 0
                            else 0.0
                        ),
                        "kurtosis": (
                            float(
                                ((increments - increments.mean()) ** 4).mean()
                                / (increments.std() ** 4)
                            )
                            if increments.std() > 0
                            else 0.0
                        ),
                    },
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "levy_simulate"}

    @mcp.tool()
    def optimal_stopping(
        dataset: str,
        column: str = "close",
        method: str = "cusum",
        threshold: float = 2.0,
    ) -> dict[str, Any]:
        """Detect optimal exit points in a price series.

        Parameters:
            dataset: Dataset containing price data.
            column: Column to analyze.
            method: Stopping method — 'cusum' (changepoint detection) or
                'ou_exit' (mean-reversion optimal exit threshold).
            threshold: CUSUM threshold or transaction cost for OU exit.
        """
        try:
            df = ctx.get_dataset(dataset)
            series = df[column].dropna()

            if method == "cusum":
                from wraquant.math.optimal_stopping import cusum_stopping

                target_mean = float(series.mean())
                result = cusum_stopping(
                    observations=series.values,
                    target_mean=target_mean,
                    threshold=threshold,
                )

                return _sanitize_for_json(
                    {
                        "tool": "optimal_stopping",
                        "dataset": dataset,
                        "column": column,
                        "method": "cusum",
                        "threshold": threshold,
                        "target_mean": target_mean,
                        **result,
                    }
                )

            elif method == "ou_exit":
                from wraquant.math.optimal_stopping import optimal_exit_threshold

                returns = series.pct_change().dropna()
                mu = float(-returns.autocorr())
                sigma = float(returns.std())

                result = optimal_exit_threshold(
                    mu=mu,
                    sigma=sigma,
                    transaction_cost=threshold,
                )

                return _sanitize_for_json(
                    {
                        "tool": "optimal_stopping",
                        "dataset": dataset,
                        "column": column,
                        "method": "ou_exit",
                        "estimated_mu": mu,
                        "estimated_sigma": sigma,
                        "transaction_cost": threshold,
                        **result,
                    }
                )

            else:
                msg = f"Unknown method '{method}'. Use 'cusum' or 'ou_exit'."
                raise ValueError(msg)
        except Exception as e:
            return {"error": str(e), "tool": "optimal_stopping"}

    @mcp.tool()
    def hawkes_fit(
        event_times_dataset: str,
        column: str = "time",
    ) -> dict[str, Any]:
        """Fit a Hawkes self-exciting process to event arrival times.

        Estimates baseline intensity (mu), excitation magnitude (alpha),
        and decay rate (beta) via maximum likelihood.

        Parameters:
            event_times_dataset: Dataset containing event times.
            column: Column with event timestamps or numeric times.
        """
        try:
            import numpy as np

            from wraquant.math.hawkes import fit_hawkes, hawkes_branching_ratio

            df = ctx.get_dataset(event_times_dataset)
            times = df[column].dropna()

            # Convert to numeric if timestamps
            if hasattr(times.iloc[0], "timestamp"):
                t0 = times.iloc[0]
                times_numeric = np.array([(t - t0).total_seconds() for t in times])
            else:
                times_numeric = times.values.astype(float)

            result = fit_hawkes(times_numeric)

            branching = hawkes_branching_ratio(result["alpha"], result["beta"])

            stored = ctx.store_model(
                f"hawkes_{event_times_dataset}",
                result,
                model_type="hawkes",
                source_dataset=event_times_dataset,
                metrics={"branching_ratio": branching},
            )

            return _sanitize_for_json(
                {
                    "tool": "hawkes_fit",
                    "dataset": event_times_dataset,
                    "n_events": len(times_numeric),
                    "mu": result["mu"],
                    "alpha": result["alpha"],
                    "beta": result["beta"],
                    "branching_ratio": branching,
                    "stationary": branching < 1.0,
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "hawkes_fit"}

    @mcp.tool()
    def spectral_analysis(
        dataset: str,
        column: str = "close",
    ) -> dict[str, Any]:
        """FFT spectral analysis: dominant frequencies, power spectrum, entropy.

        Identifies cyclical patterns in financial time series.

        Parameters:
            dataset: Dataset containing the time series.
            column: Column to analyze.
        """
        try:
            import pandas as pd

            from wraquant.math.spectral import (
                dominant_frequencies,
                fft_decompose,
                spectral_entropy,
            )

            df = ctx.get_dataset(dataset)
            series = df[column].dropna()

            fft_result = fft_decompose(series.values)
            dominant = dominant_frequencies(series.values)
            entropy = spectral_entropy(series.values)

            spectral_df = pd.DataFrame(
                {
                    "frequencies": fft_result["frequencies"],
                    "magnitudes": fft_result["magnitudes"],
                }
            )
            stored = ctx.store_dataset(
                f"spectral_{dataset}_{column}",
                spectral_df,
                source_op="spectral_analysis",
                parent=dataset,
            )

            return _sanitize_for_json(
                {
                    "tool": "spectral_analysis",
                    "dataset": dataset,
                    "column": column,
                    "spectral_entropy": float(entropy),
                    "dominant_frequencies": dominant["frequencies"].tolist(),
                    "dominant_periods": (
                        1.0 / dominant["frequencies"][dominant["frequencies"] > 0]
                    ).tolist(),
                    "dominant_magnitudes": dominant["magnitudes"].tolist(),
                    "observations": len(series),
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "spectral_analysis"}

    # ------------------------------------------------------------------
    # Additional math tools
    # ------------------------------------------------------------------

    @mcp.tool()
    def minimum_spanning_tree(
        dataset: str,
        threshold: float = 0.0,
    ) -> dict[str, Any]:
        """Build a minimum spanning tree from asset correlation matrix.

        The MST reveals the backbone of correlation structure — the most
        important pairwise relationships with no loops. Useful for portfolio
        clustering and hierarchical risk parity.

        Parameters:
            dataset: Dataset with multi-asset returns.
            threshold: Minimum correlation to include (default 0).
        """
        try:
            import numpy as np

            from wraquant.math.network import minimum_spanning_tree as _mst

            df = ctx.get_dataset(dataset)
            returns = df.select_dtypes(include=[np.number]).dropna()
            corr = returns.corr()

            result = _mst(corr.values)

            return _sanitize_for_json(
                {
                    "tool": "minimum_spanning_tree",
                    "dataset": dataset,
                    "n_assets": len(corr),
                    "assets": list(corr.columns),
                    "n_edges": len(result.get("edges", [])),
                    "edges": result.get("edges", []),
                    "total_weight": float(result.get("total_weight", 0)),
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "minimum_spanning_tree"}

    @mcp.tool()
    def community_detection(
        dataset: str,
        n_communities: int = 3,
    ) -> dict[str, Any]:
        """Detect communities (clusters) in a financial correlation network.

        Groups assets that are more correlated with each other than with
        the rest. Useful for sector identification and diversification.

        Parameters:
            dataset: Dataset with multi-asset returns.
            n_communities: Number of communities to detect.
        """
        try:
            import numpy as np

            from wraquant.math.network import community_detection as _community
            from wraquant.math.network import correlation_network as _corr_net

            df = ctx.get_dataset(dataset)
            returns = df.select_dtypes(include=[np.number]).dropna()

            net = _corr_net(returns, threshold=0.3)
            result = _community(net["adjacency"], n_communities=n_communities)

            labels = list(returns.columns)
            communities = {}
            for i, label in enumerate(labels):
                comm_id = int(result["labels"][i]) if i < len(result["labels"]) else 0
                communities.setdefault(comm_id, []).append(label)

            return _sanitize_for_json(
                {
                    "tool": "community_detection",
                    "dataset": dataset,
                    "n_communities": len(communities),
                    "communities": communities,
                    "modularity": result.get("modularity"),
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "community_detection"}

    @mcp.tool()
    def contagion_simulation(
        dataset: str,
        shock_asset: str | None = None,
        shock_magnitude: float = 0.5,
        threshold: float = 0.3,
        max_rounds: int = 10,
    ) -> dict[str, Any]:
        """Simulate financial contagion through a correlation network.

        Shocks one asset and propagates losses through the network based
        on correlation linkages. Useful for systemic risk assessment.

        Parameters:
            dataset: Dataset with multi-asset returns.
            shock_asset: Asset to shock (defaults to most connected).
            shock_magnitude: Size of initial shock (0-1).
            threshold: Correlation threshold for contagion linkage.
            max_rounds: Maximum propagation rounds.
        """
        try:
            import numpy as np

            from wraquant.math.network import contagion_simulation as _contagion
            from wraquant.math.network import correlation_network as _corr_net

            df = ctx.get_dataset(dataset)
            returns = df.select_dtypes(include=[np.number]).dropna()

            net = _corr_net(returns, threshold=threshold)
            adj = net["adjacency"]
            labels = list(returns.columns)

            # Determine shock node
            if shock_asset and shock_asset in labels:
                shock_node = labels.index(shock_asset)
            else:
                # Shock the most connected node
                shock_node = int(np.argmax(np.sum(adj != 0, axis=1)))
                shock_asset = labels[shock_node]

            result = _contagion(
                adj,
                shock_node=shock_node,
                shock_magnitude=shock_magnitude,
                threshold=threshold,
                max_rounds=max_rounds,
            )

            return _sanitize_for_json(
                {
                    "tool": "contagion_simulation",
                    "dataset": dataset,
                    "shock_asset": shock_asset,
                    "shock_magnitude": shock_magnitude,
                    "n_rounds": result.get("n_rounds", 0),
                    "n_infected": result.get("n_infected", 0),
                    "total_assets": len(labels),
                    "infection_pct": result.get("n_infected", 0) / len(labels) * 100,
                    "losses": {
                        labels[i]: float(v)
                        for i, v in enumerate(result.get("losses", []))
                        if i < len(labels)
                    },
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "contagion_simulation"}

    @mcp.tool()
    def variance_gamma_simulate(
        n_steps: int = 1000,
        sigma: float = 0.2,
        nu: float = 0.5,
        theta: float = -0.1,
        dt: float = 1.0 / 252,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Simulate a Variance Gamma process path.

        The VG process captures fat tails and skewness through a
        time-changed Brownian motion with Gamma subordinator.

        Parameters:
            n_steps: Number of time steps.
            sigma: Volatility of the Brownian motion.
            nu: Variance rate of the Gamma subordinator (controls kurtosis).
            theta: Drift parameter (controls skewness).
            dt: Time step size (default 1/252 for daily).
            seed: Random seed.
        """
        try:
            import numpy as np
            import pandas as pd

            from wraquant.math.levy import variance_gamma_simulate as _vg_sim

            path = _vg_sim(
                sigma=sigma, nu=nu, theta=theta, n_steps=n_steps, dt=dt, seed=seed
            )
            path_df = pd.DataFrame({"path": path})
            stored = ctx.store_dataset(
                "vg_path", path_df, source_op="variance_gamma_simulate"
            )

            increments = np.diff(path)
            return _sanitize_for_json(
                {
                    "tool": "variance_gamma_simulate",
                    "n_steps": n_steps,
                    "sigma": sigma,
                    "nu": nu,
                    "theta": theta,
                    "final_value": float(path[-1]),
                    "max_value": float(path.max()),
                    "min_value": float(path.min()),
                    "increment_mean": float(increments.mean()),
                    "increment_std": float(increments.std()),
                    "increment_skew": float(
                        ((increments - increments.mean()) ** 3).mean()
                        / max(increments.std() ** 3, 1e-10)
                    ),
                    "increment_kurtosis": float(
                        ((increments - increments.mean()) ** 4).mean()
                        / max(increments.std() ** 4, 1e-10)
                    ),
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "variance_gamma_simulate"}

    @mcp.tool()
    def nig_simulate(
        n_steps: int = 1000,
        alpha: float = 1.0,
        beta: float = 0.0,
        mu: float = 0.0,
        delta: float = 1.0,
        dt: float = 1.0 / 252,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Simulate a Normal Inverse Gaussian process path.

        NIG captures asymmetric fat tails. alpha controls tail heaviness,
        beta controls asymmetry. Common for FX and equity returns.

        Parameters:
            n_steps: Number of time steps.
            alpha: Tail heaviness (larger = lighter tails).
            beta: Asymmetry (-alpha < beta < alpha).
            mu: Location parameter.
            delta: Scale parameter.
            dt: Time step size.
            seed: Random seed.
        """
        try:
            import numpy as np
            import pandas as pd

            from wraquant.math.levy import nig_simulate as _nig_sim

            path = _nig_sim(
                alpha=alpha,
                beta=beta,
                mu=mu,
                delta=delta,
                n_steps=n_steps,
                dt=dt,
                seed=seed,
            )
            path_df = pd.DataFrame({"path": path})
            stored = ctx.store_dataset("nig_path", path_df, source_op="nig_simulate")

            increments = np.diff(path)
            return _sanitize_for_json(
                {
                    "tool": "nig_simulate",
                    "n_steps": n_steps,
                    "alpha": alpha,
                    "beta": beta,
                    "mu": mu,
                    "delta": delta,
                    "final_value": float(path[-1]),
                    "max_value": float(path.max()),
                    "min_value": float(path.min()),
                    "increment_mean": float(increments.mean()),
                    "increment_std": float(increments.std()),
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "nig_simulate"}

    @mcp.tool()
    def longstaff_schwartz(
        dataset: str,
        strike: float = 100.0,
        rf_rate: float = 0.05,
        option_type: str = "put",
    ) -> dict[str, Any]:
        """Price an American option via Longstaff-Schwartz Monte Carlo.

        Uses least-squares regression on simulated paths to estimate
        the optimal early exercise boundary.

        Parameters:
            dataset: Dataset with simulated price paths (columns = paths).
            strike: Option strike price.
            rf_rate: Risk-free rate (annualized).
            option_type: 'put' or 'call'.
        """
        try:
            import numpy as np

            from wraquant.math.optimal_stopping import longstaff_schwartz as _lsm

            df = ctx.get_dataset(dataset)
            paths = df.select_dtypes(include=[np.number]).values

            # If single column, reshape
            if paths.ndim == 1:
                paths = paths.reshape(-1, 1)

            n_steps = paths.shape[0]
            dt = 1.0 / 252

            result = _lsm(
                paths=paths,
                strike=strike,
                rf_rate=rf_rate,
                dt=dt,
                option_type=option_type,
            )

            return _sanitize_for_json(
                {
                    "tool": "longstaff_schwartz",
                    "dataset": dataset,
                    "strike": strike,
                    "rf_rate": rf_rate,
                    "option_type": option_type,
                    "n_paths": paths.shape[1] if paths.ndim > 1 else 1,
                    "n_steps": n_steps,
                    "price": float(
                        result.get("price", result)
                        if isinstance(result, dict)
                        else result
                    ),
                    **(result if isinstance(result, dict) else {}),
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "longstaff_schwartz"}

    @mcp.tool()
    def cusum_detect(
        dataset: str,
        column: str = "returns",
        threshold: float = 2.0,
    ) -> dict[str, Any]:
        """CUSUM changepoint detection on a time series.

        Monitors cumulative deviations from a target mean. When the CUSUM
        statistic exceeds the threshold, a changepoint is signaled.

        Parameters:
            dataset: Dataset containing the series.
            column: Column to monitor.
            threshold: Detection threshold (in standard deviations).
        """
        try:

            from wraquant.math.optimal_stopping import cusum_stopping

            df = ctx.get_dataset(dataset)
            series = df[column].dropna()
            target_mean = float(series.mean())

            result = cusum_stopping(
                observations=series.values,
                target_mean=target_mean,
                threshold=threshold,
            )

            return _sanitize_for_json(
                {
                    "tool": "cusum_detect",
                    "dataset": dataset,
                    "column": column,
                    "threshold": threshold,
                    "target_mean": target_mean,
                    "observations": len(series),
                    **(
                        result
                        if isinstance(result, dict)
                        else {"stopping_time": result}
                    ),
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "cusum_detect"}

    @mcp.tool()
    def entropy_analysis(
        dataset: str,
        column: str = "returns",
        bins: int = 50,
    ) -> dict[str, Any]:
        """Compute Shannon entropy and related information measures.

        Higher entropy = more uncertainty/randomness. Useful for
        measuring market complexity and predictability.

        Parameters:
            dataset: Dataset containing the series.
            column: Column to analyze.
            bins: Number of histogram bins for discretization.
        """
        try:
            import numpy as np

            from wraquant.math.information import entropy as _entropy

            df = ctx.get_dataset(dataset)
            series = df[column].dropna()

            h = _entropy(series.values, bins=bins)

            # Compare to maximum entropy (uniform distribution)
            max_entropy = float(np.log2(bins))
            normalized = float(h) / max_entropy if max_entropy > 0 else 0.0

            return _sanitize_for_json(
                {
                    "tool": "entropy_analysis",
                    "dataset": dataset,
                    "column": column,
                    "entropy": float(h),
                    "max_entropy": max_entropy,
                    "normalized_entropy": normalized,
                    "predictability": 1.0 - normalized,
                    "interpretation": (
                        "highly random"
                        if normalized > 0.9
                        else (
                            "moderately random"
                            if normalized > 0.7
                            else "some structure" if normalized > 0.5 else "structured"
                        )
                    ),
                    "observations": len(series),
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "entropy_analysis"}
