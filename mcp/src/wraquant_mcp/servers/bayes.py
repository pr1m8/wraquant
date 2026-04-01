"""Bayesian inference MCP tools.

Tools: bayesian_sharpe, bayesian_regression, bayesian_changepoint,
bayesian_portfolio, bayesian_volatility, model_comparison_bayesian,
hmc_sample.
"""

from __future__ import annotations

import json
from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_bayes_tools(mcp, ctx: AnalysisContext) -> None:
    """Register Bayesian inference tools on the MCP server."""

    @mcp.tool()
    def bayesian_sharpe(
        dataset: str,
        column: str = "returns",
        n_samples: int = 5000,
    ) -> dict[str, Any]:
        """Estimate the Sharpe ratio with full Bayesian uncertainty.

        Returns the posterior distribution of the Sharpe ratio including
        credible intervals, probability of positive Sharpe, and
        posterior mean/median.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
            n_samples: Number of posterior samples.
        """
        from wraquant.bayes.models import bayesian_sharpe as _bsharpe

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna().values

        result = _bsharpe(returns, n_samples=n_samples)

        return _sanitize_for_json({
            "tool": "bayesian_sharpe",
            "dataset": dataset,
            "posterior_mean": result.posterior_mean,
            "posterior_std": result.posterior_std,
            "credible_interval_95": [result.ci_lower, result.ci_upper],
            "prob_positive": result.prob_positive,
            "n_samples": n_samples,
        })

    @mcp.tool()
    def bayesian_regression(
        dataset: str,
        y_column: str,
        x_columns_json: str = "[]",
        n_samples: int = 2000,
    ) -> dict[str, Any]:
        """Conjugate Bayesian linear regression.

        Returns the full posterior distribution of coefficients,
        log marginal likelihood for model comparison, and credible
        intervals.

        Parameters:
            dataset: Dataset with dependent and independent variables.
            y_column: Dependent variable column.
            x_columns_json: JSON array of independent variable column
                names. If empty, uses all numeric columns except y.
            n_samples: Not used directly (conjugate solution is
                analytic), but kept for API consistency.
        """
        import numpy as np

        from wraquant.bayes.models import bayesian_regression as _breg

        df = ctx.get_dataset(dataset)

        x_columns = json.loads(x_columns_json)
        if not x_columns:
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            x_columns = [c for c in numeric if c != y_column]

        y = df[y_column].dropna().values
        X = df[x_columns].dropna().values

        n = min(len(y), len(X))
        y = y[:n]
        X = X[:n]

        result = _breg(y, X)

        model_name = f"bayes_reg_{dataset}"
        stored = ctx.store_model(
            model_name, result,
            model_type="bayesian_regression",
            source_dataset=dataset,
        )

        return _sanitize_for_json({
            "tool": "bayesian_regression",
            "dataset": dataset,
            "y_column": y_column,
            "x_columns": x_columns,
            "posterior_mean": result.posterior_mean,
            "posterior_std": np.sqrt(np.diag(result.posterior_cov)).tolist(),
            "log_marginal_likelihood": result.log_marginal_likelihood,
            **stored,
        })

    @mcp.tool()
    def bayesian_changepoint(
        dataset: str,
        column: str = "returns",
        hazard: float = 250.0,
        threshold: float = 0.3,
    ) -> dict[str, Any]:
        """Bayesian online changepoint detection (Adams & MacKay, 2007).

        Processes observations sequentially and maintains a posterior
        over the run length (time since last changepoint), providing
        probability estimates at each time step.

        Parameters:
            dataset: Dataset containing the time series.
            column: Column to analyze.
            hazard: Expected run length between changepoints. Converted
                to hazard rate as 1/hazard internally.
            threshold: Probability threshold for declaring a changepoint.
        """
        import pandas as pd

        from wraquant.bayes.models import bayesian_changepoint as _bcp

        df = ctx.get_dataset(dataset)
        data = df[column].dropna().values

        # Convert expected run length to hazard rate
        hazard_rate = 1.0 / max(hazard, 1.0)

        result = _bcp(data, hazard=hazard_rate, threshold=threshold)

        cp_df = pd.DataFrame({
            "changepoint_prob": result.changepoint_posterior,
        })
        stored = ctx.store_dataset(
            f"changepoints_{dataset}", cp_df,
            source_op="bayesian_changepoint", parent=dataset,
        )

        return _sanitize_for_json({
            "tool": "bayesian_changepoint",
            "dataset": dataset,
            "n_changepoints": len(result.most_likely_changepoints),
            "changepoint_indices": result.most_likely_changepoints.tolist(),
            "hazard": hazard,
            "threshold": threshold,
            **stored,
        })

    @mcp.tool()
    def bayesian_portfolio(
        dataset: str,
        n_samples: int = 2000,
    ) -> dict[str, Any]:
        """Bayesian portfolio allocation with parameter uncertainty.

        Samples from the posterior of (mu, Sigma) using a conjugate
        normal-inverse-Wishart prior, then computes the optimal
        portfolio for each posterior draw. Returns a distribution of
        optimal weights rather than a single fragile point estimate.

        Parameters:
            dataset: Dataset with multi-asset returns (one column per
                asset). All numeric columns are used.
            n_samples: Number of posterior samples to draw.
        """
        import numpy as np

        from wraquant.bayes.models import bayesian_portfolio as _bport

        df = ctx.get_dataset(dataset)
        returns = df.select_dtypes(include=[np.number]).dropna().values

        result = _bport(returns, n_samples=n_samples)

        assets = list(df.select_dtypes(include=[np.number]).columns)

        return _sanitize_for_json({
            "tool": "bayesian_portfolio",
            "dataset": dataset,
            "n_samples": n_samples,
            "assets": assets,
            "weights_mean": result.weights_mean.tolist(),
            "weights_std": result.weights_std.tolist(),
            "expected_return": float(result.expected_return),
            "expected_risk": float(result.expected_risk),
        })

    @mcp.tool()
    def bayesian_volatility(
        dataset: str,
        column: str = "returns",
        n_samples: int = 1000,
    ) -> dict[str, Any]:
        """Bayesian stochastic volatility model via MCMC.

        Estimates a time-varying volatility path using the standard
        stochastic volatility (SV) model with Metropolis-within-Gibbs
        sampling. Returns full uncertainty bands on the volatility path.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
            n_samples: Number of MCMC samples to keep after burn-in.
        """
        import pandas as pd

        from wraquant.bayes.models import bayesian_volatility as _bvol

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna().values

        result = _bvol(returns, n_samples=n_samples)

        vol_df = pd.DataFrame({
            "vol_mean": result.vol_mean,
            "vol_ci_lower": result.vol_ci_lower,
            "vol_ci_upper": result.vol_ci_upper,
        })
        stored = ctx.store_dataset(
            f"bayes_vol_{dataset}", vol_df,
            source_op="bayesian_volatility", parent=dataset,
        )

        return _sanitize_for_json({
            "tool": "bayesian_volatility",
            "dataset": dataset,
            "n_samples": n_samples,
            "mean_volatility": float(result.vol_mean.mean()),
            "current_volatility": float(result.vol_mean[-1]),
            "phi_mean": float(result.phi_posterior.mean()),
            "sigma_eta_mean": float(result.sigma_eta_posterior.mean()),
            **stored,
        })

    @mcp.tool()
    def model_comparison_bayesian(
        dataset: str,
        column: str = "returns",
        models_json: str = "[]",
    ) -> dict[str, Any]:
        """Compare Bayesian regression models via marginal likelihood.

        Fits each candidate model specification and ranks them by
        log marginal likelihood, WAIC, and Bayes factors.

        Parameters:
            dataset: Dataset with all variables.
            column: Dependent variable column.
            models_json: JSON array of model specs. Each spec is a dict
                with 'name' (str) and 'x_columns' (list of column
                names).
                Example: '[{"name": "m1", "x_columns": ["x1", "x2"]},
                           {"name": "m2", "x_columns": ["x1"]}]'
        """
        import numpy as np

        from wraquant.bayes.models import model_comparison as _compare

        df = ctx.get_dataset(dataset)
        y = df[column].dropna().values

        models_config = json.loads(models_json)
        if not models_config:
            return {"tool": "model_comparison_bayesian", "error": "No models provided."}

        X_list = []
        model_names = []
        for spec in models_config:
            cols = spec["x_columns"]
            X_list.append(df[cols].dropna().values[:len(y)])
            model_names.append(spec.get("name", "+".join(cols)))

        result = _compare(y, X_list, model_names=model_names)

        return _sanitize_for_json({
            "tool": "model_comparison_bayesian",
            "dataset": dataset,
            "column": column,
            "ranking": result.to_dict(orient="records")
            if hasattr(result, "to_dict") else result,
        })

    @mcp.tool()
    def hmc_sample(
        log_prob_fn_name: str,
        n_dim: int = 2,
        n_samples: int = 1000,
    ) -> dict[str, Any]:
        """Run Hamiltonian Monte Carlo on a named log-probability function.

        Provides a general-purpose HMC sampler using the pure-numpy
        implementation. The log-probability function must be one of the
        built-in targets ('normal', 'banana', 'funnel') or a model
        stored in the context.

        Parameters:
            log_prob_fn_name: Name of the log-probability function.
                Built-in options:
                - 'normal': Standard normal in n_dim dimensions.
                - 'banana': Rosenbrock banana distribution (2D).
                - 'funnel': Neal's funnel distribution (2D).
            n_dim: Number of dimensions for the target distribution.
            n_samples: Number of samples to draw after burn-in.
        """
        import numpy as np
        import pandas as pd

        from wraquant.bayes.mcmc import hamiltonian_monte_carlo

        # Define built-in log-probability functions and gradients
        if log_prob_fn_name == "normal":
            def log_prob(q):
                return -0.5 * np.sum(q**2)

            def grad_log_prob(q):
                return -q

        elif log_prob_fn_name == "banana":
            n_dim = 2

            def log_prob(q):
                return -0.5 * ((q[0] - 1.0)**2 + 10.0 * (q[1] - q[0]**2)**2)

            def grad_log_prob(q):
                g0 = -(q[0] - 1.0) - 20.0 * (q[1] - q[0]**2) * (-2.0 * q[0])
                g1 = -10.0 * (q[1] - q[0]**2)
                return np.array([g0, g1])

        elif log_prob_fn_name == "funnel":
            n_dim = 2

            def log_prob(q):
                v = q[0]
                x = q[1]
                return -0.5 * (v**2 / 9.0 + x**2 * np.exp(-v))

            def grad_log_prob(q):
                v = q[0]
                x = q[1]
                dv = -v / 9.0 + 0.5 * x**2 * np.exp(-v)
                dx = -x * np.exp(-v)
                return np.array([dv, dx])

        else:
            return {
                "tool": "hmc_sample",
                "error": f"Unknown log_prob_fn_name: '{log_prob_fn_name}'. "
                         f"Use 'normal', 'banana', or 'funnel'.",
            }

        initial = np.zeros(n_dim)
        result = hamiltonian_monte_carlo(
            log_prob,
            grad_log_prob,
            initial=initial,
            n_samples=n_samples,
            step_size=0.05,
            n_leapfrog=20,
            burn_in=min(500, n_samples // 2),
        )

        samples = result["samples"]
        param_names = [f"dim_{i}" for i in range(n_dim)]

        samples_df = pd.DataFrame(samples, columns=param_names)
        stored = ctx.store_dataset(
            f"hmc_{log_prob_fn_name}", samples_df,
            source_op="hmc_sample",
        )

        return _sanitize_for_json({
            "tool": "hmc_sample",
            "log_prob_fn": log_prob_fn_name,
            "n_dim": n_dim,
            "n_samples": len(samples),
            "acceptance_rate": result["acceptance_rate"],
            "posterior_means": samples.mean(axis=0).tolist(),
            "posterior_stds": samples.std(axis=0).tolist(),
            **stored,
        })
