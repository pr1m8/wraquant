"""Risk management MCP tools.

Tools: var_analysis, stress_test, beta_analysis, factor_analysis,
crisis_drawdowns, portfolio_risk, tail_risk, credit_analysis,
copula_fit, survival_analysis, monte_carlo_var, dcc_correlation,
expected_shortfall_decomposition, cornish_fisher_var, rolling_beta.
"""

from __future__ import annotations

from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_risk_tools(mcp, ctx: AnalysisContext) -> None:
    """Register risk-specific tools on the MCP server."""

    @mcp.tool()
    def var_analysis(
        dataset: str,
        column: str = "returns",
        confidence: float = 0.95,
        method: str = "historical",
    ) -> dict[str, Any]:
        """Compute Value-at-Risk and Conditional VaR.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
            confidence: Confidence level (0.90, 0.95, 0.99).
            method: 'historical' or 'parametric'.
        """
        from wraquant.risk.var import conditional_var, value_at_risk

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()

        var = value_at_risk(returns, confidence=confidence, method=method)
        cvar = conditional_var(returns, confidence=confidence, method=method)

        return _sanitize_for_json({
            "tool": "var_analysis",
            "dataset": dataset,
            "confidence": confidence,
            "method": method,
            "var": float(var),
            "cvar": float(cvar),
            "observations": len(returns),
        })

    @mcp.tool()
    def stress_test(
        dataset: str,
        column: str = "returns",
        scenarios: dict[str, float] | None = None,
        historical: bool = False,
    ) -> dict[str, Any]:
        """Run stress tests on a return series.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
            scenarios: Dict of scenario_name -> shock_magnitude.
                e.g. {"market_crash": -0.20, "rate_hike": -0.05}
            historical: If True, run historical stress test (GFC, COVID, etc.).
        """
        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()

        results = {}

        if historical:
            from wraquant.risk.stress import historical_stress_test

            hist_result = historical_stress_test(returns)
            results["historical"] = _sanitize_for_json(hist_result)

        if scenarios:
            from wraquant.risk.stress import stress_test_returns

            for name, shock in scenarios.items():
                stressed = stress_test_returns(returns, scenarios={name: shock})
                results[name] = _sanitize_for_json(stressed)

        if not scenarios and not historical:
            from wraquant.risk.stress import vol_stress_test

            vol_result = vol_stress_test(returns, vol_shocks=[1.5, 2.0, 3.0])
            results["vol_stress"] = _sanitize_for_json(vol_result)

        return {
            "tool": "stress_test",
            "dataset": dataset,
            "results": results,
        }

    @mcp.tool()
    def beta_analysis(
        dataset: str,
        benchmark_dataset: str,
        column: str = "returns",
        benchmark_column: str = "returns",
        window: int = 60,
    ) -> dict[str, Any]:
        """Compute multiple beta estimates against a benchmark.

        Returns OLS, Blume-adjusted, Vasicek-adjusted, and conditional betas.

        Parameters:
            dataset: Dataset containing asset returns.
            benchmark_dataset: Dataset containing benchmark returns.
            column: Asset returns column.
            benchmark_column: Benchmark returns column.
            window: Rolling window for time-varying beta.
        """
        from wraquant.risk.beta import (
            blume_adjusted_beta,
            conditional_beta,
            rolling_beta,
            vasicek_adjusted_beta,
        )

        df = ctx.get_dataset(dataset)
        bdf = ctx.get_dataset(benchmark_dataset)
        returns = df[column].dropna()
        benchmark = bdf[benchmark_column].dropna()

        n = min(len(returns), len(benchmark))
        returns = returns.iloc[-n:]
        benchmark = benchmark.iloc[-n:]

        rb = rolling_beta(returns, benchmark, window=window)
        # Compute raw beta for adjustment functions
        raw_beta = float(rb.dropna().iloc[-1]) if len(rb.dropna()) > 0 else 1.0
        blume = blume_adjusted_beta(raw_beta)
        vasicek = vasicek_adjusted_beta(raw_beta)
        cond = conditional_beta(returns, benchmark)

        import pandas as pd

        rb_df = pd.DataFrame({"rolling_beta": rb})
        ctx.store_dataset(
            f"beta_{dataset}", rb_df,
            source_op="beta_analysis", parent=dataset,
        )

        return _sanitize_for_json({
            "tool": "beta_analysis",
            "dataset": dataset,
            "benchmark": benchmark_dataset,
            "current_rolling_beta": float(rb.iloc[-1]) if len(rb) > 0 else None,
            "blume_adjusted": float(blume),
            "vasicek_adjusted": float(vasicek),
            "conditional_beta": cond,
        })

    @mcp.tool()
    def factor_analysis(
        dataset: str,
        column: str = "returns",
        model: str = "pca",
        n_factors: int = 3,
    ) -> dict[str, Any]:
        """Decompose returns into factor contributions.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column.
            model: 'pca' (statistical) or 'ff' (Fama-French).
            n_factors: Number of factors for PCA.
        """
        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()

        if model == "ff":
            from wraquant.risk.factor import fama_french_regression

            result = fama_french_regression(returns)
        else:
            from wraquant.risk.factor import statistical_factor_model

            result = statistical_factor_model(returns.to_frame(), n_factors=n_factors)

        model_name = f"factor_{dataset}_{model}"
        stored = ctx.store_model(
            model_name, result,
            model_type=f"factor_{model}",
            source_dataset=dataset,
        )

        return _sanitize_for_json({**stored, "model": model})

    @mcp.tool()
    def crisis_drawdowns(
        dataset: str,
        column: str = "returns",
        top_n: int = 5,
    ) -> dict[str, Any]:
        """Analyze the worst historical drawdowns.

        Returns top N drawdowns with start/end dates, depth, duration,
        and recovery time.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column.
            top_n: Number of worst drawdowns to return.
        """
        from wraquant.risk.historical import crisis_drawdowns as _crisis

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()

        result = _crisis(returns, top_n=top_n)

        return _sanitize_for_json({
            "tool": "crisis_drawdowns",
            "dataset": dataset,
            "top_n": top_n,
            "drawdowns": result,
        })

    @mcp.tool()
    def portfolio_risk(
        dataset: str,
        weights: list[float] | None = None,
    ) -> dict[str, Any]:
        """Compute portfolio-level risk decomposition.

        Requires a multi-column dataset (each column = asset returns).
        Returns portfolio volatility, risk contributions, diversification
        ratio, and component VaR.

        Parameters:
            dataset: Dataset with multi-asset returns (one column per asset).
            weights: Portfolio weights. Defaults to equal weight.
        """
        import numpy as np

        from wraquant.risk.portfolio import (
            diversification_ratio,
            portfolio_volatility,
            risk_contribution,
        )
        from wraquant.risk.portfolio_analytics import component_var, concentration_ratio

        df = ctx.get_dataset(dataset)
        returns = df.select_dtypes(include=[np.number]).dropna()

        if weights is None:
            n_assets = returns.shape[1]
            weights = [1.0 / n_assets] * n_assets

        w = np.array(weights)

        vol = portfolio_volatility(returns, w)
        rc = risk_contribution(returns, w)
        div = diversification_ratio(returns, w)
        cvar = component_var(returns, w)
        conc = concentration_ratio(returns, w)

        return _sanitize_for_json({
            "tool": "portfolio_risk",
            "dataset": dataset,
            "assets": list(returns.columns),
            "weights": weights,
            "portfolio_volatility": float(vol),
            "risk_contributions": rc,
            "diversification_ratio": float(div),
            "component_var": cvar,
            "concentration_ratio": float(conc),
        })

    @mcp.tool()
    def tail_risk(
        dataset: str,
        column: str = "returns",
        confidence: float = 0.95,
    ) -> dict[str, Any]:
        """Compute tail risk analytics.

        Includes Cornish-Fisher VaR, conditional drawdown at risk,
        tail ratio analysis, and drawdown at risk.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column.
            confidence: Confidence level.
        """
        from wraquant.risk.tail import (
            conditional_drawdown_at_risk,
            cornish_fisher_var,
            drawdown_at_risk,
            tail_ratio_analysis,
        )

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()

        alpha = 1.0 - confidence
        cf_result = cornish_fisher_var(returns, alpha=alpha)
        cdar = conditional_drawdown_at_risk(returns, alpha=alpha)
        tra = tail_ratio_analysis(returns)
        dar = drawdown_at_risk(returns, alpha=alpha)

        return _sanitize_for_json({
            "tool": "tail_risk",
            "dataset": dataset,
            "confidence": confidence,
            "cornish_fisher_var": cf_result if isinstance(cf_result, dict) else float(cf_result),
            "conditional_drawdown_at_risk": float(cdar),
            "drawdown_at_risk": float(dar),
            "tail_ratio_analysis": tra,
        })

    @mcp.tool()
    def credit_analysis(
        total_assets: float,
        total_liabilities: float,
        equity_value: float,
        risk_free_rate: float = 0.05,
        asset_volatility: float = 0.30,
        maturity: float = 1.0,
        working_capital: float | None = None,
        retained_earnings: float | None = None,
        ebit: float | None = None,
        sales: float | None = None,
        market_cap: float | None = None,
    ) -> dict[str, Any]:
        """Compute credit risk metrics using Merton model and Altman Z-score.

        Parameters:
            total_assets: Total assets value.
            total_liabilities: Total liabilities (debt face value).
            equity_value: Market value of equity.
            risk_free_rate: Risk-free interest rate.
            asset_volatility: Estimated asset volatility.
            maturity: Debt maturity in years.
            working_capital: Working capital (for Z-score).
            retained_earnings: Retained earnings (for Z-score).
            ebit: Earnings before interest and taxes (for Z-score).
            sales: Total sales (for Z-score).
            market_cap: Market capitalization (for Z-score).
        """
        from wraquant.risk.credit import merton_model

        merton = merton_model(
            equity=equity_value,
            debt=total_liabilities,
            vol=asset_volatility,
            rf_rate=risk_free_rate,
            maturity=maturity,
        )

        result = {"merton": _sanitize_for_json(merton)}

        if all(v is not None for v in [
            working_capital, retained_earnings, ebit, sales, market_cap,
        ]):
            from wraquant.risk.credit import altman_z_score

            z = altman_z_score(
                working_capital=working_capital,
                retained_earnings=retained_earnings,
                ebit=ebit,
                market_cap=market_cap,
                total_liabilities=total_liabilities,
                total_assets=total_assets,
                sales=sales,
            )
            result["altman_z_score"] = _sanitize_for_json(z)

        return {"tool": "credit_analysis", **result}

    @mcp.tool()
    def copula_fit(
        dataset: str,
        family: str = "gaussian",
    ) -> dict[str, Any]:
        """Fit a copula model to multi-asset returns.

        Models the dependence structure between assets independently
        of their marginal distributions. Essential for tail-risk
        analysis and portfolio simulation.

        Parameters:
            dataset: Dataset with multi-asset returns (one column per asset).
            family: Copula family. Options: 'gaussian', 't', 'clayton',
                'gumbel', 'frank'.
        """
        import numpy as np

        from wraquant.risk.copulas import (
            fit_clayton_copula,
            fit_frank_copula,
            fit_gaussian_copula,
            fit_gumbel_copula,
            fit_t_copula,
        )

        df = ctx.get_dataset(dataset)
        returns = df.select_dtypes(include=[np.number]).dropna()

        if family == "gaussian":
            result = fit_gaussian_copula(returns.values)
        elif family == "t":
            result = fit_t_copula(returns.values)
        elif family in ("clayton", "gumbel", "frank"):
            if returns.shape[1] != 2:
                return {
                    "error": f"'{family}' copula requires exactly 2 columns, "
                    f"got {returns.shape[1]}",
                }
            u, v = returns.iloc[:, 0].values, returns.iloc[:, 1].values
            fitters = {
                "clayton": fit_clayton_copula,
                "gumbel": fit_gumbel_copula,
                "frank": fit_frank_copula,
            }
            result = fitters[family](u, v)
        else:
            return {
                "error": f"Unknown family '{family}'. "
                "Options: gaussian, t, clayton, gumbel, frank",
            }

        model_name = f"copula_{dataset}_{family}"
        stored = ctx.store_model(
            model_name, result,
            model_type=f"copula_{family}",
            source_dataset=dataset,
        )

        return _sanitize_for_json({
            "tool": "copula_fit",
            "dataset": dataset,
            "family": family,
            "assets": list(returns.columns),
            **stored,
            "result": result,
        })

    @mcp.tool()
    def survival_analysis(
        dataset: str,
        column: str,
        method: str = "kaplan_meier",
    ) -> dict[str, Any]:
        """Time-to-event analysis for financial durations.

        Useful for modeling drawdown durations, time between trades,
        or time to recovery.

        Parameters:
            dataset: Dataset containing duration data.
            column: Column with duration values.
            method: Survival estimator ('kaplan_meier' or 'nelson_aalen').
        """
        import numpy as np

        from wraquant.risk.survival import kaplan_meier, nelson_aalen

        df = ctx.get_dataset(dataset)
        durations = df[column].dropna().values

        # Assume all events are observed (no censoring) unless
        # an 'event' column is present
        if "event" in df.columns:
            events = df["event"].loc[df[column].notna()].values.astype(int)
        else:
            events = np.ones(len(durations), dtype=int)

        if method == "nelson_aalen":
            result = nelson_aalen(durations, events)
        else:
            result = kaplan_meier(durations, events)

        import pandas as pd

        surv_df = pd.DataFrame(_sanitize_for_json(result))
        stored = ctx.store_dataset(
            f"survival_{dataset}_{method}", surv_df,
            source_op="survival_analysis", parent=dataset,
        )

        return _sanitize_for_json({
            "tool": "survival_analysis",
            "dataset": dataset,
            "method": method,
            "observations": len(durations),
            "events_observed": int(events.sum()),
            **stored,
        })

    @mcp.tool()
    def monte_carlo_var(
        dataset: str,
        column: str = "returns",
        n_sims: int = 10_000,
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Estimate Value-at-Risk via Monte Carlo simulation.

        Draws from a fitted distribution to estimate tail risk.
        More flexible than historical VaR for short return histories.

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column name.
            n_sims: Number of Monte Carlo simulations.
            alpha: Tail probability (0.05 = 5% VaR).
        """
        import numpy as np

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna().values

        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))

        rng = np.random.default_rng(42)
        simulated = rng.normal(mu, sigma, size=n_sims)

        var = float(np.percentile(simulated, alpha * 100))
        cvar = float(simulated[simulated <= var].mean())

        return _sanitize_for_json({
            "tool": "monte_carlo_var",
            "dataset": dataset,
            "n_sims": n_sims,
            "alpha": alpha,
            "var": var,
            "cvar": cvar,
            "mu": mu,
            "sigma": sigma,
            "observations": len(returns),
        })

    @mcp.tool()
    def dcc_correlation(
        dataset: str,
        columns_json: str = "[]",
    ) -> dict[str, Any]:
        """Fit a DCC-GARCH model for dynamic conditional correlations.

        Captures time-varying correlations between assets -- critical
        for understanding how diversification benefits change in crises.

        Parameters:
            dataset: Dataset with multi-asset returns.
            columns_json: JSON list of column names to include.
                If empty, uses all numeric columns.
        """
        import json

        import numpy as np

        from wraquant.risk.dcc import dcc_garch

        df = ctx.get_dataset(dataset)

        cols = json.loads(columns_json) if columns_json and columns_json != "[]" else []
        if cols:
            returns = df[cols].dropna()
        else:
            returns = df.select_dtypes(include=[np.number]).dropna()

        result = dcc_garch(returns.values)

        model_name = f"dcc_{dataset}"
        stored = ctx.store_model(
            model_name, result,
            model_type="dcc_garch",
            source_dataset=dataset,
        )

        return _sanitize_for_json({
            "tool": "dcc_correlation",
            "dataset": dataset,
            "assets": list(returns.columns),
            **stored,
            "result": {
                k: v for k, v in result.items()
                if k not in ("correlations", "conditional_covariances")
            } if isinstance(result, dict) else str(result),
        })

    @mcp.tool()
    def expected_shortfall_decomposition(
        dataset: str,
        weights_json: str = "[]",
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Decompose Expected Shortfall into per-asset contributions.

        Shows which assets drive tail losses. Contributions are
        additive and sum to total portfolio ES.

        Parameters:
            dataset: Dataset with multi-asset returns (one column per asset).
            weights_json: JSON list of portfolio weights.
                Defaults to equal weight.
            alpha: Tail probability (0.05 = worst 5%).
        """
        import json

        import numpy as np

        from wraquant.risk.tail import expected_shortfall_decomposition as _es_decomp

        df = ctx.get_dataset(dataset)
        returns = df.select_dtypes(include=[np.number]).dropna()

        weights = json.loads(weights_json) if weights_json and weights_json != "[]" else []
        if not weights:
            n = returns.shape[1]
            weights = [1.0 / n] * n

        w = np.array(weights)
        result = _es_decomp(w, returns, alpha=alpha)

        contributions = dict(zip(returns.columns, result.values.tolist()))

        return _sanitize_for_json({
            "tool": "expected_shortfall_decomposition",
            "dataset": dataset,
            "alpha": alpha,
            "weights": weights,
            "assets": list(returns.columns),
            "contributions": contributions,
            "total_es": float(result.sum()),
        })

    @mcp.tool()
    def cornish_fisher_var(
        dataset: str,
        column: str = "returns",
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Compute Cornish-Fisher VaR adjusted for skewness and kurtosis.

        More accurate than Gaussian VaR for non-normal return
        distributions (fat tails, skew).

        Parameters:
            dataset: Dataset containing returns.
            column: Returns column.
            alpha: Tail probability (0.05 = 5% VaR).
        """
        from wraquant.risk.tail import cornish_fisher_var as _cf_var

        df = ctx.get_dataset(dataset)
        returns = df[column].dropna()

        result = _cf_var(returns, alpha=alpha)

        return _sanitize_for_json({
            "tool": "cornish_fisher_var",
            "dataset": dataset,
            "alpha": alpha,
            "observations": len(returns),
            **result if isinstance(result, dict) else {"var": float(result)},
        })

    @mcp.tool()
    def rolling_beta(
        dataset: str,
        benchmark_dataset: str,
        column: str = "returns",
        benchmark_column: str = "returns",
        window: int = 60,
    ) -> dict[str, Any]:
        """Compute time-varying rolling beta against a benchmark.

        Tracks how systematic risk exposure evolves over time.
        Useful for monitoring style drift and hedge ratio stability.

        Parameters:
            dataset: Dataset containing asset returns.
            benchmark_dataset: Dataset containing benchmark returns.
            column: Asset returns column.
            benchmark_column: Benchmark returns column.
            window: Rolling window in periods.
        """
        import pandas as pd

        from wraquant.risk.beta import rolling_beta as _rolling_beta

        df = ctx.get_dataset(dataset)
        bdf = ctx.get_dataset(benchmark_dataset)
        returns = df[column].dropna()
        benchmark = bdf[benchmark_column].dropna()

        n = min(len(returns), len(benchmark))
        returns = returns.iloc[-n:]
        benchmark = benchmark.iloc[-n:]

        rb = _rolling_beta(returns, benchmark, window=window)

        rb_df = pd.DataFrame({"rolling_beta": rb})
        stored = ctx.store_dataset(
            f"rolling_beta_{dataset}", rb_df,
            source_op="rolling_beta", parent=dataset,
        )

        return _sanitize_for_json({
            "tool": "rolling_beta",
            "dataset": dataset,
            "benchmark": benchmark_dataset,
            "window": window,
            "current_beta": float(rb.iloc[-1]) if len(rb) > 0 else None,
            "mean_beta": float(rb.mean()),
            "std_beta": float(rb.std()),
            "min_beta": float(rb.min()),
            "max_beta": float(rb.max()),
            **stored,
        })
