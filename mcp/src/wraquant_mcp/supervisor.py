"""Supervisor tool — high-level orchestrator that delegates to module servers.

The supervisor understands what each wraquant module does and can
recommend which tools to use for a given analysis task. It acts as
a routing layer — the agent asks the supervisor what to do, and the
supervisor points to the right module tools.

This is different from the module servers which are domain-specific.
The supervisor has HIGH-LEVEL knowledge of all modules and their
relationships.
"""

from __future__ import annotations

from typing import Any


def register_supervisor_tools(mcp: Any, ctx: Any) -> None:
    """Register the supervisor/orchestrator tools."""

    @mcp.tool()
    def recommend_workflow(
        goal: str,
    ) -> dict[str, Any]:
        """Given an analysis goal, recommend which tools to use and in what order.

        The supervisor understands all 25 wraquant modules and their
        relationships. Describe what you want to accomplish and it will
        suggest a step-by-step workflow.

        Parameters:
            goal: Natural language description of what you want to do.
                Examples:
                - "Analyze AAPL risk"
                - "Build a pairs trading strategy"
                - "Optimize my portfolio with regime awareness"
                - "Price an exotic option"

        Returns:
            Recommended workflow with tool names and order.
        """
        goal_lower = goal.lower()

        # Route based on keywords
        workflows = {
            "risk": {
                "workflow": "Risk Analysis",
                "steps": [
                    "1. compute_returns → get return series",
                    "2. risk_metrics → Sharpe, Sortino, max drawdown",
                    "3. var_analysis → VaR and CVaR",
                    "4. stress_test → crisis scenarios",
                    "5. crisis_drawdowns → historical worst cases",
                    "6. factor_analysis → risk decomposition (if factors available)",
                ],
                "prompt": "Use the 'risk_report' prompt for guided workflow",
                "modules": ["risk", "vol", "stats"],
            },
            "regime": {
                "workflow": "Regime Detection",
                "steps": [
                    "1. compute_returns → prepare data",
                    "2. detect_regimes → HMM with 2-3 states",
                    "3. regime_statistics → per-regime Sharpe, vol, drawdown",
                    "4. select_n_states → BIC-based state selection",
                    "5. rolling_regime_probability → time-varying regime",
                ],
                "prompt": "Use the 'regime_detection' prompt for guided workflow",
                "modules": ["regimes", "risk", "backtest"],
            },
            "volatil": {
                "workflow": "Volatility Modeling",
                "steps": [
                    "1. compute_returns → prepare data",
                    "2. realized_volatility → current vol estimate",
                    "3. fit_garch → model conditional vol (try GJR, EGARCH)",
                    "4. model_selection → compare GARCH variants",
                    "5. forecast_volatility → predict future vol",
                    "6. news_impact_curve → asymmetric shock response",
                ],
                "prompt": "Use the 'volatility_deep_dive' prompt for guided workflow",
                "modules": ["vol", "risk"],
            },
            "portfolio": {
                "workflow": "Portfolio Construction",
                "steps": [
                    "1. Load multi-asset returns",
                    "2. correlation_analysis → check diversification",
                    "3. optimize_portfolio → try risk_parity, max_sharpe, hrp",
                    "4. portfolio_risk → component VaR, diversification ratio",
                    "5. detect_regimes → regime-adjust weights",
                    "6. comprehensive_tearsheet → evaluate",
                ],
                "prompt": "Use the 'portfolio_construction' prompt for guided workflow",
                "modules": ["opt", "risk", "regimes", "backtest"],
            },
            "pair": {
                "workflow": "Pairs Trading",
                "steps": [
                    "1. Load two asset price series",
                    "2. cointegration_test → are they cointegrated?",
                    "3. Compute spread and hedge ratio",
                    "4. stationarity_test → is spread stationary?",
                    "5. Generate z-score signals",
                    "6. run_backtest → test the strategy",
                    "7. backtest_metrics → evaluate",
                ],
                "prompt": "Use the 'pairs_trading' prompt for guided workflow",
                "modules": ["stats", "ts", "backtest"],
            },
            "backtest": {
                "workflow": "Strategy Backtesting",
                "steps": [
                    "1. Prepare price/return data",
                    "2. Generate signals (TA indicators, ML, or custom)",
                    "3. run_backtest → execute the backtest",
                    "4. backtest_metrics → all performance metrics",
                    "5. comprehensive_tearsheet → full report",
                    "6. walk_forward → out-of-sample validation",
                ],
                "prompt": "Use the 'strategy_tearsheet' prompt for guided workflow",
                "modules": ["backtest", "ta", "ml", "risk"],
            },
            "forecast": {
                "workflow": "Time Series Forecasting",
                "steps": [
                    "1. stationarity_test → check if differencing needed",
                    "2. decompose → trend, seasonal, residual",
                    "3. forecast → auto_forecast tries multiple models",
                    "4. Compare forecast methods",
                    "5. forecast_evaluation → RMSE, MAE, MAPE",
                ],
                "prompt": "Use the 'equity_deep_dive' prompt (includes forecasting)",
                "modules": ["ts", "stats"],
            },
            "ml": {
                "workflow": "ML Alpha Research",
                "steps": [
                    "1. build_features → returns, vol, TA features",
                    "2. train_model → gradient_boost with walk-forward",
                    "3. feature_importance → which features matter",
                    "4. run_backtest → test predictions as signals",
                    "5. backtest_metrics → evaluate financial performance",
                ],
                "prompt": "Use the 'ml_alpha_research' prompt for guided workflow",
                "modules": ["ml", "ta", "backtest", "risk"],
            },
            "option": {
                "workflow": "Option Pricing",
                "steps": [
                    "1. price_option → Black-Scholes baseline",
                    "2. compute_greeks → delta, gamma, theta, vega",
                    "3. Vol smile analysis if multiple strikes",
                    "4. simulate_process → Heston/SABR for complex models",
                ],
                "prompt": "Use the 'option_pricing' prompt for guided workflow",
                "modules": ["price"],
            },
            "micro": {
                "workflow": "Microstructure Analysis",
                "steps": [
                    "1. liquidity_metrics → Amihud, Kyle, spread",
                    "2. toxicity_analysis → VPIN, order flow",
                    "3. market_quality → efficiency, variance ratio",
                    "4. execution_cost → pre-trade cost estimate",
                ],
                "prompt": "Use the 'microstructure_analysis' prompt",
                "modules": ["microstructure", "execution"],
            },
            "causal": {
                "workflow": "Causal Analysis",
                "steps": [
                    "1. granger_causality → does X predict Y?",
                    "2. event_study → abnormal returns around events",
                    "3. diff_in_diff → policy impact estimation",
                    "4. synthetic_control → counterfactual construction",
                ],
                "prompt": "Use the 'causal_analysis' prompt",
                "modules": ["causal", "econometrics"],
            },
        }

        # Find best matching workflow
        for keyword, wf in workflows.items():
            if keyword in goal_lower:
                return wf

        # Default: general analysis
        return {
            "workflow": "General Analysis",
            "steps": [
                "1. workspace_status → see what data is available",
                "2. analyze → comprehensive one-liner analysis",
                "3. Based on results, drill into specific modules:",
                "   - risk_metrics for risk assessment",
                "   - detect_regimes for regime analysis",
                "   - fit_garch for volatility modeling",
                "   - compute_indicator for technical signals",
            ],
            "hint": "Describe your goal more specifically for a tailored workflow",
            "available_workflows": list(workflows.keys()),
            "all_prompts": [
                "equity_deep_dive",
                "sector_comparison",
                "risk_report",
                "volatility_deep_dive",
                "regime_detection",
                "pairs_trading",
                "portfolio_construction",
                "ml_alpha_research",
                "option_pricing",
            ],
        }

    @mcp.tool()
    def module_guide(module: str) -> dict[str, Any]:
        """Get a usage guide for a specific wraquant module.

        Explains what the module does, its key tools, when to use it,
        and how it connects to other modules.

        Parameters:
            module: Module name (e.g., 'risk', 'vol', 'regimes').
        """
        guides = {
            "risk": {
                "description": "Portfolio risk measurement and management",
                "key_tools": [
                    "risk_metrics",
                    "var_analysis",
                    "stress_test",
                    "beta_analysis",
                    "factor_analysis",
                    "crisis_drawdowns",
                    "portfolio_risk",
                ],
                "when_to_use": "After computing returns, to assess how much risk you're taking and where it comes from",
                "feeds_into": [
                    "opt (risk-aware optimization)",
                    "backtest (risk-adjusted metrics)",
                    "viz (risk dashboards)",
                ],
                "feeds_from": [
                    "vol (GARCH conditional vol for VaR)",
                    "regimes (regime-conditional risk)",
                    "stats (correlation for portfolio risk)",
                ],
                "functions": 95,
            },
            "vol": {
                "description": "Volatility modeling and forecasting",
                "key_tools": [
                    "fit_garch",
                    "forecast_volatility",
                    "model_selection",
                    "news_impact_curve",
                    "realized_volatility",
                ],
                "when_to_use": "To understand volatility dynamics — is vol high/low, rising/falling, symmetric/asymmetric?",
                "feeds_into": [
                    "risk (GARCH VaR)",
                    "backtest (vol-based position sizing)",
                    "regimes (vol regime detection)",
                ],
                "feeds_from": ["data (price series)", "stats (return statistics)"],
                "functions": 28,
            },
            "regimes": {
                "description": "Market regime detection and regime-aware investing",
                "key_tools": [
                    "detect_regimes",
                    "regime_statistics",
                    "select_n_states",
                    "rolling_regime_probability",
                ],
                "when_to_use": "To identify current market state (bull/bear, high/low vol) and adapt strategy accordingly",
                "feeds_into": [
                    "opt (regime-conditional weights)",
                    "backtest (regime-filtered signals)",
                    "risk (regime-specific metrics)",
                ],
                "feeds_from": ["data (return series)", "vol (for vol-based regimes)"],
                "functions": 38,
            },
            "ta": {
                "description": "265 technical analysis indicators across 19 categories",
                "key_tools": [
                    "compute_indicator",
                    "list_indicators",
                    "multi_indicator",
                    "scan_signals",
                    "momentum_indicators",
                    "trend_indicators",
                    "volatility_indicators",
                    "volume_indicators",
                    "pattern_recognition",
                    "support_resistance",
                    "ta_summary",
                    "ta_screening",
                ],
                "when_to_use": "For generating trading signals, identifying trends, measuring momentum, detecting patterns",
                "feeds_into": [
                    "ml (TA features)",
                    "backtest (signal-based strategies)",
                    "viz (chart overlays)",
                ],
                "categories": [
                    "momentum",
                    "overlap",
                    "volume",
                    "trend",
                    "volatility",
                    "patterns",
                    "signals",
                    "statistics",
                    "cycles",
                    "fibonacci",
                    "support_resistance",
                    "breadth",
                    "performance",
                    "smoothing",
                    "exotic",
                    "candles",
                    "price_action",
                    "custom",
                ],
                "functions": 265,
            },
            "stats": {
                "description": "Statistical analysis — regression, correlation, distributions, tests",
                "key_tools": [
                    "correlation_analysis",
                    "regression",
                    "distribution_fit",
                    "stationarity_tests",
                    "cointegration_test",
                ],
                "when_to_use": "For understanding data properties, relationships between assets, and statistical significance",
                "feeds_into": [
                    "risk (correlation for portfolio risk)",
                    "ml (features)",
                    "ts (stationarity for forecasting)",
                ],
                "functions": 79,
            },
            "ts": {
                "description": "Time series analysis — forecasting, decomposition, changepoints",
                "key_tools": [
                    "forecast",
                    "decompose",
                    "stationarity_test",
                    "changepoint_detect",
                    "anomaly_detect",
                ],
                "when_to_use": "For understanding time series structure and predicting future values",
                "feeds_into": [
                    "backtest (forecast-based strategies)",
                    "risk (vol forecasting)",
                ],
                "functions": 52,
            },
            "opt": {
                "description": "Portfolio optimization — MVO, risk parity, Black-Litterman, HRP",
                "key_tools": [
                    "optimize_portfolio",
                    "efficient_frontier",
                    "rebalance_analysis",
                ],
                "when_to_use": "After computing returns and risk, to find optimal portfolio weights",
                "feeds_into": [
                    "backtest (test the allocation)",
                    "execution (trade to target weights)",
                ],
                "feeds_from": [
                    "risk (portfolio vol, component VaR)",
                    "regimes (regime adjustment)",
                    "stats (shrunk covariance)",
                ],
                "functions": 26,
            },
            "backtest": {
                "description": "Strategy backtesting and performance evaluation",
                "key_tools": [
                    "run_backtest",
                    "backtest_metrics",
                    "comprehensive_tearsheet",
                    "walk_forward",
                    "strategy_comparison",
                ],
                "when_to_use": "To test whether a strategy works historically before risking capital",
                "feeds_from": [
                    "ta (signals)",
                    "ml (predictions)",
                    "opt (weights)",
                    "regimes (filters)",
                ],
                "functions": 38,
            },
            "price": {
                "description": "Derivatives pricing — options, fixed income, stochastic models",
                "key_tools": [
                    "price_option",
                    "compute_greeks",
                    "simulate_process",
                    "yield_curve_analysis",
                ],
                "when_to_use": "For pricing options, bonds, and complex derivatives",
                "functions": 50,
            },
            "ml": {
                "description": "Machine learning — features, models, pipelines, deep learning",
                "key_tools": [
                    "build_features",
                    "train_model",
                    "feature_importance",
                    "walk_forward_ml",
                ],
                "when_to_use": "For building predictive models from financial data",
                "feeds_from": [
                    "ta (indicator features)",
                    "stats (statistical features)",
                    "regimes (regime features)",
                ],
                "feeds_into": ["backtest (ML signals)", "risk (ML-based risk)"],
                "functions": 44,
            },
            "causal": {
                "description": "Causal inference — DID, synthetic control, IV, event studies",
                "key_tools": [
                    "granger_causality",
                    "event_study",
                    "diff_in_diff",
                    "synthetic_control",
                ],
                "when_to_use": "To determine CAUSAL effects of events, policies, or interventions on financial outcomes",
                "functions": 19,
            },
            "bayes": {
                "description": "Bayesian inference — MCMC, model comparison, uncertainty quantification",
                "key_tools": [
                    "bayesian_sharpe",
                    "bayesian_regression",
                    "bayesian_changepoint",
                    "model_comparison",
                ],
                "when_to_use": "When you need uncertainty estimates, not just point estimates. Short track records, model selection.",
                "functions": 29,
            },
            "microstructure": {
                "description": "Market microstructure — liquidity, toxicity, market quality",
                "key_tools": [
                    "liquidity_metrics",
                    "toxicity_analysis",
                    "market_quality",
                    "spread_decomposition",
                ],
                "when_to_use": "For analyzing market quality, informed trading, and execution conditions",
                "feeds_into": ["execution (cost estimation)"],
                "functions": 33,
            },
            "execution": {
                "description": "Execution algorithms — TWAP, VWAP, Almgren-Chriss",
                "key_tools": ["optimal_schedule", "execution_cost", "almgren_chriss"],
                "when_to_use": "For minimizing market impact when executing large orders",
                "feeds_from": ["microstructure (liquidity, spread)"],
                "functions": 21,
            },
            "econometrics": {
                "description": "Econometrics — panel data, VAR, event studies, structural breaks",
                "key_tools": ["var_model", "panel_regression", "structural_break"],
                "when_to_use": "For advanced econometric analysis of financial data",
                "functions": 34,
            },
            "forex": {
                "description": "Forex analysis — pairs, carry, sessions, risk",
                "key_tools": ["carry_analysis", "fx_risk", "currency_strength"],
                "when_to_use": "For FX-specific analysis including carry trades and currency risk",
                "functions": 23,
            },
            "math": {
                "description": "Advanced math — Lévy processes, networks, optimal stopping",
                "key_tools": [
                    "correlation_network",
                    "levy_simulate",
                    "optimal_stopping",
                ],
                "when_to_use": "For advanced mathematical modeling and network analysis",
                "functions": 55,
            },
        }

        guide = guides.get(module)
        if guide is None:
            return {
                "error": f"Unknown module '{module}'",
                "available": list(guides.keys()),
            }
        return {"module": module, **guide}
