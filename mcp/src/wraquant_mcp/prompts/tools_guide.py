"""Per-tool usage prompts — short, focused guides for every tool.

Each guide_ prompt walks an agent through using a specific tool
correctly, including parameter choices and interpretation guidance.
"""

from __future__ import annotations

from typing import Any


def register_tool_guide_prompts(mcp: Any) -> None:
    """Register per-tool guide prompts on the MCP server."""

    # ── risk/ ────────────────────────────────────────────────────────

    @mcp.prompt()
    def guide_copula_fit(dataset: str = "multi_asset_returns") -> list[dict]:
        """Guide: How to fit and interpret copulas."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Use copula_fit on {dataset}:

1. Call copula_fit(dataset="{dataset}", family="gaussian") -- baseline copula.
2. Call copula_fit(dataset="{dataset}", family="t") -- captures tail dependence.
3. Compare AIC/BIC -- lower is better. Student-t almost always wins for equities.
4. Check tail_dependence in the result -- if lambda_L > 0.1, Gaussian copula is dangerously wrong.
5. If lower tail dependence is high, also try family="clayton" for crash-specific modeling.

Interpretation:
- Correlation matrix from copula != Pearson correlation (it's on the copula scale).
- df < 10 in t-copula = heavy tails, assets crash together.
- df > 30 = basically Gaussian (no tail dependence).
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_survival_analysis(dataset: str = "returns") -> list[dict]:
        """Guide: Survival analysis for drawdown duration."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Survival analysis on {dataset}:

1. Identify drawdown periods -- when cumulative returns are negative.
2. Call survival_analysis(dataset="{dataset}", column="duration", method="kaplan_meier").
3. The survival curve shows: what fraction of drawdowns last at least N days?
4. Median survival time = typical drawdown duration.
5. If hazard rate is decreasing: drawdowns that last longer tend to recover faster (good).
6. If hazard rate is increasing: the longer you're in drawdown, the worse it gets (bad).

Use for: estimating how long to wait before cutting losses.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_monte_carlo_var(dataset: str = "returns") -> list[dict]:
        """Guide: Monte Carlo VaR estimation."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Monte Carlo VaR on {dataset}:

1. Call monte_carlo_var(dataset="{dataset}", column="returns", n_sims=10000, alpha=0.05).
2. The result gives VaR (5th percentile of simulated losses) and CVaR (mean of tail losses).
3. Compare to historical VaR from var_analysis -- large gap means the distribution fit matters.
4. Increase n_sims to 50000 for more stable estimates if results seem noisy.
5. If returns are non-normal, MC VaR may still underestimate risk (it assumes Gaussian by default).

Interpretation:
- VaR: "we expect to lose no more than X on 95% of days."
- CVaR: "when we do lose more than VaR, the average loss is Y."
- CVaR is always worse than VaR. If CVaR >> VaR, the tail is very fat.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_dcc_correlation(dataset: str = "multi_asset_returns") -> list[dict]:
        """Guide: DCC-GARCH dynamic conditional correlations."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
DCC-GARCH on {dataset}:

1. Call dcc_correlation(dataset="{dataset}").
2. The model estimates time-varying conditional correlations between all asset pairs.
3. Check if correlations spike during drawdowns -- this is the "diversification meltdown."
4. Compare current conditional correlation to the full-sample average.
5. If current correlations are elevated, diversification benefits are reduced.

Interpretation:
- Stable correlations = portfolio risk is predictable.
- Correlation spikes in crises = need to over-diversify or hedge explicitly.
- DCC alpha + beta near 1.0 = correlations are very persistent.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_expected_shortfall_decomposition(
        dataset: str = "multi_asset_returns",
    ) -> list[dict]:
        """Guide: ES decomposition into per-asset contributions."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
ES decomposition on {dataset}:

1. Call expected_shortfall_decomposition(dataset="{dataset}", alpha=0.05).
2. Each asset gets a contribution that sums to total portfolio ES.
3. A large contribution means that asset drives tail losses.
4. Compare contributions to portfolio weights -- overweight on risk contributors?
5. Negative contribution is possible (hedging asset that helps in tail events).

Interpretation:
- Use contributions to identify concentration risk in the tail.
- If one asset contributes >50% of ES, the portfolio is effectively a single-name bet.
- Rebalance toward assets with lower ES contributions for better tail diversification.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_cornish_fisher_var(dataset: str = "returns") -> list[dict]:
        """Guide: Cornish-Fisher VaR adjusted for higher moments."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Cornish-Fisher VaR on {dataset}:

1. Call cornish_fisher_var(dataset="{dataset}", column="returns", alpha=0.05).
2. Cornish-Fisher adjusts Gaussian VaR for skewness and excess kurtosis.
3. Compare to standard parametric VaR from var_analysis -- the gap is the skew/kurtosis correction.
4. For negatively skewed returns (equity), CF-VaR is worse (more negative) than Gaussian VaR.
5. For fat-tailed returns (kurtosis > 3), CF-VaR is also worse.

Interpretation:
- If CF-VaR is significantly worse than Gaussian: standard risk models understate risk.
- If they're similar: returns are roughly normal (unlikely for equities).
- Best used as a quick non-parametric VaR correction without full Monte Carlo.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_rolling_beta(
        dataset: str = "asset_returns", benchmark: str = "market_returns"
    ) -> list[dict]:
        """Guide: Rolling beta for tracking systematic risk drift."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Rolling beta: {dataset} vs {benchmark}:

1. Call rolling_beta(dataset="{dataset}", benchmark_dataset="{benchmark}", window=60).
2. The output includes current, mean, min, max beta over the sample.
3. Check std_beta -- high std means the asset's market exposure is unstable.
4. Compare current_beta to mean_beta -- is exposure drifting?
5. Try window=120 for a smoother estimate if 60-day is too noisy.

Interpretation:
- Beta > 1: amplifies market moves (aggressive).
- Beta < 1: dampens market moves (defensive).
- Rising beta = increasing systematic risk exposure (style drift toward market).
- Use for hedge ratio calibration: short beta * notional of benchmark to hedge.
""",
                },
            }
        ]

    # ── vol/ ─────────────────────────────────────────────────────────

    @mcp.prompt()
    def guide_hawkes_fit(dataset: str = "returns") -> list[dict]:
        """Guide: Hawkes process for volatility clustering."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Hawkes process on {dataset}:

1. Call hawkes_fit(dataset="{dataset}", column="returns").
2. The model estimates mu (baseline intensity), alpha (excitation), beta (decay).
3. Branching ratio = alpha/beta. If < 1, process is stationary (shocks die out).
4. If branching ratio is near 1.0, volatility clustering is extreme -- shocks self-amplify.
5. High alpha relative to beta = each shock triggers many subsequent shocks.

Interpretation:
- Hawkes captures "contagion" in volatility -- one large move increases the probability of another.
- Compare to GARCH: Hawkes is event-driven (counts), GARCH is level-driven (variance).
- Useful for modeling flash crashes and cascading sell-offs.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_stochastic_vol(dataset: str = "returns") -> list[dict]:
        """Guide: Stochastic volatility model via particle filter."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Stochastic volatility on {dataset}:

1. Call stochastic_vol(dataset="{dataset}", column="returns").
2. The SV model treats log-volatility as a latent AR(1) process.
3. Check the filtered_vol path -- it should track realized vol but be smoother.
4. Key parameters: persistence (phi), vol-of-vol (sigma_eta).
5. High phi (>0.95) = volatility is very persistent (similar to high GARCH persistence).

Interpretation:
- SV models are more flexible than GARCH -- volatility is a separate random process.
- Compare SV filtered vol to GARCH conditional vol -- they should agree qualitatively.
- SV is the continuous-time analog of what GARCH estimates in discrete time.
- Use for option pricing where GARCH is too rigid.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_variance_risk_premium(dataset: str = "vol_data") -> list[dict]:
        """Guide: Variance risk premium (implied minus realized)."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Variance risk premium on {dataset}:

1. Ensure {dataset} has both realized_vol and implied_vol columns.
2. Call variance_risk_premium(dataset="{dataset}", realized_col="realized_vol", implied_col="implied_vol").
3. VRP = implied variance - realized variance. Positive = investors overpay for vol protection.
4. A persistently positive VRP is the foundation of short-vol strategies.
5. VRP spikes negative during crises (realized > implied -- market caught off guard).

Interpretation:
- Mean VRP ~ 2-4 vol points for SPX historically.
- Sell vol when VRP is high (rich implied vol).
- Avoid selling vol when VRP is negative (realized vol is surprising the market).
- Track VRP over time to time variance swap and option strategies.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_bipower_variation(dataset: str = "returns") -> list[dict]:
        """Guide: Bipower variation for jump-robust volatility."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Bipower variation on {dataset}:

1. Call bipower_variation(dataset="{dataset}", column="returns", window=20).
2. BPV estimates the continuous (diffusive) component of volatility, filtering out jumps.
3. Compare BPV to realized variance -- the difference is the jump component.
4. If RV >> BPV: significant jump activity is present.
5. Use BPV as the "true" volatility input for option pricing models.

Interpretation:
- BPV < RV: returns contain jumps (sudden large moves).
- BPV ~ RV: price process is mostly diffusive (continuous).
- Jump component = RV - BPV. Large jump component = need jump-diffusion models.
- Pair with jump_detection for a formal statistical test.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_jump_detection(dataset: str = "returns") -> list[dict]:
        """Guide: Barndorff-Nielsen-Shephard jump test."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Jump detection on {dataset}:

1. Call jump_detection(dataset="{dataset}", column="returns").
2. The BNS test compares realized variance to bipower variation.
3. If the test statistic is significant (p < 0.05): jumps are present.
4. The jump contribution shows what fraction of total variance comes from jumps.

Interpretation:
- Significant jumps invalidate pure diffusion models (GBM, basic GARCH).
- Need jump-diffusion (Merton) or Levy process models for accurate pricing.
- Financial time series almost always have jumps -- the question is how large.
- Jumps matter most for short-dated options and tail risk estimation.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_garch_rolling(dataset: str = "returns") -> list[dict]:
        """Guide: Rolling GARCH volatility forecast."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Rolling GARCH on {dataset}:

1. Call garch_rolling(dataset="{dataset}", column="returns", window=500, horizon=1).
2. Refits GARCH on a rolling window at each point, producing out-of-sample forecasts.
3. Compare forecasts to subsequent realized volatility to assess accuracy.
4. Use window=500 (2 years daily) for stable estimation; window=250 for faster adaptation.
5. horizon=1 for next-day vol; horizon=5 for next-week vol.

Interpretation:
- Good GARCH forecasts should track realized vol with low bias.
- If forecasts consistently overshoot: model overestimates persistence.
- If forecasts consistently undershoot: model underestimates vol-of-vol.
- Use rolling forecasts to build a dynamic hedge ratio or position sizing rule.
""",
                },
            }
        ]

    # ── stats/ ───────────────────────────────────────────────────────

    @mcp.prompt()
    def guide_partial_correlation(dataset: str = "multi_asset_returns") -> list[dict]:
        """Guide: Partial correlation to find direct relationships."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Partial correlation on {dataset}:

1. Call partial_correlation(dataset="{dataset}").
2. Partial correlation removes the effect of all other variables from each pair.
3. Compare to Pearson correlation -- high Pearson but low partial = mediated relationship.
4. Low Pearson but high partial = relationship was masked by a confounding variable.

Interpretation:
- Partial correlation reveals the DIRECT linear link between two assets.
- If asset A and B have high Pearson but low partial: they co-move because of a third factor.
- Use for network construction -- edges should be partial correlations, not raw correlations.
- Essential for distinguishing causation-like structure from spurious correlation.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_distance_correlation(
        dataset: str = "returns", col_a: str = "asset_a", col_b: str = "asset_b"
    ) -> list[dict]:
        """Guide: Distance correlation for nonlinear dependence."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Distance correlation between {col_a} and {col_b} in {dataset}:

1. Call distance_correlation(dataset="{dataset}", col_a="{col_a}", col_b="{col_b}").
2. Distance correlation = 0 if and only if variables are independent (unlike Pearson).
3. It captures nonlinear and non-monotonic dependence that Pearson and Spearman miss.
4. Compare to Pearson: if dCorr >> Pearson, there is significant nonlinear dependence.

Interpretation:
- dCorr ranges from 0 (independent) to 1 (perfectly dependent).
- High dCorr + low Pearson = strong nonlinear relationship (e.g., quadratic).
- Computationally expensive: O(n^2). Use on reasonable sample sizes (<5000).
- Great for feature selection in ML -- identifies features with any dependence on target.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_mutual_information(
        dataset: str = "returns", col_a: str = "feature", col_b: str = "target"
    ) -> list[dict]:
        """Guide: Mutual information for general dependence."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Mutual information between {col_a} and {col_b} in {dataset}:

1. Call mutual_information(dataset="{dataset}", col_a="{col_a}", col_b="{col_b}").
2. MI measures how much knowing one variable reduces uncertainty about the other.
3. MI = 0 means the variables are independent. Higher = more shared information.
4. Unlike correlation, MI captures all types of dependence.

Interpretation:
- MI is in nats (natural log). Convert to bits by dividing by ln(2).
- MI > 0.1 nats typically indicates a meaningful relationship for financial data.
- Use for feature ranking: sort features by MI with the target variable.
- MI does not indicate direction (positive/negative) -- only strength.
- Complement with correlation to understand both strength and direction.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_kde_estimate(dataset: str = "returns") -> list[dict]:
        """Guide: Kernel density estimation for return distributions."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
KDE on {dataset}:

1. Call kde_estimate(dataset="{dataset}", column="returns").
2. The result is a smooth, non-parametric density estimate.
3. Compare the KDE shape to a Gaussian -- look for fat tails, skewness, multimodality.
4. Use the KDE for non-parametric VaR: find the alpha-quantile of the estimated density.

Interpretation:
- Bimodal KDE = the returns may come from a mixture (e.g., two regimes).
- Left-skewed KDE = crashes are more frequent than rallies (typical for equities).
- Fat tails in the KDE relative to Gaussian = need heavy-tailed risk models.
- KDE is the best way to visualize what the return distribution actually looks like.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_best_fit_distribution(dataset: str = "returns") -> list[dict]:
        """Guide: Automated distribution fitting and selection."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Best-fit distribution on {dataset}:

1. Call best_fit_distribution(dataset="{dataset}", column="returns").
2. Tests normal, t, skewed-t, stable, and other distributions.
3. Ranking is by AIC (lower = better fit with parsimony penalty).
4. Check KS statistic p-value -- p < 0.05 means the distribution is rejected.

Interpretation:
- Normal almost never wins for daily returns (fat tails, skew).
- Student-t usually wins -- the df parameter tells you how fat the tails are.
- Skewed-t adds asymmetry on top of fat tails.
- Stable distributions handle extreme fat tails but have infinite variance.
- Use the winning distribution for parametric VaR and Monte Carlo simulation.
""",
                },
            }
        ]

    # ── ts/ ──────────────────────────────────────────────────────────

    @mcp.prompt()
    def guide_ssa_decompose(dataset: str = "prices") -> list[dict]:
        """Guide: Singular Spectrum Analysis decomposition."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
SSA decomposition on {dataset}:

1. Call ssa_decompose(dataset="{dataset}", column="close", n_components=3).
2. Component 1 is typically the trend. Components 2-3 capture oscillations.
3. Residual = original minus all extracted components = noise.
4. Compare SSA trend to a simple moving average -- SSA adapts to the data shape.

Interpretation:
- SSA is non-parametric: no assumptions about seasonality or trend shape.
- More components = more signal extracted, but risk of overfitting.
- Use the trend component for regime detection or as a slow signal.
- Use oscillatory components to identify cyclical trading opportunities.
- SSA works best on stationary or trend-stationary series.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_arima_diagnostics(dataset: str = "returns") -> list[dict]:
        """Guide: ARIMA model fitting and residual diagnostics."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
ARIMA diagnostics on {dataset}:

1. Call arima_diagnostics(dataset="{dataset}", column="returns").
2. Auto-ARIMA selects (p, d, q) order via AIC.
3. Check Ljung-Box test on residuals -- p > 0.05 means no remaining autocorrelation (good).
4. Check normality test -- if residuals are non-normal, consider GARCH for the variance.
5. Check heteroscedasticity test -- significant means ARCH effects (use GARCH).

Interpretation:
- If Ljung-Box rejects: the model misses serial dependence. Increase p or q.
- If normality rejects: fine for point forecasts, bad for prediction intervals.
- If heteroscedasticity present: residuals have time-varying variance, need ARIMA-GARCH.
- A well-specified model has white noise residuals (no patterns left).
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_rolling_forecast(dataset: str = "returns") -> list[dict]:
        """Guide: Walk-forward rolling forecast evaluation."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Rolling forecast on {dataset}:

1. Call rolling_forecast(dataset="{dataset}", column="returns", horizon=5, window=200).
2. At each step, the model is re-fit on a window of data and forecasts ahead.
3. The result contains out-of-sample forecasts vs actual values.
4. Compute RMSE, MAE, and directional accuracy from the stored results.

Interpretation:
- This is the gold standard for forecast evaluation -- no look-ahead bias.
- If RMSE is close to naive (random walk) RMSE: the model adds no value.
- Directional accuracy > 52% is meaningful for trading (given transaction costs).
- window=200 is a good starting point. Too small = unstable, too large = slow to adapt.
- Use to compare multiple forecasting methods on the same data.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_ensemble_forecast(dataset: str = "returns") -> list[dict]:
        """Guide: Ensemble forecast combining multiple methods."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Ensemble forecast on {dataset}:

1. Call ensemble_forecast(dataset="{dataset}", column="returns", horizon=10).
2. Combines ARIMA, ETS, Theta, and other methods via inverse-RMSE weighting.
3. The ensemble often outperforms any single model because it diversifies model risk.
4. Check which sub-models get the highest weights -- they fit this data best.

Interpretation:
- If one model dominates the weight: the data has a strong pattern that model captures.
- If weights are roughly equal: no single model is clearly best (ensembling helps most here).
- Ensemble forecasts are more robust to structural breaks than any single model.
- Use as the primary forecast and single models as diagnostics.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_ornstein_uhlenbeck(dataset: str = "spread") -> list[dict]:
        """Guide: Ornstein-Uhlenbeck mean-reversion estimation."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Ornstein-Uhlenbeck on {dataset}:

1. Call ornstein_uhlenbeck(dataset="{dataset}", column="close").
2. Estimates kappa (speed), theta (long-run mean), sigma (volatility).
3. Half-life = ln(2) / kappa. This is how long it takes to revert halfway to the mean.
4. Half-life < 20 days = fast enough for pairs trading. > 60 days = too slow.

Interpretation:
- High kappa = fast mean reversion (good for trading).
- Low kappa = slow reversion (acts more like a random walk).
- theta is the equilibrium level -- trade toward it.
- sigma determines the spread's volatility -- use for position sizing.
- Use OU parameters to set entry/exit thresholds: enter at theta +/- k*sigma.
""",
                },
            }
        ]

    # ── backtest/ ────────────────────────────────────────────────────

    @mcp.prompt()
    def guide_omega_ratio(dataset: str = "strategy_returns") -> list[dict]:
        """Guide: Omega ratio -- gain/loss probability-weighted ratio."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Omega ratio on {dataset}:

1. Call omega_ratio(dataset="{dataset}", column="returns", threshold=0.0).
2. Omega = probability-weighted gains / probability-weighted losses.
3. Unlike Sharpe, Omega uses the entire return distribution (all moments).
4. Omega > 1 means gains outweigh losses. Higher is better.
5. Try threshold=risk_free_rate/252 for excess-return Omega.

Interpretation:
- Omega > 1.5 = strong strategy. Omega < 1.0 = losing strategy.
- More appropriate than Sharpe for non-normal returns (skewed, fat-tailed).
- Omega at different thresholds maps out the full gain-loss tradeoff.
- Compare strategies: higher Omega at the same threshold = better.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_kelly_fraction(dataset: str = "strategy_returns") -> list[dict]:
        """Guide: Kelly criterion for optimal position sizing."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Kelly fraction on {dataset}:

1. Call kelly_fraction(dataset="{dataset}", column="returns").
2. Full Kelly = fraction of capital that maximizes geometric growth rate.
3. Half Kelly (full_kelly / 2) is the practitioner standard -- same expected growth, much less variance.
4. Requires both winning and losing trades in the sample.

Interpretation:
- Full Kelly > 1.0 = strategy is so good it suggests leverage (be skeptical).
- Full Kelly < 0.0 = losing strategy, don't trade it.
- Full Kelly 0.2 - 0.5 = typical for decent strategies.
- ALWAYS use half Kelly or less in practice -- estimation error in win rate is large.
- Kelly assumes IID returns, which financial returns are NOT. Use conservatively.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_vectorized_backtest(
        dataset: str = "prices", signals: str = "signals"
    ) -> list[dict]:
        """Guide: Fast vectorized backtest with transaction costs."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Vectorized backtest: {dataset} + {signals}:

1. Call vectorized_backtest(dataset="{dataset}", signals_dataset="{signals}", commission=0.001).
2. Multiplies signals by returns, deducting commission on each trade.
3. The signal dataset needs a signal/signals/position column with values 1, -1, or 0.
4. commission=0.001 = 10bps per trade (reasonable for liquid equities).

Interpretation:
- Compare Sharpe with and without commission -- if it drops below 1.0, costs kill the strategy.
- n_trades / n_days = turnover. High turnover + low alpha = eaten by costs.
- total_commission shows the cumulative drag from trading.
- If max_drawdown is much worse with costs, the strategy is marginal.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_drawdown_analysis(dataset: str = "strategy_returns") -> list[dict]:
        """Guide: Detailed drawdown period analysis."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Drawdown analysis on {dataset}:

1. Call drawdown_analysis(dataset="{dataset}", column="returns", top_n=5).
2. Returns the 5 worst drawdowns with start date, end date, depth, duration, and recovery time.
3. Check current_drawdown -- are we currently in a drawdown?
4. Compare max drawdown to annualized return: a drawdown > 2x annual return is concerning.

Interpretation:
- Long recovery times (>6 months) suggest the strategy has regime vulnerability.
- If the worst drawdowns cluster in time, the strategy fails in specific market conditions.
- Drawdown depth vs duration: deep-but-short is better than shallow-but-long.
- Use drawdown analysis to set stop-loss or position reduction rules.
""",
                },
            }
        ]

    # ── ml/ ──────────────────────────────────────────────────────────

    @mcp.prompt()
    def guide_pca_factors(dataset: str = "multi_asset_returns") -> list[dict]:
        """Guide: PCA factor extraction from multi-asset returns."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
PCA factors from {dataset}:

1. Call pca_factors(dataset="{dataset}", n_factors=3).
2. PC1 is almost always "the market" (captures ~60-80% of variance for equities).
3. PC2 often captures sector rotation, value/growth, or risk-on/risk-off.
4. Check explained_variance_ratio -- if first 3 PCs explain >90%, the space is low-dimensional.

Interpretation:
- Use PCA factors as features for ML models instead of raw returns (dimensionality reduction).
- Build a statistical factor model: regress each asset on the PCA factors.
- Residuals from PCA regression = idiosyncratic returns (alpha candidates).
- If a new PC emerges with high variance: a new risk factor has appeared.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_isolation_forest(dataset: str = "returns") -> list[dict]:
        """Guide: Isolation Forest anomaly detection."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Isolation Forest on {dataset}:

1. Call isolation_forest(dataset="{dataset}", contamination=0.05).
2. contamination=0.05 means expect ~5% of observations to be anomalous.
3. Anomalous days are labeled -1 in the output.
4. Check what dates are flagged -- do they correspond to known events (flash crashes, earnings)?

Interpretation:
- Isolation Forest identifies outliers by how quickly they are "isolated" via random splits.
- Works well in high dimensions (multiple features) where z-score fails.
- Set contamination based on domain knowledge: 0.01-0.05 for daily financial data.
- Use flagged anomalies to investigate unusual market behavior or data quality issues.
- Consider removing anomalies before fitting models sensitive to outliers (e.g., OLS).
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_svm_classify(dataset: str = "features") -> list[dict]:
        """Guide: SVM classification for regime or signal prediction."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
SVM classification on {dataset}:

1. Ensure {dataset} has a target/label column and numeric feature columns.
2. Call svm_classify(dataset="{dataset}", target_col="label").
3. Uses grid search over kernels (linear, RBF, poly) with chronological train/test split.
4. Check test accuracy and the confusion matrix in the result.

Interpretation:
- SVM works best when classes are separable in feature space.
- RBF kernel handles nonlinear boundaries but can overfit with many features.
- Linear kernel is more robust and interpretable for financial data.
- If accuracy is ~50%: features don't separate regimes. Try better features.
- Feature scaling is critical for SVM -- the tool handles this internally.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_online_regression(dataset: str = "returns") -> list[dict]:
        """Guide: Online regression with time-varying coefficients."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Online regression on {dataset}:

1. Call online_regression(dataset="{dataset}", y_col="returns", halflife=60).
2. Uses Recursive Least Squares with exponential forgetting (decay from halflife).
3. Coefficients evolve over time -- tracks how factor exposures drift.
4. halflife=60 (3 months) is a good default. Lower = faster adaptation, noisier.

Interpretation:
- Plot coefficients over time to see factor exposure drift.
- If coefficients are stable: a static model is fine. If volatile: need online updating.
- Use for live hedge ratio estimation: the latest coefficient is the current optimal hedge.
- Compare to rolling OLS: online regression is smoother and more computationally efficient.
- Great for detecting when a strategy's alpha source is decaying.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_cross_asset_features(
        dataset: str = "asset_returns", benchmark: str = "market_returns"
    ) -> list[dict]:
        """Guide: Cross-asset feature construction."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Cross-asset features: {dataset} vs {benchmark}:

1. Call cross_asset_features(dataset="{dataset}", benchmark_dataset="{benchmark}", window=60).
2. Generates: rolling_corr, rolling_beta, rel_vol, rolling_alpha, tracking_error.
3. These features capture the inter-asset relationship dynamics.

Interpretation:
- Rising rolling_corr: assets are converging (less diversification benefit).
- Rolling_beta drift: systematic risk exposure is changing.
- rel_vol > 1: asset is more volatile than benchmark (amplifier).
- Positive rolling_alpha: asset outperforms risk-adjusted benchmark.
- Use these features as inputs to ML models for regime-aware portfolio construction.
""",
                },
            }
        ]

    # ── price/ ───────────────────────────────────────────────────────

    @mcp.prompt()
    def guide_implied_volatility(
        market_price: float = 10.0, spot: float = 100.0, strike: float = 100.0
    ) -> list[dict]:
        """Guide: Computing implied volatility from option prices."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Implied volatility extraction:

1. Call implied_volatility(market_price={market_price}, spot={spot}, strike={strike}, rf=0.05, maturity=0.25, option_type="call").
2. Newton's method inverts Black-Scholes to find the vol that reproduces the market price.
3. Compare implied vol to realized vol -- the difference is the variance risk premium.

Interpretation:
- IV > realized vol: options are "rich" (overpriced relative to historical).
- IV < realized vol: options are "cheap" (unusual, usually during complacency).
- IV across strikes gives the vol smile/skew -- steep skew = crash fear.
- ATM IV is the market's best estimate of future realized volatility.
- Use for vol trading: sell rich IV, buy cheap IV.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_sabr_calibrate() -> list[dict]:
        """Guide: SABR model calibration to market vol smile."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
SABR calibration:

1. Collect market implied vols at multiple strikes for the same maturity.
2. Call sabr_calibrate(forward=100.0, strikes_json="[90,95,100,105,110]", vols_json="[0.25,0.22,0.20,0.22,0.24]", maturity=0.25).
3. SABR returns alpha (vol level), beta (CEV exponent), rho (skew), nu (vol-of-vol).

Interpretation:
- beta is usually fixed (0.5 for rates, 1.0 for equities). Don't over-interpret.
- rho < 0 means negative skew (left tail is fatter) -- normal for equities.
- nu controls the smile curvature. High nu = pronounced smile (wings are expensive).
- alpha sets the overall vol level at the ATM point.
- Use calibrated SABR to interpolate vols at any strike for consistent pricing.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_simulate_heston(spot: float = 100.0) -> list[dict]:
        """Guide: Heston stochastic volatility simulation."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Heston simulation (spot={spot}):

1. Call simulate_heston(spot={spot}, v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, T=1.0, n_paths=1000).
2. v0=0.04 is initial variance (vol = 20%). theta=0.04 is long-run variance.
3. kappa=2.0 is mean-reversion speed for variance. rho=-0.7 captures leverage effect.
4. sigma_v=0.3 is vol-of-vol (controls smile curvature).

Interpretation:
- rho < 0 generates negative skew (down moves increase vol) -- realistic for equities.
- kappa * theta should satisfy Feller condition: 2*kappa*theta > sigma_v^2 to keep vol positive.
- Mean path converges to forward price. Distribution of final prices shows the smile.
- Use for option pricing: price = discounted average payoff across paths.
- Compare summary stats to observed market distribution for calibration validation.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_bond_duration() -> list[dict]:
        """Guide: Bond duration and convexity analysis."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
Bond duration analysis:

1. Call bond_duration(face_value=1000, coupon_rate=0.05, ytm=0.04, periods=20).
2. Macaulay duration: weighted average time to receive cash flows (in years).
3. Modified duration: percentage price change per 1% yield change.
4. Convexity: second-order correction -- captures the curvature of price-yield relationship.

Interpretation:
- Modified duration of 7 means: if yields rise 1%, bond price falls ~7%.
- Price change ~ -ModDur * dy + 0.5 * Convexity * dy^2 (convexity helps).
- Higher coupon = lower duration (cash flows arrive sooner).
- Higher yield = lower duration (future cash flows discounted more).
- Use for immunization: match portfolio duration to liability duration.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_fbsde_price() -> list[dict]:
        """Guide: Forward-Backward SDE option pricing."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
FBSDE pricing:

1. Call fbsde_price(spot=100, strike=100, rf=0.05, vol=0.20, T=1.0, n_paths=10000).
2. FBSDE solves the pricing PDE via Monte Carlo on the forward-backward SDE system.
3. Compare to Black-Scholes price -- they should agree for European options.

Interpretation:
- FBSDE is overkill for vanilla Europeans (use BS directly).
- Its value is for: path-dependent options, incomplete markets, American exercise.
- The framework extends naturally to nonlinear pricing (funding costs, CVA).
- If FBSDE and BS disagree significantly: check n_paths (increase for convergence).
- Use n_paths=50000+ for accurate exotic pricing.
""",
                },
            }
        ]

    # ── opt/ ─────────────────────────────────────────────────────────

    @mcp.prompt()
    def guide_black_litterman(dataset: str = "multi_asset_returns") -> list[dict]:
        """Guide: Black-Litterman portfolio optimization."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Black-Litterman on {dataset}:

1. Call black_litterman(dataset="{dataset}", views_json='{{"AAPL": 0.10, "MSFT": 0.05}}').
2. BL combines market equilibrium returns with your subjective views.
3. Views override market-implied returns for specific assets.
4. The confidence in each view is automatically scaled by estimation uncertainty.

Interpretation:
- Without views: BL gives the market-cap-weighted portfolio (equilibrium).
- With views: weights tilt toward assets you're bullish on.
- BL is more stable than raw MVO because it starts from a reasonable prior.
- Check if the resulting weights are sensible -- extreme weights mean extreme views.
- Use market_weights to properly anchor the equilibrium if available.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_hierarchical_risk_parity(
        dataset: str = "multi_asset_returns",
    ) -> list[dict]:
        """Guide: HRP portfolio optimization."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Hierarchical Risk Parity on {dataset}:

1. Call hierarchical_risk_parity(dataset="{dataset}").
2. HRP uses hierarchical clustering to group similar assets, then allocates inversely to risk.
3. No matrix inversion required -- works even when covariance is ill-conditioned.
4. Produces a well-diversified portfolio without needing expected return estimates.

Interpretation:
- HRP weights are always between 0 and 1 (no short sales, no leverage).
- Clustered assets share weight -- if 5 tech stocks cluster, they each get ~1/5 of the tech allocation.
- Compare to risk parity and min-vol -- HRP is typically between the two.
- HRP is the most robust optimizer for small samples or many assets.
- Less sensitive to estimation error than MVO or max Sharpe.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_min_volatility(dataset: str = "multi_asset_returns") -> list[dict]:
        """Guide: Minimum variance portfolio."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Min-volatility portfolio on {dataset}:

1. Call min_volatility(dataset="{dataset}").
2. Finds the portfolio with the lowest possible volatility on the efficient frontier.
3. The ONLY efficient portfolio that does not require expected return estimates.
4. Returns the optimal weights and the achieved portfolio volatility.

Interpretation:
- Min-vol tends to overweight low-vol assets and assets with low correlation to others.
- Often outperforms 1/N in practice despite ignoring expected returns (Jagannathan & Ma, 2003).
- If portfolio_volatility is only slightly below equal-weight vol: assets are similar.
- Large weight concentration = one asset dominates the minimum variance solution.
- Use as a defensive allocation or as a benchmark for other optimizers.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_max_sharpe(dataset: str = "multi_asset_returns") -> list[dict]:
        """Guide: Maximum Sharpe ratio (tangency) portfolio."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Max Sharpe portfolio on {dataset}:

1. Call max_sharpe(dataset="{dataset}", risk_free_rate=0.04).
2. Finds the portfolio on the efficient frontier with the highest Sharpe ratio.
3. This is the tangency portfolio -- the point where the capital market line touches the frontier.
4. Highly sensitive to expected return estimates.

Interpretation:
- Max Sharpe typically concentrates in a few assets with high estimated Sharpe.
- Extreme weights = estimation error is dominating. Use with caution.
- Compare to HRP and min-vol -- if max Sharpe gives very different weights, the expected returns are unreliable.
- In practice, constrain weights to max 20-30% per asset to limit estimation error.
- The reported sharpe_ratio, expected_return, volatility are in-sample -- out-of-sample will be worse.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_risk_budgeting(dataset: str = "multi_asset_returns") -> list[dict]:
        """Guide: Risk budgeting portfolio optimization."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Risk budgeting on {dataset}:

1. With equal budgets (risk parity): call risk_budgeting(dataset="{dataset}").
2. With custom budgets: call risk_budgeting(dataset="{dataset}", target_risk_json="[0.5, 0.3, 0.2]").
3. Target risk contributions must sum to 1.0.
4. The optimizer finds weights so each asset contributes exactly its target fraction of total risk.

Interpretation:
- Equal risk budgets = risk parity (each asset contributes the same amount of risk).
- Custom budgets let you express views on risk allocation, not returns.
- Compare to min-vol: risk budgeting gives more diversified risk contributions.
- If an asset needs a very large weight to hit its risk target: it's low vol and low correlation.
- Risk budgeting never produces extreme weights (stable, robust).
""",
                },
            }
        ]

    # ── microstructure/ ──────────────────────────────────────────────

    @mcp.prompt()
    def guide_spread_decomposition(dataset: str = "tick_data") -> list[dict]:
        """Guide: Huang-Stoll spread decomposition."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Spread decomposition on {dataset}:

1. Ensure {dataset} has bid, ask, close, and volume columns.
2. Call spread_decomposition(dataset="{dataset}").
3. Decomposes the spread into: adverse selection, inventory holding, and order processing.

Interpretation:
- High adverse selection component = informed traders dominate (toxic flow).
- High inventory component = market maker is absorbing imbalance.
- High order processing component = fixed costs (exchange fees, clearing).
- If adverse selection > 50% of spread: be cautious about market orders.
- Use to evaluate execution quality: am I paying more than the order processing cost?
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_price_impact(dataset: str = "trade_data") -> list[dict]:
        """Guide: Measuring permanent vs temporary price impact."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Price impact analysis on {dataset}:

1. Ensure {dataset} has close and volume columns.
2. Call price_impact(dataset="{dataset}").
3. The result measures how much prices move per unit of volume.

Interpretation:
- High mean_impact = trading moves prices significantly (illiquid market).
- If impact is asymmetric (buys > sells): there is demand pressure.
- Compare to Kyle lambda from liquidity_metrics -- they measure related concepts.
- Use price impact estimates to set optimal execution schedules.
- Impact > expected alpha from a trade = the trade is not worth executing.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_depth_analysis(dataset: str = "orderbook_data") -> list[dict]:
        """Guide: Order book depth imbalance analysis."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Depth analysis on {dataset}:

1. Ensure {dataset} has bid_depth and ask_depth columns.
2. Call depth_analysis(dataset="{dataset}").
3. Depth imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).

Interpretation:
- Imbalance > 0: more buying interest (bid-heavy book).
- Imbalance < 0: more selling interest (ask-heavy book).
- buy_pressure_pct shows what fraction of time the book was bid-heavy.
- Persistent imbalance predicts short-term price direction.
- Sudden imbalance shifts can signal large order flow or news.
""",
                },
            }
        ]

    # ── execution/ ───────────────────────────────────────────────────

    @mcp.prompt()
    def guide_transaction_cost_analysis(dataset: str = "market_data") -> list[dict]:
        """Guide: Post-trade TCA benchmarking."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Transaction Cost Analysis against {dataset}:

1. Prepare trades as JSON: '[{{"price": 150.5, "quantity": 1000, "timestamp": "2024-01-15T10:30:00"}}]'.
2. Call transaction_cost_analysis(trades_json="...", market_data_dataset="{dataset}").
3. TCA compares each trade's execution price to arrival price, VWAP, and close.

Interpretation:
- Arrival cost: execution price vs price when order was placed (true cost).
- VWAP cost: execution price vs day's VWAP (benchmark for algos).
- Close cost: execution price vs closing price (shows timing value).
- Negative cost = you got a better price than the benchmark (good execution).
- Positive cost = you paid more than the benchmark (slippage).
- Aggregate costs over time to evaluate broker/algo performance.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_close_auction() -> list[dict]:
        """Guide: Close auction allocation strategy."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
Close auction allocation:

1. Call close_auction(total_quantity=50000, close_volume_pct=0.2).
2. Splits the order between continuous-market trading and MOC (market-on-close).
3. close_volume_pct=0.2 means 20% of daily volume trades at the close (typical for liquid stocks).

Interpretation:
- MOC portion captures the close price exactly (no slippage vs close benchmark).
- Continuous portion can be executed via TWAP/VWAP throughout the day.
- Higher close_volume_pct = more allocation to MOC (more liquidity at close).
- Use when the close price is the benchmark (index rebalancing, fund NAV).
- If total_quantity > 5% of close volume: MOC may cause impact. Split across days.
""",
                },
            }
        ]

    # ── math/ ────────────────────────────────────────────────────────

    @mcp.prompt()
    def guide_systemic_risk(dataset: str = "multi_asset_returns") -> list[dict]:
        """Guide: Systemic risk scoring via Marginal Expected Shortfall."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Systemic risk on {dataset}:

1. Call systemic_risk(dataset="{dataset}").
2. Each asset gets a Marginal Expected Shortfall (MES) score.
3. MES measures how much each asset loses when the entire system is in distress.

Interpretation:
- Most systemic asset = highest MES score. It loses the most in market crashes.
- Least systemic asset = lowest MES. Best diversifier in crises.
- Underweight high-MES assets if you want crash protection.
- MES is forward-looking systemic risk -- not the same as beta (which is symmetric).
- Compare to beta_analysis: high beta + high MES = amplifies both up and down.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_math_hawkes_fit(dataset: str = "event_times") -> list[dict]:
        """Guide: Hawkes process fit for event arrival times."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Hawkes fit on {dataset}:

1. Ensure {dataset} has a time/timestamp column with event arrival times.
2. Call hawkes_fit(event_times_dataset="{dataset}", column="time").
3. Returns mu (baseline intensity), alpha (excitation), beta (decay).
4. Branching ratio = alpha/beta. Must be < 1 for stationarity.

Interpretation:
- mu: average event rate in the absence of self-excitation.
- alpha: how much each event boosts the intensity of future events.
- beta: how quickly the excitation decays.
- Branching ratio > 0.5 = strong clustering (events trigger events).
- Branching ratio near 1.0 = near-critical process (cascading events, flash crash risk).
- Use for modeling trade arrival times, default clustering, or volatility contagion.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_spectral_analysis(dataset: str = "prices") -> list[dict]:
        """Guide: FFT spectral analysis for cyclical patterns."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Spectral analysis on {dataset}:

1. Call spectral_analysis(dataset="{dataset}", column="close").
2. Returns dominant frequencies, their periods (in trading days), and spectral entropy.
3. Dominant frequencies reveal cyclical patterns invisible in the time domain.

Interpretation:
- Period ~ 252: annual seasonality. Period ~ 21: monthly cycle. Period ~ 5: weekly effect.
- High spectral entropy = no dominant frequency (noisy, random-walk-like).
- Low spectral entropy = strong cyclical component (tradeable pattern).
- If dominant_period ~ 63 (quarterly): earnings cycle or rebalancing effect.
- Use identified cycles to set trading horizons or seasonal strategy timing.
""",
                },
            }
        ]

    # ── experiment/ ──────────────────────────────────────────────────

    @mcp.prompt()
    def guide_experiment_comparison() -> list[dict]:
        """Guide: Comparing multiple strategy experiments."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
Experiment comparison:

1. Run multiple experiments first: create_experiment + run_experiment for each strategy variant.
2. Call experiment_comparison(names_json='["momentum_fast", "momentum_slow", "meanrev"]').
3. Returns side-by-side comparison of best params and metrics, plus a ranking.

Interpretation:
- Ranking is by best_sharpe across each experiment's optimal parameters.
- Check if the winning experiment's best_sharpe is significantly better (not just noise).
- Compare max_drawdown across experiments -- sometimes a lower-Sharpe strategy with smaller drawdowns is preferable.
- If all experiments have similar Sharpe: the strategy class matters more than the parameter choice.
- Always validate the winner with out-of-sample testing (walk_forward).
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_parameter_sensitivity(
        name: str = "my_experiment", param: str = "period"
    ) -> list[dict]:
        """Guide: Parameter sensitivity analysis for a strategy experiment."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Parameter sensitivity: experiment "{name}", parameter "{param}":

1. Call parameter_sensitivity(name="{name}", param_name="{param}").
2. Groups all experiment results by the chosen parameter value.
3. Shows mean Sharpe, std Sharpe, min/max Sharpe for each parameter value.

Interpretation:
- Low overall_sensitivity: performance is robust to this parameter (good).
- High overall_sensitivity: performance depends heavily on this parameter (bad -- likely overfit).
- If mean_sharpe is stable across values: parameter choice doesn't matter much. Pick the middle.
- If mean_sharpe varies wildly: the strategy is fragile. Consider a different approach.
- A parameter with monotonic Sharpe relationship is more trustworthy than one with random variation.
""",
                },
            }
        ]

    # ── workspace ────────────────────────────────────────────────────

    @mcp.prompt()
    def guide_snapshot() -> list[dict]:
        """Guide: Creating and restoring workspace snapshots."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
Workspace snapshots:

1. Before a risky operation, call snapshot(name="before_experiment").
2. This saves a copy of all datasets and models in the workspace.
3. If something goes wrong, call restore_snapshot(name="before_experiment").
4. Everything reverts to the snapshot state (datasets, models, DuckDB tables).

Use for:
- Before running destructive experiments (parameter sweeps that overwrite data).
- Before large data transformations (resampling, filtering).
- Creating "known good" checkpoints in a multi-step workflow.
- Comparing results before and after a change (snapshot, change, compare, restore).
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_restore_snapshot() -> list[dict]:
        """Guide: Restoring a workspace from a snapshot."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
Restore snapshot:

1. Call restore_snapshot(name="snapshot_name") to revert the workspace.
2. All datasets and models are replaced with the snapshot versions.
3. Any data created after the snapshot is lost.

When to restore:
- An experiment produced bad results and you want to try again.
- You accidentally deleted or overwrote a dataset.
- You want to compare current state to a previous state.
- Use workspace_status() after restore to verify the state.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_query_data() -> list[dict]:
        """Guide: SQL queries on workspace data via DuckDB."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
Query data with SQL:

1. Call query_data(sql="SELECT * FROM my_dataset LIMIT 10") to inspect data.
2. All datasets in the workspace are DuckDB tables -- full SQL available.
3. Use for filtering: query_data(sql="SELECT * FROM returns WHERE returns > 0.05").
4. Use for aggregation: query_data(sql="SELECT AVG(close) as mean_price FROM prices GROUP BY year").

Tips:
- SHOW TABLES to see all available datasets.
- DESCRIBE my_dataset to see column names and types.
- Use SQL for quick data exploration before running analysis tools.
- Results are limited to 50 rows in the response -- use LIMIT for larger queries.
- Only SELECT, SHOW, and DESCRIBE are allowed (read-only for safety).
""",
                },
            }
        ]

    # ── Core tools (server.py tier-2) ─────────────────────────────────

    @mcp.prompt()
    def guide_compute_returns(dataset: str = "prices") -> list[dict]:
        """Guide: Computing returns from price data."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Compute returns from {dataset}:

1. Call compute_returns(dataset="{dataset}", column="close", method="simple").
2. method="simple" gives (P_{{t}} - P_{{t-1}}) / P_{{t-1}}. method="log" gives ln(P_{{t}} / P_{{t-1}}).
3. Simple returns are more intuitive. Log returns are additive over time (better for modeling).
4. The result is stored as a new dataset in the workspace with a "returns" column.

When to use log vs simple:
- Log returns: GARCH fitting, regime detection, statistical tests, time series models.
- Simple returns: Sharpe ratio, portfolio returns (simple returns aggregate across assets).
- For short horizons (daily), the difference is negligible.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_risk_metrics(dataset: str = "returns") -> list[dict]:
        """Guide: Computing risk metrics (Sharpe, Sortino, drawdown)."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Risk metrics on {dataset}:

1. Call risk_metrics(dataset="{dataset}", column="returns", risk_free=0.0).
2. Key outputs: sharpe_ratio, sortino_ratio, max_drawdown, annual_vol, calmar_ratio.
3. Sharpe > 1.0 is good, > 2.0 is excellent. Sortino > Sharpe means positive skew (good).
4. Max drawdown: the worst peak-to-trough decline. Below -20% is painful.
5. Calmar = annual_return / max_drawdown. > 1.0 means you earned more than you lost at worst.

Interpretation:
- Compare Sharpe to Sortino: if Sortino >> Sharpe, downside risk is well-managed.
- Max drawdown should be paired with its duration — a -15% drawdown lasting 2 years is worse than -25% lasting 2 months.
- Use risk_free > 0 if analyzing excess returns vs a benchmark rate.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_var_analysis(dataset: str = "returns") -> list[dict]:
        """Guide: Value-at-Risk and CVaR analysis."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
VaR analysis on {dataset}:

1. Call var_analysis(dataset="{dataset}", column="returns", alpha=0.05, method="historical").
2. Methods: "historical" (empirical quantile), "parametric" (normal assumption), "cornish_fisher" (skew/kurtosis adjusted).
3. Historical VaR is most robust. Parametric understates risk for fat-tailed data.
4. CVaR (Expected Shortfall) = average loss in the tail beyond VaR. Always >= VaR.
5. If CVaR >> VaR (e.g., CVaR is 2x VaR), the tail is very fat — beware.

Regulatory context:
- Basel III prefers CVaR (ES) at 97.5% over VaR at 99%.
- Daily VaR * sqrt(10) ≈ 10-day VaR (assumes normal, conservative).
- Backtest VaR: count the days that breach VaR. Should be ~5% for 95% VaR.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_stress_test(dataset: str = "returns") -> list[dict]:
        """Guide: Stress testing against historical crisis scenarios."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Stress test on {dataset}:

1. Call stress_test(dataset="{dataset}", column="returns", scenarios="historical").
2. Tests against: GFC 2008, COVID 2020, Dot-com 2000, Taper Tantrum 2013, etc.
3. Each scenario applies historical drawdown shocks to your portfolio.
4. Key output: expected loss under each scenario, worst-case scenario, and recovery time.

Interpretation:
- If GFC loss would be > 30%, your portfolio has significant systematic risk.
- Compare COVID vs GFC losses — if similar magnitude, you're exposed to fat tails.
- Use custom scenarios: stress_test(..., custom_scenario={{"equity": -0.20, "vol": 2.0}}).
- Combine with var_analysis for a complete risk picture.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_fit_garch(dataset: str = "returns") -> list[dict]:
        """Guide: Fitting GARCH models for volatility dynamics."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Fit GARCH on {dataset}:

1. Call fit_garch(dataset="{dataset}", column="returns", model="GJR", p=1, q=1).
2. Models: "GARCH" (symmetric), "GJR" (leverage effect), "EGARCH" (log-vol), "FIGARCH" (long memory).
3. GJR is the default choice for equities — it captures the leverage effect (vol rises more after falls).
4. Key outputs: persistence (alpha + beta), omega, news impact curve shape.

Interpretation:
- Persistence > 0.98: vol shocks are very persistent, mean reversion is slow.
- Alpha/gamma ratio (GJR): gamma > 0 confirms leverage effect. gamma > alpha = strong asymmetry.
- AIC/BIC for model comparison — lower is better. Try model_selection to auto-compare.
- Use forecast_volatility after fitting to predict future vol.

Common workflow: fit_garch → forecast_volatility → var_analysis (GARCH-VaR).
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_detect_regimes(dataset: str = "returns") -> list[dict]:
        """Guide: Detecting market regimes via HMM."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Detect regimes in {dataset}:

1. Call detect_regimes(dataset="{dataset}", column="returns", n_regimes=2, method="hmm").
2. 2-state: bull (high-mean, low-vol) vs bear (low-mean, high-vol).
3. 3-state: add a crash/crisis state with very high vol and large negative mean.
4. Use select_n_states first if unsure how many regimes to use — it uses BIC.

Interpretation:
- Transition matrix: prob of staying in regime vs switching. p(bull→bull) > 0.95 is typical.
- Current state: what regime are we in now? Check regime_labels for human-readable names.
- Regime statistics: mean return, vol, Sharpe PER regime — is bull regime really better?
- Rolling probabilities: use rolling_regime_probability for time-varying regime confidence.

Follow-up: regime_statistics, regime_backtest (test regime-aware strategies).
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_correlation_analysis(dataset: str = "multi_asset_returns") -> list[dict]:
        """Guide: Correlation and diversification analysis."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Correlation analysis on {dataset}:

1. Call correlation_analysis(dataset="{dataset}", method="pearson").
2. Also try method="spearman" (rank-based, robust to outliers) or method="kendall" (concordance).
3. Key outputs: correlation matrix, eigenvalues, diversification_ratio.

Interpretation:
- Diversification ratio = portfolio vol / weighted avg vol. > 1.0 is good.
- If first eigenvalue explains > 60% of variance, there's a dominant factor (usually market beta).
- Look for negative correlations — these are your diversifiers.
- Compare to rolling_correlation to check stability. Correlations that spike in crises = bad.
- High correlation (> 0.8) between positions = concentrated risk, consider merging.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_cointegration_test(dataset: str = "prices") -> list[dict]:
        """Guide: Testing for cointegration (pairs trading prerequisite)."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Cointegration test on {dataset}:

1. Call cointegration_test(dataset="{dataset}", column1="asset_a", column2="asset_b").
2. Tests: Engle-Granger (2-step) and Johansen (multivariate).
3. If p-value < 0.05, the pair is cointegrated — their spread is mean-reverting.
4. The hedge ratio tells you how much of asset B to short per unit of asset A.

Interpretation:
- Cointegrated ≠ correlated. Correlation means they move together. Cointegration means their SPREAD is stationary.
- Hedge ratio near 1.0: dollar-neutral pair. Far from 1.0: one leg dominates.
- Hurst exponent < 0.5 on the spread confirms mean reversion.
- Use stationarity_tests on the spread to double-check.

Follow-up workflow: compute spread → stationarity_tests → z-score signals → run_backtest.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_optimize_portfolio(dataset: str = "multi_asset_returns") -> list[dict]:
        """Guide: Portfolio optimization (MVO, risk parity, HRP)."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Optimize portfolio from {dataset}:

1. Call optimize_portfolio(dataset="{dataset}", method="max_sharpe").
2. Methods: "max_sharpe", "min_volatility", "risk_parity", "hrp" (hierarchical risk parity).
3. MVO (max_sharpe/min_vol) is theoretically optimal but sensitive to estimation error.
4. Risk parity is more robust — equalizes risk contribution, doesn't need return estimates.
5. HRP uses hierarchical clustering — no matrix inversion, handles singular covariance.

Interpretation:
- Check for extreme weights (> 40% in one asset) — sign of estimation error in MVO.
- Compare Sharpe of optimized vs equal-weight. If similar, estimation error dominates.
- Use efficient_frontier to see the full risk-return tradeoff.
- For production: prefer risk_parity or HRP over MVO.

Follow-up: portfolio_risk for component VaR, rebalance_analysis for turnover.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_run_backtest(dataset: str = "prices") -> list[dict]:
        """Guide: Running a strategy backtest."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Backtest strategy on {dataset}:

1. Call run_backtest(dataset="{dataset}", signal_column="signal", price_column="close").
2. Signal column should contain: +1 (long), -1 (short), 0 (flat).
3. Use fractional signals (0.5 = half position) for position sizing.
4. Set initial_capital, commission, slippage for realistic simulation.

Interpretation:
- Call backtest_metrics after for Sharpe, max drawdown, win rate, profit factor.
- Call comprehensive_tearsheet for the full report.
- Use walk_forward for out-of-sample validation (critical for avoiding overfitting).
- Compare to benchmark: if strategy Sharpe < 0.5 and benchmark Sharpe ≈ 0.4, you're not adding value.

Pitfalls:
- Look-ahead bias: make sure signals only use past data.
- Survivorship bias: does your universe include delisted stocks?
- Transaction costs: unrealistic without commission + slippage.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_backtest_metrics(dataset: str = "backtest_results") -> list[dict]:
        """Guide: Interpreting backtest performance metrics."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Backtest metrics for {dataset}:

1. Call backtest_metrics(dataset="{dataset}").
2. Key metrics and benchmarks:
   - Sharpe > 1.0 is good (after costs). > 2.0 is excellent.
   - Max drawdown: < 15% for conservative, < 30% for aggressive.
   - Win rate: > 50% for trend following, > 65% for mean reversion.
   - Profit factor: > 1.5 is good (gross profit / gross loss).
   - Calmar ratio: > 1.0 (annual return / max drawdown).
   - Sortino > Sharpe: good sign (positive skew).

Red flags:
- Sharpe > 3.0 in daily data: likely overfitting or look-ahead bias.
- Win rate > 80%: probably a martingale strategy (blows up eventually).
- Max drawdown duration > 1 year: hard to stick with psychologically.
- Turnover > 200%/year with < 1% alpha: costs eat the edge.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_walk_forward(dataset: str = "prices") -> list[dict]:
        """Guide: Walk-forward analysis for out-of-sample validation."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Walk-forward analysis on {dataset}:

1. Call walk_forward(dataset="{dataset}", train_pct=0.7, n_splits=5).
2. Splits data into rolling train/test windows. Strategy is re-trained on each train window.
3. Out-of-sample Sharpe is the REAL performance (in-sample Sharpe is inflated by overfitting).

Interpretation:
- If OOS Sharpe < 50% of IS Sharpe, you're likely overfitting.
- Consistency across folds: if Sharpe varies wildly (0.5 to 2.0), the edge is unstable.
- Walk-forward is MORE reliable than single train/test split.
- Use with ML models: walk_forward_ml for automatic feature re-training.

Rule of thumb: If OOS Sharpe > 0.5 consistently across 5+ folds, the strategy has real edge.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_comprehensive_tearsheet(dataset: str = "backtest_results") -> list[dict]:
        """Guide: Generating a comprehensive performance tearsheet."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Tearsheet for {dataset}:

1. Call comprehensive_tearsheet(dataset="{dataset}").
2. Produces: performance summary, monthly returns, drawdown analysis, rolling metrics.
3. Monthly returns table: rows=years, columns=months. Color code: green>0, red<0.
4. Top 5 drawdowns: date, depth, recovery time. Shows tail risk profile.
5. Rolling Sharpe/vol: shows how the strategy performs in different market conditions.

How to read:
- Is the equity curve smooth or jagged? Smooth = consistent edge. Jagged = regime-dependent.
- Are drawdowns clustered? Clustered drawdowns = regime vulnerability.
- Is rolling Sharpe stable? If it degrades over time, the edge may be decaying.
- Compare to strategy_comparison for relative performance vs alternatives.
""",
                },
            }
        ]

    # ── risk/ tools ───────────────────────────────────────────────────

    @mcp.prompt()
    def guide_beta_analysis(dataset: str = "returns") -> list[dict]:
        """Guide: Beta estimation and decomposition."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Beta analysis on {dataset}:

1. Call beta_analysis(dataset="{dataset}", column="returns", benchmark_dataset="benchmark").
2. Outputs: beta, upside_beta, downside_beta, alpha (Jensen's), R-squared.
3. Beta = 1.0: moves with the market. > 1.0: amplifies market moves. < 1.0: defensive.

Interpretation:
- If downside_beta > upside_beta: the asset falls more than it rises with the market (BAD).
- If upside_beta > downside_beta: asymmetric upside exposure (GOOD).
- Alpha > 0: positive excess return after adjusting for market risk.
- R² < 0.3: market explains little of the variance — idiosyncratic risk dominates.
- Use rolling_beta to check if beta is stable or time-varying.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_factor_analysis(dataset: str = "returns") -> list[dict]:
        """Guide: Multi-factor risk decomposition."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Factor analysis on {dataset}:

1. Call factor_analysis(dataset="{dataset}", column="returns", factors_dataset="factors").
2. Factors: market, size (SMB), value (HML), momentum (UMD), quality, low_vol.
3. Outputs: factor betas, R², alpha, factor contributions to risk.

Interpretation:
- Alpha (intercept): the return NOT explained by factors. Positive alpha = genuine skill.
- Factor loadings: which systematic risks are you exposed to?
- Risk decomposition: how much of your vol comes from each factor?
- If market factor explains > 80% of risk, you're basically an expensive index fund.
- Use with crisis_drawdowns to see how factor exposures change in stress.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_crisis_drawdowns(dataset: str = "returns") -> list[dict]:
        """Guide: Historical crisis drawdown analysis."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Crisis drawdowns for {dataset}:

1. Call crisis_drawdowns(dataset="{dataset}", column="returns").
2. Identifies the top 5-10 worst drawdowns with dates, depth, duration, and recovery time.
3. Compares your drawdown profile to major market events.

Interpretation:
- Depth: how far did you fall? < -10% is mild, -20% moderate, -30%+ severe.
- Duration: how long from peak to trough? Longer is psychologically harder.
- Recovery: how long to get back to the previous high water mark?
- Underwater period = duration + recovery. This is what investors actually feel.
- If your worst drawdown aligns with GFC/COVID, you have systematic risk.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_portfolio_risk(dataset: str = "multi_asset_returns") -> list[dict]:
        """Guide: Portfolio-level risk decomposition."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Portfolio risk for {dataset}:

1. Call portfolio_risk(dataset="{dataset}", weights=[0.6, 0.4]).
2. Outputs: portfolio_vol, component_var, marginal_var, diversification_ratio.

Interpretation:
- Component VaR: each asset's contribution to total portfolio VaR. Sum = total VaR.
- If one asset contributes > 50% of VaR, the portfolio is concentrated.
- Marginal VaR: how much VaR changes if you add 1% more of this asset.
- Diversification ratio > 1.0: you're benefiting from diversification.
- Use with optimize_portfolio to find risk-parity weights (equal risk contribution).
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_tail_risk(dataset: str = "returns") -> list[dict]:
        """Guide: Tail risk analysis (EVT, tail dependence)."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Tail risk analysis on {dataset}:

1. Call tail_risk(dataset="{dataset}", column="returns").
2. Uses Extreme Value Theory (EVT) to model the tail distribution.
3. Outputs: tail index (xi), VaR at extreme quantiles, expected shortfall.

Interpretation:
- Tail index (xi) > 0: heavy tails (fat-tailed, Pareto-like). Higher xi = fatter tails.
- xi ≈ 0: exponential tails (roughly normal). xi < 0: bounded tails (rare in finance).
- EVT-based VaR is more accurate for extreme quantiles (99.9%) than parametric.
- Compare to normal VaR: if EVT VaR >> normal VaR, your risk is underestimated.
""",
                },
            }
        ]

    # ── vol/ tools ────────────────────────────────────────────────────

    @mcp.prompt()
    def guide_forecast_volatility(dataset: str = "returns") -> list[dict]:
        """Guide: Forecasting volatility with GARCH models."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Forecast vol on {dataset}:

1. First call fit_garch(dataset="{dataset}") to get a fitted model.
2. Then forecast_volatility(dataset="{dataset}", horizon=10) for 10-day forecast.
3. Output: daily vol forecasts, cumulative vol over horizon, confidence intervals.

Interpretation:
- If forecasted vol > realized vol: market expects turbulence (or GARCH captures recent shock).
- Vol forecasts mean-revert to long-run vol. Speed depends on persistence.
- Persistence > 0.98: vol shocks die slowly (~50 days to half-life).
- Use for: VaR scaling, position sizing, option pricing (term structure of vol).

Compare to: realized_volatility for backward-looking, news_impact_curve for shock response.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_model_selection(dataset: str = "returns") -> list[dict]:
        """Guide: GARCH model selection and comparison."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Model selection on {dataset}:

1. Call model_selection(dataset="{dataset}", column="returns").
2. Compares: GARCH(1,1), GJR-GARCH, EGARCH, TGARCH, FIGARCH.
3. Ranks by AIC/BIC. Lower is better. BIC penalizes complexity more.

Interpretation:
- If GJR or EGARCH wins over GARCH: leverage effect is significant (common for equities).
- If FIGARCH wins: long-memory vol (common for indices, FX).
- If plain GARCH wins: the data is simple — don't over-complicate.
- After selection, use the winning model for forecast_volatility and VaR.
- Report AIC difference: ΔAIC < 2 = models are essentially equivalent.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_realized_volatility(dataset: str = "prices") -> list[dict]:
        """Guide: Realized volatility estimation from price data."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Realized volatility on {dataset}:

1. Call realized_volatility(dataset="{dataset}", method="yang_zhang", window=21).
2. Methods: "close_to_close" (simple), "parkinson" (high-low), "garman_klass" (OHLC), "yang_zhang" (OHLC, most efficient).
3. Yang-Zhang uses open, high, low, close — 8x more efficient than close-to-close.
4. Window=21 ≈ 1 trading month. Window=63 ≈ 1 quarter.

Interpretation:
- RV is backward-looking (what vol WAS), vs GARCH which forecasts future vol.
- Compare RV to implied vol: IV > RV means options are expensive (VRP is positive).
- Rolling RV: is vol trending up or down? Check regime context.
- Use bipower_variation to separate continuous vol from jumps.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_news_impact_curve(dataset: str = "returns") -> list[dict]:
        """Guide: News impact curve — asymmetric volatility response."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
News impact curve for {dataset}:

1. First fit_garch(dataset="{dataset}", model="GJR") to get a fitted model.
2. Then news_impact_curve(dataset="{dataset}").
3. Shows how vol responds to positive vs negative shocks of the same size.

Interpretation:
- Symmetric curve = GARCH (no leverage). Steeper left side = leverage effect (GJR/EGARCH).
- The "leverage effect": a -2% shock increases vol more than a +2% shock.
- Steepness ratio: how much more does a negative shock increase vol vs positive?
- In equities, -1σ shock typically increases vol 1.5-2x more than +1σ shock.
- Useful for options: asymmetry explains the volatility skew.
""",
                },
            }
        ]

    # ── ts/ tools ─────────────────────────────────────────────────────

    @mcp.prompt()
    def guide_forecast(dataset: str = "prices") -> list[dict]:
        """Guide: Time series forecasting."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Forecast on {dataset}:

1. Call forecast(dataset="{dataset}", column="close", horizon=30, method="auto").
2. method="auto" tries ARIMA, ETS, Theta, and picks the best by AIC.
3. Also available: "arima", "ets", "theta", "prophet" (if installed).

Interpretation:
- Forecast confidence intervals widen with horizon — uncertainty grows.
- Point forecasts > 5 days out are unreliable for financial prices (EMH).
- Forecasts are more useful for: vol (persistent), macro (trending), spreads (mean-reverting).
- For trading signals: compare forecast direction to current trend.
- Use forecast_evaluation to measure RMSE, MAE, MAPE on holdout.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_decompose(dataset: str = "prices") -> list[dict]:
        """Guide: Time series decomposition (trend + seasonal + residual)."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Decompose {dataset}:

1. Call decompose(dataset="{dataset}", column="close", method="stl").
2. Methods: "classical" (additive/multiplicative), "stl" (STL Loess), "ssa" (spectral).
3. STL is most robust. Handles changing seasonality.

Interpretation:
- Trend: the underlying direction. Is it accelerating or decelerating?
- Seasonal: recurring patterns. Monthly options expiry, quarterly earnings, day-of-week.
- Residual: what's left. Should be stationary if decomposition is good.
- If residual has structure → the decomposition missed something.
- Use ssa_decompose for non-parametric decomposition (doesn't assume seasonal period).
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_stationarity_tests(dataset: str = "returns") -> list[dict]:
        """Guide: Testing for stationarity (ADF, KPSS, PP)."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Stationarity tests on {dataset}:

1. Call stationarity_tests(dataset="{dataset}", column="returns").
2. Runs ADF (null: unit root), KPSS (null: stationary), Phillips-Perron.
3. If ADF p < 0.05 AND KPSS p > 0.05: series IS stationary.
4. If ADF p > 0.05 AND KPSS p < 0.05: series is NOT stationary → difference it.

Interpretation:
- Prices: almost always non-stationary (random walk). Returns: almost always stationary.
- Spread between cointegrated pairs: should be stationary.
- Non-stationary data cannot be used for: regression (spurious), ARMA, mean-reversion strategies.
- If borderline: use changepoint_detect to check if the stationarity changed at some point.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_changepoint_detect(dataset: str = "returns") -> list[dict]:
        """Guide: Detecting structural changepoints in time series."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Changepoint detection on {dataset}:

1. Call changepoint_detect(dataset="{dataset}", column="returns", method="pelt").
2. Methods: "pelt" (fast, penalized), "binseg" (binary segmentation), "bayesian".
3. Detects changes in mean, variance, or both.

Interpretation:
- Each changepoint marks a structural break — the data-generating process changed.
- Common causes: regime shifts, policy changes, market structure changes.
- Use bayesian method for uncertainty on changepoint locations.
- If many changepoints detected: lower penalty for more sensitivity, raise for fewer.
- Pair with detect_regimes for complementary analysis (HMM = soft regimes, changepoints = hard breaks).
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_anomaly_detect(dataset: str = "returns") -> list[dict]:
        """Guide: Detecting anomalous observations."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Anomaly detection on {dataset}:

1. Call anomaly_detect(dataset="{dataset}", column="returns", method="zscore").
2. Methods: "zscore" (> 3σ), "iqr" (outside 1.5×IQR), "isolation_forest" (ML-based).
3. Isolation forest is best for multivariate anomaly detection.

Interpretation:
- Z-score anomalies: simple, assumes normality. Good for quick screening.
- IQR anomalies: robust to non-normality. Use for heavy-tailed returns.
- Isolation forest: finds anomalies in feature space. Best for multivariate data.
- Decide: are anomalies errors (fix) or signals (trade)?
- Flash crashes, earnings surprises, and circuit breakers all show as anomalies.
""",
                },
            }
        ]

    # ── stats/ tools ──────────────────────────────────────────────────

    @mcp.prompt()
    def guide_regression(dataset: str = "returns") -> list[dict]:
        """Guide: Linear regression analysis."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Regression on {dataset}:

1. Call regression(dataset="{dataset}", y_column="returns", x_columns=["factor1", "factor2"]).
2. Outputs: coefficients, t-stats, p-values, R², adjusted R², F-stat, residuals.

Interpretation:
- R² > 0.1 is noteworthy for daily returns (most of the variation is noise).
- Coefficients: beta to each factor. p < 0.05 = statistically significant.
- Check residuals: should be white noise. If autocorrelated → the model is mis-specified.
- Compare adjusted R² (penalizes adding useless variables) to R².
- Use robust_stats for heteroscedasticity-robust standard errors (Newey-West).
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_distribution_fit(dataset: str = "returns") -> list[dict]:
        """Guide: Fitting statistical distributions to returns."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Distribution fit on {dataset}:

1. Call distribution_fit(dataset="{dataset}", column="returns").
2. Fits: normal, student-t, skew-normal, generalized hyperbolic, stable.
3. Ranked by AIC/BIC and Kolmogorov-Smirnov test.

Interpretation:
- Normal almost never wins for financial returns (fat tails, skew).
- Student-t with df=4-6 is typical for equities (heavy tails).
- Skew < 0: negative skew (left tail is fatter, crashes more likely).
- Excess kurtosis > 3: fat tails, VaR underestimation with normal assumption.
- The winning distribution should be used for parametric VaR.
""",
                },
            }
        ]

    # ── ml/ tools ─────────────────────────────────────────────────────

    @mcp.prompt()
    def guide_build_features(dataset: str = "prices") -> list[dict]:
        """Guide: Building ML features from financial data."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Build ML features from {dataset}:

1. Call build_features(dataset="{dataset}", feature_sets=["returns", "ta", "vol"]).
2. Feature sets: "returns" (lagged returns), "ta" (TA indicators), "vol" (GARCH vol), "regime" (regime state).
3. Target: next-day return direction (classification) or magnitude (regression).

Best practices:
- Feature importance FIRST: don't use 200 features blindly. Use feature_importance to prune.
- Cross-validate: walk_forward_ml to avoid look-ahead bias.
- Standard scale all features before training (except tree-based models).
- Avoid future data leaking into features (e.g., forward-looking indicators).
- 20-50 features is a good range. More = overfitting risk.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_train_model(dataset: str = "features") -> list[dict]:
        """Guide: Training ML models for alpha prediction."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Train model on {dataset}:

1. Call train_model(dataset="{dataset}", target="direction", model="gradient_boost").
2. Models: "gradient_boost" (XGBoost), "random_forest", "svm", "lasso", "ridge", "lstm" (deep).
3. Gradient boosting is the default choice — best for tabular financial data.

Interpretation:
- Accuracy > 52% for daily direction = potentially tradeable (after costs).
- Feature importance: which features actually predict? Drop irrelevant ones.
- Look at confusion matrix: is the model biased toward one class?
- Validation MUST be walk-forward (not random split) — financial data has temporal structure.
- After training, use run_backtest to test the predictions as trading signals.

Warning: In-sample accuracy >> out-of-sample accuracy = overfitting. Fix with regularization or fewer features.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_feature_importance(dataset: str = "features") -> list[dict]:
        """Guide: Feature importance and selection."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Feature importance for {dataset}:

1. Call feature_importance(dataset="{dataset}", target="direction", method="gradient_boost").
2. Methods: "gradient_boost" (impurity), "permutation" (model-agnostic), "shap" (Shapley values).
3. Permutation importance is most reliable — measures actual prediction impact.

Interpretation:
- Top 5-10 features carry most of the signal. The rest are noise.
- If a feature's importance is unstable across folds → it's likely noise.
- Correlated features split importance — use partial_correlation to identify groups.
- Remove features with < 1% importance to reduce overfitting.
- SHAP values show per-prediction feature contributions (local interpretability).
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_walk_forward_ml(dataset: str = "features") -> list[dict]:
        """Guide: Walk-forward ML with automatic retraining."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Walk-forward ML on {dataset}:

1. Call walk_forward_ml(dataset="{dataset}", target="direction", model="gradient_boost", n_splits=10).
2. Automatically retrains model at each step on expanding or rolling window.
3. Produces out-of-sample predictions for the full test period.

Interpretation:
- OOS accuracy is the REAL performance. IS accuracy is inflated.
- If OOS accuracy degrades over time: the relationship is non-stationary.
- Compare to baseline: always predict +1 (long only). Does ML add value?
- If OOS Sharpe < 0.3 on the ML signal, the model isn't worth the complexity.
- Use predictions as signals in run_backtest for final evaluation.
""",
                },
            }
        ]

    # ── price/ tools ──────────────────────────────────────────────────

    @mcp.prompt()
    def guide_price_option() -> list[dict]:
        """Guide: Pricing options with Black-Scholes."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
Price an option:

1. Call price_option(S=100, K=105, T=0.25, r=0.05, sigma=0.2, option_type="call").
2. S=spot, K=strike, T=time to expiry (years), r=risk-free rate, sigma=implied vol.
3. Outputs: price, delta, gamma, theta, vega, rho.

Interpretation:
- ITM call (S > K): price ≈ intrinsic value + time value. Delta near 1.0.
- ATM (S ≈ K): delta ≈ 0.5. Highest gamma and vega.
- OTM (S < K): price ≈ time value only. Delta near 0. Cheap but low probability.
- Theta is negative: options lose value daily. ATM has highest theta decay.
- Vega: price sensitivity to vol. ATM long-dated has highest vega.

Limitations: BS assumes constant vol and lognormal prices. Use simulate_heston for vol smile.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_compute_greeks() -> list[dict]:
        """Guide: Computing and interpreting option Greeks."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
Compute Greeks:

1. Call compute_greeks(S=100, K=105, T=0.25, r=0.05, sigma=0.2).
2. Delta: price change per $1 move in underlying. Call delta ∈ [0,1], put ∈ [-1,0].
3. Gamma: delta change per $1 move. Highest ATM. Measures convexity.
4. Theta: daily time decay. Negative for long options. Accelerates near expiry.
5. Vega: price change per 1% vol move. Highest for ATM, long-dated.
6. Rho: price change per 1% rate move. Small for short-dated options.

Portfolio Greeks:
- Portfolio delta = sum(position × delta). Delta-neutral = hedged against small moves.
- Portfolio gamma > 0: you benefit from large moves (long gamma). < 0: large moves hurt.
- Portfolio theta < 0 and gamma > 0: you're paying for convexity (long options).
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_simulate_process() -> list[dict]:
        """Guide: Simulating stochastic processes (GBM, Heston, jump-diffusion)."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
Simulate stochastic process:

1. Call simulate_process(model="gbm", S0=100, T=1.0, n_steps=252, n_paths=1000, params={}).
2. Models: "gbm" (Black-Scholes), "heston" (stochastic vol), "merton" (jump-diffusion),
   "variance_gamma", "cgmy" (Lévy).
3. GBM: simplest, constant vol. Heston: realistic vol clustering. Merton: captures jumps.

Use cases:
- Monte Carlo option pricing: simulate_process → compute option payoff → average.
- Risk analysis: simulate 10,000 paths → compute VaR from simulated returns.
- Strategy stress testing: test your strategy on simulated paths with different vol/jump params.
- Model calibration: compare simulated return distribution to historical.
""",
                },
            }
        ]

    # ── backtest/ tools ───────────────────────────────────────────────

    @mcp.prompt()
    def guide_strategy_comparison(dataset: str = "prices") -> list[dict]:
        """Guide: Comparing multiple strategies side-by-side."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Compare strategies on {dataset}:

1. Call strategy_comparison(dataset="{dataset}", strategies=["momentum", "mean_reversion", "buy_hold"]).
2. Each strategy is backtested on the same data for fair comparison.
3. Outputs: Sharpe, max DD, Calmar, win rate, profit factor for each.

Interpretation:
- Sharpe comparison: which strategy has the best risk-adjusted return?
- Correlation between strategies: low correlation = combine them for diversification.
- Max drawdown: which strategy is more painful to hold?
- Turnover: high-turnover strategies need strong alpha to cover costs.
- Use walk_forward for each strategy to ensure OOS robustness.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_efficient_frontier(dataset: str = "multi_asset_returns") -> list[dict]:
        """Guide: Computing and interpreting the efficient frontier."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Efficient frontier for {dataset}:

1. Call efficient_frontier(dataset="{dataset}", n_points=50).
2. Plots the optimal risk-return tradeoff for different target returns.
3. Key points: min_volatility portfolio, max_sharpe portfolio, your current portfolio.

Interpretation:
- Portfolios below the frontier are inefficient — same return with less risk exists.
- The tangent line from the risk-free rate touches the max_sharpe portfolio.
- If your current portfolio is far below the frontier → your allocation is suboptimal.
- Caveat: the frontier is estimated from historical data. Out-of-sample, it shifts.
- Use with black_litterman views to tilt the frontier toward your expectations.
""",
                },
            }
        ]

    # ── causal/ tools ─────────────────────────────────────────────────

    @mcp.prompt()
    def guide_granger_causality(dataset: str = "multi_asset_returns") -> list[dict]:
        """Guide: Granger causality testing."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Granger causality on {dataset}:

1. Call granger_causality(dataset="{dataset}", x_column="asset_a", y_column="asset_b", max_lags=5).
2. Tests: does past asset_a predict future asset_b (beyond what asset_b predicts itself)?
3. p < 0.05: asset_a Granger-causes asset_b. Can be unidirectional or bidirectional.

Important caveats:
- Granger causality ≠ true causality. It's predictive, not causal.
- Both series must be stationary. Use stationarity_tests first.
- Spurious results if both are driven by a third variable (confounding).
- Useful for: lead-lag relationships, information flow between markets.
- Follow up with diff_in_diff or synthetic_control for actual causal inference.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_event_study(dataset: str = "returns") -> list[dict]:
        """Guide: Event study for abnormal return analysis."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Event study on {dataset}:

1. Call event_study(dataset="{dataset}", column="returns", event_dates=["2024-01-15"], window=(-5,10)).
2. Computes cumulative abnormal returns (CAR) around each event date.
3. Window=(-5,10) means 5 days before to 10 days after the event.

Interpretation:
- CAR > 0: positive abnormal return around the event (good news was priced in).
- Pre-event drift: if CAR starts rising before event date → information leakage.
- Post-event drift: if CAR continues after event → slow price adjustment (market inefficiency).
- T-statistic: is the CAR statistically significant? t > 2 ≈ p < 0.05.
- Multiple events: average CARs across events for a more robust estimate.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_diff_in_diff(dataset: str = "panel_data") -> list[dict]:
        """Guide: Difference-in-differences causal analysis."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Diff-in-diff on {dataset}:

1. Call diff_in_diff(dataset="{dataset}", outcome_col="returns", treatment_col="treated", time_col="post").
2. Needs treatment group, control group, pre-treatment and post-treatment periods.
3. The DID estimate = (treated_post - treated_pre) - (control_post - control_pre).

Interpretation:
- DID coefficient: the CAUSAL effect of treatment, controlling for group and time trends.
- Parallel trends assumption: treated and control must have similar trends BEFORE treatment.
- If parallel trends violated, DID is biased. Check with a pre-treatment trend test.
- Use synthetic_control as a robustness check — it constructs a better counterfactual.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_synthetic_control(dataset: str = "panel_data") -> list[dict]:
        """Guide: Synthetic control method for counterfactual estimation."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Synthetic control on {dataset}:

1. Call synthetic_control(dataset="{dataset}", treated="treated_unit", control_pool=["c1","c2","c3"]).
2. Constructs a weighted average of control units that best mimics the treated unit pre-treatment.
3. The treatment effect = treated - synthetic counterfactual.

Interpretation:
- Pre-treatment fit: if RMSPE is high, the synthetic control is a poor match.
- Post-treatment gap: the difference is the causal effect.
- Placebo tests: run the same analysis on each control unit. If treated gap is largest = significant.
- Weights: which control units contribute most to the synthetic? Make sure they make sense.
""",
                },
            }
        ]

    # ── bayes/ tools ──────────────────────────────────────────────────

    @mcp.prompt()
    def guide_bayesian_sharpe(dataset: str = "returns") -> list[dict]:
        """Guide: Bayesian Sharpe ratio estimation."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Bayesian Sharpe on {dataset}:

1. Call bayesian_sharpe(dataset="{dataset}", column="returns").
2. Gives a POSTERIOR DISTRIBUTION of the Sharpe ratio, not just a point estimate.
3. Key output: posterior mean, 95% credible interval, P(Sharpe > 0).

Interpretation:
- If 95% CI includes 0: insufficient evidence that the strategy is profitable.
- P(Sharpe > 0) > 95%: strong evidence of positive risk-adjusted returns.
- The width of the CI shows estimation uncertainty — wider = less data or more noisy.
- Bayesian Sharpe is more conservative than classical (smaller samples → wider intervals).
- Especially useful for short track records (< 3 years of data).
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_bayesian_regression(dataset: str = "returns") -> list[dict]:
        """Guide: Bayesian regression with uncertainty quantification."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Bayesian regression on {dataset}:

1. Call bayesian_regression(dataset="{dataset}", y_column="returns", x_columns=["factor1"]).
2. Gives posterior distributions for each coefficient (not just point estimates).
3. Outputs: posterior means, credible intervals, predictive distribution, Rhat convergence.

Interpretation:
- Coefficient 95% CI not including 0: the factor is a significant predictor.
- Posterior predictive: distribution of future observations (wider than CI for coefficients).
- Rhat near 1.0: MCMC has converged. Rhat > 1.1: run more samples.
- Compare to frequentist regression: Bayesian with informative priors can be more robust.
- Use model_comparison_bayesian for WAIC/LOO comparison between competing models.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_bayesian_changepoint(dataset: str = "returns") -> list[dict]:
        """Guide: Bayesian changepoint detection with uncertainty."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Bayesian changepoint on {dataset}:

1. Call bayesian_changepoint(dataset="{dataset}", column="returns").
2. Gives a POSTERIOR probability of a changepoint at each time point.
3. Unlike frequentist methods, you get uncertainty on changepoint locations.

Interpretation:
- High posterior probability at a date: strong evidence of a structural break.
- Multiple modes in the posterior: uncertainty about exact changepoint date.
- Number of changepoints is also inferred from the data.
- Use alongside detect_regimes — regimes are soft/recurring, changepoints are hard/permanent.
""",
                },
            }
        ]

    # ── forex/ tools ──────────────────────────────────────────────────

    @mcp.prompt()
    def guide_carry_analysis(dataset: str = "fx_data") -> list[dict]:
        """Guide: FX carry trade analysis."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Carry analysis on {dataset}:

1. Call carry_analysis(dataset="{dataset}").
2. Carry = interest rate differential between two currencies.
3. Positive carry: you earn by holding the higher-yielding currency.
4. Output: carry yield, annualized return, carry-to-risk ratio.

Interpretation:
- High carry currencies: AUD, NZD, BRL. Low carry: JPY, CHF, EUR.
- Carry works most of the time but crashes during risk-off events (carry unwind).
- Carry-to-risk ratio > 1.0: carry exceeds expected vol. Good.
- Use detect_regimes to identify carry-friendly vs carry-crash regimes.
- Combine with fx_risk for hedging analysis.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_currency_strength(dataset: str = "fx_data") -> list[dict]:
        """Guide: Currency strength ranking."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Currency strength on {dataset}:

1. Call currency_strength(dataset="{dataset}").
2. Ranks currencies by their strength across multiple pairs.
3. A strong currency appreciates against most other currencies.

Interpretation:
- Buy strong / sell weak: momentum strategy. Works in trending markets.
- Mean reversion: fading extremes. Works when strength is extended.
- USD strength typically correlates with risk-off environments.
- Compare to carry: high carry + weak = potential bounce. High carry + strong = ideal.
- Use with session_info for optimal execution timing.
""",
                },
            }
        ]

    # ── microstructure/ tools ─────────────────────────────────────────

    @mcp.prompt()
    def guide_liquidity_metrics(dataset: str = "prices") -> list[dict]:
        """Guide: Market liquidity analysis."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Liquidity metrics on {dataset}:

1. Call liquidity_metrics(dataset="{dataset}").
2. Outputs: Amihud illiquidity, bid-ask spread, Kyle's lambda, market depth.

Interpretation:
- Amihud ratio: price impact per unit of volume. Higher = less liquid.
- Kyle's lambda: permanent price impact. Higher = more informed trading.
- Bid-ask spread: direct transaction cost. Wider = less liquid.
- Market depth: volume available at best prices.
- Liquidity varies by time of day (U-shaped: high at open/close, low midday).
- Use for: execution cost estimation, determining appropriate position sizes.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_toxicity_analysis(dataset: str = "prices") -> list[dict]:
        """Guide: Order flow toxicity (VPIN) analysis."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Toxicity analysis on {dataset}:

1. Call toxicity_analysis(dataset="{dataset}").
2. VPIN (Volume-Synchronized Probability of Informed Trading): fraction of volume from informed traders.
3. VPIN > 0.7: high toxicity, informed traders are active. Market makers widen spreads.

Interpretation:
- Rising VPIN before events: informed traders know something. Potential front-running.
- VPIN spikes precede flash crashes — useful as an early warning signal.
- Order flow imbalance: buy vs sell pressure. Persistent imbalance = directional flow.
- Combine with liquidity_metrics: high VPIN + low liquidity = dangerous (adverse selection).
- Use for: execution timing (avoid high VPIN periods), market-making strategy signals.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_market_quality(dataset: str = "prices") -> list[dict]:
        """Guide: Market quality and efficiency metrics."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Market quality on {dataset}:

1. Call market_quality(dataset="{dataset}").
2. Outputs: variance ratio, autocorrelation, efficiency ratio, price discovery metrics.

Interpretation:
- Variance ratio = 1.0: random walk (efficient). > 1.0: trending. < 1.0: mean-reverting.
- Autocorrelation at lag 1: positive = momentum, negative = reversal.
- Efficiency ratio close to 1.0: prices quickly incorporate information.
- Low efficiency = potential trading opportunity (but also more noise).
- Compare across assets or time periods to identify changing market structure.
""",
                },
            }
        ]

    # ── execution/ tools ──────────────────────────────────────────────

    @mcp.prompt()
    def guide_almgren_chriss() -> list[dict]:
        """Guide: Almgren-Chriss optimal execution framework."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
Almgren-Chriss execution:

1. Call almgren_chriss(total_shares=100000, daily_volume=1000000, spread=0.01, volatility=0.02, risk_aversion=1e-6).
2. Minimizes expected cost + risk penalty over the execution horizon.
3. Higher risk_aversion → front-load execution (certainty preferred over optimal cost).

Interpretation:
- Optimal trajectory: how much to trade each interval. Usually front-loaded.
- Expected cost: spread + temporary impact + permanent impact.
- Trade-off: slower execution = lower impact but more timing risk (price drift).
- Urgency parameter: high urgency → TWAP-like (uniform). Low urgency → VWAP-like.
- Use for: large institutional orders (> 5% ADV), illiquid securities.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_execution_cost(dataset: str = "prices") -> list[dict]:
        """Guide: Pre-trade execution cost estimation."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Execution cost on {dataset}:

1. Call execution_cost(dataset="{dataset}", quantity=10000).
2. Estimates: spread cost, market impact cost, timing risk, total expected cost.

Interpretation:
- Spread cost: half the bid-ask spread × quantity. The minimum you'll pay.
- Market impact: additional cost from moving the market with your order. Scales with sqrt(quantity/ADV).
- Timing risk: cost uncertainty from price moves during execution. Higher for volatile assets.
- Total cost typically = 5-30 bps for liquid equities, 50-200 bps for illiquid.
- If estimated cost > expected alpha of the trade → don't trade.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_optimal_schedule(dataset: str = "prices") -> list[dict]:
        """Guide: Generating optimal execution schedules."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Optimal schedule for {dataset}:

1. Call optimal_schedule(dataset="{dataset}", total_quantity=50000, method="twap").
2. Methods: "twap" (uniform), "vwap" (volume-weighted), "is" (implementation shortfall).
3. TWAP: simplest, spreads order evenly. Good for low-urgency orders.
4. VWAP: trades more when volume is high. Minimizes market impact.
5. IS: balances impact vs risk. Best for alpha-driven trades.

When to use which:
- TWAP: benchmark execution, no strong view on timing.
- VWAP: client benchmark is VWAP, or you want to minimize footprint.
- IS: you have alpha that decays — need to trade fast but not too fast.
""",
                },
            }
        ]

    # ── data/ tools ───────────────────────────────────────────────────

    @mcp.prompt()
    def guide_fetch_yahoo(ticker: str = "AAPL") -> list[dict]:
        """Guide: Fetching market data from Yahoo Finance."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Fetch {ticker} data:

1. Call fetch_yahoo(ticker="{ticker}", start="2020-01-01", end="2024-01-01").
2. Returns OHLCV data stored in the workspace as a dataset.
3. Use fetch_ohlcv for other providers or more control.

After fetching:
- compute_returns to get return series.
- describe_dataset to inspect what you got.
- clean_dataset to handle missing values and outliers.
- Then proceed with analysis (risk_metrics, fit_garch, etc.).

Tips:
- Use "^GSPC" for S&P 500, "^VIX" for VIX, "BTC-USD" for Bitcoin.
- For multi-asset: fetch each ticker separately, then merge_datasets.
- Yahoo data may have gaps on holidays — clean_dataset handles this.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_clean_dataset(dataset: str = "raw_data") -> list[dict]:
        """Guide: Cleaning and validating financial data."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Clean {dataset}:

1. Call clean_dataset(dataset="{dataset}").
2. Handles: missing values, duplicates, outliers, timezone issues, column types.
3. Missing value methods: "ffill" (forward fill, default for prices), "drop", "interpolate".

Best practices:
- Always clean before analysis. Dirty data → wrong results.
- For prices: forward fill is standard (last known price).
- For returns: drop is better (avoid artificial zero returns).
- Check for survivorship bias: are delisted stocks included?
- validate_returns_tool after cleaning to verify data quality.
""",
                },
            }
        ]

    # ── econometrics/ tools ───────────────────────────────────────────

    @mcp.prompt()
    def guide_panel_regression(dataset: str = "panel_data") -> list[dict]:
        """Guide: Panel data regression (fixed/random effects)."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Panel regression on {dataset}:

1. Call panel_regression(dataset="{dataset}", y="returns", x=["size", "value"], entity="ticker", time="date").
2. Methods: "fixed_effects" (entity dummies), "random_effects" (GLS), "pooled" (OLS).
3. Fixed effects: controls for unobserved entity-specific factors. Most common in finance.

Interpretation:
- Fixed effects absorb time-invariant entity differences (company quality, sector).
- Hausman test p < 0.05: use fixed effects (random effects biased). p > 0.05: random effects OK.
- Clustered standard errors: cluster by entity for panel data (accounts for within-entity correlation).
- R² within vs between: within captures time-series variation, between captures cross-sectional.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_var_model(dataset: str = "multi_asset_returns") -> list[dict]:
        """Guide: Vector autoregression (VAR) model."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
VAR model on {dataset}:

1. Call var_model(dataset="{dataset}", columns=["asset_a", "asset_b"], max_lags=5).
2. Models the dynamic interaction between multiple time series.
3. Lag selection: AIC/BIC picks the optimal number of lags.

Interpretation:
- Coefficients: how each variable's lags affect the others.
- Granger causality: which variables predict which? Check p-values.
- Impulse response: how does a shock to asset_a propagate to asset_b?
- Variance decomposition: what fraction of asset_b's variance is explained by asset_a shocks?
- Use impulse_response for shock propagation analysis.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_structural_break(dataset: str = "returns") -> list[dict]:
        """Guide: Testing for structural breaks."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Structural break tests on {dataset}:

1. Call structural_break(dataset="{dataset}", column="returns").
2. Tests: Chow, CUSUM, Bai-Perron (multiple breaks), Zivot-Andrews (unit root with break).
3. Identifies dates where the data-generating process changed fundamentally.

Interpretation:
- Chow test: is there a break at a specific known date? (e.g., policy change).
- CUSUM: sequential test — detects breaks as they happen (monitoring tool).
- Bai-Perron: finds multiple unknown break dates. Most comprehensive.
- Breaks often align with: regime changes, policy shifts, market structure changes.
- After finding breaks: re-estimate models on each sub-period separately.
""",
                },
            }
        ]

    # ── viz/ tools ────────────────────────────────────────────────────

    @mcp.prompt()
    def guide_plot_equity_curve(dataset: str = "backtest_results") -> list[dict]:
        """Guide: Plotting equity curves."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Plot equity curve for {dataset}:

1. Call plot_equity_curve(dataset="{dataset}", column="equity").
2. Shows cumulative wealth over time with drawdown shading.
3. Add benchmark for comparison: benchmark_dataset="benchmark".

Reading the chart:
- Steep upward slope: high returns. Flat periods: drawdowns or low activity.
- Drawdown shading shows underwater periods. Deeper/longer = more painful.
- Compare to benchmark: are you above or below? When do you diverge?
- Smooth curve = consistent returns. Jagged = high-variance strategy.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_portfolio_dashboard(dataset: str = "portfolio_returns") -> list[dict]:
        """Guide: Creating interactive portfolio dashboards."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Portfolio dashboard for {dataset}:

1. Call portfolio_dashboard(dataset="{dataset}").
2. Creates interactive Plotly dashboard with: equity curve, drawdowns, rolling metrics, allocation.
3. Includes: performance table, risk metrics, regime overlay.

Sections:
- Performance: cumulative returns with benchmark comparison.
- Risk: rolling Sharpe, rolling vol, VaR breaches.
- Allocation: pie chart of current weights, weight evolution over time.
- Regimes: regime overlay on equity curve — see how you perform in each regime.
- Drawdowns: underwater chart with recovery markers.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_regime_dashboard(dataset: str = "returns") -> list[dict]:
        """Guide: Creating regime analysis dashboards."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
Regime dashboard for {dataset}:

1. First detect_regimes on the dataset.
2. Then regime_dashboard(dataset="{dataset}") for interactive visualization.
3. Shows: regime timeline, per-regime statistics, transition probabilities, rolling probabilities.

Reading the dashboard:
- Regime timeline: colored bands showing which regime was active at each point.
- Per-regime box: mean return, vol, Sharpe for each regime. Bull should be green, bear red.
- Transition matrix heatmap: probability of moving between regimes.
- Rolling probability: current confidence in each regime state.
""",
                },
            }
        ]

    # ── Remaining tool guides (batch) ─────────────────────────────────

    @mcp.prompt()
    def guide_correlation_network(dataset: str = "multi_returns") -> list[dict]:
        """Guide: Building correlation networks."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
correlation_network on {dataset}: Build network where edges = correlations above threshold. Returns centrality scores. Most central asset = most connected = highest systemic risk. Use minimum_spanning_tree for the backbone, community_detection for clusters.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_minimum_spanning_tree(dataset: str = "multi_returns") -> list[dict]:
        """Guide: Minimum spanning tree from correlations."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
minimum_spanning_tree on {dataset}: Extracts the backbone of correlation structure — shortest path connecting all assets without loops. Useful for HRP portfolio construction and visual sector identification. Highly correlated assets are connected directly.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_community_detection(dataset: str = "multi_returns") -> list[dict]:
        """Guide: Detecting asset clusters."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
community_detection on {dataset}: Groups assets into clusters based on correlation network. Assets in the same community are more correlated with each other than with others. Useful for sector identification, diversification analysis, and HRP.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_contagion_simulation(dataset: str = "multi_returns") -> list[dict]:
        """Guide: Financial contagion simulation."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
contagion_simulation on {dataset}: Shocks one asset and propagates losses through the correlation network. Shows how many assets get infected and total system loss. Use to identify "too connected to fail" assets. Higher infection_pct = more systemic risk.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_entropy_analysis(dataset: str = "returns") -> list[dict]:
        """Guide: Shannon entropy for predictability."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
entropy_analysis on {dataset}: Measures randomness/predictability of returns. Normalized entropy near 1.0 = highly random (EMH). Near 0.5 = some structure (potentially exploitable). Compare across assets or time periods to find relative predictability.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_cusum_detect(dataset: str = "returns") -> list[dict]:
        """Guide: CUSUM changepoint detection."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
cusum_detect on {dataset}: Monitors cumulative deviations from the mean. When CUSUM exceeds threshold, a structural change is signaled. Good for online monitoring — detects gradual shifts, not just sudden jumps. Use alongside structural_break for formal testing.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_longstaff_schwartz(dataset: str = "simulated_paths") -> list[dict]:
        """Guide: American option pricing via LSM."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
longstaff_schwartz on {dataset}: Prices American options via Monte Carlo + regression. Needs simulated price paths (from simulate_process). Estimates optimal early exercise boundary. Price - European price = early exercise premium. Critical for American puts and convertible bonds.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_variance_gamma_simulate() -> list[dict]:
        """Guide: Variance Gamma process simulation."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
variance_gamma_simulate: Simulates VG process — Brownian motion with Gamma time change. Captures fat tails (nu) and skewness (theta). nu > 0 = fatter tails than normal. theta < 0 = negative skew. Use for realistic return simulation and option pricing beyond Black-Scholes.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_nig_simulate() -> list[dict]:
        """Guide: Normal Inverse Gaussian simulation."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
nig_simulate: Simulates NIG process — captures asymmetric fat tails. alpha controls tail heaviness (smaller = fatter). beta controls asymmetry (negative = left-skewed, typical for equities). More flexible than VG for fitting real return distributions.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_levy_simulate() -> list[dict]:
        """Guide: General Lévy process simulation."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
levy_simulate: Unified interface for VG, NIG, CGMY, and stable processes. Choose model based on data: VG for moderate tails, NIG for asymmetry, CGMY for fine-tuned tail control, stable for extreme heavy tails. Compare simulated increments to historical returns for calibration quality.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_optimal_stopping(dataset: str = "prices") -> list[dict]:
        """Guide: Optimal stopping and exit timing."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
optimal_stopping on {dataset}: method="cusum" detects changepoints. method="ou_exit" computes optimal exit threshold for mean-reverting trades. OU exit balances expected profit vs holding cost. Use for pairs trading exit rules and stop-loss optimization.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_hawkes_fit_detail(dataset: str = "events") -> list[dict]:
        """Guide: Hawkes process for self-exciting events (detailed)."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
hawkes_fit on {dataset}: Fits self-exciting point process. Events trigger more events (e.g., trades beget trades, crashes trigger panic). Branching ratio < 1.0 = stable process. Near 1.0 = critical (flash crash territory). Use for: trade arrival modeling, jump clustering, volatility contagion.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_fit_gaussian_hmm(dataset: str = "returns") -> list[dict]:
        """Guide: Fitting Gaussian HMM for regime detection."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
fit_gaussian_hmm on {dataset}: Fits a Hidden Markov Model with Gaussian emissions. Returns states (Viterbi), transition matrix, per-state means/covariances. n_states=2 = bull/bear. n_states=3 adds crisis state. Check AIC/BIC. Use n_init=10+ for robust EM.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_fit_ms_autoregression(dataset: str = "returns") -> list[dict]:
        """Guide: Markov-switching autoregression."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
fit_ms_autoregression on {dataset}: Extends HMM with AR lags whose coefficients switch across regimes. Better than HMM when returns show serial correlation (momentum in bull, reversal in bear). Use for GDP forecasting, interest rate modeling, momentum-reversal switching.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_gaussian_mixture_regimes(dataset: str = "returns") -> list[dict]:
        """Guide: GMM-based regime detection."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
gaussian_mixture_regimes on {dataset}: Clusters returns into regimes using Gaussian Mixture Model. Unlike HMM, GMM ignores temporal ordering — purely distributional. Faster than HMM. Good for initial regime exploration. Compare to HMM results for validation.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_kalman_filter(dataset: str = "prices") -> list[dict]:
        """Guide: Kalman filter for state estimation."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
kalman_filter on {dataset}: Optimal recursive state estimator. Use for: trend extraction (smoother than MA), dynamic hedge ratio estimation, time-varying beta, signal-noise separation. Returns filtered states and smoothed states (full-sample optimal).
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_kalman_regression(dataset: str = "returns") -> list[dict]:
        """Guide: Time-varying regression via Kalman filter."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
kalman_regression on {dataset}: Estimates time-varying coefficients. Unlike OLS (fixed coefficients), Kalman tracks how relationships evolve. Use for: dynamic beta, time-varying hedge ratio in pairs trading, adaptive factor exposures. Q parameter controls how fast coefficients can change.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_regime_statistics(dataset: str = "returns") -> list[dict]:
        """Guide: Per-regime summary statistics."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
regime_statistics on {dataset}: After detect_regimes, computes mean, vol, Sharpe, max DD per regime. Bull regime should have higher mean, lower vol. If Sharpe doesn't improve per-regime, regime information isn't valuable for this asset. Use for regime-conditional position sizing.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_select_n_states(dataset: str = "returns") -> list[dict]:
        """Guide: Selecting optimal number of HMM states."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
select_n_states on {dataset}: Tests n=2,3,4,5 states and picks best by BIC. 2 states = bull/bear (most common). 3 states = adds crisis. 4+ states = usually overfitting. If BIC is flat across 2-3, use 2 (simpler). Report the BIC scores for justification.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_rolling_regime_probability(dataset: str = "returns") -> list[dict]:
        """Guide: Time-varying regime probabilities."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
rolling_regime_probability on {dataset}: Shows P(bull) and P(bear) over time. Useful for real-time monitoring. When P(bear) crosses 0.5, consider reducing exposure. Gradual transitions = regime uncertainty. Sharp transitions = clear regime change. Use for regime-filtered trading signals.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_regime_scoring(dataset: str = "returns") -> list[dict]:
        """Guide: Composite regime scoring."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
regime_scoring on {dataset}: Combines multiple regime signals (HMM, vol, momentum, correlation) into a single composite score. Score > 0 = favorable conditions. Score < 0 = unfavorable. Use as a regime filter for strategies: only trade when composite is favorable.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_regime_conditional_moments(dataset: str = "returns") -> list[dict]:
        """Guide: Regime-conditional return moments."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
regime_conditional_moments on {dataset}: Computes mean, var, skew, kurtosis separately for each regime state. Reveals how return distribution changes across regimes. Bull: positive mean, moderate vol. Bear: negative mean, high vol, negative skew. Use for regime-dependent VaR.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_regime_labels(dataset: str = "returns") -> list[dict]:
        """Guide: Assigning human-readable regime labels."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
regime_labels on {dataset}: Assigns interpretable names to numeric regime states. method="volatility" labels by vol level. method="trend" by direction. method="composite" combines multiple signals. Makes regime outputs more intuitive for reporting and dashboards.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_regime_transition(dataset: str = "returns") -> list[dict]:
        """Guide: Regime transition analysis."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
regime_transition on {dataset}: Analyzes the transition matrix — P(switch from regime i to j). Expected regime durations = 1/(1-P(i→i)). Steady state = long-run fraction of time in each regime. Use for: estimating how long current regime will last, regime-conditional asset allocation.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_regime_backtest(dataset: str = "returns") -> list[dict]:
        """Guide: Regime-conditional backtesting."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
regime_backtest on {dataset}: Compares strategy performance WITH regime filter vs WITHOUT. If regime filter improves Sharpe and reduces drawdown, the regime signal adds value. Key: no look-ahead in regime detection — use rolling probabilities, not full-sample labels.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_momentum_indicators(dataset: str = "prices") -> list[dict]:
        """Guide: Computing momentum indicators."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
momentum_indicators on {dataset}: Computes RSI, MACD, Stochastic, ROC, Williams %R, CCI, MFI at once. Returns overall momentum assessment (bullish/bearish/neutral). Use for signal generation and market timing. RSI > 70 = overbought, < 30 = oversold.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_trend_indicators(dataset: str = "prices") -> list[dict]:
        """Guide: Computing trend indicators."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
trend_indicators on {dataset}: ADX, Aroon, TRIX, PSAR + SMA crossovers. Returns trend strength (ADX: strong/moderate/weak/none) and direction (uptrend/downtrend/sideways). ADX > 25 = trending. Price > SMA50 > SMA200 = strong uptrend (golden cross).
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_volatility_indicators(dataset: str = "prices") -> list[dict]:
        """Guide: Computing volatility indicators."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
volatility_indicators on {dataset}: ATR, Bollinger Bands, Keltner Channel, Donchian Channel, BB Width. Returns vol regime (high/normal/low). BB Width < 0.03 = squeeze (expect breakout). ATR% shows vol as fraction of price. Use for position sizing and stop placement.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_volume_indicators(dataset: str = "prices") -> list[dict]:
        """Guide: Computing volume indicators."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
volume_indicators on {dataset}: OBV, AD Line, CMF, MFI, Force Index. Returns volume confirmation (confirmed_up, divergence_bearish, etc.). Price up + OBV up = confirmed. Price up + OBV down = bearish divergence (warning signal). CMF > 0.05 = accumulation. MFI < 20 = oversold on volume.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_pattern_recognition(dataset: str = "prices") -> list[dict]:
        """Guide: Candlestick pattern detection."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
pattern_recognition on {dataset}: Scans for doji, hammer, engulfing, morning/evening star, shooting star, hanging man, harami, dark cloud cover. Returns detected patterns with bullish/bearish bias. Patterns are most reliable at support/resistance levels and with volume confirmation.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_support_resistance(dataset: str = "prices") -> list[dict]:
        """Guide: Computing S/R levels."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
support_resistance on {dataset}: Identifies key price levels using fractals, price clustering, and pivot points. Also computes Fibonacci retracements. Returns nearest support/resistance with distance %. Price near support = potential bounce. Price near resistance = potential rejection.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_ta_summary(dataset: str = "prices") -> list[dict]:
        """Guide: Comprehensive TA summary."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
ta_summary on {dataset}: One-call summary across momentum, trend, vol, and moving averages. Returns overall assessment (bullish/bearish/neutral) with bullish_pct score. Includes RSI, MACD, SMA 20/50/200, ADX, ATR, Bollinger Bands. Use for quick market overview.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_ta_screening(dataset: str = "prices") -> list[dict]:
        """Guide: Multi-indicator signal screening."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
ta_screening on {dataset}: Scans RSI, MACD crossover, MA crossover, and Bollinger Bands. Returns composite score (-1 to +1) and recommendation (strong_buy/buy/hold/sell/strong_sell). Score > 0.5 = strong_buy. Multiple confirming signals = higher conviction.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_scan_signals(dataset: str = "prices") -> list[dict]:
        """Guide: Overbought/oversold signal scan."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
scan_signals on {dataset}: Quick OB/OS check across RSI, Stochastic, Williams %R, CCI. Returns consensus (overbought/oversold/neutral). 2+ indicators agreeing = stronger signal. Use for mean-reversion entry timing. Confirm with volume and trend context.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_list_indicators() -> list[dict]:
        """Guide: Listing available TA indicators."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
list_indicators: Shows all 265 TA indicators by category. Pass category="momentum" to filter. Categories: overlap, momentum, volume, trend, volatility, patterns, statistics, cycles, fibonacci, smoothing, exotic, support_resistance. Use multi_indicator to compute several at once.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_multi_indicator(dataset: str = "prices") -> list[dict]:
        """Guide: Computing multiple indicators at once."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
multi_indicator on {dataset}: Compute multiple TA indicators in one call. Pass indicators=["rsi", "macd", "bollinger_bands"]. More efficient than calling compute_indicator repeatedly. Results stored as new columns in a new dataset. Use for feature engineering before ML.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_rebalance_analysis(dataset: str = "portfolio") -> list[dict]:
        """Guide: Portfolio rebalancing analysis."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
rebalance_analysis on {dataset}: Compares current weights to target weights. Computes required trades, turnover, and estimated transaction costs. Key: rebalance when drift > threshold (typically 5% absolute). Turnover × cost = performance drag. Calendar vs threshold rebalancing tradeoff.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_yield_curve_analysis() -> list[dict]:
        """Guide: Yield curve construction and analysis."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
yield_curve_analysis: Bootstrap zero curve from bond data. Compute forward rates, duration, convexity. Curve shape signals: normal (growth), flat (slowdown), inverted (recession). Key rate durations show sensitivity to each point. Use for rate scenario analysis and bond portfolio risk.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_seasonality_analysis(dataset: str = "returns") -> list[dict]:
        """Guide: Detecting seasonal patterns."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
seasonality_analysis on {dataset}: Tests for day-of-week, month-of-year, and holiday effects. Statistically significant seasonality = potential trading signal. January effect, Monday effect, turn-of-month. Always check if the pattern persists out-of-sample and survives transaction costs.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_robust_stats(dataset: str = "returns") -> list[dict]:
        """Guide: Robust statistics (median, IQR, trimmed mean)."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
robust_stats on {dataset}: Computes median, IQR, trimmed mean, MAD — resistant to outliers. Use when data has fat tails (always in finance). Mean ≠ median = skewed distribution. MAD < std = outliers inflate std. Trimmed mean at 5% removes extreme 5% from each tail.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_cointegration_johansen(dataset: str = "prices") -> list[dict]:
        """Guide: Johansen multivariate cointegration test."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
cointegration_johansen on {dataset}: Tests for number of cointegrating relationships among multiple series. Trace and max eigenvalue statistics. Rank 0 = no cointegration. Rank 1 = one cointegrating relationship. Cointegrating vectors define the long-run equilibrium. Use for multi-leg pairs trading.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_credit_analysis(dataset: str = "financials") -> list[dict]:
        """Guide: Credit risk analysis."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
credit_analysis on {dataset}: Merton model for default probability, credit spreads, and distance-to-default. Higher distance-to-default = safer. Use with altman_z for a complementary view. Combine with stress_test to see default probability under crisis scenarios.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_ewma_volatility(dataset: str = "returns") -> list[dict]:
        """Guide: EWMA volatility estimation."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
ewma_volatility on {dataset}: Exponentially weighted moving average vol. Lambda=0.94 (RiskMetrics default). More responsive than simple rolling vol. No parameters to estimate (unlike GARCH). Good for real-time vol monitoring. Compare to GARCH conditional vol for model validation.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_impulse_response(dataset: str = "macro_data") -> list[dict]:
        """Guide: VAR impulse response functions."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
impulse_response on {dataset}: After fitting var_model, shows how a 1σ shock to variable A propagates to B over time. Peak response = maximum impact. Decay rate = how quickly the effect fades. Use for understanding cross-asset shock transmission and monetary policy transmission.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_event_study_econometric(dataset: str = "returns") -> list[dict]:
        """Guide: Formal econometric event study."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
event_study_econometric on {dataset}: Estimates abnormal returns around events using market model. Computes CAR and statistical tests. Pre-event CAR > 0 = information leakage. Post-event drift = under-reaction. t-stat > 2 = statistically significant at 5%.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_instrumental_variable(dataset: str = "panel_data") -> list[dict]:
        """Guide: Instrumental variable (2SLS) estimation."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
instrumental_variable on {dataset}: Solves endogeneity (OLS bias from omitted variables/simultaneity). Needs an instrument that's correlated with X but doesn't directly affect Y. First-stage F > 10 or you have weak instruments. IV estimate is causal (under assumptions).
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_regression_discontinuity(dataset: str = "data") -> list[dict]:
        """Guide: Regression discontinuity design."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
regression_discontinuity on {dataset}: Estimates causal effect at a threshold (e.g., index inclusion at a market cap cutoff). Treatment = above threshold. Control = below. The discontinuity in outcome at the cutoff = causal effect. Only valid locally near the cutoff.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_mediation_analysis(dataset: str = "data") -> list[dict]:
        """Guide: Causal mediation analysis."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
mediation_analysis on {dataset}: Tests if the effect of X on Y goes THROUGH a mediator M. Direct effect: X→Y. Indirect effect: X→M→Y. Total = direct + indirect. If indirect is significant, M explains HOW X affects Y. Use for factor attribution and policy transmission.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_bayesian_portfolio(dataset: str = "returns") -> list[dict]:
        """Guide: Bayesian portfolio optimization."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
bayesian_portfolio on {dataset}: Accounts for parameter uncertainty in portfolio weights. Unlike MVO (treats estimates as truth), Bayesian averages over the posterior. Result: more diversified, more robust weights. Especially valuable with short track records or many assets.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_bayesian_volatility(dataset: str = "returns") -> list[dict]:
        """Guide: Bayesian stochastic volatility."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
bayesian_volatility on {dataset}: Estimates time-varying vol with full uncertainty. Returns posterior distribution of vol at each time point. Wider CI = more vol uncertainty. Compare to GARCH point estimates. Use for Bayesian VaR (integrates over vol uncertainty). More conservative than plug-in approaches.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_hmc_sample(dataset: str = "returns") -> list[dict]:
        """Guide: Hamiltonian Monte Carlo sampling."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
hmc_sample on {dataset}: MCMC sampling using Hamiltonian dynamics. More efficient than random walk Metropolis for high-dimensional problems. Check Rhat < 1.1 for convergence. Check effective sample size > 100. Use for any Bayesian model where conjugate priors aren't available.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_model_comparison_bayesian() -> list[dict]:
        """Guide: Bayesian model comparison."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
model_comparison_bayesian: Compares models using WAIC, LOO, or Bayes factors. WAIC difference < 2 = equivalent models (choose simpler). Difference 2-10 = some evidence. Difference > 10 = strong evidence for better model. Always prefer simpler model when evidence is ambiguous.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_piotroski_score(dataset: str = "financials") -> list[dict]:
        """Guide: Piotroski F-Score quality ranking."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
piotroski_score on {dataset}: 0-9 quality score based on profitability, leverage, and operating efficiency. F-score >= 7 = high quality. F-score <= 2 = low quality. Backtest: long high-quality, short low-quality. The quality factor has been persistent across markets.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_altman_z(dataset: str = "financials") -> list[dict]:
        """Guide: Altman Z-Score for bankruptcy prediction."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
altman_z on {dataset}: Z > 2.99 = safe. 1.81 < Z < 2.99 = grey zone. Z < 1.81 = distress zone (high default probability). Combine with credit_analysis (Merton model) for complementary credit view. Z-score works best for manufacturing firms; use Z'' for non-manufacturing.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_dcf_valuation(dataset: str = "financials") -> list[dict]:
        """Guide: Discounted cash flow valuation."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
dcf_valuation on {dataset}: Intrinsic value = PV of future cash flows. Needs: projected FCF, discount rate (WACC), terminal growth rate. Highly sensitive to: discount rate (±1% changes value 20-30%), terminal growth (2-3% for mature companies), and FCF projections. Use sensitivity analysis.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_fundamental_ratios(dataset: str = "financials") -> list[dict]:
        """Guide: Fundamental financial ratios."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
fundamental_ratios on {dataset}: Computes P/E, P/B, ROE, D/E, current ratio, etc. Compare to sector medians, not absolute thresholds. Low P/E + high ROE + low D/E = quality value. High P/E alone is NOT expensive if growth justifies it (PEG ratio).
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_quality_screen(dataset: str = "financials") -> list[dict]:
        """Guide: Quality stock screening."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
quality_screen on {dataset}: Composite quality score from profitability, stability, and growth metrics. Filters for companies with consistent earnings, low leverage, and improving fundamentals. Use with piotroski_score and fundamental_ratios for a complete quality assessment.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_sentiment_score() -> list[dict]:
        """Guide: News sentiment scoring."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
sentiment_score: Scores text/headlines from -1 (very negative) to +1 (very positive). Use on news headlines, earnings call transcripts, or social media. Aggregate across sources for a robust signal. Sentiment often leads price moves by hours to days.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_sentiment_aggregate(dataset: str = "news_data") -> list[dict]:
        """Guide: Aggregating sentiment over time."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
sentiment_aggregate on {dataset}: Aggregates individual sentiment scores into daily/weekly signals. Methods: mean, exponential decay (recent news matters more), volume-weighted. Compare aggregate sentiment to returns — lead-lag analysis reveals if sentiment is predictive.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_news_signal() -> list[dict]:
        """Guide: Converting news sentiment to trading signals."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
news_signal: Converts sentiment scores into +1/-1 trading signals with thresholds. Signal quality depends on sentiment persistence and noise level. Backtest the signal before trading. Combine with TA signals for higher conviction. News alpha decays fast — act within hours.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_news_impact() -> list[dict]:
        """Guide: News impact on returns."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
news_impact: Measures how different news categories affect returns. Earnings news has biggest impact. M&A, regulatory, and analyst actions follow. Quantify the average return impact per category. Use for event-driven strategy design.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_earnings_surprise() -> list[dict]:
        """Guide: Earnings surprise analysis."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
earnings_surprise: Computes actual vs expected earnings. Positive surprise = beat estimates → typically positive drift (PEAD). Negative surprise = missed → negative drift. The magnitude of surprise and revision history matter more than direction alone.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_create_experiment() -> list[dict]:
        """Guide: Creating a research experiment."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
create_experiment: Sets up a controlled experiment to compare strategies/parameters. Define hypothesis, variables, and evaluation metrics upfront. Use for systematic strategy research. Tracks all runs with parameters and results for reproducibility.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_run_experiment() -> list[dict]:
        """Guide: Running a research experiment."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
run_experiment: Executes an experiment with specified parameters. Records all inputs, outputs, and metrics. Use with parameter_sensitivity for systematic parameter sweeps. All results stored for later comparison with experiment_comparison.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_experiment_results() -> list[dict]:
        """Guide: Viewing experiment results."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
experiment_results: Retrieves all runs from an experiment with their parameters and metrics. Sort by Sharpe, max DD, or any metric. Identify best parameters. Check if results are robust across different parameter values (sensitivity analysis).
""",
                },
            }
        ]

    # ── Data tools ────────────────────────────────────────────────────

    @mcp.prompt()
    def guide_load_csv() -> list[dict]:
        """Guide: Loading CSV data into workspace."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
load_csv: Load a CSV file into the workspace as a named dataset. Specify date_column for automatic DatetimeIndex parsing. After loading, use describe_dataset to inspect and clean_dataset to prepare for analysis.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_load_json() -> list[dict]:
        """Guide: Loading JSON data into workspace."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
load_json: Load JSON data (records format) into workspace. Useful for API responses, config files, or structured data. Converts to DataFrame automatically. Use for loading trade data, order books, or parameter sets.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_describe_dataset(dataset: str = "data") -> list[dict]:
        """Guide: Inspecting dataset properties."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
describe_dataset on {dataset}: Shows shape, columns, dtypes, basic stats, missing values. First thing to call after loading new data. Check for: correct number of rows, expected columns, reasonable value ranges, missing value patterns.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_validate_returns_tool(dataset: str = "returns") -> list[dict]:
        """Guide: Validating return data quality."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
validate_returns_tool on {dataset}: Checks for common data quality issues: extreme returns (> 50%), zero returns, negative prices, missing values, weekend data. Flags suspicious observations. Always validate before analysis — bad data leads to bad results.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_merge_datasets() -> list[dict]:
        """Guide: Merging multiple datasets."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
merge_datasets: Combines multiple datasets by date alignment. Options: inner (only common dates), outer (all dates), left/right join. For multi-asset analysis, use inner join to ensure all assets have data for every date.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_filter_dataset(dataset: str = "data") -> list[dict]:
        """Guide: Filtering dataset rows/columns."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
filter_dataset on {dataset}: Select specific date ranges, remove outliers, or keep specific columns. For SQL-level filtering, use query_data instead. Filter is simpler for common operations like date slicing and column selection.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_export_dataset(dataset: str = "data") -> list[dict]:
        """Guide: Exporting data from workspace."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
export_dataset on {dataset}: Save workspace data to CSV, Parquet, or JSON. Use Parquet for large datasets (compressed, fast). CSV for sharing with non-Python tools. JSON for API integration.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_add_column(dataset: str = "data") -> list[dict]:
        """Guide: Adding computed columns to a dataset."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
add_column on {dataset}: Add a new column computed from existing ones. Examples: rolling average, lagged returns, dummy variables, log transforms. Useful for feature engineering before ML or custom indicator creation.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_split_dataset(dataset: str = "data") -> list[dict]:
        """Guide: Splitting data for train/test."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
split_dataset on {dataset}: Split by date (time series), fraction, or explicit index. Always split by time for financial data — random splits create look-ahead bias. Typical: 70% train, 30% test. Use walk_forward for rolling splits.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_rename_columns(dataset: str = "data") -> list[dict]:
        """Guide: Renaming dataset columns."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
rename_columns on {dataset}: Standardize column names. Convention: lowercase, underscores, no spaces. Standard names: close, open, high, low, volume, returns. Consistent naming ensures tools work correctly (many expect "close" or "returns" columns).
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_resample_dataset(dataset: str = "data") -> list[dict]:
        """Guide: Resampling data frequency."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
resample_dataset on {dataset}: Change frequency (daily→weekly, minute→hourly). For OHLCV data, use resample_ohlcv which properly aggregates: O=first, H=max, L=min, C=last, V=sum. Regular resample uses last value by default.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_compute_log_returns(dataset: str = "prices") -> list[dict]:
        """Guide: Computing log returns."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
compute_log_returns on {dataset}: log_return = ln(P_{{t}} / P_{{t-1}}). Log returns are additive over time (unlike simple returns), normally distributed by assumption, and preferred for GARCH fitting and statistical modeling. Use compute_returns(method="simple") for portfolio returns.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_align_datasets() -> list[dict]:
        """Guide: Aligning multiple datasets by date."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
align_datasets: Aligns two or more datasets to common dates. Handles different trading calendars, holidays, and missing data. Use before correlation_analysis or optimize_portfolio to ensure all assets have data for the same dates.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_fetch_ohlcv() -> list[dict]:
        """Guide: Fetching OHLCV data."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
fetch_ohlcv: Fetches OHLCV data with more control than fetch_yahoo. Supports multiple providers and intervals. Specify start/end dates, frequency, and data source. Returns standardized columns: open, high, low, close, volume.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_resample_ohlcv(dataset: str = "intraday") -> list[dict]:
        """Guide: Resampling OHLCV data."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
resample_ohlcv on {dataset}: Properly aggregates OHLCV: open=first, high=max, low=min, close=last, volume=sum. Critical to use this instead of regular resample for price data — taking the last value for high/low would be wrong.
""",
                },
            }
        ]

    # ── Remaining micro/execution/forex guides ────────────────────────

    @mcp.prompt()
    def guide_kyle_lambda_rolling(dataset: str = "prices") -> list[dict]:
        """Guide: Rolling Kyle's lambda estimation."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
kyle_lambda_rolling on {dataset}: Rolling OLS estimate of permanent price impact. Trend in lambda shows changing liquidity. Rising lambda = deteriorating liquidity. Confidence intervals show significance. Use for monitoring execution conditions over time.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_amihud_rolling(dataset: str = "prices") -> list[dict]:
        """Guide: Rolling Amihud illiquidity."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
amihud_rolling on {dataset}: Rolling Amihud ratio = |return| / volume. Higher = more illiquid. Compare current to historical percentile. Spikes often precede volatility events. Use for: position sizing, execution timing, liquidity risk monitoring.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_corwin_schultz_spread(dataset: str = "prices") -> list[dict]:
        """Guide: Corwin-Schultz spread estimation."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
corwin_schultz_spread on {dataset}: Estimates effective spread from daily high-low prices. No need for bid-ask quotes. Based on the idea that daily high captures the ask and low captures the bid. Compare to Roll spread for validation. Wider spread = higher trading costs.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_roll_spread_tool(dataset: str = "prices") -> list[dict]:
        """Guide: Roll spread estimation."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
roll_spread_tool on {dataset}: Estimates spread from serial price autocovariance. Negative autocovariance → bid-ask bounce → spread. Simple but assumes constant spread. Compare to corwin_schultz for consistency. Negative values can occur (set to 0).
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_effective_spread_tool(dataset: str = "trades") -> list[dict]:
        """Guide: Effective spread computation."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
effective_spread_tool on {dataset}: Gold standard = 2 × |trade_price - midpoint|. Needs bid/ask quotes. Effective < quoted = price improvement. Effective > quoted = adverse selection. Compare across venues for best execution analysis.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_trade_classification_tool(dataset: str = "trades") -> list[dict]:
        """Guide: Classifying trades as buyer/seller initiated."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
trade_classification_tool on {dataset}: Uses Lee-Ready algorithm (tick test + quote test) to classify each trade. Buy_pct > 60% = net buying pressure. Sell_pct > 60% = net selling. Use for: order flow analysis, VPIN computation, toxicity assessment.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_order_flow_imbalance(dataset: str = "trades") -> list[dict]:
        """Guide: Order flow imbalance analysis."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
order_flow_imbalance on {dataset}: Net buy vs sell volume normalized. OFI near +1 = strong buying. OFI near -1 = strong selling. Persistent OFI predicts short-term returns. Mean-reverting OFI = noise traders dominate. Use as an intraday signal.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_information_share(dataset: str = "multi_venue") -> list[dict]:
        """Guide: Information share across venues."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
information_share on {dataset}: Measures which venue contributes most to price discovery. IS > 50% = dominant venue. Based on Hasbrouck (1995). Use for: venue selection, understanding market structure, routing decisions.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_intraday_volatility_pattern(dataset: str = "intraday") -> list[dict]:
        """Guide: Intraday volatility patterns."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
intraday_volatility_pattern on {dataset}: Maps the U-shape (or J-shape) of intraday vol. High at open (information processing), low midday (lunch lull), high at close (portfolio rebalancing). Use for: execution timing, intraday strategy design, vol regime estimation.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_liquidity_commonality(dataset: str = "prices") -> list[dict]:
        """Guide: Liquidity commonality analysis."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
liquidity_commonality on {dataset}: Measures how much an asset's liquidity co-moves with market liquidity. High commonality (R² > 0.3) = systematic liquidity risk. During crises, all liquidity dries up together. Low commonality = idiosyncratic liquidity (safer).
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_is_schedule_tool() -> list[dict]:
        """Guide: Implementation Shortfall schedule."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
is_schedule_tool: Generates IS-optimal execution schedule. Alpha parameter controls urgency: 0.0 = pure VWAP, 0.5 = balanced, 1.0 = pure TWAP. front_loaded_pct shows how much trades in the first quarter. Higher urgency = more front-loaded. Use for alpha-driven trades.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_pov_schedule_tool() -> list[dict]:
        """Guide: Percentage of Volume schedule."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
pov_schedule_tool: Executes as a fixed percentage of market volume. pov_rate=0.10 = 10% of each interval's volume. Adaptive to volume patterns. fill_pct < 100% means the order needs multiple days. Use for: benchmark tracking, minimizing footprint, patient execution.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_expected_cost_model_tool() -> list[dict]:
        """Guide: Pre-trade expected cost model."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
expected_cost_model_tool: Estimates total execution cost before trading. Uses square-root model: impact ∝ σ × √(Q/ADV). Add spread cost for total. Compare to expected alpha — if cost > alpha, don't trade. Use for go/no-go decision on trade ideas.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_bertsimas_lo_tool() -> list[dict]:
        """Guide: Bertsimas-Lo optimal execution."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
bertsimas_lo_tool: Discrete-time dynamic programming approach to optimal execution. Compare to almgren_chriss (continuous-time). BL provides expected cost AND cost variance. Lower variance = more certain execution cost. Use as validation for AC trajectory.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_slippage_estimate() -> list[dict]:
        """Guide: Quick slippage estimation."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
slippage_estimate: Quick estimate of expected slippage in bps. Uses participation rate and volatility. Participation > 20% = high impact, consider multi-day. Participation < 3% = negligible impact. Sanity check before detailed execution_cost analysis.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_cross_rate() -> list[dict]:
        """Guide: FX cross rate computation."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
cross_rate: Computes implied cross rates from two base pairs. Example: EUR/JPY from EUR/USD and USD/JPY. Compare implied to actual for triangular arbitrage opportunities. Deviations > transaction cost = potential arb.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_session_info() -> list[dict]:
        """Guide: FX session information."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
session_info: Returns FX session times (Asian, European, US) with overlap periods. Highest liquidity during London-NY overlap (13:00-17:00 UTC). Asian session: JPY/AUD pairs most active. Use for execution timing and session-based strategies.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_fx_risk(dataset: str = "fx_portfolio") -> list[dict]:
        """Guide: FX risk analysis."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
fx_risk on {dataset}: Measures currency risk contribution to portfolio volatility. Decompose total risk into asset risk + FX risk + cross terms. High FX contribution = consider hedging. Optimal hedge ratio balances risk reduction against hedging cost.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_pip_calculator() -> list[dict]:
        """Guide: FX pip value calculator."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
pip_calculator: Computes pip value for position sizing. One pip = 0.0001 for most pairs (0.01 for JPY pairs). Pip value depends on pair, position size, and account currency. Use for: risk management, stop-loss placement, position sizing.
""",
                },
            }
        ]

    # ── Viz remaining guides ──────────────────────────────────────────

    @mcp.prompt()
    def guide_plot_returns(dataset: str = "returns") -> list[dict]:
        """Guide: Plotting returns."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
plot_returns on {dataset}: Cumulative returns chart or drawdown chart. cumulative=True shows wealth growth. cumulative=False shows underwater periods. Returns base64 PNG. Compare to benchmark to evaluate relative performance.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_plot_distribution(dataset: str = "returns") -> list[dict]:
        """Guide: Plotting return distribution."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
plot_distribution on {dataset}: Histogram with KDE overlay. Shows skewness and kurtosis. Fat tails visible as heavier bars beyond ±3σ. Compare to normal distribution overlay. Negative skew = left tail fatter (more crash risk).
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_plot_drawdown(dataset: str = "returns") -> list[dict]:
        """Guide: Plotting drawdown chart."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
plot_drawdown on {dataset}: Underwater chart showing drawdown depth over time. Deeper = worse. Longer = harder to recover. Look for: max drawdown depth, recovery time, drawdown clustering. Compare to benchmark drawdowns.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_plot_correlation(dataset: str = "multi_returns") -> list[dict]:
        """Guide: Plotting correlation heatmap."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
plot_correlation on {dataset}: Heatmap of pairwise correlations. method="pearson" (linear) or "spearman" (rank). Red = positive, blue = negative. Look for: uncorrelated pairs (diversification), highly correlated clusters (concentration risk).
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_plot_heatmap(dataset: str = "multi_returns") -> list[dict]:
        """Guide: Plotting generic heatmap."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
plot_heatmap on {dataset}: Correlation heatmap for multi-asset data. Shows pairwise relationships at a glance. Identify blocks of high correlation (sectors). Off-diagonal negative values are diversifiers. Compare to rolling correlation for stability.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_plot_candlestick(dataset: str = "prices") -> list[dict]:
        """Guide: Plotting OHLCV candlestick chart."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
plot_candlestick on {dataset}: Interactive candlestick chart with volume bars. Green = close > open (bullish). Red = close < open (bearish). Long wicks = rejection at that level. High volume on breakout = conviction. Use with support_resistance for level context.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_plot_factor_exposure(dataset: str = "returns") -> list[dict]:
        """Guide: Plotting factor exposure chart."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
plot_factor_exposure on {dataset}: Bar chart showing factor betas/loadings. Positive = long exposure, negative = short. Large bars = significant tilts. Compare to intended tilts. Use after factor_analysis to visualize where risk comes from.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_plot_rolling_metrics(dataset: str = "returns") -> list[dict]:
        """Guide: Plotting rolling performance metrics."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
plot_rolling_metrics on {dataset}: Rolling Sharpe, vol, and other metrics over time. Shows how strategy performance evolves. Degrading rolling Sharpe = edge decay. Stable = robust strategy. Identify which market conditions produce best/worst performance.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_plot_regime(dataset: str = "returns") -> list[dict]:
        """Guide: Plotting regime visualization."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
plot_regime on {dataset}: Colored bands showing regime states over time on the price chart. Bull = green, bear = red, crisis = orange. Visualize how regimes align with major market moves. Use after detect_regimes to communicate regime findings.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_plot_tearsheet(dataset: str = "backtest_results") -> list[dict]:
        """Guide: Plotting performance tearsheet."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
plot_tearsheet on {dataset}: Multi-panel performance report: equity curve, drawdowns, monthly returns, rolling metrics. The standard deliverable for strategy evaluation. Covers everything an investor needs to assess a strategy.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_plot_vol_surface() -> list[dict]:
        """Guide: Plotting volatility surface."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
plot_vol_surface: 3D surface of implied vol across strikes and maturities. Skew = lower strike → higher IV (crash protection demand). Term structure = IV across maturities. Smile = both wings elevated. Use for options strategy selection and relative value.
""",
                },
            }
        ]

    # ── fundamental/ (FMP-backed) ─────────────────────────────────────

    @mcp.prompt()
    def guide_company_profile(symbol: str = "AAPL") -> list[dict]:
        """Guide: Using the company_profile tool (FMP)."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
company_profile("{symbol}"): Fetches comprehensive company information from FMP — sector, industry, market cap, description, CEO, employees, website, IPO date, and key metrics.

Parameters:
- symbol: Stock ticker (e.g., "AAPL", "MSFT").

Interpretation:
- Use sector/industry to find peer companies for relative_valuation.
- Market cap classifies the company: mega (>200B), large (10-200B), mid (2-10B), small (<2B).
- Employee count + revenue per employee = productivity metric.
- IPO date tells you how much history is available.

Next steps:
1. financial_ratios("{symbol}") — deep-dive into profitability, leverage, valuation.
2. income_analysis("{symbol}") — revenue/margin trends over time.
3. stock_news("{symbol}") — recent news that may affect the thesis.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_financial_ratios_fmp(symbol: str = "AAPL") -> list[dict]:
        """Guide: Using the financial_ratios tool (FMP, symbol-based)."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
financial_ratios("{symbol}"): Computes comprehensive ratios from live FMP data — profitability (ROE, ROA, ROIC, margins), liquidity (current ratio, quick ratio), leverage (D/E, interest coverage), efficiency (asset turnover, inventory days), valuation (P/E, P/B, EV/EBITDA), and growth (revenue, earnings, FCF growth).

Parameters:
- symbol: Stock ticker.
- period: "annual" (default) or "quarter" for quarterly data.

Interpretation:
- Compare ratios to sector medians, NOT absolute thresholds. A 15 P/E is cheap for tech but expensive for utilities.
- ROE > 15% + D/E < 1 = quality value. High ROE with high D/E = leveraged returns (riskier).
- Current ratio > 1.5 = adequate liquidity. < 1 = potential liquidity stress.
- Interest coverage < 3 = watch for debt servicing risk.
- Declining margins over 3+ years = structural problem.

Next steps:
1. dupont_analysis("{symbol}") — decompose ROE into components.
2. relative_valuation("{symbol}") — compare multiples vs peers.
3. financial_health("{symbol}") — composite health score.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_income_analysis(symbol: str = "AAPL") -> list[dict]:
        """Guide: Using the income_analysis tool."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
income_analysis("{symbol}"): Analyzes multi-year income statement trends — revenue growth, gross/operating/net margins, and profitability trajectory.

Parameters:
- symbol: Stock ticker.
- period: "annual" (default) or "quarter".

Interpretation:
- Revenue growth: > 10% annually = strong growth. < 0% = declining business.
- Gross margin stability: Consistent GM = pricing power. Declining GM = competitive pressure.
- Operating margin: Should expand with scale. Contraction despite revenue growth = cost issues.
- Net margin: After-tax bottom line. Compare to peers — wide gaps suggest different business models.
- Watch the trend over 3-5 years, not just the latest year.

Next steps:
1. balance_sheet_analysis("{symbol}") — asset/liability context.
2. cash_flow_analysis("{symbol}") — verify earnings are backed by cash.
3. earnings_quality("{symbol}") — check if earnings are real or manipulated.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_balance_sheet_analysis(symbol: str = "AAPL") -> list[dict]:
        """Guide: Using the balance_sheet_analysis tool."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
balance_sheet_analysis("{symbol}"): Analyzes balance sheet composition — asset breakdown, liability structure, working capital, leverage ratios, and year-over-year changes.

Parameters:
- symbol: Stock ticker.
- period: "annual" (default) or "quarter".

Interpretation:
- Debt/Equity: < 0.5 = conservative, 0.5-1.5 = moderate, > 2 = aggressive leverage.
- Working capital trend: Declining WC + increasing revenue = efficient. Declining WC + flat revenue = cash squeeze.
- Goodwill/Total assets: > 30% = acquisition-heavy, impairment risk.
- Cash position: Should cover at least 6 months of operating expenses.
- Tangible book value: More reliable than book value for asset-heavy industries.

Next steps:
1. altman_z("{symbol}") — bankruptcy risk from balance sheet ratios.
2. financial_health("{symbol}") — composite score across all statements.
3. cash_flow_analysis("{symbol}") — cash generation context.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_cash_flow_analysis(symbol: str = "AAPL") -> list[dict]:
        """Guide: Using the cash_flow_analysis tool."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
cash_flow_analysis("{symbol}"): Analyzes cash flow statement — free cash flow, operating cash flow quality, capex intensity, and cash conversion.

Parameters:
- symbol: Stock ticker.
- period: "annual" (default) or "quarter".

Interpretation:
- FCF margin (FCF/Revenue): > 15% = strong cash generation. < 5% = capital-intensive.
- OCF/Net Income: Should be > 1.0. If consistently < 1.0, earnings are not backed by cash (red flag).
- Capex/Revenue: High ratio = capital-intensive business. Low = asset-light (typically higher valuations).
- FCF growth: Steady FCF growth supports dividend sustainability and buyback capacity.
- Watch for divergence between net income and OCF — accruals manipulation.

Next steps:
1. dcf_valuation("{symbol}") — use FCF projections for intrinsic value.
2. earnings_quality("{symbol}") — verify cash conversion quality.
3. dividend_history("{symbol}") — check if dividends are supported by FCF.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_dcf_valuation_fmp(symbol: str = "AAPL") -> list[dict]:
        """Guide: Using the dcf_valuation tool (FMP, symbol-based)."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
dcf_valuation("{symbol}"): Estimates intrinsic value per share using discounted cash flow analysis with live FMP financial data.

Parameters:
- symbol: Stock ticker.
- discount_rate: WACC or required return (default 0.10 = 10%).
- terminal_growth: Perpetual growth rate (default 0.025 = 2.5%).

Interpretation:
- Intrinsic value > market price → potentially undervalued (margin of safety > 0).
- Margin of safety > 30% = attractive entry. < 10% = fairly valued.
- HIGHLY sensitive to assumptions: ±1% in discount rate changes value 20-30%.
- Terminal growth MUST be < long-run GDP growth (2-3%). > 4% is unrealistic.
- Run sensitivity analysis: try discount_rate 0.08, 0.10, 0.12 with terminal_growth 0.02, 0.025, 0.03.

Next steps:
1. relative_valuation("{symbol}") — cross-check DCF with peer multiples.
2. cash_flow_analysis("{symbol}") — validate FCF projections used.
3. financial_ratios("{symbol}") — check if growth assumptions align with actual growth.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_relative_valuation(symbol: str = "AAPL") -> list[dict]:
        """Guide: Using the relative_valuation tool."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
relative_valuation("{symbol}"): Compares valuation multiples (P/E, P/B, EV/EBITDA, P/S) against peer companies with percentile ranking and implied fair values.

Parameters:
- symbol: Stock ticker.
- peers_json: Optional JSON list of peer tickers, e.g. '["MSFT", "GOOGL", "META"]'. If None, auto-selects same-sector peers.

Interpretation:
- Percentile < 25th = potentially cheap vs peers. > 75th = expensive vs peers.
- Implied fair value: What the stock would trade at if it had median peer multiples.
- P/E alone is misleading — EV/EBITDA is better for leverage-adjusted comparison.
- P/S is useful for unprofitable growth companies where P/E is meaningless.
- Discount to peers may be JUSTIFIED if growth or quality is lower.

Next steps:
1. dcf_valuation("{symbol}") — intrinsic value from cash flows (independent check).
2. financial_ratios("{symbol}") — understand WHY multiples differ from peers.
3. company_profile("{symbol}") — verify sector classification for peer selection.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_financial_health(symbol: str = "AAPL") -> list[dict]:
        """Guide: Using the financial_health tool."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
financial_health("{symbol}"): Computes a composite financial health score (0-100) with letter grade by combining profitability, leverage, liquidity, efficiency, and growth metrics.

Parameters:
- symbol: Stock ticker.

Interpretation:
- Score > 80 (A): Excellent financial health — strong across all dimensions.
- Score 60-80 (B): Good health, may have weaknesses in 1-2 areas.
- Score 40-60 (C): Average, notable concerns. Investigate the weakest sub-scores.
- Score < 40 (D/F): Poor health, multiple red flags. Check altman_z for distress risk.
- Sub-scores reveal WHERE the problems are: low profitability vs high leverage vs illiquidity.

Next steps:
1. piotroski_score("{symbol}") — binary scoring for additional perspective.
2. altman_z("{symbol}") — focused bankruptcy risk assessment.
3. earnings_quality("{symbol}") — verify reported numbers are trustworthy.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_earnings_quality(symbol: str = "AAPL") -> list[dict]:
        """Guide: Using the earnings_quality tool."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
earnings_quality("{symbol}"): Assesses whether reported earnings are real and sustainable — checks accruals ratio, cash conversion, earnings persistence, and red flags.

Parameters:
- symbol: Stock ticker.

Interpretation:
- Accruals ratio: Low accruals (< 5% of assets) = high quality. High accruals = earnings from accounting, not cash.
- Cash conversion (OCF/NI): Should be > 1.0. Consistently < 0.8 = earnings manipulation risk.
- Earnings persistence: High R-squared of earnings autoregression = sustainable. Low = volatile/one-time items.
- Revenue quality: Compare revenue growth to receivables growth. If receivables grow faster = channel stuffing risk.
- Red flags: negative FCF + positive NI, growing gap between NI and OCF, frequent "one-time" charges.

Next steps:
1. cash_flow_analysis("{symbol}") — deep-dive into cash flow patterns.
2. income_analysis("{symbol}") — margin trends that contextualize quality.
3. sec_filings("{symbol}") — read 10-K footnotes for accounting policy changes.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_dupont_analysis(symbol: str = "AAPL") -> list[dict]:
        """Guide: Using the dupont_analysis tool."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
dupont_analysis("{symbol}"): Decomposes ROE into its drivers using 3-way and 5-way DuPont analysis.

Parameters:
- symbol: Stock ticker.

3-way decomposition: ROE = Profit Margin x Asset Turnover x Equity Multiplier
- Profit Margin: Net income / Revenue. Higher = better pricing power.
- Asset Turnover: Revenue / Total Assets. Higher = more efficient asset use.
- Equity Multiplier: Total Assets / Equity. Higher = more leverage.

5-way adds: Tax Burden (NI/EBT) and Interest Burden (EBT/EBIT).

Interpretation:
- High ROE from high margin = quality. High ROE from high leverage = risky.
- Improving turnover with stable margin = operational improvement.
- Falling margin offset by rising leverage = deteriorating quality masked by debt.
- Compare components across years to spot where ROE changes originate.

Next steps:
1. financial_ratios("{symbol}") — full ratio context.
2. balance_sheet_analysis("{symbol}") — understand leverage component.
3. income_analysis("{symbol}") — understand margin component.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_stock_screener() -> list[dict]:
        """Guide: Using the stock_screener tool."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
stock_screener(criteria_json, top_n=20): Screens stocks by fundamental criteria using FMP data.

Parameters:
- criteria_json: JSON dict with min/max values, e.g.:
  '{"min_roe": 0.15, "max_pe": 25, "min_dividend_yield": 0.02, "min_market_cap": 1e9}'
- top_n: Maximum number of results (default 20).

Common screening recipes:
- Value: '{"max_pe": 15, "min_roe": 0.12, "max_debt_to_equity": 1.0}'
- Growth: '{"min_revenue_growth": 0.20, "min_earnings_growth": 0.15}'
- Dividend: '{"min_dividend_yield": 0.03, "min_payout_ratio": 0.2, "max_payout_ratio": 0.7}'
- Quality: '{"min_roe": 0.20, "min_roic": 0.15, "max_debt_to_equity": 0.5}'

Interpretation:
- Results are stored as "screener_results" dataset for further analysis.
- Screen is a starting point, not a buy signal. Always follow up with deep-dive analysis.
- Combine multiple criteria to narrow results. Start broad, then tighten.

Next steps:
1. financial_health(symbol) — score top results individually.
2. dcf_valuation(symbol) — estimate intrinsic values for top picks.
3. relative_valuation(symbol) — compare top picks against each other.
""",
                },
            }
        ]

    # ── news/ (FMP-backed) ────────────────────────────────────────────

    @mcp.prompt()
    def guide_stock_news(symbol: str = "AAPL") -> list[dict]:
        """Guide: Using the stock_news tool."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
stock_news("{symbol}"): Fetches recent news articles for a stock from FMP — headlines, dates, sources, and URLs.

Parameters:
- symbol: Stock ticker.
- limit: Number of articles to return (default 20, max ~100).

Interpretation:
- Headlines are stored as "news_{{symbol}}" dataset for further analysis.
- Scan for: earnings announcements, M&A activity, regulatory actions, analyst upgrades/downgrades.
- Cluster of negative headlines = potential sentiment overshoot (contrarian opportunity).
- Absence of news for a mid/large-cap = unusual, may precede announcement.

Next steps:
1. news_sentiment("{symbol}") — quantify sentiment from the articles.
2. earnings_data("{symbol}") — check if news relates to upcoming/recent earnings.
3. insider_activity("{symbol}") — see if insiders are acting on the news.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_news_sentiment_fmp(symbol: str = "AAPL") -> list[dict]:
        """Guide: Using the news_sentiment tool (FMP, symbol-based)."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
news_sentiment("{symbol}"): Analyzes sentiment of recent news articles — aggregate score (-1 to +1), trend direction, and bullish/bearish/neutral classification.

Parameters:
- symbol: Stock ticker.
- limit: Number of articles to analyze (default 50).

Interpretation:
- Score > 0.3 = bullish sentiment. < -0.3 = bearish. Between = neutral.
- Sentiment TREND matters more than level: improving sentiment from -0.5 to -0.2 is bullish.
- Extreme sentiment (> 0.7 or < -0.7) often precedes mean reversion — contrarian signal.
- Sentiment is most predictive for small/mid-caps where information diffuses slowly.
- For large-caps, sentiment reflects rather than predicts — use as confirmation.

Next steps:
1. sentiment_signal("{symbol}") — convert sentiment to a trading signal.
2. stock_news("{symbol}") — read the actual headlines for context.
3. earnings_data("{symbol}") — check if sentiment is earnings-driven.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_earnings_data(symbol: str = "AAPL") -> list[dict]:
        """Guide: Using the earnings_data tool."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
earnings_data("{symbol}"): Gets earnings history (actual vs estimate EPS) and upcoming earnings date.

Parameters:
- symbol: Stock ticker.

Interpretation:
- Beat rate: > 75% = company consistently exceeds expectations (management guides conservatively).
- Average surprise magnitude: Large positive surprises = analysts underestimate the business.
- PEAD (post-earnings announcement drift): Stocks that beat tend to continue drifting up for 30-60 days.
- Upcoming earnings: Important for timing entry/exit and managing event risk.
- Revenue beats matter too — EPS beats via cost-cutting without revenue growth are low quality.

Next steps:
1. earnings_surprises("{symbol}") — detailed surprise data per quarter.
2. earnings_quality("{symbol}") — verify the quality behind the numbers.
3. income_analysis("{symbol}") — longer-term trend context.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_insider_activity(symbol: str = "AAPL") -> list[dict]:
        """Guide: Using the insider_activity tool."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
insider_activity("{symbol}"): Analyzes recent insider trading — buys vs sells, buy/sell ratio, and notable large transactions.

Parameters:
- symbol: Stock ticker.
- limit: Number of transactions to analyze (default 50).

Interpretation:
- Insider BUYS are more informative than sells (insiders sell for many reasons, buy for one).
- Cluster buying by multiple insiders = strong bullish signal (they agree the stock is cheap).
- Buy/sell ratio > 2 = net insider accumulation. < 0.5 = net insider distribution.
- Large purchases (> $500K) by CEO/CFO carry more weight than small purchases by directors.
- Form 4 filings must be filed within 2 business days — data is near real-time.

Next steps:
1. company_profile("{symbol}") — context on who the insiders are.
2. financial_health("{symbol}") — check if insiders are buying a healthy company.
3. stock_news("{symbol}") — correlate insider activity with news flow.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_dividend_history(symbol: str = "AAPL") -> list[dict]:
        """Guide: Using the dividend_history tool."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
dividend_history("{symbol}"): Retrieves dividend payment history — yield, growth rate, payout ratio over time.

Parameters:
- symbol: Stock ticker.

Interpretation:
- Dividend growth > 5% annually for 10+ years = Dividend Aristocrat candidate.
- Payout ratio: 30-60% = sustainable. > 80% = limited growth, cut risk. > 100% = unsustainable.
- Yield > sector average + rising payout ratio = potential yield trap (dividend at risk).
- Ex-dividend date timing matters for income strategies.
- Compare dividend growth to earnings growth — if dividends grow faster, payout ratio is rising.

Next steps:
1. cash_flow_analysis("{symbol}") — verify FCF supports the dividend.
2. financial_health("{symbol}") — overall company stability.
3. earnings_data("{symbol}") — earnings trajectory supporting future dividends.
""",
                },
            }
        ]

    @mcp.prompt()
    def guide_sec_filings_fmp(symbol: str = "AAPL") -> list[dict]:
        """Guide: Using the sec_filings tool (FMP, symbol-based)."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
sec_filings("{symbol}"): Fetches recent SEC filings (10-K, 10-Q, 8-K, etc.) with dates and links.

Parameters:
- symbol: Stock ticker.
- form_type: Filter by type — "10-K" (annual), "10-Q" (quarterly), "8-K" (material events). None = all.
- limit: Number of filings (default 20).

Interpretation:
- 10-K: Annual report. Most comprehensive. Read risk factors, MD&A, and accounting policies.
- 10-Q: Quarterly report. Check for significant changes from prior quarter.
- 8-K: Material events — M&A, executive changes, covenant violations, restatements.
- Multiple 8-Ks in short period = something significant is happening.
- Filings stored as "filings_{{symbol}}" dataset for reference.

Next steps:
1. income_analysis("{symbol}") — quantitative view of what the filing describes.
2. earnings_quality("{symbol}") — check for accounting red flags noted in filings.
3. stock_news("{symbol}") — news coverage of filing contents.
""",
                },
            }
        ]
