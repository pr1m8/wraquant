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
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_survival_analysis(dataset: str = "returns") -> list[dict]:
        """Guide: Survival analysis for drawdown duration."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Survival analysis on {dataset}:

1. Identify drawdown periods -- when cumulative returns are negative.
2. Call survival_analysis(dataset="{dataset}", column="duration", method="kaplan_meier").
3. The survival curve shows: what fraction of drawdowns last at least N days?
4. Median survival time = typical drawdown duration.
5. If hazard rate is decreasing: drawdowns that last longer tend to recover faster (good).
6. If hazard rate is increasing: the longer you're in drawdown, the worse it gets (bad).

Use for: estimating how long to wait before cutting losses.
"""}}]

    @mcp.prompt()
    def guide_monte_carlo_var(dataset: str = "returns") -> list[dict]:
        """Guide: Monte Carlo VaR estimation."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_dcc_correlation(dataset: str = "multi_asset_returns") -> list[dict]:
        """Guide: DCC-GARCH dynamic conditional correlations."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_expected_shortfall_decomposition(dataset: str = "multi_asset_returns") -> list[dict]:
        """Guide: ES decomposition into per-asset contributions."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_cornish_fisher_var(dataset: str = "returns") -> list[dict]:
        """Guide: Cornish-Fisher VaR adjusted for higher moments."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_rolling_beta(dataset: str = "asset_returns", benchmark: str = "market_returns") -> list[dict]:
        """Guide: Rolling beta for tracking systematic risk drift."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    # ── vol/ ─────────────────────────────────────────────────────────

    @mcp.prompt()
    def guide_hawkes_fit(dataset: str = "returns") -> list[dict]:
        """Guide: Hawkes process for volatility clustering."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_stochastic_vol(dataset: str = "returns") -> list[dict]:
        """Guide: Stochastic volatility model via particle filter."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_variance_risk_premium(dataset: str = "vol_data") -> list[dict]:
        """Guide: Variance risk premium (implied minus realized)."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_bipower_variation(dataset: str = "returns") -> list[dict]:
        """Guide: Bipower variation for jump-robust volatility."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_jump_detection(dataset: str = "returns") -> list[dict]:
        """Guide: Barndorff-Nielsen-Shephard jump test."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_garch_rolling(dataset: str = "returns") -> list[dict]:
        """Guide: Rolling GARCH volatility forecast."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    # ── stats/ ───────────────────────────────────────────────────────

    @mcp.prompt()
    def guide_partial_correlation(dataset: str = "multi_asset_returns") -> list[dict]:
        """Guide: Partial correlation to find direct relationships."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_distance_correlation(dataset: str = "returns", col_a: str = "asset_a", col_b: str = "asset_b") -> list[dict]:
        """Guide: Distance correlation for nonlinear dependence."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_mutual_information(dataset: str = "returns", col_a: str = "feature", col_b: str = "target") -> list[dict]:
        """Guide: Mutual information for general dependence."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_kde_estimate(dataset: str = "returns") -> list[dict]:
        """Guide: Kernel density estimation for return distributions."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_best_fit_distribution(dataset: str = "returns") -> list[dict]:
        """Guide: Automated distribution fitting and selection."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    # ── ts/ ──────────────────────────────────────────────────────────

    @mcp.prompt()
    def guide_ssa_decompose(dataset: str = "prices") -> list[dict]:
        """Guide: Singular Spectrum Analysis decomposition."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_arima_diagnostics(dataset: str = "returns") -> list[dict]:
        """Guide: ARIMA model fitting and residual diagnostics."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_rolling_forecast(dataset: str = "returns") -> list[dict]:
        """Guide: Walk-forward rolling forecast evaluation."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_ensemble_forecast(dataset: str = "returns") -> list[dict]:
        """Guide: Ensemble forecast combining multiple methods."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_ornstein_uhlenbeck(dataset: str = "spread") -> list[dict]:
        """Guide: Ornstein-Uhlenbeck mean-reversion estimation."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    # ── backtest/ ────────────────────────────────────────────────────

    @mcp.prompt()
    def guide_omega_ratio(dataset: str = "strategy_returns") -> list[dict]:
        """Guide: Omega ratio -- gain/loss probability-weighted ratio."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_kelly_fraction(dataset: str = "strategy_returns") -> list[dict]:
        """Guide: Kelly criterion for optimal position sizing."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_vectorized_backtest(dataset: str = "prices", signals: str = "signals") -> list[dict]:
        """Guide: Fast vectorized backtest with transaction costs."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_drawdown_analysis(dataset: str = "strategy_returns") -> list[dict]:
        """Guide: Detailed drawdown period analysis."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    # ── ml/ ──────────────────────────────────────────────────────────

    @mcp.prompt()
    def guide_pca_factors(dataset: str = "multi_asset_returns") -> list[dict]:
        """Guide: PCA factor extraction from multi-asset returns."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_isolation_forest(dataset: str = "returns") -> list[dict]:
        """Guide: Isolation Forest anomaly detection."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_svm_classify(dataset: str = "features") -> list[dict]:
        """Guide: SVM classification for regime or signal prediction."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_online_regression(dataset: str = "returns") -> list[dict]:
        """Guide: Online regression with time-varying coefficients."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_cross_asset_features(dataset: str = "asset_returns", benchmark: str = "market_returns") -> list[dict]:
        """Guide: Cross-asset feature construction."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    # ── price/ ───────────────────────────────────────────────────────

    @mcp.prompt()
    def guide_implied_volatility(market_price: float = 10.0, spot: float = 100.0, strike: float = 100.0) -> list[dict]:
        """Guide: Computing implied volatility from option prices."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_sabr_calibrate() -> list[dict]:
        """Guide: SABR model calibration to market vol smile."""
        return [{"role": "user", "content": {"type": "text", "text": """
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
"""}}]

    @mcp.prompt()
    def guide_simulate_heston(spot: float = 100.0) -> list[dict]:
        """Guide: Heston stochastic volatility simulation."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_bond_duration() -> list[dict]:
        """Guide: Bond duration and convexity analysis."""
        return [{"role": "user", "content": {"type": "text", "text": """
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
"""}}]

    @mcp.prompt()
    def guide_fbsde_price() -> list[dict]:
        """Guide: Forward-Backward SDE option pricing."""
        return [{"role": "user", "content": {"type": "text", "text": """
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
"""}}]

    # ── opt/ ─────────────────────────────────────────────────────────

    @mcp.prompt()
    def guide_black_litterman(dataset: str = "multi_asset_returns") -> list[dict]:
        """Guide: Black-Litterman portfolio optimization."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_hierarchical_risk_parity(dataset: str = "multi_asset_returns") -> list[dict]:
        """Guide: HRP portfolio optimization."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_min_volatility(dataset: str = "multi_asset_returns") -> list[dict]:
        """Guide: Minimum variance portfolio."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_max_sharpe(dataset: str = "multi_asset_returns") -> list[dict]:
        """Guide: Maximum Sharpe ratio (tangency) portfolio."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_risk_budgeting(dataset: str = "multi_asset_returns") -> list[dict]:
        """Guide: Risk budgeting portfolio optimization."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    # ── microstructure/ ──────────────────────────────────────────────

    @mcp.prompt()
    def guide_spread_decomposition(dataset: str = "tick_data") -> list[dict]:
        """Guide: Huang-Stoll spread decomposition."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_price_impact(dataset: str = "trade_data") -> list[dict]:
        """Guide: Measuring permanent vs temporary price impact."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_depth_analysis(dataset: str = "orderbook_data") -> list[dict]:
        """Guide: Order book depth imbalance analysis."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    # ── execution/ ───────────────────────────────────────────────────

    @mcp.prompt()
    def guide_transaction_cost_analysis(dataset: str = "market_data") -> list[dict]:
        """Guide: Post-trade TCA benchmarking."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_close_auction() -> list[dict]:
        """Guide: Close auction allocation strategy."""
        return [{"role": "user", "content": {"type": "text", "text": """
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
"""}}]

    # ── math/ ────────────────────────────────────────────────────────

    @mcp.prompt()
    def guide_systemic_risk(dataset: str = "multi_asset_returns") -> list[dict]:
        """Guide: Systemic risk scoring via Marginal Expected Shortfall."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_math_hawkes_fit(dataset: str = "event_times") -> list[dict]:
        """Guide: Hawkes process fit for event arrival times."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    @mcp.prompt()
    def guide_spectral_analysis(dataset: str = "prices") -> list[dict]:
        """Guide: FFT spectral analysis for cyclical patterns."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    # ── experiment/ ──────────────────────────────────────────────────

    @mcp.prompt()
    def guide_experiment_comparison() -> list[dict]:
        """Guide: Comparing multiple strategy experiments."""
        return [{"role": "user", "content": {"type": "text", "text": """
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
"""}}]

    @mcp.prompt()
    def guide_parameter_sensitivity(name: str = "my_experiment", param: str = "period") -> list[dict]:
        """Guide: Parameter sensitivity analysis for a strategy experiment."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
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
"""}}]

    # ── workspace ────────────────────────────────────────────────────

    @mcp.prompt()
    def guide_snapshot() -> list[dict]:
        """Guide: Creating and restoring workspace snapshots."""
        return [{"role": "user", "content": {"type": "text", "text": """
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
"""}}]

    @mcp.prompt()
    def guide_restore_snapshot() -> list[dict]:
        """Guide: Restoring a workspace from a snapshot."""
        return [{"role": "user", "content": {"type": "text", "text": """
Restore snapshot:

1. Call restore_snapshot(name="snapshot_name") to revert the workspace.
2. All datasets and models are replaced with the snapshot versions.
3. Any data created after the snapshot is lost.

When to restore:
- An experiment produced bad results and you want to try again.
- You accidentally deleted or overwrote a dataset.
- You want to compare current state to a previous state.
- Use workspace_status() after restore to verify the state.
"""}}]

    @mcp.prompt()
    def guide_query_data() -> list[dict]:
        """Guide: SQL queries on workspace data via DuckDB."""
        return [{"role": "user", "content": {"type": "text", "text": """
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
"""}}]
