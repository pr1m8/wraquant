"""Bayesian inference prompt templates."""

from __future__ import annotations

from typing import Any


def register_bayes_prompts(mcp: Any) -> None:

    @mcp.prompt()
    def bayesian_portfolio(dataset: str = "returns_universe") -> list[dict]:
        """Bayesian portfolio optimization with parameter uncertainty."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Perform Bayesian portfolio optimization on {dataset}. This uses bayes/ tools
(bayesian_portfolio, bayesian_sharpe, bayesian_regression) to construct portfolios
that account for estimation uncertainty rather than treating parameters as known.

---

## Phase 1: The Estimation Risk Problem

1. **Classical baseline**: For context, note the MVO (mean-variance optimization)
   problem: classical optimization treats estimated means and covariance as truth.
   But with N assets and T observations, estimation error in means is O(1/sqrt(T)).
   With 10 assets and 252 observations, the mean estimates are extremely noisy.
   MVO amplifies this noise -- it "maximizes estimation error."

2. **Bayesian advantage**: Instead of point estimates, Bayesian optimization
   samples from the posterior distribution of (mu, Sigma). Each posterior draw
   gives a different optimal portfolio. The average across draws accounts for
   parameter uncertainty. Result: more stable, diversified portfolios.

---

## Phase 2: Bayesian Estimation

3. **Posterior portfolio**: Run bayesian_portfolio on {dataset}.
   This uses a normal-inverse-Wishart conjugate prior and draws posterior samples
   of (mu, Sigma). For each draw, it solves the mean-variance problem.

   **Key outputs**:
   - **weights_mean**: Average optimal weights across posterior draws. These are
     the recommended allocations. More diversified than MVO.
   - **weights_std**: Uncertainty in each weight. Large std = we don't know
     the right allocation for this asset. Consider equal-weighting it instead.
   - **expected_return**: Posterior mean of portfolio return. More conservative than MVO.
   - **expected_risk**: Posterior mean of portfolio vol. More realistic than MVO.

4. **Bayesian Sharpe ratios**: For each asset, run bayesian_sharpe.
   Report for each asset:
   - **Posterior mean Sharpe**: Point estimate (similar to classical).
   - **Credible interval (95%)**: Range of plausible Sharpe ratios.
     Wide interval = high uncertainty. Narrow = reliable estimate.
   - **Prob(Sharpe > 0)**: Probability the asset has positive risk-adjusted return.
     < 70% = not confident this asset adds value.
   - **Prob(Sharpe > 0.5)**: Probability of a "good" Sharpe.

   Assets with wide credible intervals should get lower portfolio weight
   (the Bayesian optimizer does this automatically).

---

## Phase 3: Factor Model Enhancement

5. **Bayesian factor model**: If factor data available, run bayesian_regression
   with each asset's returns as Y and factors as X.
   - Posterior mean of beta: Factor exposure with uncertainty.
   - Posterior std of beta: How uncertain is the exposure estimate?
   - log_marginal_likelihood: Model evidence. Higher = better model.

   **Factor-based Bayesian portfolio**:
   - Use the posterior distribution of factor exposures to construct
     a factor-tilted portfolio that accounts for exposure uncertainty.
   - Assets with uncertain factor exposures get lower tilt.

6. **Shrinkage interpretation**: The Bayesian posterior naturally "shrinks"
   extreme estimates toward the prior. This is equivalent to:
   - Ledoit-Wolf shrinkage for covariance.
   - James-Stein shrinkage for means.
   But the Bayesian approach provides the full posterior, not just a point estimate.

---

## Phase 4: Robustness Analysis

7. **Prior sensitivity**: Run bayesian_portfolio with different n_samples.
   The prior is the conjugate normal-inverse-Wishart. With the default
   (uninformative) prior:
   - Results should converge as n_samples increases (2000+ is sufficient).
   - If results change dramatically with sample size, the posterior is diffuse
     and we need more data or a stronger prior.

8. **Comparison to classical**: Compare Bayesian vs classical MVO weights.
   - Bayesian weights are typically more diversified (closer to equal-weight).
   - Extreme long/short positions in MVO are shrunk toward zero in Bayesian.
   - Out-of-sample, Bayesian portfolios typically have lower turnover and
     higher risk-adjusted returns (Sharpe improvement of 0.1-0.3).

9. **Black-Litterman connection**: The Black-Litterman model is a special case
   of Bayesian portfolio optimization where the prior is the market equilibrium.
   Bayesian_portfolio uses an uninformative prior; BL uses an informative one.
   Consider using BL if you have strong market-implied prior beliefs.

---

## Phase 5: Portfolio Report

10. **Bayesian portfolio summary**:

    | Asset | Weight (Mean) | Weight (Std) | Sharpe (Post. Mean) | 95% CI |
    |-------|--------------|-------------|--------------------|---------|
    | ... | ... | ... | ... | [..., ...] |

11. **Key metrics**:
    - Expected portfolio return (posterior mean).
    - Expected portfolio vol (posterior mean).
    - Weight uncertainty: average weight std across assets.
    - Diversification: effective number of assets = 1 / sum(w_i^2).
    - Comparison: Bayesian Sharpe vs classical Sharpe.

12. **Recommendations**:
    - Use Bayesian weights for execution (more stable).
    - Monitor assets with high weight uncertainty.
    - Rebalancing frequency: Bayesian portfolios are more stable,
      so less frequent rebalancing is needed.

**Related prompts**: Use model_selection for comparing portfolio models,
risk_report for portfolio risk analysis, bayesian_regime_detection for
regime-aware Bayesian allocation.
""",
                },
            }
        ]

    @mcp.prompt()
    def bayesian_regime_detection(dataset: str = "returns") -> list[dict]:
        """Bayesian changepoint detection and regime analysis with full posterior."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Perform Bayesian regime detection on {dataset}. This uses bayes/ tools
(bayesian_changepoint, bayesian_volatility) alongside regimes/ tools to provide
a full probabilistic view of regime changes and volatility dynamics.

---

## Phase 1: Bayesian Changepoint Detection

1. **Online changepoint detection**: Run bayesian_changepoint on {dataset}.
   Adams & MacKay (2007) algorithm maintains a posterior over the "run length"
   (time since last changepoint) at each observation.

   **Parameter guidance**:
   - hazard=250: Expect changepoints roughly every 250 observations (~ 1 year).
     Lower = expect more frequent changes (more changepoints detected).
     Higher = expect fewer changes (only major regime shifts detected).
   - threshold=0.3: Declare a changepoint when posterior probability exceeds 30%.
     Lower = more sensitive (more false positives).
     Higher = more conservative (miss subtle changes).

2. **Interpret changepoints**: From the output:
   - **n_changepoints**: How many regime changes detected?
   - **changepoint_indices**: Where in the series do they occur?
   - Map indices to dates for economic interpretation.
   - Compare to known events: COVID, GFC, rate hikes, elections.

3. **Comparison to frequentist**: Run detect_regimes with method="hmm" on the
   same data. Compare HMM regime transition dates to Bayesian changepoint dates.
   - Agreement: Strong evidence of regime change.
   - Disagreement: The methods are detecting different types of changes
     (HMM: gradual regime shift. Bayesian CP: sudden break).
   Also run structural_break from econometrics for a third comparison.

---

## Phase 2: Bayesian Volatility Estimation

4. **Stochastic volatility**: Run bayesian_volatility on {dataset}.
   This estimates a time-varying volatility path with full uncertainty bands
   using MCMC (Metropolis-within-Gibbs sampling).

   **Key outputs**:
   - **vol_mean**: Posterior mean of the volatility path. Smoother than
     rolling window or GARCH (less noisy).
   - **vol_ci_lower / vol_ci_upper**: 95% credible interval. Width of the
     band shows uncertainty in the vol estimate.
   - **phi_mean**: Persistence of log-volatility. High phi (> 0.95) = very
     persistent vol (similar to high GARCH persistence).
   - **sigma_eta_mean**: Vol-of-vol. Higher = more volatile volatility path.

5. **Compare to GARCH**: Run fit_garch on the same data.
   - Bayesian SV provides uncertainty bands; GARCH gives a point estimate.
   - Bayesian SV is a continuous latent process; GARCH is a discrete recursion.
   - When do the Bayesian credible intervals NOT contain the GARCH estimate?
     These are periods where the models disagree most.
   - Bayesian SV often captures vol regime shifts more smoothly.

---

## Phase 3: Regime Characterization

6. **Sub-regime statistics**: Split {dataset} at the detected changepoint dates.
   For each sub-period:
   - Run analyze() for mean, vol, skewness, kurtosis.
   - Run bayesian_sharpe for posterior Sharpe with uncertainty.
   - Run risk_metrics for drawdown and tail risk.

7. **Regime-conditional volatility**: Is the Bayesian vol path different in
   each regime? Compare vol_mean in each sub-period.
   - Regime with higher posterior vol = "risk-off" state.
   - Regime with lower posterior vol = "risk-on" state.
   - The vol credible interval width may also change across regimes
     (wider in crisis = more uncertain about the vol level).

8. **Transition dynamics**: The Bayesian changepoint model gives run-length
   probabilities. The current run length posterior tells you:
   - How long the current regime has lasted (mode of run length distribution).
   - How likely a change is soon (probability mass at run length = 0).
   - If P(run_length=0) is rising, a regime change may be imminent.

---

## Phase 4: Forward-Looking Signals

9. **Current regime assessment**:
   - What is the current run length (mode of posterior)?
   - Is the changepoint probability rising?
   - Is Bayesian vol at the upper or lower credible bound?
   - Combined signal: Current regime stability assessment.

10. **Regime-aware allocation**: Based on the current regime:
    - Low-vol regime: Higher equity allocation, risk-on.
    - High-vol regime: Lower equity, higher bonds/gold, risk-off.
    - Transition period (high changepoint probability): Reduce all positions,
      increase cash. Wait for the new regime to establish.

---

## Phase 5: Summary

11. **Bayesian regime report**:

    | Period | Start | End | Mean Return | Vol (Post. Mean) | Sharpe (95% CI) |
    |--------|-------|-----|------------|-----------------|-----------------|
    | Regime 1 | ... | ... | ... | ... | [..., ...] |
    | Regime 2 | ... | ... | ... | ... | [..., ...] |

12. **Key findings**:
    - Number and timing of regime changes.
    - Current regime and stability assessment.
    - Bayesian vs frequentist consistency.
    - Volatility path with uncertainty bands.
    - Forward-looking regime change probability.

**Related prompts**: Use detect_regimes for HMM-based regimes,
structural_break_analysis for formal break tests,
bayesian_portfolio for regime-aware allocation.
""",
                },
            }
        ]

    @mcp.prompt()
    def bayesian_factor_model(
        dataset: str = "returns",
        factors_dataset: str = "factor_returns",
    ) -> list[dict]:
        """Bayesian factor model with posterior inference on exposures and alpha."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Estimate a Bayesian factor model for {dataset} using {factors_dataset}.
This uses bayes/ tools (bayesian_regression, bayesian_sharpe) to provide
full posterior distributions over factor exposures, alpha, and residual risk.

---

## Phase 1: Factor Model Setup

1. **Data preparation**: Load {dataset} (asset returns) and {factors_dataset}
   (factor returns, e.g., Fama-French, Carhart, or custom factors).
   Run align_datasets to ensure date alignment.
   Run analyze() on each factor to check stationarity and basic stats.

2. **Classical regression baseline**: For context, run a standard OLS regression
   (via analyze or correlation_analysis) of asset returns on factor returns.
   This gives point estimates of:
   - Alpha: Excess return not explained by factors.
   - Beta (factor exposures): Sensitivity to each factor.
   - R-squared: How much return variation the factors explain.

---

## Phase 2: Bayesian Estimation

3. **Bayesian regression**: Run bayesian_regression with:
   - y_column = asset returns column name
   - x_columns_json = factor column names
   - n_samples = 2000

   **Key outputs**:
   - **posterior_mean**: Bayesian point estimates of [intercept (alpha), beta1, beta2, ...].
   - **posterior_std**: Uncertainty in each coefficient.
   - **log_marginal_likelihood**: Model evidence for comparison.

   **Interpretation**:
   - Alpha posterior_mean: Bayesian estimate of excess return.
     If the 95% credible interval includes 0, alpha is not significant.
   - Beta posterior_std: Uncertainty in factor exposure.
     Wide uncertainty = we don't know the true exposure. Use the prior mean
     or shrink toward it.

4. **Alpha significance**: Run bayesian_sharpe on the regression residuals.
   If the residual Sharpe credible interval includes 0, the alpha is not
   reliably positive. The probability of positive Sharpe = probability of
   genuine alpha after accounting for factor exposures.

---

## Phase 3: Multi-Asset Factor Model

5. **Cross-sectional estimation**: If {dataset} has multiple assets (columns),
   run bayesian_regression for each asset separately.
   Build a posterior distribution of factor exposures for the entire cross-section.

   | Asset | Alpha (mean) | Alpha (95% CI) | Beta_MKT (mean) | Beta_MKT (std) |
   |-------|-------------|----------------|-----------------|----------------|
   | ... | ... | [..., ...] | ... | ... |

6. **Factor exposure uncertainty**: For portfolio construction:
   - Assets with tight beta credible intervals: Reliable factor tilts.
   - Assets with wide beta credible intervals: Uncertain exposures.
     Weight these lower in factor-tilted portfolios.

---

## Phase 4: Model Comparison

7. **Model selection**: Run model_comparison_bayesian to compare factor models.
   Example specifications:
   - Model 1: CAPM (market factor only)
   - Model 2: Fama-French 3-factor (market, size, value)
   - Model 3: Carhart 4-factor (+ momentum)
   - Model 4: 5-factor (+ profitability, investment)

   **Comparison via**:
   - log_marginal_likelihood: Higher = better model (Occam's razor built in).
   - Bayes factor: Ratio of marginal likelihoods. > 10 = strong evidence.
   - The winning model balances explanatory power with parsimony.

8. **Factor redundancy**: If adding a factor doesn't improve the marginal
   likelihood, it's redundant. The Bayesian framework automatically penalizes
   model complexity, so overfitting is less of a concern than in classical F-tests.

---

## Phase 5: Applications

9. **Factor-based risk decomposition**: Using the posterior mean exposures:
   - Systematic risk: sum(beta_i^2 * var(factor_i)) + cross terms.
   - Idiosyncratic risk: residual variance.
   - Risk contribution of each factor to total portfolio risk.

10. **Factor timing**: Run bayesian_regression on rolling windows.
    Do factor exposures change over time? If so:
    - Rising market beta in bull markets = procyclical exposure.
    - Time-varying alpha: Is alpha concentrated in certain periods?

11. **Summary**:
    - Best factor model (by marginal likelihood).
    - Alpha: Significant? Posterior mean and credible interval.
    - Factor exposures with uncertainty bands.
    - Risk decomposition: systematic vs idiosyncratic.
    - Portfolio implication: Which factors to tilt toward/away from.

**Related prompts**: Use model_selection for deeper model comparison,
bayesian_portfolio for factor-informed portfolio construction.
""",
                },
            }
        ]

    @mcp.prompt()
    def model_selection(dataset: str = "returns") -> list[dict]:
        """Bayesian model comparison and selection via marginal likelihood."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Perform Bayesian model comparison on {dataset}. This uses bayes/ tools
(model_comparison_bayesian, bayesian_regression) to systematically compare
candidate models using rigorous Bayesian criteria.

---

## Phase 1: Candidate Models

1. **Define model specifications**: Identify the candidate models to compare.
   For return prediction / factor models:
   - **Model 1 (Null)**: Intercept only (no predictors). Baseline.
   - **Model 2 (CAPM)**: Market factor only.
   - **Model 3 (FF3)**: Market + size + value.
   - **Model 4 (FF5)**: Market + size + value + profitability + investment.
   - **Model 5 (Custom)**: Your own factor set.

   For time series models:
   - AR(1), AR(2), AR(3), etc.
   - GARCH vs EGARCH vs GJR (compare fit_garch AIC/BIC).

2. **Prepare factor data**: Ensure all factor columns are in {dataset} or a
   separate factors dataset. Run analyze() on each factor for basic stats.

---

## Phase 2: Bayesian Model Comparison

3. **Run comparison**: Run model_comparison_bayesian with:
   - dataset = {dataset}
   - column = dependent variable (e.g., "returns")
   - models_json = JSON array of model specs, each with 'name' and 'x_columns'.

   This fits each model via conjugate Bayesian regression and computes:
   - **log_marginal_likelihood**: The key comparison criterion. Higher = better.
     This integrates over parameter uncertainty (not just max likelihood).
   - **Bayes factor**: Ratio of marginal likelihoods between any two models.

4. **Bayes factor interpretation**:
   - BF > 100: Decisive evidence for the better model.
   - BF 10-100: Strong evidence.
   - BF 3-10: Moderate evidence.
   - BF 1-3: Weak evidence (models are similar).
   - BF < 1: Evidence favors the other model.

---

## Phase 3: Detailed Model Assessment

5. **Individual model fits**: For the top 2-3 models, run bayesian_regression
   separately to get full posterior distributions.
   - Report posterior mean and credible intervals for each coefficient.
   - Identify which coefficients' credible intervals exclude 0 (significant).
   - Check posterior_std: Very large = model struggles with that parameter.

6. **Posterior predictive checks**: For each model:
   - Generate posterior predictions and compare to observed data.
   - If predictions are systematically biased, the model is misspecified.
   - Check if prediction intervals are well-calibrated (95% of observations
     should fall within the 95% prediction interval).

7. **Model averaging**: If no single model dominates:
   - Compute model weights: w_k = p(M_k|data) = marginal_likelihood_k / sum(all).
   - Bayesian Model Averaging (BMA) predictions = weighted average across models.
   - BMA is more robust than picking a single model.

---

## Phase 4: Robustness

8. **Prior sensitivity**: For the winning model, run bayesian_regression with
   different sample sizes (subsetting the data) to check stability.
   - Do results change with sample size? If yes, data is insufficient.
   - Does the model ranking change? If yes, evidence is not strong enough.

9. **Out-of-sample validation**: Split {dataset} into train (70%) and test (30%).
   Use split_dataset to create the split.
   - Fit all models on train data.
   - Compute predictive likelihood on test data.
   - Does the in-sample ranking match the out-of-sample ranking?
   - If not, the in-sample winner may be overfitting.

---

## Phase 5: Summary

10. **Model ranking table**:

    | Model | log ML | Bayes Factor vs Null | # Params | Interpretation |
    |-------|--------|---------------------|----------|----------------|
    | Null | ... | 1.0 (reference) | 1 | No predictors |
    | CAPM | ... | ... | 2 | Market only |
    | FF3 | ... | ... | 4 | + Size, Value |
    | FF5 | ... | ... | 6 | + Prof, Inv |

11. **Recommendation**:
    - Winning model and strength of evidence.
    - Should we use a single model or model averaging?
    - Out-of-sample validation results.
    - Practical implications: Which factors matter? Which are redundant?

**Related prompts**: Use bayesian_factor_model for deeper factor analysis,
bayesian_portfolio for model-informed portfolio construction.
""",
                },
            }
        ]

    @mcp.prompt()
    def bayesian_vol(dataset: str = "returns") -> list[dict]:
        """Bayesian stochastic volatility estimation with full posterior inference."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Estimate a Bayesian stochastic volatility model for {dataset}. This uses
bayes/ tools (bayesian_volatility, bayesian_changepoint) to extract the
latent volatility path with full uncertainty quantification.

---

## Phase 1: The Stochastic Volatility Model

1. **Model specification**: The standard SV model is:
   r_t = exp(h_t / 2) * epsilon_t,  epsilon_t ~ N(0, 1)
   h_t = mu + phi * (h_{{t-1}} - mu) + eta_t,  eta_t ~ N(0, sigma_eta^2)

   Where h_t is the log-volatility process. The parameters are:
   - **phi**: Persistence of log-vol. High phi = vol is sticky.
   - **sigma_eta**: Vol-of-vol. How much the volatility process itself fluctuates.
   - **mu**: Long-run mean of log-vol.

2. **Why Bayesian SV over GARCH?**:
   - SV treats volatility as a continuous latent variable (more realistic).
   - GARCH treats volatility as a deterministic function of past returns.
   - Bayesian SV provides uncertainty bands on the vol path (GARCH does not).
   - SV can capture "volatility surprise" -- vol moves independent of returns.

---

## Phase 2: MCMC Estimation

3. **Fit the model**: Run bayesian_volatility on {dataset}.
   - n_samples=1000: Minimum for reliable estimates.
   - n_samples=5000: Better for publication-quality inference.
   - n_samples=10000: Use if MCMC mixing is poor.

   **Key outputs**:
   - **vol_mean**: Posterior mean of the volatility path. This is the
     "best estimate" of daily annualized vol at each time point.
   - **vol_ci_lower / vol_ci_upper**: 95% credible interval. Width shows
     how uncertain we are about the vol level.
   - **mean_volatility**: Time-averaged vol across the entire sample.
   - **current_volatility**: Most recent vol estimate (for live risk management).
   - **phi_mean**: Posterior mean of persistence. > 0.95 is typical.
   - **sigma_eta_mean**: Posterior mean of vol-of-vol.

---

## Phase 3: Posterior Analysis

4. **Volatility path interpretation**:
   - Compare vol_mean to the GARCH conditional vol (from fit_garch).
     Where do they agree? Where do they diverge?
   - Bayesian SV is typically smoother (less noisy) than GARCH.
   - Wide credible intervals = high uncertainty about the vol level.
     Narrow intervals = confident vol estimate.

5. **Persistence analysis**: From phi_mean:
   - phi > 0.98: Near-integrated vol. Vol shocks are very persistent.
     Long half-life. Carry exposure slowly.
   - phi 0.90-0.98: Standard persistence. Vol mean-reverts over weeks.
   - phi < 0.90: Low persistence. Vol shocks die quickly.
     Can increase position faster after a vol spike.

6. **Vol-of-vol**: From sigma_eta_mean:
   - High sigma_eta: Vol itself is volatile. Difficult to forecast.
     VaR and risk limits should be wider.
   - Low sigma_eta: Vol is stable. More reliable risk forecasts.

---

## Phase 4: Comparison & Validation

7. **GARCH comparison**: Run fit_garch on the same dataset.
   Compare:
   - Time-averaged vol: Should be similar between SV and GARCH.
   - Vol dynamics: Does SV detect vol changes that GARCH misses?
   - Current vol: Are they in agreement on the current level?
   - Model fit: Compare via AIC/BIC for GARCH and log marginal likelihood
     for Bayesian SV (not directly comparable, but informative).

8. **Realized vol benchmark**: If OHLCV data available, compute
   realized volatility (e.g., Yang-Zhang estimator) as a "ground truth" proxy.
   Compare the Bayesian SV vol path to the realized vol.
   - SV should track realized vol well on average.
   - SV may be smoother (less noisy) than daily realized vol.

9. **Changepoint detection**: Run bayesian_changepoint on the SV vol path itself.
   Are there structural breaks in the volatility process?
   - Vol regime changes: periods where the vol level shifts permanently.
   - These are different from temporary vol spikes (captured by the SV model).

---

## Phase 5: Risk Management Applications

10. **VaR from Bayesian SV**: Using the current vol estimate (with uncertainty):
    - Point VaR: Use vol_mean for the standard VaR calculation.
    - Conservative VaR: Use vol_ci_upper for a worst-case vol assumption.
    - The range between these two is the "model uncertainty premium."

11. **Dynamic position sizing**: Based on the current Bayesian vol estimate:
    - Target constant-vol portfolio: position_size = target_vol / current_vol.
    - When vol_ci is wide (uncertain), use smaller positions.
    - When vol_ci is narrow (confident), can use full position size.

12. **Summary**:
    - Current vol estimate with 95% credible interval.
    - Persistence (phi) and vol-of-vol (sigma_eta).
    - Comparison to GARCH: agreement or disagreement?
    - Vol regime: stable, rising, falling, or uncertain?
    - Risk management recommendation: position sizing and VaR adjustment.

**Related prompts**: Use volatility_deep_dive for GARCH-based analysis,
bayesian_regime_detection for vol regime changes,
risk_report for comprehensive risk assessment.
""",
                },
            }
        ]
