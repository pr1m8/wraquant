"""Econometrics prompt templates."""

from __future__ import annotations

from typing import Any


def register_econometrics_prompts(mcp: Any) -> None:

    @mcp.prompt()
    def panel_data_analysis(dataset: str = "panel_data") -> list[dict]:
        """Panel data regression workflow: pooled OLS, fixed effects, random effects."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Perform a comprehensive panel data analysis on {dataset}. This workflow uses the
econometrics/ tools to estimate, compare, and diagnose panel regression models.

---

## Phase 1: Data Exploration

1. **Panel structure**: Run describe_dataset on {dataset} to understand shape.
   Identify: entity_col (firm, country, sector), time_col (year, quarter, date),
   dependent variable (y_col), and regressors (x_cols).
   - How many entities (N)? How many time periods (T)?
   - Is it balanced (same T for all N) or unbalanced?
   - Large N, small T = micro panel. Small N, large T = macro panel.

2. **Data quality**: Use query_data to inspect for:
   - Missing values: If > 10% missing, consider imputation or drop.
   - Outliers: Winsorize at 1%/99% if extreme observations present.
   - Within vs between variation: Do regressors vary more across entities or over time?
     If mostly between-variation, FE will absorb most signal -- RE may be better.

---

## Phase 2: Model Estimation

3. **Pooled OLS baseline**: Run panel_regression with method="pooled".
   This ignores the panel structure entirely. Coefficients are the average
   relationship across all entities and time periods.
   - Note R-squared, coefficient signs, and magnitudes.
   - This is biased if entity-specific effects are correlated with regressors.

4. **Fixed effects (within estimator)**: Run panel_regression with method="fe".
   This demeans each entity, removing all time-invariant unobserved heterogeneity.
   - **Interpret coefficient changes vs pooled**: Large change = omitted variable bias
     in pooled OLS. The FE coefficient isolates within-entity variation.
   - R-squared within: How much within-entity variation does the model explain?
   - Coefficients on time-invariant variables are not identified (dropped by FE).

5. **Random effects (GLS)**: Run panel_regression with method="re".
   Assumes entity effects are uncorrelated with regressors. More efficient than FE
   if the assumption holds, but biased if it doesn't.
   - Compare RE coefficients to FE. If very different, RE is suspect.
   - RE identifies coefficients on time-invariant variables (unlike FE).

---

## Phase 3: Model Selection & Diagnostics

6. **Hausman test** (FE vs RE): Compare the coefficient vectors.
   The test statistic is: H = (b_FE - b_RE)' * (V_FE - V_RE)^-1 * (b_FE - b_RE).
   - p < 0.05: Reject RE. Use FE (entity effects are correlated with X).
   - p > 0.05: Cannot reject RE. RE is preferred (more efficient).
   - In practice, for firm-level financial data, FE almost always wins.

7. **Diagnostic checks** for the preferred model:
   - Serial correlation: Do residuals within entities show autocorrelation?
     If yes, use Newey-West or clustered standard errors.
   - Heteroscedasticity: Do residuals vary across entities?
     If yes, use heteroscedasticity-robust (White) standard errors.
   - Cross-sectional dependence: Are residuals correlated across entities at the same time?
     If yes (common in financial panels), use Driscoll-Kraay standard errors.
   - **Rule of thumb**: Always cluster standard errors by entity for financial panels.

8. **Two-way fixed effects** (if warranted): Add time fixed effects alongside entity FE.
   This absorbs common time shocks (market-wide events, macro trends).
   Compare with one-way FE -- do time FE improve the model?

---

## Phase 4: Robustness & Interpretation

9. **Sensitivity analysis**:
   - Drop one entity at a time. Do coefficients change materially? (Influential entities.)
   - Vary the sample period. Are results stable over time?
   - Add/remove controls. Is the key coefficient robust?

10. **Economic interpretation**:
    - Translate coefficient into economic magnitude.
      Example: "A 1% increase in X is associated with a 0.5% increase in Y,
      holding entity-level factors constant."
    - Standard errors should be clustered (not naive OLS SE).
    - Report t-statistics and confidence intervals with clustered SE.

11. **Summary table**:

    | Model | Key Coeff | Std Err | R-sq | N |
    |-------|-----------|---------|------|---|
    | Pooled OLS | ... | ... | ... | ... |
    | Fixed Effects | ... | (clustered) | within: ... | ... |
    | Random Effects | ... | ... | ... | ... |

    Preferred model, economic significance, policy implications.

**Related prompts**: Use var_model_analysis for dynamic panel relationships,
cointegration_analysis for non-stationary panels.
""",
                },
            }
        ]

    @mcp.prompt()
    def var_model_analysis(dataset: str = "macro_data") -> list[dict]:
        """Vector autoregression: lag selection, IRFs, variance decomposition, forecasting."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Perform a Vector Autoregression (VAR) analysis on {dataset}. This uses the
econometrics/ tools (var_model, impulse_response) alongside stats/ tools for
pre-estimation diagnostics.

---

## Phase 1: Data Preparation

1. **Stationarity check**: Run analyze() on each variable in {dataset}.
   Check the ADF test p-value for each series.
   - p < 0.05: Stationary. Can use in levels.
   - p > 0.05: Non-stationary. Need to difference or check for cointegration.
   If ALL series are I(1), run cointegration_johansen first. If cointegrated,
   use a VECM instead of a VAR in differences (to preserve the long-run relationship).

2. **Transform if needed**: If differencing, use compute_returns (log returns = first
   difference of log prices). Re-check stationarity on the transformed data.

3. **Variable selection**: Decide which variables to include. Too many variables
   relative to observations causes over-parameterization. Rule of thumb:
   K * (K * p + 1) parameters, where K = # variables, p = # lags.
   Keep K * p * K < T / 3 for reliable estimation.

---

## Phase 2: Model Estimation

4. **Lag order selection**: Run var_model with lags set to a reasonable maximum
   (e.g., 4 for quarterly, 12 for monthly, 2-4 for daily financial data).
   Compare AIC and BIC:
   - AIC: Tends to overfit (choose more lags). Better for forecasting.
   - BIC: More parsimonious. Better for structural analysis.
   - If AIC and BIC disagree, prefer BIC for interpretation, AIC for forecasting.

5. **Estimate the VAR**: Run var_model with the selected lag order.
   Check:
   - **Stability**: All eigenvalues of the companion matrix must be inside the unit circle.
     If not, the VAR is explosive and unusable.
   - **Residual diagnostics**: Are residuals white noise? Autocorrelation in residuals
     suggests too few lags. Run analyze() on the residuals.

---

## Phase 3: Dynamic Analysis

6. **Impulse response functions**: Run impulse_response on the fitted VAR model.
   For each variable as the shock source:
   - How does a 1-standard-deviation shock propagate to other variables?
   - How quickly do responses decay? (Half-life of the response)
   - Are there delayed responses (significant at lag > 1 but not lag 0)?

   **Interpretation for financial data**:
   - Shock to equity market -> bond market: negative response = flight to quality.
   - Shock to VIX -> equity: negative, persistent = vol regime matters.
   - Shock to rates -> equity: negative, delayed = monetary policy transmission.

7. **Granger causality**: For each pair (X, Y), test if past values of X help
   predict Y beyond Y's own past. This reveals the direction of information flow.
   - X Granger-causes Y AND Y Granger-causes X = feedback system.
   - One-directional = leader-follower relationship.
   Note: Granger causality is about prediction, not true causation.

8. **Variance decomposition**: What fraction of each variable's forecast error
   variance is explained by shocks to each other variable?
   - At short horizons: mostly own shocks dominate.
   - At longer horizons: cross-variable influence grows.
   - The variable that explains the most variance in others is the "dominant" variable.

---

## Phase 4: Forecasting & Validation

9. **Out-of-sample forecast**: Use the VAR to forecast all variables jointly.
   Report the forecast for the next `horizon` periods.
   - Compare to univariate AR(p) forecasts -- does the VAR add value?
   - If VAR forecast RMSE < AR forecast RMSE, cross-variable dynamics matter.

10. **Rolling estimation**: Re-estimate the VAR on a rolling window.
    - Are coefficients stable? If not, there may be structural breaks.
    - Run structural_break on each variable to check.
    - Unstable VAR = regime-switching model may be needed.

---

## Phase 5: Synthesis

11. **Summary**:
    - Lag order: Selected by AIC/BIC. Model stable?
    - Key dynamic relationships: Which variables lead/lag?
    - Shock transmission: How do shocks propagate through the system?
    - Forecast accuracy: Does the VAR beat univariate models?
    - Dominant variable: Which variable drives the most variance?
    - **Trading implications**: If variable A leads B by 2 periods, A is a
      predictive signal for B.

**Related prompts**: Use cointegration_analysis for non-stationary systems,
structural_break_analysis if VAR coefficients are unstable.
""",
                },
            }
        ]

    @mcp.prompt()
    def structural_break_analysis(dataset: str = "returns") -> list[dict]:
        """Detect and analyze structural breaks with sub-period comparison."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Detect and analyze structural breaks in {dataset}. This workflow uses
econometrics/ break tests, stats/ diagnostics, and vol/ modeling to
understand how the data-generating process changes over time.

---

## Phase 1: Break Detection

1. **Formal break test**: Run structural_break on {dataset} with method="sup_f".
   The Andrews supremum-F test scans for an unknown break point by testing every
   possible break location and reporting the maximum F-statistic.
   - p < 0.05: At least one structural break exists.
   - The estimated break point date is the most likely location.

2. **Known break test**: If you suspect a specific date (e.g., COVID March 2020,
   Fed rate hike), run structural_break with method="chow" and the suspected
   break_point index. This is more powerful than the sup-F test when you have a prior.

3. **Multiple breaks**: Run the sup-F test on each sub-period to find additional
   breaks. Financial data often has 2-4 break points over a decade.
   Also use bayesian_changepoint for a complementary approach -- it estimates
   the full posterior distribution over changepoint locations.

4. **CUSUM cross-check**: Run optimal_stopping with method="cusum" on the same data.
   CUSUM detects gradual shifts (not just level shifts). If CUSUM triggers at a
   different date than Chow/Andrews, the break may be gradual rather than sudden.

---

## Phase 2: Sub-Period Characterization

5. **Split at break dates**: For each sub-period between consecutive breaks:
   - Run analyze() to compute mean, std, skewness, kurtosis, ADF test.
   - Run risk_metrics to get Sharpe, Sortino, max drawdown.
   - Run distribution_fit to check if the return distribution changes shape.

   **What to look for**:
   - Mean shift: Has the average return changed? (Regime change in drift)
   - Variance shift: Has volatility changed? (Regime change in risk)
   - Distribution shift: Has skewness/kurtosis changed? (Tail risk change)
   - Stationarity: Is each sub-period still stationary? If not, further breaks exist.

6. **Volatility dynamics shift**: Run fit_garch on each sub-period separately.
   - Does GARCH persistence (alpha + beta) change? Higher = more persistent vol.
   - Does the leverage effect (gamma) change? It often increases during crises.
   - Does the unconditional vol level change? This is the main volatility regime shift.

7. **Regime detection comparison**: Run detect_regimes on the full sample.
   Do HMM regime transition dates align with structural break dates?
   - Aligned: The break is a genuine regime change.
   - Misaligned: The break may be a one-time level shift, not a regime.

---

## Phase 3: Cause Identification

8. **Event mapping**: For each detected break date, identify the likely cause:
   - Market events: Flash crash, COVID selloff, Lehman collapse
   - Policy changes: Fed rate decisions, QE announcements, regulations
   - Market structure: Decimalization, circuit breakers, new ETF launches
   - Fundamental: Earnings revision, sector rotation, credit event

9. **Spillover analysis**: If multi-asset data available, test whether breaks
   are synchronized across assets (systematic) or idiosyncratic.
   Run structural_break on each asset -- do break dates cluster?

---

## Phase 4: Implications

10. **Model stability assessment**:
    - Should you estimate models on the full sample or sub-periods?
      If coefficients differ materially across sub-periods, use sub-periods.
    - Rolling-window estimation: How large a window before the oldest break?
    - Adaptive models: fit_garch with shorter windows adapts to breaks automatically.

11. **Summary**:

    | Break Date | Before Mean | After Mean | Before Vol | After Vol | Likely Cause |
    |-----------|-------------|------------|-----------|----------|--------------|
    | ... | ... | ... | ... | ... | ... |

    Number of breaks, severity of each, implications for model choice.
    Recommendation: full-sample vs sub-period estimation.

**Related prompts**: Use detect_regimes for ongoing regime monitoring,
bayesian_changepoint for probabilistic break detection.
""",
                },
            }
        ]

    @mcp.prompt()
    def cointegration_analysis(dataset: str = "prices") -> list[dict]:
        """Full cointegration workflow: Engle-Granger, Johansen, VECM, pairs trading."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Perform a comprehensive cointegration analysis on {dataset}. This workflow uses
econometrics/ tools (cointegration_johansen, var_model) alongside stats/ tools
to test for, estimate, and trade cointegrating relationships.

---

## Phase 1: Integration Order

1. **Unit root tests**: Run analyze() on each price series in {dataset}.
   Check the ADF test for each:
   - Prices: Should be non-stationary (ADF p > 0.05) -- this is I(1).
   - Returns: Should be stationary (ADF p < 0.05) -- this is I(0).
   - If prices are already stationary (I(0)), cointegration is not applicable.
   - If returns are non-stationary, the series may be I(2) -- rare for financial data.

2. **Visual inspection**: Use query_data to look at price paths. Cointegrated series
   wander together -- they may diverge temporarily but always revert. Non-cointegrated
   series can diverge permanently.

---

## Phase 2: Cointegration Testing

3. **Engle-Granger test** (for 2 variables): This is the simpler two-step approach.
   - Step 1: Regress Y on X (OLS). The coefficient is the hedge ratio.
   - Step 2: Test the regression residuals for stationarity (ADF on residuals).
   - If residuals are stationary (p < 0.05), the pair is cointegrated.
   Use correlation_analysis and then analyze() on the residuals.

   **Limitations**: Only works for 2 variables. Can only find one cointegrating
   relationship. Results depend on which variable is Y vs X.

4. **Johansen test** (multivariate): Run cointegration_johansen on {dataset}.
   This tests for the number of cointegrating relationships (rank).
   - **Trace test**: H0: rank = 0 vs H1: rank >= 1. If trace_stat > critical value
     at 5%, reject H0 (at least one cointegrating relationship).
   - **Max eigenvalue test**: Tests rank = r vs rank = r+1. More specific.
   - **Cointegration rank**: Number of independent cointegrating relationships.
     For K variables, max rank = K-1. Rank = 0 means no cointegration.
   - **Cointegrating vectors** (eigenvectors): These define the long-run
     equilibrium relationships. The first eigenvector (largest eigenvalue)
     is the strongest relationship.

5. **Deterministic terms** (det_order):
   - det_order=-1: No constant or trend. Strictest.
   - det_order=0: Constant in cointegrating equation. Most common for financial data.
   - det_order=1: Linear trend. Use if prices have deterministic trends.

---

## Phase 3: Spread Construction & Properties

6. **Hedge ratio**: From the cointegrating vector, construct the spread:
   spread_t = Y_t - beta * X_t (for 2 assets).
   For multi-asset: spread_t = w1*P1 + w2*P2 + ... + wK*PK where w = eigenvector.

7. **Spread stationarity**: Run analyze() on the spread. ADF should confirm
   stationarity. If not, the cointegration may be spurious or the hedge ratio is wrong.

8. **Half-life of mean reversion**: From the spread, estimate the OU process:
   d(spread) = -theta * spread * dt. Half-life = ln(2) / theta.
   - Half-life < 5 days: Very fast. High-frequency pairs trading.
   - Half-life 5-30 days: Sweet spot. Daily rebalancing pairs trading.
   - Half-life > 60 days: Too slow. Capital is tied up too long.
   Run analyze() on spread changes for the autocorrelation coefficient.
   theta ~ -autocorrelation at lag 1. Half-life = ln(2) / abs(theta).

9. **Spread distribution**: Run distribution_fit on the spread.
   Is it normally distributed? Fat tails = risk of larger divergences.
   Skewed = asymmetric risk. This affects stop-loss placement.

---

## Phase 4: VECM & Dynamics

10. **Vector Error Correction Model**: Run var_model on the levels data.
    The VECM adds the error correction term (lagged spread) to a VAR in differences.
    - **Speed of adjustment** coefficients: How fast does each asset respond to
      disequilibrium? Larger = faster reversion. If only one asset adjusts,
      the other is the "leader" (price discovery).
    - **Short-run dynamics**: VAR coefficients show short-term lead-lag relationships.
    - Run impulse_response on the VECM to see how shocks propagate.

---

## Phase 5: Trading Application

11. **Pairs trading signals**: Standardize the spread (z-score):
    z_t = (spread_t - mean(spread)) / std(spread).
    - Entry long spread: z < -2.0 (spread too low, expect reversion up).
    - Entry short spread: z > +2.0 (spread too high, expect reversion down).
    - Exit: z returns to 0 (or crosses zero).
    - Stop-loss: z < -4.0 or z > +4.0 (relationship may have broken down).

12. **Rolling hedge ratio**: Re-estimate the cointegrating vector on a rolling
    window (e.g., 250 days). If the hedge ratio drifts materially, the
    relationship may not be stable. Consider adaptive rebalancing.

13. **Summary**:
    - Cointegration rank and evidence strength.
    - Hedge ratio and its stability over time.
    - Half-life of mean reversion.
    - Speed of adjustment (which asset leads?).
    - Trading strategy feasibility: half-life, transaction costs, capacity.

**Related prompts**: Use var_model_analysis for dynamic analysis,
structural_break_analysis if cointegration appears to break down.
""",
                },
            }
        ]

    @mcp.prompt()
    def event_study_analysis(
        ticker: str = "AAPL",
        event_date: str = "2024-01-25",
    ) -> list[dict]:
        """Event study with abnormal returns, statistical tests, and information leakage."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Perform a rigorous event study for {ticker} around {event_date}. This uses the
econometrics/ event_study_econometric tool alongside stats/ and vol/ tools for a
complete assessment of the event's impact.

---

## Phase 1: Setup

1. **Data requirements**: Load price data for {ticker} covering at least:
   - Estimation window: -260 to -11 trading days before event (250 days for
     market model estimation). Run compute_returns on prices_{ticker.lower()}.
   - Event window: -10 to +20 trading days around {event_date}.
   - Market benchmark (SPY or relevant index) for the same period.

2. **Market model estimation**: Over the estimation window, regress
   {ticker} returns on market returns:
   R_i,t = alpha + beta * R_m,t + epsilon_t.
   This establishes "normal" returns. The beta captures market sensitivity.

---

## Phase 2: Abnormal Returns

3. **Compute abnormal returns**: Run event_study_econometric with:
   - dataset = returns_{ticker.lower()}
   - event_dates_json = '["{event_date}"]'
   - market_dataset = returns_spy (or your benchmark)
   - estimation_window = 250
   - event_window = 10

   This computes:
   - **AR_t** = R_i,t - (alpha + beta * R_m,t) for each day in the event window.
   - **CAR** = cumulative sum of ARs.
   - **t-statistic** for testing H0: CAR = 0.

4. **Key windows to report**:
   - AR(0): Immediate event-day reaction.
   - CAR(-1, +1): 3-day reaction (captures event + immediate response).
   - CAR(-5, +5): Full event window.
   - CAR(0, +10): Post-event drift.
   - CAR(0, +20): Extended drift.

---

## Phase 3: Pre-Event Analysis (Information Leakage)

5. **Pre-event abnormal returns**: Examine AR from day -10 to day -1.
   - CAR(-10, -1) significantly positive? = insider buying / information leakage.
   - CAR(-10, -1) significantly negative? = insider selling / anticipation.
   - Gradual pre-event drift suggests the information was partially priced in.

6. **Pre-event volume**: Run analyze() on the volume in days -10 to -1.
   Abnormal volume before the event suggests informed trading.
   Compare to the estimation window average volume.

7. **Pre-event volatility**: Run fit_garch on the pre-event period.
   Elevated conditional vol before the event = market uncertainty / anticipation.

---

## Phase 4: Post-Event Dynamics

8. **Post-event drift**: Does CAR continue after the initial reaction?
   - Continued drift in the same direction = underreaction (market is slow
     to fully incorporate the information). This is the "PEAD" for earnings.
   - Reversal = overreaction (initial response was excessive).
   Run analyze() on post-event returns to check for autocorrelation.

9. **Volatility impact**: Run fit_garch on the full sample spanning the event.
   - Does conditional vol spike on the event day? How long until it normalizes?
   - Compare GARCH-implied vol before and after the event.
   - Run news_impact_curve to see if the event changed the asymmetric vol response.

10. **Regime impact**: Run detect_regimes on returns spanning the event.
    Did the event trigger a regime change? If so, from which regime to which?

---

## Phase 5: Statistical Robustness

11. **Multiple testing** (if multiple events): If testing multiple events,
    correct for multiple comparisons (Bonferroni or BH correction).
    The significance threshold should be stricter.

12. **Non-parametric tests**: The standard t-test assumes normal ARs.
    If returns are fat-tailed, use the rank test or sign test as robustness checks.

13. **Summary**:

    | Window | CAR | t-stat | p-value | Interpretation |
    |--------|-----|--------|---------|----------------|
    | AR(0) | ... | ... | ... | Immediate reaction |
    | CAR(-1,+1) | ... | ... | ... | Short-term impact |
    | CAR(-10,-1) | ... | ... | ... | Pre-event leakage? |
    | CAR(0,+10) | ... | ... | ... | Post-event drift? |

    - Event impact: positive/negative, magnitude in bps.
    - Information leakage: evidence of pre-event trading?
    - Post-event drift: under/overreaction?
    - Volatility impact: spike magnitude and persistence.

**Related prompts**: Use structural_break_analysis if the event caused a regime change,
var_model_analysis for understanding cross-asset event transmission.
""",
                },
            }
        ]

    @mcp.prompt()
    def instrumental_variable_analysis(dataset: str = "panel_data") -> list[dict]:
        """Instrumental variable (2SLS) for causal identification."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

IV/2SLS analysis on {dataset}:

1. **The endogeneity problem**: Why is OLS biased? Omitted variables?
   Simultaneity? Measurement error?
2. **Instrument identification**: What variable Z satisfies:
   - Relevance: Z correlates with the endogenous regressor X.
   - Exclusion: Z affects Y ONLY through X, not directly.
3. **First stage**: instrumental_variable -- regress X on Z (and controls).
   F-stat > 10 (rule of thumb). Weak instruments bias toward OLS.
4. **Second stage**: Use predicted X from first stage to estimate effect on Y.
   2SLS coefficient is the causal effect (under IV assumptions).
5. **Diagnostics**:
   - Weak instrument test: Stock-Yogo critical values.
   - Over-identification test (Sargan): if multiple instruments, are they all valid?
   - Hausman test: compare IV to OLS. If different, OLS is biased.
6. **Summary**: IV estimate vs OLS estimate. Is endogeneity significant?
   Instrument strength. Causal interpretation and limitations.
""",
                },
            }
        ]
