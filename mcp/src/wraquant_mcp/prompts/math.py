"""Advanced math prompt templates."""

from __future__ import annotations

from typing import Any


def register_math_prompts(mcp: Any) -> None:

    @mcp.prompt()
    def network_analysis(dataset: str = "returns_universe") -> list[dict]:
        """Financial network analysis: correlation networks, centrality, systemic risk."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Perform a financial network analysis on {dataset}. This uses math/ network tools
(correlation_network, systemic_risk) to map the structure of cross-asset dependencies
and identify systemically important nodes.

---

## Phase 1: Network Construction

1. **Correlation network**: Run correlation_network on {dataset} with threshold=0.5.
   This builds a graph where:
   - Nodes = assets (one per column in the dataset).
   - Edges = correlations exceeding the threshold.
   - Edge weight = correlation magnitude.

   **Threshold selection**:
   - threshold=0.3: Dense network, many edges. Good for community detection.
   - threshold=0.5: Moderate. Standard for financial networks.
   - threshold=0.7: Sparse. Only the strongest relationships survive.
   Start with 0.5 and adjust based on network density.

2. **Network statistics**: From the correlation_network output:
   - **Number of edges**: Too many = threshold too low. Too few = too high.
   - **Density**: edges / max_possible_edges. > 0.5 = very connected market.
   - **Isolated nodes**: Assets with no edges above threshold. These are unique diversifiers.

3. **Centrality analysis**: The correlation_network result includes centrality measures.
   Report for each asset:
   - **Degree centrality**: Number of connections. High degree = hub asset.
     These are "market bellwethers" -- they move with many others.
   - Most central asset: This is the most "systemic" node.
   - Least central asset: Best diversifier (least connected).

---

## Phase 2: Systemic Risk Assessment

4. **Systemic risk scores**: Run systemic_risk on {dataset}.
   This computes Marginal Expected Shortfall (MES) for each asset:
   - MES = E[R_i | R_market < VaR_5%]. The average return of asset i
     on the worst 5% market days.
   - More negative MES = higher systemic risk contribution.
   - Most systemic asset: Contributes most to portfolio tail risk.
   - Least systemic asset: Best tail risk diversifier.

5. **Centrality vs systemic risk**: Compare the two rankings.
   - High centrality + high MES: Truly systemic (e.g., large banks in 2008).
   - High centrality + low MES: Connected but not dangerous (e.g., utilities).
   - Low centrality + high MES: Tail risk from idiosyncratic exposure (e.g., EM).
   - Low centrality + low MES: Safe diversifier. Increase allocation.

---

## Phase 3: Network Dynamics

6. **Regime-conditional networks**: If possible, split {dataset} into bull and
   bear periods (using detect_regimes). Build separate correlation_networks
   for each regime.
   - Do edges increase in bear markets? (Contagion = correlations spike in crisis.)
   - Which assets become more central in crisis? These are contagion channels.
   - Which assets become less central? These are safe havens.

7. **Rolling network analysis**: Compute correlation_network on rolling windows
   (e.g., 120-day) to track network evolution.
   - Is the network becoming denser? (Rising systemic risk.)
   - Is a new hub emerging? (Sector rotation or contagion risk.)
   - Are previously connected assets decorrelating? (Diversification improving.)

---

## Phase 4: Portfolio Implications

8. **Network-informed allocation**:
   - Reduce weight on highly central assets (diversification penalty).
   - Increase weight on peripheral assets (better diversification).
   - Avoid concentrating in a single cluster (community).

9. **Contagion risk**: If the network has a highly connected core:
   - A shock to any core asset propagates to all others.
   - Portfolio VaR underestimates risk if it ignores network structure.
   - Use stress_test with scenarios targeting core assets.

10. **Summary**:
    - Network density and key hubs.
    - Systemic risk ranking (most to least).
    - Community structure: which assets cluster together?
    - Regime sensitivity: does the network change in crisis?
    - Portfolio recommendations: reweight based on network position.

**Related prompts**: Use correlation_analysis for pairwise details,
risk_report for portfolio risk assessment.
""",
                },
            }
        ]

    @mcp.prompt()
    def levy_process_modeling(
        dataset: str = "returns",
        model: str = "variance_gamma",
    ) -> list[dict]:
        """Levy process calibration, simulation, and option pricing."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Calibrate and simulate a {model} Levy process using {dataset}. This uses
math/ Levy tools for fat-tailed return modeling, simulation, and derivative pricing.

---

## Phase 1: Empirical Distribution Analysis

1. **Return statistics**: Run analyze() on {dataset}.
   Report: mean, std, skewness, excess kurtosis.
   - Kurtosis > 3: Fat tails. Standard normal is inadequate.
   - Negative skew: Left tail is heavier. Crashes are larger than rallies.
   - These moments are the calibration targets for the Levy model.

2. **Distribution fit**: Run distribution_fit on {dataset}.
   Compare normal, Student-t, and skewed-t fits.
   The best-fit distribution guides Levy model selection:
   - High kurtosis, symmetric: Variance Gamma or NIG with beta=0.
   - High kurtosis, negative skew: Variance Gamma with theta < 0 or NIG with beta < 0.
   - Very heavy tails (kurtosis > 10): Consider stable Levy (alpha < 2).

---

## Phase 2: Levy Process Simulation

3. **Simulate the process**: Run levy_simulate with model="{model}".
   Parameter guidance:
   - **Variance Gamma**: sigma=historical vol, nu=2/(kurtosis-3) if kurtosis>3,
     theta=mean * sqrt(252) (negative for equities).
   - **NIG**: alpha=tail_heaviness (higher = lighter tails), beta=skewness parameter,
     mu=location, delta=scale.
   - **CGMY**: C=activity, G=right tail decay, M=left tail decay, Y=fine structure.
   - **Stable**: alpha=stability (1.5-1.9 for financial data), beta=skewness.

4. **Simulation analysis**: From the levy_simulate output:
   - Final value: Where did the simulated path end?
   - Max/min value: Range of the path (extreme moves).
   - Increment statistics: Compare mean, std, skew, kurtosis of simulated
     increments to the empirical moments from step 1.
   - If simulated moments match empirical: calibration is good.
   - If mismatch: adjust parameters and re-simulate.

---

## Phase 3: Model Comparison

5. **Multi-model comparison**: Run levy_simulate for each model type
   (variance_gamma, nig, stable). Compare:
   - Which model best matches the empirical distribution?
   - Increment kurtosis: Does the model capture the observed fat tails?
   - Skewness: Does the model capture the observed asymmetry?

6. **Goodness of fit**: Compare simulated vs empirical:
   - QQ-plot comparison: Do the tails match?
   - Kolmogorov-Smirnov test: Is the simulated distribution consistent?
   - AIC/BIC if available from distribution_fit.

---

## Phase 4: Applications

7. **Risk assessment**: Use the fitted Levy model for:
   - VaR estimation: Levy VaR captures tail risk better than Gaussian.
   - Expected Shortfall: Levy ES is more conservative (wider tails).
   - Compare Levy-based VaR to the var_analysis output from risk/ tools.

8. **Option pricing**: If pricing derivatives:
   - Levy models generate the volatility smile naturally (no need for
     local vol or stochastic vol hacks).
   - Use the Levy characteristic function for FFT-based option pricing.
   - Compare to Black-Scholes prices: the difference = smile effect.

9. **Scenario generation**: Use levy_simulate to generate Monte Carlo scenarios
   that respect the fat-tailed, skewed distribution.
   - Standard normal MC underestimates tail risk by 30-50%.
   - Levy MC produces realistic extreme scenarios.

---

## Phase 5: Summary

10. **Model selection**:
    - Best Levy model for this data: {model} or alternative.
    - Calibrated parameters and their interpretation.
    - Key advantage over Gaussian: tail risk capture.
    - Application recommendation: risk, pricing, or scenario generation.

**Related prompts**: Use distribution_fit for empirical distribution analysis,
exotic_pricing for Levy-based option pricing, spectral_analysis for frequency
domain characterization.
""",
                },
            }
        ]

    @mcp.prompt()
    def information_theory_analysis(dataset: str = "returns_universe") -> list[dict]:
        """Entropy, mutual information, and information-theoretic dependence analysis."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Perform an information-theoretic analysis of {dataset}. This uses math/ spectral
and network tools alongside stats/ analysis to measure dependence and complexity
using entropy-based measures.

---

## Phase 1: Entropy Analysis

1. **Spectral entropy**: Run spectral_analysis on each column of {dataset}.
   Report spectral_entropy for each asset.
   - **Interpretation**: Spectral entropy measures the "flatness" of the power spectrum.
     High entropy (near 1): White noise, unpredictable, efficient market.
     Low entropy (near 0): Concentrated frequency content, periodic/predictable.
   - Rank assets by spectral entropy. The lowest-entropy assets have the most
     structure (exploitable patterns).

2. **Dominant frequencies**: From spectral_analysis, report dominant frequencies
   and their periods for each asset.
   - Period of 5 days: Weekly cycle (day-of-week effect).
   - Period of 21 days: Monthly cycle (options expiration, rebalancing).
   - Period of 63 days: Quarterly cycle (earnings, quarter-end effects).
   - Period of 252 days: Annual cycle (seasonality).
   Are these cycles statistically significant or just noise?

---

## Phase 2: Dependence Structure

3. **Linear correlation**: Run correlation_analysis on {dataset} as baseline.
   Pearson correlation measures linear dependence. Report the correlation matrix.

4. **Non-linear dependence**: Mutual information captures ALL dependence (linear
   and non-linear). Compare mutual information to squared correlation:
   - If MI >> r-squared: Significant non-linear dependence exists.
     Linear models miss this. Consider non-linear models or copulas.
   - If MI ~ r-squared: Dependence is mostly linear. Standard correlation is sufficient.

5. **Transfer entropy**: Directional information flow between assets.
   Transfer entropy from X to Y measures how much knowing X's past reduces
   uncertainty about Y's future, beyond what Y's own past tells you.
   - TE(X->Y) > TE(Y->X): X leads Y (X is the information source).
   - TE(X->Y) ~ TE(Y->X): Bidirectional or no clear leader.
   This is the information-theoretic analog of Granger causality but captures
   non-linear predictive relationships.

---

## Phase 3: Complexity & Efficiency

6. **Market efficiency**: Low spectral entropy + high autocorrelation = predictable.
   Combine entropy with Hurst exponent:
   - H > 0.5, low entropy: Trending (momentum strategies).
   - H < 0.5, low entropy: Mean-reverting (reversion strategies).
   - H ~ 0.5, high entropy: Efficient (no exploitable patterns).

7. **Information ratio**: Compare the entropy of the asset to the entropy of
   a pure random walk with the same variance. The ratio measures departure
   from efficiency:
   - Ratio near 1: Market is efficient for this asset.
   - Ratio << 1: Highly structured, potentially predictable.

8. **Regime-conditional entropy**: Run detect_regimes first, then compute
   spectral_analysis separately for each regime.
   - Does entropy decrease in bear markets? (More structure = more predictability in crisis.)
   - Does entropy increase in bull markets? (Less structure = more efficient.)

---

## Phase 4: Network Information

9. **Correlation network**: Run correlation_network on {dataset}.
   Use mutual information instead of linear correlation for edge weights
   (captures non-linear relationships between assets).

10. **Information centrality**: Which assets are the most "informative"
    about the rest of the network? High mutual information with many others
    = information hub. These assets are leading indicators.

---

## Phase 5: Synthesis

11. **Information-theoretic dashboard**:

    | Asset | Spectral Entropy | Dominant Period | MI with Market | Efficiency Ratio |
    |-------|-----------------|----------------|----------------|-----------------|
    | ... | ... | ... | ... | ... |

12. **Implications**:
    - Most predictable assets (lowest entropy): Candidates for alpha generation.
    - Information leaders (highest transfer entropy): Leading indicators.
    - Non-linear dependencies: Where standard correlation fails.
    - Market efficiency assessment: Is this market exploitable?

**Related prompts**: Use network_analysis for graph-based dependence,
var_model_analysis for linear lead-lag relationships.
""",
                },
            }
        ]

    @mcp.prompt()
    def optimal_stopping_analysis(
        dataset: str = "prices",
        column: str = "close",
    ) -> list[dict]:
        """Optimal stopping analysis: exit timing, CUSUM detection, OU thresholds."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Apply optimal stopping theory to {dataset} (column: {column}). This uses
math/ tools (optimal_stopping, hawkes_fit) to determine optimal trade exit
timing and detect regime changes in real-time.

---

## Phase 1: CUSUM Changepoint Detection

1. **CUSUM stopping**: Run optimal_stopping on {dataset} with method="cusum".
   This monitors the cumulative sum of deviations from the target mean.
   When the CUSUM statistic exceeds the threshold, a structural change is detected.

   **Parameter guidance**:
   - threshold=2.0: Standard. Detects moderate shifts.
   - threshold=1.5: More sensitive. More false positives but faster detection.
   - threshold=3.0: Less sensitive. Fewer false alarms but slower detection.

   **Interpretation**:
   - stopped=True: A change was detected. The stop time and CUSUM value tell
     you when and how severe the shift was.
   - stopped=False: No change detected within the sample. The process is stable.
   - Direction: Positive CUSUM = upward shift. Negative = downward shift.

2. **Multi-threshold analysis**: Run optimal_stopping at thresholds 1.5, 2.0, 3.0.
   - Do all three detect a change? = Strong evidence of shift.
   - Only 1.5 detects? = Marginal shift, possibly noise.
   - All three agree on timing? = Sharp break. Disagree? = Gradual drift.

---

## Phase 2: Ornstein-Uhlenbeck Exit Analysis

3. **OU optimal exit**: Run optimal_stopping with method="ou_exit".
   This estimates the mean-reversion parameters and computes the optimal
   exit threshold for a mean-reverting position.

   **Parameters**:
   - The tool estimates mu (mean-reversion speed) and sigma from the data.
   - threshold parameter is interpreted as transaction cost for OU exit.

   **Interpretation**:
   - Optimal exit level: How far from the mean should you exit?
     Higher mu (faster reversion) = tighter exit threshold.
     Higher sigma (more noise) = wider exit threshold.
     Higher transaction cost = wider threshold (need more edge to cover costs).

4. **Mean-reversion characterization**: From the OU parameters:
   - **mu > 0**: Series is mean-reverting. Good candidate for pairs trading.
   - **Half-life**: ln(2) / mu. How long for a 50% reversion.
     < 5 days = very fast reversion. > 30 days = slow, may not be tradeable.
   - **sigma / sqrt(2*mu)**: Stationary standard deviation. This determines
     the typical range of fluctuations around the mean.

---

## Phase 3: Hawkes Self-Excitation Analysis

5. **Event clustering**: If the data represents event times (trades, jumps):
   Run hawkes_fit on the event times dataset.
   - **mu** (baseline intensity): Average event rate without self-excitation.
   - **alpha** (excitation): How much each event boosts the rate of future events.
   - **beta** (decay): How quickly the excitation fades.
   - **Branching ratio** (alpha/beta): Must be < 1 for stationarity.
     Near 1 = highly self-exciting (events cluster intensely).
     Near 0 = events are nearly independent (Poisson-like).

6. **Hawkes implications for trading**:
   - High branching ratio for trade arrivals: Volatility clustering,
     liquidity shocks are self-reinforcing. Be cautious with market orders.
   - Jump Hawkes: Large price moves trigger more large price moves.
     After a jump, expect more jumps (increase position monitoring frequency).

---

## Phase 4: American Option Stopping

7. **Longstaff-Schwartz framework**: For pricing American options:
   - The optimal exercise boundary is an optimal stopping problem.
   - Early exercise is optimal when intrinsic value > continuation value.
   - Use the OU exit analysis to estimate when a mean-reverting spread
     hits the optimal exercise threshold.

8. **Real options**: Optimal stopping applies to:
   - When to invest in a project (threshold for NPV).
   - When to exit a position (threshold for loss/profit).
   - When to abandon a strategy (threshold for drawdown).

---

## Phase 5: Synthesis

9. **Stopping decision framework**:

    | Method | Signal | Threshold | Current Status |
    |--------|--------|-----------|---------------|
    | CUSUM | Changepoint detected? | ... | Stopped/Continuing |
    | OU Exit | Optimal exit level | Transaction cost | Above/Below threshold |
    | Hawkes | Self-excitation level | Branching ratio | Clustered/Independent |

10. **Recommendations**:
    - Is a regime change detected? (CUSUM)
    - What is the optimal exit level for mean-reverting positions? (OU)
    - Is event clustering elevated? (Hawkes)
    - Combined signal: Stay in position, tighten stops, or exit?

**Related prompts**: Use structural_break_analysis for formal break tests,
detect_regimes for probabilistic regime identification.
""",
                },
            }
        ]
