"""Advanced quantitative methods with wraquant.

Demonstrates Levy processes, network analysis, optimal stopping,
causal inference, Bayesian analysis, and regime detection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# --- Levy processes ---
from wraquant.math.levy import (
    variance_gamma_simulate,
    nig_simulate,
    fit_variance_gamma,
)

print("=== Levy Processes ===")
vg_path = variance_gamma_simulate(sigma=0.2, theta=-0.1, nu=0.5, n_steps=500, seed=42)
print(f"  Variance Gamma path: {len(vg_path)} steps, final={vg_path[-1]:.4f}")

nig_path = nig_simulate(alpha=15.0, beta=-3.0, mu=0.0, delta=0.5, n_steps=500, seed=42)
print(f"  NIG path: {len(nig_path)} steps, final={nig_path[-1]:.4f}")

# Fit VG to data
returns = rng.normal(0, 0.02, 1000) + rng.exponential(0.005, 1000) * rng.choice([-1, 1], 1000)
vg_fit = fit_variance_gamma(returns)
print(f"  VG fit: sigma={vg_fit['sigma']:.4f}, theta={vg_fit['theta']:.4f}, nu={vg_fit['nu']:.4f}")

# --- Network analysis ---
from wraquant.math.network import (
    correlation_network,
    minimum_spanning_tree,
    centrality_measures,
    community_detection,
    systemic_risk_score,
)

print(f"\n=== Financial Network Analysis ===")
n_assets = 8
asset_returns = pd.DataFrame(
    rng.multivariate_normal(np.zeros(n_assets), np.eye(n_assets) * 0.01 + 0.005, 200),
    columns=[f"Asset_{i}" for i in range(n_assets)],
)

adj = correlation_network(asset_returns, threshold=0.3)
print(f"  Adjacency matrix shape: {adj.shape}")
print(f"  Edges (correlation > 0.3): {int(adj.sum() / 2)}")

mst = minimum_spanning_tree(asset_returns)
print(f"  MST edges: {int(mst.sum() / 2)}")

centrality = centrality_measures(adj)
print(f"  Degree centrality: {np.round(centrality['degree'], 3)}")

labels = community_detection(adj, n_communities=2)
print(f"  Community labels: {labels}")

risk_scores = systemic_risk_score(asset_returns)
print(f"  Systemic risk scores: {np.round(risk_scores['scores'], 4)}")

# --- Optimal stopping ---
from wraquant.math.optimal_stopping import (
    longstaff_schwartz,
    optimal_exit_threshold,
    secretary_problem_threshold,
)

print(f"\n=== Optimal Stopping ===")
# Longstaff-Schwartz for American option
paths = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, (100, 50)), axis=1))
ls = longstaff_schwartz(paths, strike=100, rf=0.05, dt=1 / 252)
print(f"  American put value (Longstaff-Schwartz): {ls['price']:.4f}")

exit_thresh = optimal_exit_threshold(mu=0.05, sigma=0.2, rf=0.03)
print(f"  Optimal exit threshold (GBM): {exit_thresh['threshold']:.4f}")

secretary = secretary_problem_threshold(n_candidates=100)
print(f"  Secretary problem: review first {secretary['review_count']} of 100 candidates")

# --- Causal inference ---
from wraquant.causal.treatment import (
    propensity_score,
    ipw_ate,
    diff_in_diff,
)

print(f"\n=== Causal Inference ===")
n_units = 200
X = rng.normal(0, 1, (n_units, 3))
treatment = (X[:, 0] + rng.normal(0, 0.5, n_units) > 0).astype(float)
outcome = 2.0 + 0.5 * X[:, 0] + 3.0 * treatment + rng.normal(0, 1, n_units)

ps = propensity_score(X, treatment)
print(f"  Propensity scores: mean={ps.mean():.3f}, std={ps.std():.3f}")

ate = ipw_ate(outcome, treatment, ps)
print(f"  IPW ATE: {ate.ate:.2f} (true=3.0), SE: {ate.se:.2f}")

# --- Bayesian analysis ---
from wraquant.bayes.models import bayesian_regression, bayesian_sharpe

print(f"\n=== Bayesian Analysis ===")
X_reg = rng.normal(0, 1, (100, 2))
y_reg = 1.0 + 2.0 * X_reg[:, 0] - 0.5 * X_reg[:, 1] + rng.normal(0, 0.5, 100)
bayes_reg = bayesian_regression(X_reg, y_reg, n_samples=2000, seed=42)
print(f"  Posterior means: {np.round(bayes_reg['posterior_mean'], 3)}")
print(f"  True betas: [1.0, 2.0, -0.5]")

strategy_returns = rng.normal(0.0005, 0.02, 252)
bs = bayesian_sharpe(strategy_returns, n_samples=5000, seed=42)
print(f"  Bayesian Sharpe: {bs['mean']:.3f}, 95% CI: [{bs['ci_lower']:.3f}, {bs['ci_upper']:.3f}]")

# --- Regime detection ---
from wraquant.regimes.hmm import fit_hmm
from wraquant.regimes.changepoint import online_changepoint

print(f"\n=== Regime Detection ===")
# Two-regime data
regime1 = rng.normal(0.001, 0.01, 200)
regime2 = rng.normal(-0.001, 0.03, 100)
regime_data = np.concatenate([regime1, regime2, rng.normal(0.001, 0.01, 200)])

hmm = fit_hmm(regime_data, n_states=2)
print(f"  HMM states: {len(np.unique(hmm['states']))}")
print(f"  State means: {np.round(hmm['means'], 5)}")
print(f"  State stds: {np.round(hmm['stds'], 4)}")

cp = online_changepoint(regime_data, hazard_rate=1 / 100)
print(f"  Online changepoint probabilities shape: {cp['run_length_probs'].shape}")
