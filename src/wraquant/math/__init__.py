"""Advanced mathematical tools for quantitative finance.

Provides specialized mathematical methods that go beyond standard
statistics, serving as building blocks for sophisticated quant models.
Covers financial network analysis, Levy process simulation and fitting,
and optimal stopping theory -- areas where standard libraries fall short
of the finance-specific requirements.

Key sub-modules:

- **Network** (``network``) -- Financial network analysis built on graph
  theory: ``correlation_network`` (construct a graph from a correlation
  matrix), ``minimum_spanning_tree`` (Mantegna's MST for portfolio
  clustering), ``centrality_measures`` (identify systemically important
  assets), ``community_detection`` (find sector-like groupings),
  ``systemic_risk_score`` (aggregate contagion risk), and
  ``granger_network`` (causal connectivity via Granger tests).
- **Levy** (``levy``) -- Fat-tailed stochastic processes beyond Gaussian
  models: ``variance_gamma_simulate`` / ``variance_gamma_pdf`` (VG
  process -- captures excess kurtosis and skewness), ``nig_simulate`` /
  ``nig_pdf`` (Normal Inverse Gaussian), ``cgmy_simulate`` (tempered
  stable), ``levy_stable_simulate``, and ``fit_variance_gamma`` /
  ``fit_nig`` for parameter estimation.
- **Optimal stopping** (``optimal_stopping``) -- Decision-theoretic tools:
  ``longstaff_schwartz`` (American option pricing via regression),
  ``binomial_american`` (binomial lattice), ``optimal_exit_threshold``
  (when to exit a mean-reverting trade),
  ``sequential_probability_ratio`` (Wald's SPRT for hypothesis testing),
  ``cusum_stopping`` (cumulative sum changepoint detection), and
  ``secretary_problem_threshold``.

Example:
    >>> from wraquant.math import correlation_network, variance_gamma_simulate
    >>> G = correlation_network(returns_df, threshold=0.5)
    >>> paths = variance_gamma_simulate(n_paths=1000, n_steps=252, sigma=0.2)

Use ``wraquant.math`` for specialized mathematical modeling.  For standard
statistical tests and regression, use ``wraquant.stats``.  For stochastic
process simulation in a pricing context, see ``wraquant.price.stochastic``.
For network visualization, see ``wraquant.viz.plotly_network_graph``.
"""

from __future__ import annotations

from wraquant.math.levy import (
    cgmy_simulate,
    characteristic_function_vg,
    fit_nig,
    fit_variance_gamma,
    levy_stable_simulate,
    nig_pdf,
    nig_simulate,
    variance_gamma_pdf,
    variance_gamma_simulate,
)
from wraquant.math.network import (
    centrality_measures,
    community_detection,
    contagion_simulation,
    correlation_network,
    granger_network,
    minimum_spanning_tree,
    systemic_risk_score,
)
from wraquant.math.optimal_stopping import (
    binomial_american,
    cusum_stopping,
    longstaff_schwartz,
    optimal_exit_threshold,
    secretary_problem_threshold,
    sequential_probability_ratio,
)

__all__ = [
    # network
    "correlation_network",
    "minimum_spanning_tree",
    "centrality_measures",
    "community_detection",
    "systemic_risk_score",
    "contagion_simulation",
    "granger_network",
    # levy
    "variance_gamma_pdf",
    "variance_gamma_simulate",
    "nig_pdf",
    "nig_simulate",
    "cgmy_simulate",
    "fit_variance_gamma",
    "fit_nig",
    "levy_stable_simulate",
    "characteristic_function_vg",
    # optimal_stopping
    "longstaff_schwartz",
    "binomial_american",
    "optimal_exit_threshold",
    "sequential_probability_ratio",
    "cusum_stopping",
    "secretary_problem_threshold",
]
