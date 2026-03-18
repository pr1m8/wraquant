"""Advanced mathematical tools for quantitative finance.

Submodules
----------
network
    Financial network analysis, centrality, and contagion.
levy
    Lévy processes for fat-tailed asset models.
optimal_stopping
    Optimal stopping theory for finance.
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
