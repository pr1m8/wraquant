"""Statistical analysis for financial data.

Covers descriptive statistics, hypothesis tests, correlation/covariance
estimation, distribution fitting, cointegration, regression, factor models,
and advanced dependence measures.
"""

from wraquant.stats.cointegration import (
    engle_granger,
    find_cointegrated_pairs,
    half_life,
    hedge_ratio,
    johansen,
    pairs_backtest_signals,
    spread,
    zscore_signal,
)
from wraquant.stats.correlation import (
    correlation_matrix,
    correlation_significance,
    distance_correlation,
    kendall_tau,
    minimum_spanning_tree_correlation,
    mutual_information,
    partial_correlation,
    rolling_correlation,
    shrunk_covariance,
)
from wraquant.stats.dependence import (
    concordance_index,
    copula_selection,
    rank_correlation_matrix,
    tail_dependence_coefficient,
)
from wraquant.stats.descriptive import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    max_drawdown,
    omega_ratio,
    return_attribution,
    risk_contribution,
    rolling_drawdown,
    rolling_sharpe,
    summary_stats,
)
from wraquant.stats.distributions import (
    anderson_darling,
    best_fit_distribution,
    fit_distribution,
    fit_stable_distribution,
    hurst_exponent,
    jarque_bera,
    kernel_density_estimate,
    kolmogorov_smirnov,
    qqplot_data,
    tail_index,
    tail_ratio,
)
from wraquant.stats.factor import (
    factor_attribution,
    fama_french_regression,
    information_coefficient,
    quantile_analysis,
)
from wraquant.stats.factor_analysis import (
    common_factors,
    factor_correlation,
    factor_exposure,
    factor_loadings,
    factor_mimicking_portfolios,
    factor_risk_decomposition,
    fama_french_factors,
    pca_factors,
    risk_factor_decomposition,
    scree_plot_data,
    varimax_rotation,
)
from wraquant.stats.regression import (
    fama_macbeth,
    newey_west_ols,
    ols,
    rolling_ols,
    wls,
)
from wraquant.stats.robust import (
    huber_mean,
    mad,
    outlier_detection,
    robust_covariance,
    robust_zscore,
    trimmed_mean,
    trimmed_std,
    winsorize,
)
from wraquant.stats.tests import (
    breusch_pagan,
    chow_test,
    durbin_watson,
    shapiro_wilk,
    test_autocorrelation,
    test_normality,
    test_stationarity,
    variance_inflation_factor,
    white_test,
)

__all__ = [
    # descriptive
    "summary_stats",
    "annualized_return",
    "annualized_volatility",
    "max_drawdown",
    "calmar_ratio",
    "omega_ratio",
    "rolling_sharpe",
    "rolling_drawdown",
    "return_attribution",
    "risk_contribution",
    # tests
    "test_normality",
    "test_stationarity",
    "test_autocorrelation",
    "shapiro_wilk",
    "durbin_watson",
    "breusch_pagan",
    "white_test",
    "chow_test",
    "variance_inflation_factor",
    # correlation
    "correlation_matrix",
    "shrunk_covariance",
    "rolling_correlation",
    "partial_correlation",
    "distance_correlation",
    "kendall_tau",
    "mutual_information",
    "correlation_significance",
    "minimum_spanning_tree_correlation",
    # dependence
    "tail_dependence_coefficient",
    "copula_selection",
    "rank_correlation_matrix",
    "concordance_index",
    # distributions
    "fit_distribution",
    "fit_stable_distribution",
    "tail_ratio",
    "tail_index",
    "hurst_exponent",
    "qqplot_data",
    "jarque_bera",
    "kolmogorov_smirnov",
    "anderson_darling",
    "best_fit_distribution",
    "kernel_density_estimate",
    # robust
    "mad",
    "winsorize",
    "trimmed_mean",
    "trimmed_std",
    "robust_zscore",
    "robust_covariance",
    "huber_mean",
    "outlier_detection",
    # cointegration
    "engle_granger",
    "johansen",
    "half_life",
    "spread",
    "zscore_signal",
    "hedge_ratio",
    "pairs_backtest_signals",
    "find_cointegrated_pairs",
    # regression
    "ols",
    "rolling_ols",
    "wls",
    "fama_macbeth",
    "newey_west_ols",
    # factor
    "fama_french_regression",
    "factor_attribution",
    "information_coefficient",
    "quantile_analysis",
    # factor_analysis
    "pca_factors",
    "factor_loadings",
    "scree_plot_data",
    "varimax_rotation",
    "factor_mimicking_portfolios",
    "risk_factor_decomposition",
    "factor_correlation",
    "common_factors",
    "fama_french_factors",
    "factor_exposure",
    "factor_risk_decomposition",
]
