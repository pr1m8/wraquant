"""Statistical analysis for financial data.

Covers descriptive statistics, hypothesis tests, correlation/covariance
estimation, distribution fitting, cointegration, regression, and factor models.
"""

from wraquant.stats.cointegration import (
    engle_granger,
    find_cointegrated_pairs,
    half_life,
    hedge_ratio,
    pairs_backtest_signals,
    spread,
    zscore_signal,
)
from wraquant.stats.correlation import (
    correlation_matrix,
    rolling_correlation,
    shrunk_covariance,
)
from wraquant.stats.descriptive import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    max_drawdown,
    omega_ratio,
    summary_stats,
)
from wraquant.stats.distributions import (
    fit_distribution,
    hurst_exponent,
    tail_ratio,
)
from wraquant.stats.factor import (
    factor_attribution,
    fama_french_regression,
    information_coefficient,
    quantile_analysis,
)
from wraquant.stats.regression import (
    fama_macbeth,
    newey_west_ols,
    ols,
    rolling_ols,
    wls,
)
from wraquant.stats.tests import (
    test_autocorrelation,
    test_normality,
    test_stationarity,
)

__all__ = [
    # descriptive
    "summary_stats",
    "annualized_return",
    "annualized_volatility",
    "max_drawdown",
    "calmar_ratio",
    "omega_ratio",
    # tests
    "test_normality",
    "test_stationarity",
    "test_autocorrelation",
    # correlation
    "correlation_matrix",
    "shrunk_covariance",
    "rolling_correlation",
    # distributions
    "fit_distribution",
    "tail_ratio",
    "hurst_exponent",
    # cointegration
    "engle_granger",
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
]
