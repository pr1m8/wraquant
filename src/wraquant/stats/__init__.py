"""Statistical analysis for financial data.

Covers descriptive statistics, hypothesis tests, correlation/covariance
estimation, and distribution fitting.
"""

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
]
