"""wraquant -- The ultimate quant finance toolkit for Python.

A composable, backend-agnostic framework for quantitative finance covering
the full research-to-production pipeline: data fetching, time series
analysis, statistical modeling, volatility modeling, risk management,
portfolio optimization, backtesting, options pricing, regime detection,
machine learning, causal inference, forex analysis, and interactive
visualization.

Key modules:

- **data** -- Fetch prices, macro data, and alternative data from Yahoo
  Finance, FRED, NASDAQ Data Link, and CSV/Parquet files.  Clean,
  validate, and transform financial time series.
- **ta** -- 263 technical analysis indicators across 19 sub-modules
  (overlap, momentum, volume, trend, volatility, patterns, signals,
  cycles, fibonacci, smoothing, exotic, support/resistance, and more).
- **stats** -- Descriptive statistics, hypothesis tests, correlation,
  cointegration, regression, factor models, and robust estimators.
- **ts** -- Time series decomposition, seasonality detection, change-point
  detection, stationarity transforms, and forecasting (ARIMA, ETS, theta,
  ensemble, stochastic processes).
- **vol** -- Realized volatility estimators (Yang-Zhang, Garman-Klass),
  GARCH family (EGARCH, GJR, FIGARCH, HARCH), DCC, stochastic vol,
  Hawkes processes, and implied vol surfaces.
- **risk** -- Performance metrics, VaR/CVaR, portfolio risk decomposition,
  beta estimation, factor models, tail risk, copulas, stress testing,
  credit risk, survival analysis, and Monte Carlo simulation.
- **regimes** -- Hidden Markov Models, Markov-switching regression,
  Kalman filter/smoother, change-point detection, and regime-aware
  portfolio construction.
- **opt** -- Portfolio optimization (MVO, risk parity, Black-Litterman,
  HRP), convex/linear/nonlinear solvers, and multi-objective optimization.
- **ml** -- Feature engineering (triple barrier, fractional
  differentiation), purged cross-validation, walk-forward training,
  LSTM/Transformer forecasting, and online learning.
- **price** -- Options pricing (Black-Scholes, binomial, Monte Carlo),
  Greeks, fixed income, yield curves, Levy process pricing, FBSDE
  solvers, and stochastic process simulators.
- **backtest** -- Vectorized and event-driven backtesting engines,
  strategy abstractions, position sizing, tearsheet generation, and
  30+ performance metrics.
- **viz** -- Interactive Plotly dashboards (portfolio, regime, risk,
  technical), candlestick charts, vol surfaces, and correlation networks.

Example:
    >>> import wraquant as wq
    >>> prices = wq.data.fetch_prices("AAPL", start="2020-01-01")
    >>> rets = wq.returns(prices["close"])
    >>> stats = wq.stats.summary_stats(rets)
    >>> regimes = wq.detect_regimes(rets, n_regimes=2)

Use ``wraquant.data`` for ingestion, ``wraquant.stats`` or ``wraquant.ta``
for analysis, ``wraquant.risk`` for risk measurement, ``wraquant.opt`` for
allocation, and ``wraquant.backtest`` for strategy evaluation.  For
higher-level workflows, see ``wraquant.compose`` and ``wraquant.recipes``.
"""

from __future__ import annotations

from typing import Any

from wraquant._compat import Backend
from wraquant.compose import (
    Workflow,
    WorkflowResult,
    ml_workflow,
    portfolio_workflow,
    quick_analysis_workflow,
    risk_workflow,
    steps,
)
from wraquant.core.config import WQConfig, get_config, reset_config
from wraquant.core.exceptions import (
    BacktestError,
    ConfigError,
    DataFetchError,
    MissingDependencyError,
    OptimizationError,
    PricingError,
    ValidationError,
    WQError,
)
from wraquant.core.types import (
    ArrayLike,
    AssetClass,
    CovarianceMatrix,
    Currency,
    DateLike,
    Frequency,
    OHLCVFrame,
    OptionStyle,
    OptionType,
    OrderSide,
    PriceFrame,
    PriceSeries,
    RegimeState,
    ReturnFrame,
    ReturnSeries,
    ReturnType,
    RiskMeasure,
    VolModel,
    WeightsArray,
)
from wraquant.frame.factory import frame, series
from wraquant.frame.ops import (
    cumulative_returns,
    drawdowns,
    ewm_mean,
    log_returns,
    resample,
    returns,
    rolling_mean,
    rolling_std,
)
from wraquant.recipes import analyze

__version__ = "1.1.0"

__all__ = [
    # Version
    "__version__",
    # Config
    "WQConfig",
    "get_config",
    "reset_config",
    "Backend",
    # Exceptions
    "WQError",
    "MissingDependencyError",
    "DataFetchError",
    "ValidationError",
    "ConfigError",
    "OptimizationError",
    "BacktestError",
    "PricingError",
    # Types & Enums
    "DateLike",
    "ArrayLike",
    "PriceSeries",
    "ReturnSeries",
    "PriceFrame",
    "ReturnFrame",
    "OHLCVFrame",
    "WeightsArray",
    "CovarianceMatrix",
    "Frequency",
    "AssetClass",
    "Currency",
    "ReturnType",
    "OptionType",
    "OptionStyle",
    "OrderSide",
    "RegimeState",
    "RiskMeasure",
    "VolModel",
    # Frame factories
    "frame",
    "series",
    # Operations
    "returns",
    "log_returns",
    "cumulative_returns",
    "drawdowns",
    "rolling_mean",
    "rolling_std",
    "ewm_mean",
    "resample",
    # Compose
    "Workflow",
    "WorkflowResult",
    "steps",
    "quick_analysis_workflow",
    "risk_workflow",
    "ml_workflow",
    "portfolio_workflow",
    # Convenience functions
    "detect_regimes",
    "forecast",
    "backtest",
    "analyze",
]

# Lazy-loaded submodules — only imported when first accessed
_LAZY_SUBMODULES = {
    "core",
    "data",
    "ts",
    "stats",
    "vol",
    "ta",
    "ml",
    "opt",
    "price",
    "regimes",
    "risk",
    "backtest",
    "forex",
    "viz",
    "math",
    "bayes",
    "io",
    "econometrics",
    "experiment",
    "microstructure",
    "execution",
    "causal",
    "dashboard",
    "news",
    "fundamental",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        import importlib

        return importlib.import_module(f"wraquant.{name}")
    raise AttributeError(f"module 'wraquant' has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Top-level convenience functions
# ---------------------------------------------------------------------------


def detect_regimes(returns, method="hmm", n_regimes=2, **kwargs):
    """Detect market regimes. See wraquant.regimes.detect_regimes."""
    from wraquant.regimes.base import detect_regimes as _detect

    return _detect(returns, method=method, n_regimes=n_regimes, **kwargs)


def forecast(data, method="auto", horizon=10, **kwargs):
    """Forecast time series. See wraquant.ts.forecasting."""
    from wraquant.ts.forecasting import auto_forecast

    return auto_forecast(data, h=horizon, **kwargs)


def backtest(strategy_fn, prices, **kwargs):
    """Run a backtest. See wraquant.backtest.engine."""
    from wraquant.backtest.engine import Backtest

    return Backtest(strategy_fn).run(prices, **kwargs)
