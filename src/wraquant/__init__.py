"""wraquant — The ultimate quant finance toolkit for Python.

Provides a composable, backend-agnostic framework for quantitative finance:
data fetching, time series analysis, risk management, portfolio optimization,
backtesting, options pricing, forex analysis, and more.

Example:
    >>> import wraquant as wq
    >>> cfg = wq.get_config()
    >>> cfg.backend
    <Backend.PANDAS: 'pandas'>
"""

from __future__ import annotations

from typing import Any

from wraquant._compat import Backend
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

__version__ = "0.1.0"

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
