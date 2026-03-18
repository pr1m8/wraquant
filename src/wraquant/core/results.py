"""Standardized result containers for wraquant.

Dataclasses that provide consistent, IDE-friendly access to results
from GARCH fitting, backtesting, forecasting, and other operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class GARCHResult:
    """Result from GARCH-family model fitting.

    Attributes:
        params: Fitted parameters dict (omega, alpha, beta, ...).
        conditional_volatility: Time series of conditional std dev.
        standardized_residuals: Residuals / conditional vol.
        aic: Akaike Information Criterion.
        bic: Bayesian Information Criterion.
        log_likelihood: Maximized log-likelihood.
        persistence: Sum of alpha + beta parameters.
        half_life: Periods for shock to decay 50%.
        unconditional_variance: Long-run variance.
        model: Underlying fitted model object.
    """

    params: dict[str, float]
    conditional_volatility: pd.Series
    standardized_residuals: np.ndarray
    aic: float
    bic: float
    log_likelihood: float
    persistence: float
    half_life: float
    unconditional_variance: float
    model: Any = None
    ljung_box: dict | None = None


@dataclass
class BacktestResult:
    """Result from a backtest run.

    Attributes:
        returns: Portfolio return series.
        equity_curve: Cumulative equity curve.
        metrics: Dict of performance metrics (Sharpe, max_dd, etc.).
        trades: List of trade records (if event-driven).
        signals: Signal series used.
    """

    returns: pd.Series
    equity_curve: pd.Series
    metrics: dict[str, float]
    trades: list[dict] = field(default_factory=list)
    signals: pd.Series | None = None

    @property
    def sharpe(self) -> float:
        return self.metrics.get("sharpe_ratio", float("nan"))

    @property
    def max_drawdown(self) -> float:
        return self.metrics.get("max_drawdown", float("nan"))


@dataclass
class ForecastResult:
    """Result from time series forecasting.

    Attributes:
        forecast: Point forecast values.
        confidence_lower: Lower confidence bound.
        confidence_upper: Upper confidence bound.
        method: Forecasting method used.
        fitted_values: In-sample fitted values.
        residuals: In-sample residuals.
        metrics: Fit metrics (AIC, BIC, RMSE).
    """

    forecast: pd.Series | np.ndarray
    confidence_lower: np.ndarray | None = None
    confidence_upper: np.ndarray | None = None
    method: str = ""
    fitted_values: np.ndarray | None = None
    residuals: np.ndarray | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    model: Any = None
