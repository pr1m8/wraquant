"""Standardized result containers for wraquant.

Dataclasses that provide consistent, IDE-friendly access to results
from GARCH fitting, backtesting, forecasting, and other operations.
Each result type carries chaining methods that lazily import downstream
modules, enabling composable workflows:

    garch_result.to_var(alpha=0.05)   # -> risk/var
    garch_result.plot()                # -> viz
    backtest_result.to_tearsheet()     # -> backtest/tearsheet
    forecast_result.plot()             # -> viz
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

    def to_var(self, alpha: float = 0.05) -> dict:
        """Compute GARCH-based Value at Risk.

        Uses the fitted conditional volatility to estimate VaR via
        ``wraquant.risk.var.garch_var``.

        Parameters:
            alpha: Significance level (default 0.05 = 95% VaR).

        Returns:
            Dict with VaR results from ``risk.var.garch_var``.
        """
        from wraquant.risk.var import garch_var

        return garch_var(
            self.conditional_volatility,
            alpha=alpha,
        )

    def plot(self) -> Any:
        """Plot conditional volatility using viz module.

        Returns:
            Plotly figure object.
        """
        from wraquant.viz.interactive import plotly_returns

        return plotly_returns(self.conditional_volatility, title="Conditional Volatility")

    def summary(self) -> str:
        """Human-readable summary of GARCH fit.

        Returns:
            Multi-line string with key diagnostics.
        """
        param_str = ", ".join(f"{k}={v:.6f}" for k, v in self.params.items())
        lines = [
            f"GARCH Result",
            f"  Parameters: {param_str}",
            f"  Persistence: {self.persistence:.4f}",
            f"  Half-life: {self.half_life:.1f} periods",
            f"  Unconditional vol: {np.sqrt(self.unconditional_variance):.4f}",
            f"  AIC: {self.aic:.2f}  BIC: {self.bic:.2f}",
            f"  Log-likelihood: {self.log_likelihood:.2f}",
        ]
        return "\n".join(lines)


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

    def to_tearsheet(self, **kwargs: Any) -> dict:
        """Generate a comprehensive tearsheet.

        Delegates to ``wraquant.backtest.tearsheet.comprehensive_tearsheet``.

        Parameters:
            **kwargs: Forwarded to ``comprehensive_tearsheet``.

        Returns:
            Dict with full tearsheet analysis.
        """
        from wraquant.backtest.tearsheet import comprehensive_tearsheet

        return comprehensive_tearsheet(self.returns, **kwargs)

    def plot(self) -> Any:
        """Plot the equity curve using viz module.

        Returns:
            Plotly figure object.
        """
        from wraquant.viz.interactive import plotly_returns

        return plotly_returns(self.equity_curve, title="Equity Curve")

    def summary(self) -> str:
        """Human-readable summary of backtest results.

        Returns:
            Multi-line string with key performance metrics.
        """
        lines = ["Backtest Result"]
        for key, value in self.metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        lines.append(f"  Trades: {len(self.trades)}")
        return "\n".join(lines)


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

    def plot(self) -> Any:
        """Plot forecast with confidence bounds using viz module.

        Returns:
            Plotly figure object.
        """
        from wraquant.viz.interactive import plotly_returns

        forecast_series = (
            self.forecast
            if isinstance(self.forecast, pd.Series)
            else pd.Series(self.forecast)
        )
        return plotly_returns(forecast_series, title=f"Forecast ({self.method})")

    def summary(self) -> str:
        """Human-readable summary of forecast.

        Returns:
            Multi-line string with method and fit metrics.
        """
        lines = [f"Forecast Result (method={self.method})"]
        for key, value in self.metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        if isinstance(self.forecast, (pd.Series, np.ndarray)):
            n = len(self.forecast)
            lines.append(f"  Horizon: {n} periods")
        return "\n".join(lines)
