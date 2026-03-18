"""Volatility models: EWMA, GARCH family."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wraquant.core.decorators import requires_extra


def ewma_volatility(
    returns: pd.Series,
    span: int = 30,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    """Exponentially weighted moving average volatility.

    Parameters:
        returns: Return series.
        span: EWMA span parameter (lambda = 1 - 2/(span+1)).
        annualize: Whether to annualize.
        periods_per_year: Periods per year.

    Returns:
        EWMA volatility series.
    """
    var = returns.ewm(span=span).var()
    vol = np.sqrt(var)
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol


@requires_extra("timeseries")
def garch_forecast(
    returns: pd.Series,
    model: str = "GARCH",
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
    horizon: int = 1,
) -> dict[str, Any]:
    """Fit a GARCH-family model and produce volatility forecast.

    Parameters:
        returns: Return series (typically scaled by 100).
        model: Model type ('GARCH', 'EGARCH', 'GJR-GARCH', 'TARCH').
        p: GARCH lag order.
        q: ARCH lag order.
        dist: Error distribution ('normal', 'studentst', 'skewt', 'ged').
        horizon: Forecast horizon in periods.

    Returns:
        Dict with 'model', 'params', 'conditional_vol', 'forecast_vol',
        'aic', 'bic'.

    Example:
        >>> result = garch_forecast(returns * 100, model='GARCH')  # doctest: +SKIP
    """
    from arch import arch_model

    vol_model = model.upper().replace("-", "")

    am = arch_model(
        returns * 100,
        vol=vol_model,
        p=p,
        q=q,
        dist=dist,
        mean="Constant",
    )

    fit = am.fit(disp="off")
    forecasts = fit.forecast(horizon=horizon)

    return {
        "model": vol_model,
        "params": dict(fit.params),
        "conditional_vol": fit.conditional_volatility / 100,
        "forecast_vol": np.sqrt(forecasts.variance.iloc[-1].values) / 100,
        "aic": fit.aic,
        "bic": fit.bic,
        "residuals": fit.resid / 100,
    }
