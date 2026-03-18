"""Time series forecasting wrappers."""

from __future__ import annotations

from typing import Any

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from wraquant.core.decorators import requires_extra


def exponential_smoothing(data: pd.Series, **kwargs: Any) -> Any:
    """Fit a Holt-Winters exponential smoothing model.

    Wraps ``statsmodels.tsa.holtwinters.ExponentialSmoothing``.

    Parameters:
        data: Time series to fit.
        **kwargs: Keyword arguments forwarded to
            ``ExponentialSmoothing``.

    Returns:
        Fitted ``HoltWintersResultsWrapper``.
    """
    model = ExponentialSmoothing(data, **kwargs)
    return model.fit()


@requires_extra("timeseries")
def auto_arima(data: pd.Series, **kwargs: Any) -> Any:
    """Automatically select and fit the best ARIMA model.

    Requires the ``timeseries`` optional dependency group
    (``pmdarima``).

    Parameters:
        data: Time series to fit.
        **kwargs: Keyword arguments forwarded to
            ``pmdarima.auto_arima``.

    Returns:
        Fitted ARIMA model from pmdarima.
    """
    import pmdarima as pm

    return pm.auto_arima(data, **kwargs)
