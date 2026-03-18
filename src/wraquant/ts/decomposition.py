"""Time series decomposition methods.

Decomposition separates a time series into interpretable components --
typically trend, seasonal, and residual -- enabling cleaner analysis and
forecasting. This module provides three approaches:

1. **Classical decomposition** (``seasonal_decompose``) -- simple moving-
   average-based extraction of trend and seasonal components. Fast and
   easy to understand, but uses fixed seasonal patterns and loses data
   at endpoints.

2. **STL decomposition** (``stl_decompose``) -- Seasonal and Trend
   decomposition using Loess. Robust to outliers, allows the seasonal
   component to vary over time, and preserves all data points. This is
   the recommended default for most applications.

3. **Hodrick-Prescott filter** (``trend_filter``) -- extracts a smooth
   trend by penalising acceleration. Popular in macroeconomics but
   controversial due to endpoint instability and spurious cycle
   extraction.

When to use each:
    - **STL**: best general-purpose choice. Handles outliers, time-
      varying seasonality, and does not lose endpoints. Use this unless
      you have a specific reason not to.
    - **Classical decompose**: when you need a simple, fast
      decomposition and are comfortable with fixed seasonality. Good
      for EDA and presentation.
    - **HP filter**: for macroeconomic trend extraction where
      convention dictates its use (GDP, unemployment). Avoid for
      financial time series where it can create spurious cycles.

References:
    - Cleveland et al. (1990), "STL: A Seasonal-Trend Decomposition
      Procedure Based on Loess"
    - Hodrick & Prescott (1997), "Postwar U.S. Business Cycles: An
      Empirical Investigation"
    - Hamilton (2018), "Why You Should Never Use the Hodrick-Prescott
      Filter"
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import seasonal_decompose as sm_seasonal_decompose


def seasonal_decompose(
    data: pd.Series,
    period: int | None = None,
    model: str = "additive",
) -> Any:
    """Decompose a time series into trend, seasonal, and residual components.

    Classical decomposition extracts the trend via a centred moving
    average of width equal to the seasonal period, then computes the
    seasonal component as the average deviation from the trend for each
    season.

    When to use:
        Use classical decomposition for quick EDA and when the seasonal
        pattern is approximately constant over time. For production
        forecasting or when outliers are present, prefer
        ``stl_decompose`` which is more robust.

    Mathematical formulation:
        Additive:       y_t = T_t + S_t + R_t
        Multiplicative: y_t = T_t * S_t * R_t

        where T is trend, S is seasonal, and R is residual.

    How to interpret:
        - ``result.trend``: long-term direction of the series. NaN at
          endpoints (half the period on each side).
        - ``result.seasonal``: periodic pattern that repeats every
          ``period`` observations. Constant across years.
        - ``result.resid``: what remains after removing trend and
          seasonal. Should look like white noise if the decomposition
          is adequate.

    Parameters:
        data: Time series to decompose.
        period: Seasonal period (e.g., 12 for monthly data with annual
            seasonality, 252 for daily with annual). Inferred from the
            index frequency when *None*.
        model: ``"additive"`` (default) -- use when seasonal amplitude
            is roughly constant. ``"multiplicative"`` -- use when
            seasonal amplitude scales with the level.

    Returns:
        ``DecomposeResult`` with ``trend``, ``seasonal``, and ``resid``
        attributes, each a ``pd.Series``.

    Example:
        >>> import pandas as pd, numpy as np
        >>> idx = pd.date_range("2020-01-01", periods=120, freq="MS")
        >>> data = pd.Series(np.arange(120.0) + 5 * np.sin(np.arange(120) * 2 * np.pi / 12), index=idx)
        >>> result = seasonal_decompose(data, period=12)
        >>> result.seasonal.iloc[0] != 0
        True

    See Also:
        stl_decompose: Robust decomposition with time-varying seasonality.
        trend_filter: Hodrick-Prescott trend extraction.
    """
    return sm_seasonal_decompose(data, model=model, period=period)


def stl_decompose(
    data: pd.Series,
    period: int | None = None,
) -> Any:
    """STL (Seasonal and Trend decomposition using Loess) decomposition.

    STL uses locally weighted regression (Loess) to extract trend and
    seasonal components. Unlike classical decomposition, STL allows the
    seasonal pattern to evolve over time and is robust to outliers via
    iterative re-weighting.

    When to use:
        STL is the recommended default decomposition method. Use it
        when:
        - The seasonal pattern changes strength or shape over time.
        - The data contains outliers or level shifts.
        - You need values at the endpoints (no NaN padding).
        Prefer classical ``seasonal_decompose`` only for quick EDA
        where simplicity matters more than accuracy.

    Mathematical background:
        STL applies two nested Loess smoothers iteratively:
        1. Inner loop: extract seasonal component via Loess on
           sub-series (one per season), then extract trend via Loess
           on the deseasonalised series.
        2. Outer loop: compute robustness weights based on residual
           magnitude, then re-run the inner loop.

    How to interpret:
        Same as classical decomposition: ``trend``, ``seasonal``,
        ``resid``. The key difference is that ``seasonal`` can vary
        over time (plot it to see evolution). The ``resid`` should be
        approximately stationary with no remaining seasonal pattern.

    Parameters:
        data: Time series to decompose.
        period: Seasonal period (e.g., 7 for daily data with weekly
            seasonality). Uses 7 when *None* and no index frequency
            is available.

    Returns:
        ``STLForecast`` result with ``trend``, ``seasonal``, and
        ``resid`` attributes, each a ``pd.Series``.

    Example:
        >>> import pandas as pd, numpy as np
        >>> idx = pd.date_range("2020-01-01", periods=365, freq="D")
        >>> data = pd.Series(np.arange(365.0) + 3 * np.sin(np.arange(365) * 2 * np.pi / 7), index=idx)
        >>> result = stl_decompose(data, period=7)
        >>> hasattr(result, "trend")
        True

    See Also:
        seasonal_decompose: Simpler classical decomposition.
        trend_filter: Hodrick-Prescott trend extraction.

    References:
        - Cleveland et al. (1990), "STL: A Seasonal-Trend
          Decomposition Procedure Based on Loess"
    """
    if period is None:
        period = 7
    return STL(data, period=period).fit()


def trend_filter(
    data: pd.Series,
    method: str = "hp",
    lamb: float = 1600,
) -> pd.Series:
    """Extract the trend component from a time series.

    Currently supports the Hodrick-Prescott (HP) filter, which separates
    a time series into a smooth trend and a cyclical component by
    minimising the sum of squared deviations from the trend subject to
    a penalty on the trend's second difference (acceleration).

    When to use:
        The HP filter is standard in macroeconomic analysis (GDP,
        unemployment, inflation) where convention dictates its use.
        For financial time series, prefer STL or moving averages -- the
        HP filter can create spurious cycles and has well-documented
        endpoint instability (Hamilton 2018).

    Mathematical formulation:
        min_tau sum_{t=1}^T (y_t - tau_t)^2 +
                lambda * sum_{t=3}^T ((tau_t - tau_{t-1}) - (tau_{t-1} - tau_{t-2}))^2

        where tau is the trend and lambda controls smoothness.
        Higher lambda => smoother trend.

    How to interpret:
        The returned series is the trend component. The cyclical
        component is ``data - trend``. Common lambda values:
        - Monthly data: lambda = 129600 (Ravn & Uhlig)
        - Quarterly data: lambda = 1600 (the Hodrick-Prescott default)
        - Annual data: lambda = 6.25

    Parameters:
        data: Time series.
        method: Filter method -- ``"hp"`` (Hodrick-Prescott, default).
        lamb: Smoothing parameter. Default 1600 (appropriate for
            quarterly data).

    Returns:
        Trend component as a Series.

    Raises:
        ValueError: If *method* is not recognized.

    Example:
        >>> import pandas as pd, numpy as np
        >>> data = pd.Series(np.cumsum(np.random.randn(100)) + np.arange(100) * 0.5)
        >>> trend = trend_filter(data, lamb=1600)
        >>> len(trend) == len(data)
        True

    See Also:
        stl_decompose: Preferred for most decomposition tasks.
        seasonal_decompose: Classical moving-average decomposition.

    References:
        - Hodrick & Prescott (1997), "Postwar U.S. Business Cycles"
        - Hamilton (2018), "Why You Should Never Use the HP Filter"
    """
    if method == "hp":
        _cycle, trend = hpfilter(data.dropna(), lamb=lamb)
        return trend
    msg = f"Unknown trend filter method: {method!r}"
    raise ValueError(msg)
