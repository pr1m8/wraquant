"""Statistical hypothesis tests for financial data."""

from __future__ import annotations

import pandas as pd
from scipy import stats as sp_stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss


def test_normality(data: pd.Series, method: str = "jarque_bera") -> dict:
    """Test whether a series is normally distributed.

    Parameters:
        data: Data series to test.
        method: Test method — ``"jarque_bera"`` (default), ``"shapiro"``,
            or ``"dagostino"``.

    Returns:
        Dictionary with ``statistic``, ``p_value``, and ``is_normal``
        (at 5% significance level).

    Raises:
        ValueError: If *method* is not recognized.
    """
    clean = data.dropna().values

    if method == "jarque_bera":
        stat, p = sp_stats.jarque_bera(clean)
    elif method == "shapiro":
        stat, p = sp_stats.shapiro(clean)
    elif method == "dagostino":
        stat, p = sp_stats.normaltest(clean)
    else:
        msg = f"Unknown normality test method: {method!r}"
        raise ValueError(msg)

    return {
        "statistic": float(stat),
        "p_value": float(p),
        "is_normal": bool(p > 0.05),
    }


def test_stationarity(data: pd.Series, method: str = "adf") -> dict:
    """Test whether a time series is stationary.

    Parameters:
        data: Time series to test.
        method: Test method — ``"adf"`` (Augmented Dickey-Fuller, default)
            or ``"kpss"``.

    Returns:
        Dictionary with ``statistic``, ``p_value``, and ``is_stationary``
        (at 5% significance level).

    Raises:
        ValueError: If *method* is not recognized.
    """
    clean = data.dropna().values

    if method == "adf":
        result = adfuller(clean, autolag="AIC")
        stat, p = result[0], result[1]
        is_stationary = bool(p < 0.05)
    elif method == "kpss":
        stat, p, _lags, _crit = kpss(clean, regression="c", nlags="auto")
        # KPSS null hypothesis is stationarity, so reject means non-stationary
        is_stationary = bool(p > 0.05)
    else:
        msg = f"Unknown stationarity test method: {method!r}"
        raise ValueError(msg)

    return {
        "statistic": float(stat),
        "p_value": float(p),
        "is_stationary": is_stationary,
    }


def test_autocorrelation(data: pd.Series, nlags: int = 10) -> dict:
    """Ljung-Box test for autocorrelation.

    Parameters:
        data: Time series to test.
        nlags: Number of lags to test.

    Returns:
        Dictionary with ``statistic`` (at max lag), ``p_value``,
        ``is_autocorrelated`` (at 5% significance), and the full
        ``results`` DataFrame.
    """
    clean = data.dropna()
    result = acorr_ljungbox(clean, lags=nlags, return_df=True)
    last_row = result.iloc[-1]
    return {
        "statistic": float(last_row["lb_stat"]),
        "p_value": float(last_row["lb_pvalue"]),
        "is_autocorrelated": bool(last_row["lb_pvalue"] < 0.05),
        "results": result,
    }
