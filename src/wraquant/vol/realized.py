"""Realized volatility estimators.

Provides various volatility estimators from OHLCV data, including
classical, range-based, and high-frequency estimators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def realized_volatility(
    returns: pd.Series,
    window: int = 20,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    """Rolling realized volatility from return series.

    Parameters:
        returns: Return series.
        window: Rolling window size.
        annualize: Whether to annualize the volatility.
        periods_per_year: Periods per year for annualization.

    Returns:
        Rolling realized volatility series.
    """
    vol = returns.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol


def parkinson(
    high: pd.Series,
    low: pd.Series,
    window: int = 20,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    """Parkinson (1980) range-based volatility estimator.

    More efficient than close-to-close for continuous processes.

    Parameters:
        high: High price series.
        low: Low price series.
        window: Rolling window size.
        annualize: Whether to annualize.
        periods_per_year: Periods per year.

    Returns:
        Parkinson volatility series.
    """
    log_hl = np.log(high / low)
    factor = 1.0 / (4.0 * np.log(2))
    var = factor * (log_hl**2).rolling(window).mean()
    vol = np.sqrt(var)
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol


def garman_klass(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    """Garman-Klass (1980) volatility estimator.

    Uses open, high, low, close for higher efficiency than
    Parkinson or close-to-close.

    Parameters:
        open_: Open price series.
        high: High price series.
        low: Low price series.
        close: Close price series.
        window: Rolling window size.
        annualize: Whether to annualize.
        periods_per_year: Periods per year.

    Returns:
        Garman-Klass volatility series.
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    var = (0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2).rolling(window).mean()
    vol = np.sqrt(var.clip(lower=0))
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol


def rogers_satchell(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    """Rogers-Satchell (1991) volatility estimator.

    Handles drift in the price process, unlike Parkinson or Garman-Klass.

    Parameters:
        open_: Open price series.
        high: High price series.
        low: Low price series.
        close: Close price series.
        window: Rolling window size.
        annualize: Whether to annualize.
        periods_per_year: Periods per year.

    Returns:
        Rogers-Satchell volatility series.
    """
    log_ho = np.log(high / open_)
    log_hc = np.log(high / close)
    log_lo = np.log(low / open_)
    log_lc = np.log(low / close)

    var = (log_ho * log_hc + log_lo * log_lc).rolling(window).mean()
    vol = np.sqrt(var.clip(lower=0))
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol


def yang_zhang(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    """Yang-Zhang (2000) volatility estimator.

    Combines overnight and Rogers-Satchell estimators for minimum
    variance under drift and opening jumps.

    Parameters:
        open_: Open price series.
        high: High price series.
        low: Low price series.
        close: Close price series.
        window: Rolling window size.
        annualize: Whether to annualize.
        periods_per_year: Periods per year.

    Returns:
        Yang-Zhang volatility series.
    """
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    # Overnight variance
    log_oc = np.log(open_ / close.shift(1))
    overnight_var = log_oc.rolling(window).var()

    # Open-to-close variance
    log_co = np.log(close / open_)
    oc_var = log_co.rolling(window).var()

    # Rogers-Satchell
    rs = rogers_satchell(open_, high, low, close, window, annualize=False)
    rs_var = rs**2

    var = overnight_var + k * oc_var + (1 - k) * rs_var
    vol = np.sqrt(var.clip(lower=0))
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol
