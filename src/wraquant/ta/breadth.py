"""Market breadth indicators.

This module provides indicators that measure the overall health and direction
of the broader market by analyzing the number of advancing/declining issues,
new highs/lows, and the percentage of components meeting certain criteria.
All functions accept ``pd.Series`` (or ``pd.DataFrame`` where noted) inputs
and return ``pd.Series``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "advance_decline_line",
    "advance_decline_ratio",
    "mcclellan_oscillator",
    "mcclellan_summation",
    "arms_index",
    "new_highs_lows",
    "percent_above_ma",
    "high_low_index",
    "bullish_percent",
    "cumulative_volume_index",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


from wraquant.ta._validators import validate_period as _validate_period
from wraquant.ta._validators import validate_series as _validate_series


def _ema(data: pd.Series, period: int) -> pd.Series:
    """Internal EMA helper to avoid circular import."""
    return data.ewm(span=period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Advance/Decline Line
# ---------------------------------------------------------------------------


def advance_decline_line(
    advancing: pd.Series,
    declining: pd.Series,
) -> pd.Series:
    """Advance/Decline Line -- cumulative sum of (advancing - declining).

    The A/D line is a breadth indicator that tracks the running total of
    the difference between the number of advancing and declining issues.

    Interpretation:
        - **Rising A/D line with rising market**: Healthy uptrend --
          broad participation confirms the rally.
        - **Falling A/D line with rising market**: Bearish divergence --
          fewer stocks participating in the rally. Distribution.
        - **Rising A/D line with falling market**: Bullish divergence --
          accumulation occurring beneath the surface.
        - The A/D line often leads the market at major turning points.

    Parameters
    ----------
    advancing : pd.Series
        Number of advancing issues per period.
    declining : pd.Series
        Number of declining issues per period.

    Returns
    -------
    pd.Series
        Cumulative A/D line values.

    Example
    -------
    >>> adv = pd.Series([200, 250, 180, 300, 220])
    >>> dec = pd.Series([100, 150, 220, 100, 180])
    >>> advance_decline_line(adv, dec)
    """
    advancing = _validate_series(advancing, "advancing")
    declining = _validate_series(declining, "declining")

    result = (advancing - declining).cumsum()
    result.name = "ad_line"
    return result


# ---------------------------------------------------------------------------
# Advance/Decline Ratio
# ---------------------------------------------------------------------------


def advance_decline_ratio(
    advancing: pd.Series,
    declining: pd.Series,
) -> pd.Series:
    """Advance/Decline Ratio — advancing / declining.

    Values above 1.0 indicate more advancers than decliners; below 1.0
    indicates more decliners.

    Parameters
    ----------
    advancing : pd.Series
        Number of advancing issues per period.
    declining : pd.Series
        Number of declining issues per period.

    Returns
    -------
    pd.Series
        A/D ratio values (NaN where declining is zero).

    Example
    -------
    >>> adv = pd.Series([200, 250, 180])
    >>> dec = pd.Series([100, 150, 220])
    >>> advance_decline_ratio(adv, dec)
    """
    advancing = _validate_series(advancing, "advancing")
    declining = _validate_series(declining, "declining")

    result = advancing / declining.replace(0, np.nan)
    result.name = "ad_ratio"
    return result


# ---------------------------------------------------------------------------
# McClellan Oscillator
# ---------------------------------------------------------------------------


def mcclellan_oscillator(
    advancing: pd.Series,
    declining: pd.Series,
    fast: int = 19,
    slow: int = 39,
) -> pd.Series:
    """McClellan Oscillator -- difference between fast and slow EMA of AD diff.

    ``McClellan = EMA(advancing - declining, fast) - EMA(advancing - declining, slow)``

    Interpretation:
        - **Above zero**: Short-term breadth momentum is positive
          (more stocks advancing than declining, accelerating).
        - **Below zero**: Short-term breadth momentum is negative.
        - **Above +100**: Very overbought breadth-wise.
        - **Below -100**: Very oversold breadth-wise.
        - **Zero-line crossover**: Breadth momentum shift.
        - Best used for timing entries: buy when the oscillator turns
          up from below -100 (oversold breadth bounce).

    Parameters
    ----------
    advancing : pd.Series
        Number of advancing issues per period.
    declining : pd.Series
        Number of declining issues per period.
    fast : int, default 19
        Fast EMA period.
    slow : int, default 39
        Slow EMA period.

    Returns
    -------
    pd.Series
        McClellan Oscillator values.

    Example
    -------
    >>> result = mcclellan_oscillator(advancing, declining)
    """
    advancing = _validate_series(advancing, "advancing")
    declining = _validate_series(declining, "declining")
    _validate_period(fast, "fast")
    _validate_period(slow, "slow")

    ad_diff = advancing - declining
    result = _ema(ad_diff, fast) - _ema(ad_diff, slow)
    result.name = "mcclellan_oscillator"
    return result


# ---------------------------------------------------------------------------
# McClellan Summation Index
# ---------------------------------------------------------------------------


def mcclellan_summation(
    advancing: pd.Series,
    declining: pd.Series,
    fast: int = 19,
    slow: int = 39,
) -> pd.Series:
    """McClellan Summation Index -- cumulative sum of the McClellan Oscillator.

    This is the running total of the McClellan Oscillator, providing a
    longer-term view of market breadth.

    Interpretation:
        - **Rising**: Long-term breadth is improving (more and more
          stocks participating in the advance).
        - **Falling**: Long-term breadth is deteriorating.
        - **Above +1000**: Strongly bullish long-term breadth.
        - **Below -1000**: Strongly bearish long-term breadth.
        - Acts as a long-term trend indicator for market internals.

    Parameters
    ----------
    advancing : pd.Series
        Number of advancing issues per period.
    declining : pd.Series
        Number of declining issues per period.
    fast : int, default 19
        Fast EMA period for the underlying oscillator.
    slow : int, default 39
        Slow EMA period for the underlying oscillator.

    Returns
    -------
    pd.Series
        McClellan Summation Index values.

    Example
    -------
    >>> result = mcclellan_summation(advancing, declining)
    """
    advancing = _validate_series(advancing, "advancing")
    declining = _validate_series(declining, "declining")
    _validate_period(fast, "fast")
    _validate_period(slow, "slow")

    osc = mcclellan_oscillator(advancing, declining, fast=fast, slow=slow)
    result = osc.cumsum()
    result.name = "mcclellan_summation"
    return result


# ---------------------------------------------------------------------------
# Arms Index (TRIN)
# ---------------------------------------------------------------------------


def arms_index(
    advancing_issues: pd.Series,
    declining_issues: pd.Series,
    advancing_volume: pd.Series,
    declining_volume: pd.Series,
) -> pd.Series:
    """Arms Index (TRIN) — Short-Term Trading Index.

    ``TRIN = (Advancing Issues / Declining Issues) /
             (Advancing Volume / Declining Volume)``

    Values below 1.0 are bullish (more volume flowing into advancers);
    values above 1.0 are bearish.

    Parameters
    ----------
    advancing_issues : pd.Series
        Number of advancing issues.
    declining_issues : pd.Series
        Number of declining issues.
    advancing_volume : pd.Series
        Total volume of advancing issues.
    declining_volume : pd.Series
        Total volume of declining issues.

    Returns
    -------
    pd.Series
        TRIN values (NaN where denominators are zero).

    Example
    -------
    >>> result = arms_index(adv_issues, dec_issues, adv_vol, dec_vol)
    """
    advancing_issues = _validate_series(advancing_issues, "advancing_issues")
    declining_issues = _validate_series(declining_issues, "declining_issues")
    advancing_volume = _validate_series(advancing_volume, "advancing_volume")
    declining_volume = _validate_series(declining_volume, "declining_volume")

    issue_ratio = advancing_issues / declining_issues.replace(0, np.nan)
    volume_ratio = advancing_volume / declining_volume.replace(0, np.nan)
    result = issue_ratio / volume_ratio.replace(0, np.nan)
    result.name = "arms_index"
    return result


# ---------------------------------------------------------------------------
# New Highs - New Lows
# ---------------------------------------------------------------------------


def new_highs_lows(
    new_highs: pd.Series,
    new_lows: pd.Series,
) -> pd.Series:
    """New Highs minus New Lows.

    A simple breadth measure: positive values indicate more new highs
    than new lows, suggesting bullish breadth.

    Parameters
    ----------
    new_highs : pd.Series
        Number of new highs per period.
    new_lows : pd.Series
        Number of new lows per period.

    Returns
    -------
    pd.Series
        New highs minus new lows.

    Example
    -------
    >>> nh = pd.Series([50, 60, 30])
    >>> nl = pd.Series([20, 40, 50])
    >>> new_highs_lows(nh, nl)
    """
    new_highs = _validate_series(new_highs, "new_highs")
    new_lows = _validate_series(new_lows, "new_lows")

    result = new_highs - new_lows
    result.name = "new_highs_lows"
    return result


# ---------------------------------------------------------------------------
# Percent Above Moving Average
# ---------------------------------------------------------------------------


def percent_above_ma(
    prices_df: pd.DataFrame,
    period: int = 50,
) -> pd.Series:
    """Percentage of components above their N-period moving average.

    For each row, computes how many columns have a value above their
    respective rolling SMA, expressed as a percentage.

    Parameters
    ----------
    prices_df : pd.DataFrame
        DataFrame where each column is a component's price series.
    period : int, default 50
        SMA look-back period.

    Returns
    -------
    pd.Series
        Percentage (0-100) of components above their SMA.

    Example
    -------
    >>> df = pd.DataFrame({"A": [10, 11, 12], "B": [20, 19, 18]})
    >>> percent_above_ma(df, period=2)
    """
    if not isinstance(prices_df, pd.DataFrame):
        raise TypeError(
            f"prices_df must be a pd.DataFrame, got {type(prices_df).__name__}"
        )
    _validate_period(period)

    sma = prices_df.rolling(window=period, min_periods=period).mean()
    above = (prices_df > sma).sum(axis=1)
    total = prices_df.notna().sum(axis=1).replace(0, np.nan)
    result = (above / total) * 100.0
    result.name = "percent_above_ma"
    return result


# ---------------------------------------------------------------------------
# High-Low Index
# ---------------------------------------------------------------------------


def high_low_index(
    new_highs: pd.Series,
    new_lows: pd.Series,
) -> pd.Series:
    """High-Low Index — new highs as a percentage of new highs + new lows.

    ``HLI = new_highs / (new_highs + new_lows) * 100``

    Values above 50 indicate more new highs; values below 50 indicate more
    new lows.

    Parameters
    ----------
    new_highs : pd.Series
        Number of new highs per period.
    new_lows : pd.Series
        Number of new lows per period.

    Returns
    -------
    pd.Series
        High-Low Index values in [0, 100] (NaN where both are zero).

    Example
    -------
    >>> nh = pd.Series([50, 60, 30])
    >>> nl = pd.Series([20, 40, 50])
    >>> high_low_index(nh, nl)
    """
    new_highs = _validate_series(new_highs, "new_highs")
    new_lows = _validate_series(new_lows, "new_lows")

    total = (new_highs + new_lows).replace(0, np.nan)
    result = (new_highs / total) * 100.0
    result.name = "high_low_index"
    return result


# ---------------------------------------------------------------------------
# Bullish Percent Index
# ---------------------------------------------------------------------------


def bullish_percent(
    prices_df: pd.DataFrame,
    period: int = 50,
) -> pd.Series:
    """Bullish Percent Index (simplified).

    Approximates the Bullish Percent Index by computing the percentage of
    components trading above their *period*-day simple moving average.
    The traditional BPI uses point-and-figure buy signals, but the SMA
    crossover is a widely accepted simplification.

    Parameters
    ----------
    prices_df : pd.DataFrame
        DataFrame where each column is a component's price series.
    period : int, default 50
        SMA look-back period (default 50-day MA).

    Returns
    -------
    pd.Series
        Bullish Percent values in [0, 100].

    Example
    -------
    >>> df = pd.DataFrame({"A": [10, 11, 12], "B": [20, 19, 18]})
    >>> bullish_percent(df, period=2)
    """
    result = percent_above_ma(prices_df, period=period)
    result.name = "bullish_percent"
    return result


# ---------------------------------------------------------------------------
# Cumulative Volume Index (CVI)
# ---------------------------------------------------------------------------


def cumulative_volume_index(
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Cumulative Volume Index (CVI).

    Adds volume on up days and subtracts volume on down days.

    ``CVI = cumsum(sign(close.diff()) * volume)``

    Parameters
    ----------
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume data.

    Returns
    -------
    pd.Series
        CVI values.

    Example
    -------
    >>> close = pd.Series([100, 102, 101, 103, 104.0])
    >>> volume = pd.Series([1000, 1500, 1200, 1800, 1600.0])
    >>> cumulative_volume_index(close, volume)
    """
    close = _validate_series(close, "close")
    volume = _validate_series(volume, "volume")

    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    result = (direction * volume).cumsum()
    result.name = "cvi"
    return result
