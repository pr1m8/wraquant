"""Volume-based indicators.

This module provides indicators that incorporate volume data to measure
buying/selling pressure and trend confirmation. All functions accept
``pd.Series`` inputs and return ``pd.Series``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Re-export vwap from overlap (canonical location)
from wraquant.ta.overlap import vwap

__all__ = [
    "obv",
    "vwap",
    "ad_line",
    "cmf",
    "mfi",
    "eom",
    "force_index",
    "nvi",
    "pvi",
    "vpt",
    "adosc",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_series(data: pd.Series, name: str = "data") -> pd.Series:
    if not isinstance(data, pd.Series):
        raise TypeError(f"{name} must be a pd.Series, got {type(data).__name__}")
    return data


def _validate_period(period: int, name: str = "period") -> int:
    if period < 1:
        raise ValueError(f"{name} must be >= 1, got {period}")
    return period


def _ema(data: pd.Series, period: int) -> pd.Series:
    """Internal EMA helper."""
    return data.ewm(span=period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# On Balance Volume (OBV)
# ---------------------------------------------------------------------------


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On Balance Volume (OBV).

    OBV is a cumulative running total of volume. Volume is added on up-close
    days and subtracted on down-close days.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume data.

    Returns
    -------
    pd.Series
        OBV values.
    """
    _validate_series(close, "close")
    _validate_series(volume, "volume")

    direction = np.sign(close.diff())
    # First value has no diff — treat as 0
    direction.iloc[0] = 0
    result = (direction * volume).cumsum()
    result.name = "obv"
    return result


# ---------------------------------------------------------------------------
# Accumulation / Distribution Line
# ---------------------------------------------------------------------------


def ad_line(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Accumulation/Distribution Line (AD Line).

    Uses the Close Location Value (CLV) money flow multiplier.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume data.

    Returns
    -------
    pd.Series
        Cumulative A/D line values.
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")
    _validate_series(volume, "volume")

    hl_range = high - low
    clv = pd.Series(
        np.where(hl_range != 0, ((close - low) - (high - close)) / hl_range, 0.0),
        index=close.index,
    )
    money_flow_volume = clv * volume
    result = money_flow_volume.cumsum()
    result.name = "ad_line"
    return result


# ---------------------------------------------------------------------------
# Chaikin Money Flow (CMF)
# ---------------------------------------------------------------------------


def cmf(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Chaikin Money Flow (CMF).

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume data.
    period : int, default 20
        Look-back period.

    Returns
    -------
    pd.Series
        CMF values in [-1, 1].
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")
    _validate_series(volume, "volume")
    _validate_period(period)

    hl_range = high - low
    clv = pd.Series(
        np.where(hl_range != 0, ((close - low) - (high - close)) / hl_range, 0.0),
        index=close.index,
    )
    mf_volume = clv * volume
    result = (
        mf_volume.rolling(window=period, min_periods=period).sum()
        / volume.rolling(window=period, min_periods=period).sum()
    )
    result.name = "cmf"
    return result


# ---------------------------------------------------------------------------
# Money Flow Index (MFI)
# ---------------------------------------------------------------------------


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Money Flow Index (MFI).

    Often called the volume-weighted RSI.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume data.
    period : int, default 14
        Look-back period.

    Returns
    -------
    pd.Series
        MFI values in [0, 100].
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")
    _validate_series(volume, "volume")
    _validate_period(period)

    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume

    tp_diff = typical_price.diff()
    pos_flow = pd.Series(np.where(tp_diff > 0, raw_money_flow, 0.0), index=close.index)
    neg_flow = pd.Series(np.where(tp_diff < 0, raw_money_flow, 0.0), index=close.index)

    pos_sum = pos_flow.rolling(window=period, min_periods=period).sum()
    neg_sum = neg_flow.rolling(window=period, min_periods=period).sum()

    money_ratio = pos_sum / neg_sum
    result = 100.0 - (100.0 / (1.0 + money_ratio))
    result.name = "mfi"
    return result


# ---------------------------------------------------------------------------
# Ease of Movement (EOM)
# ---------------------------------------------------------------------------


def eom(
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Ease of Movement (EMV / EOM).

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    volume : pd.Series
        Volume data.
    period : int, default 14
        Smoothing period.

    Returns
    -------
    pd.Series
        EOM values (smoothed).
    """
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(volume, "volume")
    _validate_period(period)

    distance = ((high + low) / 2.0) - ((high.shift(1) + low.shift(1)) / 2.0)
    box_ratio = (volume / 1e6) / (high - low)  # scale volume down
    raw_eom = distance / box_ratio
    result = raw_eom.rolling(window=period, min_periods=period).mean()
    result.name = "eom"
    return result


# ---------------------------------------------------------------------------
# Force Index
# ---------------------------------------------------------------------------


def force_index(
    close: pd.Series,
    volume: pd.Series,
    period: int = 13,
) -> pd.Series:
    """Force Index.

    ``Force = close.diff() * volume``, then smoothed with EMA.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume data.
    period : int, default 13
        EMA smoothing period.

    Returns
    -------
    pd.Series
        Smoothed Force Index.
    """
    _validate_series(close, "close")
    _validate_series(volume, "volume")
    _validate_period(period)

    raw_force = close.diff() * volume
    result = _ema(raw_force, period)
    result.name = "force_index"
    return result


# ---------------------------------------------------------------------------
# Negative Volume Index (NVI)
# ---------------------------------------------------------------------------


def nvi(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Negative Volume Index (NVI).

    NVI focuses on days when volume decreases; the assumption is that
    smart money is active on low-volume days.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume data.

    Returns
    -------
    pd.Series
        NVI values (starts at 1000).
    """
    _validate_series(close, "close")
    _validate_series(volume, "volume")

    pct_change = close.pct_change()
    vol_change = volume.diff()

    nvi_values = np.empty(len(close))
    nvi_values[0] = 1000.0

    pct_arr = pct_change.values
    vol_arr = vol_change.values

    for i in range(1, len(close)):
        if vol_arr[i] < 0:
            nvi_values[i] = nvi_values[i - 1] * (1.0 + pct_arr[i])
        else:
            nvi_values[i] = nvi_values[i - 1]

    result = pd.Series(nvi_values, index=close.index, name="nvi")
    return result


# ---------------------------------------------------------------------------
# Positive Volume Index (PVI)
# ---------------------------------------------------------------------------


def pvi(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Positive Volume Index (PVI).

    PVI focuses on days when volume increases; the assumption is that
    the crowd follows price on high-volume days.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume data.

    Returns
    -------
    pd.Series
        PVI values (starts at 1000).
    """
    _validate_series(close, "close")
    _validate_series(volume, "volume")

    pct_change = close.pct_change()
    vol_change = volume.diff()

    pvi_values = np.empty(len(close))
    pvi_values[0] = 1000.0

    pct_arr = pct_change.values
    vol_arr = vol_change.values

    for i in range(1, len(close)):
        if vol_arr[i] > 0:
            pvi_values[i] = pvi_values[i - 1] * (1.0 + pct_arr[i])
        else:
            pvi_values[i] = pvi_values[i - 1]

    result = pd.Series(pvi_values, index=close.index, name="pvi")
    return result


# ---------------------------------------------------------------------------
# Volume Price Trend (VPT)
# ---------------------------------------------------------------------------


def vpt(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Volume Price Trend (VPT).

    ``VPT = cumsum(volume * pct_change(close))``

    Parameters
    ----------
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume data.

    Returns
    -------
    pd.Series
        VPT values.
    """
    _validate_series(close, "close")
    _validate_series(volume, "volume")

    pct = close.pct_change()
    result = (volume * pct).cumsum()
    result.name = "vpt"
    return result


# ---------------------------------------------------------------------------
# Accumulation/Distribution Oscillator (Chaikin Oscillator)
# ---------------------------------------------------------------------------


def adosc(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    fast: int = 3,
    slow: int = 10,
) -> pd.Series:
    """Accumulation/Distribution Oscillator (Chaikin Oscillator).

    ``ADOSC = EMA(AD, fast) - EMA(AD, slow)``

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume data.
    fast : int, default 3
        Fast EMA period.
    slow : int, default 10
        Slow EMA period.

    Returns
    -------
    pd.Series
        Chaikin Oscillator values.
    """
    ad = ad_line(high, low, close, volume)
    result = _ema(ad, fast) - _ema(ad, slow)
    result.name = "adosc"
    return result
