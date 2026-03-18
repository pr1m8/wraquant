"""Cycle analysis indicators.

This module provides indicators based on digital signal processing
techniques, primarily those developed by John Ehlers. They detect
dominant cycle periods and separate trend from cycle components.
All functions accept ``pd.Series`` inputs and return ``pd.Series``
(or ``dict[str, pd.Series]`` for multi-output indicators).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "hilbert_transform_dominant_period",
    "hilbert_transform_trend_mode",
    "hilbert_instantaneous_phase",
    "sine_wave",
    "even_better_sinewave",
    "roofing_filter",
    "decycler",
    "bandpass_filter",
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


def _highpass_filter(data: np.ndarray, period: int) -> np.ndarray:
    """Two-pole high-pass filter (Ehlers).

    Removes components with period longer than *period* bars.
    """
    n = len(data)
    alpha = (
        (np.cos(2 * np.pi / period) + np.sin(2 * np.pi / period) - 1)
        / np.cos(2 * np.pi / period)
    )
    hp = np.zeros(n)
    for i in range(2, n):
        hp[i] = (
            (1 - alpha / 2) * (1 - alpha / 2) * (data[i] - 2 * data[i - 1] + data[i - 2])
            + 2 * (1 - alpha) * hp[i - 1]
            - (1 - alpha) * (1 - alpha) * hp[i - 2]
        )
    return hp


def _supersmoother(data: np.ndarray, period: int) -> np.ndarray:
    """Ehlers two-pole super-smoother filter."""
    n = len(data)
    a1 = np.exp(-np.sqrt(2) * np.pi / period)
    b1 = 2 * a1 * np.cos(np.sqrt(2) * np.pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    ss = np.zeros(n)
    for i in range(2, n):
        ss[i] = c1 * (data[i] + data[i - 1]) / 2 + c2 * ss[i - 1] + c3 * ss[i - 2]
    return ss


# ---------------------------------------------------------------------------
# Hilbert Transform — Dominant Period
# ---------------------------------------------------------------------------


def hilbert_transform_dominant_period(
    data: pd.Series,
    min_period: int = 6,
    max_period: int = 50,
) -> pd.Series:
    """Dominant cycle period via Hilbert Transform.

    Uses Ehlers' Hilbert Transform Discriminator to estimate the
    dominant cycle period of the price series.

    Parameters
    ----------
    data : pd.Series
        Price series (typically close).
    min_period : int, default 6
        Minimum allowed cycle period.
    max_period : int, default 50
        Maximum allowed cycle period.

    Returns
    -------
    pd.Series
        Estimated dominant cycle period in bars.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> close = pd.Series(np.sin(np.linspace(0, 8 * np.pi, 200)) * 10 + 100)
    >>> hilbert_transform_dominant_period(close)  # doctest: +SKIP
    """
    _validate_series(data)
    values = data.values.astype(float)
    n = len(values)

    # Smooth with a simple Ehlers smoother
    smooth = np.zeros(n)
    for i in range(3, n):
        smooth[i] = (
            4 * values[i]
            + 3 * values[i - 1]
            + 2 * values[i - 2]
            + values[i - 3]
        ) / 10.0

    period_out = np.full(n, np.nan)
    detrender = np.zeros(n)
    q1 = np.zeros(n)
    i1 = np.zeros(n)
    ji = np.zeros(n)
    jq = np.zeros(n)
    i2 = np.zeros(n)
    q2 = np.zeros(n)
    re_ = np.zeros(n)
    im_ = np.zeros(n)
    smooth_period = np.zeros(n)

    for i in range(6, n):
        adj = 0.075 * smooth_period[i - 1] + 0.54
        # Detrend
        detrender[i] = (
            0.0962 * smooth[i]
            + 0.5769 * smooth[i - 2]
            - 0.5769 * smooth[i - 4]
            - 0.0962 * smooth[i - 6]
        ) * adj

        # Compute InPhase and Quadrature
        q1[i] = (
            0.0962 * detrender[i]
            + 0.5769 * detrender[i - 2]
            - 0.5769 * detrender[i - 4]
            - 0.0962 * detrender[i - 6]
        ) * adj
        i1[i] = detrender[i - 3]

        # Advance phase by 90 degrees
        ji[i] = (
            0.0962 * i1[i]
            + 0.5769 * i1[i - 2]
            - 0.5769 * i1[i - 4]
            - 0.0962 * i1[i - 6]
        ) * adj
        jq[i] = (
            0.0962 * q1[i]
            + 0.5769 * q1[i - 2]
            - 0.5769 * q1[i - 4]
            - 0.0962 * q1[i - 6]
        ) * adj

        i2[i] = i1[i] - jq[i]
        q2[i] = q1[i] + ji[i]

        # Smooth
        i2[i] = 0.2 * i2[i] + 0.8 * i2[i - 1]
        q2[i] = 0.2 * q2[i] + 0.8 * q2[i - 1]

        re_[i] = i2[i] * i2[i - 1] + q2[i] * q2[i - 1]
        im_[i] = i2[i] * q2[i - 1] - q2[i] * i2[i - 1]
        re_[i] = 0.2 * re_[i] + 0.8 * re_[i - 1]
        im_[i] = 0.2 * im_[i] + 0.8 * im_[i - 1]

        if im_[i] != 0 and re_[i] != 0:
            raw = 2 * np.pi / np.arctan(im_[i] / re_[i])
        else:
            raw = smooth_period[i - 1]

        # Clamp
        raw = np.clip(raw, min_period, max_period)

        smooth_period[i] = 0.2 * raw + 0.8 * smooth_period[i - 1]
        period_out[i] = smooth_period[i]

    result = pd.Series(period_out, index=data.index, name="dominant_period")
    return result


# ---------------------------------------------------------------------------
# Hilbert Transform — Trend Mode
# ---------------------------------------------------------------------------


def hilbert_transform_trend_mode(data: pd.Series) -> pd.Series:
    """Trend vs cycle mode indicator via Hilbert Transform.

    Returns +1 when the market is in trend mode and 0 when in cycle
    mode, based on the relationship between the dominant cycle period
    and a simple moving average smoothing window.

    Parameters
    ----------
    data : pd.Series
        Price series.

    Returns
    -------
    pd.Series
        Binary series: 1 = trending, 0 = cycling.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> close = pd.Series(np.sin(np.linspace(0, 8 * np.pi, 200)) * 10 + 100)
    >>> hilbert_transform_trend_mode(close)  # doctest: +SKIP
    """
    _validate_series(data)
    dc = hilbert_transform_dominant_period(data)
    values = data.values.astype(float)
    n = len(values)

    trend = np.full(n, np.nan)
    for i in range(1, n):
        if np.isnan(dc.iloc[i]):
            continue
        period = max(int(dc.iloc[i]), 1)
        start = max(0, i - period + 1)
        sma_val = np.mean(values[start : i + 1])
        # Trend mode when price deviates significantly from the cycle SMA
        smoothed_range = np.std(values[start : i + 1])
        if smoothed_range > 0:
            deviation = abs(values[i] - sma_val) / smoothed_range
            trend[i] = 1.0 if deviation > 0.5 else 0.0
        else:
            trend[i] = 0.0

    result = pd.Series(trend, index=data.index, name="trend_mode")
    return result


# ---------------------------------------------------------------------------
# Hilbert Instantaneous Phase (Trendline)
# ---------------------------------------------------------------------------


def hilbert_instantaneous_phase(data: pd.Series) -> pd.Series:
    """Instantaneous trendline via Hilbert Transform.

    Computes a smooth trendline by applying the dominant cycle period
    as an adaptive moving average length.

    Parameters
    ----------
    data : pd.Series
        Price series.

    Returns
    -------
    pd.Series
        Instantaneous trendline values.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> close = pd.Series(np.sin(np.linspace(0, 8 * np.pi, 200)) * 10 + 100)
    >>> hilbert_instantaneous_phase(close)  # doctest: +SKIP
    """
    _validate_series(data)
    dc = hilbert_transform_dominant_period(data)
    values = data.values.astype(float)
    n = len(values)

    trendline = np.full(n, np.nan)
    for i in range(1, n):
        if np.isnan(dc.iloc[i]):
            continue
        period = max(int(dc.iloc[i]), 1)
        start = max(0, i - period + 1)
        trendline[i] = np.mean(values[start : i + 1])

    result = pd.Series(trendline, index=data.index, name="instantaneous_trendline")
    return result


# ---------------------------------------------------------------------------
# Sine Wave
# ---------------------------------------------------------------------------


def sine_wave(data: pd.Series) -> dict[str, pd.Series]:
    """Ehlers Sine Wave indicator.

    Uses the dominant cycle period to compute the sine and lead-sine
    values, generating buy/sell signals on crossovers.

    Parameters
    ----------
    data : pd.Series
        Price series.

    Returns
    -------
    dict[str, pd.Series]
        ``sine`` and ``lead_sine`` series.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> close = pd.Series(np.sin(np.linspace(0, 8 * np.pi, 200)) * 10 + 100)
    >>> result = sine_wave(close)  # doctest: +SKIP
    """
    _validate_series(data)
    dc = hilbert_transform_dominant_period(data)
    n = len(data)

    sine_out = np.full(n, np.nan)
    lead_out = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(dc.iloc[i]):
            continue
        period = dc.iloc[i]
        if period > 0:
            phase = 2 * np.pi / period
            sine_out[i] = np.sin(phase * i)
            lead_out[i] = np.sin(phase * i + np.pi / 4)

    return {
        "sine": pd.Series(sine_out, index=data.index, name="sine"),
        "lead_sine": pd.Series(lead_out, index=data.index, name="lead_sine"),
    }


# ---------------------------------------------------------------------------
# Even Better Sinewave (EBSW)
# ---------------------------------------------------------------------------


def even_better_sinewave(
    data: pd.Series,
    hp_period: int = 40,
    ss_period: int = 10,
) -> pd.Series:
    """Ehlers Even Better Sinewave (EBSW).

    Combines a high-pass filter, super-smoother, and autocorrelation
    to produce an oscillator that identifies the dominant cycle.

    Parameters
    ----------
    data : pd.Series
        Price series.
    hp_period : int, default 40
        High-pass filter period.
    ss_period : int, default 10
        Super-smoother period.

    Returns
    -------
    pd.Series
        EBSW oscillator values in approximately [-1, 1].

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> close = pd.Series(np.sin(np.linspace(0, 8 * np.pi, 200)) * 10 + 100)
    >>> even_better_sinewave(close)  # doctest: +SKIP
    """
    _validate_series(data)
    values = data.values.astype(float)
    n = len(values)

    # High-pass filter
    hp = _highpass_filter(values, hp_period)

    # Super-smoother
    filt = _supersmoother(hp, ss_period)

    # Wave computation
    wave = np.full(n, np.nan)
    for i in range(1, n):
        rms = 0.0
        count = min(i + 1, ss_period)
        for j in range(count):
            rms += filt[i - j] ** 2
        rms = np.sqrt(rms / count) if count > 0 else 0.0

        if rms > 0:
            wave[i] = filt[i] / rms
        else:
            wave[i] = 0.0

    # Clamp to [-1, 1]
    wave = np.clip(wave, -1.0, 1.0)

    result = pd.Series(wave, index=data.index, name="ebsw")
    return result


# ---------------------------------------------------------------------------
# Roofing Filter
# ---------------------------------------------------------------------------


def roofing_filter(
    data: pd.Series,
    hp_period: int = 48,
    lp_period: int = 10,
) -> pd.Series:
    """Ehlers Roofing Filter.

    Applies a high-pass filter followed by a super-smoother low-pass
    filter to isolate the dominant cycle from both trend and noise.

    Parameters
    ----------
    data : pd.Series
        Price series.
    hp_period : int, default 48
        High-pass filter cutoff period.
    lp_period : int, default 10
        Low-pass (super-smoother) cutoff period.

    Returns
    -------
    pd.Series
        Filtered cycle component.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> close = pd.Series(np.sin(np.linspace(0, 8 * np.pi, 200)) * 10 + 100)
    >>> roofing_filter(close)  # doctest: +SKIP
    """
    _validate_series(data)
    values = data.values.astype(float)

    hp = _highpass_filter(values, hp_period)
    ss = _supersmoother(hp, lp_period)

    result = pd.Series(ss, index=data.index, name="roofing_filter")
    return result


# ---------------------------------------------------------------------------
# Decycler
# ---------------------------------------------------------------------------


def decycler(data: pd.Series, hp_period: int = 125) -> pd.Series:
    """Ehlers Decycler.

    Removes the cycle component from the price series, keeping only
    the trend. Computed as ``price - highpass(price)``.

    Parameters
    ----------
    data : pd.Series
        Price series.
    hp_period : int, default 125
        High-pass filter cutoff period. Components with period shorter
        than this are removed (cycles).

    Returns
    -------
    pd.Series
        Trend-only (decycled) series.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> close = pd.Series(np.sin(np.linspace(0, 8 * np.pi, 200)) * 10 + 100)
    >>> decycler(close)  # doctest: +SKIP
    """
    _validate_series(data)
    values = data.values.astype(float)

    hp = _highpass_filter(values, hp_period)
    trend = values - hp

    result = pd.Series(trend, index=data.index, name="decycler")
    return result


# ---------------------------------------------------------------------------
# Bandpass Filter
# ---------------------------------------------------------------------------


def bandpass_filter(
    data: pd.Series,
    period: int = 20,
    bandwidth: float = 0.3,
) -> dict[str, pd.Series]:
    """Ehlers Bandpass Filter.

    Isolates the cycle component at the specified period. Returns both
    the bandpass filter output and a trigger signal (one-bar lag).

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 20
        Centre period of the bandpass.
    bandwidth : float, default 0.3
        Bandwidth as a fraction of the centre frequency.

    Returns
    -------
    dict[str, pd.Series]
        ``bp`` (bandpass) and ``trigger`` (one-bar lag of bp).

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> close = pd.Series(np.sin(np.linspace(0, 8 * np.pi, 200)) * 10 + 100)
    >>> result = bandpass_filter(close, period=20)  # doctest: +SKIP
    """
    _validate_series(data)
    _validate_period(period)
    values = data.values.astype(float)
    n = len(values)

    beta_val = np.cos(2 * np.pi / period)
    gamma_val = 1 / np.cos(2 * np.pi * bandwidth / period)
    alpha_val = gamma_val - np.sqrt(gamma_val * gamma_val - 1)

    bp = np.zeros(n)
    for i in range(2, n):
        bp[i] = (
            0.5 * (1 - alpha_val) * (values[i] - values[i - 2])
            + beta_val * (1 + alpha_val) * bp[i - 1]
            - alpha_val * bp[i - 2]
        )

    trigger = np.zeros(n)
    trigger[1:] = bp[:-1]

    return {
        "bp": pd.Series(bp, index=data.index, name="bandpass"),
        "trigger": pd.Series(trigger, index=data.index, name="bandpass_trigger"),
    }
