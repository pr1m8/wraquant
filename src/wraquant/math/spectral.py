"""Spectral analysis / FFT tools for financial data."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import signal as sp_signal

__all__ = [
    "fft_decompose",
    "dominant_frequencies",
    "spectral_density",
    "bandpass_filter",
    "spectral_entropy",
]


def fft_decompose(
    data: ArrayLike,
    n_components: int | None = None,
) -> dict[str, np.ndarray]:
    """Perform FFT decomposition of a 1-D signal.

    Parameters
    ----------
    data : array_like
        Input time series (real-valued, 1-D).
    n_components : int or None, optional
        If given, retain only the first *n_components* positive-frequency
        components (sorted by amplitude descending).  ``None`` keeps all.

    Returns
    -------
    dict
        ``frequencies`` – normalised frequencies (cycles / sample).
        ``amplitudes``  – amplitude of each frequency component.
        ``phases``      – phase angle (radians) of each component.
        ``power``       – power spectrum (amplitude squared).
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    fft_vals = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(n)
    amplitudes = np.abs(fft_vals) * 2.0 / n
    phases = np.angle(fft_vals)
    power = amplitudes**2

    if n_components is not None:
        idx = np.argsort(amplitudes)[::-1][:n_components]
        idx = np.sort(idx)  # keep frequency order
        freqs = freqs[idx]
        amplitudes = amplitudes[idx]
        phases = phases[idx]
        power = power[idx]

    return {
        "frequencies": freqs,
        "amplitudes": amplitudes,
        "phases": phases,
        "power": power,
    }


def dominant_frequencies(
    data: ArrayLike,
    n_top: int = 5,
) -> dict[str, np.ndarray]:
    """Identify the dominant cyclical frequencies in *data*.

    Parameters
    ----------
    data : array_like
        Input time series.
    n_top : int, optional
        Number of top frequencies to return (default 5).

    Returns
    -------
    dict
        ``frequency`` – normalised frequencies of the top components.
        ``period``    – corresponding period in bars (1 / frequency).
        ``amplitude`` – amplitude of each component.
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    fft_vals = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(n)
    amplitudes = np.abs(fft_vals) * 2.0 / n

    # Exclude DC component (index 0)
    if len(freqs) > 1:
        freqs = freqs[1:]
        amplitudes = amplitudes[1:]

    n_top = min(n_top, len(freqs))
    idx = np.argsort(amplitudes)[::-1][:n_top]

    top_freqs = freqs[idx]
    top_amps = amplitudes[idx]
    periods = np.where(top_freqs > 0, 1.0 / top_freqs, np.inf)

    return {
        "frequency": top_freqs,
        "period": periods,
        "amplitude": top_amps,
    }


def spectral_density(
    data: ArrayLike,
    method: str = "periodogram",
) -> dict[str, np.ndarray]:
    """Estimate the power spectral density (PSD) of *data*.

    Parameters
    ----------
    data : array_like
        Input time series.
    method : {'periodogram', 'welch'}, optional
        Estimation method (default ``'periodogram'``).

    Returns
    -------
    dict
        ``frequencies`` – frequency axis.
        ``psd``         – power spectral density values.

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    data = np.asarray(data, dtype=float)

    if method == "periodogram":
        freqs, psd = sp_signal.periodogram(data)
    elif method == "welch":
        freqs, psd = sp_signal.welch(data)
    else:
        raise ValueError(f"Unknown method {method!r}; use 'periodogram' or 'welch'.")

    return {"frequencies": freqs, "psd": psd}


def bandpass_filter(
    data: ArrayLike,
    low_freq: float,
    high_freq: float,
    sampling_rate: float = 1.0,
) -> np.ndarray:
    """Apply a bandpass filter to isolate a frequency band.

    Uses a 5th-order Butterworth filter applied forward-backward
    (zero phase shift) via :func:`scipy.signal.sosfiltfilt`.

    Parameters
    ----------
    data : array_like
        Input time series.
    low_freq : float
        Lower cut-off frequency (Hz or cycles/sample when
        ``sampling_rate=1.0``).
    high_freq : float
        Upper cut-off frequency.
    sampling_rate : float, optional
        Sampling rate of the data (default 1.0).

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    data = np.asarray(data, dtype=float)
    nyquist = sampling_rate / 2.0
    low = low_freq / nyquist
    high = high_freq / nyquist
    # Clip to valid Butterworth range (0, 1)
    low = max(low, 1e-10)
    high = min(high, 1.0 - 1e-10)
    sos = sp_signal.butter(5, [low, high], btype="band", output="sos")
    return sp_signal.sosfiltfilt(sos, data)


def spectral_entropy(data: ArrayLike) -> float:
    """Compute the spectral entropy of *data*.

    Spectral entropy measures the "flatness" of the power spectrum.
    A perfectly periodic signal has low spectral entropy; white noise
    has high spectral entropy (close to ``log2(N/2)``).

    Parameters
    ----------
    data : array_like
        Input time series.

    Returns
    -------
    float
        Normalised spectral entropy in [0, 1].
    """
    data = np.asarray(data, dtype=float)
    _, psd = sp_signal.periodogram(data)

    # Normalise PSD to a probability distribution
    psd = psd[psd > 0]
    if len(psd) == 0:
        return 0.0
    psd_norm = psd / psd.sum()

    # Shannon entropy normalised by log2(N)
    ent = -np.sum(psd_norm * np.log2(psd_norm))
    max_ent = np.log2(len(psd_norm))
    if max_ent == 0:
        return 0.0
    return float(ent / max_ent)
