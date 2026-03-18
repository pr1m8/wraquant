"""Signal processing filters for financial time series."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import median_filter as _scipy_median_filter
from scipy.signal import savgol_filter

__all__ = [
    "savitzky_golay",
    "kalman_smooth",
    "wavelet_denoise",
    "median_filter",
    "exponential_smooth",
    "hodrick_prescott",
]


def savitzky_golay(
    data: ArrayLike,
    window: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """Apply a Savitzky-Golay smoothing filter.

    Parameters
    ----------
    data : array_like
        Input time series.
    window : int, optional
        Window length (must be odd and > *polyorder*; default 11).
    polyorder : int, optional
        Polynomial order for the local fit (default 3).

    Returns
    -------
    np.ndarray
        Smoothed signal.
    """
    data = np.asarray(data, dtype=float)
    # Ensure window is odd
    if window % 2 == 0:
        window += 1
    return savgol_filter(data, window_length=window, polyorder=polyorder)


def kalman_smooth(
    data: ArrayLike,
    process_var: float = 1e-5,
    measurement_var: float = 1e-1,
) -> np.ndarray:
    """Simple 1-D Kalman smoother (forward pass).

    Implements a constant-level model with Gaussian noise.

    Parameters
    ----------
    data : array_like
        Observed (noisy) time series.
    process_var : float, optional
        Process noise variance (default 1e-5).
    measurement_var : float, optional
        Measurement noise variance (default 1e-1).

    Returns
    -------
    np.ndarray
        Kalman-smoothed estimate (same length as *data*).
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    smoothed = np.empty(n, dtype=float)

    # Initialise
    x_est = data[0]
    p_est = 1.0

    for i in range(n):
        # Predict
        x_pred = x_est
        p_pred = p_est + process_var

        # Update
        k = p_pred / (p_pred + measurement_var)
        x_est = x_pred + k * (data[i] - x_pred)
        p_est = (1.0 - k) * p_pred

        smoothed[i] = x_est

    return smoothed


def wavelet_denoise(
    data: ArrayLike,
    wavelet: str = "db4",
    level: int | None = None,
    threshold: str = "soft",
) -> np.ndarray:
    """Wavelet denoising of a time series.

    Requires the optional ``pywavelets`` package (install via
    ``pip install wraquant[timeseries]``).

    Parameters
    ----------
    data : array_like
        Input time series.
    wavelet : str, optional
        Wavelet name (default ``'db4'``).
    level : int or None, optional
        Decomposition level.  ``None`` uses the maximum useful level.
    threshold : {'soft', 'hard'}, optional
        Thresholding mode (default ``'soft'``).

    Returns
    -------
    np.ndarray
        Denoised signal.

    Raises
    ------
    ImportError
        If ``pywt`` is not installed.
    """
    try:
        import pywt  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "wavelet_denoise requires PyWavelets.  "
            "Install it with:  pip install wraquant[timeseries]"
        ) from exc

    data = np.asarray(data, dtype=float)

    if level is None:
        level = pywt.dwt_max_level(len(data), pywt.Wavelet(wavelet).dec_len)

    coeffs = pywt.wavedec(data, wavelet, level=level)

    # Universal threshold (VisuShrink)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2.0 * np.log(len(data)))

    denoised_coeffs = [coeffs[0]]  # keep approximation coefficients
    for c in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(c, value=uthresh, mode=threshold))

    return pywt.waverec(denoised_coeffs, wavelet)[: len(data)]


def median_filter(
    data: ArrayLike,
    kernel_size: int = 5,
) -> np.ndarray:
    """Median filter for spike removal.

    Parameters
    ----------
    data : array_like
        Input time series.
    kernel_size : int, optional
        Size of the median-filter kernel (default 5).

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    data = np.asarray(data, dtype=float)
    return _scipy_median_filter(data, size=kernel_size).astype(float)


def exponential_smooth(
    data: ArrayLike,
    alpha: float = 0.3,
) -> np.ndarray:
    r"""Simple exponential smoothing.

    .. math::

        s_t = \\alpha \\, x_t + (1 - \\alpha) \\, s_{t-1}

    Parameters
    ----------
    data : array_like
        Input time series.
    alpha : float, optional
        Smoothing factor in ``(0, 1]`` (default 0.3).

    Returns
    -------
    np.ndarray
        Smoothed signal.
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    smoothed = np.empty(n, dtype=float)
    smoothed[0] = data[0]
    for i in range(1, n):
        smoothed[i] = alpha * data[i] + (1.0 - alpha) * smoothed[i - 1]
    return smoothed


def hodrick_prescott(
    data: ArrayLike,
    lamb: float = 1600.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Hodrick-Prescott filter decomposition.

    Decomposes *data* into a trend and a cyclical component.

    Parameters
    ----------
    data : array_like
        Input time series.
    lamb : float, optional
        Smoothing parameter (default 1600, standard for quarterly data).
        Use 6.25 for annual data, 129600 for monthly data.

    Returns
    -------
    tuple of np.ndarray
        ``(trend, cycle)`` where ``trend + cycle == data``.
    """
    data = np.asarray(data, dtype=float)
    n = len(data)

    # Construct the second-difference matrix K (n-2 x n)
    # and solve (I + lamb * K'K) * trend = data
    # Using sparse tridiagonal solver for efficiency
    # The normal equations give a pentadiagonal system.
    # We build it directly.

    # Diagonal bands of (I + lamb * K'K)
    # K'K is a banded matrix; we solve via dense linear algebra for
    # moderate n, which is typical for financial time series.
    diag_main = np.ones(n)
    diag_main[0] += lamb
    diag_main[1] += 5.0 * lamb
    diag_main[n - 2] += 5.0 * lamb
    diag_main[n - 1] += lamb
    for i in range(2, n - 2):
        diag_main[i] += 6.0 * lamb

    diag_1 = np.zeros(n - 1)
    diag_1[0] = -2.0 * lamb
    diag_1[n - 2] = -2.0 * lamb
    for i in range(1, n - 2):
        diag_1[i] = -4.0 * lamb

    diag_2 = np.full(n - 2, lamb)

    # Build the pentadiagonal matrix
    from scipy.linalg import solve_banded

    # Banded storage for solve_banded:
    # ab[0, :] = second super-diagonal
    # ab[1, :] = first super-diagonal
    # ab[2, :] = main diagonal
    # ab[3, :] = first sub-diagonal
    # ab[4, :] = second sub-diagonal
    ab = np.zeros((5, n))
    ab[2, :] = diag_main
    # First super/sub diagonals
    ab[1, 1:] = diag_1
    ab[3, :-1] = diag_1
    # Second super/sub diagonals
    ab[0, 2:] = diag_2
    ab[4, :-2] = diag_2

    trend = solve_banded((2, 2), ab, data)
    cycle = data - trend

    return trend, cycle
