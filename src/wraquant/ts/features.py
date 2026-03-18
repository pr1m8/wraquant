"""Time series feature extraction.

Provides functions to compute summary features from time series data
for use in classification, clustering, or exploratory analysis. Features
fall into three categories:

1. **Autocorrelation features**: ACF, PACF, Ljung-Box test for serial
   dependence structure.
2. **Spectral features**: power spectral density, dominant frequency,
   spectral entropy, spectral flatness.
3. **Complexity features**: sample entropy, approximate entropy,
   permutation entropy for measuring signal regularity/predictability.
"""

from __future__ import annotations

import warnings
from math import factorial

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Autocorrelation Features
# ---------------------------------------------------------------------------


def autocorrelation_features(
    data: pd.Series,
    n_lags: int = 40,
    significance: float = 0.05,
) -> dict:
    """Compute autocorrelation and partial autocorrelation features.

    Extracts the ACF and PACF values up to ``n_lags``, identifies
    statistically significant lags, and runs the Ljung-Box test for
    overall serial correlation.

    **Interpretation**:
        - **ACF** measures linear dependence between y_t and y_{t-k}.
          For AR(p) processes, ACF decays exponentially. For MA(q),
          ACF cuts off after lag q.
        - **PACF** measures the correlation between y_t and y_{t-k}
          after removing the effect of intermediate lags. For AR(p),
          PACF cuts off after lag p.
        - **Ljung-Box**: tests H0 that the first m autocorrelations
          are jointly zero. A significant result indicates the series
          has exploitable temporal structure.

    Parameters:
        data: Time series. NaN values are dropped.
        n_lags: Number of lags to compute (default 40).
        significance: Significance level for identifying significant
            lags and Ljung-Box test (default 0.05).

    Returns:
        Dictionary with:
        - ``acf``: 1-D numpy array of ACF values (lags 0 to n_lags).
        - ``pacf``: 1-D numpy array of PACF values (lags 0 to n_lags).
        - ``significant_acf_lags``: list of lag indices where ACF
          exceeds the Bartlett confidence band.
        - ``significant_pacf_lags``: list of lag indices where PACF
          exceeds the confidence band.
        - ``ljung_box``: dict with ``statistic``, ``p_value``, and
          ``is_significant`` for the Ljung-Box test at the maximum lag.
        - ``first_significant_lag``: int or None, the first lag with
          significant ACF (excluding lag 0).

    Example:
        >>> import numpy as np, pandas as pd
        >>> rng = np.random.default_rng(42)
        >>> # AR(1) process
        >>> x = np.zeros(500)
        >>> for i in range(1, 500):
        ...     x[i] = 0.7 * x[i-1] + rng.normal()
        >>> result = autocorrelation_features(pd.Series(x), n_lags=20)
        >>> result['acf'][1] > 0.5  # strong lag-1 autocorrelation
        True
    """
    from statsmodels.tsa.stattools import acf as sm_acf
    from statsmodels.tsa.stattools import pacf as sm_pacf

    clean = data.dropna().values
    n = len(clean)
    n_lags = min(n_lags, n // 2 - 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acf_vals = sm_acf(clean, nlags=n_lags, fft=True)
        try:
            pacf_vals = sm_pacf(clean, nlags=n_lags, method="ywm")
        except Exception:  # noqa: BLE001
            pacf_vals = sm_pacf(clean, nlags=min(n_lags, n // 3), method="ywm")
            # Pad with zeros if needed
            if len(pacf_vals) < n_lags + 1:
                pacf_vals = np.concatenate(
                    [pacf_vals, np.zeros(n_lags + 1 - len(pacf_vals))]
                )

    # Bartlett confidence band: +/- z / sqrt(n)
    z = sp_stats.norm.ppf(1 - significance / 2)
    conf_band = z / np.sqrt(n)

    significant_acf = [
        int(i) for i in range(1, len(acf_vals)) if abs(acf_vals[i]) > conf_band
    ]
    significant_pacf = [
        int(i) for i in range(1, len(pacf_vals)) if abs(pacf_vals[i]) > conf_band
    ]

    # Ljung-Box test
    from statsmodels.stats.diagnostic import acorr_ljungbox

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lb = acorr_ljungbox(clean, lags=[n_lags], return_df=True)

    lb_stat = float(lb["lb_stat"].iloc[0])
    lb_pval = float(lb["lb_pvalue"].iloc[0])

    return {
        "acf": acf_vals,
        "pacf": pacf_vals,
        "significant_acf_lags": significant_acf,
        "significant_pacf_lags": significant_pacf,
        "ljung_box": {
            "statistic": lb_stat,
            "p_value": lb_pval,
            "is_significant": lb_pval < significance,
        },
        "first_significant_lag": significant_acf[0] if significant_acf else None,
    }


# ---------------------------------------------------------------------------
# Spectral Features
# ---------------------------------------------------------------------------


def spectral_features(
    data: pd.Series,
    fs: float = 1.0,
) -> dict:
    """Compute power spectral density features.

    Extracts summary statistics from the periodogram including the
    dominant frequency, spectral entropy (a measure of spectral
    complexity), and spectral flatness (how noise-like the signal is).

    Parameters:
        data: Time series. NaN values are dropped.
        fs: Sampling frequency (default 1.0). For daily data with
            annual cycles, use ``fs=1.0`` and interpret frequencies
            as cycles per observation.

    Returns:
        Dictionary with:
        - ``dominant_frequency``: float, the frequency with the highest
          spectral power (excluding DC component).
        - ``dominant_period``: float, the period corresponding to the
          dominant frequency (``1 / dominant_frequency``).
        - ``spectral_entropy``: float, Shannon entropy of the normalised
          power spectral density. Higher values indicate a more complex,
          broadband signal; lower values indicate a few dominant
          frequencies. Range: [0, log2(N/2)].
        - ``spectral_flatness``: float in [0, 1]. Ratio of geometric
          mean to arithmetic mean of the PSD. A value near 1 indicates
          white noise; near 0 indicates a tonal/periodic signal.
          Also known as Wiener entropy.
        - ``spectral_centroid``: float, the weighted mean frequency
          (centre of mass of the spectrum).
        - ``psd``: 1-D numpy array of power spectral density values.
        - ``frequencies``: 1-D numpy array of corresponding frequencies.

    Example:
        >>> import numpy as np, pandas as pd
        >>> t = np.arange(500, dtype=float)
        >>> pure_tone = pd.Series(np.sin(2 * np.pi * t / 20))  # period=20
        >>> result = spectral_features(pure_tone)
        >>> abs(result['dominant_period'] - 20) < 2
        True
        >>> result['spectral_flatness'] < 0.3  # tonal signal
        True
    """
    from scipy.signal import periodogram

    clean = data.dropna().values.astype(np.float64)

    freqs, psd = periodogram(clean, fs=fs)

    # Exclude DC component
    if len(freqs) > 1:
        freqs_no_dc = freqs[1:]
        psd_no_dc = psd[1:]
    else:
        freqs_no_dc = freqs
        psd_no_dc = psd

    if len(psd_no_dc) == 0 or np.sum(psd_no_dc) < 1e-15:
        return {
            "dominant_frequency": 0.0,
            "dominant_period": float("inf"),
            "spectral_entropy": 0.0,
            "spectral_flatness": 0.0,
            "spectral_centroid": 0.0,
            "psd": psd,
            "frequencies": freqs,
        }

    # Dominant frequency
    peak_idx = int(np.argmax(psd_no_dc))
    dominant_freq = float(freqs_no_dc[peak_idx])
    dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else float("inf")

    # Spectral entropy
    psd_norm = psd_no_dc / np.sum(psd_no_dc)
    psd_norm = psd_norm[psd_norm > 0]  # avoid log(0)
    spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm)))

    # Spectral flatness (Wiener entropy)
    log_psd = np.log(psd_no_dc[psd_no_dc > 0])
    geo_mean = np.exp(np.mean(log_psd))
    arith_mean = np.mean(psd_no_dc[psd_no_dc > 0])
    spectral_flatness = float(geo_mean / arith_mean) if arith_mean > 0 else 0.0

    # Spectral centroid
    spectral_centroid = float(
        np.sum(freqs_no_dc * psd_no_dc) / np.sum(psd_no_dc)
    )

    return {
        "dominant_frequency": dominant_freq,
        "dominant_period": dominant_period,
        "spectral_entropy": spectral_entropy,
        "spectral_flatness": spectral_flatness,
        "spectral_centroid": spectral_centroid,
        "psd": psd,
        "frequencies": freqs,
    }


# ---------------------------------------------------------------------------
# Complexity Features
# ---------------------------------------------------------------------------


def complexity_features(
    data: pd.Series,
    m: int = 2,
    r: float | None = None,
    order: int = 3,
) -> dict:
    """Compute entropy-based complexity measures for a time series.

    Extracts three entropy measures that quantify the regularity and
    predictability of the signal:

    1. **Sample entropy (SampEn)**: probability that patterns similar
       for m points remain similar for m+1 points. Lower values
       indicate more regularity (self-similarity); higher values
       indicate more complexity. Defined as -ln(A/B) where A is the
       count of m+1 length template matches and B is the count of m
       length matches, within tolerance r.

    2. **Approximate entropy (ApEn)**: similar to SampEn but includes
       self-matches, making it slightly biased toward regularity.
       Historically introduced first (Pincus 1991).

    3. **Permutation entropy (PeEn)**: based on the frequency of
       ordinal patterns of length ``order``. Robust to noise and
       monotonic transformations. Normalised to [0, 1].

    Parameters:
        data: Time series. NaN values are dropped.
        m: Embedding dimension for SampEn and ApEn (default 2).
            Typical values: 2 or 3.
        r: Tolerance threshold for SampEn and ApEn. If ``None``,
            defaults to ``0.2 * std(data)`` (standard choice).
        order: Ordinal pattern length for permutation entropy
            (default 3). Typical values: 3-7.

    Returns:
        Dictionary with:
        - ``sample_entropy``: float, SampEn(m, r). Values typically
          range from 0 (perfectly regular) to ~2.5 (random).
        - ``approximate_entropy``: float, ApEn(m, r).
        - ``permutation_entropy``: float, normalised PeEn in [0, 1].
          0 = perfectly predictable, 1 = maximally complex/random.

    Example:
        >>> import numpy as np, pandas as pd
        >>> rng = np.random.default_rng(42)
        >>> regular = pd.Series(np.sin(np.arange(500) * 0.1))
        >>> random = pd.Series(rng.normal(0, 1, 500))
        >>> reg_feat = complexity_features(regular)
        >>> rnd_feat = complexity_features(random)
        >>> reg_feat['sample_entropy'] < rnd_feat['sample_entropy']
        True

    References:
        - Richman, J.S. & Moorman, J.R. (2000), "Physiological
          time-series analysis using approximate entropy and sample
          entropy", AJP.
        - Pincus, S.M. (1991), "Approximate entropy as a measure
          of system complexity", PNAS.
        - Bandt, C. & Pompe, B. (2002), "Permutation Entropy: A
          Natural Complexity Measure for Time Series", PRL.
    """
    clean = data.dropna().values.astype(np.float64)
    n = len(clean)

    if r is None:
        r = 0.2 * np.std(clean)
        if r < 1e-10:
            r = 0.2

    # --- Sample Entropy ---
    sample_ent = _sample_entropy(clean, m, r)

    # --- Approximate Entropy ---
    approx_ent = _approximate_entropy(clean, m, r)

    # --- Permutation Entropy ---
    perm_ent = _permutation_entropy(clean, order)

    return {
        "sample_entropy": sample_ent,
        "approximate_entropy": approx_ent,
        "permutation_entropy": perm_ent,
    }


def _sample_entropy(x: np.ndarray, m: int, r: float) -> float:
    """Compute sample entropy SampEn(m, r) for a 1-D array."""
    n = len(x)

    def _count_matches(template_len: int) -> int:
        count = 0
        for i in range(n - template_len):
            for j in range(i + 1, n - template_len):
                if np.max(np.abs(x[i : i + template_len] - x[j : j + template_len])) <= r:
                    count += 1
        return count

    b = _count_matches(m)
    a = _count_matches(m + 1)

    if b == 0:
        return float("inf")
    if a == 0:
        return float("inf")

    return float(-np.log(a / b))


def _approximate_entropy(x: np.ndarray, m: int, r: float) -> float:
    """Compute approximate entropy ApEn(m, r) for a 1-D array."""
    n = len(x)

    def _phi(template_len: int) -> float:
        templates = np.array([x[i : i + template_len] for i in range(n - template_len + 1)])
        counts = np.zeros(len(templates))
        for i, t in enumerate(templates):
            for j, s in enumerate(templates):
                if np.max(np.abs(t - s)) <= r:
                    counts[i] += 1
        counts /= len(templates)
        return float(np.mean(np.log(counts)))

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)

    return float(phi_m - phi_m1)


def _permutation_entropy(x: np.ndarray, order: int) -> float:
    """Compute normalised permutation entropy for a 1-D array."""
    n = len(x)
    if n < order:
        return 0.0

    # Count ordinal patterns
    pattern_counts: dict[tuple[int, ...], int] = {}
    for i in range(n - order + 1):
        pattern = tuple(int(j) for j in np.argsort(x[i : i + order]))
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    total = sum(pattern_counts.values())
    probs = np.array([c / total for c in pattern_counts.values()])

    # Shannon entropy normalised by max possible entropy
    h = float(-np.sum(probs * np.log2(probs)))
    max_h = np.log2(factorial(order))

    if max_h < 1e-15:
        return 0.0

    return float(h / max_h)
