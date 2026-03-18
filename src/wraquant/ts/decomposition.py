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

import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import seasonal_decompose as sm_seasonal_decompose

from wraquant.core.decorators import requires_extra


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


# ---------------------------------------------------------------------------
# Singular Spectrum Analysis
# ---------------------------------------------------------------------------


def ssa_decompose(
    data: pd.Series,
    window: int | None = None,
    n_components: int | None = None,
    groups: dict[str, list[int]] | None = None,
) -> dict:
    """Singular Spectrum Analysis (SSA) decomposition.

    SSA embeds a time series into a trajectory (Hankel) matrix, applies SVD
    to extract orthogonal components, and reconstructs interpretable
    signal components (trend, oscillatory modes, noise).

    **SSA vs STL**:
        - **STL** assumes a fixed seasonal period that you specify up front.
          It decomposes into exactly three components: trend, seasonal,
          residual. Works best when the periodicity is known and stable.
        - **SSA** is data-adaptive and makes no assumption about periodicity.
          It discovers the dominant oscillatory modes directly from the data
          via singular values. Better for non-stationary signals, signals
          with multiple or changing periodicities, or when you want to
          separate trend from multiple oscillatory components.

    Algorithm:
        1. **Embedding**: construct the L x K trajectory matrix (Hankel)
           where L = window length, K = N - L + 1.
        2. **SVD**: decompose the trajectory matrix into singular triplets.
        3. **Grouping**: assign singular triplets to interpretable groups
           (trend, oscillatory, noise) either automatically or via user
           specification.
        4. **Reconstruction**: diagonal averaging (Hankelisation) of each
           group's partial matrix to recover the time-domain component.

    Parameters:
        data: Time series to decompose. NaN values are dropped.
        window: Embedding window length L. Should be large enough to
            capture the longest period of interest but not exceed N/2.
            If ``None``, defaults to ``N // 2``.
        n_components: Number of leading singular triplets to retain.
            If ``None``, all are retained.
        groups: Optional grouping of singular triplet indices into named
            components. For example ``{"trend": [0], "seasonal": [1, 2]}``.
            Each key maps to a list of 0-based component indices.
            If ``None``, each singular triplet is returned as its own
            component (``"component_0"``, ``"component_1"``, ...).

    Returns:
        Dictionary with:
        - ``components``: dict mapping component name to reconstructed
          ``pd.Series``.
        - ``singular_values``: 1-D numpy array of singular values.
        - ``explained_variance``: 1-D numpy array of explained variance
          ratio per singular value (sums to 1.0 over all values).
        - ``window``: embedding window length used.

    Example:
        >>> import numpy as np, pandas as pd
        >>> t = np.arange(200, dtype=float)
        >>> trend = 0.05 * t
        >>> osc = 3 * np.sin(2 * np.pi * t / 20)
        >>> noise = np.random.default_rng(42).normal(0, 0.3, 200)
        >>> data = pd.Series(trend + osc + noise)
        >>> result = ssa_decompose(data, window=40, n_components=5)
        >>> # Components sum back to approximately the original
        >>> recon = sum(result['components'].values())
        >>> np.allclose(recon.values, data.values, atol=1e-8)
        True

    References:
        - Golyandina, N. & Zhigljavsky, A. (2013), *Singular Spectrum
          Analysis for Time Series*. Springer.
        - Vautard, R. & Ghil, M. (1989), "Singular spectrum analysis in
          nonlinear dynamics, with applications to paleoclimatic time
          series", Physica D.
    """
    clean = data.dropna()
    x = clean.values.astype(np.float64)
    n = len(x)

    if window is None:
        window = n // 2
    if window < 2:
        window = 2
    if window > n // 2:
        window = n // 2

    k = n - window + 1

    # Step 1: Build trajectory (Hankel) matrix
    traj = np.empty((window, k), dtype=np.float64)
    for i in range(window):
        traj[i, :] = x[i : i + k]

    # Step 2: SVD
    u, s, vt = np.linalg.svd(traj, full_matrices=False)

    # Explained variance
    s_sq = s ** 2
    explained_variance = s_sq / s_sq.sum()

    # Determine how many components to keep
    if n_components is not None:
        n_components = min(n_components, len(s))
    else:
        n_components = len(s)

    # Step 3 & 4: Reconstruction via diagonal averaging
    def _reconstruct_component(indices: list[int]) -> np.ndarray:
        """Reconstruct a time-domain signal from selected singular triplets."""
        # Build partial trajectory matrix
        partial = np.zeros_like(traj)
        for idx in indices:
            if idx < len(s):
                partial += s[idx] * np.outer(u[:, idx], vt[idx, :])

        # Diagonal averaging (Hankelisation)
        result = np.zeros(n, dtype=np.float64)
        counts = np.zeros(n, dtype=np.float64)
        l_star = min(window, k)
        k_star = max(window, k)
        for i in range(n):
            start = max(0, i - k_star + 1)
            end = min(i, l_star - 1) + 1
            for j in range(start, end):
                if window <= k:
                    result[i] += partial[j, i - j]
                else:
                    result[i] += partial[i - j, j]
                counts[i] += 1.0
        result /= counts
        return result

    # Build component dictionary
    components: dict[str, pd.Series] = {}
    if groups is not None:
        for name, indices in groups.items():
            valid = [i for i in indices if i < n_components]
            if valid:
                recon = _reconstruct_component(valid)
                components[name] = pd.Series(
                    recon, index=clean.index, name=name,
                )
    else:
        for i in range(n_components):
            name = f"component_{i}"
            recon = _reconstruct_component([i])
            components[name] = pd.Series(
                recon, index=clean.index, name=name,
            )

    return {
        "components": components,
        "singular_values": s[:n_components],
        "explained_variance": explained_variance[:n_components],
        "window": window,
    }


# ---------------------------------------------------------------------------
# Empirical Mode Decomposition
# ---------------------------------------------------------------------------


def emd_decompose(
    data: pd.Series,
    max_imfs: int = 10,
    max_siftings: int = 100,
    sifting_threshold: float = 0.05,
) -> dict:
    """Empirical Mode Decomposition (EMD) via the sifting algorithm.

    EMD adaptively decomposes a non-stationary, non-linear signal into a
    finite set of Intrinsic Mode Functions (IMFs) and a monotonic residual.
    Each IMF captures a narrow-band oscillatory mode whose frequency and
    amplitude can vary over time.

    Unlike Fourier or wavelet analysis, EMD makes no assumption about
    basis functions -- the decomposition is entirely data-driven.

    Algorithm (sifting):
        1. Identify local maxima and minima of the signal.
        2. Construct upper and lower envelopes by cubic spline
           interpolation through the extrema.
        3. Compute the mean of the envelopes.
        4. Subtract the mean from the signal to obtain a candidate IMF.
        5. Repeat steps 1-4 until the candidate satisfies the IMF
           criteria (symmetric envelopes, near-zero mean).
        6. Subtract the extracted IMF from the signal and repeat from
           step 1 to extract the next IMF.

    Parameters:
        data: Time series to decompose. NaN values are dropped.
        max_imfs: Maximum number of IMFs to extract (default 10).
        max_siftings: Maximum sifting iterations per IMF (default 100).
        sifting_threshold: Convergence threshold for the normalised
            squared difference between successive sifting iterations
            (default 0.05).

    Returns:
        Dictionary with:
        - ``imfs``: 2-D numpy array of shape ``(n_imfs, N)`` where each
          row is an IMF ordered from highest to lowest frequency.
        - ``residual``: 1-D numpy array of the monotonic residual.
        - ``n_imfs``: number of IMFs extracted.

    Example:
        >>> import numpy as np, pandas as pd
        >>> t = np.linspace(0, 1, 500)
        >>> sig = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 20 * t) + t
        >>> data = pd.Series(sig)
        >>> result = emd_decompose(data, max_imfs=5)
        >>> result['imfs'].shape[0] >= 1
        True
        >>> np.all(np.isfinite(result['imfs']))
        True

    References:
        - Huang, N.E. et al. (1998), "The empirical mode decomposition and
          the Hilbert spectrum for nonlinear and non-stationary time series
          analysis", Proceedings of the Royal Society A.
    """
    from scipy.interpolate import CubicSpline

    clean = data.dropna()
    x = clean.values.astype(np.float64).copy()
    n = len(x)

    imfs: list[np.ndarray] = []
    residual = x.copy()

    for _ in range(max_imfs):
        h = residual.copy()
        prev_h = np.zeros_like(h)

        for _ in range(max_siftings):
            # Find local extrema
            maxima = []
            minima = []
            for i in range(1, n - 1):
                if h[i] > h[i - 1] and h[i] > h[i + 1]:
                    maxima.append(i)
                elif h[i] < h[i - 1] and h[i] < h[i + 1]:
                    minima.append(i)

            # Need at least 2 maxima and 2 minima for splines
            if len(maxima) < 2 or len(minima) < 2:
                break

            # Extend endpoints for better boundary behavior
            max_idx = np.array([0] + maxima + [n - 1])
            min_idx = np.array([0] + minima + [n - 1])
            max_val = h[max_idx]
            min_val = h[min_idx]

            # Cubic spline envelopes
            t_axis = np.arange(n)
            upper_env = CubicSpline(max_idx, max_val)(t_axis)
            lower_env = CubicSpline(min_idx, min_val)(t_axis)

            mean_env = (upper_env + lower_env) / 2.0
            prev_h = h.copy()
            h = h - mean_env

            # Check convergence (normalised squared difference)
            denom = np.sum(prev_h ** 2)
            if denom > 0:
                sd = np.sum((h - prev_h) ** 2) / denom
                if sd < sifting_threshold:
                    break

        imfs.append(h)
        residual = residual - h

        # Stop if the residual is monotonic or nearly constant
        diffs = np.diff(residual)
        if np.all(diffs >= 0) or np.all(diffs <= 0) or np.std(residual) < 1e-10:
            break

    imfs_array = np.array(imfs) if imfs else np.empty((0, n))

    return {
        "imfs": imfs_array,
        "residual": residual,
        "n_imfs": len(imfs),
    }


# ---------------------------------------------------------------------------
# Wavelet Decomposition
# ---------------------------------------------------------------------------


@requires_extra("timeseries")
def wavelet_decompose(
    data: pd.Series,
    wavelet: str = "db4",
    level: int | None = None,
) -> dict:
    """Multi-level discrete wavelet decomposition.

    Decomposes a time series into approximation and detail coefficients
    at multiple resolution levels using the Discrete Wavelet Transform
    (DWT). At each level, the signal is split into a low-frequency
    approximation and a high-frequency detail component.

    **When to use**:
        - Analysing time-frequency structure at multiple scales.
        - Denoising (threshold the detail coefficients).
        - Feature extraction for ML (energy at each scale).
        - When the signal has transient or localised frequency content
          that Fourier analysis would miss.

    Parameters:
        data: Time series to decompose. NaN values are dropped.
        wavelet: Wavelet family and order. Common choices:
            ``"db4"`` (Daubechies-4, good general purpose),
            ``"haar"`` (simplest, piecewise constant),
            ``"sym5"`` (near-symmetric Daubechies),
            ``"coif3"`` (Coiflet, near-symmetric with vanishing moments).
        level: Decomposition level. If ``None``, the maximum useful level
            is computed automatically via ``pywt.dwt_max_level``.

    Returns:
        Dictionary with:
        - ``approximation``: 1-D numpy array of the final-level
          approximation coefficients (lowest frequency content).
        - ``details``: list of 1-D numpy arrays, one per level, ordered
          from the finest (highest frequency, level 1) to the coarsest
          (level ``level``). ``details[0]`` is level 1 (finest detail).
        - ``wavelet``: wavelet name used.
        - ``level``: decomposition level used.
        - ``coeffs``: raw coefficient list ``[cA_n, cD_n, ..., cD_1]``
          as returned by ``pywt.wavedec`` for advanced use.

    Example:
        >>> import numpy as np, pandas as pd
        >>> data = pd.Series(np.random.default_rng(42).normal(0, 1, 256))
        >>> result = wavelet_decompose(data, wavelet="db4")  # doctest: +SKIP
        >>> len(result['details']) == result['level']  # doctest: +SKIP
        True

    References:
        - Mallat, S. (2009), *A Wavelet Tour of Signal Processing*.
          Academic Press.
    """
    import pywt

    clean = data.dropna()
    values = clean.values.astype(np.float64)

    if level is None:
        level = pywt.dwt_max_level(len(values), wavelet)

    coeffs = pywt.wavedec(values, wavelet, level=level)
    # coeffs = [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    approximation = coeffs[0]
    # Reverse detail order so details[0] = level 1 (finest)
    details = list(reversed(coeffs[1:]))

    return {
        "approximation": approximation,
        "details": details,
        "wavelet": wavelet,
        "level": level,
        "coeffs": coeffs,
    }


# ---------------------------------------------------------------------------
# Unobserved Components Model
# ---------------------------------------------------------------------------


def unobserved_components(
    data: pd.Series,
    level: bool = True,
    trend: bool = False,
    cycle: bool = False,
    seasonal: int | None = None,
    stochastic_level: bool = True,
    stochastic_trend: bool = True,
    stochastic_cycle: bool = True,
    stochastic_seasonal: bool = True,
) -> dict:
    """Unobserved Components Model (UCM) decomposition.

    Fits a structural time series model (Harvey 1989) that decomposes the
    series into interpretable latent components estimated via the Kalman
    filter and smoother:

    - **Level** (local level / random walk): stochastic intercept.
    - **Trend** (local linear trend): level + stochastic slope.
    - **Cycle**: damped stochastic trigonometric cycle.
    - **Seasonal**: time-varying seasonal pattern.
    - **Irregular**: white noise observation error.

    The model is:
        ``y_t = mu_t + gamma_t + c_t + epsilon_t``

    where mu is the trend component (level + slope), gamma is seasonal,
    c is cycle, and epsilon is irregular.

    **When to use**:
        - Macro-economic or business time series where you want
          interpretable, probabilistic decomposition.
        - When you need confidence intervals on components (STL does not
          provide these).
        - Structural break analysis: sudden changes show up as jumps in
          the stochastic level.

    Parameters:
        data: Time series to decompose. Should have a DatetimeIndex for
            best results, but integer index also works.
        level: Include a stochastic level component (default True).
        trend: Include a stochastic trend / slope component
            (default False). When True, the model is a local linear trend.
        cycle: Include a damped stochastic cycle (default False).
        seasonal: Seasonal period. If ``None``, no seasonal component
            is included.
        stochastic_level: Allow the level to vary over time
            (default True).
        stochastic_trend: Allow the slope to vary over time
            (default True).
        stochastic_cycle: Allow the cycle amplitude/phase to vary
            (default True).
        stochastic_seasonal: Allow the seasonal pattern to evolve
            (default True).

    Returns:
        Dictionary with:
        - ``trend``: pd.Series of the estimated trend (level + slope).
        - ``trend_ci``: pd.DataFrame with ``lower`` and ``upper`` columns
          (95% confidence interval for the trend).
        - ``seasonal``: pd.Series of the seasonal component (or None if
          no seasonal was specified).
        - ``seasonal_ci``: pd.DataFrame or None.
        - ``cycle``: pd.Series of the cycle component (or None).
        - ``cycle_ci``: pd.DataFrame or None.
        - ``irregular``: pd.Series of the irregular / residual component.
        - ``model``: the fitted statsmodels ``UnobservedComponentsResults``
          object for further inspection.

    Example:
        >>> import numpy as np, pandas as pd
        >>> idx = pd.date_range("2015-01-01", periods=120, freq="MS")
        >>> t = np.arange(120, dtype=float)
        >>> y = 0.5 * t + 3 * np.sin(2 * np.pi * t / 12) + np.random.default_rng(42).normal(0, 1, 120)
        >>> data = pd.Series(y, index=idx)
        >>> result = unobserved_components(data, level=True, trend=True, seasonal=12)
        >>> result['trend'] is not None
        True

    References:
        - Harvey, A.C. (1989), *Forecasting, Structural Time Series
          Models and the Kalman Filter*. Cambridge University Press.
        - Durbin, J. & Koopman, S.J. (2012), *Time Series Analysis by
          State Space Methods*. Oxford University Press.
    """
    import warnings

    from statsmodels.tsa.statespace.structural import UnobservedComponents

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = UnobservedComponents(
            data,
            level=("local linear trend" if trend else "local level") if level else False,
            cycle=cycle,
            seasonal=seasonal,
            stochastic_level=stochastic_level,
            stochastic_trend=stochastic_trend,
            stochastic_cycle=stochastic_cycle,
            stochastic_seasonal=stochastic_seasonal,
        )
        fit = model.fit(disp=False, maxiter=500)

    idx = data.index

    # Extract trend component
    trend_vals = fit.level["smoothed"]
    if hasattr(fit, "trend") and fit.trend is not None:
        trend_slope = fit.trend["smoothed"]
        trend_vals = trend_vals  # level already includes the trend effect

    trend_series = pd.Series(trend_vals, index=idx, name="trend")

    # Confidence intervals for trend (approximate via +-2 * std)
    trend_ci = None
    try:
        trend_var = fit.level["smoothed_cov"].flatten()
        trend_std = np.sqrt(np.maximum(trend_var, 0))
        trend_ci = pd.DataFrame(
            {
                "lower": trend_vals - 1.96 * trend_std,
                "upper": trend_vals + 1.96 * trend_std,
            },
            index=idx,
        )
    except Exception:  # noqa: BLE001
        pass

    # Seasonal
    seasonal_series = None
    seasonal_ci = None
    if seasonal is not None:
        try:
            seasonal_vals = fit.seasonal["smoothed"]
            seasonal_series = pd.Series(seasonal_vals, index=idx, name="seasonal")
            seasonal_var = fit.seasonal["smoothed_cov"].flatten()
            seasonal_std = np.sqrt(np.maximum(seasonal_var, 0))
            seasonal_ci = pd.DataFrame(
                {
                    "lower": seasonal_vals - 1.96 * seasonal_std,
                    "upper": seasonal_vals + 1.96 * seasonal_std,
                },
                index=idx,
            )
        except Exception:  # noqa: BLE001
            pass

    # Cycle
    cycle_series = None
    cycle_ci = None
    if cycle:
        try:
            cycle_vals = fit.cycle["smoothed"]
            cycle_series = pd.Series(cycle_vals, index=idx, name="cycle")
            cycle_var = fit.cycle["smoothed_cov"].flatten()
            cycle_std = np.sqrt(np.maximum(cycle_var, 0))
            cycle_ci = pd.DataFrame(
                {
                    "lower": cycle_vals - 1.96 * cycle_std,
                    "upper": cycle_vals + 1.96 * cycle_std,
                },
                index=idx,
            )
        except Exception:  # noqa: BLE001
            pass

    # Irregular
    irregular = pd.Series(fit.resid, index=idx, name="irregular")

    return {
        "trend": trend_series,
        "trend_ci": trend_ci,
        "seasonal": seasonal_series,
        "seasonal_ci": seasonal_ci,
        "cycle": cycle_series,
        "cycle_ci": cycle_ci,
        "irregular": irregular,
        "model": fit,
    }
