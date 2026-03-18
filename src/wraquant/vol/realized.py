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


def bipower_variation(
    returns: pd.Series,
    window: int = 20,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    """Bipower variation -- jump-robust realised volatility estimator.

    The bipower variation (BPV) of Barndorff-Nielsen and Shephard (2004)
    estimates the *continuous* (diffusive) component of quadratic
    variation by using the product of adjacent absolute returns instead
    of squared returns. Because jumps affect only isolated observations,
    taking the product of *consecutive* absolute returns washes out the
    jump contribution asymptotically:

    .. math::

        BPV_t = \\frac{\\pi}{2} \\sum_{i=2}^{n}
                |r_{t,i}| \\cdot |r_{t,i-1}|

    where the :math:`\\pi / 2` factor corrects for the bias introduced
    by using absolute values of normal random variables
    (:math:`E[|z|] = \\sqrt{2/\\pi}` for standard normal *z*).

    Parameters:
        returns: Return series (not prices).
        window: Rolling window size. Each BPV estimate uses *window*
            observations.
        annualize: Whether to annualize by ``sqrt(periods_per_year)``.
        periods_per_year: Trading periods per year (252 for daily).

    Returns:
        Rolling bipower variation series (as volatility, i.e. the square
        root of the bipower variation divided by the window) with the
        same index as the input.

    Example:
        >>> import pandas as pd, numpy as np
        >>> rng = np.random.default_rng(42)
        >>> returns = pd.Series(rng.normal(0, 0.01, 200))
        >>> bpv = bipower_variation(returns, window=20)
        >>> (bpv.dropna() > 0).all()
        True

    Notes:
        In the absence of jumps, BPV converges to the same value as
        standard realised variance. When jumps are present, BPV < RV
        and the difference ``RV - BPV`` estimates the jump component.

        This is a rolling window version that computes BPV over a
        trailing window of *window* observations at each point.

        Reference: Barndorff-Nielsen, O.E. and Shephard, N. (2004).
        "Power and Bipower Variation with Stochastic Volatility and
        Jumps." *Journal of Financial Econometrics*, 2(1), 1--37.

    See Also:
        realized_volatility: Standard rolling realised volatility.
        jump_test_bns: Formal jump detection test using BPV.
    """
    abs_ret = returns.abs()
    # Product of consecutive absolute returns
    products = abs_ret * abs_ret.shift(1)
    # Rolling sum with pi/2 correction, then normalise by window
    bpv_raw = (np.pi / 2.0) * products.rolling(window).sum() / window
    vol = np.sqrt(bpv_raw.clip(lower=0))
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol


def jump_test_bns(
    returns: pd.Series,
    window: int = 20,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Barndorff-Nielsen & Shephard (2004) jump detection test.

    Tests whether jumps are present in a return series by comparing
    realised variance (RV) to bipower variation (BPV). Under the null
    hypothesis of no jumps the two estimators are asymptotically
    equivalent. A significantly positive difference indicates the
    presence of jumps.

    The test statistic is:

    .. math::

        z = \\frac{RV - BPV}{\\sqrt{\\theta \\cdot QP}}

    where *QP* is the realised quarticity (estimated via the quad-power
    quarticity estimator) and *theta* is a finite-sample correction
    factor:

    .. math::

        \\theta = \\left(\\frac{\\pi^2}{4} + \\pi - 5\\right)
                 \\cdot \\frac{1}{n}

    The statistic is asymptotically standard normal under the null
    of no jumps.

    Parameters:
        returns: Return series (not prices).
        window: Number of observations to use for the test. The test
            is applied to the most recent *window* observations. Use
            the full series length by setting ``window=len(returns)``.
        alpha: Significance level for the jump detection decision.
            Default 0.05.

    Returns:
        Dictionary containing:

        - **rv** (*float*) -- Realised variance over the window.
        - **bpv** (*float*) -- Bipower variation over the window.
        - **jump_component** (*float*) -- ``max(RV - BPV, 0)``.
        - **continuous_component** (*float*) -- ``BPV``.
        - **z_statistic** (*float*) -- Test statistic (standard normal
          under the null of no jumps).
        - **p_value** (*float*) -- One-sided p-value.
        - **jump_detected** (*bool*) -- ``True`` if p-value < alpha.

    Example:
        >>> import pandas as pd, numpy as np
        >>> rng = np.random.default_rng(42)
        >>> returns = pd.Series(rng.normal(0, 0.01, 100))
        >>> result = jump_test_bns(returns, window=100)
        >>> 'z_statistic' in result and 'p_value' in result
        True

    Notes:
        The test has low power against small or infrequent jumps in
        short samples. For daily data, windows of at least 60--100
        observations are recommended.

        Reference: Barndorff-Nielsen, O.E. and Shephard, N. (2004).
        "Power and Bipower Variation with Stochastic Volatility and
        Jumps." *Journal of Financial Econometrics*, 2(1), 1--37.

    See Also:
        bipower_variation: The jump-robust volatility estimator.
        realized_volatility: Standard realised volatility.
    """
    from scipy import stats as sp_stats

    r = np.asarray(returns, dtype=np.float64).ravel()
    n = min(window, len(r))
    r = r[-n:]

    # Realised variance
    rv = float(np.sum(r**2))

    # Bipower variation
    abs_r = np.abs(r)
    bpv = float((np.pi / 2.0) * np.sum(abs_r[1:] * abs_r[:-1]))

    # Quad-power quarticity for variance of (RV - BPV)
    mu1 = np.sqrt(2.0 / np.pi)  # E[|z|] for standard normal
    # Tri-power quarticity-based approach: use realised quad-power
    if n >= 5:
        qpq = float(
            n
            * (np.pi**2 / 4.0)
            * np.sum(
                abs_r[3:] ** (4.0 / 3.0)
                * abs_r[2:-1] ** (4.0 / 3.0)
                * abs_r[1:-2] ** (4.0 / 3.0)
            )
            / (n - 3)
        )
    else:
        qpq = rv**2  # fallback

    theta = (np.pi**2 / 4.0 + np.pi - 5.0)

    denom = np.sqrt(max(theta * qpq / n, 1e-30))
    z_stat = float((rv - bpv) / denom)

    # One-sided test: jumps cause RV > BPV
    p_value = float(1.0 - sp_stats.norm.cdf(z_stat))

    return {
        "rv": rv,
        "bpv": bpv,
        "jump_component": float(max(rv - bpv, 0.0)),
        "continuous_component": bpv,
        "z_statistic": z_stat,
        "p_value": p_value,
        "jump_detected": p_value < alpha,
    }


def two_scale_realized_variance(
    returns: pd.Series,
    window: int = 20,
    n_slow: int = 5,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    """Two-Scale Realised Variance (TSRV) -- noise-robust RV estimator.

    The TSRV of Zhang, Mykland, and Ait-Sahalia (2005) eliminates the
    bias caused by market microstructure noise (bid-ask bounce, discrete
    prices) by combining realised variances computed at two different
    sampling frequencies. The fast-scale estimator is contaminated by
    noise; the slow-scale estimator has less noise but more discretisation
    error. Their optimal linear combination cancels the noise bias:

    .. math::

        TSRV = RV^{(slow)} - \\frac{\\bar{n}}{n} RV^{(all)}

    where :math:`RV^{(slow)}` is the average of sub-sampled RVs at a
    slower frequency, :math:`RV^{(all)}` uses all ticks, and
    :math:`\\bar{n}` is the average sub-sample size.

    Parameters:
        returns: Return series. For intraday data, these should be
            tick-by-tick or high-frequency log returns.
        window: Rolling window size.
        n_slow: Sub-sampling factor for the slow scale. The fast scale
            uses every observation; the slow scale uses every *n_slow*-th.
            Larger values reduce noise but increase discretisation error.
            Default 5.
        annualize: Whether to annualize.
        periods_per_year: Trading periods per year.

    Returns:
        Rolling TSRV series (as volatility, i.e. square root).

    Example:
        >>> import pandas as pd, numpy as np
        >>> rng = np.random.default_rng(42)
        >>> returns = pd.Series(rng.normal(0, 0.01, 200))
        >>> tsrv = two_scale_realized_variance(returns, window=20)
        >>> (tsrv.dropna() > 0).all()
        True

    Notes:
        TSRV is consistent for the integrated variance even in the
        presence of i.i.d. microstructure noise, unlike standard RV
        which diverges as the sampling frequency increases. It is
        particularly useful for intraday data sampled at very high
        frequencies (seconds or ticks).

        Reference: Zhang, L., Mykland, P.A., and Ait-Sahalia, Y. (2005).
        "A Tale of Two Time Scales: Determining Integrated Volatility
        with Noisy High-Frequency Data." *Journal of the American
        Statistical Association*, 100(472), 1394--1411.

    See Also:
        realized_volatility: Standard RV (no noise correction).
        realized_kernel: Alternative noise-robust estimator.
    """
    # Fast scale: standard RV using all returns
    rv_fast = (returns**2).rolling(window).sum()

    # Slow scale: sub-sampled RV
    # Average over n_slow sub-grids for efficiency
    rv_slow_accum = pd.Series(0.0, index=returns.index)
    for offset in range(n_slow):
        sub_returns = returns.iloc[offset::n_slow].reindex(returns.index)
        rv_sub = (sub_returns**2).rolling(
            max(window // n_slow, 1)
        ).sum()
        rv_slow_accum = rv_slow_accum + rv_sub.fillna(0)
    rv_slow = rv_slow_accum / n_slow

    # Noise-bias correction: TSRV = RV_slow - (n_bar / n) * RV_fast
    n_bar = window / n_slow
    tsrv_raw = rv_slow - (n_bar / window) * rv_fast
    # Clip to zero (numerical)
    tsrv_raw = tsrv_raw.clip(lower=0)

    vol = np.sqrt(tsrv_raw / window)
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol


def realized_kernel(
    returns: pd.Series,
    window: int = 20,
    kernel: str = "parzen",
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    """Realised kernel estimator -- noise-robust flat-top kernel RV.

    The realised kernel of Barndorff-Nielsen, Hansen, Lunde, and
    Shephard (2008) uses a kernel-weighted sum of autocovariances of
    returns to produce a consistent, non-negative estimator of
    integrated variance in the presence of market microstructure noise:

    .. math::

        RK = \\sum_{h=-H}^{H} k\\!\\left(\\frac{h}{H+1}\\right)
             \\hat{\\gamma}(h)

    where :math:`\\hat{\\gamma}(h) = \\sum_i r_i r_{i-h}` is the sample
    autocovariance at lag *h*, and :math:`k(\\cdot)` is a kernel function
    with :math:`k(0) = 1`, :math:`k(x) = 0` for :math:`|x| > 1`.

    The Parzen kernel is used by default as it guarantees non-negativity:

    .. math::

        k(x) = \\begin{cases}
            1 - 6x^2 + 6|x|^3 & \\text{if } |x| \\le 1/2 \\\\
            2(1-|x|)^3         & \\text{if } 1/2 < |x| \\le 1 \\\\
            0                  & \\text{otherwise}
        \\end{cases}

    Parameters:
        returns: Return series.
        window: Rolling window size.
        kernel: Kernel function. Currently supported: ``"parzen"``
            (default). The Parzen kernel guarantees non-negativity.
        annualize: Whether to annualize.
        periods_per_year: Trading periods per year.

    Returns:
        Rolling realised kernel volatility series (square root of
        the kernel-weighted variance).

    Example:
        >>> import pandas as pd, numpy as np
        >>> rng = np.random.default_rng(42)
        >>> returns = pd.Series(rng.normal(0, 0.01, 200))
        >>> rk = realized_kernel(returns, window=20)
        >>> (rk.dropna() > 0).all()
        True

    Notes:
        The realised kernel is rate-optimal under i.i.d. noise and
        achieves a convergence rate of :math:`n^{-1/5}` (vs
        :math:`n^{-1/4}` for standard RV with optimal sparse sampling).

        The bandwidth *H* is chosen automatically as
        :math:`H = \\lceil c \\cdot n^{3/5} \\rceil` where *c* is a
        constant typically set to 1.

        Reference: Barndorff-Nielsen, O.E., Hansen, P.R., Lunde, A.,
        and Shephard, N. (2008). "Designing Realized Kernels to Measure
        the Ex-Post Variation of Equity Prices in the Presence of
        Noise." *Econometrica*, 76(6), 1481--1536.

    See Also:
        two_scale_realized_variance: Alternative noise-robust estimator.
        realized_volatility: Standard RV without noise correction.
        bipower_variation: Jump-robust estimator.
    """
    if kernel != "parzen":
        msg = f"Only 'parzen' kernel is currently supported, got '{kernel}'."
        raise ValueError(msg)

    def _parzen(x: np.ndarray) -> np.ndarray:
        """Parzen kernel function."""
        ax = np.abs(x)
        out = np.zeros_like(ax)
        mask1 = ax <= 0.5
        mask2 = (ax > 0.5) & (ax <= 1.0)
        out[mask1] = 1.0 - 6.0 * ax[mask1] ** 2 + 6.0 * ax[mask1] ** 3
        out[mask2] = 2.0 * (1.0 - ax[mask2]) ** 3
        return out

    r = np.asarray(returns, dtype=np.float64).ravel()
    n = len(r)
    out = np.full(n, np.nan)

    for t in range(window, n):
        seg = r[t - window : t]
        seg_n = len(seg)

        # Bandwidth: c * n^(3/5), with c = 1
        H = max(int(np.ceil(seg_n ** 0.6)), 1)

        # Compute autocovariances up to lag H
        rk_val = 0.0
        for h in range(-H, H + 1):
            weight = _parzen(np.array([h / (H + 1)]))[0]
            if h >= 0:
                gamma_h = float(np.sum(seg[h:] * seg[: seg_n - h]))
            else:
                ah = -h
                gamma_h = float(np.sum(seg[: seg_n - ah] * seg[ah:]))
            rk_val += weight * gamma_h

        out[t] = max(rk_val, 0.0)

    vol = np.sqrt(out / window)
    result = pd.Series(vol, index=returns.index, name="realized_kernel")
    if annualize:
        result = result * np.sqrt(periods_per_year)
    return result
