"""Characteristic function methods for option pricing.

Characteristic functions provide a unified framework for pricing European
options under a wide variety of models.  Given the characteristic function
phi(u) of the log-price log(S_T), the call price can be recovered via
either:

* **FFT (Carr-Madan)** -- Fourier transform of the modified payoff,
  evaluated efficiently via the Fast Fourier Transform.
* **COS (Fang-Oosterlee)** -- Cosine expansion of the risk-neutral density,
  integrated analytically against the payoff coefficients.

This module provides **model-specific characteristic functions** for:

* **Heston** stochastic volatility -- the workhorse model for equity
  vol smiles.  Captures mean-reverting variance with leverage effect.
* **Variance Gamma (VG)** -- a pure-jump Lévy process that nests
  Black-Scholes as a limiting case.  Three parameters control volatility,
  skewness, and kurtosis independently.
* **Normal Inverse Gaussian (NIG)** -- a flexible Lévy process with
  semi-heavy tails; popular in credit and commodity markets.
* **CGMY** -- generalisation of VG with a parameter controlling the
  fine structure of jumps (finite/infinite activity, finite/infinite
  variation).

All characteristic functions return callables that can be plugged directly
into :func:`wraquant.price.levy_pricing.fft_option_price` or
:func:`wraquant.price.levy_pricing.cos_method`.

References:
    - Heston (1993). *A Closed-Form Solution for Options with Stochastic
      Volatility.*  Review of Financial Studies 6(2), 327-343.
    - Madan, Carr & Chang (1998). *The Variance Gamma Process and Option
      Pricing.*  European Finance Review 2, 79-105.
    - Barndorff-Nielsen (1997). *Normal Inverse Gaussian Distributions
      and Stochastic Volatility Modelling.*  Scandinavian Journal of
      Statistics 24(1), 1-13.
    - Carr, Geman, Madan & Yor (2002). *The Fine Structure of Asset
      Returns: An Empirical Investigation.*  Journal of Business 75(2).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt

from wraquant.price.levy_pricing import cos_method, fft_option_price

__all__ = [
    "characteristic_function_price",
    "heston_characteristic",
    "vg_characteristic",
    "nig_characteristic",
    "cgmy_characteristic",
]


# ---------------------------------------------------------------------------
# Generic pricing via characteristic function
# ---------------------------------------------------------------------------

def characteristic_function_price(
    char_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.complex128]],
    spot: float,
    strike: float,
    rf: float,
    T: float,
    method: str = "fft",
    n_terms: int = 4096,
) -> float:
    """Price a European call option given a characteristic function.

    This is the **universal interface** for Fourier-based option pricing.
    Any model whose characteristic function of log(S_T) is known in
    closed form can be priced through this single function, unifying
    Black-Scholes, Heston, Variance Gamma, NIG, CGMY, and others under
    one consistent API.

    The connection to the pricing PDE is via the Feynman-Kac theorem:
    the characteristic function encodes the full risk-neutral distribution
    of the log-price, and the Fourier inversion recovers the density
    (or directly the option price via the Carr-Madan or COS method).

    Parameters:
        char_fn: Characteristic function of log(S_T) under the risk-neutral
            measure.  Signature: ``char_fn(u) -> complex ndarray`` where
            ``u`` is a real-valued array of frequencies.
        spot: Current spot price of the underlying.
        strike: Strike price.
        rf: Risk-free interest rate (annualised, continuously compounded).
        T: Time to maturity in years.
        method: Pricing method -- ``"fft"`` (Carr-Madan, default) or
            ``"cos"`` (Fang-Oosterlee cosine expansion).
        n_terms: Number of terms/grid points.  For FFT this is the FFT
            size (default 4096); for COS this is the number of cosine
            terms.

    Returns:
        European call option price.

    Example:
        >>> import numpy as np
        >>> # Black-Scholes via characteristic function
        >>> S, K, r, T, sigma = 100.0, 100.0, 0.05, 1.0, 0.2
        >>> log_S = np.log(S)
        >>> drift = (r - 0.5 * sigma**2) * T
        >>> char_fn = lambda u: np.exp(1j*u*(log_S + drift) - 0.5*sigma**2*T*u**2)
        >>> price = characteristic_function_price(char_fn, S, K, r, T)
        >>> 9.0 < price < 12.0
        True
    """
    method = method.lower().strip()
    if method == "cos":
        return cos_method(char_fn, spot, strike, rf, T, n_terms=n_terms)
    elif method == "fft":
        price = fft_option_price(char_fn, spot, strike, rf, T, n_fft=n_terms)
        return float(price)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'fft' or 'cos'.")


# ---------------------------------------------------------------------------
# Heston characteristic function
# ---------------------------------------------------------------------------

def heston_characteristic(
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    rf: float,
    T: float,
    spot: float = 1.0,
) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.complex128]]:
    r"""Characteristic function of log(S_T) under the Heston model.

    The Heston (1993) stochastic volatility model specifies:

    .. math::

        dS_t &= r\,S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^1 \\
        dv_t &= \kappa(\theta - v_t)\,dt + \sigma_v\sqrt{v_t}\,dW_t^2 \\
        \text{corr}(dW^1, dW^2) &= \rho

    The characteristic function of :math:`\ln(S_T)` has a known
    closed-form expression involving complex exponentials of the
    model parameters.

    **When to use Heston vs Black-Scholes:**

    - Use **Black-Scholes** when the implied vol surface is flat
      (no smile/skew) or as a quick approximation.
    - Use **Heston** when you observe a volatility smile/skew in the
      market and need to match observed option prices across strikes.
      The negative :math:`\rho` parameter captures the leverage effect
      (negative correlation between returns and vol).
    - Heston is the standard model for equity index options and is
      widely used for calibration to the vol surface.

    Parameters:
        v0: Initial variance :math:`v_0`.
        kappa: Mean reversion speed of variance.
        theta: Long-run variance level.
        sigma_v: Volatility of variance (vol of vol).
        rho: Correlation between price and variance Brownian motions.
        rf: Risk-free rate.
        T: Time to maturity in years.
        spot: Current spot price (default 1.0).

    Returns:
        Callable ``char_fn(u)`` that returns the characteristic function
        evaluated at frequencies ``u``.

    Example:
        >>> char_fn = heston_characteristic(
        ...     v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3,
        ...     rho=-0.7, rf=0.05, T=1.0, spot=100.0,
        ... )
        >>> import numpy as np
        >>> val = char_fn(np.array([0.0]))
        >>> np.abs(val[0])  # phi(0) = 1
        1.0

    References:
        Heston, S.L. (1993). *A Closed-Form Solution for Options with
        Stochastic Volatility with Applications to Bond and Currency
        Options.*  Review of Financial Studies 6(2), 327-343.
    """
    log_spot = np.log(spot)

    def char_fn(u: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        # Use the "little Heston trap" formulation (Albrecher et al. 2007)
        # for numerical stability
        iu = 1j * u

        d = np.sqrt(
            (rho * sigma_v * iu - kappa) ** 2
            + sigma_v ** 2 * (iu + u ** 2)
        )

        g = (kappa - rho * sigma_v * iu - d) / (kappa - rho * sigma_v * iu + d)

        exp_dT = np.exp(-d * T)

        C = (rf * iu * T
             + (kappa * theta / sigma_v ** 2)
             * ((kappa - rho * sigma_v * iu - d) * T
                - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))))

        D = ((kappa - rho * sigma_v * iu - d) / sigma_v ** 2
             * (1.0 - exp_dT) / (1.0 - g * exp_dT))

        return np.exp(C + D * v0 + iu * log_spot)

    return char_fn


# ---------------------------------------------------------------------------
# Variance Gamma characteristic function
# ---------------------------------------------------------------------------

def vg_characteristic(
    sigma: float,
    nu: float,
    theta_vg: float,
    rf: float,
    T: float,
    spot: float = 1.0,
) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.complex128]]:
    r"""Characteristic function of log(S_T) under the Variance Gamma model.

    The Variance Gamma (VG) process models log-returns as a Brownian
    motion evaluated at a random (Gamma-distributed) time.  This
    introduces both skewness and excess kurtosis into the return
    distribution while remaining analytically tractable.

    .. math::

        \phi(u) = \exp\bigl(iu(r + \omega)T\bigr)
        \cdot \left(1 - iu\,\theta\,\nu
        + \tfrac{1}{2}\sigma^2\nu\,u^2\right)^{-T/\nu}

    where :math:`\omega = \frac{1}{\nu}\ln(1 - \theta\nu - \sigma^2\nu/2)`
    is the convexity correction.

    Parameters:
        sigma: Volatility parameter of the VG process.
        nu: Variance rate of the Gamma subordinator.  Controls kurtosis;
            as nu -> 0 the model converges to Black-Scholes.
        theta_vg: Drift parameter of the VG process.  Controls skewness;
            negative values produce left-skewed returns (typical for
            equity markets).
        rf: Risk-free rate.
        T: Time to maturity.
        spot: Current spot price (default 1.0).

    Returns:
        Callable ``char_fn(u)`` returning the characteristic function.

    Example:
        >>> char_fn = vg_characteristic(0.2, 0.5, -0.1, 0.05, 1.0, spot=100)
        >>> import numpy as np
        >>> np.abs(char_fn(np.array([0.0]))[0])
        1.0
    """
    log_spot = np.log(spot)

    def char_fn(u: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        omega = (1.0 / nu) * np.log(
            1.0 - theta_vg * nu - 0.5 * sigma ** 2 * nu
        )
        base = 1.0 - 1j * theta_vg * nu * u + 0.5 * sigma ** 2 * nu * u ** 2
        return np.exp(
            1j * u * (log_spot + (rf + omega) * T)
            - (T / nu) * np.log(base)
        )

    return char_fn


# ---------------------------------------------------------------------------
# Normal Inverse Gaussian characteristic function
# ---------------------------------------------------------------------------

def nig_characteristic(
    alpha: float,
    beta: float,
    mu: float,
    delta: float,
    rf: float,
    T: float,
    spot: float = 1.0,
) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.complex128]]:
    r"""Characteristic function of log(S_T) under the NIG model.

    The Normal Inverse Gaussian (NIG) distribution arises as a normal
    variance-mean mixture with inverse Gaussian mixing distribution.
    It features semi-heavy tails (heavier than Gaussian, lighter than
    Cauchy) and is popular for modelling credit spreads, commodity
    returns, and FX markets.

    .. math::

        \phi(u) = \exp\bigl(iu(r + \omega + \mu)T
        + \delta T\bigl(\gamma_0 - \sqrt{\alpha^2 - (\beta + iu)^2}\bigr)\bigr)

    where :math:`\gamma_0 = \sqrt{\alpha^2 - \beta^2}` and
    :math:`\omega = \delta(\sqrt{\alpha^2 - (\beta+1)^2} - \gamma_0) - \mu`
    ensures the martingale condition.

    Parameters:
        alpha: Tail heaviness / steepness parameter (alpha > 0).
            Larger alpha = lighter tails.
        beta: Asymmetry parameter (-alpha < beta < alpha).
            Negative beta = left skew.
        mu: Location parameter.
        delta: Scale parameter (delta > 0).
        rf: Risk-free rate.
        T: Time to maturity.
        spot: Current spot price (default 1.0).

    Returns:
        Callable ``char_fn(u)`` returning the characteristic function.

    Example:
        >>> char_fn = nig_characteristic(15.0, -3.0, 0.0, 0.5, 0.05, 1.0,
        ...                              spot=100)
        >>> import numpy as np
        >>> np.abs(char_fn(np.array([0.0]))[0])
        1.0
    """
    log_spot = np.log(spot)

    def char_fn(u: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        gamma0 = np.sqrt(alpha ** 2 - beta ** 2)
        gamma1 = np.sqrt(alpha ** 2 - (beta + 1.0) ** 2)
        omega = delta * (gamma1 - gamma0) - mu
        gamma_u = np.sqrt(alpha ** 2 - (beta + 1j * u) ** 2)
        return np.exp(
            1j * u * (log_spot + (rf + omega + mu) * T)
            + delta * T * (gamma0 - gamma_u)
        )

    return char_fn


# ---------------------------------------------------------------------------
# CGMY characteristic function
# ---------------------------------------------------------------------------

def cgmy_characteristic(
    C: float,
    G: float,
    M: float,
    Y_param: float,
    rf: float,
    T: float,
    spot: float = 1.0,
) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.complex128]]:
    r"""Characteristic function of log(S_T) under the CGMY model.

    The CGMY model (Carr, Geman, Madan & Yor, 2002) generalises the
    Variance Gamma process with an additional parameter Y controlling
    the fine structure of jumps:

    * Y < 0: finite activity (finitely many jumps in any interval)
    * 0 <= Y < 1: infinite activity, finite variation
    * 1 <= Y < 2: infinite activity, infinite variation

    .. math::

        \phi(u) = \exp\bigl(iu(r + \omega)T
        + C\,T\,\Gamma(-Y)\bigl[
            (M - iu)^Y - M^Y + (G + iu)^Y - G^Y
        \bigr]\bigr)

    where :math:`\omega = -C\,\Gamma(-Y)[(M-1)^Y - M^Y + (G+1)^Y - G^Y]`
    is the convexity correction.

    Parameters:
        C: Overall activity level (C > 0).
        G: Rate of exponential decay of the positive jump density (G > 0).
            Larger G = fewer large positive jumps.
        M: Rate of exponential decay of the negative jump density (M > 0).
            Larger M = fewer large negative jumps.
        Y_param: Fine structure parameter (Y < 2, Y != 0, Y != 1).
            See classification above.
        rf: Risk-free rate.
        T: Time to maturity.
        spot: Current spot price (default 1.0).

    Returns:
        Callable ``char_fn(u)`` returning the characteristic function.

    Example:
        >>> char_fn = cgmy_characteristic(1.0, 5.0, 10.0, 0.5,
        ...                               0.05, 1.0, spot=100)
        >>> import numpy as np
        >>> np.abs(char_fn(np.array([0.0]))[0])
        1.0

    References:
        Carr, P., Geman, H., Madan, D.B. & Yor, M. (2002). *The Fine
        Structure of Asset Returns: An Empirical Investigation.* Journal
        of Business 75(2), 305-332.
    """
    from scipy.special import gamma as gamma_fn

    log_spot = np.log(spot)
    gam_neg_Y = gamma_fn(-Y_param)

    # Convexity correction
    omega = -C * gam_neg_Y * (
        (M - 1.0) ** Y_param - M ** Y_param
        + (G + 1.0) ** Y_param - G ** Y_param
    )

    def char_fn(u: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        term = C * T * gam_neg_Y * (
            (M - 1j * u) ** Y_param - M ** Y_param
            + (G + 1j * u) ** Y_param - G ** Y_param
        )
        return np.exp(
            1j * u * (log_spot + (rf + omega) * T) + term
        )

    return char_fn
