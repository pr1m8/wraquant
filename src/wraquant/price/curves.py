"""Yield curve construction and interpolation.

Provides bootstrapping, interpolation, forward rate calculation,
and discount factor utilities for yield curve analysis.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline, interp1d

__all__ = [
    "bootstrap_zero_curve",
    "interpolate_curve",
    "forward_rate",
    "discount_factor",
]


def bootstrap_zero_curve(
    maturities: Sequence[float] | npt.NDArray[np.floating],
    par_rates: Sequence[float] | npt.NDArray[np.floating],
    freq: int = 2,
) -> npt.NDArray[np.float64]:
    """Bootstrap zero (spot) rates from par coupon rates.

    Assumes par bonds (price = 100) and iteratively strips coupons
    to derive the zero curve.

    Parameters:
        maturities: Array of maturities in years (must be evenly spaced at 1/freq).
        par_rates: Array of par coupon rates (annualized) for each maturity.
        freq: Coupon payments per year (default 2 for semiannual).

    Returns:
        Array of continuously compounded zero rates corresponding to each maturity.

    Example:
        >>> mats = [0.5, 1.0, 1.5, 2.0]
        >>> pars = [0.04, 0.042, 0.044, 0.046]
        >>> zeros = bootstrap_zero_curve(mats, pars, freq=2)
        >>> len(zeros)
        4
    """
    maturities_arr = np.asarray(maturities, dtype=np.float64)
    par_rates_arr = np.asarray(par_rates, dtype=np.float64)

    if len(maturities_arr) != len(par_rates_arr):
        raise ValueError("maturities and par_rates must have the same length.")

    n = len(maturities_arr)
    zero_rates = np.empty(n, dtype=np.float64)

    # Discount factors we have computed so far
    disc_factors: dict[int, float] = {}

    for i in range(n):
        mat = maturities_arr[i]
        coupon = par_rates_arr[i] / freq  # coupon per period as fraction of face
        n_periods = int(round(mat * freq))

        # Sum of PV of coupons using already-bootstrapped discount factors
        pv_coupons = 0.0
        for j in range(1, n_periods):
            # Find the discount factor for period j
            if j - 1 < len(zero_rates) and j - 1 < i:
                t_j = maturities_arr[j - 1] if j - 1 < i else j / freq
                r_j = zero_rates[j - 1]
                df_j = np.exp(-r_j * t_j)
            else:
                t_j = j / freq
                # For intermediate periods not directly bootstrapped,
                # use the nearest available rate
                df_j = disc_factors.get(
                    j, np.exp(-zero_rates[max(0, i - 1)] * t_j) if i > 0 else 1.0
                )
            pv_coupons += coupon * df_j
            disc_factors[j] = df_j

        # Solve for the discount factor at maturity:
        # Par bond: 1 = c * sum(DF_j) + (1 + c) * DF_n
        # pv_coupons already includes coupon weighting (coupon * df_j),
        # so: DF_n = (1 - pv_coupons) / (1 + c)
        df_n = (1.0 - pv_coupons) / (1.0 + coupon)

        # Convert discount factor to continuously compounded zero rate
        zero_rates[i] = -np.log(df_n) / mat
        disc_factors[n_periods] = df_n

    return zero_rates


def interpolate_curve(
    maturities: Sequence[float] | npt.NDArray[np.floating],
    rates: Sequence[float] | npt.NDArray[np.floating],
    target_maturities: Sequence[float] | npt.NDArray[np.floating],
    method: str = "cubic",
) -> npt.NDArray[np.float64]:
    """Interpolate a yield curve at target maturities.

    Parameters:
        maturities: Known maturities (sorted, ascending).
        rates: Known rates at those maturities.
        target_maturities: Maturities at which to interpolate.
        method: Interpolation method. One of 'linear', 'cubic', 'flat_forward'.

    Returns:
        Interpolated rates at target maturities.

    Example:
        >>> mats = [0.5, 1.0, 2.0, 5.0]
        >>> rates = [0.03, 0.035, 0.04, 0.045]
        >>> interpolate_curve(mats, rates, [0.75, 1.5, 3.0])
        array([...])
    """
    mats_arr = np.asarray(maturities, dtype=np.float64)
    rates_arr = np.asarray(rates, dtype=np.float64)
    targets_arr = np.asarray(target_maturities, dtype=np.float64)

    if method == "linear":
        f = interp1d(mats_arr, rates_arr, kind="linear", fill_value="extrapolate")
        return np.asarray(f(targets_arr), dtype=np.float64)

    elif method == "cubic":
        cs = CubicSpline(mats_arr, rates_arr, extrapolate=True)
        return np.asarray(cs(targets_arr), dtype=np.float64)

    elif method == "flat_forward":
        # Flat forward interpolation: constant forward rate between knots
        # First compute discount factors from zero rates
        disc = np.exp(-rates_arr * mats_arr)
        result = np.empty(len(targets_arr), dtype=np.float64)

        for i, t in enumerate(targets_arr):
            if t <= mats_arr[0]:
                # Extrapolate flat from first rate
                result[i] = rates_arr[0]
            elif t >= mats_arr[-1]:
                # Extrapolate flat from last rate
                result[i] = rates_arr[-1]
            else:
                # Find bracketing interval
                idx = np.searchsorted(mats_arr, t, side="right") - 1
                t0 = mats_arr[idx]
                t1 = mats_arr[idx + 1]
                df0 = disc[idx]
                df1 = disc[idx + 1]

                # Forward rate between t0 and t1
                fwd = -np.log(df1 / df0) / (t1 - t0)
                # Discount factor at target
                df_t = df0 * np.exp(-fwd * (t - t0))
                result[i] = -np.log(df_t) / t

        return result

    else:
        raise ValueError(
            f"Unknown interpolation method '{method}'. "
            "Use 'linear', 'cubic', or 'flat_forward'."
        )


def forward_rate(
    zero_rates: Sequence[float] | npt.NDArray[np.floating],
    maturities: Sequence[float] | npt.NDArray[np.floating],
    t1: float,
    t2: float,
) -> np.float64:
    """Compute the forward rate between two future dates.

    Uses continuously compounded zero rates to compute:
    f(t1, t2) = [r2 * t2 - r1 * t1] / (t2 - t1)

    Parameters:
        zero_rates: Array of zero rates.
        maturities: Corresponding maturities.
        t1: Start of the forward period.
        t2: End of the forward period.

    Returns:
        Continuously compounded forward rate between t1 and t2.

    Example:
        >>> forward_rate([0.04, 0.045], [1.0, 2.0], 1.0, 2.0)
        0.05
    """
    if t2 <= t1:
        raise ValueError("t2 must be greater than t1.")

    mats_arr = np.asarray(maturities, dtype=np.float64)
    rates_arr = np.asarray(zero_rates, dtype=np.float64)

    # Interpolate to get zero rates at t1 and t2
    cs = CubicSpline(mats_arr, rates_arr, extrapolate=True)
    r1 = float(cs(t1))
    r2 = float(cs(t2))

    fwd = (r2 * t2 - r1 * t1) / (t2 - t1)
    return np.float64(fwd)


def discount_factor(
    rate: float,
    maturity: float,
) -> np.float64:
    """Compute the discount factor from a continuously compounded rate.

    DF = exp(-rate * maturity)

    Parameters:
        rate: Continuously compounded interest rate.
        maturity: Time to maturity in years.

    Returns:
        Discount factor.

    Example:
        >>> discount_factor(0.05, 1.0)
        0.9512...
    """
    return np.float64(np.exp(-rate * maturity))
