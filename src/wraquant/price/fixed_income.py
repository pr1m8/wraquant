"""Fixed income pricing: bonds, yields, duration, and convexity.

Provides present-value based bond pricing, yield-to-maturity solvers,
and duration/convexity risk measures.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "bond_price",
    "bond_yield",
    "duration",
    "modified_duration",
    "convexity",
    "zero_rate",
]


def bond_price(
    face_value: float,
    coupon_rate: float,
    ytm: float,
    periods: int,
    freq: int = 2,
) -> np.float64:
    """Compute the price of a fixed-rate bond as the PV of its cash flows.

    Parameters:
        face_value: Par/face value of the bond.
        coupon_rate: Annual coupon rate (e.g., 0.05 for 5%).
        ytm: Yield to maturity (annualized, e.g., 0.05 for 5%).
        periods: Total number of coupon periods remaining.
        freq: Coupon payments per year (default 2 for semiannual).

    Returns:
        Present value (price) of the bond.

    Example:
        >>> bond_price(1000, 0.05, 0.05, 10, 2)
        1000.0
    """
    coupon = face_value * coupon_rate / freq
    y = ytm / freq
    n = periods

    if abs(y) < 1e-15:
        # Zero yield: simple sum of cash flows
        return np.float64(coupon * n + face_value)

    # PV of annuity (coupons) + PV of face value
    pv_coupons = coupon * (1.0 - (1.0 + y) ** (-n)) / y
    pv_face = face_value / (1.0 + y) ** n

    return np.float64(pv_coupons + pv_face)


def bond_yield(
    price: float,
    face_value: float,
    coupon_rate: float,
    periods: int,
    freq: int = 2,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> np.float64:
    """Compute the yield to maturity (YTM) of a bond via Newton's method.

    Parameters:
        price: Market price of the bond.
        face_value: Par/face value of the bond.
        coupon_rate: Annual coupon rate.
        periods: Total number of coupon periods remaining.
        freq: Coupon payments per year (default 2 for semiannual).
        tol: Convergence tolerance.
        max_iter: Maximum iterations.

    Returns:
        Annualized yield to maturity.

    Raises:
        ValueError: If the solver does not converge.

    Example:
        >>> bond_yield(950, 1000, 0.05, 10, 2)
        0.059...
    """
    coupon = face_value * coupon_rate / freq
    n = periods

    # Initial guess: current yield
    if price > 0:
        y = coupon / price
    else:
        y = 0.05 / freq

    for _ in range(max_iter):
        # Price as function of y (periodic yield)
        if abs(y) < 1e-15:
            pv = coupon * n + face_value
            dpv = 0.0
            for t in range(1, n + 1):
                dpv -= t * coupon
            dpv -= n * face_value
        else:
            discount = (1.0 + y) ** (-n)
            pv = coupon * (1.0 - discount) / y + face_value * discount

            # Derivative of PV w.r.t. y
            dpv = 0.0
            for t in range(1, n + 1):
                dpv -= t * coupon / (1.0 + y) ** (t + 1)
            dpv -= n * face_value / (1.0 + y) ** (n + 1)

        diff = pv - price
        if abs(diff) < tol:
            return np.float64(y * freq)

        if abs(dpv) < 1e-15:
            break

        y = y - diff / dpv

    raise ValueError(f"Bond yield solver did not converge after {max_iter} iterations.")


def duration(
    face_value: float,
    coupon_rate: float,
    ytm: float,
    periods: int,
    freq: int = 2,
) -> np.float64:
    """Compute the Macaulay duration of a fixed-rate bond.

    Parameters:
        face_value: Par/face value of the bond.
        coupon_rate: Annual coupon rate.
        ytm: Yield to maturity (annualized).
        periods: Total number of coupon periods remaining.
        freq: Coupon payments per year (default 2 for semiannual).

    Returns:
        Macaulay duration in years.

    Example:
        >>> duration(1000, 0.05, 0.05, 10, 2)
        4.08...
    """
    coupon = face_value * coupon_rate / freq
    y = ytm / freq
    n = periods

    # Weighted present value of cash flows
    weighted_pv = np.float64(0.0)
    total_pv = np.float64(0.0)

    for t in range(1, n + 1):
        cf = coupon
        if t == n:
            cf += face_value
        pv = cf / (1.0 + y) ** t
        weighted_pv += (t / freq) * pv
        total_pv += pv

    if total_pv == 0.0:
        return np.float64(0.0)

    return np.float64(weighted_pv / total_pv)


def modified_duration(
    face_value: float,
    coupon_rate: float,
    ytm: float,
    periods: int,
    freq: int = 2,
) -> np.float64:
    """Compute the modified duration of a fixed-rate bond.

    Modified duration = Macaulay duration / (1 + ytm/freq).

    Parameters:
        face_value: Par/face value of the bond.
        coupon_rate: Annual coupon rate.
        ytm: Yield to maturity (annualized).
        periods: Total number of coupon periods remaining.
        freq: Coupon payments per year (default 2 for semiannual).

    Returns:
        Modified duration in years.

    Example:
        >>> modified_duration(1000, 0.05, 0.05, 10, 2)
        3.98...
    """
    mac_dur = duration(face_value, coupon_rate, ytm, periods, freq)
    return np.float64(mac_dur / (1.0 + ytm / freq))


def convexity(
    face_value: float,
    coupon_rate: float,
    ytm: float,
    periods: int,
    freq: int = 2,
) -> np.float64:
    """Compute the convexity of a fixed-rate bond.

    Parameters:
        face_value: Par/face value of the bond.
        coupon_rate: Annual coupon rate.
        ytm: Yield to maturity (annualized).
        periods: Total number of coupon periods remaining.
        freq: Coupon payments per year (default 2 for semiannual).

    Returns:
        Convexity measure (in years squared, scaled by freq^2).

    Example:
        >>> convexity(1000, 0.05, 0.05, 10, 2)
        19.4...
    """
    coupon = face_value * coupon_rate / freq
    y = ytm / freq
    n = periods

    price_val = bond_price(face_value, coupon_rate, ytm, periods, freq)

    weighted_sum = np.float64(0.0)
    for t in range(1, n + 1):
        cf = coupon
        if t == n:
            cf += face_value
        pv = cf / (1.0 + y) ** (t + 2)
        weighted_sum += t * (t + 1) * pv

    return np.float64(weighted_sum / (price_val * freq**2))


def zero_rate(
    price: float,
    face_value: float,
    periods: float,
) -> np.float64:
    """Compute the zero coupon rate from a zero coupon bond price.

    Parameters:
        price: Current market price of the zero coupon bond.
        face_value: Par/face value of the bond.
        periods: Time to maturity in years.

    Returns:
        Annualized continuously compounded zero rate.

    Example:
        >>> zero_rate(950, 1000, 1.0)
        0.051...
    """
    if price <= 0.0:
        raise ValueError("Price must be positive.")
    if periods <= 0.0:
        raise ValueError("Periods must be positive.")

    return np.float64(-np.log(price / face_value) / periods)
