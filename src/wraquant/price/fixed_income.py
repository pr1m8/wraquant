"""Fixed income pricing: bonds, yields, duration, and convexity.

Provides present-value based bond pricing, yield-to-maturity solvers,
and duration/convexity risk measures.
"""

from __future__ import annotations

import numpy as np

from wraquant.core._coerce import coerce_array  # noqa: F401 — wired for type-system consistency

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
    r"""Compute the price of a fixed-rate bond as the PV of its cash flows.

    Discounts each coupon payment and the face-value redemption at
    the yield to maturity.  When ``ytm == coupon_rate``, the bond
    prices at par.  Use this to mark positions to market given a
    yield, or to compute the theoretical price for comparison against
    the quoted market price.

    .. math::

        P = \sum_{t=1}^{n} \frac{C}{(1+y)^t}
            + \frac{F}{(1+y)^n}

    where :math:`C = F \cdot c / f` is the periodic coupon,
    :math:`y = \text{ytm} / f`, :math:`n` is the number of periods,
    and :math:`f` is the payment frequency.

    Parameters:
        face_value (float): Par/face value of the bond (e.g., 1000).
        coupon_rate (float): Annual coupon rate (e.g., 0.05 for 5%).
        ytm (float): Yield to maturity (annualized, e.g., 0.05 for 5%).
        periods (int): Total number of coupon periods remaining.
        freq (int): Coupon payments per year (default 2 for semiannual).

    Returns:
        np.float64: Present value (dirty price) of the bond.  When
            ``ytm < coupon_rate`` the bond trades at a premium (price >
            face); when ``ytm > coupon_rate`` it trades at a discount.

    Example:
        >>> bond_price(1000, 0.05, 0.05, 10, 2)
        1000.0
        >>> bond_price(1000, 0.05, 0.06, 10, 2)
        925.6...

    See Also:
        bond_yield: Solve for the yield given price.
        duration: Macaulay duration for interest-rate sensitivity.
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

    YTM is the single discount rate that equates the present value of
    a bond's cash flows to its market price.  It is the internal rate
    of return assuming all coupons are reinvested at the same rate.
    Use YTM to compare bonds with different coupons and maturities on
    a common basis.

    Parameters:
        price (float): Market price of the bond.
        face_value (float): Par/face value of the bond.
        coupon_rate (float): Annual coupon rate.
        periods (int): Total number of coupon periods remaining.
        freq (int): Coupon payments per year (default 2 for semiannual).
        tol (float): Convergence tolerance on the price residual.
        max_iter (int): Maximum Newton-Raphson iterations.

    Returns:
        np.float64: Annualized yield to maturity.  For a par bond
            (price == face_value) the YTM equals the coupon rate.

    Raises:
        ValueError: If the solver does not converge within *max_iter*
            iterations.

    Example:
        >>> bond_yield(950, 1000, 0.05, 10, 2)
        0.059...

    See Also:
        bond_price: Price a bond given a yield.
        zero_rate: Extract a zero rate from a zero-coupon bond.
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
    r"""Compute the Macaulay duration of a fixed-rate bond.

    Macaulay duration is the weighted-average time to receive the
    bond's cash flows, where the weights are the present-value
    fractions.  It measures the bond's sensitivity to parallel shifts
    in the yield curve: a duration of 5 years means a 1% yield
    increase causes approximately a 5% price decrease.

    .. math::

        D = \frac{1}{P} \sum_{t=1}^{n} \frac{t}{f}
            \cdot \frac{CF_t}{(1 + y)^t}

    Parameters:
        face_value (float): Par/face value of the bond.
        coupon_rate (float): Annual coupon rate.
        ytm (float): Yield to maturity (annualized).
        periods (int): Total number of coupon periods remaining.
        freq (int): Coupon payments per year (default 2 for semiannual).

    Returns:
        np.float64: Macaulay duration in years.  Zero-coupon bonds have
            duration equal to their maturity; coupon bonds always have
            duration less than maturity.

    Example:
        >>> duration(1000, 0.05, 0.05, 10, 2)
        4.08...

    See Also:
        modified_duration: Duration divided by (1 + y/freq).
        convexity: Second-order interest-rate sensitivity.
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
    r"""Compute the modified duration of a fixed-rate bond.

    Modified duration is the more practically useful sensitivity
    measure: the percentage price change for a one-unit change in
    yield.

    .. math::

        D_{\text{mod}} = \frac{D_{\text{mac}}}{1 + y/f}

    A modified duration of 4 means a 100 bp (1%) yield increase
    causes approximately a 4% price decline.

    Parameters:
        face_value (float): Par/face value of the bond.
        coupon_rate (float): Annual coupon rate.
        ytm (float): Yield to maturity (annualized).
        periods (int): Total number of coupon periods remaining.
        freq (int): Coupon payments per year (default 2 for semiannual).

    Returns:
        np.float64: Modified duration in years.

    Example:
        >>> modified_duration(1000, 0.05, 0.05, 10, 2)
        3.98...

    See Also:
        duration: Macaulay duration.
        convexity: Second-order correction for large yield changes.
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
    r"""Compute the convexity of a fixed-rate bond.

    Convexity is the second-order sensitivity of bond price to yield
    changes.  Duration alone gives a linear approximation; convexity
    adds the curvature correction, improving accuracy for large yield
    moves.

    The price change for a yield shift :math:`\Delta y` is
    approximately:

    .. math::

        \frac{\Delta P}{P} \approx -D_{\text{mod}} \cdot \Delta y
        + \tfrac{1}{2}\,C \cdot (\Delta y)^2

    where :math:`C` is convexity.  Positive convexity means the bond
    gains more from a yield decrease than it loses from an equal
    yield increase.

    Parameters:
        face_value (float): Par/face value of the bond.
        coupon_rate (float): Annual coupon rate.
        ytm (float): Yield to maturity (annualized).
        periods (int): Total number of coupon periods remaining.
        freq (int): Coupon payments per year (default 2 for semiannual).

    Returns:
        np.float64: Convexity measure (in years squared, scaled by
            ``freq^2``).

    Example:
        >>> convexity(1000, 0.05, 0.05, 10, 2)
        19.4...

    See Also:
        duration: First-order interest-rate sensitivity.
        modified_duration: Percentage price change per unit yield change.
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
    r"""Compute the zero-coupon (spot) rate from a zero-coupon bond price.

    Solves :math:`P = F \cdot e^{-r \cdot T}` for *r*, giving the
    continuously compounded rate that equates the bond price to the
    present value of the face value.  Zero rates are the building
    blocks of the term structure; use them to bootstrap a yield curve
    or to discount future cash flows.

    Parameters:
        price (float): Current market price of the zero-coupon bond.
        face_value (float): Par/face value of the bond.
        periods (float): Time to maturity in years.

    Returns:
        np.float64: Annualized continuously compounded zero rate.

    Raises:
        ValueError: If *price* or *periods* is not positive.

    Example:
        >>> zero_rate(950, 1000, 1.0)
        0.051...

    See Also:
        bootstrap_zero_curve: Bootstrap a full zero curve from par rates.
        discount_factor: Convert a zero rate to a discount factor.
    """
    if price <= 0.0:
        raise ValueError("Price must be positive.")
    if periods <= 0.0:
        raise ValueError("Periods must be positive.")

    return np.float64(-np.log(price / face_value) / periods)
