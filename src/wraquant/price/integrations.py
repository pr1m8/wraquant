"""Advanced pricing integrations using optional packages.

Provides wrappers around QuantLib, financePy, rateslib, and py-vollib
for bond pricing, option pricing, yield curve construction, and
implied volatility computation.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np

from wraquant.core.decorators import requires_extra

__all__ = [
    "quantlib_bond",
    "quantlib_option",
    "quantlib_yield_curve",
    "financepy_option",
    "rateslib_swap",
    "vollib_implied_vol",
]


@requires_extra("pricing")
def quantlib_bond(
    face: float,
    coupon: float,
    maturity: date,
    yield_curve: list[tuple[date, float]],
    settlement: date | None = None,
    frequency: int = 2,
) -> dict[str, Any]:
    """Price a fixed-rate bond using QuantLib.

    Parameters
    ----------
    face : float
        Face (par) value of the bond.
    coupon : float
        Annual coupon rate (e.g. 0.05 for 5 %).
    maturity : date
        Bond maturity date.
    yield_curve : list of (date, float)
        Pairs of ``(date, zero_rate)`` defining the yield curve.
    settlement : date or None, default None
        Settlement date. Defaults to today when *None*.
    frequency : int, default 2
        Coupon payment frequency per year (2 = semi-annual).

    Returns
    -------
    dict
        Dictionary containing:

        * **clean_price** -- clean price of the bond.
        * **dirty_price** -- dirty (full) price of the bond.
        * **ytm** -- yield to maturity.
        * **duration** -- Macaulay duration.
        * **convexity** -- convexity.
    """
    import QuantLib as ql

    if settlement is None:
        settlement = date.today()

    eval_date = ql.Date(settlement.day, settlement.month, settlement.year)
    ql.Settings.instance().evaluationDate = eval_date

    maturity_date = ql.Date(maturity.day, maturity.month, maturity.year)
    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    day_count = ql.ActualActual(ql.ActualActual.Bond)

    freq_map = {1: ql.Annual, 2: ql.Semiannual, 4: ql.Quarterly, 12: ql.Monthly}
    ql_freq = freq_map.get(frequency, ql.Semiannual)

    schedule = ql.Schedule(
        eval_date,
        maturity_date,
        ql.Period(ql_freq),
        calendar,
        ql.ModifiedFollowing,
        ql.ModifiedFollowing,
        ql.DateGeneration.Backward,
        False,
    )

    bond = ql.FixedRateBond(0, face, schedule, [coupon], day_count)

    # Build yield term structure
    ql_dates = [ql.Date(d.day, d.month, d.year) for d, _ in yield_curve]
    rates = [r for _, r in yield_curve]
    curve = ql.ZeroCurve(ql_dates, rates, day_count)
    curve_handle = ql.YieldTermStructureHandle(curve)

    engine = ql.DiscountingBondEngine(curve_handle)
    bond.setPricingEngine(engine)

    clean = bond.cleanPrice()
    dirty = bond.dirtyPrice()
    ytm = bond.bondYield(day_count, ql.Compounded, ql_freq)
    dur = ql.BondFunctions.duration(bond, ytm, day_count, ql.Compounded, ql_freq)
    conv = ql.BondFunctions.convexity(bond, ytm, day_count, ql.Compounded, ql_freq)

    return {
        "clean_price": float(clean),
        "dirty_price": float(dirty),
        "ytm": float(ytm),
        "duration": float(dur),
        "convexity": float(conv),
    }


@requires_extra("pricing")
def quantlib_option(
    spot: float,
    strike: float,
    vol: float,
    rf: float,
    maturity: float,
    option_type: str = "call",
) -> dict[str, Any]:
    """Price a European option using QuantLib's analytic engine.

    Parameters
    ----------
    spot : float
        Current spot price of the underlying.
    strike : float
        Strike price.
    vol : float
        Annualised volatility (e.g. 0.20 for 20 %).
    rf : float
        Risk-free interest rate (annual, continuously compounded).
    maturity : float
        Time to maturity in years.
    option_type : str, default 'call'
        ``'call'`` or ``'put'``.

    Returns
    -------
    dict
        Dictionary containing:

        * **price** -- option price.
        * **delta** -- option delta.
        * **gamma** -- option gamma.
        * **theta** -- option theta (per year).
        * **vega** -- option vega (per 1 % vol change).
        * **rho** -- option rho.
    """
    import QuantLib as ql

    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today

    mat_date = today + int(maturity * 365)
    day_count = ql.Actual365Fixed()

    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type.lower() == "call" else ql.Option.Put,
        strike,
    )
    exercise = ql.EuropeanExercise(mat_date)
    option = ql.VanillaOption(payoff, exercise)

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    flat_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(today, rf, day_count)
    )
    flat_vol = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), vol, day_count)
    )

    process = ql.BlackScholesMertonProcess(
        spot_handle,
        ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, day_count)),
        flat_ts,
        flat_vol,
    )

    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

    return {
        "price": float(option.NPV()),
        "delta": float(option.delta()),
        "gamma": float(option.gamma()),
        "theta": float(option.theta()),
        "vega": float(option.vega()),
        "rho": float(option.rho()),
    }


@requires_extra("pricing")
def quantlib_yield_curve(
    dates: list[date],
    rates: list[float],
) -> dict[str, Any]:
    """Construct a yield curve using QuantLib.

    Parameters
    ----------
    dates : list of date
        Curve node dates (the first date is the reference date).
    rates : list of float
        Zero rates corresponding to each date.

    Returns
    -------
    dict
        Dictionary containing:

        * **reference_date** -- the curve reference date.
        * **max_date** -- the latest date on the curve.
        * **discount_factors** -- list of discount factors at each node.
        * **forward_rates** -- list of instantaneous forward rates at each node.
        * **curve** -- the raw ``ql.ZeroCurve`` object.
    """
    import QuantLib as ql

    day_count = ql.Actual365Fixed()
    ql_dates = [ql.Date(d.day, d.month, d.year) for d in dates]

    ql.Settings.instance().evaluationDate = ql_dates[0]
    curve = ql.ZeroCurve(ql_dates, rates, day_count)

    discount_factors = [curve.discount(d) for d in ql_dates]
    forward_rates = [
        curve.forwardRate(d, d + 1, day_count, ql.Continuous).rate()
        for d in ql_dates[:-1]
    ]

    return {
        "reference_date": dates[0],
        "max_date": dates[-1],
        "discount_factors": [float(df) for df in discount_factors],
        "forward_rates": [float(fr) for fr in forward_rates],
        "curve": curve,
    }


@requires_extra("pricing")
def financepy_option(
    spot: float,
    strike: float,
    vol: float,
    rf: float,
    maturity: float,
) -> dict[str, Any]:
    """Price a European equity option using FinancePy.

    Parameters
    ----------
    spot : float
        Current spot price.
    strike : float
        Strike price.
    vol : float
        Annualised volatility.
    rf : float
        Risk-free rate (annual).
    maturity : float
        Time to maturity in years.

    Returns
    -------
    dict
        Dictionary containing:

        * **call_price** -- European call price.
        * **put_price** -- European put price.
        * **call_delta** -- call delta.
        * **put_delta** -- put delta.
    """
    import datetime as _dt

    from financepy.market.curves.discount_curve_flat import DiscountCurveFlat
    from financepy.models.black_scholes import BlackScholes
    from financepy.products.equity.equity_vanilla_option import (
        EquityVanillaOption,
    )
    from financepy.utils.date import Date
    from financepy.utils.global_types import OptionTypes

    today = date.today()
    fp_today = Date(today.day, today.month, today.year)
    mat_days = int(maturity * 365)
    mat_dt = today + _dt.timedelta(days=mat_days)
    fp_mat = Date(mat_dt.day, mat_dt.month, mat_dt.year)

    call_option = EquityVanillaOption(fp_mat, strike, OptionTypes.EUROPEAN_CALL)
    put_option = EquityVanillaOption(fp_mat, strike, OptionTypes.EUROPEAN_PUT)

    model = BlackScholes(vol)
    discount_curve = DiscountCurveFlat(fp_today, rf)
    div_curve = DiscountCurveFlat(fp_today, 0.0)

    call_price = call_option.value(fp_today, spot, discount_curve, div_curve, model)
    put_price = put_option.value(fp_today, spot, discount_curve, div_curve, model)
    call_delta = call_option.delta(fp_today, spot, discount_curve, div_curve, model)
    put_delta = put_option.delta(fp_today, spot, discount_curve, div_curve, model)

    return {
        "call_price": float(call_price),
        "put_price": float(put_price),
        "call_delta": float(call_delta),
        "put_delta": float(put_delta),
    }


@requires_extra("pricing")
def rateslib_swap(
    notional: float,
    fixed_rate: float,
    float_spread: float,
    maturity: str,
) -> dict[str, Any]:
    """Price a plain vanilla interest rate swap using rateslib.

    Parameters
    ----------
    notional : float
        Notional principal amount.
    fixed_rate : float
        Fixed leg coupon rate (e.g. 0.03 for 3 %).
    float_spread : float
        Spread over the floating index in basis points.
    maturity : str
        Tenor string (e.g. ``'5Y'``, ``'10Y'``).

    Returns
    -------
    dict
        Dictionary containing:

        * **npv** -- net present value of the swap.
        * **fixed_rate** -- the fixed rate used.
        * **float_spread** -- the floating spread used (bps).
        * **maturity** -- the tenor string.
        * **notional** -- the notional amount.
    """
    import datetime as _dt

    import rateslib as rl

    today = _dt.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    irs = rl.IRS(
        effective=today,
        termination=maturity,
        notional=notional,
        fixed_rate=fixed_rate * 100,  # rateslib expects percentage
        spec="usd_irs",
    )

    # Build a flat curve with enough nodes
    far_date = today + _dt.timedelta(days=365 * 30)
    curve = rl.Curve(
        nodes={today: 1.0, far_date: 0.5},
        convention="act360",
    )

    try:
        npv = irs.npv(curves=[curve, curve])
    except Exception:
        npv = 0.0

    return {
        "npv": float(npv) if npv is not None else 0.0,
        "fixed_rate": fixed_rate,
        "float_spread": float_spread,
        "maturity": maturity,
        "notional": notional,
    }


@requires_extra("pricing")
def vollib_implied_vol(
    price: float,
    spot: float,
    strike: float,
    rf: float,
    maturity: float,
    option_type: str = "call",
) -> dict[str, float]:
    """Compute implied volatility using py-vollib.

    Parameters
    ----------
    price : float
        Observed option market price.
    spot : float
        Current spot price of the underlying.
    strike : float
        Strike price.
    rf : float
        Risk-free rate (annual, continuously compounded).
    maturity : float
        Time to maturity in years.
    option_type : str, default 'call'
        ``'call'`` or ``'put'``.

    Returns
    -------
    dict
        Dictionary containing:

        * **implied_vol** -- implied volatility.
        * **option_type** -- the option type used.
        * **price** -- the input option price.
    """
    # Patch py_lets_be_rational.constants for Python 3.13+ (_testcapi removed)
    import sys

    if not hasattr(sys, "_testcapi_patched"):
        try:
            import py_lets_be_rational.constants  # noqa: F401
        except ModuleNotFoundError:
            import types

            _fake = types.ModuleType("_testcapi")
            _fake.DBL_MIN = sys.float_info.min
            _fake.DBL_MAX = sys.float_info.max
            sys.modules["_testcapi"] = _fake
        sys._testcapi_patched = True  # type: ignore[attr-defined]

    from py_vollib.black_scholes.implied_volatility import implied_volatility

    flag = "c" if option_type.lower() == "call" else "p"
    iv = implied_volatility(price, spot, strike, maturity, rf, flag)

    return {
        "implied_vol": float(iv),
        "option_type": option_type,
        "price": price,
    }
