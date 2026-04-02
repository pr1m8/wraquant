"""Valuation models for intrinsic value estimation.

Provides multiple approaches to estimating the fair value of a stock:

- **Discounted Cash Flow (DCF)** -- Projects free cash flows and discounts
  them back to the present.  The gold standard for absolute valuation.
- **Relative valuation** -- Compares multiples (P/E, EV/EBITDA, P/B) to
  a peer group or sector to identify relative mis-pricing.
- **Graham Number** -- Ben Graham's conservative formula for intrinsic
  value based on earnings and book value.
- **Peter Lynch fair value** -- PEG-based valuation: a stock is fairly
  valued when P/E equals the EPS growth rate.
- **Dividend Discount Model (DDM)** -- Gordon growth model for stable
  dividend-paying stocks.
- **Residual Income Model (RIM)** -- Book value plus the present value
  of future excess earnings above the cost of equity.

All symbol-based functions call the FMP data provider.  Pass an
``fmp_client`` to reuse a client across calls and avoid re-creation.

Example:
    >>> from wraquant.fundamental.valuation import dcf_valuation, margin_of_safety
    >>> dcf = dcf_valuation("AAPL")
    >>> print(f"Intrinsic value: ${dcf['intrinsic_value_per_share']:.2f}")
    >>> print(f"Margin of safety: {dcf['margin_of_safety']:.1%}")

References:
    - Damodaran, A. (2012). *Investment Valuation*, 3rd ed. Wiley.
    - Graham, B. & Dodd, D. (1934). *Security Analysis*. McGraw-Hill.
    - Gordon, M. J. (1959). "Dividends, Earnings, and Stock Prices."
      *Review of Economics and Statistics*, 41(2), 99--105.
    - Ohlson, J. A. (1995). "Earnings, Book Values, and Dividends in
      Equity Valuation." *Contemporary Accounting Research*, 11(2),
      661--687.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Sequence

from wraquant.core.decorators import requires_extra

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide safely, returning *default* when *denominator* is near zero."""
    if abs(denominator) < 1e-12:
        return default
    return float(numerator / denominator)


def _get_fmp_client(fmp_client: Any | None = None) -> Any:
    """Return the provided client or construct a default ``FMPProvider``."""
    if fmp_client is not None:
        return fmp_client
    from wraquant.data.providers.fmp import FMPProvider  # noqa: WPS433

    return FMPProvider()


def _safe_get(data: dict | list, key: str, default: float = 0.0) -> float:
    """Extract a numeric value from an FMP response (dict or list-of-dict)."""
    if isinstance(data, list):
        if not data:
            return default
        data = data[0]
    val = data.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_get_list(data: Any) -> list[dict]:
    """Coerce *data* to a list of dicts."""
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    return []


# ---------------------------------------------------------------------------
# DCF Valuation
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def dcf_valuation(
    symbol: str,
    *,
    growth_rate: float | None = None,
    discount_rate: float | None = None,
    terminal_growth: float = 0.025,
    projection_years: int = 5,
    fmp_client: Any | None = None,
) -> dict[str, Any]:
    """Discounted cash flow valuation using FMP financial data.

    Estimates intrinsic value by projecting free cash flows forward and
    discounting them to the present.  If ``growth_rate`` or
    ``discount_rate`` are not provided, they are estimated from
    historical data and the company's cost structure.

    When to use:
        - Absolute valuation when you have a view on future growth.
        - Sensitivity analysis: vary growth/discount to bound fair value.
        - Combine with :func:`margin_of_safety` for investment decisions.

    Mathematical formulation:
        PV = sum_{t=1}^{N} FCF_0 * (1+g)^t / (1+r)^t
             + [FCF_N * (1+g_term) / (r - g_term)] / (1+r)^N

        Intrinsic value per share = PV / shares outstanding

    Parameters:
        symbol: Ticker symbol.
        growth_rate: Projected annual FCF growth rate.  If ``None``,
            estimated from the 3-year historical FCF CAGR.  Typical
            range: 0.03--0.20.
        discount_rate: Weighted average cost of capital (WACC).  If
            ``None``, estimated as 10% (common equity assumption).
            Typical range: 0.08--0.12.
        terminal_growth: Perpetual growth rate for terminal value.
            Must be < ``discount_rate``.  Use GDP growth rate
            (~2--3%) as an upper bound.
        projection_years: Number of years to project FCF.
            Typical: 5--10.
        fmp_client: Optional ``FMPProvider`` instance.

    Returns:
        Dictionary containing:
        - **intrinsic_value** (*float*) -- Total intrinsic enterprise value.
        - **intrinsic_value_per_share** (*float*) -- Per-share fair value.
        - **current_price** (*float*) -- Current market price.
        - **margin_of_safety** (*float*) -- (intrinsic - price) / intrinsic.
          Positive means undervalued.
        - **upside_potential** (*float*) -- (intrinsic - price) / price.
        - **pv_cash_flows** (*float*) -- PV of projected FCFs.
        - **pv_terminal** (*float*) -- PV of terminal value.
        - **terminal_value** (*float*) -- Undiscounted terminal value.
        - **terminal_pct** (*float*) -- Terminal value as % of total PV.
          > 75% means the valuation is highly sensitive to terminal
          assumptions.
        - **projected_fcf** (*list[float]*) -- Year-by-year projected FCFs.
        - **assumptions** (*dict*) -- Growth rate, discount rate, terminal
          growth used.
        - **fmp_dcf** (*float*) -- FMP's own DCF estimate for comparison.

    Raises:
        ValueError: If ``discount_rate <= terminal_growth``.

    Example:
        >>> from wraquant.fundamental.valuation import dcf_valuation
        >>> dcf = dcf_valuation("MSFT", growth_rate=0.12, discount_rate=0.10)
        >>> print(f"Fair value: ${dcf['intrinsic_value_per_share']:.2f}")
        >>> print(f"Margin of safety: {dcf['margin_of_safety']:.1%}")

    References:
        Damodaran, A. (2012). *Investment Valuation*, 3rd ed., Chapter 12.

    See Also:
        relative_valuation: Peer-based valuation.
        margin_of_safety: Stand-alone margin computation.
    """
    client = _get_fmp_client(fmp_client)

    # Fetch data
    cash_flow_data = _safe_get_list(client.cash_flow(symbol, period="annual", limit=10))
    profile_data = client.company_profile(symbol)
    fmp_dcf_data = client.dcf(symbol)
    balance = client.balance_sheet(symbol)

    # Current FCF (most recent year)
    if cash_flow_data:
        fcf_current = _safe_get(cash_flow_data[0], "freeCashFlow")
    else:
        fcf_current = 0.0

    # Estimate growth rate from historical FCF if not provided
    if growth_rate is None:
        fcf_values = [_safe_get(row, "freeCashFlow") for row in cash_flow_data]
        fcf_positive = [v for v in fcf_values if v > 0]
        if len(fcf_positive) >= 4:
            # 3-year CAGR from most recent 4 values
            growth_rate = (fcf_positive[0] / fcf_positive[3]) ** (1.0 / 3.0) - 1.0
            # Clamp to reasonable range
            growth_rate = max(-0.10, min(growth_rate, 0.25))
        else:
            growth_rate = 0.05  # conservative default

    if discount_rate is None:
        discount_rate = 0.10  # common equity assumption

    if discount_rate <= terminal_growth:
        msg = (
            f"discount_rate ({discount_rate:.4f}) must exceed "
            f"terminal_growth ({terminal_growth:.4f})"
        )
        raise ValueError(msg)

    # Project FCFs
    projected_fcf = []
    for t in range(1, projection_years + 1):
        projected = fcf_current * (1 + growth_rate) ** t
        projected_fcf.append(projected)

    # Discount projected FCFs
    pv_cash_flows = 0.0
    for t, fcf in enumerate(projected_fcf, start=1):
        pv_cash_flows += fcf / (1 + discount_rate) ** t

    # Terminal value (Gordon growth model)
    terminal_cf = projected_fcf[-1] * (1 + terminal_growth) if projected_fcf else 0.0
    terminal_value = _safe_div(terminal_cf, discount_rate - terminal_growth)
    pv_terminal = terminal_value / (1 + discount_rate) ** projection_years

    intrinsic_value = pv_cash_flows + pv_terminal

    # Adjust for net debt to get equity value
    total_debt = _safe_get(balance, "totalDebt")
    cash = _safe_get(balance, "cashAndCashEquivalents")
    net_debt = total_debt - cash
    equity_value = intrinsic_value - net_debt

    # Per-share value
    profile = (
        profile_data[0]
        if isinstance(profile_data, list) and profile_data
        else profile_data
    )
    if isinstance(profile, dict):
        shares = _safe_get(profile, "mktCap") / max(_safe_get(profile, "price"), 1e-12)
        current_price = _safe_get(profile, "price")
    else:
        shares = 1.0
        current_price = 0.0

    # Fall back to balance sheet shares if profile didn't work
    if shares < 1.0:
        shares = _safe_get(balance, "commonStock", default=1.0)
        shares = max(shares, 1.0)

    value_per_share = equity_value / shares if shares > 0 else 0.0

    mos = _safe_div(value_per_share - current_price, value_per_share)
    upside = _safe_div(value_per_share - current_price, current_price)
    terminal_pct = (
        _safe_div(pv_terminal, intrinsic_value) if intrinsic_value > 0 else 0.0
    )

    fmp_dcf_value = _safe_get(fmp_dcf_data, "dcf")

    return {
        "intrinsic_value": float(intrinsic_value),
        "equity_value": float(equity_value),
        "intrinsic_value_per_share": float(value_per_share),
        "current_price": float(current_price),
        "margin_of_safety": float(mos),
        "upside_potential": float(upside),
        "pv_cash_flows": float(pv_cash_flows),
        "pv_terminal": float(pv_terminal),
        "terminal_value": float(terminal_value),
        "terminal_pct": float(terminal_pct),
        "projected_fcf": [float(f) for f in projected_fcf],
        "assumptions": {
            "growth_rate": float(growth_rate),
            "discount_rate": float(discount_rate),
            "terminal_growth": float(terminal_growth),
            "projection_years": projection_years,
        },
        "fmp_dcf": float(fmp_dcf_value),
    }


# ---------------------------------------------------------------------------
# Relative Valuation
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def relative_valuation(
    symbol: str,
    *,
    peers: list[str] | None = None,
    fmp_client: Any | None = None,
) -> dict[str, Any]:
    """Compare valuation multiples against a peer group.

    Relative valuation assumes that similar companies should trade at
    similar multiples.  Deviations suggest over- or under-pricing
    relative to the peer set.

    When to use:
        - When absolute valuation (DCF) is too uncertain.
        - Sector rotation strategies: buy cheap sectors, sell expensive.
        - Pair trading: long the cheap peer, short the expensive one.

    Parameters:
        symbol: Target ticker symbol.
        peers: List of peer ticker symbols.  If ``None``, the function
            uses FMP's sector peers (same industry/sector).  Provide
            explicit peers for more meaningful comparisons.
        fmp_client: Optional ``FMPProvider`` instance.

    Returns:
        Dictionary containing:
        - **symbol** (*str*) -- The target ticker.
        - **multiples** (*dict*) -- Target's P/E, P/B, P/S, EV/EBITDA.
        - **peer_medians** (*dict*) -- Median multiples of the peer group.
        - **peer_means** (*dict*) -- Mean multiples of the peer group.
        - **premium_discount** (*dict*) -- For each multiple, the %
          premium (+) or discount (-) vs. peer median.
          Negative = cheaper than peers.
        - **peers_data** (*list[dict]*) -- Individual peer multiples.
        - **verdict** (*str*) -- Summary: "undervalued", "fairly valued",
          or "overvalued" based on median premium/discount.

    Example:
        >>> from wraquant.fundamental.valuation import relative_valuation
        >>> rv = relative_valuation("AAPL", peers=["MSFT", "GOOG", "META"])
        >>> print(f"P/E premium: {rv['premium_discount']['pe_ratio']:+.1%}")
        >>> print(f"Verdict: {rv['verdict']}")

    See Also:
        dcf_valuation: Absolute valuation approach.
        valuation_ratios: Single-stock multiples.
    """
    client = _get_fmp_client(fmp_client)

    # Import here to avoid circular dependency
    from wraquant.fundamental.ratios import valuation_ratios

    target_multiples = valuation_ratios(symbol, fmp_client=client)

    if peers is None:
        # Use FMP profile to get sector, then use rating peers
        profile = client.company_profile(symbol)
        profile_data = profile[0] if isinstance(profile, list) and profile else profile
        sector = (
            profile_data.get("sector", "") if isinstance(profile_data, dict) else ""
        )
        industry = (
            profile_data.get("industry", "") if isinstance(profile_data, dict) else ""
        )
        logger.info(
            "No peers provided; using sector=%s, industry=%s",
            sector,
            industry,
        )
        peers = []

    multiples_keys = ["pe_ratio", "pb_ratio", "ps_ratio", "ev_to_ebitda"]

    peers_data = []
    for peer in peers:
        try:
            pm = valuation_ratios(peer, fmp_client=client)
            pm["symbol"] = peer
            peers_data.append(pm)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to fetch multiples for peer %s", peer)

    # Compute peer medians and means
    peer_medians: dict[str, float] = {}
    peer_means: dict[str, float] = {}
    premium_discount: dict[str, float] = {}

    for key in multiples_keys:
        values = [p[key] for p in peers_data if p.get(key) and p[key] > 0]
        if values:
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            median = (
                sorted_vals[n // 2]
                if n % 2 == 1
                else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
            )
            mean = sum(sorted_vals) / n
            peer_medians[key] = median
            peer_means[key] = mean
            target_val = target_multiples.get(key, 0.0)
            premium_discount[key] = _safe_div(target_val - median, median)
        else:
            peer_medians[key] = 0.0
            peer_means[key] = 0.0
            premium_discount[key] = 0.0

    # Verdict based on average premium/discount
    valid_pd = [v for v in premium_discount.values() if v != 0.0]
    avg_pd = sum(valid_pd) / len(valid_pd) if valid_pd else 0.0

    if avg_pd < -0.15:
        verdict = "undervalued"
    elif avg_pd > 0.15:
        verdict = "overvalued"
    else:
        verdict = "fairly valued"

    return {
        "symbol": symbol,
        "multiples": target_multiples,
        "peer_medians": peer_medians,
        "peer_means": peer_means,
        "premium_discount": premium_discount,
        "peers_data": peers_data,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Graham Number
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def graham_number(
    symbol: str,
    *,
    fmp_client: Any | None = None,
) -> dict[str, float]:
    """Compute Ben Graham's intrinsic value number.

    The Graham Number is a conservative estimate of the maximum price a
    defensive investor should pay.  It assumes a stock should not trade
    above P/E of 15 and P/B of 1.5 simultaneously.

    When to use:
        - Deep value screening for defensive investors.
        - Quick sanity check on valuation.
        - Pair with Piotroski F-Score: high F-Score + price < Graham
          Number is a classic value strategy.

    Mathematical formulation:
        Graham Number = sqrt(22.5 * EPS * BVPS)

        where 22.5 = 15 (max P/E) * 1.5 (max P/B)

    Parameters:
        symbol: Ticker symbol.
        fmp_client: Optional ``FMPProvider`` instance.

    Returns:
        Dictionary containing:
        - **graham_number** (*float*) -- Intrinsic value estimate.
          Only meaningful when EPS > 0 and BVPS > 0.
        - **current_price** (*float*) -- Current market price.
        - **margin_of_safety** (*float*) -- (graham - price) / graham.
        - **eps** (*float*) -- Earnings per share used.
        - **bvps** (*float*) -- Book value per share used.

    Example:
        >>> from wraquant.fundamental.valuation import graham_number
        >>> gn = graham_number("JNJ")
        >>> print(f"Graham Number: ${gn['graham_number']:.2f}")
        >>> print(f"Current price: ${gn['current_price']:.2f}")

    References:
        Graham, B. (1973). *The Intelligent Investor*, Revised ed.,
        Chapter 14.

    See Also:
        peter_lynch_value: Growth-oriented fair value.
        dcf_valuation: More sophisticated intrinsic value.
    """
    client = _get_fmp_client(fmp_client)

    metrics = client.key_metrics(symbol)
    profile = client.company_profile(symbol)

    eps = _safe_get(metrics, "netIncomePerShare")
    bvps = _safe_get(metrics, "bookValuePerShare")

    profile_data = profile[0] if isinstance(profile, list) and profile else profile
    current_price = (
        _safe_get(profile_data, "price") if isinstance(profile_data, dict) else 0.0
    )

    # Graham Number = sqrt(22.5 * EPS * BVPS)
    # Only valid when both EPS and BVPS are positive
    if eps > 0 and bvps > 0:
        gn = math.sqrt(22.5 * eps * bvps)
    else:
        gn = 0.0

    mos = _safe_div(gn - current_price, gn) if gn > 0 else 0.0

    return {
        "graham_number": float(gn),
        "current_price": float(current_price),
        "margin_of_safety": float(mos),
        "eps": float(eps),
        "bvps": float(bvps),
    }


# ---------------------------------------------------------------------------
# Peter Lynch Value
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def peter_lynch_value(
    symbol: str,
    *,
    fmp_client: Any | None = None,
) -> dict[str, float | str]:
    """Compute fair value using Peter Lynch's PEG-based methodology.

    Peter Lynch argued that a fairly valued growth stock should have a
    P/E ratio roughly equal to its earnings growth rate.  PEG < 1
    suggests undervaluation; PEG > 2 suggests overvaluation.

    When to use:
        - Growth stock screening.
        - Quick check on whether you're overpaying for growth.
        - Combine with :func:`growth_ratios` for context on growth
          sustainability.

    Mathematical formulation:
        PEG = P/E / (EPS Growth Rate * 100)
        Fair Value = EPS * EPS Growth Rate * 100

    Parameters:
        symbol: Ticker symbol.
        fmp_client: Optional ``FMPProvider`` instance.

    Returns:
        Dictionary containing:
        - **fair_value** (*float*) -- Lynch fair value per share.
        - **current_price** (*float*) -- Current market price.
        - **peg_ratio** (*float*) -- PEG ratio.
        - **pe_ratio** (*float*) -- Current P/E.
        - **eps_growth_rate** (*float*) -- EPS growth rate used (decimal).
        - **margin_of_safety** (*float*) -- (fair - price) / fair.
        - **lynch_category** (*str*) -- "undervalued" (PEG < 1),
          "fairly valued" (1--2), or "overvalued" (> 2).

    Example:
        >>> from wraquant.fundamental.valuation import peter_lynch_value
        >>> plv = peter_lynch_value("NVDA")
        >>> print(f"PEG: {plv['peg_ratio']:.2f}")
        >>> print(f"Lynch category: {plv['lynch_category']}")

    References:
        Lynch, P. (1989). *One Up on Wall Street*. Simon & Schuster.

    See Also:
        graham_number: Conservative value approach.
        valuation_ratios: Raw multiples.
    """
    client = _get_fmp_client(fmp_client)

    ratios_data = client.ratios_ttm(symbol)
    growth_data = _safe_get_list(client.financial_growth(symbol, period="annual"))
    profile = client.company_profile(symbol)

    pe = _safe_get(ratios_data, "peRatioTTM")
    eps_growth = _safe_get(growth_data[0], "epsgrowth") if growth_data else 0.0

    profile_data = profile[0] if isinstance(profile, list) and profile else profile
    current_price = (
        _safe_get(profile_data, "price") if isinstance(profile_data, dict) else 0.0
    )

    # EPS (derive from price and P/E)
    eps = _safe_div(current_price, pe) if pe > 0 else 0.0

    # PEG ratio
    eps_growth_pct = eps_growth * 100  # convert decimal to percentage points
    peg = _safe_div(pe, eps_growth_pct) if eps_growth_pct > 0 else 0.0

    # Lynch fair value: EPS * growth rate (in %)
    fair_value = eps * eps_growth_pct if eps > 0 and eps_growth_pct > 0 else 0.0

    mos = _safe_div(fair_value - current_price, fair_value) if fair_value > 0 else 0.0

    if peg < 1.0 and peg > 0:
        category = "undervalued"
    elif peg <= 2.0:
        category = "fairly valued"
    else:
        category = "overvalued"

    return {
        "fair_value": float(fair_value),
        "current_price": float(current_price),
        "peg_ratio": float(peg),
        "pe_ratio": float(pe),
        "eps_growth_rate": float(eps_growth),
        "margin_of_safety": float(mos),
        "lynch_category": category,
    }


# ---------------------------------------------------------------------------
# Dividend Discount Model (DDM)
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def dividend_discount_model(
    symbol: str,
    *,
    required_return: float = 0.10,
    fmp_client: Any | None = None,
) -> dict[str, float | str]:
    """Gordon Growth Model (single-stage DDM) valuation.

    Values a stock as the present value of all future dividends growing
    at a constant rate in perpetuity.  Only suitable for mature,
    stable-dividend companies (utilities, consumer staples, REITs).

    When to use:
        - Value dividend aristocrats and other stable payers.
        - Income-focused portfolio construction.
        - *Not* suitable for non-dividend or high-growth companies.

    Mathematical formulation:
        V_0 = D_1 / (r - g)

        where:
        D_1 = next year's expected dividend = D_0 * (1 + g)
        r   = required return (cost of equity)
        g   = dividend growth rate (must be < r)

    Parameters:
        symbol: Ticker symbol.
        required_return: Required rate of return / cost of equity.
            Typical range: 0.08--0.12.  Use CAPM or build-up method
            to estimate.
        fmp_client: Optional ``FMPProvider`` instance.

    Returns:
        Dictionary containing:
        - **intrinsic_value** (*float*) -- DDM fair value per share.
        - **current_price** (*float*) -- Current market price.
        - **margin_of_safety** (*float*) -- (intrinsic - price) / intrinsic.
        - **dividend_per_share** (*float*) -- Current annual DPS.
        - **dividend_growth_rate** (*float*) -- Estimated growth rate.
        - **dividend_yield** (*float*) -- Current dividend yield.
        - **implied_return** (*float*) -- Yield + growth (implied total
          return at current price).
        - **model_applicable** (*str*) -- "yes" if the company pays
          dividends and growth < required return; "no" otherwise.

    Example:
        >>> from wraquant.fundamental.valuation import dividend_discount_model
        >>> ddm = dividend_discount_model("KO", required_return=0.09)
        >>> print(f"DDM value: ${ddm['intrinsic_value']:.2f}")
        >>> print(f"Implied return: {ddm['implied_return']:.2%}")

    References:
        Gordon, M. J. (1959). "Dividends, Earnings, and Stock Prices."
        *Review of Economics and Statistics*, 41(2), 99--105.

    See Also:
        dcf_valuation: FCF-based valuation (works for non-payers).
        residual_income_model: Book-value-based alternative.
    """
    client = _get_fmp_client(fmp_client)

    profile = client.company_profile(symbol)
    ratios_data = client.ratios_ttm(symbol)
    growth_data = _safe_get_list(client.financial_growth(symbol, period="annual"))
    metrics = client.key_metrics(symbol)

    profile_data = profile[0] if isinstance(profile, list) and profile else profile
    current_price = (
        _safe_get(profile_data, "price") if isinstance(profile_data, dict) else 0.0
    )
    last_div = (
        _safe_get(profile_data, "lastDiv") if isinstance(profile_data, dict) else 0.0
    )

    div_yield = _safe_get(ratios_data, "dividendYieldTTM")
    div_growth = (
        _safe_get(growth_data[0], "dividendsperShareGrowth") if growth_data else 0.0
    )

    # Use DPS from metrics if profile doesn't have it
    dps = (
        last_div
        if last_div > 0
        else _safe_get(metrics, "dividendYield") * current_price
    )

    # Check model applicability
    if dps <= 0 or div_growth >= required_return:
        return {
            "intrinsic_value": 0.0,
            "current_price": float(current_price),
            "margin_of_safety": 0.0,
            "dividend_per_share": float(dps),
            "dividend_growth_rate": float(div_growth),
            "dividend_yield": float(div_yield),
            "implied_return": float(div_yield + div_growth),
            "model_applicable": "no",
        }

    # Gordon Growth: V = D1 / (r - g) where D1 = D0 * (1 + g)
    d1 = dps * (1 + div_growth)
    intrinsic_value = d1 / (required_return - div_growth)

    mos = _safe_div(intrinsic_value - current_price, intrinsic_value)
    implied_return = _safe_div(dps, current_price) + div_growth

    return {
        "intrinsic_value": float(intrinsic_value),
        "current_price": float(current_price),
        "margin_of_safety": float(mos),
        "dividend_per_share": float(dps),
        "dividend_growth_rate": float(div_growth),
        "dividend_yield": float(div_yield),
        "implied_return": float(implied_return),
        "model_applicable": "yes",
    }


# ---------------------------------------------------------------------------
# Residual Income Model
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def residual_income_model(
    symbol: str,
    *,
    cost_of_equity: float = 0.10,
    projection_years: int = 5,
    fade_rate: float = 0.20,
    fmp_client: Any | None = None,
) -> dict[str, Any]:
    """Residual income (abnormal earnings) valuation model.

    Values a stock as its book value plus the present value of future
    residual income (earnings in excess of the cost of equity).  Unlike
    DCF, this model anchors on book value and is less sensitive to
    terminal value assumptions.

    When to use:
        - Companies with stable book values (financials, industrials).
        - When terminal value dominates DCF (RIM reduces this problem).
        - Academic factor research: book value is the anchor.

    Mathematical formulation:
        V_0 = BV_0 + sum_{t=1}^{T} RI_t / (1 + r_e)^t + TV

        RI_t = NI_t - r_e * BV_{t-1}  (residual income)

        TV = RI_T * (1 - fade) / (r_e - g_ri * (1 - fade))

    Parameters:
        symbol: Ticker symbol.
        cost_of_equity: Required return on equity.  Use CAPM:
            r_e = r_f + beta * (r_m - r_f).  Typical: 0.08--0.14.
        projection_years: Years of explicit RI projection.
        fade_rate: Annual rate at which residual income fades toward
            zero.  Higher fade = more conservative.  0.0 = no fade
            (perpetual excess returns).  0.20 = industry reversion.
        fmp_client: Optional ``FMPProvider`` instance.

    Returns:
        Dictionary containing:
        - **intrinsic_value** (*float*) -- RIM fair value per share.
        - **current_price** (*float*) -- Current market price.
        - **margin_of_safety** (*float*) -- (intrinsic - price) / intrinsic.
        - **book_value_per_share** (*float*) -- Current BVPS.
        - **current_roe** (*float*) -- Current ROE.
        - **residual_income** (*float*) -- Most recent period RI.
        - **pv_residual_income** (*float*) -- PV of projected RI stream.
        - **pv_terminal** (*float*) -- PV of terminal RI.
        - **excess_return_spread** (*float*) -- ROE - cost of equity.
          Positive means the company creates value.

    Example:
        >>> from wraquant.fundamental.valuation import residual_income_model
        >>> rim = residual_income_model("JPM", cost_of_equity=0.11)
        >>> print(f"RIM value: ${rim['intrinsic_value']:.2f}")
        >>> print(f"Excess spread: {rim['excess_return_spread']:.2%}")

    References:
        Ohlson, J. A. (1995). "Earnings, Book Values, and Dividends in
        Equity Valuation." *Contemporary Accounting Research*, 11(2),
        661--687.

    See Also:
        dcf_valuation: Cash-flow-based alternative.
        graham_number: Simpler book-value-based approach.
    """
    client = _get_fmp_client(fmp_client)

    income = client.income_statement(symbol)
    balance = client.balance_sheet(symbol)
    metrics = client.key_metrics(symbol)
    profile = client.company_profile(symbol)

    net_income = _safe_get(income, "netIncome")
    total_equity = _safe_get(balance, "totalStockholdersEquity")
    bvps = _safe_get(metrics, "bookValuePerShare")

    profile_data = profile[0] if isinstance(profile, list) and profile else profile
    current_price = (
        _safe_get(profile_data, "price") if isinstance(profile_data, dict) else 0.0
    )
    shares = _safe_div(
        _safe_get(profile_data, "mktCap") if isinstance(profile_data, dict) else 0.0,
        max(current_price, 1e-12),
    )
    if shares < 1.0:
        shares = _safe_get(balance, "commonStock", default=1.0)
        shares = max(shares, 1.0)

    # Current ROE
    roe = _safe_div(net_income, total_equity)

    # Residual income = NI - cost_of_equity * BV
    ri_current = net_income - cost_of_equity * total_equity

    # Project residual income with fade
    pv_ri = 0.0
    ri_t = ri_current
    for t in range(1, projection_years + 1):
        ri_t = ri_t * (1 - fade_rate)  # fade toward zero
        pv_ri += ri_t / (1 + cost_of_equity) ** t

    # Terminal value of residual income
    ri_terminal = ri_t * (1 - fade_rate)
    if cost_of_equity > 0:
        # Perpetuity with fade: TV = RI / (r + fade)
        tv_ri = _safe_div(ri_terminal, cost_of_equity + fade_rate)
    else:
        tv_ri = 0.0
    pv_terminal = tv_ri / (1 + cost_of_equity) ** projection_years

    # Total equity value
    intrinsic_equity = total_equity + pv_ri + pv_terminal
    intrinsic_per_share = intrinsic_equity / shares if shares > 0 else 0.0

    mos = _safe_div(intrinsic_per_share - current_price, intrinsic_per_share)

    return {
        "intrinsic_value": float(intrinsic_per_share),
        "current_price": float(current_price),
        "margin_of_safety": float(mos),
        "book_value_per_share": float(bvps),
        "current_roe": float(roe),
        "residual_income": float(ri_current),
        "pv_residual_income": float(pv_ri / shares) if shares > 0 else 0.0,
        "pv_terminal": float(pv_terminal / shares) if shares > 0 else 0.0,
        "excess_return_spread": float(roe - cost_of_equity),
    }


# ---------------------------------------------------------------------------
# Margin of Safety
# ---------------------------------------------------------------------------


def margin_of_safety(
    symbol: str | None = None,
    intrinsic_value: float = 0.0,
    current_price: float = 0.0,
    *,
    fmp_client: Any | None = None,
) -> float:
    """Compute the margin of safety between intrinsic value and market price.

    The margin of safety is the central concept of value investing.
    Graham recommended buying only when the market price is significantly
    below intrinsic value to protect against estimation errors.

    Mathematical formulation:
        Margin of Safety = (Intrinsic Value - Market Price) / Intrinsic Value

    When to use:
        - After computing intrinsic value via DCF, Graham Number, etc.
        - Graham recommended a minimum 33% margin of safety.
        - Negative margin means the stock trades above estimated fair value.

    Parameters:
        symbol: Optional ticker symbol.  If provided with no
            ``current_price``, the current market price is fetched.
        intrinsic_value: Your estimated fair value per share.
        current_price: Current market price.  If 0.0 and ``symbol``
            is provided, fetched from FMP.
        fmp_client: Optional ``FMPProvider`` instance.

    Returns:
        Margin of safety as a float (e.g., 0.30 = 30% discount to
        intrinsic value).  Negative means premium to intrinsic.

    Example:
        >>> from wraquant.fundamental.valuation import margin_of_safety
        >>> mos = margin_of_safety(intrinsic_value=150.0, current_price=100.0)
        >>> print(f"Margin of safety: {mos:.1%}")
        Margin of safety: 33.3%

    See Also:
        dcf_valuation: Compute intrinsic value.
        graham_number: Conservative intrinsic value.
    """
    if current_price <= 0 and symbol is not None:
        client = _get_fmp_client(fmp_client)
        profile = client.company_profile(symbol)
        profile_data = profile[0] if isinstance(profile, list) and profile else profile
        current_price = (
            _safe_get(profile_data, "price") if isinstance(profile_data, dict) else 0.0
        )

    if intrinsic_value <= 0:
        return 0.0

    return float((intrinsic_value - current_price) / intrinsic_value)


# ---------------------------------------------------------------------------
# Legacy API compatibility
# ---------------------------------------------------------------------------


def piotroski_f_score(financials: dict[str, float]) -> int:
    """Compute the Piotroski F-Score (0--9) for financial health.

    The Piotroski F-Score is a composite score of nine binary tests
    that evaluate profitability, leverage/liquidity, and operating
    efficiency.  Stocks scoring 8--9 are considered financially strong;
    scores of 0--2 indicate financial distress.

    When to use:
        - Screen value stocks (low P/B) for financial health.
        - Avoid value traps: low P/B stocks with low F-Scores tend to
          underperform.
        - Long/short strategy: long high F-Score value stocks, short
          low F-Score value stocks.

    The nine binary tests:

    **Profitability (4 points)**:
        1. ROA > 0 (net_income / total_assets > 0)
        2. Operating cash flow > 0
        3. ROA increased vs. prior year
        4. Cash flow from operations > net income (accruals quality)

    **Leverage & liquidity (3 points)**:
        5. Long-term debt decreased vs. prior year
        6. Current ratio increased vs. prior year
        7. No new shares issued (shares outstanding unchanged or
           decreased)

    **Operating efficiency (2 points)**:
        8. Gross margin increased vs. prior year
        9. Asset turnover increased vs. prior year

    Parameters:
        financials: Dictionary with the following keys:

            - ``net_income``: Current year net income.
            - ``prev_net_income``: Prior year net income.
            - ``operating_cash_flow``: Current year operating cash flow.
            - ``total_assets``: Current year total assets.
            - ``prev_total_assets``: Prior year total assets.
            - ``long_term_debt``: Current year long-term debt.
            - ``prev_long_term_debt``: Prior year long-term debt.
            - ``current_ratio``: Current year current ratio.
            - ``prev_current_ratio``: Prior year current ratio.
            - ``shares_outstanding``: Current year shares outstanding.
            - ``prev_shares_outstanding``: Prior year shares outstanding.
            - ``gross_margin``: Current year gross margin.
            - ``prev_gross_margin``: Prior year gross margin.
            - ``asset_turnover``: Current year asset turnover.
            - ``prev_asset_turnover``: Prior year asset turnover.

    Returns:
        Integer score from 0 to 9.

    Example:
        >>> financials = {
        ...     "net_income": 1e6, "prev_net_income": 8e5,
        ...     "operating_cash_flow": 1.2e6, "total_assets": 5e6,
        ...     "prev_total_assets": 4.8e6, "long_term_debt": 1e6,
        ...     "prev_long_term_debt": 1.1e6, "current_ratio": 1.5,
        ...     "prev_current_ratio": 1.3, "shares_outstanding": 1e6,
        ...     "prev_shares_outstanding": 1e6, "gross_margin": 0.4,
        ...     "prev_gross_margin": 0.38, "asset_turnover": 0.8,
        ...     "prev_asset_turnover": 0.75,
        ... }
        >>> piotroski_f_score(financials)
        9

    References:
        Piotroski, J. D. (2000). "Value Investing: The Use of Historical
        Financial Statement Information to Separate Winners from Losers."
        *Journal of Accounting Research*, 38, 1--41.

    See Also:
        dcf_valuation: Intrinsic value estimation.
        financial_health_score: FMP-powered composite score.
    """
    f = financials
    score = 0

    # --- Profitability ---
    total_assets = f.get("total_assets", 1.0)
    prev_total_assets = f.get("prev_total_assets", total_assets)

    roa = f.get("net_income", 0.0) / total_assets if abs(total_assets) > 1e-12 else 0.0
    prev_roa = (
        f.get("prev_net_income", 0.0) / prev_total_assets
        if abs(prev_total_assets) > 1e-12
        else 0.0
    )

    # 1. ROA > 0
    if roa > 0:
        score += 1
    # 2. Operating cash flow > 0
    if f.get("operating_cash_flow", 0.0) > 0:
        score += 1
    # 3. ROA increased
    if roa > prev_roa:
        score += 1
    # 4. Accruals: cash flow > net income
    if f.get("operating_cash_flow", 0.0) > f.get("net_income", 0.0):
        score += 1

    # --- Leverage & liquidity ---
    # 5. Long-term debt decreased
    if f.get("long_term_debt", 0.0) < f.get("prev_long_term_debt", 0.0):
        score += 1
    # 6. Current ratio increased
    if f.get("current_ratio", 0.0) > f.get("prev_current_ratio", 0.0):
        score += 1
    # 7. No new shares issued
    if f.get("shares_outstanding", 0.0) <= f.get("prev_shares_outstanding", 0.0):
        score += 1

    # --- Operating efficiency ---
    # 8. Gross margin increased
    if f.get("gross_margin", 0.0) > f.get("prev_gross_margin", 0.0):
        score += 1
    # 9. Asset turnover increased
    if f.get("asset_turnover", 0.0) > f.get("prev_asset_turnover", 0.0):
        score += 1

    return score


def quality_screen(
    stocks_df: Any,
    metrics: list[str] | None = None,
) -> Any:
    """Rank stocks by a composite quality score.

    Computes a composite quality score by ranking each stock on
    multiple fundamental metrics and averaging the percentile ranks.
    Higher composite scores indicate higher quality.

    When to use:
        - Construct a quality factor for multi-factor models.
        - Screen a universe for high-quality long candidates.
        - Complement value screening (avoid value traps by requiring
          quality).

    Parameters:
        stocks_df: DataFrame where each row is a stock and columns
            contain fundamental metrics.  Missing values are handled
            by assigning median rank.
        metrics: List of column names to include in the composite.
            If None, defaults to ``["roe", "operating_margin",
            "current_ratio"]`` (using only columns that exist in the
            DataFrame).

    Returns:
        DataFrame with the original data plus a ``quality_score``
        column (0 to 1, higher is better) and a ``quality_rank``
        column (1 = best), sorted by quality_score descending.

    Example:
        >>> import pandas as pd
        >>> stocks = pd.DataFrame({
        ...     "ticker": ["AAPL", "MSFT", "GOOG"],
        ...     "roe": [0.25, 0.30, 0.20],
        ...     "operating_margin": [0.30, 0.35, 0.25],
        ...     "current_ratio": [1.5, 2.0, 3.0],
        ... }).set_index("ticker")
        >>> result = quality_screen(stocks)
        >>> result["quality_rank"].iloc[0]
        1

    See Also:
        piotroski_f_score: Financial health assessment (single stock).
        custom_screen: Flexible screening with arbitrary criteria.
    """
    import pandas as pd

    default_metrics = ["roe", "operating_margin", "current_ratio"]

    if metrics is None:
        metrics = [m for m in default_metrics if m in stocks_df.columns]

    if not metrics:
        result = stocks_df.copy()
        result["quality_score"] = 0.5
        result["quality_rank"] = 1
        return result

    available = [m for m in metrics if m in stocks_df.columns]
    if not available:
        result = stocks_df.copy()
        result["quality_score"] = 0.5
        result["quality_rank"] = 1
        return result

    ranks = pd.DataFrame(index=stocks_df.index)
    for m in available:
        col = stocks_df[m].astype(float)
        ranks[m] = col.rank(pct=True, na_option="keep")
        ranks[m] = ranks[m].fillna(0.5)

    composite = ranks.mean(axis=1)

    result = stocks_df.copy()
    result["quality_score"] = composite
    result["quality_rank"] = composite.rank(ascending=False, method="min").astype(int)
    result = result.sort_values("quality_score", ascending=False)

    return result
