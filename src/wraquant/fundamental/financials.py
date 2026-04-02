"""Financial statement analysis and quality assessment.

Provides deep analysis of income statements, balance sheets, and cash
flow statements -- going beyond raw ratios to uncover trends, quality
signals, and composite health scores.

This module answers questions that simple ratio snapshots cannot:
- Are margins expanding or contracting?
- Is earnings quality high (cash-backed) or low (accruals-driven)?
- What does the balance sheet composition tell us about risk?
- Is the company's overall financial health improving or deteriorating?

All functions call the FMP data provider for financial statement data.
Pass an ``fmp_client`` to reuse a single client across multiple calls.

Example:
    >>> from wraquant.fundamental.financials import financial_health_score
    >>> health = financial_health_score("AAPL")
    >>> print(f"Health score: {health['total_score']}/100")
    >>> print(f"Category: {health['category']}")

References:
    - Sloan, R. G. (1996). "Do Stock Prices Fully Reflect Information
      in Accruals and Cash Flows about Future Earnings?" *The Accounting
      Review*, 71(3), 289--315.
    - Beneish, M. D. (1999). "The Detection of Earnings Manipulation."
      *Financial Analysts Journal*, 55(5), 24--36.
    - Palepu, K. G. & Healy, P. M. (2013). *Business Analysis and
      Valuation*, 5th edition. Cengage.
"""

from __future__ import annotations

import logging
from typing import Any

from wraquant.core.decorators import requires_extra

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide safely, returning *default* on zero/near-zero denominator."""
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
    """Extract a numeric value from an FMP response dict/list."""
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


def _safe_get_str(data: dict | list, key: str, default: str = "") -> str:
    """Extract a string value from an FMP response dict/list."""
    if isinstance(data, list):
        if not data:
            return default
        data = data[0]
    val = data.get(key)
    return str(val) if val is not None else default


def _safe_get_list(data: Any) -> list[dict]:
    """Coerce *data* to a list of dicts."""
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    return []


def _pct_change(current: float, previous: float) -> float:
    """Compute percentage change, handling zero denominators."""
    return _safe_div(current - previous, abs(previous))


# ---------------------------------------------------------------------------
# Income Statement Analysis
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def income_analysis(
    symbol: str,
    *,
    period: str = "annual",
    limit: int = 5,
    fmp_client: Any | None = None,
) -> dict[str, Any]:
    """Analyse income statement trends over multiple periods.

    Goes beyond single-period ratios to reveal the *trajectory* of
    revenue, margins, and bottom-line profitability.  Trend analysis
    is critical because a company with a 20% margin that is declining
    is very different from one with a 15% margin that is expanding.

    Parameters:
        symbol: Ticker symbol.
        period: ``"annual"`` or ``"quarter"``.
        limit: Number of historical periods to analyse.
        fmp_client: Optional ``FMPProvider`` instance.

    Returns:
        Dictionary containing:
        - **revenue** (*list[float]*) -- Revenue by period (most recent first).
        - **revenue_growth** (*list[float]*) -- YoY revenue growth rates.
        - **gross_margin** (*list[float]*) -- Gross margin by period.
        - **operating_margin** (*list[float]*) -- Operating margin by period.
        - **net_margin** (*list[float]*) -- Net margin by period.
        - **ebitda_margin** (*list[float]*) -- EBITDA margin by period.
        - **margin_trend** (*str*) -- "expanding", "contracting", or "stable"
          based on operating margin trajectory.
        - **revenue_cagr_3y** (*float*) -- 3-year revenue CAGR.
        - **revenue_cagr_5y** (*float*) -- 5-year revenue CAGR.
        - **eps** (*list[float]*) -- EPS by period.
        - **dates** (*list[str]*) -- Period end dates.
        - **periods_analysed** (*int*) -- Number of periods included.

    Example:
        >>> from wraquant.fundamental.financials import income_analysis
        >>> inc = income_analysis("MSFT")
        >>> print(f"Revenue CAGR (3Y): {inc['revenue_cagr_3y']:.1%}")
        >>> print(f"Margin trend: {inc['margin_trend']}")

    See Also:
        balance_sheet_analysis: Asset/liability composition trends.
        cash_flow_analysis: Cash flow quality and FCF trends.
    """
    client = _get_fmp_client(fmp_client)
    data = _safe_get_list(client.income_statement(symbol, period=period, limit=limit))

    if not data:
        return {
            "revenue": [],
            "revenue_growth": [],
            "gross_margin": [],
            "operating_margin": [],
            "net_margin": [],
            "ebitda_margin": [],
            "margin_trend": "unknown",
            "revenue_cagr_3y": 0.0,
            "revenue_cagr_5y": 0.0,
            "eps": [],
            "dates": [],
            "periods_analysed": 0,
        }

    revenues = [_safe_get(d, "revenue") for d in data]
    gross_profits = [_safe_get(d, "grossProfit") for d in data]
    op_incomes = [_safe_get(d, "operatingIncome") for d in data]
    net_incomes = [_safe_get(d, "netIncome") for d in data]
    dep_amort = [_safe_get(d, "depreciationAndAmortization") for d in data]
    eps_list = [_safe_get(d, "eps") for d in data]
    dates = [_safe_get_str(d, "date") for d in data]

    # Compute margins
    gross_margins = [
        _safe_div(gp, rev) for gp, rev in zip(gross_profits, revenues, strict=False)
    ]
    op_margins = [
        _safe_div(oi, rev) for oi, rev in zip(op_incomes, revenues, strict=False)
    ]
    net_margins = [
        _safe_div(ni, rev) for ni, rev in zip(net_incomes, revenues, strict=False)
    ]
    ebitda_margins = [
        _safe_div(oi + da, rev)
        for oi, da, rev in zip(op_incomes, dep_amort, revenues, strict=False)
    ]

    # Revenue growth (YoY)
    rev_growth = []
    for i in range(len(revenues) - 1):
        rev_growth.append(_pct_change(revenues[i], revenues[i + 1]))

    # CAGR
    def _cagr(values: list[float], years: int) -> float:
        if len(values) <= years or values[years] <= 0 or values[0] <= 0:
            return 0.0
        return (values[0] / values[years]) ** (1.0 / years) - 1.0

    # Margin trend: compare average of first 2 periods to last 2
    if len(op_margins) >= 4:
        recent_avg = sum(op_margins[:2]) / 2
        older_avg = sum(op_margins[-2:]) / 2
        diff = recent_avg - older_avg
        if diff > 0.02:
            margin_trend = "expanding"
        elif diff < -0.02:
            margin_trend = "contracting"
        else:
            margin_trend = "stable"
    elif len(op_margins) >= 2:
        diff = op_margins[0] - op_margins[-1]
        if diff > 0.02:
            margin_trend = "expanding"
        elif diff < -0.02:
            margin_trend = "contracting"
        else:
            margin_trend = "stable"
    else:
        margin_trend = "unknown"

    return {
        "revenue": revenues,
        "revenue_growth": rev_growth,
        "gross_margin": gross_margins,
        "operating_margin": op_margins,
        "net_margin": net_margins,
        "ebitda_margin": ebitda_margins,
        "margin_trend": margin_trend,
        "revenue_cagr_3y": _cagr(revenues, 3),
        "revenue_cagr_5y": (
            _cagr(revenues, min(5, len(revenues) - 1)) if len(revenues) > 1 else 0.0
        ),
        "eps": eps_list,
        "dates": dates,
        "periods_analysed": len(data),
    }


# ---------------------------------------------------------------------------
# Balance Sheet Analysis
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def balance_sheet_analysis(
    symbol: str,
    *,
    period: str = "annual",
    limit: int = 5,
    fmp_client: Any | None = None,
) -> dict[str, Any]:
    """Analyse balance sheet composition and leverage trends.

    Reveals the structure of assets (tangible vs. intangible, current
    vs. long-term), the financing mix (debt vs. equity), and how
    these have evolved.  Essential for credit analysis and for
    understanding a company's capital intensity.

    Parameters:
        symbol: Ticker symbol.
        period: ``"annual"`` or ``"quarter"``.
        limit: Number of historical periods.
        fmp_client: Optional ``FMPProvider`` instance.

    Returns:
        Dictionary containing:
        - **total_assets** (*list[float]*) -- Total assets by period.
        - **total_equity** (*list[float]*) -- Stockholders' equity by period.
        - **total_debt** (*list[float]*) -- Total debt by period.
        - **cash** (*list[float]*) -- Cash & equivalents by period.
        - **net_debt** (*list[float]*) -- Debt minus cash by period.
        - **debt_to_equity** (*list[float]*) -- D/E ratio by period.
        - **debt_to_assets** (*list[float]*) -- Debt ratio by period.
        - **current_ratio** (*list[float]*) -- Current ratio by period.
        - **equity_pct** (*list[float]*) -- Equity as % of total assets.
        - **intangible_pct** (*list[float]*) -- Intangibles + goodwill as %
          of total assets.  High values (>50%) mean most "assets" are
          goodwill from acquisitions.
        - **leverage_trend** (*str*) -- "increasing", "decreasing", or
          "stable" based on D/E trajectory.
        - **book_value_per_share** (*list[float]*) -- BVPS by period.
        - **tangible_bvps** (*list[float]*) -- BVPS excluding intangibles.
        - **dates** (*list[str]*) -- Period end dates.

    Example:
        >>> from wraquant.fundamental.financials import balance_sheet_analysis
        >>> bs = balance_sheet_analysis("AAPL")
        >>> print(f"Net debt: ${bs['net_debt'][0]:,.0f}")
        >>> print(f"Leverage trend: {bs['leverage_trend']}")

    See Also:
        income_analysis: Revenue and margin trends.
        financial_health_score: Composite assessment.
    """
    client = _get_fmp_client(fmp_client)
    data = _safe_get_list(client.balance_sheet(symbol, period=period, limit=limit))

    if not data:
        return {
            "total_assets": [],
            "total_equity": [],
            "total_debt": [],
            "cash": [],
            "net_debt": [],
            "debt_to_equity": [],
            "debt_to_assets": [],
            "current_ratio": [],
            "equity_pct": [],
            "intangible_pct": [],
            "leverage_trend": "unknown",
            "book_value_per_share": [],
            "tangible_bvps": [],
            "dates": [],
        }

    total_assets = [_safe_get(d, "totalAssets") for d in data]
    total_equity = [_safe_get(d, "totalStockholdersEquity") for d in data]
    total_debt = [_safe_get(d, "totalDebt") for d in data]
    cash = [_safe_get(d, "cashAndCashEquivalents") for d in data]
    current_assets = [_safe_get(d, "totalCurrentAssets") for d in data]
    current_liabilities = [_safe_get(d, "totalCurrentLiabilities") for d in data]
    goodwill = [_safe_get(d, "goodwill") for d in data]
    intangibles = [_safe_get(d, "intangibleAssets") for d in data]
    shares = [_safe_get(d, "commonStock", default=1.0) for d in data]
    dates = [_safe_get_str(d, "date") for d in data]

    net_debt = [d - c for d, c in zip(total_debt, cash, strict=False)]
    de_ratios = [
        _safe_div(d, e) for d, e in zip(total_debt, total_equity, strict=False)
    ]
    da_ratios = [
        _safe_div(d, a) for d, a in zip(total_debt, total_assets, strict=False)
    ]
    cr_ratios = [
        _safe_div(ca, cl)
        for ca, cl in zip(current_assets, current_liabilities, strict=False)
    ]
    eq_pct = [_safe_div(e, a) for e, a in zip(total_equity, total_assets, strict=False)]
    intang_pct = [
        _safe_div(g + i, a)
        for g, i, a in zip(goodwill, intangibles, total_assets, strict=False)
    ]

    bvps = [_safe_div(e, s) for e, s in zip(total_equity, shares, strict=False)]
    tangible_bvps = [
        _safe_div(e - g - i, s)
        for e, g, i, s in zip(total_equity, goodwill, intangibles, shares, strict=False)
    ]

    # Leverage trend
    if len(de_ratios) >= 3:
        recent = de_ratios[0]
        older = de_ratios[-1]
        diff = recent - older
        if diff > 0.15:
            leverage_trend = "increasing"
        elif diff < -0.15:
            leverage_trend = "decreasing"
        else:
            leverage_trend = "stable"
    else:
        leverage_trend = "unknown"

    return {
        "total_assets": total_assets,
        "total_equity": total_equity,
        "total_debt": total_debt,
        "cash": cash,
        "net_debt": net_debt,
        "debt_to_equity": de_ratios,
        "debt_to_assets": da_ratios,
        "current_ratio": cr_ratios,
        "equity_pct": eq_pct,
        "intangible_pct": intang_pct,
        "leverage_trend": leverage_trend,
        "book_value_per_share": bvps,
        "tangible_bvps": tangible_bvps,
        "dates": dates,
    }


# ---------------------------------------------------------------------------
# Cash Flow Analysis
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def cash_flow_analysis(
    symbol: str,
    *,
    period: str = "annual",
    limit: int = 5,
    fmp_client: Any | None = None,
) -> dict[str, Any]:
    """Analyse cash flow statement trends and free cash flow quality.

    Cash flow analysis reveals whether reported earnings are backed by
    real cash generation.  A company can report growing profits while
    hemorrhaging cash -- this analysis catches that.

    The key metric is **free cash flow (FCF)** = operating cash flow
    minus capital expenditures.  FCF is what's actually available for
    dividends, buybacks, debt reduction, and reinvestment.

    Parameters:
        symbol: Ticker symbol.
        period: ``"annual"`` or ``"quarter"``.
        limit: Number of historical periods.
        fmp_client: Optional ``FMPProvider`` instance.

    Returns:
        Dictionary containing:
        - **operating_cash_flow** (*list[float]*) -- OCF by period.
        - **capital_expenditures** (*list[float]*) -- CapEx by period
          (negative = spending).
        - **free_cash_flow** (*list[float]*) -- FCF by period.
        - **fcf_margin** (*list[float]*) -- FCF / Revenue by period.
        - **fcf_growth** (*list[float]*) -- YoY FCF growth rates.
        - **fcf_yield** (*float*) -- FCF / Market Cap (most recent).
          > 5% is attractive.
        - **cash_conversion** (*list[float]*) -- OCF / Net Income.
          > 1.0 means the company generates more cash than accounting
          earnings -- a sign of high earnings quality.
        - **capex_to_revenue** (*list[float]*) -- CapEx intensity.
        - **capex_to_ocf** (*list[float]*) -- What portion of OCF is
          consumed by maintenance/growth CapEx.
        - **dividends_paid** (*list[float]*) -- Total dividends paid.
        - **buybacks** (*list[float]*) -- Net share repurchases.
        - **total_shareholder_return** (*list[float]*) -- Dividends + buybacks.
        - **fcf_payout_ratio** (*list[float]*) -- (Dividends + buybacks) / FCF.
        - **dates** (*list[str]*) -- Period end dates.

    Example:
        >>> from wraquant.fundamental.financials import cash_flow_analysis
        >>> cf = cash_flow_analysis("MSFT")
        >>> print(f"FCF margin: {cf['fcf_margin'][0]:.1%}")
        >>> print(f"Cash conversion: {cf['cash_conversion'][0]:.2f}x")

    References:
        Sloan, R. G. (1996). "Do Stock Prices Fully Reflect Information
        in Accruals and Cash Flows about Future Earnings?" *The Accounting
        Review*, 71(3), 289--315.

    See Also:
        earnings_quality: Detailed accruals analysis.
        income_analysis: Margin trends.
    """
    client = _get_fmp_client(fmp_client)
    cf_data = _safe_get_list(client.cash_flow(symbol, period=period, limit=limit))
    income_data = _safe_get_list(
        client.income_statement(symbol, period=period, limit=limit)
    )
    profile = client.company_profile(symbol)

    if not cf_data:
        return {
            "operating_cash_flow": [],
            "capital_expenditures": [],
            "free_cash_flow": [],
            "fcf_margin": [],
            "fcf_growth": [],
            "fcf_yield": 0.0,
            "cash_conversion": [],
            "capex_to_revenue": [],
            "capex_to_ocf": [],
            "dividends_paid": [],
            "buybacks": [],
            "total_shareholder_return": [],
            "fcf_payout_ratio": [],
            "dates": [],
        }

    ocf = [_safe_get(d, "operatingCashFlow") for d in cf_data]
    capex = [_safe_get(d, "capitalExpenditure") for d in cf_data]
    fcf = [_safe_get(d, "freeCashFlow") for d in cf_data]
    divs = [abs(_safe_get(d, "dividendsPaid")) for d in cf_data]
    buybacks_raw = [_safe_get(d, "commonStockRepurchased") for d in cf_data]
    buybacks = [abs(b) for b in buybacks_raw]
    dates = [_safe_get_str(d, "date") for d in cf_data]

    revenues = [_safe_get(d, "revenue") for d in income_data[: len(cf_data)]]
    net_incomes = [_safe_get(d, "netIncome") for d in income_data[: len(cf_data)]]

    # Pad lists if income data is shorter
    while len(revenues) < len(cf_data):
        revenues.append(0.0)
    while len(net_incomes) < len(cf_data):
        net_incomes.append(0.0)

    fcf_margins = [_safe_div(f, r) for f, r in zip(fcf, revenues, strict=False)]
    cash_conv = [_safe_div(o, ni) for o, ni in zip(ocf, net_incomes, strict=False)]
    capex_rev = [_safe_div(abs(c), r) for c, r in zip(capex, revenues, strict=False)]
    capex_ocf = [_safe_div(abs(c), o) for c, o in zip(capex, ocf, strict=False)]

    fcf_growth = []
    for i in range(len(fcf) - 1):
        fcf_growth.append(_pct_change(fcf[i], fcf[i + 1]))

    total_sh_return = [d + b for d, b in zip(divs, buybacks, strict=False)]
    fcf_payout = [
        _safe_div(tsr, f) for tsr, f in zip(total_sh_return, fcf, strict=False)
    ]

    # FCF yield
    profile_data = profile[0] if isinstance(profile, list) and profile else profile
    mkt_cap = (
        _safe_get(profile_data, "mktCap") if isinstance(profile_data, dict) else 0.0
    )
    fcf_yield = _safe_div(fcf[0], mkt_cap) if fcf else 0.0

    return {
        "operating_cash_flow": ocf,
        "capital_expenditures": capex,
        "free_cash_flow": fcf,
        "fcf_margin": fcf_margins,
        "fcf_growth": fcf_growth,
        "fcf_yield": fcf_yield,
        "cash_conversion": cash_conv,
        "capex_to_revenue": capex_rev,
        "capex_to_ocf": capex_ocf,
        "dividends_paid": divs,
        "buybacks": buybacks,
        "total_shareholder_return": total_sh_return,
        "fcf_payout_ratio": fcf_payout,
        "dates": dates,
    }


# ---------------------------------------------------------------------------
# Financial Health Score
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def financial_health_score(
    symbol: str,
    *,
    fmp_client: Any | None = None,
) -> dict[str, Any]:
    """Compute a composite financial health score (0--100).

    Aggregates profitability, liquidity, leverage, efficiency, and
    cash flow quality into a single score.  This is a modernised,
    continuous version of the binary Piotroski F-Score -- it
    captures *how much* better or worse a metric is, not just
    whether it passes a threshold.

    The score is computed as a weighted average of five sub-scores:
    - Profitability (30 points): ROE, ROA, margins
    - Liquidity (15 points): current ratio, quick ratio
    - Leverage (20 points): D/E, interest coverage
    - Efficiency (15 points): asset turnover, cash conversion cycle
    - Cash Flow Quality (20 points): FCF margin, accruals quality

    Parameters:
        symbol: Ticker symbol.
        fmp_client: Optional ``FMPProvider`` instance.

    Returns:
        Dictionary containing:
        - **total_score** (*float*) -- Composite score 0--100.
        - **category** (*str*) -- "excellent" (80+), "good" (60--79),
          "fair" (40--59), "weak" (20--39), "critical" (< 20).
        - **profitability_score** (*float*) -- Sub-score 0--30.
        - **liquidity_score** (*float*) -- Sub-score 0--15.
        - **leverage_score** (*float*) -- Sub-score 0--20.
        - **efficiency_score** (*float*) -- Sub-score 0--15.
        - **cash_flow_score** (*float*) -- Sub-score 0--20.
        - **strengths** (*list[str]*) -- Top-performing areas.
        - **weaknesses** (*list[str]*) -- Areas of concern.
        - **piotroski_f_score** (*int*) -- Traditional F-Score for reference.
        - **fmp_score** (*float*) -- FMP's own financial score.

    Example:
        >>> from wraquant.fundamental.financials import financial_health_score
        >>> health = financial_health_score("MSFT")
        >>> print(f"Score: {health['total_score']:.0f}/100 ({health['category']})")
        >>> print(f"Strengths: {', '.join(health['strengths'])}")

    See Also:
        earnings_quality: Deep dive into earnings reliability.
        comprehensive_ratios: All ratios in one call.
    """
    client = _get_fmp_client(fmp_client)

    income = client.income_statement(symbol)
    balance = client.balance_sheet(symbol)
    cash_flow = client.cash_flow(symbol)
    client.ratios(symbol)
    fmp_score_data = client.score(symbol)

    # Extract key values
    revenue = _safe_get(income, "revenue")
    net_income = _safe_get(income, "netIncome")
    gross_profit = _safe_get(income, "grossProfit")
    op_income = _safe_get(income, "operatingIncome")
    total_assets = _safe_get(balance, "totalAssets")
    total_equity = _safe_get(balance, "totalStockholdersEquity")
    total_debt = _safe_get(balance, "totalDebt")
    current_assets = _safe_get(balance, "totalCurrentAssets")
    current_liabilities = _safe_get(balance, "totalCurrentLiabilities")
    inventory = _safe_get(balance, "inventory")
    ocf = _safe_get(cash_flow, "operatingCashFlow")
    fcf = _safe_get(cash_flow, "freeCashFlow")
    ebit = _safe_get(income, "operatingIncome")
    interest_expense = abs(_safe_get(income, "interestExpense"))

    strengths: list[str] = []
    weaknesses: list[str] = []

    # --- Profitability sub-score (0--30) ---
    roe = _safe_div(net_income, total_equity)
    roa = _safe_div(net_income, total_assets)
    _ = _safe_div(gross_profit, revenue)
    op_margin = _safe_div(op_income, revenue)
    net_margin = _safe_div(net_income, revenue)

    prof_score = 0.0
    # ROE: scale 0--10 (0% = 0, 25%+ = 10)
    prof_score += min(max(roe / 0.25, 0.0), 1.0) * 10
    # ROA: scale 0--5 (0% = 0, 10%+ = 5)
    prof_score += min(max(roa / 0.10, 0.0), 1.0) * 5
    # Operating margin: scale 0--8 (0% = 0, 30%+ = 8)
    prof_score += min(max(op_margin / 0.30, 0.0), 1.0) * 8
    # Net margin: scale 0--7 (0% = 0, 20%+ = 7)
    prof_score += min(max(net_margin / 0.20, 0.0), 1.0) * 7

    if roe > 0.15:
        strengths.append("strong ROE")
    elif roe < 0.05:
        weaknesses.append("low ROE")
    if op_margin > 0.20:
        strengths.append("high operating margin")
    elif op_margin < 0.05:
        weaknesses.append("thin operating margins")

    # --- Liquidity sub-score (0--15) ---
    current_ratio = _safe_div(current_assets, current_liabilities)
    quick_ratio = _safe_div(current_assets - inventory, current_liabilities)

    liq_score = 0.0
    # Current ratio: 1.0 = 5, 2.0+ = 10, <0.8 = 0
    if current_ratio >= 2.0:
        liq_score += 10
    elif current_ratio >= 1.0:
        liq_score += 5 + (current_ratio - 1.0) * 5
    else:
        liq_score += max(current_ratio / 1.0, 0.0) * 5
    # Quick ratio: 1.0+ = 5, 0.5 = 2.5
    liq_score += min(max(quick_ratio / 1.0, 0.0), 1.0) * 5

    if current_ratio > 1.5:
        strengths.append("healthy liquidity")
    elif current_ratio < 1.0:
        weaknesses.append("liquidity risk (current ratio < 1)")

    # --- Leverage sub-score (0--20) ---
    de_ratio = _safe_div(total_debt, total_equity)
    interest_coverage = (
        _safe_div(ebit, interest_expense) if interest_expense > 0 else 20.0
    )

    lev_score = 0.0
    # D/E: 0 = 10, 1.0 = 5, 3.0+ = 0
    if de_ratio <= 0.5:
        lev_score += 10
    elif de_ratio <= 1.0:
        lev_score += 10 - (de_ratio - 0.5) * 10
    elif de_ratio <= 3.0:
        lev_score += max(5 - (de_ratio - 1.0) * 2.5, 0.0)
    # Interest coverage: <1.5 = 0, 3 = 5, 10+ = 10
    if interest_coverage >= 10:
        lev_score += 10
    elif interest_coverage >= 3:
        lev_score += 5 + (interest_coverage - 3) / 7 * 5
    elif interest_coverage >= 1.5:
        lev_score += (interest_coverage - 1.5) / 1.5 * 5
    else:
        lev_score += 0

    if de_ratio < 0.5:
        strengths.append("low leverage")
    elif de_ratio > 2.0:
        weaknesses.append("high leverage")
    if interest_coverage < 2.0 and interest_expense > 0:
        weaknesses.append("weak interest coverage")

    # --- Efficiency sub-score (0--15) ---
    asset_turnover = _safe_div(revenue, total_assets)

    eff_score = 0.0
    # Asset turnover: 0.5 = 5, 1.0+ = 10
    eff_score += min(max(asset_turnover / 1.0, 0.0), 1.0) * 10
    # Positive net income signals efficiency
    if net_income > 0:
        eff_score += 5

    # --- Cash Flow Quality sub-score (0--20) ---
    fcf_margin = _safe_div(fcf, revenue)
    accruals_quality = _safe_div(ocf, net_income) if net_income > 0 else 0.0

    cf_score = 0.0
    # FCF margin: 0% = 0, 15%+ = 10
    cf_score += min(max(fcf_margin / 0.15, 0.0), 1.0) * 10
    # Cash conversion > 1: scale 0--10
    if accruals_quality >= 1.0:
        cf_score += min(accruals_quality / 1.5, 1.0) * 10
    elif accruals_quality > 0:
        cf_score += accruals_quality * 5
    else:
        cf_score += 0

    if fcf_margin > 0.10:
        strengths.append("strong FCF generation")
    elif fcf < 0:
        weaknesses.append("negative free cash flow")
    if accruals_quality > 1.0 and net_income > 0:
        strengths.append("high earnings quality (cash-backed)")
    elif 0 < accruals_quality < 0.7 and net_income > 0:
        weaknesses.append("low earnings quality (accruals-driven)")

    # --- Piotroski F-Score ---
    # Simplified: use income/balance/cash_flow from latest period
    # (This is a lightweight version; for full F-Score, compare 2 periods)
    f_score = 0
    if roa > 0:
        f_score += 1
    if ocf > 0:
        f_score += 1
    if ocf > net_income:
        f_score += 1
    if current_ratio > 1.0:
        f_score += 1
    if net_income > 0:
        f_score += 1

    # FMP's own score
    fmp_scr = _safe_get(fmp_score_data, "altmanZScore")

    # Total score
    total = prof_score + liq_score + lev_score + eff_score + cf_score
    total = min(max(total, 0.0), 100.0)

    if total >= 80:
        category = "excellent"
    elif total >= 60:
        category = "good"
    elif total >= 40:
        category = "fair"
    elif total >= 20:
        category = "weak"
    else:
        category = "critical"

    return {
        "total_score": float(total),
        "category": category,
        "profitability_score": float(min(prof_score, 30.0)),
        "liquidity_score": float(min(liq_score, 15.0)),
        "leverage_score": float(min(lev_score, 20.0)),
        "efficiency_score": float(min(eff_score, 15.0)),
        "cash_flow_score": float(min(cf_score, 20.0)),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "piotroski_f_score": f_score,
        "fmp_score": fmp_scr,
    }


# ---------------------------------------------------------------------------
# Earnings Quality
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def earnings_quality(
    symbol: str,
    *,
    period: str = "annual",
    limit: int = 5,
    fmp_client: Any | None = None,
) -> dict[str, Any]:
    """Assess the quality and sustainability of reported earnings.

    High-quality earnings are cash-backed, persistent, and free of
    accounting manipulation.  Low-quality earnings are driven by
    accruals, non-recurring items, or aggressive accounting.

    This function computes multiple earnings quality metrics from the
    academic literature on earnings management and accruals.

    Key metrics:
    - **Accruals ratio**: Total accruals / total assets.  High
      accruals (>10% of assets) predict future earnings reversals
      (Sloan, 1996).
    - **Cash conversion**: OCF / net income.  Should be > 1.0.
    - **Earnings persistence**: Correlation of earnings across periods.
      High persistence = sustainable earnings.

    Parameters:
        symbol: Ticker symbol.
        period: ``"annual"`` or ``"quarter"``.
        limit: Number of historical periods.
        fmp_client: Optional ``FMPProvider`` instance.

    Returns:
        Dictionary containing:
        - **accruals_ratio** (*float*) -- Total accruals / average
          total assets.  < 5% is high quality; > 10% is a red flag.
        - **cash_conversion_ratio** (*float*) -- OCF / Net Income.
          > 1.0 means cash earnings exceed accounting earnings.
        - **earnings_persistence** (*float*) -- Autocorrelation of
          earnings.  High (>0.7) = stable; low (<0.3) = volatile.
        - **fcf_to_net_income** (*float*) -- FCF / Net Income.
          Measures how much of earnings converts to free cash.
        - **quality_grade** (*str*) -- "A" (excellent), "B" (good),
          "C" (fair), "D" (poor), "F" (manipulated/unreliable).
        - **accruals_trend** (*list[float]*) -- Accruals ratio history.
        - **red_flags** (*list[str]*) -- Specific concerns identified.
        - **periods_analysed** (*int*) -- Number of periods used.

    Example:
        >>> from wraquant.fundamental.financials import earnings_quality
        >>> eq = earnings_quality("AAPL")
        >>> print(f"Quality grade: {eq['quality_grade']}")
        >>> print(f"Accruals ratio: {eq['accruals_ratio']:.2%}")

    References:
        - Sloan, R. G. (1996). "Do Stock Prices Fully Reflect Information
          in Accruals and Cash Flows about Future Earnings?" *The Accounting
          Review*, 71(3), 289--315.
        - Beneish, M. D. (1999). "The Detection of Earnings Manipulation."
          *Financial Analysts Journal*, 55(5), 24--36.

    See Also:
        cash_flow_analysis: Detailed cash flow trends.
        financial_health_score: Composite score.
    """
    client = _get_fmp_client(fmp_client)
    income_data = _safe_get_list(
        client.income_statement(symbol, period=period, limit=limit)
    )
    balance_data = _safe_get_list(
        client.balance_sheet(symbol, period=period, limit=limit)
    )
    cf_data = _safe_get_list(client.cash_flow(symbol, period=period, limit=limit))

    if not income_data or not balance_data or not cf_data:
        return {
            "accruals_ratio": 0.0,
            "cash_conversion_ratio": 0.0,
            "earnings_persistence": 0.0,
            "fcf_to_net_income": 0.0,
            "quality_grade": "N/A",
            "accruals_trend": [],
            "red_flags": ["insufficient data"],
            "periods_analysed": 0,
        }

    net_incomes = [_safe_get(d, "netIncome") for d in income_data]
    total_assets_list = [_safe_get(d, "totalAssets") for d in balance_data]
    ocf_list = [_safe_get(d, "operatingCashFlow") for d in cf_data]
    fcf_list = [_safe_get(d, "freeCashFlow") for d in cf_data]

    # Accruals = Net Income - Operating Cash Flow
    min_len = min(len(net_incomes), len(ocf_list), len(total_assets_list))
    accruals = [
        ni - ocf
        for ni, ocf in zip(net_incomes[:min_len], ocf_list[:min_len], strict=False)
    ]

    # Accruals ratio = accruals / average total assets
    accruals_ratios = []
    for i in range(min_len):
        if i + 1 < len(total_assets_list):
            avg_assets = (total_assets_list[i] + total_assets_list[i + 1]) / 2
        else:
            avg_assets = total_assets_list[i]
        accruals_ratios.append(_safe_div(accruals[i], avg_assets))

    latest_accruals = accruals_ratios[0] if accruals_ratios else 0.0

    # Cash conversion: OCF / Net Income
    cash_conv = _safe_div(ocf_list[0], net_incomes[0]) if net_incomes[0] > 0 else 0.0

    # FCF / Net Income
    fcf_ni = _safe_div(fcf_list[0], net_incomes[0]) if net_incomes[0] > 0 else 0.0

    # Earnings persistence: simple autocorrelation of net income
    persistence = 0.0
    if len(net_incomes) >= 3:
        # Simple: correlation between NI(t) and NI(t-1)
        series_a = net_incomes[:-1]  # NI at t
        series_b = net_incomes[1:]  # NI at t-1
        n = len(series_a)
        mean_a = sum(series_a) / n
        mean_b = sum(series_b) / n
        cov = (
            sum(
                (a - mean_a) * (b - mean_b)
                for a, b in zip(series_a, series_b, strict=False)
            )
            / n
        )
        std_a = (sum((a - mean_a) ** 2 for a in series_a) / n) ** 0.5
        std_b = (sum((b - mean_b) ** 2 for b in series_b) / n) ** 0.5
        if std_a > 1e-12 and std_b > 1e-12:
            persistence = cov / (std_a * std_b)

    # Red flags
    red_flags: list[str] = []
    if abs(latest_accruals) > 0.10:
        red_flags.append(f"high accruals ratio ({latest_accruals:.1%})")
    if cash_conv < 0.7 and net_incomes[0] > 0:
        red_flags.append("low cash conversion (OCF < 70% of net income)")
    if net_incomes[0] > 0 and ocf_list[0] < 0:
        red_flags.append("positive earnings but negative operating cash flow")
    if fcf_ni < 0.5 and net_incomes[0] > 0:
        red_flags.append("FCF significantly below net income")
    if persistence < 0.3 and len(net_incomes) >= 3:
        red_flags.append("low earnings persistence (volatile)")

    # Check for growing accruals (deteriorating quality)
    if len(accruals_ratios) >= 3:
        if accruals_ratios[0] > accruals_ratios[-1] + 0.03:
            red_flags.append("accruals ratio increasing over time")

    # Quality grade
    score = 0.0
    # Accruals: low is good (< 5% = 3, 5-10% = 2, 10%+ = 0)
    if abs(latest_accruals) < 0.05:
        score += 3
    elif abs(latest_accruals) < 0.10:
        score += 2
    # Cash conversion > 1 = 3, > 0.8 = 2, > 0.5 = 1
    if cash_conv >= 1.0:
        score += 3
    elif cash_conv >= 0.8:
        score += 2
    elif cash_conv >= 0.5:
        score += 1
    # Persistence > 0.7 = 2, > 0.4 = 1
    if persistence > 0.7:
        score += 2
    elif persistence > 0.4:
        score += 1
    # FCF/NI > 0.8 = 2, > 0.5 = 1
    if fcf_ni >= 0.8:
        score += 2
    elif fcf_ni >= 0.5:
        score += 1

    if score >= 9:
        grade = "A"
    elif score >= 7:
        grade = "B"
    elif score >= 5:
        grade = "C"
    elif score >= 3:
        grade = "D"
    else:
        grade = "F"

    return {
        "accruals_ratio": float(latest_accruals),
        "cash_conversion_ratio": float(cash_conv),
        "earnings_persistence": float(persistence),
        "fcf_to_net_income": float(fcf_ni),
        "quality_grade": grade,
        "accruals_trend": [float(a) for a in accruals_ratios],
        "red_flags": red_flags,
        "periods_analysed": min_len,
    }


# ---------------------------------------------------------------------------
# Common Size Analysis
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def common_size_analysis(
    symbol: str,
    *,
    period: str = "annual",
    limit: int = 5,
    fmp_client: Any | None = None,
) -> dict[str, Any]:
    """Generate common-size financial statements.

    Common-size analysis expresses each line item as a percentage of a
    base figure: revenue for the income statement, total assets for the
    balance sheet.  This normalisation makes it easy to compare companies
    of different sizes and to track composition changes over time.

    When to use:
        - Cross-company comparison regardless of size.
        - Trend analysis: detect shifts in cost structure or asset mix.
        - Input to peer-relative valuation and industry benchmarking.

    Parameters:
        symbol: Ticker symbol.
        period: ``"annual"`` or ``"quarter"``.
        limit: Number of historical periods.
        fmp_client: Optional ``FMPProvider`` instance.

    Returns:
        Dictionary containing:
        - **income_statement** (*list[dict]*) -- Each period as a dict
          of line items expressed as % of revenue.  Keys include
          ``cost_of_revenue_pct``, ``gross_profit_pct``, ``rd_pct``,
          ``sga_pct``, ``operating_income_pct``, ``net_income_pct``.
        - **balance_sheet** (*list[dict]*) -- Each period as a dict
          of line items expressed as % of total assets.  Keys include
          ``current_assets_pct``, ``fixed_assets_pct``,
          ``intangibles_pct``, ``current_liabilities_pct``,
          ``long_term_debt_pct``, ``equity_pct``.
        - **dates** (*list[str]*) -- Period end dates.
        - **periods_analysed** (*int*) -- Number of periods.

    Example:
        >>> from wraquant.fundamental.financials import common_size_analysis
        >>> cs = common_size_analysis("AAPL")
        >>> latest = cs['income_statement'][0]
        >>> print(f"COGS: {latest['cost_of_revenue_pct']:.1%}")
        >>> print(f"R&D: {latest['rd_pct']:.1%}")

    See Also:
        income_analysis: Absolute values and growth rates.
        balance_sheet_analysis: Composition and leverage trends.
    """
    client = _get_fmp_client(fmp_client)
    income_data = _safe_get_list(
        client.income_statement(symbol, period=period, limit=limit)
    )
    balance_data = _safe_get_list(
        client.balance_sheet(symbol, period=period, limit=limit)
    )

    dates = [_safe_get_str(d, "date") for d in income_data]

    # Common-size income statement (% of revenue)
    cs_income: list[dict[str, float]] = []
    for row in income_data:
        rev = _safe_get(row, "revenue")
        if rev <= 0:
            cs_income.append(
                {
                    "cost_of_revenue_pct": 0.0,
                    "gross_profit_pct": 0.0,
                    "rd_pct": 0.0,
                    "sga_pct": 0.0,
                    "operating_income_pct": 0.0,
                    "interest_expense_pct": 0.0,
                    "income_tax_pct": 0.0,
                    "net_income_pct": 0.0,
                    "ebitda_pct": 0.0,
                }
            )
            continue

        cs_income.append(
            {
                "cost_of_revenue_pct": _safe_div(_safe_get(row, "costOfRevenue"), rev),
                "gross_profit_pct": _safe_div(_safe_get(row, "grossProfit"), rev),
                "rd_pct": _safe_div(
                    _safe_get(row, "researchAndDevelopmentExpenses"), rev
                ),
                "sga_pct": _safe_div(
                    _safe_get(row, "sellingGeneralAndAdministrativeExpenses"),
                    rev,
                ),
                "operating_income_pct": _safe_div(
                    _safe_get(row, "operatingIncome"), rev
                ),
                "interest_expense_pct": _safe_div(
                    abs(_safe_get(row, "interestExpense")),
                    rev,
                ),
                "income_tax_pct": _safe_div(_safe_get(row, "incomeTaxExpense"), rev),
                "net_income_pct": _safe_div(_safe_get(row, "netIncome"), rev),
                "ebitda_pct": _safe_div(
                    _safe_get(row, "operatingIncome")
                    + _safe_get(row, "depreciationAndAmortization"),
                    rev,
                ),
            }
        )

    # Common-size balance sheet (% of total assets)
    cs_balance: list[dict[str, float]] = []
    for row in balance_data:
        assets = _safe_get(row, "totalAssets")
        if assets <= 0:
            cs_balance.append(
                {
                    "current_assets_pct": 0.0,
                    "cash_pct": 0.0,
                    "receivables_pct": 0.0,
                    "inventory_pct": 0.0,
                    "fixed_assets_pct": 0.0,
                    "intangibles_pct": 0.0,
                    "goodwill_pct": 0.0,
                    "current_liabilities_pct": 0.0,
                    "long_term_debt_pct": 0.0,
                    "total_debt_pct": 0.0,
                    "equity_pct": 0.0,
                    "retained_earnings_pct": 0.0,
                }
            )
            continue

        cs_balance.append(
            {
                "current_assets_pct": _safe_div(
                    _safe_get(row, "totalCurrentAssets"),
                    assets,
                ),
                "cash_pct": _safe_div(
                    _safe_get(row, "cashAndCashEquivalents"),
                    assets,
                ),
                "receivables_pct": _safe_div(
                    _safe_get(row, "netReceivables"),
                    assets,
                ),
                "inventory_pct": _safe_div(_safe_get(row, "inventory"), assets),
                "fixed_assets_pct": _safe_div(
                    _safe_get(row, "propertyPlantEquipmentNet"),
                    assets,
                ),
                "intangibles_pct": _safe_div(
                    _safe_get(row, "intangibleAssets"),
                    assets,
                ),
                "goodwill_pct": _safe_div(_safe_get(row, "goodwill"), assets),
                "current_liabilities_pct": _safe_div(
                    _safe_get(row, "totalCurrentLiabilities"),
                    assets,
                ),
                "long_term_debt_pct": _safe_div(
                    _safe_get(row, "longTermDebt"),
                    assets,
                ),
                "total_debt_pct": _safe_div(_safe_get(row, "totalDebt"), assets),
                "equity_pct": _safe_div(
                    _safe_get(row, "totalStockholdersEquity"),
                    assets,
                ),
                "retained_earnings_pct": _safe_div(
                    _safe_get(row, "retainedEarnings"),
                    assets,
                ),
            }
        )

    return {
        "income_statement": cs_income,
        "balance_sheet": cs_balance,
        "dates": dates,
        "periods_analysed": len(income_data),
    }
