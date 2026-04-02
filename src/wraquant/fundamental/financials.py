"""Financial statement analysis using FMP data.

Provides deep analytical functions that go beyond raw financial statements
to deliver trend analysis, growth decomposition, health scoring, and
earnings quality assessment.  These are the tools a fundamental analyst
reaches for after reading the 10-K: they answer "what happened?", "is it
getting better?", and "can I trust the numbers?"

Functions in this module compute derived analytics from the three core
financial statements (income statement, balance sheet, cash flow statement).
All data is fetched from the FMP (Financial Modeling Prep) API.

Key capabilities:

1. **Income analysis** -- Revenue and margin trends, growth rates, and
   operating leverage across multiple periods.
2. **Balance sheet analysis** -- Asset composition, leverage evolution,
   book value trends, and working capital dynamics.
3. **Cash flow analysis** -- Free cash flow generation, cash conversion
   efficiency, and CapEx intensity.
4. **Financial health score** -- Composite 0--100 score aggregating
   profitability, liquidity, solvency, and efficiency into a single
   grade (A--F).
5. **Earnings quality** -- Accruals analysis and cash conversion to
   detect potential earnings manipulation.
6. **Common-size analysis** -- Vertical analysis expressing every line
   item as a percentage of revenue (income statement) or total assets
   (balance sheet).

Example:
    >>> from wraquant.fundamental.financials import income_analysis
    >>> result = income_analysis("AAPL", period="annual")
    >>> print(f"Revenue CAGR (3Y): {result['revenue_cagr_3y']:.1%}")
    >>> print(f"Margin trend: {result['margin_trend']}")

References:
    - Sloan, R. G. (1996). "Do Stock Prices Fully Reflect Information
      in Accruals and Cash Flows about Future Earnings?" *The Accounting
      Review*, 71(3), 289--315.
    - Dechow, P. M. & Dichev, I. D. (2002). "The Quality of Accruals
      and Earnings." *The Accounting Review*, 77(s-1), 35--59.
    - Beneish, M. D. (1999). "The Detection of Earnings Manipulation."
      *Financial Analysts Journal*, 55(5), 24--36.
    - Piotroski, J. D. (2000). "Value Investing: The Use of Historical
      Financial Statement Information to Separate Winners from Losers."
      *Journal of Accounting Research*, 38, 1--41.
    - Palepu, K. G. & Healy, P. M. (2013). *Business Analysis and
      Valuation*, 5th edition. Cengage.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from wraquant.core.decorators import requires_extra

logger = logging.getLogger(__name__)

__all__ = [
    "income_analysis",
    "balance_sheet_analysis",
    "cash_flow_analysis",
    "financial_health_score",
    "earnings_quality",
    "common_size_analysis",
    "revenue_decomposition",
    "working_capital_analysis",
    "capex_analysis",
    "shareholder_returns",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide safely, returning *default* on zero/near-zero denominator."""
    if abs(denominator) < 1e-12:
        return default
    return float(numerator / denominator)


def _get_fmp_client(fmp_client: Any | None = None) -> Any:
    """Return the provided client or construct a default ``FMPClient``."""
    if fmp_client is not None:
        return fmp_client
    from wraquant.data.providers.fmp import FMPClient  # noqa: WPS433

    return FMPClient()


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


def _safe_get_str(data: dict | list, key: str, default: str = "") -> str:
    """Extract a string value from an FMP response dict/list."""
    if isinstance(data, list):
        if not data:
            return default
        data = data[0]
    val = data.get(key)
    return str(val) if val is not None else default


def _safe_get_list(data: Any) -> list[dict]:
    """Coerce *data* to a list of dicts (handles DataFrame, list, or dict)."""
    if isinstance(data, pd.DataFrame):
        return data.to_dict("records")
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    return []


def _pct_change(current: float, previous: float) -> float:
    """Compute percentage change, handling zero denominators."""
    return _safe_div(current - previous, abs(previous))


def _cagr(values: list[float], years: int) -> float:
    """Compute compound annual growth rate.

    *values* are ordered most-recent-first: ``values[0]`` is the latest
    period and ``values[years]`` is *years* periods earlier.
    """
    if len(values) <= years or values[years] <= 0 or values[0] <= 0:
        return 0.0
    return (values[0] / values[years]) ** (1.0 / years) - 1.0


# ---------------------------------------------------------------------------
# Income Statement Analysis
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def income_analysis(
    symbol: str,
    period: str = "annual",
    *,
    fmp_client: Any | None = None,
) -> dict[str, Any]:
    """Analyse income statement trends over multiple periods.

    Goes beyond single-period ratios to reveal the *trajectory* of
    revenue, margins, and bottom-line profitability.  Use this when you
    need to understand whether a company's earning power is structurally
    improving, temporarily inflated, or in secular decline.  A company
    with a 20 % margin that is declining is very different from one with
    a 15 % margin that is expanding.

    This function is the starting point for fundamental stock analysis:
    it answers "is the business growing?" and "are margins expanding or
    compressing?"

    Parameters:
        symbol: Ticker symbol (e.g., ``"AAPL"``).
        period: ``"annual"`` or ``"quarter"``.  Annual data smooths out
            seasonality; quarterly reveals recent momentum.
        fmp_client: Optional pre-configured ``FMPClient`` instance.
            If ``None``, a default client is created using the
            ``FMP_API_KEY`` environment variable.

    Returns:
        Dictionary containing:

        **Time-series data (most recent first):**
        - **revenue** (*list[float]*) -- Revenue by period.
        - **revenue_growth** (*list[float]*) -- YoY revenue growth rates.
        - **gross_margin** (*list[float]*) -- Gross profit / revenue per period.
        - **operating_margin** (*list[float]*) -- Operating income / revenue.
        - **net_margin** (*list[float]*) -- Net income / revenue.
        - **ebitda_margin** (*list[float]*) -- EBITDA / revenue.
        - **eps** (*list[float]*) -- Diluted EPS by period.
        - **dates** (*list[str]*) -- Reporting period dates.

        **Trend analysis:**
        - **margin_trend** (*str*) -- ``"expanding"``, ``"contracting"``,
          or ``"stable"`` based on operating margin trajectory.

        **Growth rates:**
        - **revenue_cagr_3y** (*float*) -- 3-year revenue CAGR.  > 10 %
          is strong organic growth; negative signals secular decline.
        - **revenue_cagr_5y** (*float*) -- 5-year revenue CAGR.

        **Metadata:**
        - **periods_analysed** (*int*) -- Number of periods returned.

    Example:
        >>> from wraquant.fundamental.financials import income_analysis
        >>> inc = income_analysis("MSFT")
        >>> print(f"Revenue CAGR (3Y): {inc['revenue_cagr_3y']:.1%}")
        >>> print(f"Margin trend: {inc['margin_trend']}")

    See Also:
        balance_sheet_analysis: Asset/liability composition trends.
        cash_flow_analysis: Cash flow quality and FCF trends.
        common_size_analysis: Line items as % of revenue.
    """
    client = _get_fmp_client(fmp_client)
    data = _safe_get_list(client.income_statement(symbol, period=period, limit=10))

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
    rev_growth: list[float] = []
    for i in range(len(revenues) - 1):
        rev_growth.append(_pct_change(revenues[i], revenues[i + 1]))

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
    period: str = "annual",
    *,
    fmp_client: Any | None = None,
) -> dict[str, Any]:
    """Analyse balance sheet composition, leverage, and capital structure.

    Reveals the structure of assets (tangible vs. intangible, current
    vs. long-term), the financing mix (debt vs. equity), and how these
    have evolved.  Use this for:

    - **Credit analysis**: Is the company over-leveraged?  Is the debt-
      to-equity ratio trending upward?
    - **Equity screening**: Is book value growing?  What fraction of
      assets is goodwill from acquisitions?
    - **Factor investing**: Value (P/B), investment (asset growth).
    - **Distress prediction**: Working capital and liquidity trends.

    Parameters:
        symbol: Ticker symbol (e.g., ``"AAPL"``).
        period: ``"annual"`` or ``"quarter"``.
        fmp_client: Optional pre-configured ``FMPClient`` instance.
            If ``None``, a default client is created.

    Returns:
        Dictionary containing:

        **Time-series data (most recent first):**
        - **total_assets** (*list[float]*) -- Total assets by period.
        - **total_equity** (*list[float]*) -- Stockholders' equity by period.
        - **total_debt** (*list[float]*) -- Total debt by period.
        - **cash** (*list[float]*) -- Cash & equivalents by period.
        - **net_debt** (*list[float]*) -- Debt minus cash by period.
          Negative means the company has more cash than debt.
        - **debt_to_equity** (*list[float]*) -- D/E ratio by period.
          > 2.0 is high leverage for most industries.
        - **debt_to_assets** (*list[float]*) -- Debt ratio by period.
        - **current_ratio** (*list[float]*) -- Current ratio by period.
          < 1.0 is a liquidity warning.
        - **equity_pct** (*list[float]*) -- Equity as % of total assets.
        - **intangible_pct** (*list[float]*) -- (Intangibles + goodwill) /
          total assets.  > 50 % means most "assets" are goodwill from
          acquisitions -- a risk in downturns.
        - **book_value_per_share** (*list[float]*) -- BVPS by period.
        - **tangible_bvps** (*list[float]*) -- BVPS excluding intangibles.
        - **dates** (*list[str]*) -- Period end dates.

        **Trend analysis:**
        - **leverage_trend** (*str*) -- ``"increasing"``, ``"decreasing"``,
          or ``"stable"`` based on D/E ratio trajectory.

    Example:
        >>> from wraquant.fundamental.financials import balance_sheet_analysis
        >>> bs = balance_sheet_analysis("AAPL")
        >>> print(f"Net debt: ${bs['net_debt'][0]:,.0f}")
        >>> print(f"Leverage trend: {bs['leverage_trend']}")

    See Also:
        income_analysis: Revenue and margin trends.
        cash_flow_analysis: Cash flow quality and FCF trends.
        financial_health_score: Composite assessment.
    """
    client = _get_fmp_client(fmp_client)
    data = _safe_get_list(client.balance_sheet(symbol, period=period, limit=10))

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
    period: str = "annual",
    *,
    fmp_client: Any | None = None,
) -> dict[str, Any]:
    """Analyse cash flow statement trends and free cash flow quality.

    Cash flow analysis reveals whether reported earnings are backed by
    real cash generation.  A company can report growing profits while
    hemorrhaging cash -- this analysis catches that.

    The key metric is **free cash flow (FCF)** = operating cash flow
    minus capital expenditures.  FCF is what is actually available for
    dividends, buybacks, debt reduction, and reinvestment.

    Use this function to:
    - Verify that reported earnings translate into actual cash.
    - Assess CapEx requirements and how much free cash flow remains.
    - Track whether the company is self-funding or reliant on external
      capital.
    - Compare cash returned to shareholders vs. cash generated.

    Parameters:
        symbol: Ticker symbol (e.g., ``"AAPL"``).
        period: ``"annual"`` or ``"quarter"``.
        fmp_client: Optional pre-configured ``FMPClient`` instance.
            If ``None``, a default client is created.

    Returns:
        Dictionary containing:

        **Time-series data (most recent first):**
        - **operating_cash_flow** (*list[float]*) -- OCF by period.
          Should consistently exceed net income for healthy companies.
        - **capital_expenditures** (*list[float]*) -- CapEx by period
          (negative = spending).
        - **free_cash_flow** (*list[float]*) -- FCF by period.
        - **fcf_margin** (*list[float]*) -- FCF / revenue by period.
          > 10 % is strong; indicates each revenue dollar generates
          substantial free cash.
        - **fcf_growth** (*list[float]*) -- YoY FCF growth rates.
        - **cash_conversion** (*list[float]*) -- OCF / net income.
          > 1.0 means cash earnings exceed accounting earnings -- a
          sign of high earnings quality.
        - **capex_to_revenue** (*list[float]*) -- |CapEx| / revenue.
          > 15 % indicates capital-intensive business.
        - **capex_to_ocf** (*list[float]*) -- |CapEx| / OCF.  > 50 %
          means heavy reinvestment requirements.
        - **dividends_paid** (*list[float]*) -- Absolute dividends paid.
        - **buybacks** (*list[float]*) -- Absolute share repurchases.
        - **total_shareholder_return** (*list[float]*) -- Dividends + buybacks.
        - **fcf_payout_ratio** (*list[float]*) -- (Dividends + buybacks) / FCF.
          > 1.0 means the company is returning more than it generates.
        - **dates** (*list[str]*) -- Period end dates.

        **Point estimates:**
        - **fcf_yield** (*float*) -- FCF / market cap (most recent).
          > 5 % is typically attractive for value investors.

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
        income_analysis: Margin trends for context.
    """
    client = _get_fmp_client(fmp_client)
    cf_data = _safe_get_list(client.cash_flow(symbol, period=period, limit=10))
    income_data = _safe_get_list(
        client.income_statement(symbol, period=period, limit=10)
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

    fcf_growth: list[float] = []
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
    """Compute a composite financial health score (0--100) with letter grade.

    Aggregates profitability, liquidity, leverage, efficiency, and cash
    flow quality into a single score.  This is a modernised, continuous
    version of the binary Piotroski F-Score -- it captures *how much*
    better or worse a metric is, not just whether it passes a threshold.

    Use this for:
    - **Screening universes**: Filter out financially distressed companies
      before building factor portfolios.
    - **Risk management**: Flag holdings whose fundamentals are
      deteriorating.
    - **Quick triage**: Rapidly assess health before deep-diving into
      individual statements.

    The score is computed as a weighted average of five sub-scores:
        1. **Profitability (30 pts)**: ROE, ROA, operating margin,
           net margin.
        2. **Liquidity (15 pts)**: Current ratio, quick ratio.
        3. **Leverage (20 pts)**: D/E ratio, interest coverage.
        4. **Efficiency (15 pts)**: Asset turnover, positive NI.
        5. **Cash flow quality (20 pts)**: FCF margin, cash conversion.

    Grading scale:
        A (80--100 "excellent"), B (60--79 "good"), C (40--59 "fair"),
        D (20--39 "weak"), F (0--19 "critical").

    Parameters:
        symbol: Ticker symbol (e.g., ``"AAPL"``).
        fmp_client: Optional pre-configured ``FMPClient`` instance.
            If ``None``, a default client is created.

    Returns:
        Dictionary containing:
        - **total_score** (*float*) -- Composite score 0--100.
        - **grade** (*str*) -- Letter grade: ``"A"`` through ``"F"``.
        - **category** (*str*) -- ``"excellent"``, ``"good"``, ``"fair"``,
          ``"weak"``, or ``"critical"``.
        - **profitability_score** (*float*) -- Sub-score out of 30.
        - **liquidity_score** (*float*) -- Sub-score out of 15.
        - **leverage_score** (*float*) -- Sub-score out of 20.
        - **efficiency_score** (*float*) -- Sub-score out of 15.
        - **cash_flow_score** (*float*) -- Sub-score out of 20.
        - **strengths** (*list[str]*) -- Top-performing areas.
        - **weaknesses** (*list[str]*) -- Areas of concern.
        - **piotroski_f_score** (*int*) -- Traditional F-Score (0--9)
          for reference.
        - **symbol** (*str*) -- The ticker analysed.

    Example:
        >>> from wraquant.fundamental.financials import financial_health_score
        >>> health = financial_health_score("MSFT")
        >>> print(f"Score: {health['total_score']:.0f}/100 ({health['grade']})")
        >>> print(f"Strengths: {', '.join(health['strengths'])}")

    See Also:
        earnings_quality: Deep dive into earnings reliability.
        comprehensive_ratios: All ratios in one call.
    """
    client = _get_fmp_client(fmp_client)

    income = client.income_statement(symbol)
    balance = client.balance_sheet(symbol)
    cash_flow = client.cash_flow(symbol)
    _ = client.score(symbol)

    # Extract key values
    revenue = _safe_get(income, "revenue")
    net_income = _safe_get(income, "netIncome")
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
    if current_ratio >= 2.0:
        liq_score += 10
    elif current_ratio >= 1.0:
        liq_score += 5 + (current_ratio - 1.0) * 5
    else:
        liq_score += max(current_ratio / 1.0, 0.0) * 5
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
    if de_ratio <= 0.5:
        lev_score += 10
    elif de_ratio <= 1.0:
        lev_score += 10 - (de_ratio - 0.5) * 10
    elif de_ratio <= 3.0:
        lev_score += max(5 - (de_ratio - 1.0) * 2.5, 0.0)
    if interest_coverage >= 10:
        lev_score += 10
    elif interest_coverage >= 3:
        lev_score += 5 + (interest_coverage - 3) / 7 * 5
    elif interest_coverage >= 1.5:
        lev_score += (interest_coverage - 1.5) / 1.5 * 5

    if de_ratio < 0.5:
        strengths.append("low leverage")
    elif de_ratio > 2.0:
        weaknesses.append("high leverage")
    if interest_coverage < 2.0 and interest_expense > 0:
        weaknesses.append("weak interest coverage")

    # --- Efficiency sub-score (0--15) ---
    asset_turnover = _safe_div(revenue, total_assets)

    eff_score = 0.0
    eff_score += min(max(asset_turnover / 1.0, 0.0), 1.0) * 10
    if net_income > 0:
        eff_score += 5

    # --- Cash Flow Quality sub-score (0--20) ---
    fcf_margin = _safe_div(fcf, revenue)
    accruals_quality = _safe_div(ocf, net_income) if net_income > 0 else 0.0

    cf_score = 0.0
    cf_score += min(max(fcf_margin / 0.15, 0.0), 1.0) * 10
    if accruals_quality >= 1.0:
        cf_score += min(accruals_quality / 1.5, 1.0) * 10
    elif accruals_quality > 0:
        cf_score += accruals_quality * 5

    if fcf_margin > 0.10:
        strengths.append("strong FCF generation")
    elif fcf < 0:
        weaknesses.append("negative free cash flow")
    if accruals_quality > 1.0 and net_income > 0:
        strengths.append("high earnings quality (cash-backed)")
    elif 0 < accruals_quality < 0.7 and net_income > 0:
        weaknesses.append("low earnings quality (accruals-driven)")

    # --- Piotroski F-Score (simplified single-period) ---
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

    # Total score
    total = prof_score + liq_score + lev_score + eff_score + cf_score
    total = min(max(total, 0.0), 100.0)

    if total >= 80:
        category = "excellent"
        grade = "A"
    elif total >= 60:
        category = "good"
        grade = "B"
    elif total >= 40:
        category = "fair"
        grade = "C"
    elif total >= 20:
        category = "weak"
        grade = "D"
    else:
        category = "critical"
        grade = "F"

    return {
        "symbol": symbol,
        "total_score": round(float(total), 1),
        "grade": grade,
        "category": category,
        "profitability_score": round(float(min(prof_score, 30.0)), 1),
        "liquidity_score": round(float(min(liq_score, 15.0)), 1),
        "leverage_score": round(float(min(lev_score, 20.0)), 1),
        "efficiency_score": round(float(min(eff_score, 15.0)), 1),
        "cash_flow_score": round(float(min(cf_score, 20.0)), 1),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "piotroski_f_score": f_score,
    }


# ---------------------------------------------------------------------------
# Earnings Quality
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def earnings_quality(
    symbol: str,
    *,
    fmp_client: Any | None = None,
) -> dict[str, Any]:
    """Assess the quality and sustainability of reported earnings.

    High-quality earnings are cash-backed, persistent, and free of
    accounting manipulation.  Low-quality earnings are driven by
    accruals, non-recurring items, or aggressive accounting.

    This function computes multiple earnings quality metrics from the
    academic literature on earnings management and accruals.  Use it as
    a quality filter in stock selection: prefer companies with low
    accruals and high cash conversion.

    Key metrics:

    - **Accruals ratio**: Total accruals / average total assets.  High
      accruals (> 10 % of assets) predict future earnings reversals
      (Sloan, 1996).  This is the single most powerful quality signal.
    - **Cash conversion**: OCF / net income.  Should be > 1.0 for
      healthy companies.  Consistently < 0.7 is a red flag.
    - **Earnings persistence**: Autocorrelation of earnings across
      periods.  High persistence (> 0.7) = sustainable earnings.

    Mathematical formulations:
        Accruals = Net Income - Operating Cash Flow
        Accruals Ratio = Accruals / Average Total Assets
        Cash Conversion Ratio = Operating CF / Net Income
        FCF to Net Income = Free Cash Flow / Net Income

    Parameters:
        symbol: Ticker symbol (e.g., ``"AAPL"``).
        fmp_client: Optional pre-configured ``FMPClient`` instance.
            If ``None``, a default client is created.

    Returns:
        Dictionary containing:
        - **accruals_ratio** (*float*) -- Accruals / avg total assets.
          < 5 % is high quality; > 10 % is a red flag.
        - **cash_conversion_ratio** (*float*) -- OCF / net income.
          > 1.0 means earnings are backed by cash.
        - **earnings_persistence** (*float*) -- Autocorrelation of NI.
          > 0.7 = stable; < 0.3 = volatile.
        - **fcf_to_net_income** (*float*) -- FCF / net income.
          > 0.8 is strong; < 0.5 means heavy CapEx eats into earnings.
        - **quality_grade** (*str*) -- ``"A"`` (excellent) through
          ``"F"`` (manipulated/unreliable).
        - **accruals_trend** (*list[float]*) -- Accruals ratio history
          (most recent first).
        - **red_flags** (*list[str]*) -- Specific concerns identified.
        - **periods_analysed** (*int*) -- Number of periods used.
        - **symbol** (*str*) -- The ticker analysed.

    Example:
        >>> from wraquant.fundamental.financials import earnings_quality
        >>> eq = earnings_quality("AAPL")
        >>> print(f"Quality grade: {eq['quality_grade']}")
        >>> print(f"Accruals ratio: {eq['accruals_ratio']:.2%}")
        >>> if eq['red_flags']:
        ...     print(f"Warnings: {', '.join(eq['red_flags'])}")

    Notes:
        Reference: Sloan, R. G. (1996). "Do Stock Prices Fully Reflect
        Information in Accruals and Cash Flows about Future Earnings?"
        *The Accounting Review*, 71(3), 289--315.

    See Also:
        cash_flow_analysis: Detailed cash flow trends.
        financial_health_score: Composite score.
    """
    client = _get_fmp_client(fmp_client)
    income_data = _safe_get_list(
        client.income_statement(symbol, period="annual", limit=5)
    )
    balance_data = _safe_get_list(
        client.balance_sheet(symbol, period="annual", limit=5)
    )
    cf_data = _safe_get_list(client.cash_flow(symbol, period="annual", limit=5))

    if not income_data or not balance_data or not cf_data:
        return {
            "symbol": symbol,
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
    accruals_ratios: list[float] = []
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
        series_a = net_incomes[:-1]
        series_b = net_incomes[1:]
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
    if len(accruals_ratios) >= 3:
        if accruals_ratios[0] > accruals_ratios[-1] + 0.03:
            red_flags.append("accruals ratio increasing over time")

    # Quality grade
    score = 0.0
    if abs(latest_accruals) < 0.05:
        score += 3
    elif abs(latest_accruals) < 0.10:
        score += 2
    if cash_conv >= 1.0:
        score += 3
    elif cash_conv >= 0.8:
        score += 2
    elif cash_conv >= 0.5:
        score += 1
    if persistence > 0.7:
        score += 2
    elif persistence > 0.4:
        score += 1
    if fcf_ni >= 0.8:
        score += 2
    elif fcf_ni >= 0.5:
        score += 1

    if score >= 9:
        quality_grade = "A"
    elif score >= 7:
        quality_grade = "B"
    elif score >= 5:
        quality_grade = "C"
    elif score >= 3:
        quality_grade = "D"
    else:
        quality_grade = "F"

    return {
        "symbol": symbol,
        "accruals_ratio": float(latest_accruals),
        "cash_conversion_ratio": float(cash_conv),
        "earnings_persistence": float(persistence),
        "fcf_to_net_income": float(fcf_ni),
        "quality_grade": quality_grade,
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
    period: str = "annual",
    *,
    fmp_client: Any | None = None,
) -> pd.DataFrame:
    """Generate a common-size DataFrame combining income and balance sheet.

    Common-size analysis expresses each line item as a percentage of a
    base figure: revenue for the income statement, total assets for the
    balance sheet.  This normalisation makes it easy to:

    - **Compare companies of different sizes** on an apples-to-apples
      basis.
    - **Track composition changes** over time (e.g., is R&D spending
      growing as a share of revenue?).
    - **Benchmark against sector medians** to spot outliers.

    When to use:
        Use common-size analysis as input to peer-relative valuation
        and industry benchmarking.  It is also a prerequisite for
        detecting structural shifts in cost structure or asset mix
        across reporting periods.

    Parameters:
        symbol: Ticker symbol (e.g., ``"AAPL"``).
        period: ``"annual"`` or ``"quarter"``.
        fmp_client: Optional pre-configured ``FMPClient`` instance.
            If ``None``, a default client is created.

    Returns:
        DataFrame with one row per reporting period and columns for each
        line item expressed as a ratio (0--1 scale):

        **Income statement (% of revenue):**
        - **date** (*str*) -- Reporting period date.
        - **cost_of_revenue_pct** (*float*) -- COGS / revenue.
        - **gross_profit_pct** (*float*) -- Gross profit / revenue.
        - **rd_pct** (*float*) -- R&D expense / revenue.
        - **sga_pct** (*float*) -- SG&A expense / revenue.
        - **operating_income_pct** (*float*) -- Operating income / revenue.
        - **interest_expense_pct** (*float*) -- |Interest| / revenue.
        - **income_tax_pct** (*float*) -- Income tax / revenue.
        - **net_income_pct** (*float*) -- Net income / revenue.
        - **ebitda_pct** (*float*) -- EBITDA / revenue.

        **Balance sheet (% of total assets):**
        - **current_assets_pct** (*float*) -- Current assets / total assets.
        - **cash_pct** (*float*) -- Cash / total assets.
        - **receivables_pct** (*float*) -- Net receivables / total assets.
        - **inventory_pct** (*float*) -- Inventory / total assets.
        - **fixed_assets_pct** (*float*) -- PP&E / total assets.
        - **intangibles_pct** (*float*) -- Intangible assets / total assets.
        - **goodwill_pct** (*float*) -- Goodwill / total assets.
        - **current_liabilities_pct** (*float*) -- Current liabilities /
          total assets.
        - **long_term_debt_pct** (*float*) -- Long-term debt / total assets.
        - **total_debt_pct** (*float*) -- Total debt / total assets.
        - **equity_pct** (*float*) -- Equity / total assets.
        - **retained_earnings_pct** (*float*) -- Retained earnings /
          total assets.

    Example:
        >>> from wraquant.fundamental.financials import common_size_analysis
        >>> cs = common_size_analysis("AAPL")
        >>> print(cs[["date", "gross_profit_pct", "rd_pct",
        ...           "operating_income_pct"]].head())

    See Also:
        income_analysis: Absolute values and growth rates.
        balance_sheet_analysis: Composition and leverage trends.
    """
    client = _get_fmp_client(fmp_client)
    income_data = _safe_get_list(
        client.income_statement(symbol, period=period, limit=10)
    )
    balance_data = _safe_get_list(client.balance_sheet(symbol, period=period, limit=10))

    _all_cols = [
        "date",
        "cost_of_revenue_pct",
        "gross_profit_pct",
        "rd_pct",
        "sga_pct",
        "operating_income_pct",
        "interest_expense_pct",
        "income_tax_pct",
        "net_income_pct",
        "ebitda_pct",
        "current_assets_pct",
        "cash_pct",
        "receivables_pct",
        "inventory_pct",
        "fixed_assets_pct",
        "intangibles_pct",
        "goodwill_pct",
        "current_liabilities_pct",
        "long_term_debt_pct",
        "total_debt_pct",
        "equity_pct",
        "retained_earnings_pct",
    ]

    if not income_data:
        return pd.DataFrame(columns=_all_cols)

    _zero_income = {
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
    _zero_balance = {
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

    records: list[dict[str, Any]] = []
    n_periods = max(len(income_data), len(balance_data))

    for i in range(n_periods):
        inc = income_data[i] if i < len(income_data) else {}
        bs = balance_data[i] if i < len(balance_data) else {}

        rev = _safe_get(inc, "revenue")
        assets = _safe_get(bs, "totalAssets")

        record: dict[str, Any] = {
            "date": _safe_get_str(inc, "date") or _safe_get_str(bs, "date"),
        }

        # Income statement as % of revenue
        if rev > 0:
            record["cost_of_revenue_pct"] = _safe_div(
                _safe_get(inc, "costOfRevenue"), rev
            )
            record["gross_profit_pct"] = _safe_div(_safe_get(inc, "grossProfit"), rev)
            record["rd_pct"] = _safe_div(
                _safe_get(inc, "researchAndDevelopmentExpenses"), rev
            )
            record["sga_pct"] = _safe_div(
                _safe_get(inc, "sellingGeneralAndAdministrativeExpenses"),
                rev,
            )
            record["operating_income_pct"] = _safe_div(
                _safe_get(inc, "operatingIncome"), rev
            )
            record["interest_expense_pct"] = _safe_div(
                abs(_safe_get(inc, "interestExpense")), rev
            )
            record["income_tax_pct"] = _safe_div(
                _safe_get(inc, "incomeTaxExpense"), rev
            )
            record["net_income_pct"] = _safe_div(_safe_get(inc, "netIncome"), rev)
            record["ebitda_pct"] = _safe_div(
                _safe_get(inc, "operatingIncome")
                + _safe_get(inc, "depreciationAndAmortization"),
                rev,
            )
        else:
            record.update(_zero_income)

        # Balance sheet as % of total assets
        if assets > 0:
            record["current_assets_pct"] = _safe_div(
                _safe_get(bs, "totalCurrentAssets"), assets
            )
            record["cash_pct"] = _safe_div(
                _safe_get(bs, "cashAndCashEquivalents"), assets
            )
            record["receivables_pct"] = _safe_div(
                _safe_get(bs, "netReceivables"), assets
            )
            record["inventory_pct"] = _safe_div(_safe_get(bs, "inventory"), assets)
            record["fixed_assets_pct"] = _safe_div(
                _safe_get(bs, "propertyPlantEquipmentNet"), assets
            )
            record["intangibles_pct"] = _safe_div(
                _safe_get(bs, "intangibleAssets"), assets
            )
            record["goodwill_pct"] = _safe_div(_safe_get(bs, "goodwill"), assets)
            record["current_liabilities_pct"] = _safe_div(
                _safe_get(bs, "totalCurrentLiabilities"), assets
            )
            record["long_term_debt_pct"] = _safe_div(
                _safe_get(bs, "longTermDebt"), assets
            )
            record["total_debt_pct"] = _safe_div(_safe_get(bs, "totalDebt"), assets)
            record["equity_pct"] = _safe_div(
                _safe_get(bs, "totalStockholdersEquity"), assets
            )
            record["retained_earnings_pct"] = _safe_div(
                _safe_get(bs, "retainedEarnings"), assets
            )
        else:
            record.update(_zero_balance)

        records.append(record)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Revenue Decomposition
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def revenue_decomposition(
    symbol: str,
    *,
    fmp_client: Any | None = None,
) -> dict[str, Any]:
    """Break down revenue by product segment and geographic region.

    Understanding *where* revenue comes from is essential for assessing
    concentration risk, growth drivers, and geographic diversification.
    A company with 80 % of revenue from one product is far riskier than
    one with balanced segment mix.  Similarly, heavy geographic
    concentration creates FX and geopolitical risk.

    When to use:
        - Identify revenue concentration risk (single product/region).
        - Assess growth drivers: which segments are accelerating?
        - Geographic diversification analysis for risk management.
        - Input to :func:`sum_of_parts_valuation` for SOTP analysis.

    Parameters:
        symbol: Ticker symbol (e.g., ``"AAPL"``).
        fmp_client: Optional pre-configured ``FMPClient`` instance.

    Returns:
        Dictionary containing:
        - **symbol** (*str*) -- The ticker analysed.
        - **product_segments** (*list[dict]*) -- Per-product breakdown
          with ``name``, ``revenue``, ``pct_of_total``.  Sorted by
          revenue descending.
        - **geographic_segments** (*list[dict]*) -- Per-region breakdown
          with ``name``, ``revenue``, ``pct_of_total``.
        - **total_revenue** (*float*) -- Total revenue for reference.
        - **concentration_risk** (*str*) -- ``"high"`` if top segment
          > 60 % of total, ``"moderate"`` if > 40 %, ``"low"`` otherwise.
        - **top_product_pct** (*float*) -- Largest product segment as
          fraction of total revenue.
        - **top_geo_pct** (*float*) -- Largest geographic region as
          fraction of total revenue.

    Example:
        >>> from wraquant.fundamental.financials import revenue_decomposition
        >>> rd = revenue_decomposition("AAPL")
        >>> for seg in rd["product_segments"]:
        ...     print(f"{seg['name']}: {seg['pct_of_total']:.1%}")
        >>> print(f"Concentration risk: {rd['concentration_risk']}")

    See Also:
        income_analysis: Revenue trends over time.
        common_size_analysis: Line items as % of revenue.
    """
    client = _get_fmp_client(fmp_client)

    # Fetch segmentation data
    try:
        product_data = _safe_get_list(client.revenue_product_segmentation(symbol))
    except Exception:  # noqa: BLE001
        product_data = []

    try:
        geo_data = _safe_get_list(client.revenue_geographic_segmentation(symbol))
    except Exception:  # noqa: BLE001
        geo_data = []

    # Get total revenue for reference
    income = _safe_get_list(client.income_statement(symbol, period="annual", limit=1))
    total_revenue = _safe_get(income[0], "revenue") if income else 0.0

    def _parse_segments(data_list: list[dict]) -> list[dict[str, Any]]:
        """Parse FMP segment data into a clean list of segment dicts."""
        if not data_list:
            return []
        latest = data_list[0] if data_list else {}
        segments: dict[str, float] = {}
        if isinstance(latest, dict):
            for key, val in latest.items():
                if key in ("date", "symbol", "cik", "period"):
                    continue
                try:
                    rev = float(val) if val else 0.0
                    if rev > 0:
                        segments[key] = rev
                except (TypeError, ValueError):
                    continue

        seg_total = sum(segments.values()) or total_revenue or 1.0
        result = [
            {
                "name": name,
                "revenue": float(rev),
                "pct_of_total": float(rev / seg_total) if seg_total > 0 else 0.0,
            }
            for name, rev in segments.items()
        ]
        result.sort(key=lambda x: x["revenue"], reverse=True)
        return result

    product_segments = _parse_segments(product_data)
    geographic_segments = _parse_segments(geo_data)

    # Concentration risk
    top_product_pct = product_segments[0]["pct_of_total"] if product_segments else 0.0
    top_geo_pct = geographic_segments[0]["pct_of_total"] if geographic_segments else 0.0

    max_concentration = max(top_product_pct, top_geo_pct)
    if max_concentration > 0.60:
        concentration_risk = "high"
    elif max_concentration > 0.40:
        concentration_risk = "moderate"
    else:
        concentration_risk = "low"

    return {
        "symbol": symbol,
        "product_segments": product_segments,
        "geographic_segments": geographic_segments,
        "total_revenue": float(total_revenue),
        "concentration_risk": concentration_risk,
        "top_product_pct": float(top_product_pct),
        "top_geo_pct": float(top_geo_pct),
    }


# ---------------------------------------------------------------------------
# Working Capital Analysis
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def working_capital_analysis(
    symbol: str,
    *,
    periods: int = 5,
    fmp_client: Any | None = None,
) -> dict[str, Any]:
    """Analyse working capital efficiency and cash conversion cycle trends.

    Working capital management directly impacts free cash flow.  A company
    that collects receivables faster, turns inventory quicker, and delays
    payables generates more cash from the same level of sales.  The Cash
    Conversion Cycle (CCC) captures all three dynamics in one number.

    Deteriorating working capital (rising CCC) is an early warning of
    operational problems -- even before it shows up in earnings.

    When to use:
        - Cash flow quality assessment: rising DSO may signal aggressive
          revenue recognition.
        - Operational efficiency benchmarking vs. peers.
        - Early warning system: deteriorating CCC often precedes earnings
          misses.
        - Complement to :func:`earnings_quality` for detecting manipulation.

    Mathematical formulations:
        DSO = Accounts Receivable / (Revenue / 365)
        DIO = Inventory / (COGS / 365)
        DPO = Accounts Payable / (COGS / 365)
        CCC = DSO + DIO - DPO

    Parameters:
        symbol: Ticker symbol (e.g., ``"WMT"``).
        periods: Number of annual periods to analyse for trend detection.
        fmp_client: Optional pre-configured ``FMPClient`` instance.

    Returns:
        Dictionary containing:
        - **symbol** (*str*) -- The ticker analysed.
        - **periods_analysed** (*int*) -- Actual periods with data.
        - **dso** (*list[float]*) -- Days Sales Outstanding by period
          (most recent first).  Rising DSO = slower collections.
        - **dio** (*list[float]*) -- Days Inventory Outstanding by period.
          Rising DIO = inventory building up (demand problem?).
        - **dpo** (*list[float]*) -- Days Payable Outstanding by period.
          Rising DPO = stretching supplier payments.
        - **ccc** (*list[float]*) -- Cash Conversion Cycle by period.
          Negative CCC (like Amazon) = funded by suppliers.
        - **working_capital** (*list[float]*) -- Net working capital
          (current assets - current liabilities) by period.
        - **wc_to_revenue** (*list[float]*) -- Working capital as % of
          revenue.  Rising ratio = more capital tied up.
        - **ccc_trend** (*str*) -- ``"improving"`` (CCC declining),
          ``"deteriorating"`` (rising), or ``"stable"``.
        - **dates** (*list[str]*) -- Period end dates.

    Example:
        >>> from wraquant.fundamental.financials import working_capital_analysis
        >>> wc = working_capital_analysis("WMT")
        >>> print(f"CCC: {wc['ccc'][0]:.0f} days (trend: {wc['ccc_trend']})")
        >>> print(f"DSO: {wc['dso'][0]:.0f} days")

    References:
        Richards, V. D. & Laughlin, E. J. (1980). "A Cash Conversion
        Cycle Approach to Liquidity Analysis." *Financial Management*,
        9(1), 32--38.

    See Also:
        efficiency_ratios: Point-in-time turnover ratios.
        cash_flow_analysis: Broader cash flow trends.
    """
    client = _get_fmp_client(fmp_client)

    income_data = _safe_get_list(
        client.income_statement(symbol, period="annual", limit=periods)
    )
    balance_data = _safe_get_list(
        client.balance_sheet(symbol, period="annual", limit=periods)
    )

    n = min(len(income_data), len(balance_data))
    if n == 0:
        return {
            "symbol": symbol,
            "periods_analysed": 0,
            "dso": [],
            "dio": [],
            "dpo": [],
            "ccc": [],
            "working_capital": [],
            "wc_to_revenue": [],
            "ccc_trend": "unknown",
            "dates": [],
        }

    dso_list: list[float] = []
    dio_list: list[float] = []
    dpo_list: list[float] = []
    ccc_list: list[float] = []
    wc_list: list[float] = []
    wc_rev_list: list[float] = []
    dates: list[str] = []

    for i in range(n):
        revenue = _safe_get(income_data[i], "revenue")
        cogs = _safe_get(income_data[i], "costOfRevenue")
        receivables = _safe_get(balance_data[i], "netReceivables")
        inventory = _safe_get(balance_data[i], "inventory")
        payables = _safe_get(balance_data[i], "accountPayables")
        current_assets = _safe_get(balance_data[i], "totalCurrentAssets")
        current_liab = _safe_get(balance_data[i], "totalCurrentLiabilities")

        daily_revenue = _safe_div(revenue, 365.0)
        daily_cogs = _safe_div(cogs, 365.0)

        dso = _safe_div(receivables, daily_revenue)
        dio = _safe_div(inventory, daily_cogs)
        dpo = _safe_div(payables, daily_cogs)
        ccc = dso + dio - dpo

        wc = current_assets - current_liab
        wc_rev = _safe_div(wc, revenue) if revenue > 0 else 0.0

        dso_list.append(float(dso))
        dio_list.append(float(dio))
        dpo_list.append(float(dpo))
        ccc_list.append(float(ccc))
        wc_list.append(float(wc))
        wc_rev_list.append(float(wc_rev))
        dates.append(_safe_get_str(balance_data[i], "date"))

    # CCC trend
    if len(ccc_list) >= 2:
        ccc_change = ccc_list[0] - ccc_list[-1]
        if ccc_change < -5:
            ccc_trend = "improving"
        elif ccc_change > 5:
            ccc_trend = "deteriorating"
        else:
            ccc_trend = "stable"
    else:
        ccc_trend = "unknown"

    return {
        "symbol": symbol,
        "periods_analysed": n,
        "dso": dso_list,
        "dio": dio_list,
        "dpo": dpo_list,
        "ccc": ccc_list,
        "working_capital": wc_list,
        "wc_to_revenue": wc_rev_list,
        "ccc_trend": ccc_trend,
        "dates": dates,
    }


# ---------------------------------------------------------------------------
# CapEx Analysis
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def capex_analysis(
    symbol: str,
    *,
    periods: int = 5,
    fmp_client: Any | None = None,
) -> dict[str, Any]:
    """Analyse capital expenditure intensity, maintenance vs. growth split.

    Not all CapEx is created equal.  **Maintenance CapEx** merely sustains
    the existing asset base (roughly equal to depreciation).  **Growth
    CapEx** expands productive capacity and drives future revenue.
    Understanding this split is crucial for:

    - True free cash flow: FCF = OCF - Maintenance CapEx (not total CapEx).
    - Growth assessment: high growth CapEx signals management confidence.
    - Capital intensity: CapEx/Revenue reveals how capital-hungry the
      business model is.

    When to use:
        - Distinguish between asset-light (SaaS) and capital-heavy
          (industrials, utilities) business models.
        - Estimate "owner earnings" (Buffett): NI + D&A - Maintenance CapEx.
        - Identify companies investing aggressively for future growth.
        - Input to valuation: only maintenance CapEx should be deducted
          in a "normalised FCF" model.

    Mathematical formulations:
        Maintenance CapEx ≈ Depreciation & Amortisation
        Growth CapEx = Total CapEx - Maintenance CapEx
        CapEx Intensity = |CapEx| / Revenue
        CapEx / OCF = |CapEx| / Operating Cash Flow
        Owner Earnings = Net Income + D&A - Maintenance CapEx

    Parameters:
        symbol: Ticker symbol (e.g., ``"AAPL"``).
        periods: Number of annual periods to analyse.
        fmp_client: Optional pre-configured ``FMPClient`` instance.

    Returns:
        Dictionary containing:
        - **symbol** (*str*) -- The ticker analysed.
        - **periods_analysed** (*int*) -- Actual periods with data.
        - **total_capex** (*list[float]*) -- Total CapEx by period
          (negative = spending).
        - **depreciation** (*list[float]*) -- D&A by period (proxy for
          maintenance CapEx).
        - **maintenance_capex** (*list[float]*) -- Estimated maintenance
          CapEx (≈ D&A).
        - **growth_capex** (*list[float]*) -- Total CapEx minus
          maintenance.  Positive = investing for growth.
        - **capex_to_revenue** (*list[float]*) -- |CapEx| / revenue.
          > 15 % = capital-intensive; < 5 % = asset-light.
        - **capex_to_ocf** (*list[float]*) -- |CapEx| / OCF.  > 80 %
          leaves little FCF.
        - **growth_capex_pct** (*list[float]*) -- Growth CapEx as % of
          total CapEx.
        - **owner_earnings** (*list[float]*) -- NI + D&A - maintenance
          CapEx (Buffett's preferred measure).
        - **capex_trend** (*str*) -- ``"increasing"``, ``"decreasing"``,
          or ``"stable"`` based on CapEx/Revenue trajectory.
        - **dates** (*list[str]*) -- Period end dates.

    Example:
        >>> from wraquant.fundamental.financials import capex_analysis
        >>> ca = capex_analysis("AMZN")
        >>> print(f"CapEx intensity: {ca['capex_to_revenue'][0]:.1%}")
        >>> print(f"Growth CapEx %: {ca['growth_capex_pct'][0]:.1%}")
        >>> print(f"Owner earnings: ${ca['owner_earnings'][0]:,.0f}")

    References:
        Buffett, W. (1986). Berkshire Hathaway Shareholder Letter
        ("owner earnings" concept).

    See Also:
        cash_flow_analysis: Broader cash flow metrics.
        shareholder_returns: How CapEx competes with buybacks/dividends.
    """
    client = _get_fmp_client(fmp_client)

    income_data = _safe_get_list(
        client.income_statement(symbol, period="annual", limit=periods)
    )
    cf_data = _safe_get_list(client.cash_flow(symbol, period="annual", limit=periods))

    n = min(len(income_data), len(cf_data))
    if n == 0:
        return {
            "symbol": symbol,
            "periods_analysed": 0,
            "total_capex": [],
            "depreciation": [],
            "maintenance_capex": [],
            "growth_capex": [],
            "capex_to_revenue": [],
            "capex_to_ocf": [],
            "growth_capex_pct": [],
            "owner_earnings": [],
            "capex_trend": "unknown",
            "dates": [],
        }

    total_capex_list: list[float] = []
    depreciation_list: list[float] = []
    maintenance_list: list[float] = []
    growth_list: list[float] = []
    capex_rev_list: list[float] = []
    capex_ocf_list: list[float] = []
    growth_pct_list: list[float] = []
    owner_earnings_list: list[float] = []
    dates: list[str] = []

    for i in range(n):
        capex = _safe_get(cf_data[i], "capitalExpenditure")
        dep = _safe_get(income_data[i], "depreciationAndAmortization")
        revenue = _safe_get(income_data[i], "revenue")
        ocf = _safe_get(cf_data[i], "operatingCashFlow")
        net_income = _safe_get(income_data[i], "netIncome")

        abs_capex = abs(capex)
        maintenance = dep  # D&A as proxy for maintenance CapEx
        growth = max(abs_capex - maintenance, 0.0)

        capex_rev = _safe_div(abs_capex, revenue) if revenue > 0 else 0.0
        capex_ocf = _safe_div(abs_capex, ocf) if ocf > 0 else 0.0
        growth_pct = _safe_div(growth, abs_capex) if abs_capex > 0 else 0.0

        # Owner earnings = NI + D&A - maintenance CapEx
        owner_earn = net_income + dep - maintenance

        total_capex_list.append(float(capex))
        depreciation_list.append(float(dep))
        maintenance_list.append(float(maintenance))
        growth_list.append(float(growth))
        capex_rev_list.append(float(capex_rev))
        capex_ocf_list.append(float(capex_ocf))
        growth_pct_list.append(float(growth_pct))
        owner_earnings_list.append(float(owner_earn))
        dates.append(_safe_get_str(cf_data[i], "date"))

    # CapEx trend based on CapEx/Revenue
    if len(capex_rev_list) >= 2:
        change = capex_rev_list[0] - capex_rev_list[-1]
        if change > 0.02:
            capex_trend = "increasing"
        elif change < -0.02:
            capex_trend = "decreasing"
        else:
            capex_trend = "stable"
    else:
        capex_trend = "unknown"

    return {
        "symbol": symbol,
        "periods_analysed": n,
        "total_capex": total_capex_list,
        "depreciation": depreciation_list,
        "maintenance_capex": maintenance_list,
        "growth_capex": growth_list,
        "capex_to_revenue": capex_rev_list,
        "capex_to_ocf": capex_ocf_list,
        "growth_capex_pct": growth_pct_list,
        "owner_earnings": owner_earnings_list,
        "capex_trend": capex_trend,
        "dates": dates,
    }


# ---------------------------------------------------------------------------
# Shareholder Returns
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def shareholder_returns(
    symbol: str,
    *,
    periods: int = 5,
    fmp_client: Any | None = None,
) -> dict[str, Any]:
    """Analyse total shareholder yield: dividends + buybacks + debt reduction.

    Total shareholder yield captures *all* cash returned to shareholders,
    not just dividends.  In the modern market, share buybacks often exceed
    dividends by a wide margin (e.g., Apple returns 3-4x more via buybacks
    than dividends).  Focusing only on dividend yield misses half the story.

    When to use:
        - Income-focused investing: total yield (dividends + buybacks)
          is a better income proxy than dividend yield alone.
        - Sustainability analysis: is the company returning more cash
          than it generates?  Payout ratio > 100 % is unsustainable.
        - Shareholder-friendly management: track trends in capital
          allocation policy.
        - Factor construction: total yield is a more predictive value
          signal than dividend yield.

    Mathematical formulations:
        Total Yield = (Dividends + Buybacks) / Market Cap
        Payout Ratio = (Dividends + Buybacks) / Net Income
        FCF Payout Ratio = (Dividends + Buybacks) / FCF
        Buyback Yield = Net Buybacks / Market Cap
        Dividend Yield = Dividends / Market Cap

    Parameters:
        symbol: Ticker symbol (e.g., ``"AAPL"``).
        periods: Number of annual periods to analyse.
        fmp_client: Optional pre-configured ``FMPClient`` instance.

    Returns:
        Dictionary containing:
        - **symbol** (*str*) -- The ticker analysed.
        - **periods_analysed** (*int*) -- Actual periods with data.
        - **dividends_paid** (*list[float]*) -- Absolute dividends by period.
        - **buybacks** (*list[float]*) -- Absolute buybacks by period.
        - **total_returned** (*list[float]*) -- Dividends + buybacks.
        - **dividend_yield** (*float*) -- Current dividend yield (TTM).
        - **buyback_yield** (*float*) -- Most recent buyback yield.
        - **total_yield** (*float*) -- Dividend + buyback yield combined.
        - **payout_ratio** (*list[float]*) -- Total returned / net income.
          > 1.0 means returning more than earned (using balance sheet).
        - **fcf_payout_ratio** (*list[float]*) -- Total returned / FCF.
          > 1.0 means returning more than free cash flow.
        - **sustainability** (*str*) -- ``"sustainable"`` if FCF payout
          < 80 %, ``"caution"`` if 80--120 %, ``"unsustainable"`` if > 120 %.
        - **trend** (*str*) -- ``"increasing"`` if total returned is
          growing, ``"decreasing"`` or ``"stable"``.
        - **dates** (*list[str]*) -- Period end dates.

    Example:
        >>> from wraquant.fundamental.financials import shareholder_returns
        >>> sr = shareholder_returns("AAPL")
        >>> print(f"Total yield: {sr['total_yield']:.2%}")
        >>> print(f"Sustainability: {sr['sustainability']}")
        >>> for i, d in enumerate(sr['dates'][:3]):
        ...     print(f"  {d}: ${sr['total_returned'][i]:,.0f}")

    References:
        Mauboussin, M. J. & Callahan, D. (2014). "Capital Allocation:
        Evidence, Analytical Methods, and Assessment Guidance."
        *Credit Suisse Global Financial Strategies*.

    See Also:
        cash_flow_analysis: Broader cash flow metrics.
        capex_analysis: How CapEx competes with shareholder returns.
    """
    client = _get_fmp_client(fmp_client)

    cf_data = _safe_get_list(client.cash_flow(symbol, period="annual", limit=periods))
    income_data = _safe_get_list(
        client.income_statement(symbol, period="annual", limit=periods)
    )
    profile = client.company_profile(symbol)

    profile_data = profile[0] if isinstance(profile, list) and profile else profile
    mkt_cap = (
        _safe_get(profile_data, "mktCap") if isinstance(profile_data, dict) else 0.0
    )

    n = min(len(cf_data), len(income_data))
    if n == 0:
        return {
            "symbol": symbol,
            "periods_analysed": 0,
            "dividends_paid": [],
            "buybacks": [],
            "total_returned": [],
            "dividend_yield": 0.0,
            "buyback_yield": 0.0,
            "total_yield": 0.0,
            "payout_ratio": [],
            "fcf_payout_ratio": [],
            "sustainability": "unknown",
            "trend": "unknown",
            "dates": [],
        }

    divs_list: list[float] = []
    buybacks_list: list[float] = []
    total_returned_list: list[float] = []
    payout_ratio_list: list[float] = []
    fcf_payout_list: list[float] = []
    dates: list[str] = []

    for i in range(n):
        divs = abs(_safe_get(cf_data[i], "dividendsPaid"))
        buybacks_raw = _safe_get(cf_data[i], "commonStockRepurchased")
        buybacks = abs(buybacks_raw)
        total = divs + buybacks
        net_income = _safe_get(income_data[i], "netIncome")
        fcf = _safe_get(cf_data[i], "freeCashFlow")

        payout = _safe_div(total, net_income) if net_income > 0 else 0.0
        fcf_payout = _safe_div(total, fcf) if fcf > 0 else 0.0

        divs_list.append(float(divs))
        buybacks_list.append(float(buybacks))
        total_returned_list.append(float(total))
        payout_ratio_list.append(float(payout))
        fcf_payout_list.append(float(fcf_payout))
        dates.append(_safe_get_str(cf_data[i], "date"))

    # Yields based on most recent period and current market cap
    latest_divs = divs_list[0] if divs_list else 0.0
    latest_buybacks = buybacks_list[0] if buybacks_list else 0.0
    div_yield = _safe_div(latest_divs, mkt_cap) if mkt_cap > 0 else 0.0
    buyback_yield = _safe_div(latest_buybacks, mkt_cap) if mkt_cap > 0 else 0.0
    total_yield = div_yield + buyback_yield

    # Sustainability
    latest_fcf_payout = fcf_payout_list[0] if fcf_payout_list else 0.0
    if latest_fcf_payout < 0.80:
        sustainability = "sustainable"
    elif latest_fcf_payout < 1.20:
        sustainability = "caution"
    else:
        sustainability = "unsustainable"

    # Trend
    if len(total_returned_list) >= 2:
        change = _pct_change(total_returned_list[0], total_returned_list[-1])
        if change > 0.10:
            trend = "increasing"
        elif change < -0.10:
            trend = "decreasing"
        else:
            trend = "stable"
    else:
        trend = "unknown"

    return {
        "symbol": symbol,
        "periods_analysed": n,
        "dividends_paid": divs_list,
        "buybacks": buybacks_list,
        "total_returned": total_returned_list,
        "dividend_yield": float(div_yield),
        "buyback_yield": float(buyback_yield),
        "total_yield": float(total_yield),
        "payout_ratio": payout_ratio_list,
        "fcf_payout_ratio": fcf_payout_list,
        "sustainability": sustainability,
        "trend": trend,
        "dates": dates,
    }
