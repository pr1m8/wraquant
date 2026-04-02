"""Financial ratio analysis using FMP data.

Provides comprehensive financial ratio computation across six categories:
profitability, liquidity, leverage, efficiency, valuation, and growth.
Includes DuPont decomposition (3-way and 5-way) and a convenience function
that aggregates all ratios into a single dictionary.

All functions accept a ticker symbol and optionally an FMP client instance.
When no client is provided, one is created automatically (requires the
``market-data`` extra and an FMP API key in the environment).

The ratios computed here form the foundation of:
- **Value investing**: P/E, P/B, EV/EBITDA for identifying underpriced assets.
- **Quality screening**: ROE, ROIC, margins for separating winners from losers.
- **Factor models**: Fama-French HML (P/B), RMW (profitability), CMA (investment).
- **Credit analysis**: leverage and liquidity ratios for default prediction.

Example:
    >>> from wraquant.fundamental.ratios import comprehensive_ratios
    >>> ratios = comprehensive_ratios("AAPL")
    >>> print(f"ROE: {ratios['profitability']['roe']:.2%}")
    >>> print(f"D/E: {ratios['leverage']['debt_to_equity']:.2f}")

References:
    - Fama, E. F. & French, K. R. (1993). "Common risk factors in the
      returns on stocks and bonds." *Journal of Financial Economics*, 33, 3--56.
    - Piotroski, J. D. (2000). "Value Investing: The Use of Historical
      Financial Statement Information to Separate Winners from Losers."
      *Journal of Accounting Research*, 38, 1--41.
    - Palepu, K. G. & Healy, P. M. (2013). *Business Analysis and
      Valuation*, 5th edition.
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
    """Divide *numerator* by *denominator*, returning *default* on zero/near-zero."""
    if abs(denominator) < 1e-12:
        return default
    return float(numerator / denominator)


def _get_fmp_client(fmp_client: Any | None = None) -> Any:
    """Return the provided client or construct a default one."""
    if fmp_client is not None:
        return fmp_client
    from wraquant.data.providers.fmp import FMPClient  # noqa: WPS433

    return FMPClient()


def _safe_get(data: dict | list, key: str, default: float = 0.0) -> float:
    """Extract a numeric value from an FMP response dict/list.

    FMP endpoints often return a list with a single dict, or a plain dict.
    This helper handles both shapes and coerces ``None`` to *default*.
    """
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
    """Ensure *data* is a list of dicts."""
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    return []


# ---------------------------------------------------------------------------
# Profitability ratios
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def profitability_ratios(
    symbol: str,
    *,
    period: str = "annual",
    fmp_client: Any | None = None,
) -> dict[str, float]:
    """Compute core profitability ratios for a company.

    Profitability ratios measure how effectively a company converts
    revenue into profit at various stages of the income statement.  Use
    these to compare operating efficiency across peers and to track
    margin trends over time.

    Mathematical formulations:
        ROE   = Net Income / Shareholders' Equity
        ROA   = Net Income / Total Assets
        ROIC  = NOPAT / Invested Capital
              = EBIT * (1 - tax rate) / (Total Debt + Equity - Cash)
        Gross Margin     = Gross Profit / Revenue
        Operating Margin = Operating Income / Revenue
        Net Margin       = Net Income / Revenue

    Parameters:
        symbol: Ticker symbol (e.g., ``"AAPL"``).
        period: ``"annual"`` or ``"quarter"``.  Annual data is more
            stable; quarterly reveals recent trends.
        fmp_client: Optional pre-configured ``FMPClient`` instance.
            If ``None``, a default client is created.

    Returns:
        Dictionary containing:
        - **roe** (*float*) -- Return on equity.  > 0.15 is generally strong.
        - **roa** (*float*) -- Return on assets.  > 0.05 is solid.
        - **roic** (*float*) -- Return on invested capital.  > WACC means
          the company creates value.
        - **gross_margin** (*float*) -- Gross profit / revenue.
        - **operating_margin** (*float*) -- Operating income / revenue.
        - **net_margin** (*float*) -- Net income / revenue.
        - **period** (*str*) -- The period used.

    Example:
        >>> from wraquant.fundamental.ratios import profitability_ratios
        >>> p = profitability_ratios("MSFT")
        >>> print(f"ROE: {p['roe']:.2%}, ROIC: {p['roic']:.2%}")

    See Also:
        dupont_decomposition: Breaks ROE into its drivers.
        efficiency_ratios: Asset utilisation metrics.
    """
    client = _get_fmp_client(fmp_client)

    income = client.income_statement(symbol, period=period)
    balance = client.balance_sheet(symbol, period=period)
    ratios_data = client.ratios(symbol, period=period)

    # Extract values from latest period
    revenue = _safe_get(income, "revenue")
    gross_profit = _safe_get(income, "grossProfit")
    operating_income = _safe_get(income, "operatingIncome")
    net_income = _safe_get(income, "netIncome")
    ebit = _safe_get(income, "operatingIncome")
    income_tax = _safe_get(income, "incomeTaxExpense")
    income_before_tax = _safe_get(income, "incomeBeforeTax")

    total_assets = _safe_get(balance, "totalAssets")
    total_equity = _safe_get(balance, "totalStockholdersEquity")
    total_debt = _safe_get(balance, "totalDebt")
    cash = _safe_get(balance, "cashAndCashEquivalents")

    # Effective tax rate
    tax_rate = _safe_div(income_tax, income_before_tax, default=0.21)
    tax_rate = max(0.0, min(tax_rate, 1.0))  # clamp to [0, 1]

    # ROIC: NOPAT / Invested Capital
    nopat = ebit * (1.0 - tax_rate)
    invested_capital = total_debt + total_equity - cash
    roic = _safe_div(nopat, invested_capital)

    # Prefer FMP-computed ratios when available, fall back to manual calc
    roe_val = _safe_get(ratios_data, "returnOnEquity", default=None)  # type: ignore[arg-type]
    if roe_val is None or roe_val == 0.0:
        roe_val = _safe_div(net_income, total_equity)
    roa_val = _safe_get(ratios_data, "returnOnAssets", default=None)  # type: ignore[arg-type]
    if roa_val is None or roa_val == 0.0:
        roa_val = _safe_div(net_income, total_assets)

    return {
        "roe": roe_val,
        "roa": roa_val,
        "roic": roic,
        "gross_margin": _safe_div(gross_profit, revenue),
        "operating_margin": _safe_div(operating_income, revenue),
        "net_margin": _safe_div(net_income, revenue),
        "period": period,
    }


# ---------------------------------------------------------------------------
# Liquidity ratios
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def liquidity_ratios(
    symbol: str,
    *,
    period: str = "annual",
    fmp_client: Any | None = None,
) -> dict[str, float]:
    """Compute short-term liquidity ratios.

    Liquidity ratios assess a company's ability to meet its short-term
    obligations.  They are critical for credit analysis, bankruptcy
    prediction (Altman Z-Score), and the Piotroski F-Score.

    Mathematical formulations:
        Current Ratio = Current Assets / Current Liabilities
        Quick Ratio   = (Current Assets - Inventory) / Current Liabilities
        Cash Ratio    = Cash & Equivalents / Current Liabilities
        Working Capital = Current Assets - Current Liabilities

    Parameters:
        symbol: Ticker symbol.
        period: ``"annual"`` or ``"quarter"``.
        fmp_client: Optional ``FMPClient`` instance.

    Returns:
        Dictionary containing:
        - **current_ratio** (*float*) -- > 1.5 is healthy; < 1.0 is a red flag.
        - **quick_ratio** (*float*) -- Acid-test; excludes illiquid inventory.
        - **cash_ratio** (*float*) -- Most conservative; cash only.
        - **working_capital** (*float*) -- Absolute dollar liquidity buffer.
        - **period** (*str*) -- The period used.

    Example:
        >>> from wraquant.fundamental.ratios import liquidity_ratios
        >>> liq = liquidity_ratios("AAPL")
        >>> print(f"Current ratio: {liq['current_ratio']:.2f}")

    References:
        Altman, E. I. (1968). "Financial Ratios, Discriminant Analysis and
        the Prediction of Corporate Bankruptcy." *Journal of Finance*, 23(4),
        589--609.

    See Also:
        leverage_ratios: Long-term solvency.
    """
    client = _get_fmp_client(fmp_client)
    balance = client.balance_sheet(symbol, period=period)

    current_assets = _safe_get(balance, "totalCurrentAssets")
    current_liabilities = _safe_get(balance, "totalCurrentLiabilities")
    inventory = _safe_get(balance, "inventory")
    cash = _safe_get(balance, "cashAndCashEquivalents")

    return {
        "current_ratio": _safe_div(current_assets, current_liabilities),
        "quick_ratio": _safe_div(current_assets - inventory, current_liabilities),
        "cash_ratio": _safe_div(cash, current_liabilities),
        "working_capital": current_assets - current_liabilities,
        "period": period,
    }


# ---------------------------------------------------------------------------
# Leverage ratios
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def leverage_ratios(
    symbol: str,
    *,
    period: str = "annual",
    fmp_client: Any | None = None,
) -> dict[str, float]:
    """Compute leverage and solvency ratios.

    Leverage ratios measure the extent to which a company uses debt to
    finance its assets.  Higher leverage amplifies both returns and risk.
    These ratios are essential for:
    - Credit risk modeling (probability of default).
    - Merton structural models (distance to default).
    - Factor investing (leverage as a risk factor).

    Mathematical formulations:
        Debt-to-Equity  = Total Debt / Shareholders' Equity
        Debt Ratio      = Total Debt / Total Assets
        Interest Coverage = EBIT / Interest Expense
        Equity Multiplier = Total Assets / Shareholders' Equity
        Debt-to-EBITDA  = Total Debt / EBITDA

    Parameters:
        symbol: Ticker symbol.
        period: ``"annual"`` or ``"quarter"``.
        fmp_client: Optional ``FMPClient`` instance.

    Returns:
        Dictionary containing:
        - **debt_to_equity** (*float*) -- D/E ratio; > 2.0 is high leverage.
        - **debt_ratio** (*float*) -- Portion of assets financed by debt.
        - **interest_coverage** (*float*) -- EBIT / interest; < 1.5 is
          distressed.  Higher is safer.
        - **equity_multiplier** (*float*) -- Assets per dollar of equity;
          captures leverage in DuPont.
        - **debt_to_ebitda** (*float*) -- How many years of EBITDA to repay
          debt; < 3 is conservative.
        - **period** (*str*) -- The period used.

    Example:
        >>> from wraquant.fundamental.ratios import leverage_ratios
        >>> lev = leverage_ratios("AAPL")
        >>> print(f"D/E: {lev['debt_to_equity']:.2f}")

    See Also:
        liquidity_ratios: Short-term solvency.
        dupont_decomposition: How leverage drives ROE.
    """
    client = _get_fmp_client(fmp_client)
    balance = client.balance_sheet(symbol, period=period)
    income = client.income_statement(symbol, period=period)

    total_debt = _safe_get(balance, "totalDebt")
    total_equity = _safe_get(balance, "totalStockholdersEquity")
    total_assets = _safe_get(balance, "totalAssets")
    ebit = _safe_get(income, "operatingIncome")
    interest_expense = _safe_get(income, "interestExpense")
    depreciation = _safe_get(income, "depreciationAndAmortization")

    ebitda = ebit + depreciation

    return {
        "debt_to_equity": _safe_div(total_debt, total_equity),
        "debt_ratio": _safe_div(total_debt, total_assets),
        "interest_coverage": _safe_div(
            ebit, abs(interest_expense) if interest_expense else 0.0
        ),
        "equity_multiplier": _safe_div(total_assets, total_equity),
        "debt_to_ebitda": _safe_div(total_debt, ebitda),
        "period": period,
    }


# ---------------------------------------------------------------------------
# Efficiency ratios
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def efficiency_ratios(
    symbol: str,
    *,
    period: str = "annual",
    fmp_client: Any | None = None,
) -> dict[str, float]:
    """Compute asset utilisation and efficiency ratios.

    Efficiency ratios (also called activity ratios) measure how
    effectively a company uses its assets to generate revenue.  They
    are the ``turnover`` component in DuPont analysis and are critical
    for comparing capital-light vs. capital-intensive businesses.

    Mathematical formulations:
        Asset Turnover      = Revenue / Total Assets
        Inventory Turnover  = COGS / Average Inventory
        Receivable Turnover = Revenue / Accounts Receivable
        Payable Turnover    = COGS / Accounts Payable
        Days Sales Outstanding = 365 / Receivable Turnover
        Days Inventory Outstanding = 365 / Inventory Turnover
        Days Payable Outstanding = 365 / Payable Turnover
        Cash Conversion Cycle = DSO + DIO - DPO

    Parameters:
        symbol: Ticker symbol.
        period: ``"annual"`` or ``"quarter"``.
        fmp_client: Optional ``FMPClient`` instance.

    Returns:
        Dictionary containing:
        - **asset_turnover** (*float*) -- Revenue per dollar of assets.
        - **inventory_turnover** (*float*) -- How many times inventory
          is sold per period.  Higher is better for retail.
        - **receivable_turnover** (*float*) -- How quickly receivables
          convert to cash.
        - **payable_turnover** (*float*) -- How quickly the company pays
          suppliers.
        - **days_sales_outstanding** (*float*) -- Average collection period
          in days.
        - **days_inventory_outstanding** (*float*) -- Average days to sell
          inventory.
        - **days_payable_outstanding** (*float*) -- Average days to pay
          suppliers.
        - **cash_conversion_cycle** (*float*) -- DSO + DIO - DPO; shorter
          is better.  Negative means the company is funded by suppliers
          (e.g., Amazon).
        - **period** (*str*) -- The period used.

    Example:
        >>> from wraquant.fundamental.ratios import efficiency_ratios
        >>> eff = efficiency_ratios("WMT")
        >>> print(f"Inventory turnover: {eff['inventory_turnover']:.1f}x")
        >>> print(f"Cash conversion cycle: {eff['cash_conversion_cycle']:.0f} days")

    See Also:
        dupont_decomposition: Efficiency is a component of ROE.
    """
    client = _get_fmp_client(fmp_client)
    income = client.income_statement(symbol, period=period)
    balance = client.balance_sheet(symbol, period=period)

    revenue = _safe_get(income, "revenue")
    cogs = _safe_get(income, "costOfRevenue")
    total_assets = _safe_get(balance, "totalAssets")
    inventory = _safe_get(balance, "inventory")
    receivables = _safe_get(balance, "netReceivables")
    payables = _safe_get(balance, "accountPayables")

    asset_turnover = _safe_div(revenue, total_assets)
    inventory_turnover = _safe_div(cogs, inventory)
    receivable_turnover = _safe_div(revenue, receivables)
    payable_turnover = _safe_div(cogs, payables)

    dso = _safe_div(365.0, receivable_turnover) if receivable_turnover else 0.0
    dio = _safe_div(365.0, inventory_turnover) if inventory_turnover else 0.0
    dpo = _safe_div(365.0, payable_turnover) if payable_turnover else 0.0

    return {
        "asset_turnover": asset_turnover,
        "inventory_turnover": inventory_turnover,
        "receivable_turnover": receivable_turnover,
        "payable_turnover": payable_turnover,
        "days_sales_outstanding": dso,
        "days_inventory_outstanding": dio,
        "days_payable_outstanding": dpo,
        "cash_conversion_cycle": dso + dio - dpo,
        "period": period,
    }


# ---------------------------------------------------------------------------
# Valuation ratios
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def valuation_ratios(
    symbol: str,
    *,
    fmp_client: Any | None = None,
) -> dict[str, float]:
    """Compute market-based valuation multiples.

    Valuation ratios relate the market price to fundamentals.  They are
    the workhorses of relative valuation and the value factor in
    Fama-French models.  Use them to compare a stock to its sector
    median, its own history, or the broad market.

    Mathematical formulations:
        P/E        = Price / Earnings Per Share
        P/B        = Price / Book Value Per Share
        P/S        = Price / Revenue Per Share
        EV/EBITDA  = Enterprise Value / EBITDA
        PEG        = P/E / EPS Growth Rate (%)
        Div Yield  = Dividend Per Share / Price

    Parameters:
        symbol: Ticker symbol.
        fmp_client: Optional ``FMPClient`` instance.

    Returns:
        Dictionary containing:
        - **pe_ratio** (*float*) -- Price-to-earnings.  Market median ~15--20.
        - **pb_ratio** (*float*) -- Price-to-book.
        - **ps_ratio** (*float*) -- Price-to-sales.  Useful for unprofitable
          companies.
        - **ev_to_ebitda** (*float*) -- Enterprise value / EBITDA.  < 10 is
          often considered cheap.
        - **peg_ratio** (*float*) -- Growth-adjusted P/E.  < 1 may be
          undervalued relative to growth.
        - **dividend_yield** (*float*) -- Annual dividend / price.
        - **earnings_yield** (*float*) -- Inverse of P/E; comparable to
          bond yields.
        - **price_to_fcf** (*float*) -- Price / free cash flow per share.

    Example:
        >>> from wraquant.fundamental.ratios import valuation_ratios
        >>> val = valuation_ratios("AAPL")
        >>> print(f"P/E: {val['pe_ratio']:.1f}, EV/EBITDA: {val['ev_to_ebitda']:.1f}")

    References:
        Fama, E. F. & French, K. R. (1992). "The Cross-Section of Expected
        Stock Returns." *Journal of Finance*, 47(2), 427--465.

    See Also:
        relative_valuation: Compare multiples to peers.
    """
    client = _get_fmp_client(fmp_client)
    ratios_data = client.ratios_ttm(symbol)
    metrics = client.key_metrics(symbol)
    ev_data = client.enterprise_value(symbol)

    pe = _safe_get(ratios_data, "peRatioTTM")
    pb = _safe_get(ratios_data, "priceToBookRatioTTM")
    ps = _safe_get(ratios_data, "priceToSalesRatioTTM")
    peg = _safe_get(ratios_data, "pegRatioTTM")
    dividend_yield = _safe_get(ratios_data, "dividendYieldTTM")
    price_to_fcf = _safe_get(ratios_data, "priceToFreeCashFlowsRatioTTM")

    ev_to_ebitda = _safe_get(metrics, "evToOperatingCashFlow")
    # Prefer direct EV/EBITDA from enterprise value data
    ev_ebitda_direct = _safe_get(ev_data, "evToEBITDA")
    if ev_ebitda_direct and ev_ebitda_direct != 0.0:
        ev_to_ebitda = ev_ebitda_direct

    earnings_yield = _safe_div(1.0, pe) if pe and pe != 0.0 else 0.0

    return {
        "pe_ratio": pe,
        "pb_ratio": pb,
        "ps_ratio": ps,
        "ev_to_ebitda": ev_to_ebitda,
        "peg_ratio": peg,
        "dividend_yield": dividend_yield,
        "earnings_yield": earnings_yield,
        "price_to_fcf": price_to_fcf,
    }


# ---------------------------------------------------------------------------
# Growth ratios
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def growth_ratios(
    symbol: str,
    *,
    period: str = "annual",
    fmp_client: Any | None = None,
) -> dict[str, float | list[float]]:
    """Compute revenue, earnings, and dividend growth rates.

    Growth ratios quantify the trajectory of a company's top line,
    bottom line, and shareholder distributions.  They are core inputs
    to the PEG ratio, DCF terminal growth assumptions, and growth
    factor construction.

    Mathematical formulations:
        Revenue Growth = (Revenue_t - Revenue_{t-1}) / Revenue_{t-1}
        EPS Growth     = (EPS_t - EPS_{t-1}) / EPS_{t-1}
        Dividend Growth = (DPS_t - DPS_{t-1}) / DPS_{t-1}
        EBITDA Growth  = (EBITDA_t - EBITDA_{t-1}) / EBITDA_{t-1}
        FCF Growth     = (FCF_t - FCF_{t-1}) / FCF_{t-1}

    Parameters:
        symbol: Ticker symbol.
        period: ``"annual"`` or ``"quarter"``.  Annual is less noisy;
            quarterly captures recent momentum.
        fmp_client: Optional ``FMPClient`` instance.

    Returns:
        Dictionary containing:
        - **revenue_growth** (*float*) -- Most recent period YoY revenue
          growth.
        - **eps_growth** (*float*) -- Most recent period YoY EPS growth.
        - **dividend_growth** (*float*) -- Most recent period YoY dividend
          growth.
        - **ebitda_growth** (*float*) -- Most recent period YoY EBITDA growth.
        - **fcf_growth** (*float*) -- Most recent period YoY FCF growth.
        - **revenue_growth_3y** (*float*) -- 3-year CAGR of revenue.
        - **revenue_growth_5y** (*float*) -- 5-year CAGR of revenue.
        - **revenue_growth_history** (*list[float]*) -- Growth for each
          available period.
        - **period** (*str*) -- The period used.

    Example:
        >>> from wraquant.fundamental.ratios import growth_ratios
        >>> g = growth_ratios("NVDA")
        >>> print(f"Revenue growth: {g['revenue_growth']:.1%}")
        >>> print(f"3Y CAGR: {g['revenue_growth_3y']:.1%}")

    See Also:
        valuation_ratios: PEG uses growth.
    """
    client = _get_fmp_client(fmp_client)
    growth_data = client.financial_growth(symbol, period=period)

    rows = _safe_get_list(growth_data)

    # Most recent period
    latest = rows[0] if rows else {}
    revenue_growth = _safe_get(latest, "revenueGrowth")
    eps_growth = _safe_get(latest, "epsgrowth")
    dividend_growth = _safe_get(latest, "dividendsperShareGrowth")
    ebitda_growth = _safe_get(latest, "ebitdagrowth")
    fcf_growth = _safe_get(latest, "freeCashFlowGrowth")

    # Historical revenue growth for CAGR calculation
    rev_history = [_safe_get(r, "revenueGrowth") for r in rows]

    # CAGR helpers: use income statement revenue across periods
    income_list = _safe_get_list(
        client.income_statement(symbol, period=period, limit=10)
    )
    revenues = [_safe_get(r, "revenue") for r in income_list]

    def _cagr(values: list[float], years: int) -> float:
        """Compute compound annual growth rate over *years* periods."""
        if len(values) <= years or values[years] <= 0 or values[0] <= 0:
            return 0.0
        # values[0] is most recent, values[years] is *years* periods ago
        return (values[0] / values[years]) ** (1.0 / years) - 1.0

    return {
        "revenue_growth": revenue_growth,
        "eps_growth": eps_growth,
        "dividend_growth": dividend_growth,
        "ebitda_growth": ebitda_growth,
        "fcf_growth": fcf_growth,
        "revenue_growth_3y": _cagr(revenues, 3),
        "revenue_growth_5y": _cagr(revenues, 5),
        "revenue_growth_history": rev_history,
        "period": period,
    }


# ---------------------------------------------------------------------------
# DuPont decomposition
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def dupont_decomposition(
    symbol: str,
    *,
    period: str = "annual",
    fmp_client: Any | None = None,
) -> dict[str, float]:
    """Perform 3-way and 5-way DuPont decomposition of ROE.

    DuPont analysis decomposes return on equity into its fundamental
    drivers, revealing *why* a company's ROE is high or low.  This is
    essential for distinguishing between companies that earn high ROE
    through operational excellence vs. financial leverage.

    Mathematical formulations:

    **3-way DuPont:**
        ROE = Net Margin x Asset Turnover x Equity Multiplier
            = (NI / Rev) x (Rev / Assets) x (Assets / Equity)

    **5-way DuPont (extended):**
        ROE = Tax Burden x Interest Burden x Operating Margin
              x Asset Turnover x Equity Multiplier
            = (NI / EBT) x (EBT / EBIT) x (EBIT / Rev)
              x (Rev / Assets) x (Assets / Equity)

    Parameters:
        symbol: Ticker symbol.
        period: ``"annual"`` or ``"quarter"``.
        fmp_client: Optional ``FMPClient`` instance.

    Returns:
        Dictionary containing:

        **3-way components:**
        - **net_margin** (*float*) -- NI / Revenue.
        - **asset_turnover** (*float*) -- Revenue / Total Assets.
        - **equity_multiplier** (*float*) -- Total Assets / Equity.
        - **roe_3way** (*float*) -- Product of the three components.

        **5-way components:**
        - **tax_burden** (*float*) -- NI / EBT.  Closer to 1.0 means lower
          effective tax rate.
        - **interest_burden** (*float*) -- EBT / EBIT.  Closer to 1.0 means
          less interest cost relative to operating profit.
        - **operating_margin** (*float*) -- EBIT / Revenue.
        - **roe_5way** (*float*) -- Product of the five components.
        - **period** (*str*) -- The period used.

    Example:
        >>> from wraquant.fundamental.ratios import dupont_decomposition
        >>> dp = dupont_decomposition("AAPL")
        >>> print(f"ROE (3-way): {dp['roe_3way']:.2%}")
        >>> print(f"  = {dp['net_margin']:.2%} margin")
        >>> print(f"  x {dp['asset_turnover']:.2f} turnover")
        >>> print(f"  x {dp['equity_multiplier']:.2f} leverage")

    References:
        Soliman, M. T. (2008). "The Use of DuPont Analysis by Market
        Participants." *The Accounting Review*, 83(3), 823--853.

    See Also:
        profitability_ratios: Individual profitability metrics.
        leverage_ratios: Detailed leverage analysis.
    """
    client = _get_fmp_client(fmp_client)
    income = client.income_statement(symbol, period=period)
    balance = client.balance_sheet(symbol, period=period)

    revenue = _safe_get(income, "revenue")
    net_income = _safe_get(income, "netIncome")
    ebit = _safe_get(income, "operatingIncome")
    ebt = _safe_get(income, "incomeBeforeTax")
    total_assets = _safe_get(balance, "totalAssets")
    total_equity = _safe_get(balance, "totalStockholdersEquity")

    # 3-way
    net_margin = _safe_div(net_income, revenue)
    asset_turnover = _safe_div(revenue, total_assets)
    equity_multiplier = _safe_div(total_assets, total_equity)
    roe_3way = net_margin * asset_turnover * equity_multiplier

    # 5-way
    tax_burden = _safe_div(net_income, ebt)
    interest_burden = _safe_div(ebt, ebit)
    operating_margin = _safe_div(ebit, revenue)
    roe_5way = (
        tax_burden
        * interest_burden
        * operating_margin
        * asset_turnover
        * equity_multiplier
    )

    return {
        # 3-way
        "net_margin": net_margin,
        "asset_turnover": asset_turnover,
        "equity_multiplier": equity_multiplier,
        "roe_3way": roe_3way,
        # 5-way
        "tax_burden": tax_burden,
        "interest_burden": interest_burden,
        "operating_margin": operating_margin,
        "roe_5way": roe_5way,
        "period": period,
    }


# ---------------------------------------------------------------------------
# Comprehensive ratios
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def comprehensive_ratios(
    symbol: str,
    *,
    period: str = "annual",
    fmp_client: Any | None = None,
) -> dict[str, Any]:
    """Compute all financial ratios in a single call.

    Convenience function that aggregates profitability, liquidity,
    leverage, efficiency, valuation, growth, and DuPont ratios into a
    single nested dictionary.  Useful for building screening dashboards
    and factor databases.

    Parameters:
        symbol: Ticker symbol.
        period: ``"annual"`` or ``"quarter"`` (used for non-valuation
            ratios; valuation ratios always use TTM).
        fmp_client: Optional ``FMPClient`` instance.  Passing one
            avoids creating multiple clients.

    Returns:
        Nested dictionary with keys:
        - **profitability** -- From :func:`profitability_ratios`.
        - **liquidity** -- From :func:`liquidity_ratios`.
        - **leverage** -- From :func:`leverage_ratios`.
        - **efficiency** -- From :func:`efficiency_ratios`.
        - **valuation** -- From :func:`valuation_ratios`.
        - **growth** -- From :func:`growth_ratios`.
        - **dupont** -- From :func:`dupont_decomposition`.
        - **symbol** (*str*) -- The ticker analysed.
        - **period** (*str*) -- The period used.

    Example:
        >>> from wraquant.fundamental.ratios import comprehensive_ratios
        >>> all_ratios = comprehensive_ratios("AAPL")
        >>> print(f"ROE: {all_ratios['profitability']['roe']:.2%}")
        >>> print(f"D/E: {all_ratios['leverage']['debt_to_equity']:.2f}")
        >>> print(f"P/E: {all_ratios['valuation']['pe_ratio']:.1f}")

    See Also:
        financial_health_score: Composite score from these ratios.
    """
    client = _get_fmp_client(fmp_client)
    kwargs: dict[str, Any] = {"fmp_client": client}

    return {
        "symbol": symbol,
        "period": period,
        "profitability": profitability_ratios(symbol, period=period, **kwargs),
        "liquidity": liquidity_ratios(symbol, period=period, **kwargs),
        "leverage": leverage_ratios(symbol, period=period, **kwargs),
        "efficiency": efficiency_ratios(symbol, period=period, **kwargs),
        "valuation": valuation_ratios(symbol, **kwargs),
        "growth": growth_ratios(symbol, period=period, **kwargs),
        "dupont": dupont_decomposition(symbol, period=period, **kwargs),
    }


# ---------------------------------------------------------------------------
# Ratio Comparison (peer benchmarking)
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def ratio_comparison(
    symbol: str,
    *,
    peers: list[str] | None = None,
    fmp_client: Any | None = None,
) -> dict[str, Any]:
    """Compare a stock's key ratios to its peer group with percentile ranking.

    Relative ratio analysis is the backbone of peer-group valuation and
    factor-based stock selection.  Rather than asking "is this ratio good
    in absolute terms?", this function asks "how does it rank against
    comparable companies?"  A 20 % ROE might be outstanding in utilities
    but mediocre in tech.

    When to use:
        - Building peer-relative factor scores for multi-factor models.
        - Identifying companies that are outliers within their industry.
        - Due-diligence: confirming whether a stock's ratios are truly
          exceptional or merely in-line with its sector.

    Mathematical formulation:
        Percentile_i = (# peers with ratio <= target_ratio_i) / N_peers

        A percentile of 0.80 means the stock ranks in the 80th
        percentile among its peers on that metric.

    Parameters:
        symbol: Ticker symbol (e.g., ``"AAPL"``).
        peers: List of peer ticker symbols.  If ``None``, peers are
            auto-discovered via FMP's ``stock_peers()`` endpoint,
            which returns companies in the same industry.
        fmp_client: Optional pre-configured ``FMPClient`` instance.

    Returns:
        Dictionary containing:
        - **symbol** (*str*) -- The target ticker.
        - **peers** (*list[str]*) -- Peers used for comparison.
        - **target_ratios** (*dict*) -- The target's key ratios.
        - **peer_averages** (*dict*) -- Mean of each ratio across peers.
        - **peer_medians** (*dict*) -- Median of each ratio across peers.
        - **percentile_rank** (*dict*) -- Percentile rank (0--1) of the
          target vs. peers for each ratio.  Higher is better for
          profitability/efficiency; lower is better for leverage.
        - **peers_detail** (*list[dict]*) -- Individual peer ratios.

    Example:
        >>> from wraquant.fundamental.ratios import ratio_comparison
        >>> comp = ratio_comparison("AAPL")
        >>> print(f"ROE percentile: {comp['percentile_rank']['roe']:.0%}")
        >>> print(f"D/E percentile: {comp['percentile_rank']['debt_to_equity']:.0%}")

    See Also:
        comprehensive_ratios: All ratios for a single company.
        sector_comparison: Compare against sector-wide averages.
    """
    client = _get_fmp_client(fmp_client)

    # Discover peers if not provided
    if peers is None:
        try:
            peers = client.stock_peers(symbol)
        except Exception:  # noqa: BLE001
            logger.warning("Could not fetch peers for %s; using empty list", symbol)
            peers = []
    if not peers:
        logger.warning("No peers found for %s", symbol)

    # Ratios to compare
    ratio_keys = [
        "roe",
        "roa",
        "roic",
        "gross_margin",
        "operating_margin",
        "net_margin",
        "debt_to_equity",
        "debt_ratio",
        "current_ratio",
        "asset_turnover",
    ]

    # Fetch target ratios
    target_prof = profitability_ratios(symbol, fmp_client=client)
    target_lev = leverage_ratios(symbol, fmp_client=client)
    target_liq = liquidity_ratios(symbol, fmp_client=client)
    target_eff = efficiency_ratios(symbol, fmp_client=client)

    target_ratios = {
        "roe": target_prof.get("roe", 0.0),
        "roa": target_prof.get("roa", 0.0),
        "roic": target_prof.get("roic", 0.0),
        "gross_margin": target_prof.get("gross_margin", 0.0),
        "operating_margin": target_prof.get("operating_margin", 0.0),
        "net_margin": target_prof.get("net_margin", 0.0),
        "debt_to_equity": target_lev.get("debt_to_equity", 0.0),
        "debt_ratio": target_lev.get("debt_ratio", 0.0),
        "current_ratio": target_liq.get("current_ratio", 0.0),
        "asset_turnover": target_eff.get("asset_turnover", 0.0),
    }

    # Fetch peer ratios
    peers_detail: list[dict[str, Any]] = []
    for peer in peers:
        try:
            p_prof = profitability_ratios(peer, fmp_client=client)
            p_lev = leverage_ratios(peer, fmp_client=client)
            p_liq = liquidity_ratios(peer, fmp_client=client)
            p_eff = efficiency_ratios(peer, fmp_client=client)
            peer_data = {
                "symbol": peer,
                "roe": p_prof.get("roe", 0.0),
                "roa": p_prof.get("roa", 0.0),
                "roic": p_prof.get("roic", 0.0),
                "gross_margin": p_prof.get("gross_margin", 0.0),
                "operating_margin": p_prof.get("operating_margin", 0.0),
                "net_margin": p_prof.get("net_margin", 0.0),
                "debt_to_equity": p_lev.get("debt_to_equity", 0.0),
                "debt_ratio": p_lev.get("debt_ratio", 0.0),
                "current_ratio": p_liq.get("current_ratio", 0.0),
                "asset_turnover": p_eff.get("asset_turnover", 0.0),
            }
            peers_detail.append(peer_data)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to fetch ratios for peer %s", peer)

    # Compute averages, medians, and percentile ranks
    peer_averages: dict[str, float] = {}
    peer_medians: dict[str, float] = {}
    percentile_rank: dict[str, float] = {}

    for key in ratio_keys:
        values = [p[key] for p in peers_detail if p.get(key) is not None]
        if values:
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            peer_averages[key] = sum(values) / n
            peer_medians[key] = (
                sorted_vals[n // 2]
                if n % 2 == 1
                else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
            )
            # Percentile: fraction of peers the target beats
            target_val = target_ratios[key]
            count_below = sum(1 for v in values if v <= target_val)
            percentile_rank[key] = count_below / n
        else:
            peer_averages[key] = 0.0
            peer_medians[key] = 0.0
            percentile_rank[key] = 0.5  # no data, assume median

    return {
        "symbol": symbol,
        "peers": peers,
        "target_ratios": target_ratios,
        "peer_averages": peer_averages,
        "peer_medians": peer_medians,
        "percentile_rank": percentile_rank,
        "peers_detail": peers_detail,
    }


# ---------------------------------------------------------------------------
# Ratio Trends (multi-year trajectory)
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def ratio_trends(
    symbol: str,
    *,
    periods: int = 5,
    fmp_client: Any | None = None,
) -> dict[str, Any]:
    """Analyse multi-year ratio trends to identify improving or deteriorating fundamentals.

    Static ratios show where a company is *now*; trends show where it is
    *going*.  A company with a 12 % ROE that has risen from 8 % is far more
    attractive than one at 15 % declining from 20 %.  This function computes
    trend direction and magnitude for every major ratio category.

    When to use:
        - Momentum-quality strategies: buy stocks with *improving*
          fundamentals (rising margins, declining leverage).
        - Turnaround detection: identify companies transitioning from
          weak to strong.
        - Risk monitoring: catch early signs of fundamental deterioration
          in existing holdings.

    Mathematical formulation:
        Trend direction = sign(ratio_latest - ratio_oldest)
        Trend magnitude = (ratio_latest - ratio_oldest) / |ratio_oldest|
        Trend is classified as "improving", "deteriorating", or "stable"
        based on a ±5% threshold.

    Parameters:
        symbol: Ticker symbol (e.g., ``"AAPL"``).
        periods: Number of annual periods to analyse.  Default 5 gives
            a full business-cycle view.  Use 3 for recent trajectory.
        fmp_client: Optional pre-configured ``FMPClient`` instance.

    Returns:
        Dictionary containing:
        - **symbol** (*str*) -- The ticker analysed.
        - **periods_analysed** (*int*) -- Actual number of periods with data.
        - **roe** (*dict*) -- ``{"values": [...], "direction": str,
          "magnitude": float}``.  Direction is ``"improving"``,
          ``"deteriorating"``, or ``"stable"``.
        - **roa** (*dict*) -- Same structure as ROE.
        - **gross_margin** (*dict*) -- Margin trajectory.
        - **operating_margin** (*dict*) -- Operating efficiency trend.
        - **net_margin** (*dict*) -- Bottom-line margin trend.
        - **debt_to_equity** (*dict*) -- Leverage trend.  "improving"
          means leverage is *decreasing*.
        - **current_ratio** (*dict*) -- Liquidity trend.
        - **asset_turnover** (*dict*) -- Efficiency trend.
        - **summary** (*str*) -- Overall assessment: ``"fundamentals
          improving"``, ``"fundamentals deteriorating"``, or
          ``"fundamentals mixed"``.

    Example:
        >>> from wraquant.fundamental.ratios import ratio_trends
        >>> trends = ratio_trends("MSFT", periods=5)
        >>> print(f"ROE trend: {trends['roe']['direction']}")
        >>> print(f"Summary: {trends['summary']}")

    See Also:
        ratio_comparison: Cross-sectional peer comparison.
        comprehensive_ratios: Point-in-time ratio snapshot.
    """
    client = _get_fmp_client(fmp_client)

    # Fetch multi-period data
    ratios_data = _safe_get_list(client.ratios(symbol, period="annual", limit=periods))
    income_data = _safe_get_list(
        client.income_statement(symbol, period="annual", limit=periods)
    )
    balance_data = _safe_get_list(
        client.balance_sheet(symbol, period="annual", limit=periods)
    )

    n_periods = min(len(ratios_data), len(income_data), len(balance_data))
    if n_periods == 0:
        return {
            "symbol": symbol,
            "periods_analysed": 0,
            "summary": "insufficient data",
        }

    def _extract_series(data_list: list[dict], key: str) -> list[float]:
        return [_safe_get(d, key) for d in data_list[:n_periods]]

    def _compute_trend(values: list[float], invert: bool = False) -> dict[str, Any]:
        """Compute trend direction and magnitude from a time series.

        *values* are most-recent-first.  When *invert* is True, a
        *decrease* in the ratio is classified as ``"improving"`` (e.g.,
        debt-to-equity: lower is better).
        """
        if len(values) < 2:
            return {"values": values, "direction": "unknown", "magnitude": 0.0}

        latest = values[0]
        oldest = values[-1]
        if abs(oldest) < 1e-12:
            magnitude = 0.0
        else:
            magnitude = (latest - oldest) / abs(oldest)

        threshold = 0.05
        if abs(magnitude) < threshold:
            direction = "stable"
        elif magnitude > 0:
            direction = "deteriorating" if invert else "improving"
        else:
            direction = "improving" if invert else "deteriorating"

        return {"values": values, "direction": direction, "magnitude": float(magnitude)}

    # Extract ratio time series
    roe_vals = [
        _safe_div(
            _safe_get(income_data[i], "netIncome"),
            _safe_get(balance_data[i], "totalStockholdersEquity"),
        )
        for i in range(n_periods)
    ]
    roa_vals = [
        _safe_div(
            _safe_get(income_data[i], "netIncome"),
            _safe_get(balance_data[i], "totalAssets"),
        )
        for i in range(n_periods)
    ]
    gross_margins = [
        _safe_div(
            _safe_get(income_data[i], "grossProfit"),
            _safe_get(income_data[i], "revenue"),
        )
        for i in range(n_periods)
    ]
    op_margins = [
        _safe_div(
            _safe_get(income_data[i], "operatingIncome"),
            _safe_get(income_data[i], "revenue"),
        )
        for i in range(n_periods)
    ]
    net_margins = [
        _safe_div(
            _safe_get(income_data[i], "netIncome"),
            _safe_get(income_data[i], "revenue"),
        )
        for i in range(n_periods)
    ]
    de_vals = [
        _safe_div(
            _safe_get(balance_data[i], "totalDebt"),
            _safe_get(balance_data[i], "totalStockholdersEquity"),
        )
        for i in range(n_periods)
    ]
    cr_vals = [
        _safe_div(
            _safe_get(balance_data[i], "totalCurrentAssets"),
            _safe_get(balance_data[i], "totalCurrentLiabilities"),
        )
        for i in range(n_periods)
    ]
    at_vals = [
        _safe_div(
            _safe_get(income_data[i], "revenue"),
            _safe_get(balance_data[i], "totalAssets"),
        )
        for i in range(n_periods)
    ]

    trends = {
        "roe": _compute_trend(roe_vals),
        "roa": _compute_trend(roa_vals),
        "gross_margin": _compute_trend(gross_margins),
        "operating_margin": _compute_trend(op_margins),
        "net_margin": _compute_trend(net_margins),
        "debt_to_equity": _compute_trend(de_vals, invert=True),
        "current_ratio": _compute_trend(cr_vals),
        "asset_turnover": _compute_trend(at_vals),
    }

    # Summary: count improving vs deteriorating
    improving = sum(1 for t in trends.values() if t.get("direction") == "improving")
    deteriorating = sum(
        1 for t in trends.values() if t.get("direction") == "deteriorating"
    )

    if improving >= 5:
        summary = "fundamentals improving"
    elif deteriorating >= 5:
        summary = "fundamentals deteriorating"
    else:
        summary = "fundamentals mixed"

    return {
        "symbol": symbol,
        "periods_analysed": n_periods,
        **trends,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Sector Comparison
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def sector_comparison(
    symbol: str,
    *,
    fmp_client: Any | None = None,
) -> dict[str, Any]:
    """Compare a stock's ratios to its sector and industry averages.

    While :func:`ratio_comparison` benchmarks against specific peers, this
    function compares against the broader sector using FMP sector PE data
    and the company's own profile metadata.  This is useful for answering:
    "Is this stock cheap *for its sector*?" and "Does this company's
    profitability justify a sector-premium valuation?"

    When to use:
        - Sector rotation: compare company metrics to sector norms.
        - Relative valuation: determine if a premium/discount to sector
          is warranted by superior/inferior fundamentals.
        - Factor construction: normalise ratios by sector before ranking.

    Parameters:
        symbol: Ticker symbol (e.g., ``"AAPL"``).
        fmp_client: Optional pre-configured ``FMPClient`` instance.

    Returns:
        Dictionary containing:
        - **symbol** (*str*) -- The target ticker.
        - **sector** (*str*) -- The company's GICS sector.
        - **industry** (*str*) -- The company's industry classification.
        - **company_ratios** (*dict*) -- The company's key ratios.
        - **sector_pe** (*float*) -- Average P/E for the sector.
        - **company_pe** (*float*) -- The company's P/E.
        - **pe_premium_to_sector** (*float*) -- (company_PE - sector_PE)
          / sector_PE.  Positive = premium; negative = discount.
        - **sector_performance** (*float*) -- Current-day sector
          performance (% change).
        - **assessment** (*str*) -- ``"premium to sector"``,
          ``"discount to sector"``, or ``"in-line with sector"``.

    Example:
        >>> from wraquant.fundamental.ratios import sector_comparison
        >>> sc = sector_comparison("AAPL")
        >>> print(f"Sector: {sc['sector']}")
        >>> print(f"PE premium: {sc['pe_premium_to_sector']:+.1%}")
        >>> print(f"Assessment: {sc['assessment']}")

    See Also:
        ratio_comparison: Peer-level comparison with percentile ranking.
        relative_valuation: Full multi-metric peer comparison.
    """
    client = _get_fmp_client(fmp_client)

    # Get company profile for sector/industry classification
    profile = client.company_profile(symbol)
    profile_data = profile[0] if isinstance(profile, list) and profile else profile
    sector = profile_data.get("sector", "") if isinstance(profile_data, dict) else ""
    industry = (
        profile_data.get("industry", "") if isinstance(profile_data, dict) else ""
    )

    # Get company's own ratios
    target_val = valuation_ratios(symbol, fmp_client=client)
    target_prof = profitability_ratios(symbol, fmp_client=client)
    target_lev = leverage_ratios(symbol, fmp_client=client)

    company_pe = target_val.get("pe_ratio", 0.0)

    company_ratios = {
        "pe_ratio": company_pe,
        "pb_ratio": target_val.get("pb_ratio", 0.0),
        "ps_ratio": target_val.get("ps_ratio", 0.0),
        "ev_to_ebitda": target_val.get("ev_to_ebitda", 0.0),
        "roe": target_prof.get("roe", 0.0),
        "operating_margin": target_prof.get("operating_margin", 0.0),
        "debt_to_equity": target_lev.get("debt_to_equity", 0.0),
    }

    # Get sector performance data
    sector_pe = 0.0
    sector_perf = 0.0
    try:
        sector_data = client.sector_performance()
        if isinstance(sector_data, list):
            for item in sector_data:
                if (
                    isinstance(item, dict)
                    and item.get("sector", "").lower() == sector.lower()
                ):
                    sector_perf = float(item.get("changesPercentage", 0) or 0)
                    break
    except Exception:  # noqa: BLE001
        logger.warning("Could not fetch sector performance data")

    # Estimate sector PE from peers as a fallback
    try:
        peers = client.stock_peers(symbol)
        peer_pes: list[float] = []
        for peer in peers[:10]:  # limit API calls
            try:
                pr = client.ratios_ttm(peer)
                pe_val = float(pr.get("peRatioTTM", 0) or 0)
                if 0 < pe_val < 200:  # filter outliers
                    peer_pes.append(pe_val)
            except Exception:  # noqa: BLE001
                continue
        if peer_pes:
            sorted_pes = sorted(peer_pes)
            n = len(sorted_pes)
            sector_pe = (
                sorted_pes[n // 2]
                if n % 2 == 1
                else (sorted_pes[n // 2 - 1] + sorted_pes[n // 2]) / 2.0
            )
    except Exception:  # noqa: BLE001
        logger.warning("Could not compute sector PE for %s", symbol)

    # PE premium/discount to sector
    pe_premium = _safe_div(company_pe - sector_pe, sector_pe) if sector_pe > 0 else 0.0

    if pe_premium > 0.15:
        assessment = "premium to sector"
    elif pe_premium < -0.15:
        assessment = "discount to sector"
    else:
        assessment = "in-line with sector"

    return {
        "symbol": symbol,
        "sector": sector,
        "industry": industry,
        "company_ratios": company_ratios,
        "sector_pe": float(sector_pe),
        "company_pe": float(company_pe),
        "pe_premium_to_sector": float(pe_premium),
        "sector_performance": float(sector_perf),
        "assessment": assessment,
    }
