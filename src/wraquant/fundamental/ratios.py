"""Financial ratios for fundamental analysis.

Provides standard valuation and profitability ratios used in
fundamental-driven quant strategies and factor investing.
"""

from __future__ import annotations


def pe_ratio(price: float, earnings: float) -> float:
    """Price-to-earnings ratio.

    The P/E ratio is the most widely used valuation metric.  It
    measures how much investors are willing to pay per dollar of
    earnings.  High P/E suggests growth expectations or overvaluation;
    low P/E suggests value or distress.

    Mathematical formulation:
        P/E = price / earnings

    When to use:
        - Screen for value stocks (low P/E) or growth stocks (high P/E).
        - Compare valuations across companies in the same sector.
        - As a factor in multi-factor models (value factor).

    Parameters:
        price: Current share price.
        earnings: Earnings per share (EPS).  Use trailing twelve months
            (TTM) for backward-looking or consensus estimates for
            forward P/E.

    Returns:
        P/E ratio as a float.  Returns 0.0 if earnings is zero or
        near-zero.

    Example:
        >>> pe_ratio(price=150.0, earnings=7.5)
        20.0
        >>> pe_ratio(price=50.0, earnings=2.0)
        25.0

    See Also:
        pb_ratio: Price-to-book ratio.
        roe: Return on equity.
    """
    if abs(earnings) < 1e-12:
        return 0.0
    return float(price / earnings)


def pb_ratio(price: float, book_value: float) -> float:
    """Price-to-book ratio.

    Measures the market price relative to the accounting book value of
    equity.  P/B < 1 suggests the stock trades below its net asset
    value (potential value opportunity or sign of distress).

    Mathematical formulation:
        P/B = price / book_value_per_share

    When to use:
        - Value screening: low P/B stocks form the HML (high-minus-low)
          factor in Fama-French models.
        - Financial sector analysis where book value is a meaningful
          anchor.
        - Tangible-asset-heavy industries (real estate, banks).

    Parameters:
        price: Current share price.
        book_value: Book value per share.

    Returns:
        P/B ratio as a float.  Returns 0.0 if book_value is zero or
        near-zero.

    Example:
        >>> pb_ratio(price=100.0, book_value=50.0)
        2.0

    See Also:
        pe_ratio: Price-to-earnings ratio.
    """
    if abs(book_value) < 1e-12:
        return 0.0
    return float(price / book_value)


def roe(net_income: float, equity: float) -> float:
    """Return on equity.

    Measures how efficiently a company generates profit from
    shareholder equity.  Higher ROE indicates more efficient use of
    equity capital.

    Mathematical formulation:
        ROE = net_income / shareholders_equity

    When to use:
        - Profitability screening: high ROE is a key component of the
          RMW (robust-minus-weak) factor in the Fama-French 5-factor
          model.
        - DuPont decomposition: ROE = margin * turnover * leverage.
        - Quality factor construction.

    Parameters:
        net_income: Net income (annual or TTM).
        equity: Total shareholders' equity.

    Returns:
        ROE as a float (e.g., 0.15 = 15%).  Returns 0.0 if equity is
        zero or near-zero.

    Example:
        >>> roe(net_income=1_000_000, equity=5_000_000)
        0.2

    See Also:
        operating_margin: Profitability before interest and taxes.
        debt_to_equity: Leverage measure.
    """
    if abs(equity) < 1e-12:
        return 0.0
    return float(net_income / equity)


def debt_to_equity(total_debt: float, equity: float) -> float:
    """Debt-to-equity ratio.

    Measures financial leverage -- the extent to which a company funds
    its operations with debt versus equity.  Higher D/E implies greater
    financial risk but also potential for higher returns on equity
    through leverage.

    Mathematical formulation:
        D/E = total_debt / shareholders_equity

    When to use:
        - Risk screening: high D/E increases bankruptcy risk.
        - Leverage factor in multi-factor models.
        - Sector-relative comparisons (capital-intensive industries
          naturally carry more debt).

    Parameters:
        total_debt: Total debt (short-term + long-term).
        equity: Total shareholders' equity.

    Returns:
        D/E ratio as a float.  Returns 0.0 if equity is zero or
        near-zero.

    Example:
        >>> debt_to_equity(total_debt=2_000_000, equity=5_000_000)
        0.4

    See Also:
        current_ratio: Short-term liquidity measure.
        roe: Return on equity.
    """
    if abs(equity) < 1e-12:
        return 0.0
    return float(total_debt / equity)


def current_ratio(
    current_assets: float,
    current_liabilities: float,
) -> float:
    """Current ratio (short-term liquidity).

    Measures a company's ability to pay short-term obligations with
    current assets.  A ratio above 1.0 indicates the company can cover
    its near-term liabilities; below 1.0 signals potential liquidity
    stress.

    Mathematical formulation:
        current_ratio = current_assets / current_liabilities

    When to use:
        - Liquidity screening: current_ratio < 1.0 is a red flag.
        - Credit analysis and bankruptcy prediction models.
        - Piotroski F-Score component.

    Parameters:
        current_assets: Total current assets.
        current_liabilities: Total current liabilities.

    Returns:
        Current ratio as a float.  Returns 0.0 if current_liabilities
        is zero or near-zero.

    Example:
        >>> current_ratio(current_assets=500_000, current_liabilities=300_000)
        1.6666666666666667

    See Also:
        debt_to_equity: Long-term leverage measure.
    """
    if abs(current_liabilities) < 1e-12:
        return 0.0
    return float(current_assets / current_liabilities)


def operating_margin(
    operating_income: float,
    revenue: float,
) -> float:
    """Operating margin.

    Measures operating profitability as a fraction of revenue.
    Operating margin strips out interest and taxes, showing how
    efficiently the core business generates profit.

    Mathematical formulation:
        operating_margin = operating_income / revenue

    When to use:
        - Profitability comparison across companies.
        - Trend analysis: declining margins may signal competitive
          pressure.
        - Quality screening: stable or expanding margins indicate
          pricing power and operational efficiency.

    Parameters:
        operating_income: Operating income (EBIT).
        revenue: Total revenue.

    Returns:
        Operating margin as a float (e.g., 0.20 = 20%).  Returns 0.0
        if revenue is zero or near-zero.

    Example:
        >>> operating_margin(operating_income=200_000, revenue=1_000_000)
        0.2

    See Also:
        roe: Return on equity.
        pe_ratio: Valuation metric.
    """
    if abs(revenue) < 1e-12:
        return 0.0
    return float(operating_income / revenue)
