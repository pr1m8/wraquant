"""Fundamental analysis for quantitative finance.

Provides tools for computing financial ratios, valuation models, and
quality screens -- the building blocks of fundamental-driven quant
strategies.  While wraquant excels at technical and statistical analysis,
many successful strategies blend price-based signals with fundamental
data.

This module covers three areas:

1. **Financial ratios** (``ratios`` submodule) -- Standard valuation and
   profitability ratios: P/E, P/B, ROE, debt-to-equity, current ratio,
   and operating margin.  These are the foundation of value investing
   and factor-based strategies.

2. **Valuation models** (``valuation`` submodule) -- Discounted cash
   flow (DCF) valuation, Piotroski F-Score for financial health
   assessment, and composite quality screening.

3. **Quality screening** (``valuation`` submodule) -- Rank stocks by a
   composite quality score combining profitability, leverage, and
   efficiency metrics.

Example:
    >>> from wraquant.fundamental import pe_ratio, piotroski_f_score
    >>> pe = pe_ratio(price=150.0, earnings=7.5)
    >>> print(f"P/E ratio: {pe:.1f}")
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
    >>> score = piotroski_f_score(financials)
    >>> print(f"Piotroski F-Score: {score}")

References:
    - Graham & Dodd (1934), "Security Analysis"
    - Piotroski (2000), "Value Investing: The Use of Historical
      Financial Statement Information to Separate Winners from Losers"
"""

from wraquant.fundamental.ratios import (
    current_ratio,
    debt_to_equity,
    operating_margin,
    pb_ratio,
    pe_ratio,
    roe,
)
from wraquant.fundamental.valuation import (
    dcf_valuation,
    piotroski_f_score,
    quality_screen,
)

__all__ = [
    "pe_ratio",
    "pb_ratio",
    "roe",
    "debt_to_equity",
    "current_ratio",
    "operating_margin",
    "piotroski_f_score",
    "dcf_valuation",
    "quality_screen",
]
