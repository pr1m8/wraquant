"""Fundamental analysis for quantitative finance.

Provides FMP-backed tools for financial ratios, valuation models,
financial statement analysis, earnings quality, and stock screening.

Modules:
    ratios: Profitability, liquidity, leverage, efficiency, valuation ratios
    valuation: DCF, relative valuation, Graham number, DDM, residual income
    financials: Income/balance sheet/cash flow analysis, health scores
    screening: Stock screeners (value, growth, quality, Piotroski, magic formula)

Example:
    >>> from wraquant.fundamental import comprehensive_ratios, dcf_valuation
    >>> ratios = comprehensive_ratios("AAPL")
    >>> dcf = dcf_valuation("AAPL", discount_rate=0.10)
    >>> print(f"Intrinsic value: ${dcf['intrinsic_value_per_share']:.2f}")

References:
    - Graham & Dodd (1934), "Security Analysis"
    - Piotroski (2000), "Value Investing"
    - Greenblatt (2006), "The Little Book That Beats the Market"
    - Damodaran (2012), "Investment Valuation"
"""

from __future__ import annotations

from wraquant.fundamental.ratios import (
    comprehensive_ratios,
    dupont_decomposition,
    efficiency_ratios,
    growth_ratios,
    leverage_ratios,
    liquidity_ratios,
    profitability_ratios,
    valuation_ratios,
)
from wraquant.fundamental.valuation import (
    dcf_valuation,
    dividend_discount_model,
    graham_number,
    margin_of_safety,
    peter_lynch_value,
    piotroski_f_score,
    quality_screen,
    relative_valuation,
    residual_income_model,
)

__all__ = [
    # Ratios
    "profitability_ratios",
    "liquidity_ratios",
    "leverage_ratios",
    "efficiency_ratios",
    "valuation_ratios",
    "growth_ratios",
    "dupont_decomposition",
    "comprehensive_ratios",
    # Valuation
    "dcf_valuation",
    "relative_valuation",
    "graham_number",
    "peter_lynch_value",
    "dividend_discount_model",
    "residual_income_model",
    "margin_of_safety",
    "piotroski_f_score",
    "quality_screen",
]
