"""Fundamental analysis for quantitative finance.

Provides tools for computing financial ratios, valuation models,
financial statement analysis, and stock screening -- the building
blocks of fundamental-driven quant strategies.

This module covers four areas:

1. **Financial ratios** (``ratios`` submodule) -- Profitability, liquidity,
   leverage, efficiency, valuation, and growth ratios.  Includes DuPont
   decomposition (3-way and 5-way) and a convenience function that
   aggregates all ratios.

2. **Valuation models** (``valuation`` submodule) -- DCF valuation,
   relative valuation, Graham Number, Peter Lynch fair value, Dividend
   Discount Model, Residual Income Model, and margin of safety.
   Also retains the Piotroski F-Score and DataFrame quality screen.

3. **Financial statement analysis** (``financials`` submodule) -- Income
   statement, balance sheet, and cash flow trend analysis.  Composite
   financial health scoring (0--100), earnings quality assessment, and
   common-size statements.

4. **Stock screening** (``screening`` submodule) -- Pre-built screens for
   value, growth, and quality investing.  Classic strategies: Piotroski
   F-Score screen and Greenblatt's Magic Formula.  Flexible custom
   screening with arbitrary criteria.

All FMP-based functions accept an optional ``fmp_client`` parameter to
reuse a single provider instance across calls.

Example:
    >>> from wraquant.fundamental import profitability_ratios, dcf_valuation
    >>> prof = profitability_ratios("AAPL")
    >>> print(f"ROE: {prof['roe']:.2%}")
    >>> dcf = dcf_valuation("AAPL")
    >>> print(f"Fair value: ${dcf['intrinsic_value_per_share']:.2f}")

References:
    - Graham & Dodd (1934), "Security Analysis"
    - Piotroski (2000), "Value Investing: The Use of Historical
      Financial Statement Information to Separate Winners from Losers"
    - Greenblatt (2006), "The Little Book That Beats the Market"
    - Damodaran (2012), "Investment Valuation", 3rd edition
"""

# --- Financial statement analysis ---
from wraquant.fundamental.financials import (
    balance_sheet_analysis,
    capex_analysis,
    cash_flow_analysis,
    common_size_analysis,
    earnings_quality,
    financial_health_score,
    income_analysis,
    revenue_decomposition,
    shareholder_returns,
    working_capital_analysis,
)

# --- Ratios ---
from wraquant.fundamental.ratios import (
    comprehensive_ratios,
    dupont_decomposition,
    efficiency_ratios,
    growth_ratios,
    leverage_ratios,
    liquidity_ratios,
    profitability_ratios,
    ratio_comparison,
    ratio_trends,
    sector_comparison,
    valuation_ratios,
)

# --- Screening ---
from wraquant.fundamental.screening import (
    custom_screen,
    dividend_aristocrat_screen,
    growth_screen,
    insider_buying_screen,
    magic_formula_screen,
    momentum_value_screen,
    piotroski_screen,
)
from wraquant.fundamental.screening import quality_screen as quality_factor_screen
from wraquant.fundamental.screening import (
    turnaround_screen,
    value_screen,
)

# --- Valuation ---
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
    "ratio_comparison",
    "ratio_trends",
    "sector_comparison",
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
    # Financial statement analysis
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
    # Screening
    "value_screen",
    "growth_screen",
    "quality_factor_screen",
    "piotroski_screen",
    "magic_formula_screen",
    "custom_screen",
    "dividend_aristocrat_screen",
    "turnaround_screen",
    "insider_buying_screen",
    "momentum_value_screen",
]
