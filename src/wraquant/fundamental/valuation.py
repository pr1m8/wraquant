"""Valuation models and quality screening.

Provides Piotroski F-Score, discounted cash flow (DCF) valuation, and
composite quality screening for fundamental-driven quant strategies.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


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
        Piotroski, J. D. (2000). "Value Investing: The Use of
        Historical Financial Statement Information to Separate Winners
        from Losers." *Journal of Accounting Research*, 38, 1--41.

    See Also:
        dcf_valuation: Intrinsic value estimation.
        quality_screen: Composite quality ranking.
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


def dcf_valuation(
    cash_flows: Sequence[float],
    discount_rate: float,
    terminal_growth: float = 0.02,
) -> dict[str, float]:
    """Discounted cash flow (DCF) valuation.

    Estimates intrinsic value by discounting projected future cash flows
    and a terminal value back to the present.  The terminal value uses
    the Gordon growth model: ``TV = CF_n * (1 + g) / (r - g)``.

    When to use:
        - Intrinsic value estimation for individual stocks.
        - Sensitivity analysis: vary discount_rate and terminal_growth
          to understand valuation range.
        - Compare intrinsic value to market price for alpha signals.

    Mathematical formulation:
        PV = sum_{t=1}^{n} CF_t / (1 + r)^t + TV / (1 + r)^n

        TV = CF_n * (1 + g) / (r - g)

        where r = discount_rate and g = terminal_growth.

    Parameters:
        cash_flows: Projected free cash flows for each future period.
            The last element is used as the basis for terminal value.
        discount_rate: Discount rate (WACC or required return).
            Typical range: 0.08 -- 0.12 for equities.
        terminal_growth: Perpetual growth rate for the terminal value.
            Must be less than discount_rate.  Typical range:
            0.02 -- 0.03 (GDP growth rate).

    Returns:
        Dictionary containing:
        - **present_value** (*float*) -- Total present value (intrinsic
          value estimate).
        - **pv_cash_flows** (*float*) -- Present value of explicit
          cash flow projections.
        - **pv_terminal** (*float*) -- Present value of terminal value.
        - **terminal_value** (*float*) -- Undiscounted terminal value.

    Raises:
        ValueError: If discount_rate <= terminal_growth (terminal value
            would be infinite or negative).

    Example:
        >>> dcf_valuation([100, 110, 121], discount_rate=0.10,
        ...               terminal_growth=0.02)['present_value']
        1280.7917...

    See Also:
        piotroski_f_score: Financial health assessment.
        quality_screen: Multi-metric quality ranking.

    References:
        - Damodaran, A. (2012). "Investment Valuation", 3rd edition.
    """
    if discount_rate <= terminal_growth:
        msg = (
            f"discount_rate ({discount_rate}) must be greater than "
            f"terminal_growth ({terminal_growth})."
        )
        raise ValueError(msg)

    cfs = list(cash_flows)
    if not cfs:
        return {
            "present_value": 0.0,
            "pv_cash_flows": 0.0,
            "pv_terminal": 0.0,
            "terminal_value": 0.0,
        }

    n = len(cfs)

    # PV of explicit cash flows
    pv_cfs = 0.0
    for t, cf in enumerate(cfs, start=1):
        pv_cfs += cf / (1 + discount_rate) ** t

    # Terminal value (Gordon growth model)
    terminal_cf = cfs[-1] * (1 + terminal_growth)
    terminal_value = terminal_cf / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / (1 + discount_rate) ** n

    present_value = pv_cfs + pv_terminal

    return {
        "present_value": float(present_value),
        "pv_cash_flows": float(pv_cfs),
        "pv_terminal": float(pv_terminal),
        "terminal_value": float(terminal_value),
    }


def quality_screen(
    stocks_df: pd.DataFrame,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
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
        dcf_valuation: Intrinsic value estimation (single stock).
    """
    default_metrics = ["roe", "operating_margin", "current_ratio"]

    if metrics is None:
        metrics = [m for m in default_metrics if m in stocks_df.columns]

    if not metrics:
        result = stocks_df.copy()
        result["quality_score"] = 0.5
        result["quality_rank"] = 1
        return result

    # Available metrics only
    available = [m for m in metrics if m in stocks_df.columns]

    if not available:
        result = stocks_df.copy()
        result["quality_score"] = 0.5
        result["quality_rank"] = 1
        return result

    # Compute percentile ranks for each metric
    ranks = pd.DataFrame(index=stocks_df.index)
    for m in available:
        col = stocks_df[m].astype(float)
        ranks[m] = col.rank(pct=True, na_option="keep")
        # Fill NaN with median rank (0.5)
        ranks[m] = ranks[m].fillna(0.5)

    # Composite: average of percentile ranks
    composite = ranks.mean(axis=1)

    result = stocks_df.copy()
    result["quality_score"] = composite
    result["quality_rank"] = composite.rank(ascending=False, method="min").astype(int)
    result = result.sort_values("quality_score", ascending=False)

    return result
