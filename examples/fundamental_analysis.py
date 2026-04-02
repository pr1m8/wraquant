"""Complete fundamental analysis of a stock using wraquant.

This example demonstrates how to use wraquant's fundamental module
to analyze a company's financial health, compute valuation, and
make an investment decision.

The workflow mirrors what a buy-side analyst does before initiating
a position:

    1. Company profile -- who are they, what sector, what size?
    2. Income analysis -- is the business growing, are margins expanding?
    3. Balance sheet -- is the capital structure safe?
    4. Financial ratios -- profitability, efficiency, valuation
    5. DuPont decomposition -- what drives ROE?
    6. DCF valuation -- what is the intrinsic value?
    7. Graham Number -- conservative floor estimate
    8. Relative valuation -- cheap or expensive vs peers?
    9. Financial health score -- composite 0-100 grade
   10. Earnings quality -- can we trust the numbers?
   11. Final verdict -- buy, hold, or avoid?

Usage:
    FMP_API_KEY=your_key python examples/fundamental_analysis.py AAPL
    python examples/fundamental_analysis.py MSFT  # uses synthetic data as fallback

Requirements:
    pip install wraquant[market-data]
"""

from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Synthetic data fallback -- when no FMP API key is available, we generate
# realistic-looking data so the example still runs end-to-end.
# ---------------------------------------------------------------------------

SYNTHETIC_PROFILE = {
    "companyName": "Apple Inc.",
    "symbol": "AAPL",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "marketCap": 2_850_000_000_000,
    "price": 185.50,
    "beta": 1.24,
    "description": "Apple Inc. designs, manufactures, and markets smartphones, "
    "personal computers, tablets, wearables, and accessories.",
    "ceo": "Tim Cook",
    "exchange": "NASDAQ",
    "country": "US",
    "fullTimeEmployees": 164_000,
    "ipoDate": "1980-12-12",
}

SYNTHETIC_INCOME = {
    "revenue": [383_285_000_000, 394_328_000_000, 365_817_000_000, 274_515_000_000],
    "revenue_growth": [0.078, -0.028, 0.333, 0.054],
    "gross_margin": [0.438, 0.433, 0.417, 0.382],
    "operating_margin": [0.302, 0.308, 0.286, 0.246],
    "net_margin": [0.253, 0.259, 0.241, 0.209],
    "ebitda_margin": [0.328, 0.338, 0.311, 0.276],
    "eps": [6.13, 6.11, 5.61, 3.69],
    "dates": ["2023-09-30", "2022-10-01", "2021-09-25", "2020-09-26"],
    "margin_trend": "stable",
    "revenue_cagr_3y": 0.118,
    "revenue_cagr_5y": 0.087,
}

SYNTHETIC_BALANCE = {
    "total_assets": [352_583_000_000, 352_755_000_000, 351_002_000_000],
    "total_liabilities": [290_437_000_000, 302_083_000_000, 287_912_000_000],
    "total_equity": [62_146_000_000, 50_672_000_000, 63_090_000_000],
    "current_ratio": [0.988, 0.879, 1.075],
    "debt_to_equity": [1.80, 2.37, 1.73],
    "book_value_per_share": [3.97, 3.19, 3.84],
    "working_capital": [-1_742_000_000, -18_084_000_000, 9_355_000_000],
    "dates": ["2023-09-30", "2022-10-01", "2021-09-25"],
    "leverage_trend": "improving",
}

SYNTHETIC_PROFITABILITY = {
    "roe": 1.565,
    "roa": 0.275,
    "roic": 0.498,
    "gross_margin": 0.438,
    "operating_margin": 0.302,
    "net_margin": 0.253,
    "period": "annual",
}

SYNTHETIC_LIQUIDITY = {
    "current_ratio": 0.988,
    "quick_ratio": 0.943,
    "cash_ratio": 0.221,
    "working_capital": -1_742_000_000,
    "period": "annual",
}

SYNTHETIC_LEVERAGE = {
    "debt_to_equity": 1.80,
    "debt_to_assets": 0.316,
    "interest_coverage": 29.1,
    "equity_multiplier": 5.67,
    "long_term_debt_to_equity": 1.62,
    "period": "annual",
}

SYNTHETIC_EFFICIENCY = {
    "asset_turnover": 1.087,
    "receivables_turnover": 18.2,
    "inventory_turnover": 34.1,
    "days_sales_outstanding": 20.1,
    "days_inventory": 10.7,
    "days_payable": 63.8,
    "cash_conversion_cycle": -33.0,
    "period": "annual",
}

SYNTHETIC_VALUATION = {
    "pe_ratio": 30.3,
    "pb_ratio": 46.7,
    "ps_ratio": 7.43,
    "ev_ebitda": 23.6,
    "peg_ratio": 2.82,
    "earnings_yield": 0.033,
    "fcf_yield": 0.034,
    "dividend_yield": 0.005,
    "period": "annual",
}

SYNTHETIC_GROWTH = {
    "revenue_growth": 0.078,
    "net_income_growth": 0.069,
    "eps_growth": 0.093,
    "revenue_growth_3y_cagr": 0.118,
    "net_income_growth_3y_cagr": 0.143,
    "period": "annual",
}

SYNTHETIC_DUPONT = {
    "roe": 1.565,
    "net_margin": 0.253,
    "asset_turnover": 1.087,
    "equity_multiplier": 5.67,
    "roe_check": 1.561,
    "tax_burden": 0.846,
    "interest_burden": 0.985,
    "operating_margin": 0.302,
    "roe_5way_check": 1.432,
    "decomposition_3way": {
        "net_margin": 0.253,
        "asset_turnover": 1.087,
        "equity_multiplier": 5.67,
    },
    "decomposition_5way": {
        "tax_burden": 0.846,
        "interest_burden": 0.985,
        "operating_margin": 0.302,
        "asset_turnover": 1.087,
        "equity_multiplier": 5.67,
    },
}

SYNTHETIC_DCF = {
    "intrinsic_value": 2_650_000_000_000,
    "intrinsic_value_per_share": 169.30,
    "current_price": 185.50,
    "margin_of_safety": -0.096,
    "upside_potential": -0.087,
    "pv_cash_flows": 530_000_000_000,
    "pv_terminal": 2_120_000_000_000,
    "terminal_value": 3_450_000_000_000,
    "terminal_pct": 0.80,
    "projected_fcf": [115e9, 124e9, 134e9, 145e9, 157e9],
    "growth_rate": 0.08,
    "discount_rate": 0.10,
    "terminal_growth": 0.025,
}

SYNTHETIC_GRAHAM = {
    "graham_number": 54.12,
    "current_price": 185.50,
    "margin_of_safety": -0.708,
    "is_undervalued": False,
    "eps": 6.13,
    "book_value_per_share": 3.97,
}

SYNTHETIC_RELATIVE = {
    "pe_vs_sector": {"stock": 30.3, "sector_median": 24.1, "premium": 0.257},
    "ev_ebitda_vs_sector": {"stock": 23.6, "sector_median": 18.2, "premium": 0.297},
    "pb_vs_sector": {"stock": 46.7, "sector_median": 5.8, "premium": 7.052},
    "composite_premium": 0.297,
    "assessment": "Trading at a 30% premium to sector -- justified if growth persists.",
}

SYNTHETIC_HEALTH = {
    "score": 72,
    "grade": "B",
    "components": {
        "profitability": 95,
        "liquidity": 45,
        "solvency": 55,
        "efficiency": 92,
    },
    "flags": ["Current ratio below 1.0", "High debt-to-equity (1.80)"],
}

SYNTHETIC_EARNINGS_QUALITY = {
    "accruals_ratio": -0.032,
    "cash_conversion": 1.12,
    "quality_score": 88,
    "assessment": "high",
    "details": {
        "accruals_ratio": -0.032,
        "cash_conversion_ratio": 1.12,
        "revenue_quality": 0.95,
        "earnings_persistence": 0.91,
    },
}


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def header(title: str) -> None:
    """Print a prominent section header."""
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def subheader(title: str) -> None:
    """Print a subsection header."""
    print(f"\n--- {title} ---")


def pct(value: float) -> str:
    """Format a float as a percentage string."""
    return f"{value:.1%}"


def dollar(value: float) -> str:
    """Format a number as a dollar amount."""
    if abs(value) >= 1e12:
        return f"${value / 1e12:.2f}T"
    if abs(value) >= 1e9:
        return f"${value / 1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:.2f}M"
    return f"${value:,.2f}"


def grade_color(grade: str) -> str:
    """Return a text annotation for a letter grade."""
    mapping = {
        "A": "Excellent",
        "B": "Good",
        "C": "Fair",
        "D": "Weak",
        "F": "Failing",
    }
    return mapping.get(grade, "Unknown")


# ---------------------------------------------------------------------------
# Main analysis workflow
# ---------------------------------------------------------------------------

def run_analysis(ticker: str, use_live: bool = True) -> None:
    """Run the complete fundamental analysis pipeline."""

    has_api_key = bool(os.environ.get("FMP_API_KEY"))

    if use_live and has_api_key:
        print(f"[Using LIVE FMP data for {ticker}]")
        from wraquant.fundamental import (
            income_analysis,
            balance_sheet_analysis,
            profitability_ratios,
            liquidity_ratios,
            leverage_ratios,
            efficiency_ratios,
            valuation_ratios,
            growth_ratios,
            dupont_decomposition,
            dcf_valuation,
            graham_number,
            relative_valuation,
            financial_health_score,
            earnings_quality,
        )
        # In live mode, call the real functions
        profile = SYNTHETIC_PROFILE  # FMP profile is separate; keep synthetic for display
        income = income_analysis(ticker, period="annual")
        balance = balance_sheet_analysis(ticker, period="annual")
        prof = profitability_ratios(ticker, period="annual")
        liq = liquidity_ratios(ticker, period="annual")
        lev = leverage_ratios(ticker, period="annual")
        eff = efficiency_ratios(ticker, period="annual")
        val = valuation_ratios(ticker, period="annual")
        growth = growth_ratios(ticker, period="annual")
        dupont = dupont_decomposition(ticker, period="annual")
        dcf = dcf_valuation(ticker)
        graham = graham_number(ticker)
        relative = relative_valuation(ticker)
        health = financial_health_score(ticker)
        eq = earnings_quality(ticker)
    else:
        if not has_api_key:
            print(f"[No FMP_API_KEY found -- using synthetic data for {ticker}]")
        else:
            print(f"[Using synthetic data for {ticker}]")
        # Use pre-built synthetic data
        profile = SYNTHETIC_PROFILE
        income = SYNTHETIC_INCOME
        balance = SYNTHETIC_BALANCE
        prof = SYNTHETIC_PROFITABILITY
        liq = SYNTHETIC_LIQUIDITY
        lev = SYNTHETIC_LEVERAGE
        eff = SYNTHETIC_EFFICIENCY
        val = SYNTHETIC_VALUATION
        growth = SYNTHETIC_GROWTH
        dupont = SYNTHETIC_DUPONT
        dcf = SYNTHETIC_DCF
        graham = SYNTHETIC_GRAHAM
        relative = SYNTHETIC_RELATIVE
        health = SYNTHETIC_HEALTH
        eq = SYNTHETIC_EARNINGS_QUALITY

    # ------------------------------------------------------------------
    # 1. Company Profile
    # ------------------------------------------------------------------
    header(f"FUNDAMENTAL ANALYSIS: {ticker}")

    print(f"\n  Company:    {profile['companyName']}")
    print(f"  Sector:     {profile['sector']}")
    print(f"  Industry:   {profile['industry']}")
    print(f"  Market Cap: {dollar(profile['marketCap'])}")
    print(f"  Price:      ${profile['price']:.2f}")
    print(f"  Beta:       {profile['beta']:.2f}")
    print(f"  Exchange:   {profile['exchange']}")

    # ------------------------------------------------------------------
    # 2. Income Statement Analysis
    # ------------------------------------------------------------------
    header("INCOME STATEMENT ANALYSIS")

    print(f"\n  Revenue CAGR (3Y): {pct(income['revenue_cagr_3y'])}")
    if "revenue_cagr_5y" in income:
        print(f"  Revenue CAGR (5Y): {pct(income['revenue_cagr_5y'])}")
    print(f"  Margin Trend:      {income['margin_trend']}")

    subheader("Revenue by Period (most recent first)")
    for i, (date, rev) in enumerate(zip(income["dates"], income["revenue"])):
        growth_str = f"  ({pct(income['revenue_growth'][i])} YoY)" if i < len(income["revenue_growth"]) else ""
        print(f"    {date}:  {dollar(rev)}{growth_str}")

    subheader("Margin Trends")
    print(f"    {'Period':<12} {'Gross':>8} {'Operating':>10} {'Net':>8} {'EBITDA':>8}")
    print(f"    {'-' * 48}")
    for i, date in enumerate(income["dates"]):
        print(
            f"    {date:<12} "
            f"{pct(income['gross_margin'][i]):>8} "
            f"{pct(income['operating_margin'][i]):>10} "
            f"{pct(income['net_margin'][i]):>8} "
            f"{pct(income['ebitda_margin'][i]):>8}"
        )

    subheader("EPS Trend")
    for date, eps in zip(income["dates"], income["eps"]):
        print(f"    {date}:  ${eps:.2f}")

    # ------------------------------------------------------------------
    # 3. Balance Sheet Analysis
    # ------------------------------------------------------------------
    header("BALANCE SHEET ANALYSIS")

    print(f"\n  Leverage Trend: {balance.get('leverage_trend', 'N/A')}")

    subheader("Capital Structure")
    print(f"    {'Period':<12} {'Assets':>14} {'Liabilities':>14} {'Equity':>14}")
    print(f"    {'-' * 56}")
    for i, date in enumerate(balance["dates"]):
        print(
            f"    {date:<12} "
            f"{dollar(balance['total_assets'][i]):>14} "
            f"{dollar(balance['total_liabilities'][i]):>14} "
            f"{dollar(balance['total_equity'][i]):>14}"
        )

    subheader("Key Balance Sheet Ratios")
    print(f"    Current Ratio:       {balance['current_ratio'][0]:.3f}")
    print(f"    Debt-to-Equity:      {balance['debt_to_equity'][0]:.2f}")
    print(f"    Book Value/Share:    ${balance['book_value_per_share'][0]:.2f}")

    # ------------------------------------------------------------------
    # 4. Financial Ratios
    # ------------------------------------------------------------------
    header("FINANCIAL RATIOS")

    subheader("Profitability")
    print(f"    ROE:              {pct(prof['roe']):>10}")
    print(f"    ROA:              {pct(prof['roa']):>10}")
    print(f"    ROIC:             {pct(prof['roic']):>10}")
    print(f"    Gross Margin:     {pct(prof['gross_margin']):>10}")
    print(f"    Operating Margin: {pct(prof['operating_margin']):>10}")
    print(f"    Net Margin:       {pct(prof['net_margin']):>10}")

    subheader("Liquidity")
    print(f"    Current Ratio:    {liq['current_ratio']:>10.3f}")
    print(f"    Quick Ratio:      {liq['quick_ratio']:>10.3f}")
    print(f"    Cash Ratio:       {liq['cash_ratio']:>10.3f}")

    subheader("Leverage")
    print(f"    Debt/Equity:         {lev['debt_to_equity']:>8.2f}")
    print(f"    Debt/Assets:         {lev['debt_to_assets']:>8.3f}")
    print(f"    Interest Coverage:   {lev['interest_coverage']:>8.1f}x")
    print(f"    Equity Multiplier:   {lev['equity_multiplier']:>8.2f}")

    subheader("Efficiency")
    print(f"    Asset Turnover:      {eff['asset_turnover']:>8.3f}")
    print(f"    Days Sales Out:      {eff['days_sales_outstanding']:>8.1f}")
    print(f"    Days Inventory:      {eff['days_inventory']:>8.1f}")
    print(f"    Cash Conv. Cycle:    {eff['cash_conversion_cycle']:>8.1f} days")

    subheader("Valuation")
    print(f"    P/E Ratio:           {val['pe_ratio']:>8.1f}")
    print(f"    P/B Ratio:           {val['pb_ratio']:>8.1f}")
    print(f"    P/S Ratio:           {val['ps_ratio']:>8.2f}")
    print(f"    EV/EBITDA:           {val['ev_ebitda']:>8.1f}")
    print(f"    Earnings Yield:      {pct(val['earnings_yield']):>8}")
    print(f"    FCF Yield:           {pct(val['fcf_yield']):>8}")
    print(f"    Dividend Yield:      {pct(val['dividend_yield']):>8}")

    subheader("Growth")
    print(f"    Revenue Growth:      {pct(growth['revenue_growth']):>8}")
    print(f"    Net Income Growth:   {pct(growth['net_income_growth']):>8}")
    print(f"    EPS Growth:          {pct(growth['eps_growth']):>8}")

    # ------------------------------------------------------------------
    # 5. DuPont Decomposition
    # ------------------------------------------------------------------
    header("DUPONT DECOMPOSITION")

    subheader("3-Way DuPont (ROE = Net Margin x Asset Turnover x Equity Multiplier)")
    d3 = dupont["decomposition_3way"]
    print(f"    Net Margin:        {pct(d3['net_margin'])}")
    print(f"    x Asset Turnover:  {d3['asset_turnover']:.3f}")
    print(f"    x Equity Mult.:    {d3['equity_multiplier']:.2f}")
    print(f"    = ROE:             {pct(dupont['roe'])}")

    subheader("5-Way DuPont (decomposes margin into tax, interest, and operating)")
    d5 = dupont["decomposition_5way"]
    print(f"    Tax Burden:        {d5['tax_burden']:.3f}  (Net Income / EBT)")
    print(f"    Interest Burden:   {d5['interest_burden']:.3f}  (EBT / EBIT)")
    print(f"    Operating Margin:  {pct(d5['operating_margin'])}")
    print(f"    Asset Turnover:    {d5['asset_turnover']:.3f}")
    print(f"    Equity Multiplier: {d5['equity_multiplier']:.2f}")

    # Interpret
    print("\n  Interpretation:")
    if d5["equity_multiplier"] > 3:
        print("    - High leverage amplifies ROE -- watch debt levels.")
    if d5["operating_margin"] > 0.20:
        print("    - Strong operating margin indicates pricing power.")
    if d5["tax_burden"] > 0.80:
        print("    - Favorable tax efficiency (tax burden > 80%).")

    # ------------------------------------------------------------------
    # 6. DCF Valuation
    # ------------------------------------------------------------------
    header("DCF VALUATION")

    print(f"\n  Growth Rate:       {pct(dcf['growth_rate'])}")
    print(f"  Discount Rate:     {pct(dcf['discount_rate'])}")
    print(f"  Terminal Growth:   {pct(dcf['terminal_growth'])}")
    print(f"  PV(Cash Flows):    {dollar(dcf['pv_cash_flows'])}")
    print(f"  PV(Terminal):      {dollar(dcf['pv_terminal'])}")
    print(f"  Terminal % of PV:  {pct(dcf['terminal_pct'])}")

    subheader("Projected Free Cash Flows")
    for i, fcf in enumerate(dcf["projected_fcf"], 1):
        print(f"    Year {i}: {dollar(fcf)}")

    subheader("Valuation Summary")
    print(f"    Intrinsic Value:   ${dcf['intrinsic_value_per_share']:.2f}")
    print(f"    Current Price:     ${dcf['current_price']:.2f}")
    print(f"    Margin of Safety:  {pct(dcf['margin_of_safety'])}")
    print(f"    Upside Potential:  {pct(dcf['upside_potential'])}")

    if dcf["terminal_pct"] > 0.75:
        print("\n    WARNING: Terminal value is >75% of total PV.")
        print("    The valuation is highly sensitive to terminal assumptions.")

    # ------------------------------------------------------------------
    # 7. Graham Number
    # ------------------------------------------------------------------
    header("GRAHAM NUMBER")

    print(f"\n  EPS:               ${graham['eps']:.2f}")
    print(f"  Book Value/Share:  ${graham['book_value_per_share']:.2f}")
    print(f"  Graham Number:     ${graham['graham_number']:.2f}")
    print(f"  Current Price:     ${graham['current_price']:.2f}")
    print(f"  Margin of Safety:  {pct(graham['margin_of_safety'])}")
    print(f"  Undervalued?       {'YES' if graham['is_undervalued'] else 'NO'}")

    if not graham["is_undervalued"]:
        print("\n  Note: Graham Number is conservative. Growth stocks typically")
        print("  trade well above their Graham Number due to intangible value.")

    # ------------------------------------------------------------------
    # 8. Relative Valuation
    # ------------------------------------------------------------------
    header("RELATIVE VALUATION (vs Sector)")

    pe_data = relative["pe_vs_sector"]
    ev_data = relative["ev_ebitda_vs_sector"]
    print(f"\n  P/E:      Stock {pe_data['stock']:.1f}  vs  Sector {pe_data['sector_median']:.1f}  ({'+' if pe_data['premium'] > 0 else ''}{pct(pe_data['premium'])} premium)")
    print(f"  EV/EBITDA: Stock {ev_data['stock']:.1f}  vs  Sector {ev_data['sector_median']:.1f}  ({'+' if ev_data['premium'] > 0 else ''}{pct(ev_data['premium'])} premium)")

    if "composite_premium" in relative:
        print(f"\n  Composite Premium:  {pct(relative['composite_premium'])}")

    if "assessment" in relative:
        print(f"  Assessment:  {relative['assessment']}")

    # ------------------------------------------------------------------
    # 9. Financial Health Score
    # ------------------------------------------------------------------
    header("FINANCIAL HEALTH SCORE")

    print(f"\n  Overall Score:  {health['score']}/100  (Grade: {health['grade']} -- {grade_color(health['grade'])})")

    subheader("Component Scores")
    for component, score in health["components"].items():
        bar = "#" * (score // 5) + "-" * (20 - score // 5)
        print(f"    {component.capitalize():<15} [{bar}] {score}/100")

    if health.get("flags"):
        subheader("Risk Flags")
        for flag in health["flags"]:
            print(f"    ! {flag}")

    # ------------------------------------------------------------------
    # 10. Earnings Quality
    # ------------------------------------------------------------------
    header("EARNINGS QUALITY")

    print(f"\n  Quality Score:     {eq['quality_score']}/100")
    print(f"  Assessment:        {eq['assessment'].upper()}")
    print(f"  Accruals Ratio:    {eq['accruals_ratio']:.3f}")
    print(f"  Cash Conversion:   {eq['cash_conversion']:.2f}x")

    # Interpret
    print("\n  Interpretation:")
    if eq["accruals_ratio"] < 0:
        print("    - Negative accruals: earnings are backed by cash flow (good).")
    else:
        print("    - Positive accruals: earnings exceed cash flow (investigate).")
    if eq["cash_conversion"] > 1.0:
        print("    - Cash conversion > 1x: company generates more cash than reported earnings.")
    else:
        print("    - Cash conversion < 1x: working capital absorbing cash (monitor).")

    # ------------------------------------------------------------------
    # 11. Final Verdict
    # ------------------------------------------------------------------
    header("INVESTMENT VERDICT")

    # Build the verdict from the data
    positives = []
    negatives = []
    neutrals = []

    # Profitability
    if prof["roe"] > 0.15:
        positives.append(f"Strong ROE ({pct(prof['roe'])})")
    if prof["roic"] > 0.15:
        positives.append(f"High ROIC ({pct(prof['roic'])}) -- creating shareholder value")

    # Growth
    if income["revenue_cagr_3y"] > 0.10:
        positives.append(f"Solid revenue growth (3Y CAGR: {pct(income['revenue_cagr_3y'])})")
    elif income["revenue_cagr_3y"] < 0:
        negatives.append(f"Revenue declining (3Y CAGR: {pct(income['revenue_cagr_3y'])})")

    # Margins
    if income["margin_trend"] == "expanding":
        positives.append("Expanding margins -- operating leverage")
    elif income["margin_trend"] == "contracting":
        negatives.append("Contracting margins -- competitive pressure")

    # Valuation
    if dcf["margin_of_safety"] > 0.15:
        positives.append(f"DCF undervalued (MoS: {pct(dcf['margin_of_safety'])})")
    elif dcf["margin_of_safety"] < -0.15:
        negatives.append(f"DCF overvalued (MoS: {pct(dcf['margin_of_safety'])})")
    else:
        neutrals.append("DCF close to fair value")

    # Earnings quality
    if eq["quality_score"] >= 80:
        positives.append(f"High earnings quality (score: {eq['quality_score']})")
    elif eq["quality_score"] < 50:
        negatives.append(f"Low earnings quality (score: {eq['quality_score']})")

    # Health
    if health["score"] >= 80:
        positives.append(f"Strong financial health (score: {health['score']}, grade: {health['grade']})")
    elif health["score"] < 50:
        negatives.append(f"Weak financial health (score: {health['score']}, grade: {health['grade']})")

    # Leverage
    if lev["debt_to_equity"] > 2.0:
        negatives.append(f"Elevated leverage (D/E: {lev['debt_to_equity']:.2f})")
    if lev["interest_coverage"] < 3.0:
        negatives.append(f"Low interest coverage ({lev['interest_coverage']:.1f}x)")

    # Liquidity
    if liq["current_ratio"] < 1.0:
        negatives.append(f"Current ratio below 1.0 ({liq['current_ratio']:.3f})")

    subheader("Positives")
    for p in positives:
        print(f"    + {p}")
    if not positives:
        print("    (none)")

    subheader("Negatives")
    for n in negatives:
        print(f"    - {n}")
    if not negatives:
        print("    (none)")

    if neutrals:
        subheader("Neutral")
        for n in neutrals:
            print(f"    ~ {n}")

    # Overall recommendation
    score = len(positives) - len(negatives)
    subheader("Recommendation")
    if score >= 3:
        verdict = "BUY"
        reasoning = "Strong fundamentals with multiple positive catalysts."
    elif score >= 1:
        verdict = "HOLD"
        reasoning = "Solid business but valuation or risks temper enthusiasm."
    elif score == 0:
        verdict = "NEUTRAL"
        reasoning = "Positives and negatives roughly balanced."
    else:
        verdict = "AVOID"
        reasoning = "Fundamental concerns outweigh positives."

    print(f"\n    >>> {verdict} <<<")
    print(f"    {reasoning}")

    # Disclaimer
    print("\n" + "-" * 60)
    print("  DISCLAIMER: This is a quantitative analysis template.")
    print("  It is NOT investment advice. Always do your own research.")
    print("-" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Complete fundamental analysis of a stock using wraquant.",
        epilog="Set FMP_API_KEY environment variable for live data.",
    )
    parser.add_argument(
        "ticker",
        nargs="?",
        default="AAPL",
        help="Ticker symbol to analyze (default: AAPL)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force synthetic data even if FMP_API_KEY is set",
    )
    args = parser.parse_args()

    run_analysis(args.ticker, use_live=not args.synthetic)
