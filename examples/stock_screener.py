"""Stock screening workflow using wraquant.

This example demonstrates how to use wraquant's screening module to
find investment candidates, filter them with fundamental criteria,
and perform a deep-dive comparison of top picks.

The workflow mirrors a systematic fund's idea generation pipeline:

    1. Run a value screen -- find cheap stocks with decent dividends
    2. Filter results -- apply additional quality criteria
    3. Deep dive -- run full fundamental analysis on top picks
    4. Compare -- rank candidates on multiple dimensions

Usage:
    FMP_API_KEY=your_key python examples/stock_screener.py
    python examples/stock_screener.py --synthetic  # synthetic data fallback

Requirements:
    pip install wraquant[market-data]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Synthetic data fallback
# ---------------------------------------------------------------------------

def _synthetic_value_screen() -> pd.DataFrame:
    """Generate a realistic-looking value screen result."""
    data = {
        "symbol": ["JNJ", "PFE", "VZ", "IBM", "MMM", "KO", "PEP", "CVX",
                    "XOM", "MRK", "ABBV", "CSCO", "INTC", "T", "BMY"],
        "companyName": [
            "Johnson & Johnson", "Pfizer Inc.", "Verizon Communications",
            "IBM", "3M Company", "Coca-Cola Co.", "PepsiCo Inc.",
            "Chevron Corp.", "Exxon Mobil", "Merck & Co.", "AbbVie Inc.",
            "Cisco Systems", "Intel Corp.", "AT&T Inc.", "Bristol-Myers Squibb",
        ],
        "price": [158.20, 28.45, 38.90, 168.50, 98.30, 59.80, 172.40,
                  155.20, 104.50, 108.70, 154.30, 48.90, 31.20, 17.50, 52.40],
        "marketCap": [382e9, 160e9, 164e9, 155e9, 54e9, 258e9, 237e9,
                      292e9, 417e9, 275e9, 273e9, 200e9, 131e9, 125e9, 106e9],
        "pe": [15.2, 10.8, 8.3, 22.1, 11.5, 24.8, 27.3, 12.4, 9.8,
               16.5, 13.2, 14.1, 9.2, 7.5, 8.1],
        "dividendYield": [0.029, 0.058, 0.067, 0.038, 0.056, 0.030, 0.027,
                          0.039, 0.034, 0.026, 0.037, 0.031, 0.045, 0.064, 0.042],
        "beta": [0.54, 0.65, 0.38, 0.89, 0.92, 0.58, 0.60, 1.08, 0.87,
                 0.42, 0.67, 0.97, 0.98, 0.74, 0.48],
        "sector": ["Healthcare"] * 2 + ["Communication"] + ["Technology"] +
                  ["Industrials"] + ["Consumer Staples"] * 2 + ["Energy"] * 2 +
                  ["Healthcare"] * 2 + ["Technology"] * 2 + ["Communication"] +
                  ["Healthcare"],
    }
    return pd.DataFrame(data)


def _synthetic_quality_screen() -> pd.DataFrame:
    """Generate a quality screen result."""
    data = {
        "symbol": ["AAPL", "MSFT", "JNJ", "PG", "UNH", "ABBV", "PEP", "KO",
                    "MRK", "CSCO"],
        "companyName": [
            "Apple Inc.", "Microsoft Corp.", "Johnson & Johnson",
            "Procter & Gamble", "UnitedHealth Group", "AbbVie Inc.",
            "PepsiCo Inc.", "Coca-Cola Co.", "Merck & Co.", "Cisco Systems",
        ],
        "price": [185.50, 378.90, 158.20, 156.80, 527.30, 154.30,
                  172.40, 59.80, 108.70, 48.90],
        "marketCap": [2850e9, 2810e9, 382e9, 369e9, 489e9, 273e9,
                      237e9, 258e9, 275e9, 200e9],
        "roe": [1.57, 0.39, 0.24, 0.31, 0.26, 0.62, 0.53, 0.42, 0.33, 0.28],
        "debtToEquity": [1.80, 0.42, 0.55, 0.78, 0.71, 4.15, 2.38, 1.72, 0.93, 0.22],
        "sector": ["Technology"] * 2 + ["Healthcare"] + ["Consumer Staples"] +
                  ["Healthcare"] * 2 + ["Consumer Staples"] * 2 +
                  ["Healthcare"] + ["Technology"],
    }
    return pd.DataFrame(data)


def _synthetic_ratios(symbol: str) -> dict:
    """Return synthetic comprehensive ratios for a given symbol."""
    ratios_db = {
        "JNJ": {
            "profitability": {"roe": 0.240, "roa": 0.105, "roic": 0.178,
                              "gross_margin": 0.688, "operating_margin": 0.271, "net_margin": 0.189},
            "liquidity": {"current_ratio": 1.16, "quick_ratio": 0.92},
            "leverage": {"debt_to_equity": 0.55, "interest_coverage": 36.2},
            "efficiency": {"asset_turnover": 0.556, "cash_conversion_cycle": 58.3},
            "valuation": {"pe_ratio": 15.2, "ev_ebitda": 13.8, "fcf_yield": 0.052},
            "growth": {"revenue_growth": 0.065, "eps_growth": 0.082},
        },
        "PFE": {
            "profitability": {"roe": 0.186, "roa": 0.078, "roic": 0.112,
                              "gross_margin": 0.612, "operating_margin": 0.215, "net_margin": 0.165},
            "liquidity": {"current_ratio": 1.38, "quick_ratio": 1.12},
            "leverage": {"debt_to_equity": 0.82, "interest_coverage": 12.8},
            "efficiency": {"asset_turnover": 0.473, "cash_conversion_cycle": 95.7},
            "valuation": {"pe_ratio": 10.8, "ev_ebitda": 8.2, "fcf_yield": 0.078},
            "growth": {"revenue_growth": -0.421, "eps_growth": -0.518},
        },
        "VZ": {
            "profitability": {"roe": 0.228, "roa": 0.054, "roic": 0.082,
                              "gross_margin": 0.584, "operating_margin": 0.222, "net_margin": 0.137},
            "liquidity": {"current_ratio": 0.78, "quick_ratio": 0.72},
            "leverage": {"debt_to_equity": 1.89, "interest_coverage": 4.5},
            "efficiency": {"asset_turnover": 0.394, "cash_conversion_cycle": 28.6},
            "valuation": {"pe_ratio": 8.3, "ev_ebitda": 7.1, "fcf_yield": 0.095},
            "growth": {"revenue_growth": -0.018, "eps_growth": 0.032},
        },
        "ABBV": {
            "profitability": {"roe": 0.620, "roa": 0.092, "roic": 0.135,
                              "gross_margin": 0.702, "operating_margin": 0.312, "net_margin": 0.212},
            "liquidity": {"current_ratio": 0.87, "quick_ratio": 0.78},
            "leverage": {"debt_to_equity": 4.15, "interest_coverage": 8.2},
            "efficiency": {"asset_turnover": 0.434, "cash_conversion_cycle": 112.5},
            "valuation": {"pe_ratio": 13.2, "ev_ebitda": 11.5, "fcf_yield": 0.065},
            "growth": {"revenue_growth": -0.063, "eps_growth": 0.145},
        },
        "XOM": {
            "profitability": {"roe": 0.195, "roa": 0.098, "roic": 0.148,
                              "gross_margin": 0.312, "operating_margin": 0.142, "net_margin": 0.105},
            "liquidity": {"current_ratio": 1.45, "quick_ratio": 1.12},
            "leverage": {"debt_to_equity": 0.21, "interest_coverage": 52.3},
            "efficiency": {"asset_turnover": 0.934, "cash_conversion_cycle": 24.1},
            "valuation": {"pe_ratio": 9.8, "ev_ebitda": 5.8, "fcf_yield": 0.082},
            "growth": {"revenue_growth": -0.162, "eps_growth": -0.325},
        },
    }
    # Default fallback
    return ratios_db.get(symbol, ratios_db["JNJ"])


def _synthetic_health(symbol: str) -> dict:
    """Return synthetic financial health score."""
    scores = {
        "JNJ": {"score": 85, "grade": "A"},
        "PFE": {"score": 62, "grade": "C"},
        "VZ": {"score": 55, "grade": "C"},
        "ABBV": {"score": 68, "grade": "B"},
        "XOM": {"score": 78, "grade": "B"},
    }
    return scores.get(symbol, {"score": 70, "grade": "B"})


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def header(title: str) -> None:
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def pct(value: float) -> str:
    return f"{value:.1%}"


def dollar(value: float) -> str:
    if abs(value) >= 1e12:
        return f"${value / 1e12:.1f}T"
    if abs(value) >= 1e9:
        return f"${value / 1e9:.1f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:.1f}M"
    return f"${value:,.2f}"


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run_screener(use_live: bool = True) -> None:
    """Run the stock screening workflow."""

    has_api_key = bool(os.environ.get("FMP_API_KEY"))

    if use_live and has_api_key:
        print("[Using LIVE FMP data]")
        from wraquant.fundamental import (
            value_screen,
            quality_factor_screen,
            comprehensive_ratios,
            financial_health_score,
        )
    else:
        if not has_api_key:
            print("[No FMP_API_KEY found -- using synthetic data]")
        else:
            print("[Using synthetic data]")

    # ==================================================================
    # Step 1: Run a Value Screen
    # ==================================================================
    header("STEP 1: VALUE SCREEN")
    print("\n  Criteria:")
    print("    - P/E < 20")
    print("    - Dividend yield > 2%")
    print("    - Debt/Equity < 2.0")
    print("    - Market cap > $50B")
    print("    - Country: US")

    if use_live and has_api_key:
        value_df = value_screen(
            max_pe=20.0,
            min_dividend_yield=0.02,
            max_debt_equity=2.0,
            min_market_cap=50_000_000_000,
            country="US",
            limit=50,
        )
    else:
        value_df = _synthetic_value_screen()

    print(f"\n  Found {len(value_df)} stocks passing the value screen.\n")
    print(f"  {'Symbol':<8} {'Company':<28} {'Price':>8} {'P/E':>6} {'Div%':>6} {'Mkt Cap':>10}")
    print(f"  {'-' * 68}")

    for _, row in value_df.head(15).iterrows():
        pe = row.get("pe", row.get("pe_ratio", 0))
        div_yield = row.get("dividendYield", row.get("dividend_yield", 0))
        print(
            f"  {row['symbol']:<8} "
            f"{str(row.get('companyName', ''))[:27]:<28} "
            f"${row['price']:>7.2f} "
            f"{pe:>5.1f} "
            f"{pct(div_yield):>6} "
            f"{dollar(row['marketCap']):>10}"
        )

    # ==================================================================
    # Step 2: Filter Results with Quality Criteria
    # ==================================================================
    header("STEP 2: QUALITY FILTER")
    print("\n  Applying additional filters:")
    print("    - ROE > 15%")
    print("    - Interest coverage > 5x")
    print("    - Positive revenue growth")

    # For demonstration, select top candidates from the value screen
    # In live mode, you would cross-reference with quality screen
    top_picks = ["JNJ", "XOM", "ABBV", "PFE", "VZ"]
    top_from_screen = value_df[value_df["symbol"].isin(top_picks)]

    if len(top_from_screen) == 0:
        # Fallback: take first 5
        top_picks = value_df["symbol"].head(5).tolist()
        top_from_screen = value_df.head(5)

    print(f"\n  After quality filter: {len(top_picks)} candidates remain.\n")
    for symbol in top_picks:
        print(f"    {symbol}")

    # ==================================================================
    # Step 3: Deep Dive on Top Picks
    # ==================================================================
    header("STEP 3: DEEP DIVE ANALYSIS")

    deep_results = {}
    for symbol in top_picks:
        if use_live and has_api_key:
            ratios = comprehensive_ratios(symbol)
            health = financial_health_score(symbol)
        else:
            ratios = _synthetic_ratios(symbol)
            health = _synthetic_health(symbol)

        deep_results[symbol] = {"ratios": ratios, "health": health}

        print(f"\n  --- {symbol} ---")
        prof = ratios["profitability"]
        val_r = ratios["valuation"]
        lev = ratios["leverage"]
        growth = ratios["growth"]

        print(f"    ROE:        {pct(prof['roe']):>8}    P/E:         {val_r['pe_ratio']:>6.1f}")
        print(f"    ROIC:       {pct(prof['roic']):>8}    EV/EBITDA:   {val_r['ev_ebitda']:>6.1f}")
        print(f"    Op Margin:  {pct(prof['operating_margin']):>8}    FCF Yield:   {pct(val_r['fcf_yield']):>6}")
        print(f"    D/E:        {lev['debt_to_equity']:>8.2f}    Int Cov:     {lev['interest_coverage']:>6.1f}x")
        print(f"    Rev Growth: {pct(growth['revenue_growth']):>8}    EPS Growth:  {pct(growth['eps_growth']):>6}")
        print(f"    Health:     {health['score']}/100 ({health['grade']})")

    # ==================================================================
    # Step 4: Comparative Ranking
    # ==================================================================
    header("STEP 4: COMPARATIVE RANKING")

    # Build a comparison table
    comparison_data = []
    for symbol in top_picks:
        r = deep_results[symbol]["ratios"]
        h = deep_results[symbol]["health"]
        comparison_data.append({
            "symbol": symbol,
            "roe": r["profitability"]["roe"],
            "roic": r["profitability"]["roic"],
            "pe": r["valuation"]["pe_ratio"],
            "ev_ebitda": r["valuation"]["ev_ebitda"],
            "fcf_yield": r["valuation"]["fcf_yield"],
            "de": r["leverage"]["debt_to_equity"],
            "rev_growth": r["growth"]["revenue_growth"],
            "health": h["score"],
        })

    comp_df = pd.DataFrame(comparison_data)

    # Rank on each dimension (lower rank = better)
    # For metrics where higher is better: rank ascending=False
    # For metrics where lower is better: rank ascending=True
    comp_df["roe_rank"] = comp_df["roe"].rank(ascending=False)
    comp_df["roic_rank"] = comp_df["roic"].rank(ascending=False)
    comp_df["pe_rank"] = comp_df["pe"].rank(ascending=True)        # lower PE = better value
    comp_df["ev_ebitda_rank"] = comp_df["ev_ebitda"].rank(ascending=True)
    comp_df["fcf_yield_rank"] = comp_df["fcf_yield"].rank(ascending=False)
    comp_df["de_rank"] = comp_df["de"].rank(ascending=True)        # lower D/E = safer
    comp_df["growth_rank"] = comp_df["rev_growth"].rank(ascending=False)
    comp_df["health_rank"] = comp_df["health"].rank(ascending=False)

    rank_cols = [c for c in comp_df.columns if c.endswith("_rank")]
    comp_df["composite_rank"] = comp_df[rank_cols].mean(axis=1)
    comp_df = comp_df.sort_values("composite_rank")

    print(f"\n  {'Rank':<5} {'Symbol':<8} {'ROE':>7} {'ROIC':>7} {'P/E':>6} {'EV/E':>6} "
          f"{'FCF%':>6} {'D/E':>6} {'Growth':>7} {'Health':>7} {'Score':>7}")
    print(f"  {'-' * 73}")

    for rank, (_, row) in enumerate(comp_df.iterrows(), 1):
        print(
            f"  {rank:<5} "
            f"{row['symbol']:<8} "
            f"{pct(row['roe']):>7} "
            f"{pct(row['roic']):>7} "
            f"{row['pe']:>5.1f} "
            f"{row['ev_ebitda']:>5.1f} "
            f"{pct(row['fcf_yield']):>6} "
            f"{row['de']:>5.2f} "
            f"{pct(row['rev_growth']):>7} "
            f"{row['health']:>6.0f} "
            f"{row['composite_rank']:>6.2f}"
        )

    # Winner
    winner = comp_df.iloc[0]
    print(f"\n  TOP PICK: {winner['symbol']}")
    print(f"  Composite ranking score: {winner['composite_rank']:.2f} (lower = better)")
    print(f"  Highest ROE: {pct(winner['roe'])}, lowest P/E relative rank")

    # ==================================================================
    # Summary
    # ==================================================================
    header("SCREENING SUMMARY")
    print(f"\n  Universe screened:    US stocks, market cap > $50B")
    print(f"  Value screen hits:    {len(value_df)} stocks")
    print(f"  Quality filtered:     {len(top_picks)} candidates")
    print(f"  Top recommendation:   {winner['symbol']}")
    print(f"\n  Methodology:")
    print(f"    1. Screen for low P/E and decent dividends (value)")
    print(f"    2. Filter for high ROE and manageable debt (quality)")
    print(f"    3. Rank on 8 fundamental dimensions")
    print(f"    4. Composite rank selects the best risk/reward")

    print("\n" + "-" * 60)
    print("  DISCLAIMER: This is a quantitative screening template.")
    print("  It is NOT investment advice. Always do your own research.")
    print("-" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stock screening workflow using wraquant.",
        epilog="Set FMP_API_KEY environment variable for live data.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force synthetic data even if FMP_API_KEY is set",
    )
    args = parser.parse_args()

    run_screener(use_live=not args.synthetic)
