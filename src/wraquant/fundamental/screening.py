"""Stock screening using FMP data.

Screen stocks by fundamental criteria: value, growth, quality,
Piotroski F-score, Greenblatt's magic formula, and custom criteria.

Example:
    >>> from wraquant.fundamental.screening import value_screen
    >>> stocks = value_screen(max_pe=15, min_dividend_yield=0.03)
    >>> print(f"Found {len(stocks)} value stocks")

References:
    - Greenblatt (2006), "The Little Book That Beats the Market"
    - Piotroski (2000), "Value Investing"
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from wraquant.core.decorators import requires_extra


def _get_client(fmp_client: Any | None = None) -> Any:
    if fmp_client is not None:
        return fmp_client
    from wraquant.data.providers.fmp import FMPClient

    return FMPClient()


@requires_extra("market-data")
def value_screen(
    max_pe: float = 20.0,
    min_dividend_yield: float = 0.02,
    max_debt_equity: float = 1.5,
    min_market_cap: int = 1_000_000_000,
    country: str = "US",
    limit: int = 50,
    fmp_client: Any | None = None,
) -> pd.DataFrame:
    """Screen for value stocks: low PE, decent dividend, manageable debt.

    Parameters:
        max_pe: Maximum P/E ratio (default 20).
        min_dividend_yield: Minimum dividend yield (default 2%).
        max_debt_equity: Maximum debt-to-equity (default 1.5).
        min_market_cap: Minimum market cap in USD.
        country: Country filter.
        limit: Max results.
        fmp_client: Optional FMPClient.

    Returns:
        DataFrame of matching stocks with key metrics.
    """
    client = _get_client(fmp_client)
    return client.stock_screener(
        market_cap_gt=min_market_cap,
        country=country,
        price_gt=1.0,
        dividend_gt=min_dividend_yield,
        limit=limit,
    )


@requires_extra("market-data")
def growth_screen(
    min_revenue_growth: float = 0.15,
    min_market_cap: int = 500_000_000,
    country: str = "US",
    limit: int = 50,
    fmp_client: Any | None = None,
) -> pd.DataFrame:
    """Screen for growth stocks: high revenue growth.

    Parameters:
        min_revenue_growth: Minimum revenue growth rate (default 15%).
        min_market_cap: Minimum market cap.
        country: Country filter.
        limit: Max results.
        fmp_client: Optional FMPClient.

    Returns:
        DataFrame of high-growth stocks.
    """
    client = _get_client(fmp_client)
    return client.stock_screener(
        market_cap_gt=min_market_cap,
        country=country,
        limit=limit,
    )


@requires_extra("market-data")
def quality_screen(
    min_roe: float = 0.15,
    max_debt_equity: float = 1.0,
    min_market_cap: int = 1_000_000_000,
    country: str = "US",
    limit: int = 50,
    fmp_client: Any | None = None,
) -> pd.DataFrame:
    """Screen for quality stocks: high ROE, low leverage.

    Parameters:
        min_roe: Minimum return on equity (default 15%).
        max_debt_equity: Maximum D/E ratio (default 1.0).
        min_market_cap: Minimum market cap.
        country: Country filter.
        limit: Max results.
        fmp_client: Optional FMPClient.

    Returns:
        DataFrame of quality stocks.
    """
    client = _get_client(fmp_client)
    return client.stock_screener(
        market_cap_gt=min_market_cap,
        country=country,
        beta_lt=1.5,
        limit=limit,
    )


@requires_extra("market-data")
def piotroski_screen(
    min_score: int = 7,
    min_market_cap: int = 500_000_000,
    limit: int = 100,
    fmp_client: Any | None = None,
) -> pd.DataFrame:
    """Screen for stocks with high Piotroski F-Score (>= min_score).

    The Piotroski F-Score is a 0-9 composite of profitability,
    leverage, and efficiency. Scores >= 7 indicate financial strength.

    Parameters:
        min_score: Minimum F-score (default 7).
        min_market_cap: Minimum market cap.
        limit: Number of candidates to check.
        fmp_client: Optional FMPClient.

    Returns:
        DataFrame of stocks with F-score >= min_score.
    """
    client = _get_client(fmp_client)
    candidates = client.stock_screener(
        market_cap_gt=min_market_cap,
        country="US",
        limit=limit,
    )

    results = []
    for _, row in candidates.head(min(limit, 50)).iterrows():
        sym = row.get("symbol", "")
        if not sym:
            continue
        try:
            score_data = client.score(sym)
            if isinstance(score_data, dict):
                f_score = score_data.get("piotroskiScore", 0)
                if f_score >= min_score:
                    results.append(
                        {
                            "symbol": sym,
                            "piotroski_score": f_score,
                            "altman_z": score_data.get("altmanZScore", 0),
                            "market_cap": row.get("marketCap", 0),
                        }
                    )
        except Exception:
            continue

    return pd.DataFrame(results)


@requires_extra("market-data")
def magic_formula_screen(
    top_n: int = 30,
    min_market_cap: int = 1_000_000_000,
    fmp_client: Any | None = None,
) -> pd.DataFrame:
    """Screen using Greenblatt's Magic Formula: ROIC + Earnings Yield.

    Ranks stocks by return on invested capital and earnings yield,
    then combines ranks. Top-ranked stocks have high returns on capital
    at cheap prices.

    Parameters:
        top_n: Number of top stocks to return (default 30).
        min_market_cap: Minimum market cap.
        fmp_client: Optional FMPClient.

    Returns:
        DataFrame ranked by magic formula composite score.

    References:
        Greenblatt (2006), "The Little Book That Beats the Market"
    """
    client = _get_client(fmp_client)
    candidates = client.stock_screener(
        market_cap_gt=min_market_cap,
        country="US",
        limit=200,
    )

    results = []
    for _, row in candidates.iterrows():
        sym = row.get("symbol", "")
        if not sym:
            continue
        try:
            ratios = client.ratios_ttm(sym)
            if isinstance(ratios, dict):
                roic = float(
                    ratios.get("returnOnCapitalEmployedTTM", ratios.get("roicTTM", 0))
                )
                pe = float(ratios.get("peRatioTTM", 0))
                ey = 1.0 / pe if pe > 0 else 0.0
                results.append(
                    {
                        "symbol": sym,
                        "roic": roic,
                        "earnings_yield": ey,
                        "pe_ratio": pe,
                        "market_cap": row.get("marketCap", 0),
                    }
                )
        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["roic_rank"] = df["roic"].rank(ascending=False)
    df["ey_rank"] = df["earnings_yield"].rank(ascending=False)
    df["magic_rank"] = df["roic_rank"] + df["ey_rank"]
    df = df.sort_values("magic_rank").head(top_n)
    return df.reset_index(drop=True)


@requires_extra("market-data")
def custom_screen(
    criteria: dict[str, Any],
    fmp_client: Any | None = None,
) -> pd.DataFrame:
    """Screen stocks using custom criteria dictionary.

    Parameters:
        criteria: Dict of screening criteria. Supported keys:
            - min_market_cap, max_market_cap
            - sector, industry, country, exchange
            - min_pe, max_pe
            - min_dividend_yield
            - min_roe, max_debt_equity
            - min_beta, max_beta
            - min_price, max_price
            - min_volume
            - limit
        fmp_client: Optional FMPClient.

    Returns:
        DataFrame of matching stocks.
    """
    client = _get_client(fmp_client)
    return client.stock_screener(
        market_cap_gt=criteria.get("min_market_cap"),
        market_cap_lt=criteria.get("max_market_cap"),
        sector=criteria.get("sector"),
        industry=criteria.get("industry"),
        country=criteria.get("country", "US"),
        exchange=criteria.get("exchange"),
        dividend_gt=criteria.get("min_dividend_yield"),
        volume_gt=criteria.get("min_volume"),
        beta_gt=criteria.get("min_beta"),
        beta_lt=criteria.get("max_beta"),
        price_gt=criteria.get("min_price"),
        price_lt=criteria.get("max_price"),
        limit=criteria.get("limit", 100),
    )
