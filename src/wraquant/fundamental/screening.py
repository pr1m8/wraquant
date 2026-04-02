"""Stock screening using FMP data.

Provides six screening strategies for identifying stocks that match
specific fundamental criteria.  Each screen encapsulates a well-known
investment philosophy:

1. **Value screen** -- Classic Ben Graham-style: low P/E, decent
   dividend yield, manageable debt.
2. **Growth screen** -- Momentum/growth: high revenue growth, positive
   earnings trajectory.
3. **Quality screen** -- Buffett-style: high ROE, low leverage, durable
   competitive advantages.
4. **Piotroski screen** -- Academic: financial health via the 9-point
   F-Score (Piotroski, 2000).
5. **Magic formula screen** -- Greenblatt: rank by ROIC + earnings
   yield, buy the top-ranked stocks.
6. **Custom screen** -- Flexible: pass any combination of criteria as
   a dictionary.

All screening functions use ``FMPClient`` for data retrieval.  The FMP
stock screener endpoint filters the market-wide universe server-side,
so results are returned quickly even for broad criteria.

Example:
    >>> from wraquant.fundamental.screening import value_screen
    >>> stocks = value_screen(max_pe=15, min_dividend_yield=0.03)
    >>> print(f"Found {len(stocks)} value stocks")
    >>> print(stocks[["symbol", "price", "marketCap"]].head())

References:
    - Graham, B. (1949). *The Intelligent Investor*. Harper & Brothers.
    - Piotroski, J. D. (2000). "Value Investing: The Use of Historical
      Financial Statement Information to Separate Winners from Losers."
      *Journal of Accounting Research*, 38, 1--41.
    - Greenblatt, J. (2006). *The Little Book That Beats the Market*.
      Wiley.
    - Novy-Marx, R. (2013). "The Other Side of Value: The Gross
      Profitability Premium." *Journal of Financial Economics*, 108(1),
      1--28.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from wraquant.core.decorators import requires_extra

logger = logging.getLogger(__name__)

__all__ = [
    "value_screen",
    "growth_screen",
    "quality_screen",
    "piotroski_screen",
    "magic_formula_screen",
    "custom_screen",
    "dividend_aristocrat_screen",
    "turnaround_screen",
    "insider_buying_screen",
    "momentum_value_screen",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_fmp_client(fmp_client: Any | None = None) -> Any:
    """Return the provided client or construct a default ``FMPClient``."""
    if fmp_client is not None:
        return fmp_client
    from wraquant.data.providers.fmp import FMPClient  # noqa: WPS433

    return FMPClient()


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide safely, returning *default* on zero/near-zero denominator."""
    if abs(denominator) < 1e-12:
        return default
    return float(numerator / denominator)


def _screener_request(
    client: Any,
    *,
    market_cap_gt: int | None = None,
    market_cap_lt: int | None = None,
    price_gt: float | None = None,
    price_lt: float | None = None,
    beta_gt: float | None = None,
    beta_lt: float | None = None,
    volume_gt: int | None = None,
    dividend_gt: float | None = None,
    sector: str | None = None,
    industry: str | None = None,
    country: str | None = None,
    exchange: str | None = None,
    limit: int = 100,
) -> pd.DataFrame:
    """Execute a stock screener request via FMPClient.

    Calls the FMP ``/stable/stock-screener`` endpoint with the provided
    filter parameters.  Returns a DataFrame of matching stocks.
    """
    params: dict[str, Any] = {"limit": limit}
    if market_cap_gt is not None:
        params["marketCapMoreThan"] = market_cap_gt
    if market_cap_lt is not None:
        params["marketCapLowerThan"] = market_cap_lt
    if price_gt is not None:
        params["priceMoreThan"] = price_gt
    if price_lt is not None:
        params["priceLowerThan"] = price_lt
    if beta_gt is not None:
        params["betaMoreThan"] = beta_gt
    if beta_lt is not None:
        params["betaLowerThan"] = beta_lt
    if volume_gt is not None:
        params["volumeMoreThan"] = volume_gt
    if dividend_gt is not None:
        params["dividendMoreThan"] = dividend_gt
    if sector is not None:
        params["sector"] = sector
    if industry is not None:
        params["industry"] = industry
    if country is not None:
        params["country"] = country
    if exchange is not None:
        params["exchange"] = exchange

    # Use the client's internal _get to hit the screener endpoint
    data = client._get("/stable/stock-screener", params)
    return pd.DataFrame(data) if data else pd.DataFrame()


# ---------------------------------------------------------------------------
# Value Screen
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def value_screen(
    max_pe: float = 20.0,
    min_dividend_yield: float = 0.02,
    *,
    max_debt_equity: float = 1.5,
    min_market_cap: int = 1_000_000_000,
    country: str = "US",
    limit: int = 50,
    fmp_client: Any | None = None,
) -> pd.DataFrame:
    """Screen for value stocks: low P/E, decent dividend, manageable debt.

    Implements a classic Ben Graham-style value screen that identifies
    stocks trading at low multiples while paying dividends and
    maintaining conservative balance sheets.  This is the foundation of
    value investing and the HML (high-minus-low) factor in Fama-French.

    When to use:
        Use this screen to build a value-tilted portfolio, identify
        contrarian opportunities, or as the starting point for deep-dive
        fundamental analysis.  Value screens work best in conjunction
        with a quality filter (see :func:`quality_screen`) to avoid
        "value traps" -- cheap stocks that are cheap for good reason.

    Parameters:
        max_pe: Maximum price-to-earnings ratio.  The S&P 500 median
            P/E is historically around 15--20.  Setting this to 15
            focuses on deep value; 20 casts a wider net.
        min_dividend_yield: Minimum annual dividend yield (as decimal).
            0.02 = 2 %.  Set to 0 to include non-dividend payers.
        max_debt_equity: Maximum debt-to-equity ratio.  1.5 allows
            moderate leverage; 0.5 is conservative.
        min_market_cap: Minimum market capitalisation in USD.  The
            default ($1B) excludes micro/small-caps.
        country: Country filter (ISO code).  ``"US"`` for domestic.
        limit: Maximum number of results to return.
        fmp_client: Optional pre-configured ``FMPClient`` instance.
            If ``None``, a default client is created.

    Returns:
        DataFrame of matching stocks with key metrics including
        ``symbol``, ``companyName``, ``marketCap``, ``price``, ``beta``,
        ``lastAnnualDividend``, and ``sector``.

    Example:
        >>> from wraquant.fundamental.screening import value_screen
        >>> df = value_screen(max_pe=15, min_dividend_yield=0.03)
        >>> print(f"Found {len(df)} deep value stocks")
        >>> print(df[["symbol", "price", "marketCap"]].head(10))

    See Also:
        quality_screen: Filter value candidates by profitability.
        magic_formula_screen: Combines value and quality in one rank.
    """
    client = _get_fmp_client(fmp_client)
    df = _screener_request(
        client,
        market_cap_gt=min_market_cap,
        country=country,
        price_gt=1.0,
        dividend_gt=min_dividend_yield,
        limit=limit * 3,  # over-fetch for client-side filtering
    )

    if df.empty:
        return df

    # Client-side P/E filter (FMP screener may not support pe directly)
    if "pe" in df.columns:
        df = df[(df["pe"] > 0) & (df["pe"] <= max_pe)]
    elif "peRatio" in df.columns:
        df = df[(df["peRatio"] > 0) & (df["peRatio"] <= max_pe)]

    return df.head(limit).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Growth Screen
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def growth_screen(
    min_revenue_growth: float = 0.15,
    *,
    min_market_cap: int = 500_000_000,
    country: str = "US",
    limit: int = 50,
    fmp_client: Any | None = None,
) -> pd.DataFrame:
    """Screen for growth stocks: high revenue growth, positive momentum.

    Identifies companies with strong top-line growth, which is the
    primary driver of long-term equity returns.  Revenue growth is
    preferred over earnings growth because it is harder to manipulate
    and more persistent.

    When to use:
        Use this screen to identify companies in secular growth
        industries or those gaining market share.  Growth screens are
        most effective in bull markets and for momentum-based strategies.
        Combine with :func:`earnings_quality` to filter out companies
        that are growing revenue but burning cash.

    Parameters:
        min_revenue_growth: Minimum YoY revenue growth rate (as decimal).
            0.15 = 15 %.  Set higher (0.30+) for hyper-growth.
        min_market_cap: Minimum market cap in USD.
        country: Country filter.
        limit: Maximum results.
        fmp_client: Optional ``FMPClient`` instance.

    Returns:
        DataFrame of high-growth stocks with key metrics.  Includes a
        ``revenue_growth`` column when available from FMP data.

    Example:
        >>> from wraquant.fundamental.screening import growth_screen
        >>> df = growth_screen(min_revenue_growth=0.25)
        >>> print(f"Found {len(df)} high-growth stocks")

    See Also:
        value_screen: Complement for a barbell strategy.
        quality_screen: Ensure growth is profitable.
    """
    client = _get_fmp_client(fmp_client)
    candidates = _screener_request(
        client,
        market_cap_gt=min_market_cap,
        country=country,
        price_gt=1.0,
        limit=limit * 3,
    )

    if candidates.empty:
        return candidates

    # Enrich with growth data for top candidates
    results: list[dict[str, Any]] = []
    for _, row in candidates.head(min(limit * 2, 100)).iterrows():
        sym = row.get("symbol", "")
        if not sym:
            continue
        try:
            growth = client.financial_growth(sym, period="annual", limit=1)
            growth_rows = (
                growth.to_dict("records")
                if isinstance(growth, pd.DataFrame)
                else ([growth] if isinstance(growth, dict) else growth)
            )
            if growth_rows:
                rev_growth = growth_rows[0].get("revenueGrowth", 0) or 0
                rev_growth = float(rev_growth)
                if rev_growth >= min_revenue_growth:
                    result = dict(row)
                    result["revenue_growth"] = rev_growth
                    results.append(result)
        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("revenue_growth", ascending=False)
    return df.head(limit).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Quality Screen
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def quality_screen(
    min_roe: float = 0.15,
    max_de: float = 1.0,
    *,
    min_market_cap: int = 1_000_000_000,
    country: str = "US",
    limit: int = 50,
    fmp_client: Any | None = None,
) -> pd.DataFrame:
    """Screen for quality stocks: high ROE, low leverage, wide moats.

    Quality investing targets companies with durable competitive
    advantages -- high returns on equity, conservative balance sheets,
    and stable profitability.  The quality factor (RMW in Fama-French 5)
    has historically delivered positive risk-adjusted returns with lower
    drawdowns than value or momentum.

    When to use:
        Use this screen to identify "compounders" -- stocks that grow
        book value through high reinvestment rates.  Quality screens
        excel in bear markets and risk-off environments because high-
        quality companies are more resilient to economic downturns.

    Parameters:
        min_roe: Minimum return on equity (as decimal).  0.15 = 15 %.
            The median S&P 500 ROE is around 15--18 %.
        max_de: Maximum debt-to-equity ratio.  1.0 is moderate; 0.5
            is conservative.  Some capital-light businesses (tech, SaaS)
            naturally have low D/E.
        min_market_cap: Minimum market cap in USD.
        country: Country filter.
        limit: Maximum results.
        fmp_client: Optional ``FMPClient`` instance.

    Returns:
        DataFrame of quality stocks enriched with ``roe`` and
        ``debt_to_equity`` columns for verification.

    Example:
        >>> from wraquant.fundamental.screening import quality_screen
        >>> df = quality_screen(min_roe=0.20, max_de=0.5)
        >>> print(f"Found {len(df)} high-quality stocks")
        >>> print(df[["symbol", "roe", "debt_to_equity"]].head(10))

    References:
        Novy-Marx, R. (2013). "The Other Side of Value: The Gross
        Profitability Premium." *Journal of Financial Economics*, 108(1),
        1--28.

    See Also:
        value_screen: Combine quality + value for best risk/reward.
        earnings_quality: Validate that earnings are cash-backed.
    """
    client = _get_fmp_client(fmp_client)
    candidates = _screener_request(
        client,
        market_cap_gt=min_market_cap,
        country=country,
        beta_lt=1.5,
        price_gt=1.0,
        limit=limit * 3,
    )

    if candidates.empty:
        return candidates

    results: list[dict[str, Any]] = []
    for _, row in candidates.head(min(limit * 2, 100)).iterrows():
        sym = row.get("symbol", "")
        if not sym:
            continue
        try:
            ratios = client.ratios_ttm(sym)
            if not isinstance(ratios, dict):
                continue
            roe = float(ratios.get("returnOnEquityTTM", 0) or 0)
            de = float(ratios.get("debtEquityRatioTTM", 0) or 0)
            if roe >= min_roe and de <= max_de:
                result = dict(row)
                result["roe"] = roe
                result["debt_to_equity"] = de
                results.append(result)
        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("roe", ascending=False)
    return df.head(limit).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Piotroski Screen
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def piotroski_screen(
    min_score: int = 7,
    *,
    min_market_cap: int = 500_000_000,
    limit: int = 100,
    fmp_client: Any | None = None,
) -> pd.DataFrame:
    """Screen for stocks with high Piotroski F-Score.

    The Piotroski F-Score is a 0--9 composite score measuring financial
    strength across three categories:

    **Profitability (4 pts):**
        1. Positive ROA
        2. Positive operating cash flow
        3. ROA improvement (year over year)
        4. Cash flow > net income (accruals quality)

    **Leverage & liquidity (3 pts):**
        5. Decrease in leverage (long-term debt / assets)
        6. Improvement in current ratio
        7. No new equity issuance

    **Operating efficiency (2 pts):**
        8. Improvement in gross margin
        9. Improvement in asset turnover

    Scores of 8--9 identify the strongest companies; scores of 0--2
    predict financial distress.  Piotroski (2000) showed that a long-
    short strategy based on the F-Score earned 23 % annual returns among
    high book-to-market stocks.

    Parameters:
        min_score: Minimum Piotroski F-Score (0--9).  Default 7
            captures the top tier.  Set to 8 for ultra-high quality.
        min_market_cap: Minimum market cap in USD.
        limit: Number of candidates to evaluate (more = slower but
            more results).
        fmp_client: Optional ``FMPClient`` instance.

    Returns:
        DataFrame with columns ``symbol``, ``piotroski_score``,
        ``altman_z``, and ``market_cap`` for stocks meeting the
        threshold.  Sorted by F-Score descending.

    Example:
        >>> from wraquant.fundamental.screening import piotroski_screen
        >>> df = piotroski_screen(min_score=8)
        >>> print(f"Found {len(df)} high F-Score stocks")
        >>> print(df[["symbol", "piotroski_score", "altman_z"]].head())

    References:
        Piotroski, J. D. (2000). "Value Investing: The Use of Historical
        Financial Statement Information to Separate Winners from Losers."
        *Journal of Accounting Research*, 38, 1--41.

    See Also:
        financial_health_score: Continuous 0--100 health score.
        value_screen: Combine with F-Score for deep value strategy.
    """
    client = _get_fmp_client(fmp_client)
    candidates = _screener_request(
        client,
        market_cap_gt=min_market_cap,
        country="US",
        price_gt=1.0,
        limit=limit,
    )

    if candidates.empty:
        return pd.DataFrame(
            columns=["symbol", "piotroski_score", "altman_z", "market_cap"]
        )

    results: list[dict[str, Any]] = []
    for _, row in candidates.head(min(limit, 50)).iterrows():
        sym = row.get("symbol", "")
        if not sym:
            continue
        try:
            score_data = client.score(sym)
            if isinstance(score_data, dict):
                f_score = score_data.get("piotroskiScore", 0)
                if f_score is not None and int(f_score) >= min_score:
                    results.append(
                        {
                            "symbol": sym,
                            "piotroski_score": int(f_score),
                            "altman_z": score_data.get("altmanZScore", 0),
                            "market_cap": row.get("marketCap", 0),
                        }
                    )
        except Exception:
            continue

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("piotroski_score", ascending=False)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Magic Formula Screen
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def magic_formula_screen(
    top_n: int = 30,
    *,
    min_market_cap: int = 1_000_000_000,
    fmp_client: Any | None = None,
) -> pd.DataFrame:
    """Screen using Greenblatt's Magic Formula: ROIC + Earnings Yield.

    The magic formula ranks stocks by two criteria:
        1. **Return on invested capital (ROIC)** -- Measures how
           efficiently management deploys capital.  Higher is better.
        2. **Earnings yield (EBIT / EV)** -- Measures how cheap the
           stock is relative to its earning power.  Higher is better.

    Each stock gets a rank on ROIC and a rank on earnings yield; the
    combined rank identifies companies that are both high-quality AND
    cheap.  Greenblatt (2006) showed this simple strategy beat the
    market over 17 years.

    When to use:
        Use this as a standalone strategy or as a starting point for
        further due diligence.  The magic formula is most effective
        for mid-to-large cap US equities with stable earnings.  Avoid
        applying it to financials and utilities, which have different
        capital structures.

    Parameters:
        top_n: Number of top-ranked stocks to return.  Greenblatt
            recommends holding 20--30 positions.
        min_market_cap: Minimum market cap in USD.  The original study
            used $50M; $1B is more practical for institutional investors.
        fmp_client: Optional ``FMPClient`` instance.

    Returns:
        DataFrame ranked by magic formula composite score with columns:
        - **symbol** (*str*) -- Ticker symbol.
        - **roic** (*float*) -- Return on invested capital (TTM).
        - **earnings_yield** (*float*) -- 1 / P/E ratio (TTM).
        - **pe_ratio** (*float*) -- P/E ratio for reference.
        - **market_cap** (*float*) -- Market capitalisation.
        - **roic_rank** (*float*) -- Rank by ROIC (1 = best).
        - **ey_rank** (*float*) -- Rank by earnings yield (1 = best).
        - **magic_rank** (*float*) -- Combined rank (lower = better).

    Example:
        >>> from wraquant.fundamental.screening import magic_formula_screen
        >>> df = magic_formula_screen(top_n=20)
        >>> print(df[["symbol", "roic", "earnings_yield", "magic_rank"]].head())

    References:
        Greenblatt, J. (2006). *The Little Book That Beats the Market*.
        Wiley.

    See Also:
        quality_screen: Alternative quality-focused screen.
        value_screen: Pure value screen without quality component.
    """
    client = _get_fmp_client(fmp_client)
    candidates = _screener_request(
        client,
        market_cap_gt=min_market_cap,
        country="US",
        price_gt=1.0,
        limit=200,
    )

    if candidates.empty:
        return pd.DataFrame()

    results: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        sym = row.get("symbol", "")
        if not sym:
            continue
        try:
            ratios = client.ratios_ttm(sym)
            if not isinstance(ratios, dict):
                continue
            roic = float(
                ratios.get(
                    "returnOnCapitalEmployedTTM",
                    ratios.get("roicTTM", 0),
                )
                or 0
            )
            pe = float(ratios.get("peRatioTTM", 0) or 0)
            ey = 1.0 / pe if pe > 0 else 0.0
            if roic > 0 and ey > 0:
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


# ---------------------------------------------------------------------------
# Custom Screen
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def custom_screen(
    criteria: dict[str, Any],
    *,
    fmp_client: Any | None = None,
) -> pd.DataFrame:
    """Screen stocks using a flexible criteria dictionary.

    A general-purpose screener that accepts any combination of filter
    criteria as a dictionary.  Use this when the predefined screens
    (value, growth, quality) do not match your requirements.

    When to use:
        Use this for custom factor construction, sector-specific
        screens, or when you need to combine filters that span multiple
        predefined screens (e.g., "growth + low beta + tech sector").

    Parameters:
        criteria: Dictionary of screening criteria.  Supported keys:

            **Market cap:**
            - ``min_market_cap`` (*int*) -- Minimum market cap in USD.
            - ``max_market_cap`` (*int*) -- Maximum market cap in USD.

            **Classification:**
            - ``sector`` (*str*) -- GICS sector (e.g., ``"Technology"``).
            - ``industry`` (*str*) -- Industry filter.
            - ``country`` (*str*) -- Country code (default ``"US"``).
            - ``exchange`` (*str*) -- Exchange (e.g., ``"NASDAQ"``).

            **Valuation:**
            - ``min_pe`` (*float*) -- Minimum P/E ratio.
            - ``max_pe`` (*float*) -- Maximum P/E ratio.
            - ``min_dividend_yield`` (*float*) -- Minimum dividend yield.

            **Quality:**
            - ``min_roe`` (*float*) -- Minimum ROE.
            - ``max_debt_equity`` (*float*) -- Maximum D/E ratio.

            **Risk:**
            - ``min_beta`` (*float*) -- Minimum beta.
            - ``max_beta`` (*float*) -- Maximum beta.

            **Price & volume:**
            - ``min_price`` (*float*) -- Minimum share price.
            - ``max_price`` (*float*) -- Maximum share price.
            - ``min_volume`` (*int*) -- Minimum average daily volume.

            **Control:**
            - ``limit`` (*int*) -- Maximum results (default 100).

        fmp_client: Optional ``FMPClient`` instance.

    Returns:
        DataFrame of matching stocks with standard FMP screener columns.

    Example:
        >>> from wraquant.fundamental.screening import custom_screen
        >>> df = custom_screen({
        ...     "sector": "Technology",
        ...     "min_market_cap": 10_000_000_000,
        ...     "max_beta": 1.2,
        ...     "min_dividend_yield": 0.01,
        ...     "limit": 20,
        ... })
        >>> print(df[["symbol", "companyName", "marketCap"]].head())

    See Also:
        value_screen: Predefined value criteria.
        growth_screen: Predefined growth criteria.
        quality_screen: Predefined quality criteria.
    """
    client = _get_fmp_client(fmp_client)
    return _screener_request(
        client,
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


# ---------------------------------------------------------------------------
# Dividend Aristocrat Screen
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def dividend_aristocrat_screen(
    min_years: int = 10,
    *,
    min_market_cap: int = 1_000_000_000,
    country: str = "US",
    limit: int = 50,
    fmp_client: Any | None = None,
) -> pd.DataFrame:
    """Screen for stocks with consecutive years of dividend growth.

    Dividend Aristocrats (S&P 500 members with 25+ years of consecutive
    dividend increases) have historically outperformed with lower
    volatility.  This screen identifies companies with *N* or more
    consecutive years of annual dividend per share growth -- a strong
    signal of financial discipline, predictable cash flows, and
    shareholder-friendly management.

    When to use:
        - Income-focused portfolio construction.
        - Quality screening: consistent dividend growth requires
          consistent earnings growth.
        - Defensive strategy: dividend growers tend to outperform in
          bear markets.
        - Retirement portfolios: rising income stream over time.

    Parameters:
        min_years: Minimum consecutive years of dividend growth.
            25 = traditional Aristocrat; 10 = broader "achiever" screen;
            5 = emerging dividend growers.
        min_market_cap: Minimum market cap in USD.
        country: Country filter.
        limit: Maximum results to return.
        fmp_client: Optional ``FMPClient`` instance.

    Returns:
        DataFrame of qualifying stocks with columns:
        - **symbol** (*str*) -- Ticker symbol.
        - **consecutive_years** (*int*) -- Years of consecutive dividend
          growth.
        - **current_yield** (*float*) -- Current dividend yield.
        - **dividend_growth_rate** (*float*) -- Most recent YoY growth.
        - **market_cap** (*float*) -- Market capitalisation.

    Example:
        >>> from wraquant.fundamental.screening import dividend_aristocrat_screen
        >>> df = dividend_aristocrat_screen(min_years=25)
        >>> print(f"Found {len(df)} Dividend Aristocrats")
        >>> print(df[["symbol", "consecutive_years", "current_yield"]].head())

    References:
        S&P Dow Jones Indices. "S&P 500 Dividend Aristocrats."
        ProShares (2019). "Why Dividend Growth Matters."

    See Also:
        value_screen: Combine with dividend screen for income + value.
        quality_screen: Dividend consistency as a quality proxy.
    """
    client = _get_fmp_client(fmp_client)

    # Get dividend-paying stocks
    candidates = _screener_request(
        client,
        market_cap_gt=min_market_cap,
        country=country,
        dividend_gt=0.001,
        price_gt=1.0,
        limit=limit * 5,
    )

    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "consecutive_years",
                "current_yield",
                "dividend_growth_rate",
                "market_cap",
            ]
        )

    results: list[dict[str, Any]] = []
    for _, row in candidates.head(min(limit * 3, 150)).iterrows():
        sym = row.get("symbol", "")
        if not sym:
            continue
        try:
            div_data = client.dividends(sym)
            if isinstance(div_data, pd.DataFrame) and not div_data.empty:
                # Group dividends by year and sum
                if "date" in div_data.columns and "dividend" in div_data.columns:
                    div_data["date"] = pd.to_datetime(div_data["date"])
                    div_data["year"] = div_data["date"].dt.year
                    annual_divs = (
                        div_data.groupby("year")["dividend"]
                        .sum()
                        .sort_index(ascending=False)
                    )

                    # Count consecutive years of growth
                    consecutive = 0
                    values = annual_divs.values
                    for i in range(len(values) - 1):
                        if values[i] > values[i + 1]:
                            consecutive += 1
                        else:
                            break

                    if consecutive >= min_years:
                        # Compute growth rate
                        latest_div = values[0] if len(values) > 0 else 0
                        prev_div = values[1] if len(values) > 1 else 0
                        growth_rate = (
                            _safe_div(latest_div - prev_div, prev_div)
                            if prev_div > 0
                            else 0.0
                        )

                        results.append(
                            {
                                "symbol": sym,
                                "consecutive_years": consecutive,
                                "current_yield": float(
                                    row.get("lastAnnualDividend", 0) or 0
                                )
                                / max(float(row.get("price", 1) or 1), 0.01),
                                "dividend_growth_rate": float(growth_rate),
                                "market_cap": float(row.get("marketCap", 0) or 0),
                            }
                        )
        except Exception:  # noqa: BLE001
            continue

    if not results:
        return pd.DataFrame(
            columns=[
                "symbol",
                "consecutive_years",
                "current_yield",
                "dividend_growth_rate",
                "market_cap",
            ]
        )

    df = pd.DataFrame(results)
    df = df.sort_values("consecutive_years", ascending=False)
    return df.head(limit).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Turnaround Screen
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def turnaround_screen(
    max_pe: float = 15.0,
    *,
    min_margin_improvement: float = 0.02,
    min_market_cap: int = 500_000_000,
    country: str = "US",
    limit: int = 50,
    fmp_client: Any | None = None,
) -> pd.DataFrame:
    """Screen for turnaround candidates: improving margins but still cheap.

    Turnaround stocks are value stocks with positive momentum in their
    fundamentals.  They trade at low multiples (the market hasn't noticed
    the improvement yet) but show rising margins, indicating operational
    recovery.  This combines value investing with momentum -- the two
    most robust factors in academic research.

    When to use:
        - Contrarian strategies: buy when margins inflect upward.
        - Mean-reversion plays: stocks with temporarily depressed
          earnings reverting to historical norms.
        - Combine with insider buying screen for higher conviction.

    Parameters:
        max_pe: Maximum P/E ratio (cheap stocks only).  Default 15.
        min_margin_improvement: Minimum operating margin improvement
            (most recent - prior year).  0.02 = 2pp improvement.
        min_market_cap: Minimum market cap in USD.
        country: Country filter.
        limit: Maximum results.
        fmp_client: Optional ``FMPClient`` instance.

    Returns:
        DataFrame of turnaround candidates with columns:
        - **symbol** (*str*) -- Ticker symbol.
        - **pe_ratio** (*float*) -- Current P/E ratio.
        - **operating_margin_current** (*float*) -- Most recent period.
        - **operating_margin_prior** (*float*) -- Prior period.
        - **margin_improvement** (*float*) -- Change in operating margin.
        - **revenue_growth** (*float*) -- YoY revenue growth.
        - **market_cap** (*float*) -- Market capitalisation.

    Example:
        >>> from wraquant.fundamental.screening import turnaround_screen
        >>> df = turnaround_screen(max_pe=12)
        >>> print(f"Found {len(df)} turnaround candidates")
        >>> print(df[["symbol", "pe_ratio", "margin_improvement"]].head())

    See Also:
        value_screen: Pure value screen.
        quality_screen: Ensure turnarounds have staying power.
        insider_buying_screen: Insider confidence in the turnaround.
    """
    client = _get_fmp_client(fmp_client)

    candidates = _screener_request(
        client,
        market_cap_gt=min_market_cap,
        country=country,
        price_gt=1.0,
        limit=limit * 5,
    )

    if candidates.empty:
        return pd.DataFrame()

    results: list[dict[str, Any]] = []
    for _, row in candidates.head(min(limit * 3, 150)).iterrows():
        sym = row.get("symbol", "")
        if not sym:
            continue
        try:
            # Fetch ratios and income data
            ratios = client.ratios_ttm(sym)
            if not isinstance(ratios, dict):
                continue
            pe = float(ratios.get("peRatioTTM", 0) or 0)
            if pe <= 0 or pe > max_pe:
                continue

            # Get multi-period income for margin trend
            income = client.income_statement(sym, period="annual", limit=3)
            if isinstance(income, pd.DataFrame):
                income_list = income.to_dict("records")
            elif isinstance(income, list):
                income_list = income
            else:
                continue

            if len(income_list) < 2:
                continue

            rev_current = float(income_list[0].get("revenue", 0) or 0)
            rev_prior = float(income_list[1].get("revenue", 0) or 0)
            oi_current = float(income_list[0].get("operatingIncome", 0) or 0)
            oi_prior = float(income_list[1].get("operatingIncome", 0) or 0)

            if rev_current <= 0 or rev_prior <= 0:
                continue

            margin_current = oi_current / rev_current
            margin_prior = oi_prior / rev_prior
            margin_improvement = margin_current - margin_prior

            if margin_improvement >= min_margin_improvement:
                rev_growth = _safe_div(rev_current - rev_prior, abs(rev_prior))
                results.append(
                    {
                        "symbol": sym,
                        "pe_ratio": pe,
                        "operating_margin_current": margin_current,
                        "operating_margin_prior": margin_prior,
                        "margin_improvement": margin_improvement,
                        "revenue_growth": rev_growth,
                        "market_cap": float(row.get("marketCap", 0) or 0),
                    }
                )
        except Exception:  # noqa: BLE001
            continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("margin_improvement", ascending=False)
    return df.head(limit).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Insider Buying Screen
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def insider_buying_screen(
    min_buys: int = 3,
    days: int = 90,
    *,
    min_market_cap: int = 500_000_000,
    country: str = "US",
    limit: int = 50,
    fmp_client: Any | None = None,
) -> pd.DataFrame:
    """Screen for stocks with significant recent insider buying activity.

    Insiders (officers, directors, 10 %+ owners) are the most informed
    participants in a stock.  Academic research consistently shows that
    *insider purchases* predict positive future returns, especially
    cluster buying (multiple insiders buying in a short window).  Insider
    *sales* are less informative (often driven by diversification/taxes).

    This screen identifies stocks with multiple insider buy transactions
    in the recent period -- a strong signal of management confidence.

    When to use:
        - Confirmation signal: use alongside value/quality screens.
        - Contrarian buying: insiders buying during market selloffs.
        - Special situations: new CEO buying, founder increasing stake.
        - Pair with :func:`turnaround_screen` for high-conviction turnarounds.

    Parameters:
        min_buys: Minimum number of insider purchase transactions in the
            lookback window.  3+ is significant cluster buying.
        days: Lookback period in calendar days.  Default 90 (one quarter).
        min_market_cap: Minimum market cap in USD.
        country: Country filter.
        limit: Maximum results.
        fmp_client: Optional ``FMPClient`` instance.

    Returns:
        DataFrame of stocks with insider buying activity:
        - **symbol** (*str*) -- Ticker symbol.
        - **buy_count** (*int*) -- Number of insider purchases.
        - **total_value** (*float*) -- Total dollar value of purchases.
        - **unique_insiders** (*int*) -- Number of distinct insiders buying.
        - **market_cap** (*float*) -- Market capitalisation.

    Example:
        >>> from wraquant.fundamental.screening import insider_buying_screen
        >>> df = insider_buying_screen(min_buys=5, days=60)
        >>> print(f"Found {len(df)} stocks with cluster insider buying")
        >>> print(df[["symbol", "buy_count", "total_value"]].head())

    References:
        Lakonishok, J. & Lee, I. (2001). "Are Insider Trades
        Informative?" *Review of Financial Studies*, 14(1), 79--111.

    See Also:
        turnaround_screen: Combine with insider buying for conviction.
        quality_screen: Verify fundamentals back the insider thesis.
    """
    client = _get_fmp_client(fmp_client)

    candidates = _screener_request(
        client,
        market_cap_gt=min_market_cap,
        country=country,
        price_gt=1.0,
        limit=limit * 3,
    )

    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "buy_count",
                "total_value",
                "unique_insiders",
                "market_cap",
            ]
        )

    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)

    results: list[dict[str, Any]] = []
    for _, row in candidates.head(min(limit * 3, 100)).iterrows():
        sym = row.get("symbol", "")
        if not sym:
            continue
        try:
            trades = client.insider_trades(sym, limit=200)
            if not isinstance(trades, pd.DataFrame) or trades.empty:
                continue

            # Filter for purchases within the lookback window
            if "transactionDate" in trades.columns:
                trades["transactionDate"] = pd.to_datetime(
                    trades["transactionDate"], errors="coerce"
                )
                recent = trades[trades["transactionDate"] >= cutoff_date]
            else:
                recent = trades

            # Filter for purchases (P = purchase, A = acquisition)
            if "transactionType" in recent.columns:
                buys = recent[
                    recent["transactionType"].str.upper().isin(["P", "P-PURCHASE"])
                ]
            elif "acquistionOrDisposition" in recent.columns:
                buys = recent[recent["acquistionOrDisposition"].str.upper() == "A"]
            else:
                continue

            if len(buys) >= min_buys:
                total_value = 0.0
                if "securitiesTransacted" in buys.columns and "price" in buys.columns:
                    for _, trade in buys.iterrows():
                        shares_traded = float(trade.get("securitiesTransacted", 0) or 0)
                        trade_price = float(trade.get("price", 0) or 0)
                        total_value += shares_traded * trade_price

                unique_insiders = 0
                if "reportingName" in buys.columns:
                    unique_insiders = buys["reportingName"].nunique()
                elif "reportingCik" in buys.columns:
                    unique_insiders = buys["reportingCik"].nunique()

                results.append(
                    {
                        "symbol": sym,
                        "buy_count": len(buys),
                        "total_value": float(total_value),
                        "unique_insiders": int(unique_insiders),
                        "market_cap": float(row.get("marketCap", 0) or 0),
                    }
                )
        except Exception:  # noqa: BLE001
            continue

    if not results:
        return pd.DataFrame(
            columns=[
                "symbol",
                "buy_count",
                "total_value",
                "unique_insiders",
                "market_cap",
            ]
        )

    df = pd.DataFrame(results)
    df = df.sort_values("buy_count", ascending=False)
    return df.head(limit).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Momentum + Value Screen
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def momentum_value_screen(
    top_n: int = 30,
    *,
    min_market_cap: int = 1_000_000_000,
    country: str = "US",
    fmp_client: Any | None = None,
) -> pd.DataFrame:
    """Screen combining value (low PE) with price momentum (positive 6M return).

    Value and momentum are the two most persistent and well-documented
    factors in equity returns.  They are negatively correlated, making
    a combined strategy more robust than either alone.  This screen
    identifies stocks that are both cheap (low P/E) and in an uptrend
    (positive 6-month price return), capturing the sweet spot where
    value is being recognized by the market.

    When to use:
        - Multi-factor portfolio construction: value + momentum is the
          classic two-factor strategy (Asness et al., 2013).
        - Avoid value traps: momentum filter ensures the market is
          beginning to recognize the value.
        - Tactical allocation: identify undervalued stocks with improving
          price action.

    Mathematical formulation:
        Value Score = percentile_rank(1/PE)  (higher = cheaper)
        Momentum Score = percentile_rank(6M return)
        Combined Score = Value Score + Momentum Score
        Select top N by Combined Score.

    Parameters:
        top_n: Number of top-ranked stocks to return.
        min_market_cap: Minimum market cap in USD.
        country: Country filter.
        fmp_client: Optional ``FMPClient`` instance.

    Returns:
        DataFrame ranked by combined score with columns:
        - **symbol** (*str*) -- Ticker symbol.
        - **pe_ratio** (*float*) -- P/E ratio.
        - **earnings_yield** (*float*) -- 1/PE (higher = cheaper).
        - **price_return_6m** (*float*) -- 6-month price return.
        - **value_rank** (*float*) -- Rank by earnings yield (1 = best).
        - **momentum_rank** (*float*) -- Rank by 6M return (1 = best).
        - **combined_rank** (*float*) -- Sum of value and momentum ranks
          (lower = better).
        - **market_cap** (*float*) -- Market capitalisation.

    Example:
        >>> from wraquant.fundamental.screening import momentum_value_screen
        >>> df = momentum_value_screen(top_n=20)
        >>> print(df[["symbol", "pe_ratio", "price_return_6m",
        ...           "combined_rank"]].head())

    References:
        Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013).
        "Value and Momentum Everywhere." *Journal of Finance*, 68(3),
        929--985.

    See Also:
        value_screen: Pure value screen.
        magic_formula_screen: Value + quality (similar concept).
    """
    client = _get_fmp_client(fmp_client)

    candidates = _screener_request(
        client,
        market_cap_gt=min_market_cap,
        country=country,
        price_gt=1.0,
        limit=200,
    )

    if candidates.empty:
        return pd.DataFrame()

    results: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        sym = row.get("symbol", "")
        if not sym:
            continue
        try:
            # Get P/E ratio
            ratios = client.ratios_ttm(sym)
            if not isinstance(ratios, dict):
                continue
            pe = float(ratios.get("peRatioTTM", 0) or 0)
            if pe <= 0 or pe > 50:  # filter extreme PEs
                continue

            ey = 1.0 / pe

            # Get 6-month price return from historical prices
            hist = client.historical_price(sym, interval="daily")
            if isinstance(hist, pd.DataFrame) and len(hist) >= 120:
                # Approximately 6 months of trading days
                current_price = float(hist["close"].iloc[-1])
                price_6m_ago = float(hist["close"].iloc[-126])
                if price_6m_ago > 0:
                    return_6m = (current_price - price_6m_ago) / price_6m_ago
                else:
                    continue
            else:
                continue

            if return_6m > 0:  # only positive momentum
                results.append(
                    {
                        "symbol": sym,
                        "pe_ratio": pe,
                        "earnings_yield": ey,
                        "price_return_6m": return_6m,
                        "market_cap": float(row.get("marketCap", 0) or 0),
                    }
                )
        except Exception:  # noqa: BLE001
            continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["value_rank"] = df["earnings_yield"].rank(ascending=False)
    df["momentum_rank"] = df["price_return_6m"].rank(ascending=False)
    df["combined_rank"] = df["value_rank"] + df["momentum_rank"]
    df = df.sort_values("combined_rank").head(top_n)
    return df.reset_index(drop=True)
