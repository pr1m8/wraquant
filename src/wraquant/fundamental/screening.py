"""Stock screening based on fundamental criteria.

Provides pre-built screens for common investment styles (value, growth,
quality) and classic quantitative strategies (Piotroski, Magic Formula),
plus a flexible custom screening framework.

Each screen accepts threshold parameters and returns a list of matching
stocks with their metrics.  Screens call the FMP data provider for
financial data and profile information.

Screen philosophy:
- **Value screens** find cheap stocks (low P/E, high yield).  Risk:
  value traps -- cheap for a reason.  Pair with quality screens.
- **Growth screens** find companies with accelerating fundamentals.
  Risk: overpaying for growth (high PEG).
- **Quality screens** find companies with strong, sustainable economics.
  Risk: usually expensive -- combine with valuation for timing.

Example:
    >>> from wraquant.fundamental.screening import magic_formula_screen
    >>> mf = magic_formula_screen(top_n=20)
    >>> print(mf[['symbol', 'earnings_yield', 'roic', 'magic_rank']])

References:
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

from wraquant.core.decorators import requires_extra

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide safely, returning *default* on zero/near-zero denominator."""
    if abs(denominator) < 1e-12:
        return default
    return float(numerator / denominator)


def _get_fmp_client(fmp_client: Any | None = None) -> Any:
    """Return the provided client or construct a default ``FMPProvider``."""
    if fmp_client is not None:
        return fmp_client
    from wraquant.data.providers.fmp import FMPProvider  # noqa: WPS433

    return FMPProvider()


def _safe_get(data: dict | list, key: str, default: float = 0.0) -> float:
    """Extract a numeric value from an FMP response dict/list."""
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


def _safe_get_str(data: dict | list, key: str, default: str = "") -> str:
    """Extract a string value from an FMP response dict/list."""
    if isinstance(data, list):
        if not data:
            return default
        data = data[0]
    val = data.get(key)
    return str(val) if val is not None else default


def _safe_get_list(data: Any) -> list[dict]:
    """Coerce *data* to a list of dicts."""
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    return []


def _screen_universe(
    symbols: list[str] | None,
    fmp_client: Any,
) -> list[str]:
    """Resolve the screening universe.

    If *symbols* is provided, use it directly.  Otherwise, attempt to
    fetch a broad universe from FMP (e.g., active stocks list).
    """
    if symbols is not None:
        return symbols
    try:
        stock_list = fmp_client.stock_list()
        if isinstance(stock_list, list):
            return [
                s.get("symbol", "")
                for s in stock_list[:500]
                if s.get("symbol") and s.get("type") in ("stock", "Stock", None)
            ]
    except Exception:  # noqa: BLE001
        logger.warning("Could not fetch stock universe from FMP")
    return []


def _fetch_stock_metrics(
    symbol: str,
    fmp_client: Any,
) -> dict[str, Any] | None:
    """Fetch key metrics and ratios for a single stock.

    Returns ``None`` if data cannot be retrieved.
    """
    try:
        ratios = fmp_client.ratios_ttm(symbol)
        metrics = fmp_client.key_metrics(symbol)
        profile = fmp_client.company_profile(symbol)

        profile_data = profile[0] if isinstance(profile, list) and profile else profile
        if not isinstance(profile_data, dict):
            return None

        pe = _safe_get(ratios, "peRatioTTM")
        return {
            "symbol": symbol,
            "company_name": profile_data.get("companyName", ""),
            "sector": profile_data.get("sector", ""),
            "industry": profile_data.get("industry", ""),
            "market_cap": _safe_get(profile_data, "mktCap"),
            "price": _safe_get(profile_data, "price"),
            # Valuation
            "pe_ratio": pe,
            "pb_ratio": _safe_get(ratios, "priceToBookRatioTTM"),
            "ps_ratio": _safe_get(ratios, "priceToSalesRatioTTM"),
            "peg_ratio": _safe_get(ratios, "pegRatioTTM"),
            "ev_to_ebitda": _safe_get(metrics, "enterpriseValueOverEBITDATTM"),
            "price_to_fcf": _safe_get(ratios, "priceToFreeCashFlowsRatioTTM"),
            "dividend_yield": _safe_get(ratios, "dividendYieldTTM"),
            "earnings_yield": _safe_div(1.0, pe) if pe > 0 else 0.0,
            # Profitability
            "roe": _safe_get(ratios, "returnOnEquityTTM"),
            "roa": _safe_get(ratios, "returnOnAssetsTTM"),
            "roic": _safe_get(ratios, "returnOnCapitalEmployedTTM"),
            "gross_margin": _safe_get(ratios, "grossProfitMarginTTM"),
            "operating_margin": _safe_get(ratios, "operatingProfitMarginTTM"),
            "net_margin": _safe_get(ratios, "netProfitMarginTTM"),
            # Leverage
            "debt_to_equity": _safe_get(ratios, "debtEquityRatioTTM"),
            "current_ratio": _safe_get(ratios, "currentRatioTTM"),
            "interest_coverage": _safe_get(ratios, "interestCoverageTTM"),
            # Growth
            "revenue_growth": _safe_get(metrics, "revenueGrowth"),
            "eps_growth": _safe_get(metrics, "epsgrowth"),
            # Per share
            "eps": _safe_get(metrics, "netIncomePerShare"),
            "bvps": _safe_get(metrics, "bookValuePerShare"),
            "fcf_per_share": _safe_get(metrics, "freeCashFlowPerShare"),
        }
    except Exception:  # noqa: BLE001
        logger.debug("Failed to fetch metrics for %s", symbol)
        return None


# ---------------------------------------------------------------------------
# Value Screen
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def value_screen(
    *,
    symbols: list[str] | None = None,
    min_pe: float | None = None,
    max_pe: float = 20.0,
    min_dividend_yield: float = 0.0,
    max_pb: float | None = None,
    max_ps: float | None = None,
    min_earnings_yield: float = 0.0,
    min_market_cap: float = 0.0,
    fmp_client: Any | None = None,
) -> list[dict[str, Any]]:
    """Screen for value stocks based on valuation criteria.

    Value investing seeks stocks trading below their intrinsic worth.
    This screen identifies candidates with low valuation multiples
    and/or high yields -- the quantitative implementation of Graham
    and Dodd's margin-of-safety concept.

    When to use:
        - Build a value factor portfolio.
        - Find candidates for deeper fundamental analysis.
        - Pair with :func:`quality_screen` to avoid value traps.

    Parameters:
        symbols: List of ticker symbols to screen.  If ``None``, uses
            a default universe (FMP stock list).
        min_pe: Minimum P/E ratio.  ``None`` means no lower bound.
            Set > 0 to exclude negative-earnings companies.
        max_pe: Maximum P/E ratio.  Lower values = stricter value filter.
            Classic value: 15.  Deep value: 10.
        min_dividend_yield: Minimum dividend yield (decimal).
            0.02 = 2% yield.  Set to 0 for non-dividend stocks.
        max_pb: Maximum price-to-book ratio.  ``None`` = no filter.
            Classic Graham: 1.5.
        max_ps: Maximum price-to-sales ratio.  ``None`` = no filter.
        min_earnings_yield: Minimum earnings yield (1/PE, decimal).
        min_market_cap: Minimum market cap in dollars.  Filter out
            micro-caps (e.g., 1e9 for >$1B).
        fmp_client: Optional ``FMPProvider`` instance.

    Returns:
        List of dicts, each containing the stock's metrics and
        screening results.  Sorted by P/E ascending (cheapest first).

    Example:
        >>> from wraquant.fundamental.screening import value_screen
        >>> hits = value_screen(
        ...     symbols=["AAPL", "MSFT", "BRK-B", "JNJ", "XOM"],
        ...     max_pe=18, min_dividend_yield=0.02,
        ... )
        >>> for h in hits:
        ...     print(f"{h['symbol']}: P/E={h['pe_ratio']:.1f}, yield={h['dividend_yield']:.2%}")

    References:
        Graham, B. & Dodd, D. (1934). *Security Analysis*. McGraw-Hill.

    See Also:
        growth_screen: Growth-oriented screening.
        quality_screen: Profitability and leverage screening.
    """
    client = _get_fmp_client(fmp_client)
    universe = _screen_universe(symbols, client)

    results: list[dict[str, Any]] = []
    for sym in universe:
        data = _fetch_stock_metrics(sym, client)
        if data is None:
            continue

        pe = data["pe_ratio"]
        dy = data["dividend_yield"]
        pb = data["pb_ratio"]
        ps = data["ps_ratio"]
        ey = data["earnings_yield"]
        mc = data["market_cap"]

        # Apply filters
        if pe <= 0:
            continue  # skip negative earnings
        if min_pe is not None and pe < min_pe:
            continue
        if pe > max_pe:
            continue
        if dy < min_dividend_yield:
            continue
        if max_pb is not None and pb > max_pb:
            continue
        if max_ps is not None and ps > max_ps:
            continue
        if ey < min_earnings_yield:
            continue
        if mc < min_market_cap:
            continue

        data["screen"] = "value"
        results.append(data)

    # Sort by P/E ascending (cheapest first)
    results.sort(key=lambda x: x.get("pe_ratio", float("inf")))
    return results


# ---------------------------------------------------------------------------
# Growth Screen
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def growth_screen(
    *,
    symbols: list[str] | None = None,
    min_revenue_growth: float = 0.15,
    min_eps_growth: float = 0.10,
    min_gross_margin: float = 0.0,
    max_peg: float | None = None,
    min_market_cap: float = 0.0,
    fmp_client: Any | None = None,
) -> list[dict[str, Any]]:
    """Screen for growth stocks based on revenue and earnings momentum.

    Growth investing seeks companies with above-average fundamental
    growth.  This screen identifies companies with strong revenue
    acceleration, earnings expansion, and (optionally) reasonable
    valuation relative to that growth (PEG filter).

    When to use:
        - Build a momentum/growth factor portfolio.
        - Find high-growth candidates before they become expensive.
        - Combine with PEG filter to avoid overpaying for growth.

    Parameters:
        symbols: Ticker symbols to screen.  ``None`` = default universe.
        min_revenue_growth: Minimum YoY revenue growth (decimal).
            0.15 = 15%.  Growth stocks typically grow > 15%.
        min_eps_growth: Minimum YoY EPS growth (decimal).
        min_gross_margin: Minimum gross margin.  Filters out low-margin
            growers.  Set > 0.40 for software/tech.
        max_peg: Maximum PEG ratio.  ``None`` = no PEG filter.
            < 1.0 = growth at a reasonable price (GARP).
        min_market_cap: Minimum market cap.
        fmp_client: Optional ``FMPProvider`` instance.

    Returns:
        List of dicts sorted by revenue growth descending.

    Example:
        >>> from wraquant.fundamental.screening import growth_screen
        >>> growers = growth_screen(
        ...     symbols=["NVDA", "MSFT", "AAPL", "TSLA"],
        ...     min_revenue_growth=0.10, min_eps_growth=0.05,
        ... )
        >>> for g in growers:
        ...     print(f"{g['symbol']}: rev growth={g['revenue_growth']:.1%}")

    See Also:
        value_screen: Low-valuation approach.
        quality_screen: High-profitability approach.
    """
    client = _get_fmp_client(fmp_client)
    universe = _screen_universe(symbols, client)

    results: list[dict[str, Any]] = []
    for sym in universe:
        data = _fetch_stock_metrics(sym, client)
        if data is None:
            continue

        rg = data["revenue_growth"]
        eg = data["eps_growth"]
        gm = data["gross_margin"]
        peg = data["peg_ratio"]
        mc = data["market_cap"]

        if rg < min_revenue_growth:
            continue
        if eg < min_eps_growth:
            continue
        if gm < min_gross_margin:
            continue
        if max_peg is not None and peg > 0 and peg > max_peg:
            continue
        if mc < min_market_cap:
            continue

        data["screen"] = "growth"
        results.append(data)

    results.sort(key=lambda x: x.get("revenue_growth", 0.0), reverse=True)
    return results


# ---------------------------------------------------------------------------
# Quality Screen
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def quality_screen(
    *,
    symbols: list[str] | None = None,
    min_roe: float = 0.15,
    max_de: float = 1.0,
    min_current_ratio: float = 1.5,
    min_operating_margin: float = 0.0,
    min_roic: float = 0.0,
    min_market_cap: float = 0.0,
    fmp_client: Any | None = None,
) -> list[dict[str, Any]]:
    """Screen for high-quality companies.

    Quality investing targets companies with durable competitive
    advantages: high returns on capital, low leverage, and strong
    operating margins.  The quality factor (QMJ -- quality minus junk)
    is well-documented in academic literature.

    When to use:
        - Build a quality factor portfolio (Fama-French RMW factor).
        - Find companies with moats and sustainable economics.
        - Pair with value screening to find quality at a discount.

    Parameters:
        symbols: Ticker symbols to screen.
        min_roe: Minimum return on equity.  > 0.15 filters for
            capital-efficient businesses.
        max_de: Maximum debt-to-equity ratio.  < 1.0 favours low
            leverage.  Set higher for capital-intensive sectors.
        min_current_ratio: Minimum current ratio.  > 1.5 ensures
            short-term solvency.
        min_operating_margin: Minimum operating margin.
        min_roic: Minimum return on invested capital.  > WACC means
            the company creates economic value.
        min_market_cap: Minimum market cap.
        fmp_client: Optional ``FMPProvider`` instance.

    Returns:
        List of dicts sorted by ROE descending.

    Example:
        >>> from wraquant.fundamental.screening import quality_screen
        >>> quality = quality_screen(
        ...     symbols=["MSFT", "AAPL", "GOOG", "META"],
        ...     min_roe=0.20, max_de=0.8,
        ... )
        >>> for q in quality:
        ...     print(f"{q['symbol']}: ROE={q['roe']:.1%}, D/E={q['debt_to_equity']:.2f}")

    References:
        Asness, C. S., Frazzini, A. & Pedersen, L. H. (2019). "Quality
        Minus Junk." *Review of Accounting Studies*, 24(1), 34--112.

    See Also:
        value_screen: Find cheap stocks.
        piotroski_screen: Binary financial health filter.
    """
    client = _get_fmp_client(fmp_client)
    universe = _screen_universe(symbols, client)

    results: list[dict[str, Any]] = []
    for sym in universe:
        data = _fetch_stock_metrics(sym, client)
        if data is None:
            continue

        roe = data["roe"]
        de = data["debt_to_equity"]
        cr = data["current_ratio"]
        om = data["operating_margin"]
        roic = data["roic"]
        mc = data["market_cap"]

        if roe < min_roe:
            continue
        if de > max_de:
            continue
        if cr < min_current_ratio and cr > 0:
            continue
        if om < min_operating_margin:
            continue
        if roic < min_roic:
            continue
        if mc < min_market_cap:
            continue

        data["screen"] = "quality"
        results.append(data)

    results.sort(key=lambda x: x.get("roe", 0.0), reverse=True)
    return results


# ---------------------------------------------------------------------------
# Piotroski Screen
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def piotroski_screen(
    *,
    symbols: list[str] | None = None,
    min_score: int = 7,
    fmp_client: Any | None = None,
) -> list[dict[str, Any]]:
    """Screen stocks using the Piotroski F-Score.

    The Piotroski F-Score (0--9) evaluates financial health across
    profitability, leverage, and efficiency.  Historically, high-score
    value stocks (F >= 8, low P/B) outperform low-score value stocks
    by ~7.5% annually.

    When to use:
        - Filter a value portfolio for financial strength.
        - Avoid value traps.
        - Long/short strategy: long high F-Score, short low F-Score.

    Parameters:
        symbols: Ticker symbols.
        min_score: Minimum F-Score to pass (0--9).  7+ captures the
            top tier.  Use 8+ for the most selective filter.
        fmp_client: Optional ``FMPProvider`` instance.

    Returns:
        List of dicts including ``f_score`` and component scores,
        sorted by F-Score descending.

    Example:
        >>> from wraquant.fundamental.screening import piotroski_screen
        >>> strong = piotroski_screen(
        ...     symbols=["AAPL", "MSFT", "F", "GE"],
        ...     min_score=7,
        ... )
        >>> for s in strong:
        ...     print(f"{s['symbol']}: F-Score={s['f_score']}")

    References:
        Piotroski, J. D. (2000). "Value Investing: The Use of Historical
        Financial Statement Information to Separate Winners from Losers."
        *Journal of Accounting Research*, 38, 1--41.

    See Also:
        quality_screen: Continuous quality ranking.
        magic_formula_screen: Greenblatt's ROIC + earnings yield.
    """
    client = _get_fmp_client(fmp_client)
    universe = _screen_universe(symbols, client)

    results: list[dict[str, Any]] = []
    for sym in universe:
        try:
            income_data = _safe_get_list(
                client.income_statement(sym, period="annual", limit=2)
            )
            balance_data = _safe_get_list(
                client.balance_sheet(sym, period="annual", limit=2)
            )
            cf_data = _safe_get_list(client.cash_flow(sym, period="annual", limit=2))

            if len(income_data) < 2 or len(balance_data) < 2 or len(cf_data) < 1:
                continue

            curr_inc, prev_inc = income_data[0], income_data[1]
            curr_bal, prev_bal = balance_data[0], balance_data[1]
            curr_cf = cf_data[0]

            # Extract values
            ni = _safe_get(curr_inc, "netIncome")
            prev_ni = _safe_get(prev_inc, "netIncome")
            ocf = _safe_get(curr_cf, "operatingCashFlow")
            ta = _safe_get(curr_bal, "totalAssets")
            prev_ta = _safe_get(prev_bal, "totalAssets")
            ltd = _safe_get(curr_bal, "longTermDebt")
            prev_ltd = _safe_get(prev_bal, "longTermDebt")
            ca = _safe_get(curr_bal, "totalCurrentAssets")
            cl = _safe_get(curr_bal, "totalCurrentLiabilities")
            prev_ca = _safe_get(prev_bal, "totalCurrentAssets")
            prev_cl = _safe_get(prev_bal, "totalCurrentLiabilities")
            shares = _safe_get(curr_bal, "commonStock", default=1.0)
            prev_shares = _safe_get(prev_bal, "commonStock", default=1.0)
            revenue = _safe_get(curr_inc, "revenue")
            prev_revenue = _safe_get(prev_inc, "revenue")
            gp = _safe_get(curr_inc, "grossProfit")
            prev_gp = _safe_get(prev_inc, "grossProfit")

            # Compute ratios
            roa = _safe_div(ni, ta)
            prev_roa = _safe_div(prev_ni, prev_ta)
            curr_ratio = _safe_div(ca, cl)
            prev_curr_ratio = _safe_div(prev_ca, prev_cl)
            gm = _safe_div(gp, revenue)
            prev_gm = _safe_div(prev_gp, prev_revenue)
            at = _safe_div(revenue, ta)
            prev_at = _safe_div(prev_revenue, prev_ta)

            # Score components
            components: dict[str, bool] = {}

            # Profitability
            components["positive_roa"] = roa > 0
            components["positive_ocf"] = ocf > 0
            components["roa_increasing"] = roa > prev_roa
            components["ocf_gt_ni"] = ocf > ni

            # Leverage
            components["debt_decreasing"] = ltd < prev_ltd
            components["current_ratio_increasing"] = curr_ratio > prev_curr_ratio
            components["no_dilution"] = shares <= prev_shares

            # Efficiency
            components["margin_increasing"] = gm > prev_gm
            components["turnover_increasing"] = at > prev_at

            score = sum(1 for v in components.values() if v)

            if score < min_score:
                continue

            base = _fetch_stock_metrics(sym, client)
            if base is None:
                base = {"symbol": sym}

            base["f_score"] = score
            base["f_score_components"] = components
            base["screen"] = "piotroski"
            results.append(base)

        except Exception:  # noqa: BLE001
            logger.debug("Piotroski screen failed for %s", sym)

    results.sort(key=lambda x: x.get("f_score", 0), reverse=True)
    return results


# ---------------------------------------------------------------------------
# Magic Formula Screen
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def magic_formula_screen(
    *,
    symbols: list[str] | None = None,
    top_n: int = 30,
    min_market_cap: float = 1e8,
    fmp_client: Any | None = None,
) -> list[dict[str, Any]]:
    """Screen using Joel Greenblatt's Magic Formula.

    The Magic Formula ranks stocks by two factors and combines ranks:
    1. **Earnings yield** (EBIT / Enterprise Value) -- cheapness.
    2. **Return on capital** (EBIT / (Net Working Capital + Net Fixed
       Assets)) -- quality.

    Stocks that rank well on *both* dimensions are selected.  Greenblatt
    showed this simple formula outperformed the S&P 500 by ~10%/year
    over 17 years.

    When to use:
        - Simple, systematic value+quality portfolio.
        - As a starting universe for deeper analysis.
        - Educational: demonstrates the power of combining value and
          quality factors.

    Mathematical formulation:
        Earnings Yield = EBIT / Enterprise Value
        ROIC = EBIT / (Net Working Capital + Net Fixed Assets)
        Magic Rank = EY_rank + ROIC_rank  (lower combined rank is better)

    Parameters:
        symbols: Ticker symbols.
        top_n: Number of top-ranked stocks to return.
        min_market_cap: Minimum market cap filter.  Greenblatt
            recommended > $50M to ensure investability.
        fmp_client: Optional ``FMPProvider`` instance.

    Returns:
        List of top *top_n* dicts sorted by Magic Formula combined rank,
        including ``earnings_yield``, ``roic``, ``ey_rank``,
        ``roic_rank``, and ``magic_rank``.

    Example:
        >>> from wraquant.fundamental.screening import magic_formula_screen
        >>> mf = magic_formula_screen(
        ...     symbols=["AAPL", "MSFT", "GOOG", "META", "AMZN"],
        ...     top_n=3,
        ... )
        >>> for m in mf:
        ...     print(f"{m['symbol']}: rank={m['magic_rank']}, "
        ...           f"EY={m['earnings_yield']:.1%}, ROIC={m['roic']:.1%}")

    References:
        Greenblatt, J. (2006). *The Little Book That Beats the Market*.
        Wiley.

    See Also:
        piotroski_screen: Binary financial health filter.
        value_screen: Pure valuation-based screening.
    """
    client = _get_fmp_client(fmp_client)
    universe = _screen_universe(symbols, client)

    candidates: list[dict[str, Any]] = []
    for sym in universe:
        try:
            income = client.income_statement(sym)
            balance = client.balance_sheet(sym)
            ev_data = client.enterprise_value(sym)
            profile = client.company_profile(sym)

            profile_data = (
                profile[0] if isinstance(profile, list) and profile else profile
            )
            mc = (
                _safe_get(profile_data, "mktCap")
                if isinstance(profile_data, dict)
                else 0.0
            )

            if mc < min_market_cap:
                continue

            ebit = _safe_get(income, "operatingIncome")
            if ebit <= 0:
                continue  # skip unprofitable

            ev = _safe_get(ev_data, "enterpriseValue")
            if ev <= 0:
                continue

            # Net working capital + net fixed assets
            current_assets = _safe_get(balance, "totalCurrentAssets")
            current_liabilities = _safe_get(balance, "totalCurrentLiabilities")
            ppe = _safe_get(balance, "propertyPlantEquipmentNet")
            nwc = current_assets - current_liabilities
            invested = nwc + ppe
            if invested <= 0:
                invested = _safe_get(balance, "totalAssets")  # fallback

            ey = _safe_div(ebit, ev)
            roic = _safe_div(ebit, invested)

            base = _fetch_stock_metrics(sym, client) or {"symbol": sym}
            base["earnings_yield"] = ey
            base["roic"] = roic
            candidates.append(base)

        except Exception:  # noqa: BLE001
            logger.debug("Magic formula failed for %s", sym)

    if not candidates:
        return []

    # Rank by earnings yield (descending = rank 1 is highest EY)
    candidates.sort(key=lambda x: x.get("earnings_yield", 0.0), reverse=True)
    for i, c in enumerate(candidates):
        c["ey_rank"] = i + 1

    # Rank by ROIC (descending = rank 1 is highest ROIC)
    candidates.sort(key=lambda x: x.get("roic", 0.0), reverse=True)
    for i, c in enumerate(candidates):
        c["roic_rank"] = i + 1

    # Combined rank (lower is better)
    for c in candidates:
        c["magic_rank"] = c["ey_rank"] + c["roic_rank"]

    candidates.sort(key=lambda x: x.get("magic_rank", float("inf")))

    for c in candidates[:top_n]:
        c["screen"] = "magic_formula"

    return candidates[:top_n]


# ---------------------------------------------------------------------------
# Custom Screen
# ---------------------------------------------------------------------------


@requires_extra("market-data")
def custom_screen(
    criteria: dict[str, Any],
    *,
    symbols: list[str] | None = None,
    sort_by: str | None = None,
    sort_ascending: bool = False,
    top_n: int | None = None,
    fmp_client: Any | None = None,
) -> list[dict[str, Any]]:
    """Screen stocks with arbitrary criteria.

    Flexible screening engine that accepts a dictionary of filter
    conditions.  Each key is a metric name (matching the keys returned
    by the internal ``_fetch_stock_metrics`` helper); each value is
    a tuple ``(operator, threshold)`` or a simple threshold (treated
    as minimum).

    When to use:
        - Build custom factor screens beyond the pre-built templates.
        - Research new factor combinations.
        - Backtest screening strategies with varying thresholds.

    Parameters:
        criteria: Dictionary of screening criteria.  Values can be:

            - ``float`` -- treated as minimum (e.g., ``{"roe": 0.15}``
              means ROE >= 15%).
            - ``tuple(str, float)`` -- operator and threshold.
              Operators: ``">"``, ``">="``, ``"<"``, ``"<="``,
              ``"=="``, ``"!="``.

            Available metric keys:
            ``pe_ratio``, ``pb_ratio``, ``ps_ratio``, ``peg_ratio``,
            ``ev_to_ebitda``, ``dividend_yield``, ``earnings_yield``,
            ``roe``, ``roa``, ``roic``, ``gross_margin``,
            ``operating_margin``, ``net_margin``, ``debt_to_equity``,
            ``current_ratio``, ``interest_coverage``,
            ``revenue_growth``, ``eps_growth``, ``market_cap``.

        symbols: Ticker symbols to screen.
        sort_by: Metric key to sort results by.
        sort_ascending: Sort direction.
        top_n: Maximum results to return.
        fmp_client: Optional ``FMPProvider`` instance.

    Returns:
        List of matching stock dicts.

    Example:
        >>> from wraquant.fundamental.screening import custom_screen
        >>> hits = custom_screen(
        ...     criteria={
        ...         "roe": (">=", 0.20),
        ...         "debt_to_equity": ("<", 0.5),
        ...         "pe_ratio": ("<", 25),
        ...         "market_cap": (">=", 1e10),
        ...     },
        ...     symbols=["AAPL", "MSFT", "GOOG", "META", "AMZN"],
        ...     sort_by="roe",
        ... )
        >>> for h in hits:
        ...     print(f"{h['symbol']}: ROE={h['roe']:.1%}")

    See Also:
        value_screen: Pre-built value filter.
        growth_screen: Pre-built growth filter.
        quality_screen: Pre-built quality filter.
    """
    client = _get_fmp_client(fmp_client)
    universe = _screen_universe(symbols, client)

    def _check_criterion(value: float, condition: Any) -> bool:
        """Evaluate a single criterion against a value."""
        if isinstance(condition, (int, float)):
            return value >= condition
        if isinstance(condition, tuple) and len(condition) == 2:
            op, threshold = condition
            if op in (">", "gt"):
                return value > threshold
            if op in (">=", "gte", "ge"):
                return value >= threshold
            if op in ("<", "lt"):
                return value < threshold
            if op in ("<=", "lte", "le"):
                return value <= threshold
            if op in ("==", "eq"):
                return abs(value - threshold) < 1e-12
            if op in ("!=", "ne"):
                return abs(value - threshold) >= 1e-12
        return True

    results: list[dict[str, Any]] = []
    for sym in universe:
        data = _fetch_stock_metrics(sym, client)
        if data is None:
            continue

        passes = True
        for key, condition in criteria.items():
            val = data.get(key)
            if val is None:
                passes = False
                break
            if not _check_criterion(float(val), condition):
                passes = False
                break

        if passes:
            data["screen"] = "custom"
            results.append(data)

    if sort_by:
        results.sort(
            key=lambda x: x.get(sort_by, 0.0),
            reverse=not sort_ascending,
        )

    if top_n is not None:
        results = results[:top_n]

    return results
