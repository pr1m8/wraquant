"""Financial Modeling Prep (FMP) data provider and REST client.

Provides comprehensive access to FMP's 550+ endpoints covering price data,
financial statements, fundamental metrics, valuation models, earnings,
SEC filings, institutional ownership, news, and economic indicators.

All endpoints use the ``https://financialmodelingprep.com/stable/`` base URL.
Authentication is via ``apikey`` query parameter.  Set the ``FMP_API_KEY``
environment variable or pass the key directly to :class:`FMPClient`.

Requires the ``market-data`` extra group (``pdm install -G market-data``).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import pandas as pd

from wraquant.core.decorators import requires_extra
from wraquant.core.types import DateLike
from wraquant.data.utils import clean_symbol, parse_date

logger = logging.getLogger(__name__)

_BASE_URL = "https://financialmodelingprep.com"

# --------------------------------------------------------------------------
# Interval mapping for intraday / daily historical prices
# --------------------------------------------------------------------------

_INTERVAL_MAP: dict[str, str] = {
    "1min": "/stable/historical-chart/1min",
    "5min": "/stable/historical-chart/5min",
    "15min": "/stable/historical-chart/15min",
    "30min": "/stable/historical-chart/30min",
    "1hour": "/stable/historical-chart/1hour",
    "4hour": "/stable/historical-chart/4hour",
    "daily": "/stable/historical-price-eod/full",
}


class FMPClient:
    """Comprehensive REST client for the Financial Modeling Prep API.

    Wraps FMP's stable API surface with typed Python methods that return
    ``pd.DataFrame`` or ``dict`` results.  Every HTTP call goes through
    :meth:`_get`, which handles authentication, rate-limit back-off, and
    error handling.

    The client is intentionally *not* a :class:`DataProvider` subclass:
    ``DataProvider`` is a thin fetch-prices/fetch-ohlcv abstraction,
    whereas ``FMPClient`` exposes the full breadth of FMP's fundamental,
    valuation, earnings, SEC, news, and economic data.  If you only need
    OHLCV data for the backtest/TA pipeline, use :class:`FMPProvider`
    (registered automatically when *httpx* is available).

    Parameters:
        api_key: FMP API key.  Falls back to the ``FMP_API_KEY``
            environment variable when *None*.
        timeout: HTTP request timeout in seconds.  Defaults to 30.
        max_retries: Number of retries on transient failures (429 / 5xx).
            Defaults to 3.

    Raises:
        ValueError: If no API key is provided and ``FMP_API_KEY`` is unset.

    Example:
        >>> from wraquant.data.providers.fmp import FMPClient
        >>> fmp = FMPClient()
        >>> profile = fmp.company_profile("AAPL")
        >>> print(profile["companyName"])
        Apple Inc.
        >>> df = fmp.income_statement("AAPL", period="annual", limit=5)
        >>> print(df.columns.tolist()[:5])
    """

    @requires_extra("market-data")
    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key or os.environ.get("FMP_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "FMP API key required.  Pass api_key= or set FMP_API_KEY env var."
            )
        self._timeout = timeout
        self._max_retries = max_retries
        self._last_request_ts: float = 0.0
        # Minimum interval between requests (seconds) to respect rate limits
        self._min_interval: float = 0.12  # ~8 req/s

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict | list:
        """Execute an authenticated GET request against the FMP API.

        Handles rate-limit throttling (HTTP 429) with exponential back-off,
        retries on transient server errors (5xx), and raises on client
        errors (4xx).

        Parameters:
            path: API path **including** the ``/stable/`` prefix,
                e.g. ``"/stable/income-statement"``.
            params: Query parameters (``symbol``, ``period``, etc.).
                The ``apikey`` parameter is injected automatically.

        Returns:
            Parsed JSON response — typically a ``list[dict]`` or ``dict``.

        Raises:
            httpx.HTTPStatusError: On non-retryable HTTP errors.
            httpx.TimeoutException: If all retries are exhausted.
        """
        import httpx

        params = dict(params or {})
        params["apikey"] = self._api_key

        url = f"{_BASE_URL}{path}"

        for attempt in range(1, self._max_retries + 1):
            # Simple rate-limit throttle
            elapsed = time.monotonic() - self._last_request_ts
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)

            try:
                self._last_request_ts = time.monotonic()
                resp = httpx.get(url, params=params, timeout=self._timeout)

                if resp.status_code == 429:
                    wait = 2**attempt
                    logger.warning(
                        "FMP rate-limited (429), retrying in %ds (attempt %d/%d)",
                        wait,
                        attempt,
                        self._max_retries,
                    )
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json()  # type: ignore[no-any-return]

            except httpx.HTTPStatusError:
                if resp.status_code >= 500 and attempt < self._max_retries:
                    wait = 2**attempt
                    logger.warning(
                        "FMP server error (%d), retrying in %ds (attempt %d/%d)",
                        resp.status_code,
                        wait,
                        attempt,
                        self._max_retries,
                    )
                    time.sleep(wait)
                    continue
                raise

            except httpx.TimeoutException:
                if attempt < self._max_retries:
                    logger.warning(
                        "FMP request timed out, retrying (attempt %d/%d)",
                        attempt,
                        self._max_retries,
                    )
                    continue
                raise

        # Should not reach here, but satisfy the type checker
        raise RuntimeError("FMP request failed after all retries")  # pragma: no cover

    @staticmethod
    def _fmt_date(d: DateLike | None) -> str | None:
        """Convert a DateLike to ``YYYY-MM-DD`` string for FMP query params."""
        if d is None:
            return None
        ts = parse_date(d)
        return ts.strftime("%Y-%m-%d") if ts is not None else None

    # ==================================================================
    # 1. Price data
    # ==================================================================

    def historical_price(
        self,
        symbol: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
        interval: str = "daily",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV price data.

        Retrieves open, high, low, close, and volume data at the requested
        resolution.  For daily data the dividend-adjusted ``/full`` endpoint
        is used; for intraday the corresponding chart endpoint is selected.

        Parameters:
            symbol: Ticker symbol (e.g. ``"AAPL"``).
            start: Inclusive start date.  Omit for all available history.
            end: Inclusive end date.  Omit for data through today.
            interval: Bar resolution.  One of ``"1min"``, ``"5min"``,
                ``"15min"``, ``"30min"``, ``"1hour"``, ``"4hour"``,
                ``"daily"`` (default).

        Returns:
            DataFrame with columns ``[date, open, high, low, close, volume]``
            and any additional fields the endpoint provides (e.g. *vwap*,
            *changePercent*).  Rows are sorted ascending by date.

        Raises:
            ValueError: If *interval* is not a recognised value.

        Example:
            >>> df = fmp.historical_price("AAPL", start="2024-01-01",
            ...                           end="2024-06-30")
            >>> print(df.shape)
        """
        path = _INTERVAL_MAP.get(interval)
        if path is None:
            raise ValueError(
                f"Unsupported interval '{interval}'.  "
                f"Choose from: {sorted(_INTERVAL_MAP)}"
            )

        params: dict[str, Any] = {"symbol": clean_symbol(symbol)}
        from_date = self._fmt_date(start)
        to_date = self._fmt_date(end)
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        data = self._get(path, params)
        df = pd.DataFrame(data)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df

    def quote(self, symbol: str) -> dict[str, Any]:
        """Fetch a real-time quote for a single symbol.

        Returns the latest bid/ask, last price, volume, market cap,
        52-week high/low, and other quote fields.

        Parameters:
            symbol: Ticker symbol (e.g. ``"AAPL"``).

        Returns:
            Dictionary of quote fields.  Key fields include *price*,
            *changesPercentage*, *volume*, *marketCap*, *yearHigh*,
            *yearLow*, *eps*, and *pe*.

        Example:
            >>> q = fmp.quote("AAPL")
            >>> print(f"${q['price']:.2f}  ({q['changesPercentage']:+.2f}%)")
        """
        data = self._get("/stable/quote", {"symbol": clean_symbol(symbol)})
        if isinstance(data, list) and len(data) > 0:
            return data[0]  # type: ignore[return-value]
        return data if isinstance(data, dict) else {}

    def batch_quote(self, symbols: list[str]) -> pd.DataFrame:
        """Fetch real-time quotes for multiple symbols in a single call.

        Parameters:
            symbols: List of ticker symbols (e.g. ``["AAPL", "MSFT"]``).

        Returns:
            DataFrame with one row per symbol and quote fields as columns.

        Example:
            >>> df = fmp.batch_quote(["AAPL", "MSFT", "GOOG"])
            >>> print(df[["symbol", "price", "marketCap"]])
        """
        joined = ",".join(clean_symbol(s) for s in symbols)
        data = self._get("/stable/batch-quote", {"symbols": joined})
        return pd.DataFrame(data)

    # ==================================================================
    # 2. Financial statements
    # ==================================================================

    def income_statement(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> pd.DataFrame:
        """Fetch income statement data.

        Returns line items such as revenue, gross profit, operating income,
        net income, EPS, and EBITDA for the requested periods.

        Parameters:
            symbol: Ticker symbol.
            period: ``"annual"`` or ``"quarter"``.
            limit: Maximum number of periods to return (most recent first).

        Returns:
            DataFrame with one row per reporting period and financial line
            items as columns.

        Example:
            >>> df = fmp.income_statement("AAPL", period="annual", limit=5)
            >>> print(df[["date", "revenue", "netIncome"]].head())
        """
        data = self._get(
            "/stable/income-statement",
            {"symbol": clean_symbol(symbol), "period": period, "limit": limit},
        )
        return pd.DataFrame(data)

    def balance_sheet(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> pd.DataFrame:
        """Fetch balance sheet statement data.

        Returns assets, liabilities, and shareholders' equity line items
        including total assets, total debt, cash, and retained earnings.

        Parameters:
            symbol: Ticker symbol.
            period: ``"annual"`` or ``"quarter"``.
            limit: Maximum number of periods to return.

        Returns:
            DataFrame with one row per reporting period.

        Example:
            >>> df = fmp.balance_sheet("AAPL", limit=5)
            >>> print(df[["date", "totalAssets", "totalDebt"]].head())
        """
        data = self._get(
            "/stable/balance-sheet-statement",
            {"symbol": clean_symbol(symbol), "period": period, "limit": limit},
        )
        return pd.DataFrame(data)

    def cash_flow(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> pd.DataFrame:
        """Fetch cash flow statement data.

        Returns operating, investing, and financing cash flow line items
        including free cash flow, capital expenditures, and dividends paid.

        Parameters:
            symbol: Ticker symbol.
            period: ``"annual"`` or ``"quarter"``.
            limit: Maximum number of periods to return.

        Returns:
            DataFrame with one row per reporting period.

        Example:
            >>> df = fmp.cash_flow("AAPL", limit=5)
            >>> print(df[["date", "operatingCashFlow", "freeCashFlow"]].head())
        """
        data = self._get(
            "/stable/cash-flow-statement",
            {"symbol": clean_symbol(symbol), "period": period, "limit": limit},
        )
        return pd.DataFrame(data)

    def financial_growth(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> pd.DataFrame:
        """Fetch financial growth rates.

        Returns year-over-year or quarter-over-quarter growth rates for
        revenue, net income, EPS, free cash flow, and other key metrics.

        Parameters:
            symbol: Ticker symbol.
            period: ``"annual"`` or ``"quarter"``.
            limit: Maximum number of periods to return.

        Returns:
            DataFrame with growth rate columns (e.g. *revenueGrowth*,
            *netIncomeGrowth*, *epsgrowth*).

        Example:
            >>> df = fmp.financial_growth("AAPL", limit=5)
            >>> print(df[["date", "revenueGrowth", "netIncomeGrowth"]].head())
        """
        data = self._get(
            "/stable/financial-growth",
            {"symbol": clean_symbol(symbol), "period": period, "limit": limit},
        )
        return pd.DataFrame(data)

    # ==================================================================
    # 3. Fundamental data
    # ==================================================================

    def company_profile(self, symbol: str) -> dict[str, Any]:
        """Fetch a company profile.

        Returns a rich metadata dictionary covering company name, sector,
        industry, CEO, headquarters, description, market cap, beta,
        number of employees, and more.

        Parameters:
            symbol: Ticker symbol.

        Returns:
            Dictionary of profile fields.  Key fields include
            *companyName*, *sector*, *industry*, *mktCap*, *beta*,
            *description*, *fullTimeEmployees*, and *website*.

        Example:
            >>> p = fmp.company_profile("AAPL")
            >>> print(f"{p['companyName']} — {p['sector']}")
        """
        data = self._get("/stable/profile", {"symbol": clean_symbol(symbol)})
        if isinstance(data, list) and len(data) > 0:
            return data[0]  # type: ignore[return-value]
        return data if isinstance(data, dict) else {}

    def key_metrics(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> pd.DataFrame:
        """Fetch key financial metrics.

        Returns per-share and profitability metrics including revenue per
        share, net income per share, PE ratio, PB ratio, debt-to-equity,
        ROE, ROA, current ratio, and many more.

        Parameters:
            symbol: Ticker symbol.
            period: ``"annual"`` or ``"quarter"``.
            limit: Maximum number of periods to return.

        Returns:
            DataFrame with one row per period and metric columns.

        Example:
            >>> df = fmp.key_metrics("AAPL", limit=5)
            >>> print(df[["date", "peRatio", "debtToEquity"]].head())
        """
        data = self._get(
            "/stable/key-metrics",
            {"symbol": clean_symbol(symbol), "period": period, "limit": limit},
        )
        return pd.DataFrame(data)

    def ratios(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> pd.DataFrame:
        """Fetch financial ratios.

        Returns profitability, liquidity, leverage, efficiency, and
        valuation ratios including gross margin, operating margin, ROE,
        current ratio, quick ratio, debt-to-assets, and price-to-earnings.

        Parameters:
            symbol: Ticker symbol.
            period: ``"annual"`` or ``"quarter"``.
            limit: Maximum number of periods to return.

        Returns:
            DataFrame with one row per period and ratio columns.

        Example:
            >>> df = fmp.ratios("AAPL", limit=5)
            >>> print(df[["date", "returnOnEquity", "currentRatio"]].head())
        """
        data = self._get(
            "/stable/ratios",
            {"symbol": clean_symbol(symbol), "period": period, "limit": limit},
        )
        return pd.DataFrame(data)

    def ratios_ttm(self, symbol: str) -> dict[str, Any]:
        """Fetch trailing-twelve-month (TTM) financial ratios.

        Returns the most recent TTM ratios for the company, useful for
        comparing current valuation and profitability against historical
        annual snapshots.

        Parameters:
            symbol: Ticker symbol.

        Returns:
            Dictionary of TTM ratio values.

        Example:
            >>> r = fmp.ratios_ttm("AAPL")
            >>> print(f"ROE (TTM): {r['returnOnEquityTTM']:.2%}")
        """
        data = self._get("/stable/ratios-ttm", {"symbol": clean_symbol(symbol)})
        if isinstance(data, list) and len(data) > 0:
            return data[0]  # type: ignore[return-value]
        return data if isinstance(data, dict) else {}

    def enterprise_value(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> pd.DataFrame:
        """Fetch enterprise value data.

        Returns enterprise value and its components (market cap plus debt
        minus cash) across reporting periods, along with number of shares
        outstanding.

        Parameters:
            symbol: Ticker symbol.
            period: ``"annual"`` or ``"quarter"``.
            limit: Maximum number of periods to return.

        Returns:
            DataFrame with columns including *enterpriseValue*,
            *marketCapitalization*, *addTotalDebt*, and
            *minusCashAndCashEquivalents*.

        Example:
            >>> df = fmp.enterprise_value("AAPL", limit=5)
            >>> print(df[["date", "enterpriseValue"]].head())
        """
        data = self._get(
            "/stable/enterprise-values",
            {"symbol": clean_symbol(symbol), "period": period, "limit": limit},
        )
        return pd.DataFrame(data)

    # ==================================================================
    # 4. Valuation
    # ==================================================================

    def dcf(self, symbol: str) -> dict[str, Any]:
        """Fetch the discounted cash flow (DCF) intrinsic value estimate.

        Uses FMP's unlevered DCF model to estimate fair value based on
        projected free cash flows discounted at WACC.

        Parameters:
            symbol: Ticker symbol.

        Returns:
            Dictionary containing *dcf* (intrinsic value per share),
            *stockPrice* (current market price), and *date*.

        Example:
            >>> d = fmp.dcf("AAPL")
            >>> print(f"DCF: ${d['dcf']:.2f}  vs  price: ${d['stockPrice']:.2f}")
        """
        data = self._get(
            "/stable/discounted-cash-flow", {"symbol": clean_symbol(symbol)}
        )
        if isinstance(data, list) and len(data) > 0:
            return data[0]  # type: ignore[return-value]
        return data if isinstance(data, dict) else {}

    def advanced_dcf(self, symbol: str) -> dict[str, Any]:
        """Fetch the levered DCF valuation.

        Like :meth:`dcf` but factors in the company's debt structure
        to produce a post-debt intrinsic value, which is more appropriate
        for highly leveraged firms.

        Parameters:
            symbol: Ticker symbol.

        Returns:
            Dictionary containing the levered DCF value and supporting
            metrics.

        Example:
            >>> d = fmp.advanced_dcf("AAPL")
            >>> print(f"Levered DCF: ${d.get('dcf', 'N/A')}")
        """
        data = self._get(
            "/stable/levered-discounted-cash-flow",
            {"symbol": clean_symbol(symbol)},
        )
        if isinstance(data, list) and len(data) > 0:
            return data[0]  # type: ignore[return-value]
        return data if isinstance(data, dict) else {}

    def rating(self, symbol: str) -> dict[str, Any]:
        """Fetch FMP's proprietary company rating.

        Returns a composite rating based on DCF, ROE, ROA, DE ratio,
        PE ratio, and PB ratio, scored from *S* (strong buy) to *F*
        (strong sell).

        Parameters:
            symbol: Ticker symbol.

        Returns:
            Dictionary with *rating*, *ratingScore*,
            *ratingRecommendation*, and individual component scores.

        Example:
            >>> r = fmp.rating("AAPL")
            >>> print(f"{r['rating']}  ({r['ratingRecommendation']})")
        """
        data = self._get("/stable/ratings-snapshot", {"symbol": clean_symbol(symbol)})
        if isinstance(data, list) and len(data) > 0:
            return data[0]  # type: ignore[return-value]
        return data if isinstance(data, dict) else {}

    def score(self, symbol: str) -> dict[str, Any]:
        """Fetch financial scoring models (Piotroski, Altman Z, etc.).

        Returns quantitative scoring models that assess financial health
        and bankruptcy risk.

        Parameters:
            symbol: Ticker symbol.

        Returns:
            Dictionary containing *altmanZScore*, *piotroskiScore*, and
            related sub-scores.

        Example:
            >>> s = fmp.score("AAPL")
            >>> print(f"Altman Z: {s.get('altmanZScore', 'N/A')}")
            >>> print(f"Piotroski: {s.get('piotroskiScore', 'N/A')}")
        """
        data = self._get("/stable/financial-scores", {"symbol": clean_symbol(symbol)})
        if isinstance(data, list) and len(data) > 0:
            return data[0]  # type: ignore[return-value]
        return data if isinstance(data, dict) else {}

    # ==================================================================
    # 5. Earnings & events
    # ==================================================================

    def earnings(self, symbol: str) -> pd.DataFrame:
        """Fetch historical earnings reports for a company.

        Returns EPS estimates, actual EPS, revenue estimates, actual
        revenue, and the reporting time (before/after market) for each
        earnings announcement.

        Parameters:
            symbol: Ticker symbol.

        Returns:
            DataFrame with one row per earnings period.

        Example:
            >>> df = fmp.earnings("AAPL")
            >>> print(df[["date", "epsEstimated", "epsActual"]].head())
        """
        data = self._get("/stable/earnings", {"symbol": clean_symbol(symbol)})
        return pd.DataFrame(data)

    def earnings_surprises(self, symbol: str) -> pd.DataFrame:
        """Fetch earnings surprise history for a company.

        Shows the difference between estimated and actual EPS for each
        quarter, useful for identifying consistent beat/miss patterns.

        Parameters:
            symbol: Ticker symbol.

        Returns:
            DataFrame with columns including *date*, *estimatedEarning*,
            *actualEarning*, and the surprise amount.

        Example:
            >>> df = fmp.earnings_surprises("AAPL")
            >>> df["surprise"] = df["actualEarningResult"] - df["estimatedEarning"]
            >>> print(df[["date", "surprise"]].head())

        Notes:
            This endpoint uses the v3 legacy path as the stable API
            does not expose a per-symbol earnings-surprises endpoint.
        """
        sym = clean_symbol(symbol)
        data = self._get(f"/api/v3/earnings-surprises/{sym}", {})
        return pd.DataFrame(data)

    def earnings_calendar(
        self,
        from_date: DateLike | None = None,
        to_date: DateLike | None = None,
    ) -> pd.DataFrame:
        """Fetch the market-wide earnings announcement calendar.

        Returns upcoming and recent earnings dates for all companies,
        filterable by date range.

        Parameters:
            from_date: Inclusive start date for the calendar window.
            to_date: Inclusive end date for the calendar window.

        Returns:
            DataFrame with columns *date*, *symbol*, *eps*, *epsEstimated*,
            *revenue*, *revenueEstimated*, and *time*.

        Example:
            >>> df = fmp.earnings_calendar("2024-07-01", "2024-07-31")
            >>> print(df.shape)
        """
        params: dict[str, Any] = {}
        f = self._fmt_date(from_date)
        t = self._fmt_date(to_date)
        if f:
            params["from"] = f
        if t:
            params["to"] = t
        data = self._get("/stable/earnings-calendar", params)
        return pd.DataFrame(data)

    def dividends(self, symbol: str) -> pd.DataFrame:
        """Fetch dividend payment history for a company.

        Returns declaration date, record date, payment date, and
        dividend amount per share for each distribution.

        Parameters:
            symbol: Ticker symbol.

        Returns:
            DataFrame with dividend event details.

        Example:
            >>> df = fmp.dividends("AAPL")
            >>> print(df[["date", "dividend"]].head())
        """
        data = self._get("/stable/dividends", {"symbol": clean_symbol(symbol)})
        return pd.DataFrame(data)

    def stock_splits(self, symbol: str) -> pd.DataFrame:
        """Fetch stock split history for a company.

        Parameters:
            symbol: Ticker symbol.

        Returns:
            DataFrame with split dates and ratios (e.g. 4-for-1).

        Example:
            >>> df = fmp.stock_splits("AAPL")
            >>> print(df[["date", "numerator", "denominator"]].head())
        """
        data = self._get("/stable/splits", {"symbol": clean_symbol(symbol)})
        return pd.DataFrame(data)

    def ipo_calendar(
        self,
        from_date: DateLike | None = None,
        to_date: DateLike | None = None,
    ) -> pd.DataFrame:
        """Fetch the IPO calendar.

        Lists upcoming and past initial public offerings with filing
        details, expected pricing, share counts, and exchange listing.

        Parameters:
            from_date: Inclusive start date.
            to_date: Inclusive end date.

        Returns:
            DataFrame with IPO event details.

        Example:
            >>> df = fmp.ipo_calendar("2024-01-01", "2024-06-30")
            >>> print(df[["date", "company", "exchange"]].head())
        """
        params: dict[str, Any] = {}
        f = self._fmt_date(from_date)
        t = self._fmt_date(to_date)
        if f:
            params["from"] = f
        if t:
            params["to"] = t
        data = self._get("/stable/ipos-calendar", params)
        return pd.DataFrame(data)

    # ==================================================================
    # 6. Market data
    # ==================================================================

    def search(
        self,
        query: str,
        limit: int = 10,
        exchange: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for symbols by company name or ticker fragment.

        Parameters:
            query: Free-text search query (e.g. ``"apple"`` or ``"AA"``).
            limit: Maximum results to return.
            exchange: Optional exchange filter (e.g. ``"NASDAQ"``).

        Returns:
            List of matching symbol dictionaries with *symbol*, *name*,
            *currency*, *stockExchange*, and *exchangeShortName* fields.

        Example:
            >>> results = fmp.search("Tesla", limit=5)
            >>> for r in results:
            ...     print(f"{r['symbol']:>10}  {r['name']}")
        """
        params: dict[str, Any] = {"query": query, "limit": limit}
        if exchange:
            params["exchange"] = exchange
        data = self._get("/stable/search-name", params)
        return data if isinstance(data, list) else []

    def stock_list(self) -> pd.DataFrame:
        """Fetch the full list of available stock symbols.

        Returns every symbol FMP covers along with exchange, name,
        price, and type information.

        Returns:
            DataFrame with columns including *symbol*, *name*, *price*,
            *exchange*, and *type*.

        Example:
            >>> df = fmp.stock_list()
            >>> print(f"Total symbols: {len(df)}")
        """
        data = self._get("/stable/stock-list", {})
        return pd.DataFrame(data)

    def sector_performance(self) -> dict[str, Any]:
        """Fetch real-time sector performance snapshot.

        Returns the current-day percentage change for each market sector
        (Technology, Healthcare, Financials, etc.).

        Returns:
            Dictionary (or list of dicts) with sector names and their
            performance percentages.

        Example:
            >>> perf = fmp.sector_performance()
            >>> for item in (perf if isinstance(perf, list) else [perf]):
            ...     print(f"{item['sector']:>25}: {item['changesPercentage']:+.2f}%")
        """
        data = self._get("/stable/sector-performance-snapshot", {})
        return data  # type: ignore[return-value]

    def market_gainers(self) -> pd.DataFrame:
        """Fetch today's biggest market gainers.

        Returns:
            DataFrame of top gaining stocks with price, change,
            and volume data.

        Example:
            >>> df = fmp.market_gainers()
            >>> print(df[["symbol", "changesPercentage"]].head(10))
        """
        data = self._get("/stable/biggest-gainers", {})
        return pd.DataFrame(data)

    def market_losers(self) -> pd.DataFrame:
        """Fetch today's biggest market losers.

        Returns:
            DataFrame of top losing stocks with price, change,
            and volume data.

        Example:
            >>> df = fmp.market_losers()
            >>> print(df[["symbol", "changesPercentage"]].head(10))
        """
        data = self._get("/stable/biggest-losers", {})
        return pd.DataFrame(data)

    def market_most_active(self) -> pd.DataFrame:
        """Fetch today's most actively traded stocks.

        Returns:
            DataFrame of the most active stocks ranked by volume.

        Example:
            >>> df = fmp.market_most_active()
            >>> print(df[["symbol", "volume", "price"]].head(10))
        """
        data = self._get("/stable/most-actives", {})
        return pd.DataFrame(data)

    # ==================================================================
    # 7. SEC & institutional
    # ==================================================================

    def sec_filings(
        self,
        symbol: str,
        type: str | None = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch SEC filings for a company.

        Returns filing metadata including form type (10-K, 10-Q, 8-K, etc.),
        filing date, accepted date, and links to the full document.

        Parameters:
            symbol: Ticker symbol.
            type: Optional form type filter (e.g. ``"10-K"``, ``"8-K"``).
                When *None*, returns all filing types.
            limit: Maximum number of filings to return.

        Returns:
            DataFrame with filing records.

        Example:
            >>> df = fmp.sec_filings("AAPL", type="10-K", limit=10)
            >>> print(df[["date", "type", "link"]].head())
        """
        params: dict[str, Any] = {"symbol": clean_symbol(symbol), "limit": limit}
        if type:
            params["formType"] = type
            data = self._get("/stable/sec-filings-search/form-type", params)
        else:
            data = self._get("/stable/sec-filings-search/symbol", params)
        return pd.DataFrame(data)

    def insider_trades(
        self,
        symbol: str,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch insider trading activity for a company.

        Returns Form 4 filings showing transactions by officers,
        directors, and 10%+ owners, including buy/sell type, shares
        traded, price, and value.

        Parameters:
            symbol: Ticker symbol.
            limit: Maximum number of records to return.

        Returns:
            DataFrame with insider transaction details.

        Example:
            >>> df = fmp.insider_trades("AAPL", limit=20)
            >>> print(df[["transactionDate", "transactionType",
            ...            "securitiesTransacted"]].head())
        """
        data = self._get(
            "/stable/insider-trading/search",
            {"symbol": clean_symbol(symbol), "limit": limit},
        )
        return pd.DataFrame(data)

    def institutional_holders(self, symbol: str) -> pd.DataFrame:
        """Fetch institutional ownership data (13F filings) for a company.

        Returns the latest institutional holders, their share counts,
        and portfolio weights derived from 13F filings.

        Parameters:
            symbol: Ticker symbol.

        Returns:
            DataFrame with one row per institutional holder.

        Example:
            >>> df = fmp.institutional_holders("AAPL")
            >>> print(df[["holder", "shares"]].head(10))
        """
        data = self._get(
            "/stable/institutional-ownership/symbol-positions-summary",
            {"symbol": clean_symbol(symbol)},
        )
        return pd.DataFrame(data)

    # ==================================================================
    # 8. News
    # ==================================================================

    def stock_news(
        self,
        symbol: str,
        limit: int = 50,
    ) -> pd.DataFrame:
        """Fetch recent news articles for a specific stock.

        Parameters:
            symbol: Ticker symbol.
            limit: Maximum number of articles to return.

        Returns:
            DataFrame with columns *title*, *text*, *publishedDate*,
            *site*, *url*, and *image*.

        Example:
            >>> df = fmp.stock_news("AAPL", limit=5)
            >>> for _, row in df.iterrows():
            ...     print(f"[{row['publishedDate']}] {row['title']}")
        """
        data = self._get(
            "/stable/news/stock",
            {"symbols": clean_symbol(symbol), "limit": limit},
        )
        return pd.DataFrame(data)

    def general_news(self, limit: int = 50) -> pd.DataFrame:
        """Fetch the latest general financial news headlines.

        Returns:
            DataFrame of news articles not tied to a specific symbol.

        Example:
            >>> df = fmp.general_news(limit=10)
            >>> print(df["title"].tolist())
        """
        data = self._get("/stable/news/general-latest", {"limit": limit})
        return pd.DataFrame(data)

    def press_releases(
        self,
        symbol: str,
        limit: int = 50,
    ) -> pd.DataFrame:
        """Fetch press releases issued by a company.

        Parameters:
            symbol: Ticker symbol.
            limit: Maximum number of press releases to return.

        Returns:
            DataFrame with press release titles, dates, and full text.

        Example:
            >>> df = fmp.press_releases("AAPL", limit=5)
            >>> print(df[["date", "title"]].head())
        """
        data = self._get(
            "/stable/news/press-releases",
            {"symbol": clean_symbol(symbol), "limit": limit},
        )
        return pd.DataFrame(data)

    # ==================================================================
    # 9. Economic data
    # ==================================================================

    def economic_indicator(self, name: str) -> pd.DataFrame:
        """Fetch a macroeconomic indicator time series.

        Returns historical values for indicators such as GDP, CPI,
        unemployment rate, federal funds rate, consumer sentiment, and
        more.

        Parameters:
            name: Indicator name as recognised by FMP (e.g. ``"GDP"``,
                ``"CPI"``, ``"unemploymentRate"``,
                ``"federalFundsRate"``).

        Returns:
            DataFrame with *date* and *value* columns, sorted ascending.

        Example:
            >>> df = fmp.economic_indicator("GDP")
            >>> print(df.tail())

        See Also:
            treasury_rates: For daily Treasury yield curve data.
        """
        data = self._get("/stable/economic-indicators", {"name": name})
        df = pd.DataFrame(data)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df

    def treasury_rates(self) -> pd.DataFrame:
        """Fetch daily U.S. Treasury yield curve rates.

        Returns yields for 1-month through 30-year maturities,
        updated daily.

        Returns:
            DataFrame with one row per date and columns for each
            maturity tenor (e.g. *month1*, *month3*, *year1*, *year5*,
            *year10*, *year30*).

        Example:
            >>> df = fmp.treasury_rates()
            >>> print(df[["date", "year2", "year10"]].tail())

        See Also:
            economic_indicator: For broader macro series (GDP, CPI, etc.).
        """
        data = self._get("/stable/treasury-rates", {})
        df = pd.DataFrame(data)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df

    # ── Analyst & Ratings ─────────────────────────────────────────────

    def analyst_estimates(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> pd.DataFrame:
        """Fetch analyst earnings and revenue estimates.

        Returns:
            DataFrame with estimatedRevenueAvg, estimatedEpsAvg,
            numberAnalystEstimatedRevenue, etc.
        """
        return self._get_df(
            "/stable/analyst-estimates",
            {"symbol": symbol, "period": period, "limit": limit},
        )

    def price_target(self, symbol: str) -> pd.DataFrame:
        """Fetch analyst price target history.

        Returns:
            DataFrame with publishedDate, analystName, priceTarget.
        """
        return self._get_df("/stable/price-target", {"symbol": symbol})

    def price_target_consensus(self, symbol: str) -> dict[str, Any]:
        """Get consensus price target.

        Returns:
            Dict with targetHigh, targetLow, targetConsensus, targetMedian.
        """
        data = self._get("/stable/price-target-consensus", {"symbol": symbol})
        return data[0] if isinstance(data, list) and data else data

    def upgrades_downgrades(self, symbol: str) -> pd.DataFrame:
        """Fetch analyst upgrades and downgrades.

        Returns:
            DataFrame with publishedDate, action, newGrade, previousGrade.
        """
        return self._get_df("/stable/upgrades-downgrades", {"symbol": symbol})

    def stock_peers(self, symbol: str) -> list[str]:
        """Get peer companies for a stock.

        Returns:
            List of peer ticker symbols.
        """
        data = self._get("/stable/stock-peers", {"symbol": symbol})
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0].get("peersList", [])
        return []

    # ── ESG ───────────────────────────────────────────────────────────

    def esg_ratings(self, symbol: str) -> dict[str, Any]:
        """Get ESG ratings for a company.

        Returns:
            Dict with environmentalScore, socialScore,
            governanceScore, ESGScore.
        """
        data = self._get(
            "/stable/esg-environmental-social-governance-data", {"symbol": symbol}
        )
        return data[0] if isinstance(data, list) and data else data

    # ── Index Constituents ────────────────────────────────────────────

    def index_constituents(self, index: str = "sp500") -> pd.DataFrame:
        """Get constituents of a major index.

        Parameters:
            index: 'sp500', 'nasdaq', or 'dowjones'.

        Returns:
            DataFrame with symbol, name, sector, subSector.
        """
        path_map = {
            "sp500": "/stable/sp500-constituents",
            "nasdaq": "/stable/nasdaq-constituents",
            "dowjones": "/stable/dowjones-constituents",
        }
        return self._get_df(path_map.get(index, f"/stable/{index}-constituents"))

    # ── Revenue Segments ──────────────────────────────────────────────

    def revenue_product_segmentation(self, symbol: str) -> pd.DataFrame:
        """Fetch revenue breakdown by product segment."""
        return self._get_df("/stable/revenue-product-segmentation", {"symbol": symbol})

    def revenue_geographic_segmentation(self, symbol: str) -> pd.DataFrame:
        """Fetch revenue breakdown by geographic region."""
        return self._get_df(
            "/stable/revenue-geographic-segmentation", {"symbol": symbol}
        )

    # ── Government Trading ────────────────────────────────────────────

    def senate_trades(self, symbol: str | None = None) -> pd.DataFrame:
        """Fetch US Senate stock trading disclosures.

        Parameters:
            symbol: Optional ticker to filter. None = all senators.
        """
        params: dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        return self._get_df("/stable/senate-trading", params)

    def house_trades(self, symbol: str | None = None) -> pd.DataFrame:
        """Fetch US House of Representatives trading disclosures.

        Parameters:
            symbol: Optional ticker to filter.
        """
        params: dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        return self._get_df("/stable/house-disclosure", params)

    # ── Stock Screener ────────────────────────────────────────────────

    def stock_screener(
        self,
        market_cap_gt: int | None = None,
        market_cap_lt: int | None = None,
        sector: str | None = None,
        industry: str | None = None,
        country: str | None = None,
        exchange: str | None = None,
        dividend_gt: float | None = None,
        volume_gt: int | None = None,
        beta_gt: float | None = None,
        beta_lt: float | None = None,
        price_gt: float | None = None,
        price_lt: float | None = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Screen stocks by fundamental criteria.

        Parameters:
            market_cap_gt: Minimum market cap.
            market_cap_lt: Maximum market cap.
            sector: Filter by sector (e.g., 'Technology').
            industry: Filter by industry.
            country: Filter by country (e.g., 'US').
            exchange: Filter by exchange.
            dividend_gt: Minimum dividend yield.
            volume_gt: Minimum average volume.
            beta_gt: Minimum beta.
            beta_lt: Maximum beta.
            price_gt: Minimum price.
            price_lt: Maximum price.
            limit: Max results.

        Returns:
            DataFrame of matching stocks with key metrics.
        """
        params: dict[str, Any] = {"limit": limit}
        if market_cap_gt is not None:
            params["marketCapMoreThan"] = market_cap_gt
        if market_cap_lt is not None:
            params["marketCapLowerThan"] = market_cap_lt
        if sector:
            params["sector"] = sector
        if industry:
            params["industry"] = industry
        if country:
            params["country"] = country
        if exchange:
            params["exchange"] = exchange
        if dividend_gt is not None:
            params["dividendMoreThan"] = dividend_gt
        if volume_gt is not None:
            params["volumeMoreThan"] = volume_gt
        if beta_gt is not None:
            params["betaMoreThan"] = beta_gt
        if beta_lt is not None:
            params["betaLowerThan"] = beta_lt
        if price_gt is not None:
            params["priceMoreThan"] = price_gt
        if price_lt is not None:
            params["priceLowerThan"] = price_lt
        return self._get_df("/stable/stock-screener", params)

    # ── Shares Float ──────────────────────────────────────────────────

    def shares_float(self, symbol: str) -> dict[str, Any]:
        """Get shares float data.

        Returns:
            Dict with freeFloat, floatShares, outstandingShares.
        """
        data = self._get("/stable/shares-float", {"symbol": symbol})
        return data[0] if isinstance(data, list) and data else data

    # ── Executive Compensation ────────────────────────────────────────

    def executive_compensation(self, symbol: str) -> pd.DataFrame:
        """Fetch executive compensation data.

        Returns:
            DataFrame with name, title, salary, bonus, stockAward.
        """
        return self._get_df("/stable/executive-compensation", {"symbol": symbol})


# ======================================================================
# DataProvider adapter — thin OHLCV bridge for the provider registry
# ======================================================================


class FMPProvider:
    """Minimal DataProvider-compatible adapter for FMP price data.

    Registered automatically in :func:`register_builtin_providers` when
    *httpx* is importable.  Delegates to :class:`FMPClient` for the
    actual HTTP calls.

    The ``FMP_API_KEY`` environment variable must be set.
    """

    name = "fmp"

    @requires_extra("market-data")
    def fetch_prices(
        self,
        symbol: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
        **kwargs: Any,
    ) -> pd.Series:
        """Fetch adjusted close prices from FMP.

        Parameters:
            symbol: Ticker symbol (e.g. ``"AAPL"``).
            start: Start date.
            end: End date.

        Returns:
            Adjusted close price series with DatetimeIndex.
        """
        client = FMPClient()
        df = client.historical_price(symbol, start=start, end=end, interval="daily")
        if df.empty:
            return pd.Series(dtype=float, name=clean_symbol(symbol))
        series = df.set_index("date")["close"]
        series.name = clean_symbol(symbol)
        return series

    @requires_extra("market-data")
    def fetch_ohlcv(
        self,
        symbol: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from FMP.

        Parameters:
            symbol: Ticker symbol.
            start: Start date.
            end: End date.

        Returns:
            DataFrame with open, high, low, close, volume columns.
        """
        client = FMPClient()
        df = client.historical_price(symbol, start=start, end=end, interval="daily")
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = df.set_index("date")
        cols = [
            c for c in ["open", "high", "low", "close", "volume"] if c in df.columns
        ]
        return df[cols]


__all__ = [
    "FMPClient",
    "FMPProvider",
]
