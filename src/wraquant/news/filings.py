"""SEC filings retrieval and analysis.

Provides functions for searching and retrieving SEC filings (10-K, 10-Q,
8-K, and others) from the FMP data provider.  SEC filings are the
authoritative source of corporate financial data and material event
disclosures.

Filing types commonly used in quant strategies:

- **10-K** (annual report) -- Complete financial statements, risk factors,
  management discussion.  The definitive source for fundamental analysis.
- **10-Q** (quarterly report) -- Interim financials, updated risk factors.
  Useful for tracking intra-year trends.
- **8-K** (current report) -- Material events: executive changes, M&A,
  contract awards, covenant violations.  The most time-sensitive filing
  type for event-driven strategies.
- **4** (insider transactions) -- Officer/director stock trades.  Filed
  within 2 business days of the transaction.
- **13F** (institutional holdings) -- Quarterly disclosure of equity
  positions by institutional managers with >$100M AUM.

References:
    - SEC EDGAR: https://www.sec.gov/edgar
    - Loughran & McDonald (2011), "When Is a Liability Not a Liability?
      Textual Analysis, Dictionaries, and 10-Ks"
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from wraquant.core.decorators import requires_extra


@requires_extra("market-data")
def recent_filings(
    symbol: str,
    form_type: str | None = None,
    limit: int = 20,
) -> pd.DataFrame:
    """Fetch recent SEC filings for a company.

    Returns a DataFrame of SEC filings sorted by date (most recent
    first).  Optionally filter by form type to focus on specific
    filing categories.

    Parameters:
        symbol: Ticker symbol (e.g., ``"AAPL"``).
        form_type: SEC form type to filter by (e.g., ``"10-K"``,
            ``"10-Q"``, ``"8-K"``, ``"4"``, ``"13F"``).  If None,
            returns all filing types.
        limit: Maximum number of filings to return.

    Returns:
        DataFrame with columns:
        - **date** (*str*) -- Filing date.
        - **type** (*str*) -- SEC form type.
        - **title** (*str*) -- Filing title/description.
        - **url** (*str*) -- Link to the filing on SEC EDGAR.
        - **cik** (*str*) -- SEC Central Index Key.

    Example:
        >>> from wraquant.news.filings import recent_filings
        >>> filings = recent_filings("MSFT", form_type="10-K", limit=5)
        >>> print(filings[["date", "type", "title"]])

    See Also:
        annual_reports: Shortcut for 10-K filings.
        quarterly_reports: Shortcut for 10-Q filings.
        material_events: Shortcut for 8-K filings.
    """
    from wraquant.data.providers.fmp import FMPClient

    client = FMPClient()
    df = client.sec_filings(symbol, type=form_type, limit=limit)

    if df.empty:
        return pd.DataFrame(columns=["date", "type", "title", "url", "cik"])

    # Standardize columns
    col_map = _build_col_map(df)
    result = _standardize_filings_df(df, col_map)

    return result.head(limit).reset_index(drop=True)


@requires_extra("market-data")
def filing_search(
    query: str,
    from_date: str | date | None = None,
    to_date: str | date | None = None,
    *,
    limit: int = 50,
) -> pd.DataFrame:
    """Search SEC filings by keyword across recent filings.

    Performs a text search over filing titles and descriptions to find
    filings related to a specific topic (e.g., ``"merger"``,
    ``"restatement"``, ``"executive departure"``).

    This performs client-side filtering on bulk filing data, so it works
    best with a focused symbol-level query.  For market-wide filing
    searches, use the SEC EDGAR full-text search API directly.

    Parameters:
        query: Search query string.  Matched case-insensitively against
            filing titles.  Can include a ticker symbol prefix separated
            by ``":"`` (e.g., ``"AAPL:merger"``).
        from_date: Start date filter (inclusive).
        to_date: End date filter (inclusive).
        limit: Maximum number of results to return.

    Returns:
        DataFrame with columns: ``date``, ``type``, ``title``, ``url``,
        ``cik``, ``symbol``.

    Example:
        >>> from wraquant.news.filings import filing_search
        >>> results = filing_search("TSLA:merger", limit=10)
        >>> results = filing_search(
        ...     "AAPL:executive",
        ...     from_date="2023-01-01",
        ...     to_date="2023-12-31",
        ... )

    See Also:
        recent_filings: Browse filings by symbol and type.
    """
    from wraquant.data.providers.fmp import FMPClient

    client = FMPClient()

    # Parse query for symbol prefix
    symbol: str | None = None
    search_term = query
    if ":" in query:
        parts = query.split(":", maxsplit=1)
        symbol = parts[0].strip().upper()
        search_term = parts[1].strip()

    if not symbol:
        # Try to use the query itself as the symbol
        symbol = search_term.strip().upper()
        search_term = ""

    df = client.sec_filings(symbol, limit=limit * 3)  # Fetch extra for filtering

    if df.empty:
        return pd.DataFrame(columns=["date", "type", "title", "url", "cik", "symbol"])

    col_map = _build_col_map(df)
    result = _standardize_filings_df(df, col_map)
    result["symbol"] = symbol

    # Text filter
    if search_term:
        title_lower = result["title"].str.lower()
        mask = title_lower.str.contains(search_term.lower(), na=False)
        result = result.loc[mask]

    # Date filter
    if from_date or to_date:
        result["_date_parsed"] = pd.to_datetime(result["date"], errors="coerce")
        if from_date:
            from_ts = pd.Timestamp(str(from_date))
            result = result.loc[result["_date_parsed"] >= from_ts]
        if to_date:
            to_ts = pd.Timestamp(str(to_date))
            result = result.loc[result["_date_parsed"] <= to_ts]
        result = result.drop(columns=["_date_parsed"])

    return result.head(limit).reset_index(drop=True)


@requires_extra("market-data")
def annual_reports(
    symbol: str,
    limit: int = 5,
) -> pd.DataFrame:
    """Fetch 10-K annual report filings for a company.

    Convenience wrapper around ``recent_filings`` that filters for
    10-K filings only.  Annual reports are the most comprehensive
    disclosure and include audited financial statements, MD&A
    (management discussion and analysis), risk factors, and more.

    Parameters:
        symbol: Ticker symbol (e.g., ``"GOOG"``).
        limit: Maximum number of annual reports to return.

    Returns:
        DataFrame with columns: ``date``, ``type``, ``title``, ``url``,
        ``cik``.  Sorted by date descending.

    Example:
        >>> from wraquant.news.filings import annual_reports
        >>> reports = annual_reports("AMZN", limit=3)
        >>> print(reports[["date", "title"]])

    See Also:
        quarterly_reports: 10-Q filings.
        material_events: 8-K filings.
        recent_filings: All filing types.
    """
    return recent_filings(symbol, form_type="10-K", limit=limit)


@requires_extra("market-data")
def quarterly_reports(
    symbol: str,
    limit: int = 10,
) -> pd.DataFrame:
    """Fetch 10-Q quarterly report filings for a company.

    Convenience wrapper around ``recent_filings`` that filters for
    10-Q filings only.  Quarterly reports provide interim financial
    statements and updated risk disclosures between annual reports.

    Parameters:
        symbol: Ticker symbol (e.g., ``"NFLX"``).
        limit: Maximum number of quarterly reports to return.

    Returns:
        DataFrame with columns: ``date``, ``type``, ``title``, ``url``,
        ``cik``.  Sorted by date descending.

    Example:
        >>> from wraquant.news.filings import quarterly_reports
        >>> reports = quarterly_reports("META", limit=4)
        >>> print(reports[["date", "title"]])

    See Also:
        annual_reports: 10-K filings.
        material_events: 8-K filings.
        recent_filings: All filing types.
    """
    return recent_filings(symbol, form_type="10-Q", limit=limit)


@requires_extra("market-data")
def material_events(
    symbol: str,
    limit: int = 20,
) -> pd.DataFrame:
    """Fetch 8-K current report filings for a company.

    8-K filings disclose material events that shareholders should know
    about between regular SEC reporting periods.  These include:

    - Entry into or termination of a material agreement
    - Bankruptcy or receivership
    - Departure or appointment of officers/directors
    - Changes in fiscal year
    - Amendments to articles of incorporation
    - Material impairments
    - Unregistered sales of equity securities
    - Changes in certifying accountant

    For event-driven strategies, 8-K filings are the most actionable
    filing type because they signal discrete, potentially price-moving
    corporate actions.

    Parameters:
        symbol: Ticker symbol (e.g., ``"BA"``).
        limit: Maximum number of 8-K filings to return.

    Returns:
        DataFrame with columns: ``date``, ``type``, ``title``, ``url``,
        ``cik``.  Sorted by date descending.

    Example:
        >>> from wraquant.news.filings import material_events
        >>> events = material_events("GM", limit=10)
        >>> print(events[["date", "title"]])

    See Also:
        annual_reports: 10-K filings.
        quarterly_reports: 10-Q filings.
        recent_filings: All filing types.
    """
    return recent_filings(symbol, form_type="8-K", limit=limit)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_col_map(df: pd.DataFrame) -> dict[str, str | None]:
    """Build a mapping from standardized names to actual column names.

    Parameters:
        df: DataFrame with potentially varying column names.

    Returns:
        Dictionary mapping standard names to actual column names.
    """
    return {
        "date": _resolve_col(df, ["fillingDate", "filingDate", "date", "acceptedDate"]),
        "type": _resolve_col(df, ["type", "formType", "form_type"]),
        "title": _resolve_col(df, ["title", "description", "link", "finalLink"]),
        "url": _resolve_col(df, ["link", "finalLink", "url", "edgarUrl"]),
        "cik": _resolve_col(df, ["cik", "CIK"]),
    }


def _standardize_filings_df(
    df: pd.DataFrame,
    col_map: dict[str, str | None],
) -> pd.DataFrame:
    """Standardize a filings DataFrame to consistent column names.

    Parameters:
        df: Raw filings DataFrame.
        col_map: Mapping from standard names to actual column names
            (from ``_build_col_map``).

    Returns:
        DataFrame with standardized columns: ``date``, ``type``,
        ``title``, ``url``, ``cik``.
    """
    result = pd.DataFrame()

    for std_name, actual_col in col_map.items():
        if actual_col and actual_col in df.columns:
            result[std_name] = df[actual_col].astype(str)
        else:
            result[std_name] = ""

    return result


def _resolve_col(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """Find the first matching column name from a list of candidates.

    Parameters:
        df: DataFrame to search.
        candidates: Ordered list of possible column names.

    Returns:
        The first matching column name, or None if none match.
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None
