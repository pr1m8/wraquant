"""Currency pair definitions and cross rate calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from wraquant.core.types import Currency


@dataclass(frozen=True)
class CurrencyPair:
    """A forex currency pair.

    Parameters:
        base: Base currency (e.g., EUR in EURUSD).
        quote: Quote currency (e.g., USD in EURUSD).

    Example:
        >>> pair = CurrencyPair(Currency.EUR, Currency.USD)
        >>> pair.symbol
        'EURUSD'
    """

    base: Currency
    quote: Currency

    @property
    def symbol(self) -> str:
        """Standard pair symbol (e.g., 'EURUSD')."""
        return f"{self.base}{self.quote}"

    @property
    def yahoo_symbol(self) -> str:
        """Yahoo Finance ticker format."""
        return f"{self.base}{self.quote}=X"

    @property
    def is_jpy_pair(self) -> bool:
        """Whether this pair involves JPY (different pip size)."""
        return Currency.JPY in (self.base, self.quote)

    @property
    def pip_size(self) -> float:
        """Size of one pip for this pair."""
        return 0.01 if self.is_jpy_pair else 0.0001

    def inverse(self) -> CurrencyPair:
        """Return the inverse pair (e.g., EURUSD -> USDEUR)."""
        return CurrencyPair(self.quote, self.base)

    @classmethod
    def from_string(cls, s: str) -> CurrencyPair:
        """Parse a pair from string like 'EURUSD' or 'EUR/USD'.

        Parameters:
            s: Pair string (6 chars or with separator).

        Returns:
            CurrencyPair instance.
        """
        s = s.upper().replace("/", "").replace("-", "").replace("=X", "")
        if len(s) != 6:
            raise ValueError(f"Invalid currency pair: {s}")
        return cls(Currency(s[:3]), Currency(s[3:]))


def major_pairs() -> list[CurrencyPair]:
    """Return the 7 major forex pairs.

    Returns:
        List of major currency pairs.
    """
    return [
        CurrencyPair(Currency.EUR, Currency.USD),
        CurrencyPair(Currency.GBP, Currency.USD),
        CurrencyPair(Currency.USD, Currency.JPY),
        CurrencyPair(Currency.USD, Currency.CHF),
        CurrencyPair(Currency.AUD, Currency.USD),
        CurrencyPair(Currency.USD, Currency.CAD),
        CurrencyPair(Currency.NZD, Currency.USD),
    ]


def cross_rate(
    pair1_rate: float,
    pair2_rate: float,
    method: str = "divide",
) -> float:
    """Calculate a cross rate from two pairs sharing a common currency.

    Use cross rates to derive the exchange rate for a currency pair that
    is not directly quoted.  For example, EUR/JPY can be derived from
    EUR/USD and USD/JPY.

    The method depends on how the pairs share a common currency:

    - ``'multiply'``: when pair1 = A/B and pair2 = B/C, result = A/C.
    - ``'divide'``: when pair1 = A/B and pair2 = C/B, result = A/C.

    Parameters:
        pair1_rate: Rate for first pair.
        pair2_rate: Rate for second pair.
        method: ``'divide'`` (pair1/pair2) or ``'multiply'``
            (pair1 * pair2).

    Returns:
        Cross rate.

    Example:
        >>> cross_rate(1.1000, 110.00, method="multiply")  # EURJPY from EURUSD * USDJPY
        121.0
        >>> cross_rate(1.1000, 1.3000, method="divide")  # EURGBP from EURUSD / GBPUSD
        0.8461538461538461

    See Also:
        CurrencyPair: Currency pair representation.
    """
    if method == "multiply":
        return pair1_rate * pair2_rate
    return pair1_rate / pair2_rate


def correlation_matrix(
    pairs_df: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """Rolling correlation matrix between currency pairs.

    Use this to identify which currency pairs move together and which
    diverge.  High positive correlation means two pairs track each
    other closely (little diversification benefit); negative correlation
    offers hedging opportunities.

    Computes pairwise Pearson correlations of returns over a rolling
    window.  Returns the most recent window's correlation matrix.

    Parameters:
        pairs_df: DataFrame where each column is the price series of a
            currency pair (e.g., columns ``['EURUSD', 'GBPUSD', 'USDJPY']``).
            Index should be datetime.
        window: Rolling window size in periods (default 60, roughly
            3 months of daily data).  Shorter windows capture recent
            regime shifts; longer windows are more stable.

    Returns:
        Correlation matrix as a DataFrame (pairs x pairs).  Values
        range from -1.0 (perfect negative correlation) to +1.0
        (perfect positive correlation).

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> prices = pd.DataFrame({
        ...     'EURUSD': np.cumsum(rng.normal(0, 0.001, 100)) + 1.10,
        ...     'GBPUSD': np.cumsum(rng.normal(0, 0.001, 100)) + 1.30,
        ... })
        >>> corr = correlation_matrix(prices, window=30)
        >>> corr.shape
        (2, 2)

    See Also:
        currency_strength: Relative strength of individual currencies.
    """
    returns = pairs_df.pct_change().dropna()
    if len(returns) < window:
        # Fall back to full-sample correlation if not enough data
        return returns.corr()
    return returns.tail(window).corr()


def currency_strength(
    pairs_df: pd.DataFrame,
    window: int | None = None,
) -> pd.Series:
    """Compute relative strength of each currency from cross rates.

    Use this to identify which currencies are strengthening and which
    are weakening across the board.  A currency that is appreciating
    against most counterparts will have a high strength score.

    The algorithm extracts individual currency codes from pair column
    names (e.g., ``'EURUSD'`` yields ``EUR`` and ``USD``), computes
    returns, and averages each currency's performance across all pairs
    it appears in (positive for appreciation, negative for depreciation).

    Parameters:
        pairs_df: DataFrame where each column is named as a 6-character
            pair (e.g., ``'EURUSD'``, ``'USDJPY'``).  Values are prices.
        window: Number of recent periods to use for strength calculation.
            If *None*, uses the full history.

    Returns:
        Series indexed by currency code with mean return as the strength
        score.  Positive values indicate the currency is strengthening
        on average; negative values indicate weakening.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> prices = pd.DataFrame({
        ...     'EURUSD': np.cumsum(rng.normal(0.0001, 0.001, 100)) + 1.10,
        ...     'USDJPY': np.cumsum(rng.normal(0.0001, 0.001, 100)) + 110.0,
        ... })
        >>> strength = currency_strength(prices)
        >>> 'EUR' in strength.index
        True

    See Also:
        correlation_matrix: Pairwise correlation between pairs.
    """
    import pandas as pd  # noqa: F811

    returns = pairs_df.pct_change().dropna()
    if window is not None and len(returns) > window:
        returns = returns.tail(window)

    # Accumulate per-currency contributions
    currency_returns: dict[str, list[float]] = {}

    for col in returns.columns:
        name = str(col).upper().replace("/", "").replace("-", "").replace("=X", "")
        if len(name) < 6:
            continue
        base = name[:3]
        quote = name[3:6]

        mean_ret = float(returns[col].mean())

        # Base currency appreciates when pair goes up
        currency_returns.setdefault(base, []).append(mean_ret)
        # Quote currency depreciates when pair goes up
        currency_returns.setdefault(quote, []).append(-mean_ret)

    strength_scores = {
        ccy: float(np.mean(rets)) for ccy, rets in currency_returns.items()
    }
    import pandas as pd  # noqa: F811

    return pd.Series(strength_scores).sort_values(ascending=False)


def volatility_by_session(
    prices: pd.DataFrame | pd.Series,
    sessions: dict[str, tuple[int, int]] | None = None,
) -> dict[str, float]:
    """Compute price volatility during each forex trading session.

    Use this to identify which session carries the most volatility for
    a given currency pair.  Typically London and the London/New York
    overlap have the highest volatility for major pairs.

    The function groups intraday returns by session (based on UTC hour)
    and computes annualised volatility for each.

    Parameters:
        prices: Intraday price series or DataFrame with a
            DatetimeIndex.  For a DataFrame, uses the first column.
            Must have sub-daily frequency (e.g., 1H, 15min).
        sessions: Dictionary mapping session name to (start_hour,
            end_hour) in UTC.  Hours are inclusive of start, exclusive
            of end.  Defaults to the four major sessions:
            Sydney (21-6), Tokyo (0-9), London (7-16), New York (12-21).

    Returns:
        Dictionary mapping session name to annualised volatility
        (assuming 252 trading days).  Higher values indicate more
        volatile sessions.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> idx = pd.date_range('2024-01-01', periods=240, freq='1h')
        >>> prices = pd.Series(np.cumsum(np.random.default_rng(42).normal(0, 0.001, 240)) + 1.10, index=idx)
        >>> vol = volatility_by_session(prices)
        >>> 'London' in vol
        True

    Notes:
        For pairs involving Asian currencies, Tokyo session volatility
        is often the highest.  For EUR and GBP pairs, London dominates.

    See Also:
        wraquant.forex.session.ForexSession: Session definitions.
        wraquant.forex.session.current_session: Active session detection.
    """
    import pandas as pd  # noqa: F811

    if sessions is None:
        sessions = {
            "Sydney": (21, 6),
            "Tokyo": (0, 9),
            "London": (7, 16),
            "New York": (12, 21),
        }

    if isinstance(prices, pd.DataFrame):
        series = prices.iloc[:, 0]
    else:
        series = prices

    returns = series.pct_change().dropna()
    hours = returns.index.hour  # type: ignore[union-attr]

    result: dict[str, float] = {}
    for name, (start_h, end_h) in sessions.items():
        if start_h <= end_h:
            mask = (hours >= start_h) & (hours < end_h)
        else:
            # Crosses midnight
            mask = (hours >= start_h) | (hours < end_h)

        session_returns = returns[mask]
        if len(session_returns) > 1:
            # Annualise: assume number of observations per day from the session
            periods_per_day = max(1, int((end_h - start_h) % 24))
            annualised = float(session_returns.std()) * np.sqrt(
                periods_per_day * 252
            )
            result[name] = annualised
        else:
            result[name] = 0.0

    return result
