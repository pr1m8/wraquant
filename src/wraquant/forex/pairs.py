"""Currency pair definitions and cross rate calculations."""

from __future__ import annotations

from dataclasses import dataclass

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
    """Calculate a cross rate from two pairs.

    Parameters:
        pair1_rate: Rate for first pair.
        pair2_rate: Rate for second pair.
        method: 'divide' (pair1/pair2) or 'multiply' (pair1*pair2).

    Returns:
        Cross rate.

    Example:
        >>> cross_rate(1.1000, 110.00, method="multiply")  # EURJPY from EURUSD * USDJPY
        121.0
    """
    if method == "multiply":
        return pair1_rate * pair2_rate
    return pair1_rate / pair2_rate
