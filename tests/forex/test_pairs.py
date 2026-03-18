"""Tests for forex pair handling."""

from __future__ import annotations

import pytest

from wraquant.core.types import Currency
from wraquant.forex.pairs import CurrencyPair, cross_rate, major_pairs


class TestCurrencyPair:
    def test_symbol(self) -> None:
        pair = CurrencyPair(Currency.EUR, Currency.USD)
        assert pair.symbol == "EURUSD"

    def test_yahoo_symbol(self) -> None:
        pair = CurrencyPair(Currency.EUR, Currency.USD)
        assert pair.yahoo_symbol == "EURUSD=X"

    def test_jpy_pair(self) -> None:
        pair = CurrencyPair(Currency.USD, Currency.JPY)
        assert pair.is_jpy_pair
        assert pair.pip_size == 0.01

    def test_non_jpy_pair(self) -> None:
        pair = CurrencyPair(Currency.EUR, Currency.USD)
        assert not pair.is_jpy_pair
        assert pair.pip_size == 0.0001

    def test_inverse(self) -> None:
        pair = CurrencyPair(Currency.EUR, Currency.USD)
        inv = pair.inverse()
        assert inv.base == Currency.USD
        assert inv.quote == Currency.EUR

    def test_from_string(self) -> None:
        pair = CurrencyPair.from_string("EURUSD")
        assert pair.base == Currency.EUR
        assert pair.quote == Currency.USD

    def test_from_string_with_slash(self) -> None:
        pair = CurrencyPair.from_string("EUR/USD")
        assert pair.symbol == "EURUSD"

    def test_from_string_invalid(self) -> None:
        with pytest.raises(ValueError):
            CurrencyPair.from_string("EU")


class TestMajorPairs:
    def test_seven_majors(self) -> None:
        pairs = major_pairs()
        assert len(pairs) == 7

    def test_eurusd_in_majors(self) -> None:
        pairs = major_pairs()
        symbols = [p.symbol for p in pairs]
        assert "EURUSD" in symbols


class TestCrossRate:
    def test_divide(self) -> None:
        rate = cross_rate(1.1000, 0.8500)
        assert abs(rate - 1.1000 / 0.8500) < 1e-10

    def test_multiply(self) -> None:
        rate = cross_rate(1.1000, 110.00, method="multiply")
        assert abs(rate - 121.0) < 1e-10
