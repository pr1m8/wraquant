"""Tests for forex pair handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.core.types import Currency
from wraquant.forex.pairs import (
    CurrencyPair,
    correlation_matrix,
    cross_rate,
    currency_strength,
    major_pairs,
    volatility_by_session,
)


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


class TestCorrelationMatrix:
    def test_shape(self) -> None:
        rng = np.random.default_rng(42)
        prices = pd.DataFrame({
            "EURUSD": np.cumsum(rng.normal(0, 0.001, 100)) + 1.10,
            "GBPUSD": np.cumsum(rng.normal(0, 0.001, 100)) + 1.30,
            "USDJPY": np.cumsum(rng.normal(0, 0.001, 100)) + 110.0,
        })
        corr = correlation_matrix(prices, window=30)
        assert corr.shape == (3, 3)

    def test_diagonal_is_one(self) -> None:
        rng = np.random.default_rng(42)
        prices = pd.DataFrame({
            "EURUSD": np.cumsum(rng.normal(0, 0.001, 100)) + 1.10,
            "GBPUSD": np.cumsum(rng.normal(0, 0.001, 100)) + 1.30,
        })
        corr = correlation_matrix(prices, window=30)
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-10)

    def test_symmetric(self) -> None:
        rng = np.random.default_rng(42)
        prices = pd.DataFrame({
            "EURUSD": np.cumsum(rng.normal(0, 0.001, 100)) + 1.10,
            "GBPUSD": np.cumsum(rng.normal(0, 0.001, 100)) + 1.30,
        })
        corr = correlation_matrix(prices, window=30)
        np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-10)

    def test_short_data_fallback(self) -> None:
        rng = np.random.default_rng(42)
        prices = pd.DataFrame({
            "EURUSD": np.cumsum(rng.normal(0, 0.001, 10)) + 1.10,
            "GBPUSD": np.cumsum(rng.normal(0, 0.001, 10)) + 1.30,
        })
        corr = correlation_matrix(prices, window=60)
        assert corr.shape == (2, 2)


class TestCurrencyStrength:
    def test_returns_series(self) -> None:
        rng = np.random.default_rng(42)
        prices = pd.DataFrame({
            "EURUSD": np.cumsum(rng.normal(0.0001, 0.001, 100)) + 1.10,
            "USDJPY": np.cumsum(rng.normal(0.0001, 0.001, 100)) + 110.0,
        })
        strength = currency_strength(prices)
        assert isinstance(strength, pd.Series)

    def test_contains_currencies(self) -> None:
        rng = np.random.default_rng(42)
        prices = pd.DataFrame({
            "EURUSD": np.cumsum(rng.normal(0.0001, 0.001, 100)) + 1.10,
            "USDJPY": np.cumsum(rng.normal(0.0001, 0.001, 100)) + 110.0,
        })
        strength = currency_strength(prices)
        assert "EUR" in strength.index
        assert "USD" in strength.index
        assert "JPY" in strength.index

    def test_sorted_descending(self) -> None:
        rng = np.random.default_rng(42)
        prices = pd.DataFrame({
            "EURUSD": np.cumsum(rng.normal(0.001, 0.001, 100)) + 1.10,
            "GBPUSD": np.cumsum(rng.normal(-0.001, 0.001, 100)) + 1.30,
        })
        strength = currency_strength(prices)
        values = strength.values
        assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))

    def test_with_window(self) -> None:
        rng = np.random.default_rng(42)
        prices = pd.DataFrame({
            "EURUSD": np.cumsum(rng.normal(0, 0.001, 100)) + 1.10,
        })
        strength = currency_strength(prices, window=20)
        assert len(strength) == 2  # EUR and USD


class TestVolatilityBySession:
    def test_default_sessions(self) -> None:
        idx = pd.date_range("2024-01-01", periods=480, freq="1h")
        rng = np.random.default_rng(42)
        prices = pd.Series(
            np.cumsum(rng.normal(0, 0.001, 480)) + 1.10, index=idx
        )
        vol = volatility_by_session(prices)
        assert "London" in vol
        assert "New York" in vol
        assert "Tokyo" in vol
        assert "Sydney" in vol

    def test_all_positive(self) -> None:
        idx = pd.date_range("2024-01-01", periods=480, freq="1h")
        rng = np.random.default_rng(42)
        prices = pd.Series(
            np.cumsum(rng.normal(0, 0.001, 480)) + 1.10, index=idx
        )
        vol = volatility_by_session(prices)
        for name, v in vol.items():
            assert v >= 0.0, f"Volatility for {name} should be non-negative"

    def test_custom_sessions(self) -> None:
        idx = pd.date_range("2024-01-01", periods=480, freq="1h")
        rng = np.random.default_rng(42)
        prices = pd.Series(
            np.cumsum(rng.normal(0, 0.001, 480)) + 1.10, index=idx
        )
        custom = {"Morning": (8, 12), "Afternoon": (12, 17)}
        vol = volatility_by_session(prices, sessions=custom)
        assert "Morning" in vol
        assert "Afternoon" in vol

    def test_dataframe_input(self) -> None:
        idx = pd.date_range("2024-01-01", periods=480, freq="1h")
        rng = np.random.default_rng(42)
        prices = pd.DataFrame({
            "EURUSD": np.cumsum(rng.normal(0, 0.001, 480)) + 1.10,
        }, index=idx)
        vol = volatility_by_session(prices)
        assert "London" in vol
