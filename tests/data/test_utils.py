"""Tests for data utility functions."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from wraquant.data.utils import clean_symbol, infer_frequency, parse_date


class TestParseDate:
    def test_string(self) -> None:
        result = parse_date("2020-01-01")
        assert isinstance(result, pd.Timestamp)
        assert result.year == 2020

    def test_date(self) -> None:
        result = parse_date(date(2020, 6, 15))
        assert result == pd.Timestamp("2020-06-15")

    def test_datetime(self) -> None:
        result = parse_date(datetime(2020, 6, 15, 12, 30))
        assert result == pd.Timestamp("2020-06-15 12:30:00")

    def test_timestamp(self) -> None:
        ts = pd.Timestamp("2020-01-01")
        assert parse_date(ts) is ts

    def test_numpy_datetime(self) -> None:
        dt = np.datetime64("2020-01-01")
        result = parse_date(dt)
        assert isinstance(result, pd.Timestamp)

    def test_none(self) -> None:
        assert parse_date(None) is None

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError):
            parse_date(12345)


class TestCleanSymbol:
    def test_uppercase(self) -> None:
        assert clean_symbol("aapl") == "AAPL"

    def test_strip_whitespace(self) -> None:
        assert clean_symbol("  AAPL  ") == "AAPL"

    def test_forex_pair(self) -> None:
        assert clean_symbol("eurusd=x") == "EURUSD=X"


class TestInferFrequency:
    def test_daily(self) -> None:
        idx = pd.bdate_range("2020-01-01", periods=20)
        freq = infer_frequency(idx)
        assert freq is not None

    def test_too_short(self) -> None:
        idx = pd.DatetimeIndex(["2020-01-01"])
        assert infer_frequency(idx) is None
