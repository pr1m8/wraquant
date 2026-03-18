"""Tests for advanced data cleaning integrations."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

_has_janitor = importlib.util.find_spec("janitor") is not None
_has_rapidfuzz = importlib.util.find_spec("rapidfuzz") is not None
_has_dateparser = importlib.util.find_spec("dateparser") is not None
_has_price_parser = importlib.util.find_spec("price_parser") is not None
_has_country_converter = importlib.util.find_spec("country_converter") is not None
_has_ftfy = importlib.util.find_spec("ftfy") is not None
_has_unidecode = importlib.util.find_spec("unidecode") is not None


class TestJanitorCleanNames:
    @pytest.mark.skipif(not _has_janitor, reason="pyjanitor not installed")
    def test_lowercase_columns(self) -> None:
        from wraquant.data.cleaning_advanced import janitor_clean_names

        df = pd.DataFrame({"First Name": [1], "Last Name": [2], "AGE": [3]})
        result = janitor_clean_names(df)
        for col in result.columns:
            assert col == col.lower()
            assert " " not in col

    @pytest.mark.skipif(not _has_janitor, reason="pyjanitor not installed")
    def test_preserves_data(self) -> None:
        from wraquant.data.cleaning_advanced import janitor_clean_names

        df = pd.DataFrame({"Column A": [10, 20], "Column B": [30, 40]})
        result = janitor_clean_names(df)
        assert result.shape == df.shape
        assert result.iloc[0, 0] == 10


class TestJanitorRemoveEmpty:
    @pytest.mark.skipif(not _has_janitor, reason="pyjanitor not installed")
    def test_removes_empty_rows(self) -> None:
        from wraquant.data.cleaning_advanced import janitor_remove_empty

        df = pd.DataFrame({
            "a": [1, np.nan, 3],
            "b": [4, np.nan, 6],
        })
        result = janitor_remove_empty(df)
        assert len(result) == 2

    @pytest.mark.skipif(not _has_janitor, reason="pyjanitor not installed")
    def test_removes_empty_columns(self) -> None:
        from wraquant.data.cleaning_advanced import janitor_remove_empty

        df = pd.DataFrame({
            "a": [1, 2],
            "b": [np.nan, np.nan],
        })
        result = janitor_remove_empty(df)
        assert "b" not in result.columns


class TestFuzzyMerge:
    @pytest.mark.skipif(not _has_rapidfuzz, reason="rapidfuzz not installed")
    def test_basic_fuzzy_merge(self) -> None:
        from wraquant.data.cleaning_advanced import fuzzy_merge

        df1 = pd.DataFrame({"name": ["Apple Inc", "Microsft", "Googl"]})
        df2 = pd.DataFrame({
            "company": ["Apple Inc.", "Microsoft Corp", "Google LLC"],
            "ticker": ["AAPL", "MSFT", "GOOGL"],
        })
        result = fuzzy_merge(df1, df2, left_col="name", right_col="company", threshold=60)
        assert "match_score" in result.columns
        assert len(result) == 3

    @pytest.mark.skipif(not _has_rapidfuzz, reason="rapidfuzz not installed")
    def test_no_match_below_threshold(self) -> None:
        from wraquant.data.cleaning_advanced import fuzzy_merge

        df1 = pd.DataFrame({"name": ["xyz_nomatch"]})
        df2 = pd.DataFrame({"company": ["Apple"], "val": [1]})
        result = fuzzy_merge(df1, df2, left_col="name", right_col="company", threshold=99)
        assert result["match_score"].iloc[0] == 0.0


class TestParseDatesFlexible:
    @pytest.mark.skipif(not _has_dateparser, reason="dateparser not installed")
    def test_mixed_formats(self) -> None:
        from wraquant.data.cleaning_advanced import parse_dates_flexible

        series = pd.Series(["2023-01-15", "January 15, 2023", "15/01/2023"])
        result = parse_dates_flexible(series)
        assert result.notna().all()
        assert len(result) == 3

    @pytest.mark.skipif(not _has_dateparser, reason="dateparser not installed")
    def test_nan_preserved(self) -> None:
        from wraquant.data.cleaning_advanced import parse_dates_flexible

        series = pd.Series([np.nan, "2023-01-01"])
        result = parse_dates_flexible(series)
        assert pd.isna(result.iloc[0])
        assert pd.notna(result.iloc[1])


class TestParsePrices:
    @pytest.mark.skipif(not _has_price_parser, reason="price-parser not installed")
    def test_usd_prices(self) -> None:
        from wraquant.data.cleaning_advanced import parse_prices

        series = pd.Series(["$1,234.56", "$99.99", "$0.50"])
        result = parse_prices(series)
        assert "amount" in result.columns
        assert "currency" in result.columns
        assert result["amount"].iloc[0] == pytest.approx(1234.56)

    @pytest.mark.skipif(not _has_price_parser, reason="price-parser not installed")
    def test_nan_input(self) -> None:
        from wraquant.data.cleaning_advanced import parse_prices

        series = pd.Series([np.nan, "$10.00"])
        result = parse_prices(series)
        assert pd.isna(result["amount"].iloc[0])
        assert result["amount"].iloc[1] == pytest.approx(10.0)


class TestNormalizeCountries:
    @pytest.mark.skipif(
        not _has_country_converter, reason="country-converter not installed"
    )
    def test_basic_normalization(self) -> None:
        from wraquant.data.cleaning_advanced import normalize_countries

        series = pd.Series(["USA", "United Kingdom", "DE"])
        result = normalize_countries(series)
        assert "name_short" in result.columns
        assert "iso3" in result.columns
        assert "iso2" in result.columns
        assert len(result) == 3


class TestFixText:
    @pytest.mark.skipif(
        not (_has_ftfy and _has_unidecode), reason="ftfy or unidecode not installed"
    )
    def test_ascii_transliteration(self) -> None:
        from wraquant.data.cleaning_advanced import fix_text

        series = pd.Series(["caf\u00e9", "na\u00efve", "r\u00e9sum\u00e9"])
        result = fix_text(series)
        for val in result:
            assert val.isascii()

    @pytest.mark.skipif(
        not (_has_ftfy and _has_unidecode), reason="ftfy or unidecode not installed"
    )
    def test_nan_preserved(self) -> None:
        from wraquant.data.cleaning_advanced import fix_text

        series = pd.Series([np.nan, "hello"])
        result = fix_text(series)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == "hello"
