"""Advanced data cleaning integrations using optional packages.

Provides wrappers around pyjanitor, rapidfuzz, dateparser,
price-parser, country-converter, ftfy, and unidecode for column
name cleaning, fuzzy merging, flexible date parsing, price parsing,
country normalisation, and text encoding fixes.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from wraquant.core.decorators import requires_extra

__all__ = [
    "janitor_clean_names",
    "janitor_remove_empty",
    "fuzzy_merge",
    "parse_dates_flexible",
    "parse_prices",
    "normalize_countries",
    "fix_text",
]


@requires_extra("cleaning")
def janitor_clean_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame column names using pyjanitor.

    Converts column names to lowercase snake_case, strips whitespace,
    and replaces special characters with underscores.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with messy column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned column names.
    """
    import janitor  # noqa: F401 — registers .clean_names accessor

    return df.clean_names(remove_special=True)


@requires_extra("cleaning")
def janitor_remove_empty(df: pd.DataFrame) -> pd.DataFrame:
    """Remove empty rows and columns using pyjanitor.

    Drops rows and columns that are entirely NaN or empty.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame possibly containing empty rows/columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with empty rows and columns removed.
    """
    import janitor  # noqa: F401

    return df.remove_empty()


@requires_extra("cleaning")
def fuzzy_merge(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    left_col: str,
    right_col: str,
    threshold: float = 80.0,
) -> pd.DataFrame:
    """Merge two DataFrames using fuzzy string matching via rapidfuzz.

    For each value in *left_col* of *df1*, the best match above
    *threshold* in *right_col* of *df2* is found. Matched rows are
    joined; unmatched rows from *df1* are retained with NaN for *df2*
    columns.

    Parameters
    ----------
    df1 : pd.DataFrame
        Left DataFrame.
    df2 : pd.DataFrame
        Right DataFrame.
    left_col : str
        Column name in *df1* to match on.
    right_col : str
        Column name in *df2* to match on.
    threshold : float, default 80.0
        Minimum similarity score (0--100) to consider a match.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with an additional ``match_score`` column
        indicating the similarity score for each matched pair.
    """
    from rapidfuzz import fuzz, process

    right_values = df2[right_col].astype(str).tolist()
    matches: list[dict[str, Any]] = []

    for idx, left_val in df1[left_col].items():
        result = process.extractOne(
            str(left_val),
            right_values,
            scorer=fuzz.WRatio,
            score_cutoff=threshold,
        )
        if result is not None:
            match_str, score, match_idx = result
            matches.append({
                "left_idx": idx,
                "right_idx": df2.index[match_idx],
                "match_score": score,
            })
        else:
            matches.append({
                "left_idx": idx,
                "right_idx": None,
                "match_score": 0.0,
            })

    match_df = pd.DataFrame(matches)

    # Build result: left rows joined with matching right rows
    result = df1.copy()
    result["match_score"] = match_df["match_score"].values

    right_indices = match_df["right_idx"].values
    for col in df2.columns:
        if col == right_col:
            col_name = f"{right_col}_matched"
        else:
            col_name = col if col not in result.columns else f"{col}_right"
        values = []
        for ri in right_indices:
            if ri is not None:
                values.append(df2.loc[ri, col])
            else:
                values.append(None)
        result[col_name] = values

    return result


@requires_extra("cleaning")
def parse_dates_flexible(series: pd.Series) -> pd.Series:
    """Parse mixed-format date strings using dateparser.

    Handles a wide variety of date formats and natural language dates
    (e.g. ``'yesterday'``, ``'3 days ago'``).

    Parameters
    ----------
    series : pd.Series
        Series of date strings in potentially mixed formats.

    Returns
    -------
    pd.Series
        Series of ``datetime`` objects. Values that cannot be parsed
        are set to ``NaT``.
    """
    import dateparser

    def _parse(val: Any) -> Any:
        if pd.isna(val):
            return pd.NaT
        parsed = dateparser.parse(str(val))
        return parsed if parsed is not None else pd.NaT

    return series.apply(_parse)


@requires_extra("cleaning")
def parse_prices(series: pd.Series) -> pd.DataFrame:
    """Parse price strings into numeric amounts and currencies.

    Uses the ``price-parser`` library to extract amounts and currency
    codes from strings like ``'$1,234.56'`` or ``'EUR 99.99'``.

    Parameters
    ----------
    series : pd.Series
        Series of price strings.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:

        * **amount** -- extracted numeric price (float, NaN if unparseable).
        * **currency** -- extracted currency code (str or None).
    """
    from price_parser import Price

    amounts: list[float | None] = []
    currencies: list[str | None] = []

    for val in series:
        if pd.isna(val):
            amounts.append(None)
            currencies.append(None)
            continue
        price = Price.fromstring(str(val))
        amounts.append(float(price.amount) if price.amount is not None else None)
        currencies.append(price.currency)

    return pd.DataFrame(
        {"amount": amounts, "currency": currencies},
        index=series.index,
    )


@requires_extra("cleaning")
def normalize_countries(series: pd.Series) -> pd.DataFrame:
    """Standardise country names and codes using country-converter.

    Parameters
    ----------
    series : pd.Series
        Series of country names, ISO codes, or other country
        identifiers in various formats.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:

        * **name_short** -- standardised short country name.
        * **iso3** -- ISO 3166-1 alpha-3 code.
        * **iso2** -- ISO 3166-1 alpha-2 code.
    """
    import country_converter as coco

    cc = coco.CountryConverter()
    values = series.astype(str).tolist()

    return pd.DataFrame(
        {
            "name_short": cc.convert(values, to="name_short"),
            "iso3": cc.convert(values, to="ISO3"),
            "iso2": cc.convert(values, to="ISO2"),
        },
        index=series.index,
    )


@requires_extra("cleaning")
def fix_text(series: pd.Series) -> pd.Series:
    """Fix text encoding issues using ftfy and unidecode.

    Repairs mojibake, normalises Unicode, and transliterates non-ASCII
    characters to their closest ASCII equivalents.

    Parameters
    ----------
    series : pd.Series
        Series of strings that may contain encoding artefacts.

    Returns
    -------
    pd.Series
        Series with fixed text encoding. NaN values are preserved.
    """
    import ftfy
    from unidecode import unidecode

    def _fix(val: Any) -> Any:
        if pd.isna(val):
            return val
        text = str(val)
        text = ftfy.fix_text(text)
        text = unidecode(text)
        return text

    return series.apply(_fix)
