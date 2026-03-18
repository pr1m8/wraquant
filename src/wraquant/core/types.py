"""Type definitions, enums, and type aliases for wraquant.

Provides strongly-typed enumerations for financial concepts and
type aliases used throughout the package.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from enum import StrEnum
from typing import TypeAlias, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

# ---------------------------------------------------------------------------
# Core type aliases
# ---------------------------------------------------------------------------

DateLike: TypeAlias = Union[str, date, datetime, pd.Timestamp, np.datetime64]
"""Anything that can be interpreted as a date."""

ArrayLike: TypeAlias = Union[
    npt.NDArray[np.floating], pd.Series, Sequence[float], list[float]
]
"""Numeric array-like input accepted by most functions."""

PriceSeries: TypeAlias = pd.Series
"""A pandas Series of prices indexed by datetime."""

ReturnSeries: TypeAlias = pd.Series
"""A pandas Series of returns indexed by datetime."""

PriceFrame: TypeAlias = pd.DataFrame
"""A DataFrame of prices with datetime index and asset columns."""

ReturnFrame: TypeAlias = pd.DataFrame
"""A DataFrame of returns with datetime index and asset columns."""

OHLCVFrame: TypeAlias = pd.DataFrame
"""A DataFrame with open/high/low/close/volume columns and datetime index."""

WeightsArray: TypeAlias = Union[npt.NDArray[np.floating], pd.Series, Sequence[float]]
"""Portfolio weight vector."""

CovarianceMatrix: TypeAlias = Union[npt.NDArray[np.floating], pd.DataFrame]
"""Covariance matrix (2D)."""


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Frequency(StrEnum):
    """Time series frequency."""

    TICK = "tick"
    SECOND = "1s"
    MINUTE = "1min"
    FIVE_MIN = "5min"
    FIFTEEN_MIN = "15min"
    THIRTY_MIN = "30min"
    HOURLY = "1h"
    FOUR_HOUR = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1mo"
    QUARTERLY = "1q"
    YEARLY = "1y"


class AssetClass(StrEnum):
    """Financial asset classes."""

    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    FX = "fx"
    CRYPTO = "crypto"
    DERIVATIVE = "derivative"
    INDEX = "index"
    ETF = "etf"
    FUND = "fund"
    REAL_ESTATE = "real_estate"
    ALTERNATIVE = "alternative"


class Currency(StrEnum):
    """Major world currencies (ISO 4217)."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CHF = "CHF"
    AUD = "AUD"
    CAD = "CAD"
    NZD = "NZD"
    SEK = "SEK"
    NOK = "NOK"
    DKK = "DKK"
    SGD = "SGD"
    HKD = "HKD"
    CNY = "CNY"
    CNH = "CNH"
    INR = "INR"
    KRW = "KRW"
    BRL = "BRL"
    MXN = "MXN"
    ZAR = "ZAR"
    TRY = "TRY"
    PLN = "PLN"
    CZK = "CZK"
    HUF = "HUF"
    THB = "THB"
    TWD = "TWD"
    RUB = "RUB"


class ReturnType(StrEnum):
    """Type of return calculation."""

    SIMPLE = "simple"
    LOG = "log"
    EXCESS = "excess"


class OptionType(StrEnum):
    """Option contract type."""

    CALL = "call"
    PUT = "put"


class OptionStyle(StrEnum):
    """Option exercise style."""

    EUROPEAN = "european"
    AMERICAN = "american"
    BERMUDAN = "bermudan"
    ASIAN = "asian"


class OrderSide(StrEnum):
    """Trade order direction."""

    BUY = "buy"
    SELL = "sell"


class RegimeState(StrEnum):
    """Market regime states."""

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    CRISIS = "crisis"


class RiskMeasure(StrEnum):
    """Risk measure types."""

    VAR = "var"
    CVAR = "cvar"
    VOLATILITY = "volatility"
    MAX_DRAWDOWN = "max_drawdown"
    SEMI_DEVIATION = "semi_deviation"
    ENTROPIC = "entropic"


class VolModel(StrEnum):
    """Volatility model types."""

    GARCH = "garch"
    EGARCH = "egarch"
    GJR_GARCH = "gjr_garch"
    TARCH = "tarch"
    EWMA = "ewma"
    REALIZED = "realized"
    IMPLIED = "implied"
    HESTON = "heston"
    SABR = "sabr"
