"""Financial time series types built on pandas.

Provides PriceSeries, ReturnSeries, OHLCVFrame, and ReturnFrame --
pd.Series/pd.DataFrame subclasses that carry financial metadata
(frequency, currency, return_type) and offer domain-specific methods
(to_returns, to_prices, sharpe, annualized_vol, correlation, covariance).

These types are the foundation of wraquant's type system.  Every module
can accept and return them; downstream code can query metadata without
parsing column names or guessing conventions.

Design decisions:
    - Subclass pd.Series/pd.DataFrame rather than wrap them.  This means
      every pandas method works out of the box; we only *add* behaviour.
    - ``_metadata`` ensures custom attributes survive slicing, groupby,
      and other pandas operations that construct new objects internally.
    - Frequency detection is automatic from DatetimeIndex but can be
      overridden at construction time.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd


# ---------------------------------------------------------------------------
# Legacy protocols (kept for backwards compatibility)
# ---------------------------------------------------------------------------


@runtime_checkable
class AbstractSeries(Protocol):
    """Protocol for a 1-D labeled array of financial data."""

    @property
    def name(self) -> str | None: ...

    @property
    def dtype(self) -> Any: ...

    def __len__(self) -> int: ...

    def to_numpy(self) -> npt.NDArray[np.floating]: ...

    def to_pandas(self) -> Any: ...

    def to_polars(self) -> Any: ...


@runtime_checkable
class AbstractFrame(Protocol):
    """Protocol for a 2-D labeled table of financial data."""

    @property
    def columns(self) -> list[str]: ...

    @property
    def shape(self) -> tuple[int, int]: ...

    def __len__(self) -> int: ...

    def to_numpy(self) -> npt.NDArray[np.floating]: ...

    def to_pandas(self) -> Any: ...

    def to_polars(self) -> Any: ...


# ---------------------------------------------------------------------------
# Frequency detection
# ---------------------------------------------------------------------------

_FREQ_MAP: dict[str, tuple[str, int]] = {
    # pandas offset alias -> (human label, periods_per_year)
    "B": ("daily", 252),
    "D": ("daily", 252),
    "C": ("daily", 252),
    "W": ("weekly", 52),
    "M": ("monthly", 12),
    "MS": ("monthly", 12),
    "ME": ("monthly", 12),
    "BM": ("monthly", 12),
    "BMS": ("monthly", 12),
    "BME": ("monthly", 12),
    "Q": ("quarterly", 4),
    "QS": ("quarterly", 4),
    "QE": ("quarterly", 4),
    "BQ": ("quarterly", 4),
    "BQS": ("quarterly", 4),
    "BQE": ("quarterly", 4),
    "Y": ("yearly", 1),
    "A": ("yearly", 1),
    "YS": ("yearly", 1),
    "YE": ("yearly", 1),
    "AS": ("yearly", 1),
    "BA": ("yearly", 1),
    "BAS": ("yearly", 1),
    "BAE": ("yearly", 1),
    "h": ("hourly", 252 * 7),
    "H": ("hourly", 252 * 7),
    "min": ("minute", 252 * 7 * 60),
    "T": ("minute", 252 * 7 * 60),
    "s": ("second", 252 * 7 * 3600),
    "S": ("second", 252 * 7 * 3600),
}


def _detect_frequency(index: pd.Index) -> tuple[str, int]:
    """Detect frequency from a DatetimeIndex.

    Uses pd.infer_freq, falling back to median-delta heuristics.

    Returns:
        Tuple of (frequency_label, periods_per_year).
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return ("unknown", 252)

    freq = pd.infer_freq(index)
    if freq is not None:
        # Strip any leading digits (e.g. "5T" -> "T")
        alpha = "".join(c for c in freq if c.isalpha())
        if alpha in _FREQ_MAP:
            return _FREQ_MAP[alpha]

    # Fallback: median timedelta
    deltas = np.diff(index.values).astype("timedelta64[s]").astype(float)
    median_seconds = float(np.median(deltas))

    if median_seconds < 120:
        return ("second", 252 * 7 * 3600)
    if median_seconds < 7200:
        return ("minute", 252 * 7 * 60)
    if median_seconds < 18 * 3600:
        return ("hourly", 252 * 7)
    if median_seconds < 5 * 86400:
        return ("daily", 252)
    if median_seconds < 20 * 86400:
        return ("weekly", 52)
    if median_seconds < 100 * 86400:
        return ("monthly", 12)
    if median_seconds < 200 * 86400:
        return ("quarterly", 4)
    return ("yearly", 1)


# ---------------------------------------------------------------------------
# PriceSeries
# ---------------------------------------------------------------------------


class PriceSeries(pd.Series):
    """A pandas Series of asset prices with financial metadata.

    Carries ``frequency`` (auto-detected or explicit) and ``currency``
    through all pandas operations.  Provides ``to_returns()`` for
    conversion to ReturnSeries, and ``periods_per_year`` for
    annualisation.

    Parameters:
        *args: Forwarded to pd.Series.
        frequency: Override auto-detected frequency (e.g. "daily").
        currency: ISO currency code (e.g. "USD").
        **kwargs: Forwarded to pd.Series.

    Example:
        >>> import pandas as pd
        >>> idx = pd.bdate_range("2023-01-02", periods=5)
        >>> p = PriceSeries([100, 101, 99, 102, 103], index=idx)
        >>> p.frequency
        'daily'
        >>> p.periods_per_year
        252
        >>> r = p.to_returns()
        >>> isinstance(r, ReturnSeries)
        True
    """

    _metadata = ["_frequency", "_periods_per_year", "_currency"]

    def __init__(
        self,
        *args: Any,
        frequency: str | None = None,
        currency: str = "USD",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if frequency is not None:
            self._frequency = frequency
            # Look up periods_per_year from the freq map
            ppy = 252
            for _alias, (label, ppy_val) in _FREQ_MAP.items():
                if label == frequency:
                    ppy = ppy_val
                    break
            self._periods_per_year = ppy
        else:
            label, ppy = _detect_frequency(self.index)
            self._frequency = label
            self._periods_per_year = ppy
        self._currency = currency

    @property
    def _constructor(self) -> type:
        return PriceSeries

    @property
    def frequency(self) -> str:
        """Detected or explicit time series frequency."""
        return getattr(self, "_frequency", "unknown")

    @property
    def periods_per_year(self) -> int:
        """Number of periods per year for annualisation."""
        return getattr(self, "_periods_per_year", 252)

    @property
    def currency(self) -> str:
        """ISO currency code."""
        return getattr(self, "_currency", "USD")

    def to_returns(self, method: str = "simple") -> ReturnSeries:
        """Convert prices to returns.

        Parameters:
            method: ``"simple"`` for arithmetic returns (P_t/P_{t-1} - 1),
                ``"log"`` for log returns (ln(P_t/P_{t-1})).

        Returns:
            ReturnSeries with the same metadata.
        """
        if method == "log":
            ret = np.log(self / self.shift(1))
        else:
            ret = self.pct_change()
        ret = ret.iloc[1:]  # drop first NaN
        result = ReturnSeries(
            ret,
            frequency=self.frequency,
            currency=self.currency,
            return_type=method,
        )
        result._periods_per_year = self.periods_per_year
        return result


# ---------------------------------------------------------------------------
# ReturnSeries
# ---------------------------------------------------------------------------


class ReturnSeries(pd.Series):
    """A pandas Series of asset returns with financial metadata.

    Carries ``frequency``, ``currency``, and ``return_type`` through
    all pandas operations.  Provides convenience analytics
    (``sharpe``, ``annualized_vol``, ``annualized_return``) and
    ``to_prices()`` for conversion back to PriceSeries.

    Parameters:
        *args: Forwarded to pd.Series.
        frequency: Override auto-detected frequency.
        currency: ISO currency code.
        return_type: ``"simple"`` or ``"log"``.
        **kwargs: Forwarded to pd.Series.

    Example:
        >>> import pandas as pd, numpy as np
        >>> idx = pd.bdate_range("2023-01-03", periods=4)
        >>> r = ReturnSeries([0.01, -0.02, 0.03, 0.01], index=idx)
        >>> r.frequency
        'daily'
        >>> isinstance(r.annualized_vol(), float)
        True
    """

    _metadata = [
        "_frequency",
        "_periods_per_year",
        "_currency",
        "_return_type",
    ]

    def __init__(
        self,
        *args: Any,
        frequency: str | None = None,
        currency: str = "USD",
        return_type: str = "simple",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if frequency is not None:
            self._frequency = frequency
            ppy = 252
            for _alias, (label, ppy_val) in _FREQ_MAP.items():
                if label == frequency:
                    ppy = ppy_val
                    break
            self._periods_per_year = ppy
        else:
            label, ppy = _detect_frequency(self.index)
            self._frequency = label
            self._periods_per_year = ppy
        self._currency = currency
        self._return_type = return_type

    @property
    def _constructor(self) -> type:
        return ReturnSeries

    @property
    def frequency(self) -> str:
        """Detected or explicit time series frequency."""
        return getattr(self, "_frequency", "unknown")

    @property
    def periods_per_year(self) -> int:
        """Number of periods per year for annualisation."""
        return getattr(self, "_periods_per_year", 252)

    @property
    def currency(self) -> str:
        """ISO currency code."""
        return getattr(self, "_currency", "USD")

    @property
    def return_type(self) -> str:
        """Return computation method: 'simple' or 'log'."""
        return getattr(self, "_return_type", "simple")

    def to_prices(self, initial_price: float = 100.0) -> PriceSeries:
        """Convert returns back to a price series.

        Parameters:
            initial_price: Starting price level.

        Returns:
            PriceSeries starting at ``initial_price``.
        """
        if self.return_type == "log":
            cum = np.exp(self.cumsum())
        else:
            cum = (1 + self).cumprod()
        prices = cum * initial_price
        result = PriceSeries(
            prices,
            frequency=self.frequency,
            currency=self.currency,
        )
        result._periods_per_year = self.periods_per_year
        return result

    def sharpe(self, risk_free: float = 0.0) -> float:
        """Annualized Sharpe ratio.

        Parameters:
            risk_free: Annual risk-free rate (e.g. 0.05 for 5%).

        Returns:
            Annualized Sharpe ratio.
        """
        excess = self - risk_free / self.periods_per_year
        std = excess.std()
        if std == 0 or np.isnan(std):
            return 0.0
        return float(excess.mean() / std * np.sqrt(self.periods_per_year))

    def annualized_vol(self) -> float:
        """Annualized volatility (standard deviation * sqrt(N))."""
        return float(self.std() * np.sqrt(self.periods_per_year))

    def annualized_return(self) -> float:
        """Annualized compound return.

        Computes geometric mean return annualised to ``periods_per_year``.
        """
        if len(self) == 0:
            return 0.0
        if self.return_type == "log":
            return float(self.mean() * self.periods_per_year)
        total = float((1 + self).prod())
        n_years = len(self) / self.periods_per_year
        if n_years <= 0 or total <= 0:
            return 0.0
        return float(total ** (1.0 / n_years) - 1)


# ---------------------------------------------------------------------------
# OHLCVFrame
# ---------------------------------------------------------------------------


class OHLCVFrame(pd.DataFrame):
    """A DataFrame with open/high/low/close/volume columns.

    Provides typed accessors for each OHLCV column, returning
    PriceSeries to preserve frequency and currency metadata.

    Column names are case-insensitive: ``Open``, ``open``, ``OPEN``
    all resolve correctly.

    Parameters:
        *args: Forwarded to pd.DataFrame.
        frequency: Override auto-detected frequency.
        currency: ISO currency code.
        **kwargs: Forwarded to pd.DataFrame.

    Example:
        >>> import pandas as pd, numpy as np
        >>> idx = pd.bdate_range("2023-01-02", periods=3)
        >>> df = OHLCVFrame({
        ...     "open": [100, 101, 99],
        ...     "high": [102, 103, 101],
        ...     "low": [99, 100, 98],
        ...     "close": [101, 99, 100],
        ...     "volume": [1000, 1200, 900],
        ... }, index=idx)
        >>> isinstance(df.close, PriceSeries)
        True
    """

    _metadata = ["_frequency", "_periods_per_year", "_currency"]

    def __init__(
        self,
        *args: Any,
        frequency: str | None = None,
        currency: str = "USD",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if frequency is not None:
            self._frequency = frequency
            ppy = 252
            for _alias, (label, ppy_val) in _FREQ_MAP.items():
                if label == frequency:
                    ppy = ppy_val
                    break
            self._periods_per_year = ppy
        else:
            label, ppy = _detect_frequency(self.index)
            self._frequency = label
            self._periods_per_year = ppy
        self._currency = currency

    @property
    def _constructor(self) -> type:
        return OHLCVFrame

    @property
    def frequency(self) -> str:
        """Detected or explicit time series frequency."""
        return getattr(self, "_frequency", "unknown")

    @property
    def periods_per_year(self) -> int:
        """Number of periods per year for annualisation."""
        return getattr(self, "_periods_per_year", 252)

    @property
    def currency(self) -> str:
        """ISO currency code."""
        return getattr(self, "_currency", "USD")

    def _find_col(self, name: str) -> str:
        """Find a column by case-insensitive match."""
        lower = name.lower()
        for col in self.columns:
            if col.lower() == lower:
                return col
        raise KeyError(f"No '{name}' column found. Available: {list(self.columns)}")

    @property
    def close(self) -> PriceSeries:
        """Close prices as a PriceSeries."""
        col = self._find_col("close")
        return PriceSeries(
            self[col],
            frequency=self.frequency,
            currency=self.currency,
        )

    @property
    def open(self) -> PriceSeries:
        """Open prices as a PriceSeries."""
        col = self._find_col("open")
        return PriceSeries(
            self[col],
            frequency=self.frequency,
            currency=self.currency,
        )

    @property
    def high(self) -> PriceSeries:
        """High prices as a PriceSeries."""
        col = self._find_col("high")
        return PriceSeries(
            self[col],
            frequency=self.frequency,
            currency=self.currency,
        )

    @property
    def low(self) -> PriceSeries:
        """Low prices as a PriceSeries."""
        col = self._find_col("low")
        return PriceSeries(
            self[col],
            frequency=self.frequency,
            currency=self.currency,
        )

    @property
    def volume(self) -> pd.Series:
        """Volume as a plain pd.Series."""
        col = self._find_col("volume")
        return self[col]


# ---------------------------------------------------------------------------
# ReturnFrame
# ---------------------------------------------------------------------------


class ReturnFrame(pd.DataFrame):
    """A DataFrame of multi-asset returns with financial metadata.

    Provides ``correlation()`` and ``covariance()`` with optional
    annualisation for portfolio construction.

    Parameters:
        *args: Forwarded to pd.DataFrame.
        frequency: Override auto-detected frequency.
        currency: ISO currency code.
        return_type: ``"simple"`` or ``"log"``.
        **kwargs: Forwarded to pd.DataFrame.

    Example:
        >>> import pandas as pd, numpy as np
        >>> idx = pd.bdate_range("2023-01-03", periods=100)
        >>> np.random.seed(42)
        >>> rf = ReturnFrame({
        ...     "A": np.random.normal(0.001, 0.01, 100),
        ...     "B": np.random.normal(0.0005, 0.015, 100),
        ... }, index=idx)
        >>> rf.correlation().shape
        (2, 2)
    """

    _metadata = [
        "_frequency",
        "_periods_per_year",
        "_currency",
        "_return_type",
    ]

    def __init__(
        self,
        *args: Any,
        frequency: str | None = None,
        currency: str = "USD",
        return_type: str = "simple",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if frequency is not None:
            self._frequency = frequency
            ppy = 252
            for _alias, (label, ppy_val) in _FREQ_MAP.items():
                if label == frequency:
                    ppy = ppy_val
                    break
            self._periods_per_year = ppy
        else:
            label, ppy = _detect_frequency(self.index)
            self._frequency = label
            self._periods_per_year = ppy
        self._currency = currency
        self._return_type = return_type

    @property
    def _constructor(self) -> type:
        return ReturnFrame

    @property
    def frequency(self) -> str:
        """Detected or explicit time series frequency."""
        return getattr(self, "_frequency", "unknown")

    @property
    def periods_per_year(self) -> int:
        """Number of periods per year for annualisation."""
        return getattr(self, "_periods_per_year", 252)

    @property
    def currency(self) -> str:
        """ISO currency code."""
        return getattr(self, "_currency", "USD")

    @property
    def return_type(self) -> str:
        """Return computation method: 'simple' or 'log'."""
        return getattr(self, "_return_type", "simple")

    def correlation(self) -> pd.DataFrame:
        """Pairwise return correlation matrix.

        Returns:
            DataFrame of correlations (N x N).
        """
        return self.corr()

    def covariance(self, annualize: bool = False) -> pd.DataFrame:
        """Pairwise return covariance matrix.

        Parameters:
            annualize: If True, multiply by ``periods_per_year``.

        Returns:
            DataFrame of covariances (N x N).
        """
        cov = self.cov()
        if annualize:
            cov = cov * self.periods_per_year
        return cov
