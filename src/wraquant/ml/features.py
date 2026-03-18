"""Feature engineering utilities for financial machine learning.

All functions in this module use only numpy and pandas -- no external TA
libraries are required.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "rolling_features",
    "return_features",
    "technical_features",
    "volatility_features",
    "microstructure_features",
    "label_fixed_horizon",
    "label_triple_barrier",
]


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------


def rolling_features(
    data: pd.Series | pd.DataFrame,
    windows: Sequence[int] = (5, 10, 21, 63),
) -> pd.DataFrame:
    """Generate rolling statistical features for each window length.

    For every window the following statistics are computed: mean, std,
    skew, kurtosis, min, and max.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Numeric time-series data.  If a DataFrame is passed, features are
        generated independently for each column.
    windows : Sequence[int]
        Rolling-window sizes (default ``(5, 10, 21, 63)``).

    Returns
    -------
    pd.DataFrame
        DataFrame whose columns are named
        ``{col}_{stat}_w{window}`` (or ``{stat}_w{window}`` when *data*
        is a Series).  The number of feature columns equals
        ``n_cols * len(windows) * 6``.
    """
    if isinstance(data, pd.Series):
        data = data.to_frame(name=data.name or "value")
        was_series = True
    else:
        was_series = False

    frames: list[pd.DataFrame] = []
    stats = ["mean", "std", "skew", "kurt", "min", "max"]

    for w in windows:
        roll = data.rolling(window=w, min_periods=w)
        rm = roll.mean()
        rs = roll.std()
        rsk = roll.apply(lambda x: x.skew(), raw=False)
        rk = roll.apply(lambda x: x.kurt(), raw=False)
        rmin = roll.min()
        rmax = roll.max()

        for col in data.columns:
            prefix = f"{col}_" if not was_series else ""
            for stat_name, stat_df in zip(
                stats, [rm, rs, rsk, rk, rmin, rmax], strict=True
            ):
                frames.append(
                    stat_df[[col]].rename(columns={col: f"{prefix}{stat_name}_w{w}"})
                )

    return pd.concat(frames, axis=1)


# ---------------------------------------------------------------------------
# Return-based features
# ---------------------------------------------------------------------------


def return_features(
    prices: pd.Series,
    lags: Sequence[int] = (1, 2, 3, 5, 10, 21),
) -> pd.DataFrame:
    """Compute lagged and cumulative return features from a price series.

    Parameters
    ----------
    prices : pd.Series
        Price series (e.g. adjusted close).
    lags : Sequence[int]
        Lag periods for returns (default ``(1, 2, 3, 5, 10, 21)``).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``ret_lag{l}`` (simple return *l* periods
        ago) and ``cum_ret_{l}`` (cumulative return over the last *l*
        periods) for each lag *l*.
    """
    result: dict[str, pd.Series] = {}

    log_ret = np.log(prices / prices.shift(1))

    for lag in lags:
        # Simple return lagged by *lag* periods
        result[f"ret_lag{lag}"] = log_ret.shift(lag)
        # Cumulative return over *lag* periods
        result[f"cum_ret_{lag}"] = np.log(prices / prices.shift(lag))

    return pd.DataFrame(result, index=prices.index)


# ---------------------------------------------------------------------------
# Technical features (inline, no dependency on ta/ module)
# ---------------------------------------------------------------------------


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _macd_histogram(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.Series:
    """MACD histogram (MACD line minus signal line)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def _bollinger_pctb(
    close: pd.Series, period: int = 20, n_std: float = 2.0
) -> pd.Series:
    """Bollinger Band %B."""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    return (close - lower) / (upper - lower).replace(0, np.nan)


def _atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def technical_features(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute common technical analysis features.

    Computes RSI, MACD histogram, Bollinger Band %B, and ATR.  If
    *volume* is provided, On-Balance Volume (OBV) is also included.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    volume : pd.Series or None
        Trade volume (optional).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``rsi``, ``macd_hist``, ``bb_pctb``,
        ``atr``, and optionally ``obv``.
    """
    result: dict[str, pd.Series] = {
        "rsi": _rsi(close),
        "macd_hist": _macd_histogram(close),
        "bb_pctb": _bollinger_pctb(close),
        "atr": _atr(high, low, close),
    }

    if volume is not None:
        direction = np.sign(close.diff()).fillna(0)
        obv = (direction * volume).cumsum()
        result["obv"] = obv

    return pd.DataFrame(result, index=close.index)


# ---------------------------------------------------------------------------
# Volatility features
# ---------------------------------------------------------------------------


def volatility_features(
    returns: pd.Series,
    windows: Sequence[int] = (5, 10, 21, 63),
) -> pd.DataFrame:
    """Compute realised-volatility-related features.

    Parameters
    ----------
    returns : pd.Series
        Log or simple return series.
    windows : Sequence[int]
        Window sizes for rolling calculations.

    Returns
    -------
    pd.DataFrame
        Columns: ``realized_vol_w{w}``, ``vol_of_vol_w{w}``, and
        ``vol_ratio_w{w1}_w{w2}`` for consecutive window pairs.
    """
    result: dict[str, pd.Series] = {}

    vol_series: dict[int, pd.Series] = {}
    for w in windows:
        rv = returns.rolling(w).std() * np.sqrt(252)
        vol_series[w] = rv
        result[f"realized_vol_w{w}"] = rv
        # Vol-of-vol: rolling std of the rolling vol
        result[f"vol_of_vol_w{w}"] = rv.rolling(w).std()

    sorted_windows = sorted(windows)
    for i in range(len(sorted_windows) - 1):
        w_short = sorted_windows[i]
        w_long = sorted_windows[i + 1]
        denominator = vol_series[w_long].replace(0, np.nan)
        result[f"vol_ratio_w{w_short}_w{w_long}"] = vol_series[w_short] / denominator

    return pd.DataFrame(result, index=returns.index)


# ---------------------------------------------------------------------------
# Microstructure features
# ---------------------------------------------------------------------------


def microstructure_features(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.DataFrame:
    """Compute market-microstructure features.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    volume : pd.Series
        Trade volume.

    Returns
    -------
    pd.DataFrame
        Columns: ``amihud_illiq``, ``kyle_lambda``,
        ``log_volume``, ``volume_ma_ratio``, ``dollar_volume``.
    """
    returns = close.pct_change()
    dollar_volume = close * volume

    # Amihud illiquidity = |return| / dollar volume (rolling 21-day mean)
    amihud = (returns.abs() / dollar_volume.replace(0, np.nan)).rolling(21).mean()

    # Kyle's lambda estimate (rolling regression slope of |price change| on
    # signed sqrt-volume over 21-day windows)
    abs_dp = close.diff().abs()
    signed_sqrt_vol = np.sign(returns) * np.sqrt(volume.abs())

    def _ols_slope(y: np.ndarray, x: np.ndarray) -> float:
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 3:
            return np.nan
        xm = x[mask]
        ym = y[mask]
        xm_dm = xm - xm.mean()
        denom = (xm_dm**2).sum()
        if denom == 0:
            return np.nan
        return float((xm_dm * (ym - ym.mean())).sum() / denom)

    kyle_lambda_vals = np.full(len(close), np.nan)
    dp_arr = abs_dp.values.astype(float)
    sv_arr = signed_sqrt_vol.values.astype(float)
    for i in range(21, len(close)):
        kyle_lambda_vals[i] = _ols_slope(dp_arr[i - 21 : i], sv_arr[i - 21 : i])
    kyle_lambda = pd.Series(kyle_lambda_vals, index=close.index, name="kyle_lambda")

    vol_ma21 = volume.rolling(21).mean().replace(0, np.nan)

    return pd.DataFrame(
        {
            "amihud_illiq": amihud,
            "kyle_lambda": kyle_lambda,
            "log_volume": np.log1p(volume),
            "volume_ma_ratio": volume / vol_ma21,
            "dollar_volume": dollar_volume,
        },
        index=close.index,
    )


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------


def label_fixed_horizon(
    returns: pd.Series,
    horizon: int = 5,
    threshold: float = 0.0,
) -> pd.Series:
    """Label future return direction over a fixed horizon.

    Parameters
    ----------
    returns : pd.Series
        Period (e.g. daily) returns.
    horizon : int
        Number of periods to accumulate forward returns.
    threshold : float
        If ``threshold > 0``, three labels are produced (``1`` / ``0`` /
        ``-1``).  If ``threshold == 0``, binary labels (``1`` / ``0``)
        are produced where ``1`` means positive cumulative return.

    Returns
    -------
    pd.Series
        Integer labels aligned to the original index.  The last
        *horizon* rows will be ``NaN``.
    """
    # For each index i, accumulate returns[i+1] through returns[i+horizon].
    # Use a forward-looking rolling sum.
    fwd_returns = returns.shift(-1)
    cum_fwd = fwd_returns.rolling(window=horizon, min_periods=horizon).sum()
    # Shift so that the value at index i represents the sum of the next
    # *horizon* returns starting from i+1.
    cum_fwd = cum_fwd.shift(-(horizon - 1))

    if threshold > 0:
        labels = pd.Series(
            np.where(
                cum_fwd > threshold,
                1,
                np.where(cum_fwd < -threshold, -1, 0),
            ),
            index=returns.index,
            dtype="Int64",
        )
    else:
        labels = pd.Series(
            np.where(cum_fwd > 0, 1, 0),
            index=returns.index,
            dtype="Int64",
        )

    labels[cum_fwd.isna()] = pd.NA
    return labels


def label_triple_barrier(
    close: pd.Series,
    upper: float | None = None,
    lower: float | None = None,
    max_holding: int = 10,
) -> pd.Series:
    """Triple-barrier labelling (Lopez de Prado).

    For each bar the method sets three barriers:
    * **Upper**: price rises by *upper* fraction  ->  label = 1
    * **Lower**: price falls by *lower* fraction  ->  label = -1
    * **Vertical**: *max_holding* bars elapse     ->  label = sign of return

    If *upper* or *lower* is ``None`` the corresponding horizontal
    barrier is disabled.

    Parameters
    ----------
    close : pd.Series
        Close price series.
    upper : float or None
        Fractional distance for the upper barrier (e.g. ``0.02`` for 2 %).
    lower : float or None
        Fractional distance for the lower barrier (positive value; e.g.
        ``0.02`` for -2 %).
    max_holding : int
        Maximum holding period in bars (vertical barrier).

    Returns
    -------
    pd.Series
        Integer labels in ``{-1, 0, 1}`` aligned to the input index.
        The last *max_holding* entries may be ``NaN``.
    """
    n = len(close)
    labels = pd.Series(np.full(n, np.nan), index=close.index, dtype="Int64")
    close_arr = close.values.astype(float)

    for i in range(n):
        entry = close_arr[i]
        if np.isnan(entry):
            continue
        end = min(i + max_holding, n - 1)
        label: int | None = None

        for j in range(i + 1, end + 1):
            price = close_arr[j]
            ret = (price - entry) / entry

            if upper is not None and ret >= upper:
                label = 1
                break
            if lower is not None and ret <= -lower:
                label = -1
                break

        if label is None:
            # Vertical barrier hit
            if end <= i or i + max_holding > n - 1:
                labels.iloc[i] = pd.NA
                continue
            final_ret = (close_arr[end] - entry) / entry
            if final_ret > 0:
                label = 1
            elif final_ret < 0:
                label = -1
            else:
                label = 0

        labels.iloc[i] = label

    return labels
