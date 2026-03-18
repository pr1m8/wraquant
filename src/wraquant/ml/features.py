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
    "interaction_features",
    "cross_asset_features",
    "regime_features",
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


# ---------------------------------------------------------------------------
# Interaction features
# ---------------------------------------------------------------------------


def interaction_features(
    data: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Create pairwise interaction terms between features.

    For each pair of selected columns ``(A, B)``, computes:
    - ``A_x_B``: element-wise product (captures multiplicative relationships)
    - ``A_div_B``: element-wise ratio A / B (captures relative magnitudes)

    Parameters
    ----------
    data : pd.DataFrame
        Feature DataFrame.
    columns : Sequence[str] or None
        Columns to use for interaction terms. If None, all columns are used.

    Returns
    -------
    pd.DataFrame
        DataFrame containing all pairwise interaction features, with column
        names like ``col1_x_col2`` and ``col1_div_col2``.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
    >>> result = interaction_features(df, columns=['a', 'b'])
    >>> 'a_x_b' in result.columns
    True
    >>> 'a_div_b' in result.columns
    True
    """
    from itertools import combinations as _combinations

    if columns is None:
        columns = list(data.columns)

    result: dict[str, pd.Series] = {}

    for col_a, col_b in _combinations(columns, 2):
        result[f"{col_a}_x_{col_b}"] = data[col_a] * data[col_b]
        denominator = data[col_b].replace(0, np.nan)
        result[f"{col_a}_div_{col_b}"] = data[col_a] / denominator

    return pd.DataFrame(result, index=data.index)


# ---------------------------------------------------------------------------
# Cross-asset features
# ---------------------------------------------------------------------------


def cross_asset_features(
    asset: pd.Series,
    benchmark: pd.Series,
    windows: Sequence[int] = (10, 21, 63),
) -> pd.DataFrame:
    """Compute cross-asset relationship features.

    Given an asset return series and a benchmark (or related asset) return
    series, computes rolling correlation, rolling beta, and relative
    strength for each window.

    Parameters
    ----------
    asset : pd.Series
        Return series for the asset of interest.
    benchmark : pd.Series
        Return series for the benchmark or related asset.
    windows : Sequence[int]
        Rolling window sizes for correlation and beta calculations.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - ``rolling_corr_w{w}``: rolling Pearson correlation
        - ``rolling_beta_w{w}``: rolling OLS beta (cov / var of benchmark)
        - ``relative_strength_w{w}``: cumulative return ratio (asset / benchmark)
          over the window

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(0)
    >>> asset = pd.Series(np.random.randn(200) * 0.01, name='asset')
    >>> bench = pd.Series(np.random.randn(200) * 0.01, name='bench')
    >>> result = cross_asset_features(asset, bench, windows=[10, 21])
    >>> 'rolling_corr_w10' in result.columns
    True
    >>> 'rolling_beta_w21' in result.columns
    True
    """
    aligned = pd.DataFrame({"asset": asset, "benchmark": benchmark}).dropna()
    a = aligned["asset"]
    b = aligned["benchmark"]

    result: dict[str, pd.Series] = {}

    for w in windows:
        # Rolling correlation
        result[f"rolling_corr_w{w}"] = a.rolling(w).corr(b)

        # Rolling beta = cov(asset, benchmark) / var(benchmark)
        cov = a.rolling(w).cov(b)
        var = b.rolling(w).var()
        result[f"rolling_beta_w{w}"] = cov / var.replace(0, np.nan)

        # Relative strength: cumulative return of asset vs benchmark
        cum_asset = (1 + a).rolling(w).apply(np.prod, raw=True)
        cum_bench = (1 + b).rolling(w).apply(np.prod, raw=True)
        result[f"relative_strength_w{w}"] = cum_asset / cum_bench.replace(
            0, np.nan
        )

    return pd.DataFrame(result, index=aligned.index)


# ---------------------------------------------------------------------------
# Regime features
# ---------------------------------------------------------------------------


def regime_features(
    regime_probabilities: pd.DataFrame,
    regime_labels: pd.Series | None = None,
) -> pd.DataFrame:
    """Create features from regime probabilities or labels.

    Given regime probabilities (e.g., from an HMM or Markov-switching model),
    constructs features useful for downstream ML models: current regime
    identity, regime duration (how many consecutive periods in the current
    regime), and estimated transition probability (rolling mean of regime
    changes).

    Parameters
    ----------
    regime_probabilities : pd.DataFrame
        DataFrame where each column is the probability of a regime
        (e.g., columns ``['bull', 'bear']`` with probabilities summing to 1).
    regime_labels : pd.Series or None
        Hard regime labels. If None, the most probable regime at each step
        is used (argmax of the probability columns).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - ``current_regime``: integer label of the current regime
        - ``regime_duration``: number of consecutive periods in the
          current regime
        - ``regime_change``: binary indicator (1 if regime changed)
        - ``transition_prob_w{w}``: rolling mean of regime changes
          for w in [5, 10, 21]
        - one column per regime probability from the input

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> probs = pd.DataFrame({
    ...     'bull': np.random.dirichlet([5, 2], size=100)[:, 0],
    ...     'bear': np.random.dirichlet([5, 2], size=100)[:, 1],
    ... })
    >>> result = regime_features(probs)
    >>> 'current_regime' in result.columns
    True
    >>> 'regime_duration' in result.columns
    True
    """
    result: dict[str, pd.Series] = {}

    # Current regime (argmax)
    if regime_labels is not None:
        current = regime_labels.astype(int)
    else:
        current = pd.Series(
            regime_probabilities.values.argmax(axis=1),
            index=regime_probabilities.index,
            name="current_regime",
        )
    result["current_regime"] = current

    # Regime change indicator
    regime_change = (current != current.shift(1)).astype(int)
    regime_change.iloc[0] = 0
    result["regime_change"] = regime_change

    # Regime duration (consecutive periods in current regime)
    duration = np.zeros(len(current), dtype=int)
    duration[0] = 1
    current_vals = current.values
    for i in range(1, len(current_vals)):
        if current_vals[i] == current_vals[i - 1]:
            duration[i] = duration[i - 1] + 1
        else:
            duration[i] = 1
    result["regime_duration"] = pd.Series(
        duration, index=regime_probabilities.index
    )

    # Rolling transition probability (how frequently regimes change)
    for w in [5, 10, 21]:
        result[f"transition_prob_w{w}"] = regime_change.rolling(
            w, min_periods=1
        ).mean()

    # Include raw probabilities
    for col in regime_probabilities.columns:
        result[f"prob_{col}"] = regime_probabilities[col]

    return pd.DataFrame(result, index=regime_probabilities.index)
