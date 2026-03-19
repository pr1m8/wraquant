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
    "ta_features",
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

    Use rolling features as a general-purpose feature engineering step
    before training ML models on time-series data.  The rolling statistics
    capture time-varying moments that can signal changes in trend (mean),
    risk (std), asymmetry (skew), and tail behaviour (kurtosis).

    For every window the following statistics are computed: mean, std,
    skew, kurtosis, min, and max.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Numeric time-series data.  If a DataFrame is passed, features are
        generated independently for each column.
    windows : Sequence[int]
        Rolling-window sizes (default ``(5, 10, 21, 63)``), corresponding
        roughly to 1-week, 2-week, 1-month, and 1-quarter horizons.

    Returns
    -------
    pd.DataFrame
        DataFrame whose columns are named
        ``{col}_{stat}_w{window}`` (or ``{stat}_w{window}`` when *data*
        is a Series).  The number of feature columns equals
        ``n_cols * len(windows) * 6``.  Early rows contain NaN where the
        window has insufficient data.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(0)
    >>> returns = pd.Series(np.random.randn(100) * 0.01, name='ret')
    >>> feats = rolling_features(returns, windows=(5, 21))
    >>> feats.columns.tolist()[:3]
    ['mean_w5', 'std_w5', 'skew_w5']
    >>> feats.shape[1]  # 6 stats * 2 windows
    12

    See Also
    --------
    return_features : Lagged and cumulative return features.
    volatility_features : Realised volatility and vol-of-vol features.
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

    Use return features as inputs to ML models predicting future returns
    or direction.  Lagged returns capture momentum and mean-reversion
    signals at multiple horizons; cumulative returns capture trend strength.

    Parameters
    ----------
    prices : pd.Series
        Price series (e.g. adjusted close).
    lags : Sequence[int]
        Lag periods for returns (default ``(1, 2, 3, 5, 10, 21)``).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``ret_lag{l}`` (log return *l* periods
        ago, a momentum/mean-reversion signal) and ``cum_ret_{l}``
        (cumulative log return over the last *l* periods, a trend
        signal) for each lag *l*.  Early rows are NaN.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> prices = pd.Series([100, 101, 102, 100, 103, 105, 104],
    ...                     name='close')
    >>> feats = return_features(prices, lags=(1, 3))
    >>> list(feats.columns)
    ['ret_lag1', 'cum_ret_1', 'ret_lag3', 'cum_ret_3']
    >>> feats['cum_ret_3'].iloc[-1] > 0  # cumulative 3-period return
    True

    See Also
    --------
    rolling_features : Rolling statistical features.
    technical_features : Technical analysis features (RSI, MACD, etc.).
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
    """Compute common technical analysis features for ML pipelines.

    Use these features as inputs to ML models when you want to capture
    classic technical signals without depending on the full ``wraquant.ta``
    module.  Combines momentum (RSI, MACD), volatility (ATR, Bollinger),
    and optionally volume (OBV) into a single DataFrame.

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
        Trade volume (optional).  When provided, adds OBV which tracks
        cumulative buying/selling pressure.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:

        - ``rsi``: Relative Strength Index (0-100).  Values above 70
          indicate overbought; below 30 indicate oversold.
        - ``macd_hist``: MACD histogram.  Positive values indicate
          bullish momentum; negative values indicate bearish.
        - ``bb_pctb``: Bollinger Band %B (0-1 range typically).
          Values above 1 mean price is above the upper band.
        - ``atr``: Average True Range.  Higher values indicate more
          volatile price action.
        - ``obv`` (optional): On-Balance Volume.  Rising OBV confirms
          an uptrend.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(0)
    >>> n = 100
    >>> close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5))
    >>> high = close + np.abs(np.random.randn(n) * 0.3)
    >>> low = close - np.abs(np.random.randn(n) * 0.3)
    >>> feats = technical_features(high, low, close)
    >>> list(feats.columns)
    ['rsi', 'macd_hist', 'bb_pctb', 'atr']

    See Also
    --------
    return_features : Lagged and cumulative return features.
    volatility_features : Realised volatility features.
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

    Use volatility features to capture the current risk environment and
    volatility regime.  Realised volatility is the most important feature
    in many financial ML models because volatility clusters (GARCH effect)
    and predicts future volatility better than returns predict future
    returns.

    Parameters
    ----------
    returns : pd.Series
        Log or simple return series.
    windows : Sequence[int]
        Window sizes for rolling calculations (default ``(5, 10, 21, 63)``).

    Returns
    -------
    pd.DataFrame
        Columns:

        - ``realized_vol_w{w}``: Annualised rolling standard deviation
          (sqrt(252) scaling).  Interpretation: a value of 0.20 means
          ~20% annualised volatility.
        - ``vol_of_vol_w{w}``: Rolling std of the rolling vol.  High
          values indicate unstable volatility (vol-of-vol regime).
        - ``vol_ratio_w{w1}_w{w2}``: Ratio of short-window vol to
          long-window vol.  Values > 1 indicate vol is spiking
          (risk-off signal); values < 1 indicate vol compression.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(0)
    >>> rets = pd.Series(np.random.randn(200) * 0.01, name='daily_ret')
    >>> feats = volatility_features(rets, windows=(5, 21))
    >>> 'realized_vol_w5' in feats.columns
    True
    >>> 'vol_ratio_w5_w21' in feats.columns
    True

    See Also
    --------
    rolling_features : General rolling statistical features.
    wraquant.vol : Full volatility modelling (GARCH, stochastic vol).
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

    Use microstructure features to capture liquidity conditions,
    information asymmetry, and trading activity.  These are particularly
    valuable for short-horizon alpha models and execution-aware strategies
    where liquidity predicts future returns or trading costs.

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
        Columns:

        - ``amihud_illiq``: Amihud illiquidity ratio (21-day rolling
          mean of |return| / dollar_volume).  Higher values indicate
          less liquid, more price-impactful markets.
        - ``kyle_lambda``: Kyle's lambda (21-day rolling OLS slope of
          |price change| on signed sqrt-volume).  Measures the price
          impact per unit of informed flow.  Higher values suggest
          more information asymmetry.
        - ``log_volume``: Natural log of volume.  Smooths the skewed
          volume distribution for ML model consumption.
        - ``volume_ma_ratio``: Current volume / 21-day moving average.
          Values > 1 indicate above-average activity (potential event).
        - ``dollar_volume``: Price * volume.  Absolute measure of
          trading activity and liquidity.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(0)
    >>> n = 100
    >>> close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5))
    >>> high = close + np.abs(np.random.randn(n) * 0.3)
    >>> low = close - np.abs(np.random.randn(n) * 0.3)
    >>> volume = pd.Series(np.random.randint(1_000_000, 5_000_000, n))
    >>> feats = microstructure_features(high, low, close, volume)
    >>> list(feats.columns)
    ['amihud_illiq', 'kyle_lambda', 'log_volume', 'volume_ma_ratio', 'dollar_volume']

    References
    ----------
    - Amihud (2002), "Illiquidity and stock returns"
    - Kyle (1985), "Continuous Auctions and Insider Trading"

    See Also
    --------
    technical_features : Price-based technical indicators.
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

    Use fixed-horizon labelling as the simplest way to create supervised
    learning targets for directional prediction.  Each observation is
    labelled based on the cumulative return over the next *horizon*
    periods.  This is the standard approach for "will the price go up
    or down over the next N days?" classification.

    Parameters
    ----------
    returns : pd.Series
        Period (e.g. daily) returns.
    horizon : int
        Number of periods to accumulate forward returns (default 5,
        i.e. one trading week).
    threshold : float
        If ``threshold > 0``, three labels are produced: ``1`` (up
        beyond threshold), ``0`` (flat), ``-1`` (down beyond threshold).
        If ``threshold == 0``, binary labels (``1`` / ``0``) are
        produced where ``1`` means positive cumulative return.

    Returns
    -------
    pd.Series
        Integer labels aligned to the original index.  The last
        *horizon* rows will be ``NaN`` (no future data available).

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> rets = pd.Series([0.01, -0.005, 0.02, 0.01, -0.03, 0.015, 0.005])
    >>> labels = label_fixed_horizon(rets, horizon=3, threshold=0.0)
    >>> labels.iloc[0]  # sum of rets[1:4] = -0.005+0.02+0.01 > 0
    1

    Notes
    -----
    Fixed-horizon labelling does not adapt to volatility.  In high-vol
    regimes, the threshold is hit more often; in low-vol regimes, most
    labels become ``0``.  For volatility-adaptive labels, use
    ``label_triple_barrier``.

    See Also
    --------
    label_triple_barrier : Volatility-adaptive labelling (Lopez de Prado).
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

    Use triple-barrier labelling when you want targets that adapt to
    market conditions.  Unlike fixed-horizon labels, this method defines
    a profit-taking barrier (upper), a stop-loss barrier (lower), and a
    maximum holding period (vertical).  Whichever barrier is hit first
    determines the label.  This produces cleaner labels in volatile
    markets because the barriers can be scaled by volatility.

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
        ``1`` = profit-taking barrier hit first (bullish),
        ``-1`` = stop-loss barrier hit first (bearish),
        ``0`` = vertical barrier hit with zero return.
        The last *max_holding* entries may be ``NaN``.

    Example
    -------
    >>> import pandas as pd
    >>> close = pd.Series([100, 101, 102, 103, 100, 97, 98, 99, 100, 101])
    >>> labels = label_triple_barrier(close, upper=0.03, lower=0.03, max_holding=5)
    >>> labels.iloc[0]  # price rises 3% by bar 3 (103/100 - 1 = 0.03)
    1

    Notes
    -----
    In practice, set ``upper`` and ``lower`` proportional to recent
    volatility (e.g., ``upper = lower = daily_vol * sqrt(max_holding)``).
    This makes the labels regime-adaptive.

    References
    ----------
    - Lopez de Prado (2018), "Advances in Financial Machine Learning", Ch. 3

    See Also
    --------
    label_fixed_horizon : Simpler fixed-horizon labelling.
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

    Use interaction features when you suspect that predictive power lies
    in the *combination* of features rather than individual signals.  For
    example, ``momentum * volatility`` captures whether momentum is
    occurring in a high- or low-volatility environment, which may predict
    returns differently.

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

    Use cross-asset features to capture how an asset co-moves with a
    benchmark or related instrument.  Rolling correlation and beta
    detect changing exposures (useful for regime detection); relative
    strength identifies momentum divergence between the asset and its
    benchmark.

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
        result[f"relative_strength_w{w}"] = cum_asset / cum_bench.replace(0, np.nan)

    return pd.DataFrame(result, index=aligned.index)


# ---------------------------------------------------------------------------
# Regime features
# ---------------------------------------------------------------------------


def regime_features(
    regime_probabilities: pd.DataFrame,
    regime_labels: pd.Series | None = None,
) -> pd.DataFrame:
    """Create features from regime probabilities or labels.

    Use regime features when you have upstream regime detection (e.g.,
    HMM, Markov-switching) and want to feed regime state into downstream
    ML models.  Regime duration and transition probability are predictive
    because regimes tend to persist (duration) but eventually break down
    (transition probability rises before a switch).

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
    result["regime_duration"] = pd.Series(duration, index=regime_probabilities.index)

    # Rolling transition probability (how frequently regimes change)
    for w in [5, 10, 21]:
        result[f"transition_prob_w{w}"] = regime_change.rolling(w, min_periods=1).mean()

    # Include raw probabilities
    for col in regime_probabilities.columns:
        result[f"prob_{col}"] = regime_probabilities[col]

    return pd.DataFrame(result, index=regime_probabilities.index)


# ---------------------------------------------------------------------------
# TA-integrated features (imports from wraquant.ta)
# ---------------------------------------------------------------------------


def ta_features(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series | None = None,
    include: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Generate ML features using wraquant's full technical analysis library.

    Unlike ``technical_features`` (which uses inline implementations),
    this function imports directly from ``wraquant.ta`` to leverage the
    full 263-indicator library.  This bridges the ``ml`` and ``ta``
    modules so that ML pipelines can access production-quality TA
    indicators without manual wiring.

    By default, computes a curated set of the most ML-relevant
    indicators: RSI, MACD histogram, Bollinger Band %B, ATR, and
    optionally OBV.  Use the *include* parameter to select additional
    indicators.

    Parameters:
        high: High prices.
        low: Low prices.
        close: Close prices.
        volume: Trade volume (optional).  Required for volume-based
            indicators (OBV, MFI).
        include: Subset of indicators to include.  Options:
            ``'rsi'``, ``'macd'``, ``'bbands'``, ``'atr'``, ``'obv'``.
            If *None*, includes all available indicators.

    Returns:
        DataFrame with one column per indicator, indexed like the
        input series.  Column names are descriptive (e.g., ``ta_rsi``,
        ``ta_macd_hist``, ``ta_bb_pctb``, ``ta_atr``, ``ta_obv``).

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(0)
        >>> n = 100
        >>> close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5))
        >>> high = close + np.abs(np.random.randn(n) * 0.3)
        >>> low = close - np.abs(np.random.randn(n) * 0.3)
        >>> feats = ta_features(high, low, close)
        >>> 'ta_rsi' in feats.columns
        True

    See Also:
        technical_features: Inline implementation (no ta/ dependency).
        wraquant.ta.momentum.rsi: Full RSI implementation.
        wraquant.ta.momentum.macd: Full MACD implementation.
    """
    from wraquant.ta.momentum import macd, rsi
    from wraquant.ta.overlap import bollinger_bands
    from wraquant.ta.volatility import atr

    all_indicators = {"rsi", "macd", "bbands", "atr", "obv"}
    if include is None:
        selected = all_indicators.copy()
    else:
        selected = set(include) & all_indicators

    result: dict[str, pd.Series] = {}

    if "rsi" in selected:
        result["ta_rsi"] = rsi(close, period=14)

    if "macd" in selected:
        macd_result = macd(close)
        if isinstance(macd_result, dict):
            result["ta_macd_hist"] = macd_result.get(
                "histogram", macd_result.get("macd_hist", pd.Series(dtype=float))
            )
        else:
            result["ta_macd_hist"] = macd_result

    if "bbands" in selected:
        bb = bollinger_bands(close, period=20)
        if isinstance(bb, dict):
            upper = bb.get("upper", pd.Series(dtype=float))
            lower = bb.get("lower", pd.Series(dtype=float))
            bb_range = (upper - lower).replace(0, np.nan)
            result["ta_bb_pctb"] = (close - lower) / bb_range
        else:
            result["ta_bb_pctb"] = bb

    if "atr" in selected:
        result["ta_atr"] = atr(high, low, close, period=14)

    if "obv" in selected and volume is not None:
        from wraquant.ta.volume import obv

        result["ta_obv"] = obv(close, volume)

    return pd.DataFrame(result, index=close.index)
