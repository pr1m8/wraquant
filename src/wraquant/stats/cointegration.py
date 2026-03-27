"""Cointegration tests and pairs trading utilities for financial data."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from wraquant.core.decorators import requires_extra


def engle_granger(
    y1: pd.Series,
    y2: pd.Series,
    max_lag: int | None = None,
) -> dict:
    """Engle-Granger two-step cointegration test.

    Regresses *y1* on *y2* via OLS, then tests the residuals for a unit
    root using the Augmented Dickey-Fuller test.

    Parameters:
        y1: First price series.
        y2: Second price series.
        max_lag: Maximum number of lags for the ADF test.  When *None*,
            ``adfuller`` selects lags automatically via AIC.

    Returns:
        Dictionary with ``statistic`` (ADF test statistic), ``p_value``,
        ``is_cointegrated`` (at 5 % significance), ``hedge_ratio``
        (OLS slope coefficient), and ``residuals`` (pd.Series).
    """
    y1_clean = y1.dropna()
    y2_clean = y2.dropna()
    # Align on common index
    common = y1_clean.index.intersection(y2_clean.index)
    y1_vals = y1_clean.loc[common].values.astype(float)
    y2_vals = y2_clean.loc[common].values.astype(float)

    # Step 1: OLS regression  y1 = beta * y2 + alpha + eps — canonical import
    from wraquant.stats.regression import ols as _ols

    _eg_result = _ols(y1_vals, y2_vals, add_constant=True)
    # coefficients[0] = intercept (alpha), coefficients[1] = slope (beta)
    beta = _eg_result["coefficients"][1]

    residuals = y1_vals - beta * y2_vals

    # Step 2: ADF test on residuals
    adf_result = adfuller(residuals, maxlag=max_lag, autolag="AIC")
    stat, p_value = adf_result[0], adf_result[1]

    residual_series = pd.Series(residuals, index=common, name="residuals")

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "is_cointegrated": bool(p_value < 0.05),
        "hedge_ratio": float(beta),
        "residuals": residual_series,
    }


@requires_extra("timeseries")
def johansen(
    data: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> dict:
    """Johansen cointegration test for multiple time series.

    Requires the ``timeseries`` optional dependency group (provides
    ``statsmodels.tsa.vector_ar.vecm``).

    Parameters:
        data: DataFrame of price series (columns = assets).
        det_order: Deterministic term order.  ``-1`` for no deterministic
            term, ``0`` for constant, ``1`` for linear trend.
        k_ar_diff: Number of lagged differences in the model.

    Returns:
        Dictionary with ``trace_stats`` (array of trace statistics),
        ``eigenvalues``, ``critical_values_95`` (95 % critical values),
        and ``n_cointegrating`` (number of cointegrating relationships
        at the 5 % level).
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    clean = data.dropna()
    result = coint_johansen(clean.values, det_order, k_ar_diff)

    trace_stats = result.lr1
    crit_95 = result.cvt[:, 1]  # 95% critical values
    eigenvalues = result.eig

    # Count cointegrating relationships (trace stat > critical value)
    n_coint = int(np.sum(trace_stats > crit_95))

    return {
        "trace_stats": trace_stats,
        "eigenvalues": eigenvalues,
        "critical_values_95": crit_95,
        "n_cointegrating": n_coint,
    }


def half_life(spread: pd.Series) -> float:
    """Estimate the half-life of mean reversion for a spread series.

    Fits an OLS regression of the change in spread on the lagged spread
    level:  ``delta_spread = phi * spread_lag + eps``.  The half-life is
    ``-log(2) / log(1 + phi)``.

    Parameters:
        spread: Spread (or residual) series.

    Returns:
        Half-life in the same time units as the spread index.
        Returns ``float('inf')`` when the spread is not mean-reverting.
    """
    clean = spread.dropna().values.astype(float)

    lag = clean[:-1]
    delta = np.diff(clean)

    # OLS: delta = phi * lag + eps  (no intercept needed for half-life)
    phi = float(np.dot(lag, delta) / np.dot(lag, lag))

    if phi >= 0:
        # Not mean-reverting
        return float("inf")

    hl = -np.log(2) / np.log(1 + phi)
    return float(hl)


def spread(
    y1: pd.Series,
    y2: pd.Series,
    hedge_ratio: float | None = None,
) -> pd.Series:
    """Compute the spread between two price series.

    When *hedge_ratio* is ``None`` it is estimated via OLS.

    Parameters:
        y1: First price series (the dependent variable).
        y2: Second price series (the independent variable).
        hedge_ratio: Explicit hedge ratio.  If ``None``, the ratio is
            estimated from the data using OLS.

    Returns:
        Spread series (``y1 - hedge_ratio * y2``).
    """
    common = y1.dropna().index.intersection(y2.dropna().index)
    y1_aligned = y1.loc[common]
    y2_aligned = y2.loc[common]

    if hedge_ratio is None:
        y1_vals = y1_aligned.values.astype(float)
        y2_vals = y2_aligned.values.astype(float)
        from wraquant.stats.regression import ols as _ols

        _sp_result = _ols(y1_vals, y2_vals, add_constant=True)
        beta = _sp_result["coefficients"][1]  # slope (hedge ratio)
    else:
        beta = hedge_ratio

    result = y1_aligned - beta * y2_aligned
    result.name = "spread"
    return result


def zscore_signal(spread: pd.Series, window: int = 20) -> pd.Series:
    """Compute a rolling z-score of the spread for trading signals.

    Parameters:
        spread: Spread series.
        window: Rolling window size for mean and standard deviation.

    Returns:
        Rolling z-score series.
    """
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std(ddof=1)
    z = (spread - rolling_mean) / rolling_std
    z.name = "zscore"
    return z


def hedge_ratio(
    y1: pd.Series,
    y2: pd.Series,
    method: str = "ols",
) -> float:
    """Estimate the hedge ratio between two price series.

    Parameters:
        y1: First (dependent) price series.
        y2: Second (independent) price series.
        method: Estimation method — ``"ols"`` for ordinary least squares
            or ``"tls"`` for total least squares (orthogonal regression).

    Returns:
        Hedge ratio as a float.

    Raises:
        ValueError: If *method* is not recognized.
    """
    common = y1.dropna().index.intersection(y2.dropna().index)
    y1_vals = y1.loc[common].values.astype(float)
    y2_vals = y2.loc[common].values.astype(float)

    if method == "ols":
        from wraquant.stats.regression import ols as _ols

        _hr_result = _ols(y1_vals, y2_vals, add_constant=True)
        return float(_hr_result["coefficients"][1])  # slope
    elif method == "tls":
        # Total least squares via SVD
        data = np.column_stack([y2_vals - y2_vals.mean(), y1_vals - y1_vals.mean()])
        _u, _s, vt = np.linalg.svd(data, full_matrices=False)
        # The TLS slope is -V[0,1] / V[1,1]
        beta = -vt[-1, 0] / vt[-1, 1]
        return float(beta)
    else:
        msg = f"Unknown hedge ratio method: {method!r}"
        raise ValueError(msg)


def pairs_backtest_signals(
    spread: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> pd.Series:
    """Generate pairs trading signals based on z-score thresholds.

    The strategy goes short the spread when z-score > *entry_z*, goes
    long when z-score < -*entry_z*, and exits when the z-score crosses
    back inside ``[-exit_z, exit_z]``.

    Parameters:
        spread: Spread series (raw, not z-scored).
        entry_z: Z-score threshold for entry (absolute value).
        exit_z: Z-score threshold for exit (absolute value).

    Returns:
        Signal series with values in ``{-1, 0, 1}``.  ``1`` means long
        the spread, ``-1`` means short, ``0`` means flat.
    """
    z = zscore_signal(spread)
    signals = pd.Series(0, index=spread.index, name="signal", dtype=int)

    position = 0
    for i in range(len(z)):
        if np.isnan(z.iloc[i]):
            signals.iloc[i] = 0
            continue

        zval = z.iloc[i]
        if position == 0:
            if zval > entry_z:
                position = -1  # short spread
            elif zval < -entry_z:
                position = 1  # long spread
        elif position == 1:
            if zval > -exit_z:
                position = 0
        elif position == -1:
            if zval < exit_z:
                position = 0

        signals.iloc[i] = position

    return signals


def find_cointegrated_pairs(
    prices_df: pd.DataFrame,
    significance: float = 0.05,
) -> list[tuple]:
    """Scan a DataFrame of price series and find all cointegrated pairs.

    For each pair of columns, the Engle-Granger cointegration test is
    applied.  Pairs with a p-value below *significance* are returned.

    Parameters:
        prices_df: DataFrame of price series (columns = asset names).
        significance: Significance level for the cointegration test.

    Returns:
        List of tuples ``(asset1, asset2, p_value, hedge_ratio)`` for
        each cointegrated pair, sorted by p-value ascending.
    """
    columns = prices_df.columns.tolist()
    pairs: list[tuple] = []

    for col1, col2 in combinations(columns, 2):
        y1 = prices_df[col1].dropna()
        y2 = prices_df[col2].dropna()

        if len(y1) < 30 or len(y2) < 30:
            continue

        result = engle_granger(y1, y2)
        if result["p_value"] < significance:
            pairs.append((col1, col2, result["p_value"], result["hedge_ratio"]))

    pairs.sort(key=lambda x: x[2])
    return pairs
