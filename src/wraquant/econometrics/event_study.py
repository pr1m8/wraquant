"""Event study methodology.

Implements the classic event study framework widely used in empirical
finance to measure the impact of corporate events (earnings announcements,
M&A, etc.) on security prices.  Follows the methodology of MacKinlay (1997)
and Campbell, Lo, and MacKinlay (1997, ch. 4).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from wraquant.core._coerce import coerce_series


def event_study(
    returns: pd.DataFrame | pd.Series,
    event_dates: list | pd.DatetimeIndex,
    estimation_window: tuple[int, int] = (-250, -10),
    event_window: tuple[int, int] = (-5, 5),
    market_returns: pd.Series | None = None,
) -> dict[str, Any]:
    """Classic event study with market-model abnormal returns.

    For each event date, estimates a market model over the *estimation_window*
    and computes abnormal returns (AR) and cumulative abnormal returns (CAR)
    over the *event_window*.

    If *market_returns* is ``None``, a constant-mean-return model is used
    instead of the market model.

    Parameters:
        returns: Return series for the security (or DataFrame with one column
            per security).  Must have a DatetimeIndex.
        event_dates: List of event dates.
        estimation_window: ``(start, end)`` offsets in trading days relative
            to the event date for the estimation period.  Default
            ``(-250, -10)``.
        event_window: ``(start, end)`` offsets for the event window.  Default
            ``(-5, 5)``.
        market_returns: Market return series (must overlap in time with
            *returns*).

    Returns:
        Dictionary with:
        - ``abnormal_returns``: DataFrame (n_events, event_window_len) of AR.
        - ``car``: Series of cumulative abnormal returns for each event.
        - ``mean_car``: Average CAR across events.
        - ``t_stat``: Cross-sectional t-statistic for mean CAR.
        - ``p_value``: Two-sided p-value.
        - ``event_dates``: The event dates used.
        - ``n_events``: Number of valid events.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] == 1:
            returns = returns.iloc[:, 0]
        else:
            msg = "Pass a single-column DataFrame or Series for returns."
            raise ValueError(msg)

    returns = coerce_series(returns, name="returns")
    returns = returns.sort_index()
    event_dates = pd.DatetimeIndex(event_dates)

    ew_start, ew_end = event_window
    est_start, est_end = estimation_window
    event_len = ew_end - ew_start + 1

    all_ar = []
    valid_dates = []

    for edate in event_dates:
        # Find the position of the event date in the index
        idx_positions = returns.index.get_indexer([edate], method="nearest")
        event_pos = idx_positions[0]

        if event_pos < 0:
            continue

        # Estimation window positions
        est_start_pos = event_pos + est_start
        est_end_pos = event_pos + est_end
        ew_start_pos = event_pos + ew_start
        ew_end_pos = event_pos + ew_end

        # Bounds check
        if est_start_pos < 0 or ew_end_pos >= len(returns):
            continue

        est_ret = returns.iloc[est_start_pos : est_end_pos + 1].values
        event_ret = returns.iloc[ew_start_pos : ew_end_pos + 1].values

        if len(event_ret) != event_len:
            continue

        if market_returns is not None:
            # Market model: R_i = alpha + beta * R_m + epsilon
            mkt = market_returns.sort_index()
            mkt_positions = mkt.index.get_indexer(
                returns.index[est_start_pos : est_end_pos + 1], method="nearest"
            )
            mkt_est = mkt.iloc[mkt_positions].values

            mkt_ew_positions = mkt.index.get_indexer(
                returns.index[ew_start_pos : ew_end_pos + 1], method="nearest"
            )
            mkt_event = mkt.iloc[mkt_ew_positions].values

            # OLS: R_i = alpha + beta * R_m — canonical import
            try:
                from wraquant.stats.regression import ols as _ols

                _mkt_result = _ols(est_ret, mkt_est, add_constant=True)
                beta_hat = _mkt_result["coefficients"]
            except (np.linalg.LinAlgError, Exception):
                continue

            X_event = np.column_stack([np.ones(len(mkt_event)), mkt_event])
            expected_ret = X_event @ beta_hat
        else:
            # Constant mean return model
            expected_ret = np.mean(est_ret) * np.ones(event_len)

        ar = event_ret - expected_ret
        all_ar.append(ar)
        valid_dates.append(edate)

    if len(all_ar) == 0:
        return {
            "abnormal_returns": pd.DataFrame(),
            "car": pd.Series(dtype=float),
            "mean_car": 0.0,
            "t_stat": 0.0,
            "p_value": 1.0,
            "event_dates": pd.DatetimeIndex([]),
            "n_events": 0,
        }

    ar_array = np.array(all_ar)  # (n_events, event_len)
    col_labels = list(range(ew_start, ew_end + 1))

    ar_df = pd.DataFrame(ar_array, index=valid_dates, columns=col_labels)

    # CAR for each event
    car_values = ar_array.sum(axis=1)
    car_series = pd.Series(car_values, index=valid_dates, name="CAR")

    # Cross-sectional test
    n_events = len(car_values)
    mean_car = float(car_values.mean())
    std_car = float(car_values.std(ddof=1)) if n_events > 1 else 1.0
    t_stat = mean_car / (std_car / np.sqrt(n_events)) if std_car > 0 else 0.0
    p_value = 2.0 * (1.0 - sp_stats.t.cdf(abs(t_stat), df=max(n_events - 1, 1)))

    return {
        "abnormal_returns": ar_df,
        "car": car_series,
        "mean_car": mean_car,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "event_dates": pd.DatetimeIndex(valid_dates),
        "n_events": n_events,
    }


def cumulative_abnormal_return(
    returns: pd.Series,
    expected_returns: pd.Series,
    event_window: tuple[int, int] | None = None,
) -> pd.Series:
    """Compute cumulative abnormal returns (CAR).

    Parameters:
        returns: Actual return series with DatetimeIndex.
        expected_returns: Expected (normal) return series, aligned with
            *returns*.
        event_window: Optional ``(start_idx, end_idx)`` integer slice into
            the series.  If ``None``, the full series is used.

    Returns:
        Series of cumulative abnormal returns.
    """
    returns = coerce_series(returns, name="returns")
    expected_returns = coerce_series(expected_returns, name="expected_returns")
    ar = returns - expected_returns

    if event_window is not None:
        start, end = event_window
        ar = ar.iloc[start : end + 1] if isinstance(start, int) else ar.loc[start:end]

    return ar.cumsum()


def buy_and_hold_abnormal_return(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    event_window: tuple[int, int] | None = None,
) -> float:
    """Buy-and-hold abnormal return (BHAR).

    BHAR = product(1 + R_i) - product(1 + R_benchmark)

    Unlike CAR, BHAR accounts for compounding and is preferred for
    longer-horizon event studies (Barber and Lyon, 1997).

    Parameters:
        returns: Security return series.
        benchmark_returns: Benchmark return series, aligned with *returns*.
        event_window: Optional ``(start_idx, end_idx)`` integer slice.
            If ``None``, the full series is used.

    Returns:
        BHAR as a float.
    """
    returns = coerce_series(returns, name="returns")
    benchmark_returns = coerce_series(benchmark_returns, name="benchmark_returns")
    if event_window is not None:
        start, end = event_window
        ret_slice = returns.iloc[start : end + 1]
        bench_slice = benchmark_returns.iloc[start : end + 1]
    else:
        ret_slice = returns
        bench_slice = benchmark_returns

    bhar = float(np.prod(1.0 + ret_slice.values) - np.prod(1.0 + bench_slice.values))
    return bhar
