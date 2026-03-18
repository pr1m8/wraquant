"""Enhanced tearsheet / reporting utilities for backtests.

Generates comprehensive performance summaries, monthly return tables,
drawdown analysis, rolling metrics, and trade-level analytics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "generate_tearsheet",
    "monthly_returns_table",
    "drawdown_table",
    "rolling_metrics_table",
    "trade_analysis",
]


def generate_tearsheet(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Generate a comprehensive performance tearsheet dictionary.

    Parameters
    ----------
    returns : pd.Series
        Portfolio return series (simple, not log).
    benchmark : pd.Series, optional
        Benchmark return series for relative metrics.
    risk_free : float
        Annualised risk-free rate.
    periods_per_year : int
        Trading periods per year (252 for daily).

    Returns
    -------
    dict[str, Any]
        Dictionary containing absolute and (optionally) relative
        performance metrics.
    """
    returns = returns.dropna()
    n = len(returns)

    if n == 0:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "n_periods": 0,
        }

    # --- Absolute metrics ---
    total_return = float((1 + returns).prod() - 1)
    ann_factor = periods_per_year / n
    ann_return = float((1 + total_return) ** ann_factor - 1)
    ann_vol = float(returns.std() * np.sqrt(periods_per_year))
    sharpe = (ann_return - risk_free) / ann_vol if ann_vol > 0 else 0.0

    # Sortino
    downside = returns[returns < 0]
    down_std = float(downside.std() * np.sqrt(periods_per_year)) if len(downside) > 0 else 0.0
    sortino = (ann_return - risk_free) / down_std if down_std > 0 else 0.0

    # Drawdown
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = float(dd.min())
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    # Skew / Kurtosis
    skew = float(returns.skew()) if n > 2 else 0.0
    kurt = float(returns.kurtosis()) if n > 3 else 0.0

    # VaR / CVaR at 95 %
    var_95 = float(np.percentile(returns, 5))
    cvar_95 = float(returns[returns <= var_95].mean()) if (returns <= var_95).any() else var_95

    # Win rate
    win_rate = float((returns > 0).sum()) / n

    # Profit factor
    gains = float(returns[returns > 0].sum())
    losses = float(abs(returns[returns < 0].sum()))
    profit_factor = gains / losses if losses > 0 else float("inf")

    result: dict[str, Any] = {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "skewness": skew,
        "kurtosis": kurt,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "n_periods": n,
    }

    # --- Benchmark-relative metrics ---
    if benchmark is not None:
        benchmark = benchmark.dropna()
        # Align on common index
        common = returns.index.intersection(benchmark.index)
        r = returns.loc[common]
        b = benchmark.loc[common]
        excess = r - b

        # Tracking error
        te = float(excess.std() * np.sqrt(periods_per_year))

        # Information ratio
        ann_excess = float(excess.mean() * periods_per_year)
        ir = ann_excess / te if te > 0 else 0.0

        # Beta / Alpha (CAPM)
        cov_rb = float(np.cov(r, b)[0, 1])
        var_b = float(b.var())
        beta = cov_rb / var_b if var_b > 0 else 0.0
        bm_ann_return = float((1 + b).prod() ** (periods_per_year / len(b)) - 1)
        alpha = ann_return - (risk_free + beta * (bm_ann_return - risk_free))

        # Up/down capture
        up_mask = b > 0
        down_mask = b < 0
        up_capture = (
            float(r[up_mask].mean() / b[up_mask].mean()) if up_mask.any() else 0.0
        )
        down_capture = (
            float(r[down_mask].mean() / b[down_mask].mean()) if down_mask.any() else 0.0
        )

        result.update(
            {
                "benchmark_total_return": float((1 + b).prod() - 1),
                "tracking_error": te,
                "information_ratio": ir,
                "beta": beta,
                "alpha": alpha,
                "up_capture": up_capture,
                "down_capture": down_capture,
            }
        )

    return result


def monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """Compute a table of monthly returns suitable for heatmap display.

    Parameters
    ----------
    returns : pd.Series
        Daily (or intraday) return series with a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Rows = years, columns = months (1-12).  Values are total
        returns for that month.
    """
    returns = returns.dropna()
    if returns.empty:
        return pd.DataFrame()

    # Compound daily returns into monthly
    monthly = (1 + returns).groupby(
        [returns.index.year, returns.index.month]
    ).prod() - 1
    monthly.index.names = ["year", "month"]
    table = monthly.unstack(level="month")
    # Flatten any MultiIndex columns that arise from unstacking
    if isinstance(table.columns, pd.MultiIndex):
        table.columns = table.columns.droplevel(0)
    table.columns.name = "month"
    return table


def drawdown_table(
    returns: pd.Series,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return the top *N* drawdown periods with metadata.

    Parameters
    ----------
    returns : pd.Series
        Portfolio return series.
    top_n : int
        Number of worst drawdowns to report.

    Returns
    -------
    pd.DataFrame
        Columns: ``peak_date``, ``trough_date``, ``recovery_date``,
        ``depth``, ``duration`` (periods from peak to recovery or end).
    """
    returns = returns.dropna()
    if returns.empty:
        return pd.DataFrame(
            columns=["peak_date", "trough_date", "recovery_date", "depth", "duration"]
        )

    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak

    # Segment drawdown periods
    events: list[dict[str, Any]] = []
    dd_start = None
    trough_val = 0.0
    trough_idx = None

    for idx, val in dd.items():
        if val < 0:
            if dd_start is None:
                dd_start = idx
                trough_val = val
                trough_idx = idx
            if val < trough_val:
                trough_val = val
                trough_idx = idx
        else:
            if dd_start is not None:
                events.append(
                    {
                        "peak_date": dd_start,
                        "trough_date": trough_idx,
                        "recovery_date": idx,
                        "depth": float(trough_val),
                    }
                )
                dd_start = None

    # Open drawdown (no recovery)
    if dd_start is not None:
        events.append(
            {
                "peak_date": dd_start,
                "trough_date": trough_idx,
                "recovery_date": None,
                "depth": float(trough_val),
            }
        )

    if not events:
        return pd.DataFrame(
            columns=["peak_date", "trough_date", "recovery_date", "depth", "duration"]
        )

    df = pd.DataFrame(events)
    # Duration in periods
    idx_list = list(returns.index)

    def _duration(row: pd.Series) -> int:
        start = idx_list.index(row["peak_date"])
        end_date = row["recovery_date"]
        if end_date is None or (isinstance(end_date, float) and np.isnan(end_date)) or pd.isna(end_date):
            end_date = idx_list[-1]
        end = idx_list.index(end_date)
        return end - start

    df["duration"] = df.apply(_duration, axis=1)
    df = df.sort_values("depth").head(top_n).reset_index(drop=True)
    return df


def rolling_metrics_table(
    returns: pd.Series,
    windows: list[int] | None = None,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Compute rolling Sharpe, volatility, and return at multiple windows.

    Parameters
    ----------
    returns : pd.Series
        Portfolio return series.
    windows : list[int], optional
        Rolling window sizes in periods.  Default ``[21, 63, 126, 252]``.
    periods_per_year : int
        Trading periods per year.

    Returns
    -------
    pd.DataFrame
        MultiIndex columns: ``(window, metric)`` with metrics
        ``rolling_return``, ``rolling_vol``, ``rolling_sharpe``.
    """
    if windows is None:
        windows = [21, 63, 126, 252]

    returns = returns.dropna()
    frames: dict[tuple[int, str], pd.Series] = {}

    for w in windows:
        roll_ret = returns.rolling(w).apply(
            lambda x: float((1 + x).prod() - 1), raw=False
        )
        roll_vol = returns.rolling(w).std() * np.sqrt(periods_per_year)
        roll_mean = returns.rolling(w).mean() * periods_per_year
        roll_sharpe = roll_mean / roll_vol.replace(0, np.nan)

        frames[(w, "rolling_return")] = roll_ret
        frames[(w, "rolling_vol")] = roll_vol
        frames[(w, "rolling_sharpe")] = roll_sharpe

    result = pd.DataFrame(frames)
    result.columns = pd.MultiIndex.from_tuples(
        result.columns, names=["window", "metric"]
    )
    return result


def trade_analysis(trades_df: pd.DataFrame) -> dict[str, float]:
    """Analyse trade-level performance.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Must contain a ``pnl`` column with per-trade profit/loss values.
        Optionally includes ``entry_price``, ``exit_price``, ``side``, etc.

    Returns
    -------
    dict[str, float]
        Dictionary with ``win_rate``, ``avg_pnl``, ``avg_win``,
        ``avg_loss``, ``profit_factor``, ``expectancy``,
        ``max_win``, ``max_loss``, ``n_trades``.
    """
    if "pnl" not in trades_df.columns:
        raise ValueError("trades_df must contain a 'pnl' column")

    pnl = trades_df["pnl"].dropna()
    n = len(pnl)
    if n == 0:
        return {
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_win": 0.0,
            "max_loss": 0.0,
            "n_trades": 0.0,
        }

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    win_rate = len(wins) / n
    avg_pnl = float(pnl.mean())
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    total_wins = float(wins.sum()) if len(wins) > 0 else 0.0
    total_losses = float(abs(losses.sum())) if len(losses) > 0 else 0.0
    profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

    # Expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    return {
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "max_win": float(pnl.max()),
        "max_loss": float(pnl.min()),
        "n_trades": float(n),
    }
