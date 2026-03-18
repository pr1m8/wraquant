"""Advanced backtesting integrations using optional packages.

Provides wrappers around vectorbt, quantstats, empyrical, pyfolio,
and ffn for backtesting, reporting, and performance analytics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wraquant.core.decorators import requires_extra

__all__ = [
    "vectorbt_backtest",
    "quantstats_report",
    "empyrical_metrics",
    "pyfolio_tearsheet_data",
    "ffn_stats",
]


@requires_extra("backtesting")
def vectorbt_backtest(
    prices: pd.Series | pd.DataFrame,
    entries: pd.Series | pd.DataFrame,
    exits: pd.Series | pd.DataFrame,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run a vectorised backtest using vectorbt.

    Parameters
    ----------
    prices : pd.Series or pd.DataFrame
        Price data aligned with the entry/exit signals.
    entries : pd.Series or pd.DataFrame
        Boolean series/DataFrame indicating entry signals.
    exits : pd.Series or pd.DataFrame
        Boolean series/DataFrame indicating exit signals.
    **kwargs
        Additional keyword arguments forwarded to
        ``vectorbt.Portfolio.from_signals``.

    Returns
    -------
    dict
        Dictionary containing:

        * **total_return** -- total portfolio return.
        * **sharpe_ratio** -- annualised Sharpe ratio.
        * **max_drawdown** -- maximum drawdown.
        * **total_trades** -- number of trades executed.
        * **win_rate** -- fraction of winning trades.
        * **portfolio** -- the raw ``vectorbt.Portfolio`` object.
    """
    import vectorbt as vbt

    # Ensure index has a compatible freq for vectorbt
    if isinstance(prices.index, pd.DatetimeIndex) and prices.index.freq is not None:
        prices = prices.copy()
        prices.index.freq = None

    pf = vbt.Portfolio.from_signals(prices, entries, exits, **kwargs)

    return {
        "total_return": float(pf.total_return()),
        "sharpe_ratio": float(pf.sharpe_ratio()),
        "max_drawdown": float(pf.max_drawdown()),
        "total_trades": int(pf.trades.count()),
        "win_rate": float(pf.trades.win_rate()) if pf.trades.count() > 0 else 0.0,
        "portfolio": pf,
    }


@requires_extra("backtesting")
def quantstats_report(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    output: str | None = None,
) -> dict[str, Any]:
    """Generate performance analytics using quantstats.

    Parameters
    ----------
    returns : pd.Series
        Strategy return series (simple returns).
    benchmark : pd.Series or None, default None
        Benchmark return series for comparison.
    output : str or None, default None
        File path for the HTML report. When *None*, no file is written.

    Returns
    -------
    dict
        Dictionary containing:

        * **sharpe** -- annualised Sharpe ratio.
        * **sortino** -- annualised Sortino ratio.
        * **max_drawdown** -- maximum drawdown.
        * **cagr** -- compound annual growth rate.
        * **volatility** -- annualised volatility.
        * **calmar** -- Calmar ratio.
    """
    import quantstats as qs

    if output is not None:
        qs.reports.html(returns, benchmark=benchmark, output=output)

    return {
        "sharpe": float(qs.stats.sharpe(returns)),
        "sortino": float(qs.stats.sortino(returns)),
        "max_drawdown": float(qs.stats.max_drawdown(returns)),
        "cagr": float(qs.stats.cagr(returns)),
        "volatility": float(qs.stats.volatility(returns)),
        "calmar": float(qs.stats.calmar(returns)),
    }


@requires_extra("backtesting")
def empyrical_metrics(returns: pd.Series) -> dict[str, float]:
    """Compute a comprehensive set of metrics using empyrical.

    Parameters
    ----------
    returns : pd.Series
        Simple return series.

    Returns
    -------
    dict
        Dictionary containing:

        * **annual_return** -- annualised return.
        * **annual_volatility** -- annualised volatility.
        * **sharpe_ratio** -- annualised Sharpe ratio.
        * **sortino_ratio** -- annualised Sortino ratio.
        * **max_drawdown** -- maximum drawdown.
        * **calmar_ratio** -- Calmar ratio.
        * **omega_ratio** -- Omega ratio.
        * **tail_ratio** -- tail ratio (95th / 5th percentile).
        * **stability** -- R-squared of cumulative log returns.
    """
    clean = returns.dropna()
    ann = 252

    annual_ret = float((1 + clean.mean()) ** ann - 1)
    annual_vol = float(clean.std() * np.sqrt(ann))
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0.0
    neg = clean[clean < 0]
    downside_std = float(neg.std() * np.sqrt(ann)) if len(neg) > 0 else 1e-12
    sortino = annual_ret / downside_std

    cum = (1 + clean).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = float(dd.min())

    calmar = annual_ret / abs(max_dd) if abs(max_dd) > 0 else 0.0

    threshold = 0.0
    excess = clean - threshold
    omega = float(excess[excess > 0].sum() / abs(excess[excess <= 0].sum())) if (excess <= 0).any() else float("inf")

    q95 = float(np.abs(clean.quantile(0.95)))
    q05 = float(np.abs(clean.quantile(0.05)))
    tail = q95 / q05 if q05 > 0 else float("inf")

    log_cum = np.log1p(clean).cumsum()
    x = np.arange(len(log_cum))
    if len(x) > 1:
        slope, intercept = np.polyfit(x, log_cum.values, 1)
        ss_res = np.sum((log_cum.values - (slope * x + intercept)) ** 2)
        ss_tot = np.sum((log_cum.values - log_cum.values.mean()) ** 2)
        stability = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        stability = 0.0

    return {
        "annual_return": annual_ret,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "omega_ratio": omega,
        "tail_ratio": tail,
        "stability": stability,
    }


@requires_extra("backtesting")
def pyfolio_tearsheet_data(
    returns: pd.Series,
    positions: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Prepare data in the format expected by pyfolio tearsheets.

    This function does not render plots but returns the intermediate
    data structures that pyfolio uses internally, making it possible
    to inspect results programmatically.

    Parameters
    ----------
    returns : pd.Series
        Strategy return series with a DatetimeIndex.
    positions : pd.DataFrame or None, default None
        Position sizes over time. Columns are asset names, values are
        dollar positions.
    transactions : pd.DataFrame or None, default None
        Trade log. Expected columns: ``amount``, ``price``, ``symbol``.

    Returns
    -------
    dict
        Dictionary containing:

        * **returns** -- the input return series.
        * **cum_returns** -- cumulative return series.
        * **drawdown** -- drawdown series.
        * **positions** -- positions DataFrame (or None).
        * **transactions** -- transactions DataFrame (or None).
    """
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max

    return {
        "returns": returns,
        "cum_returns": cum_returns,
        "drawdown": drawdown,
        "positions": positions,
        "transactions": transactions,
    }


@requires_extra("backtesting")
def ffn_stats(prices: pd.Series | pd.DataFrame) -> dict[str, Any]:
    """Compute performance statistics using ffn.

    Parameters
    ----------
    prices : pd.Series or pd.DataFrame
        Price series or multi-asset price DataFrame.

    Returns
    -------
    dict
        Dictionary containing key performance metrics:

        * **total_return** -- total return over the period.
        * **cagr** -- compound annual growth rate.
        * **daily_sharpe** -- daily Sharpe ratio.
        * **max_drawdown** -- maximum drawdown.
        * **avg_drawdown** -- average drawdown.
        * **monthly_sharpe** -- monthly Sharpe ratio.
        * **stats_object** -- the raw ``ffn.PerformanceStats`` object.
    """
    import ffn

    perf = ffn.calc_perf_stats(prices)

    return {
        "total_return": float(perf.stats["total_return"]),
        "cagr": float(perf.stats["cagr"]),
        "daily_sharpe": float(perf.stats["daily_sharpe"]),
        "max_drawdown": float(perf.stats["max_drawdown"]),
        "avg_drawdown": float(perf.stats["avg_drawdown"]),
        "monthly_sharpe": float(perf.stats.get("monthly_sharpe", np.nan)),
        "stats_object": perf,
    }
