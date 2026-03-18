"""Liquidity analytics for market microstructure.

Provides measures of market liquidity including bid-ask spread estimators,
price impact coefficients, and turnover metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def amihud_illiquidity(
    returns: pd.Series,
    volume: pd.Series,
    window: int | None = None,
) -> pd.Series | float:
    """Amihud (2002) illiquidity ratio: mean of |return| / dollar volume.

    A higher value indicates less liquid (more illiquid) markets.

    Parameters:
        returns: Asset return series.
        volume: Dollar volume series (price * shares traded).
        window: Rolling window size. If *None*, returns a single scalar
            average over the entire sample.

    Returns:
        Rolling Amihud illiquidity ratio (or a single float when
        *window* is *None*).
    """
    ratio = np.abs(returns) / volume
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    if window is None:
        return float(np.nanmean(ratio))
    return ratio.rolling(window).mean()


def kyle_lambda(
    prices: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Kyle's lambda -- price impact coefficient via rolling OLS.

    Regresses price changes on signed order flow (volume) to estimate the
    permanent price impact per unit of volume.

    Parameters:
        prices: Price series.
        volume: Signed volume series (positive for buys, negative for sells).
        window: Rolling regression window.

    Returns:
        Rolling Kyle's lambda series.
    """
    delta_p = prices.diff()
    # Rolling OLS: lambda = cov(dp, v) / var(v)
    cov_pv = delta_p.rolling(window).cov(volume)
    var_v = volume.rolling(window).var()
    lam = cov_pv / var_v
    lam = lam.replace([np.inf, -np.inf], np.nan)
    lam.name = "kyle_lambda"
    return lam


def roll_spread(prices: pd.Series) -> float:
    """Roll (1984) implied bid-ask spread from serial covariance.

    Estimates the effective spread from the negative first-order
    autocovariance of price changes: spread = 2 * sqrt(-cov).

    Parameters:
        prices: Price series.

    Returns:
        Estimated implied spread. Returns *NaN* if the serial
        covariance is non-negative (model assumption violated).
    """
    dp = prices.diff().dropna()
    cov = np.cov(dp.values[:-1], dp.values[1:])[0, 1]
    if cov >= 0:
        return np.nan
    return 2.0 * np.sqrt(-cov)


def effective_spread(
    trade_prices: pd.Series | NDArray[np.floating],
    midpoints: pd.Series | NDArray[np.floating],
) -> pd.Series | NDArray[np.floating]:
    """Effective bid-ask spread: 2 * |trade_price - midpoint|.

    Parameters:
        trade_prices: Executed trade prices.
        midpoints: Prevailing bid-ask midpoints at time of each trade.

    Returns:
        Per-trade effective spread, same type as the inputs.
    """
    return 2.0 * np.abs(np.asarray(trade_prices) - np.asarray(midpoints))


def realized_spread(
    trade_prices: pd.Series,
    midpoints: pd.Series,
    delay: int = 5,
) -> pd.Series:
    """Realized spread incorporating a post-trade midpoint delay.

    Measures the revenue to the liquidity provider:
    ``2 * direction * (trade_price - midpoint_{t+delay})``.

    Parameters:
        trade_prices: Executed trade prices.
        midpoints: Mid-quote series aligned to trades.
        delay: Number of observations to shift the midpoint forward.

    Returns:
        Per-trade realized spread series (NaN for the last *delay* rows).
    """
    direction = np.sign(trade_prices - midpoints)
    future_mid = midpoints.shift(-delay)
    return 2.0 * direction * (trade_prices - future_mid)


def price_impact(
    trade_prices: pd.Series,
    volume: pd.Series,
    direction: pd.Series,
) -> pd.Series:
    """Permanent price impact per trade.

    Computed as ``direction * (midpoint_{t+1} - midpoint_t) / volume``,
    approximated here via successive trade prices.

    Parameters:
        trade_prices: Executed trade prices.
        volume: Volume for each trade.
        direction: Trade direction indicator (+1 buy, -1 sell).

    Returns:
        Per-trade permanent price impact series.
    """
    dp = trade_prices.diff().shift(-1)
    impact = direction * dp / volume
    impact = impact.replace([np.inf, -np.inf], np.nan)
    impact.name = "price_impact"
    return impact


def turnover_ratio(
    volume: pd.Series,
    shares_outstanding: pd.Series | float,
) -> pd.Series:
    """Turnover ratio: volume / shares outstanding.

    Parameters:
        volume: Daily trading volume.
        shares_outstanding: Total shares outstanding (scalar or series).

    Returns:
        Daily turnover ratio.
    """
    ratio = volume / shares_outstanding
    ratio.name = "turnover_ratio"
    return ratio
