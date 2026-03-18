"""Order flow toxicity metrics.

Provides measures of adverse selection and informed trading probability
used in market microstructure analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize


def vpin(
    volume: pd.Series | NDArray[np.floating],
    buy_volume: pd.Series | NDArray[np.floating],
    n_buckets: int = 50,
) -> NDArray[np.floating]:
    """Volume-Synchronized Probability of Informed Trading (VPIN).

    Aggregates volume into equal-sized buckets and measures the
    absolute order imbalance in each bucket.

    Parameters:
        volume: Total volume per observation.
        buy_volume: Buy-initiated volume per observation.
        n_buckets: Number of volume buckets.

    Returns:
        VPIN values, one per bucket.
    """
    volume = np.asarray(volume, dtype=np.float64)
    buy_volume = np.asarray(buy_volume, dtype=np.float64)
    sell_volume = volume - buy_volume

    total_volume = np.sum(volume)
    bucket_size = total_volume / n_buckets

    vpin_values: list[float] = []
    cum_buy = 0.0
    cum_sell = 0.0
    cum_vol = 0.0

    for i in range(len(volume)):
        remaining = volume[i]
        buy_frac = buy_volume[i] / volume[i] if volume[i] > 0 else 0.5
        sell_frac = 1.0 - buy_frac

        while remaining > 0:
            space = bucket_size - cum_vol
            fill = min(remaining, space)
            cum_buy += fill * buy_frac
            cum_sell += fill * sell_frac
            cum_vol += fill
            remaining -= fill

            if cum_vol >= bucket_size - 1e-10:
                bucket_vol = cum_buy + cum_sell
                if bucket_vol > 0:
                    vpin_values.append(abs(cum_buy - cum_sell) / bucket_vol)
                else:
                    vpin_values.append(0.0)
                cum_buy = 0.0
                cum_sell = 0.0
                cum_vol = 0.0

    return np.array(vpin_values, dtype=np.float64)


def pin_model(
    buy_trades: pd.Series | NDArray[np.floating],
    sell_trades: pd.Series | NDArray[np.floating],
) -> dict[str, float]:
    """Estimate the Probability of Informed Trading (PIN).

    Uses maximum likelihood on the Easley-Kiefer-O'Hara-Paperman model.
    The PIN is defined as ``alpha * mu / (alpha * mu + eps_b + eps_s)``.

    Parameters:
        buy_trades: Number of buy-initiated trades per period.
        sell_trades: Number of sell-initiated trades per period.

    Returns:
        Dictionary with keys ``'pin'``, ``'alpha'``, ``'delta'``,
        ``'mu'``, ``'eps_b'``, ``'eps_s'``.
    """
    b = np.asarray(buy_trades, dtype=np.float64)
    s = np.asarray(sell_trades, dtype=np.float64)
    n = len(b)

    # Initial parameter guesses
    total = b + s
    alpha0 = 0.5
    delta0 = 0.5
    mu0 = float(np.mean(np.abs(b - s)))
    eps_b0 = float(np.mean(b)) * 0.5
    eps_s0 = float(np.mean(s)) * 0.5

    def neg_log_lik(params: NDArray[np.floating]) -> float:
        alpha, delta, mu, eps_b, eps_s = params
        # Clamp for numerical stability
        alpha = np.clip(alpha, 1e-6, 1 - 1e-6)
        delta = np.clip(delta, 1e-6, 1 - 1e-6)
        mu = max(mu, 1e-6)
        eps_b = max(eps_b, 1e-6)
        eps_s = max(eps_s, 1e-6)

        ll = 0.0
        for i in range(n):
            bi, si = b[i], s[i]
            # Three regimes: no event, bad-news event, good-news event
            # Use log-sum-exp for stability
            log_parts = np.array([
                np.log(1 - alpha) + _poisson_ll(bi, eps_b) + _poisson_ll(si, eps_s),
                np.log(alpha * delta) + _poisson_ll(bi, eps_b) + _poisson_ll(si, eps_s + mu),
                np.log(alpha * (1 - delta)) + _poisson_ll(bi, eps_b + mu) + _poisson_ll(si, eps_s),
            ])
            max_lp = np.max(log_parts)
            ll += max_lp + np.log(np.sum(np.exp(log_parts - max_lp)))
        return -ll

    x0 = np.array([alpha0, delta0, mu0, eps_b0, eps_s0])
    bounds = [(1e-4, 1 - 1e-4), (1e-4, 1 - 1e-4), (1e-4, None), (1e-4, None), (1e-4, None)]
    result = minimize(neg_log_lik, x0, bounds=bounds, method="L-BFGS-B")
    alpha, delta, mu, eps_b, eps_s = result.x

    pin_value = (alpha * mu) / (alpha * mu + eps_b + eps_s)
    return {
        "pin": float(pin_value),
        "alpha": float(alpha),
        "delta": float(delta),
        "mu": float(mu),
        "eps_b": float(eps_b),
        "eps_s": float(eps_s),
    }


def _poisson_ll(k: float, lam: float) -> float:
    """Log-likelihood of Poisson(lam) evaluated at k."""
    from scipy.special import gammaln

    lam = max(lam, 1e-300)
    return k * np.log(lam) - lam - gammaln(k + 1)


def order_flow_imbalance(
    buy_volume: pd.Series,
    sell_volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Rolling order flow imbalance (OFI).

    Defined as ``(buy_volume - sell_volume) / (buy_volume + sell_volume)``
    averaged over a rolling window.

    Parameters:
        buy_volume: Buy-initiated volume per period.
        sell_volume: Sell-initiated volume per period.
        window: Rolling window size.

    Returns:
        Rolling OFI series in [-1, 1].
    """
    total = buy_volume + sell_volume
    ofi = (buy_volume - sell_volume) / total
    ofi = ofi.replace([np.inf, -np.inf], np.nan)
    return ofi.rolling(window).mean()


def trade_classification(
    trade_prices: pd.Series,
    bid: pd.Series,
    ask: pd.Series,
) -> pd.Series:
    """Lee-Ready trade classification combining quote test and tick test.

    Classifies each trade as buyer-initiated (+1) or seller-initiated (-1).

    The quote test assigns direction based on whether the trade price is
    above or below the midpoint.  When the trade is at the midpoint, the
    tick test (based on successive price changes) is used as a fallback.

    Parameters:
        trade_prices: Executed trade prices.
        bid: Best bid prices at time of each trade.
        ask: Best ask prices at time of each trade.

    Returns:
        Classification series with values +1 (buy) or -1 (sell).
    """
    mid = (bid + ask) / 2.0

    # Quote test
    direction = pd.Series(np.where(trade_prices > mid, 1, np.where(trade_prices < mid, -1, 0)),
                          index=trade_prices.index)

    # Tick test fallback for trades at midpoint
    tick = np.sign(trade_prices.diff())
    tick = tick.ffill().fillna(1)
    mask = direction == 0
    direction[mask] = tick[mask].astype(int)

    # Ensure no zeros remain
    direction = direction.replace(0, 1)
    direction.name = "trade_direction"
    return direction


def information_share(
    prices_list: list[pd.Series],
) -> NDArray[np.floating]:
    """Hasbrouck's information share across multiple venues.

    Estimates each venue's contribution to price discovery using the
    variance decomposition of a VECM residual.  A simplified approach
    is used: the information share for venue *j* is proportional to
    ``1 / var(price_j - mean_price)``.

    Parameters:
        prices_list: List of price series from different venues, all
            sharing the same DatetimeIndex.

    Returns:
        Array of information shares summing to 1, one per venue.
    """
    n_venues = len(prices_list)
    if n_venues == 0:
        return np.array([], dtype=np.float64)

    # Align all series
    df = pd.concat(prices_list, axis=1).dropna()
    mean_price = df.mean(axis=1)

    # Deviation-based proxy for information share
    inv_vars: list[float] = []
    for j in range(n_venues):
        deviation = df.iloc[:, j] - mean_price
        var_dev = np.var(deviation)
        inv_vars.append(1.0 / var_dev if var_dev > 0 else 0.0)

    total = sum(inv_vars)
    if total == 0:
        return np.ones(n_venues, dtype=np.float64) / n_venues

    return np.array([v / total for v in inv_vars], dtype=np.float64)
