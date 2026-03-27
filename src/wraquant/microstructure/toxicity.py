"""Order flow toxicity metrics.

Provides measures of adverse selection and informed trading probability
used in market microstructure analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize

from wraquant.core._coerce import coerce_array, coerce_series


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
        VPIN values, one per bucket.  Higher values indicate more
        informed trading activity.  Values above 0.4 are typically
        elevated.

    Example:
        >>> import numpy as np
        >>> from wraquant.microstructure.toxicity import vpin
        >>> volume = np.array([1000, 1500, 800, 1200, 900])
        >>> buy_vol = np.array([600, 400, 500, 800, 300])
        >>> result = vpin(volume, buy_vol, n_buckets=2)
        >>> len(result) >= 1
        True

    See Also:
        pin_model: Static PIN estimation from daily trade counts.
        toxicity_index: Composite toxicity score combining VPIN and OFI.
        bulk_volume_classification: Estimate buy/sell volume from OHLCV.
    """
    volume = coerce_array(volume, "volume")
    buy_volume = coerce_array(buy_volume, "buy_volume")
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
        ``'mu'``, ``'eps_b'``, ``'eps_s'``.  PIN values above 0.20
        indicate significant informed trading.

    Example:
        >>> import numpy as np
        >>> from wraquant.microstructure.toxicity import pin_model
        >>> rng = np.random.default_rng(42)
        >>> buys = rng.poisson(50, size=60)
        >>> sells = rng.poisson(50, size=60)
        >>> result = pin_model(buys, sells)
        >>> 0 <= result['pin'] <= 1
        True

    See Also:
        adjusted_pin: PIN corrected for symmetric liquidity shocks.
        vpin: Volume-synchronized alternative (real-time).
    """
    b = coerce_array(buy_trades, "buy_trades")
    s = coerce_array(sell_trades, "sell_trades")
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
        Rolling OFI series in [-1, 1].  Values near +1 indicate
        strong buying pressure; near -1 indicates selling pressure.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> buys = pd.Series(np.random.uniform(100, 500, 50))
        >>> sells = pd.Series(np.random.uniform(100, 500, 50))
        >>> ofi = order_flow_imbalance(buys, sells, window=10)
        >>> len(ofi) == 50
        True

    See Also:
        vpin: Volume-synchronized informed trading probability.
        wraquant.microstructure.liquidity.depth_imbalance:
            Order book depth imbalance.
    """
    buy_volume = coerce_series(buy_volume, "buy_volume")
    sell_volume = coerce_series(sell_volume, "sell_volume")
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

    Example:
        >>> import pandas as pd
        >>> trades = pd.Series([100.05, 99.95, 100.00, 100.03])
        >>> bid = pd.Series([99.98, 99.93, 99.98, 100.00])
        >>> ask = pd.Series([100.02, 99.97, 100.02, 100.06])
        >>> direction = trade_classification(trades, bid, ask)
        >>> int(direction.iloc[0])  # trade above midpoint -> buy
        1

    See Also:
        bulk_volume_classification: Classify from OHLCV data (no quotes).
        order_flow_imbalance: Aggregate classified volume into a signal.
    """
    trade_prices = coerce_series(trade_prices, "trade_prices")
    bid = coerce_series(bid, "bid")
    ask = coerce_series(ask, "ask")
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
        A venue with a higher share contributes more to price discovery.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> base = pd.Series(100 + np.cumsum(np.random.randn(200) * 0.1))
        >>> venue_a = base + np.random.randn(200) * 0.01
        >>> venue_b = base + np.random.randn(200) * 0.05
        >>> shares = information_share([venue_a, venue_b])
        >>> abs(shares.sum() - 1.0) < 1e-10
        True

    See Also:
        wraquant.microstructure.market_quality.hasbrouck_information_share:
            Cholesky-based information share with bounds.
        wraquant.microstructure.market_quality.gonzalo_granger_component:
            GG permanent-transitory decomposition.
    """
    n_venues = len(prices_list)
    if n_venues == 0:
        return np.array([], dtype=np.float64)

    prices_list = [coerce_series(p, f"prices_{i}") for i, p in enumerate(prices_list)]
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


# ---------------------------------------------------------------------------
# Enhanced toxicity analytics
# ---------------------------------------------------------------------------


def bulk_volume_classification(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
) -> pd.DataFrame:
    """Bulk Volume Classification (BVC).

    Classifies aggregate volume into buy- and sell-initiated components
    using the position of the close price relative to the high-low range.
    This avoids the need for tick-by-tick data or quote data, making it
    practical for daily-frequency analysis.

    The buy fraction is estimated as::

        Z = (close - low) / (high - low)
        buy_fraction = Z  (linear interpolation)

    The CDF of a standard normal evaluated at ``Z`` is sometimes used,
    but the linear version performs comparably and is more transparent.

    **When to use**: When you need to estimate buy/sell volume from daily
    OHLCV data without tick-by-tick trade records.  Suitable as an input
    to VPIN calculations, order flow imbalance metrics, or any model
    requiring buy/sell volume decomposition.

    **Interpretation**: When the close is near the high, most volume is
    classified as buys.  When the close is near the low, most volume is
    classified as sells.  The BVC estimate is noisier than Lee-Ready for
    individual trades but aggregates well across bars.

    Parameters:
        close: Closing prices.
        high: High prices.
        low: Low prices.
        volume: Total volume per bar.

    Returns:
        DataFrame with columns ``'buy_volume'``, ``'sell_volume'``,
        ``'buy_fraction'``.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> n = 50
        >>> close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5))
        >>> high = close + np.abs(np.random.randn(n))
        >>> low = close - np.abs(np.random.randn(n))
        >>> volume = pd.Series(np.random.uniform(1e4, 5e4, n))
        >>> result = bulk_volume_classification(close, high, low, volume)
        >>> list(result.columns)
        ['buy_volume', 'sell_volume', 'buy_fraction']

    References:
        Easley, D., Lopez de Prado, M. M. & O'Hara, M. (2012). "Bulk
        Classification of Trading Activity." *Working Paper*, Johnson
        School Research Paper Series.

    See Also:
        trade_classification: Lee-Ready classification from tick data.
        vpin: Uses buy/sell volume to compute informed trading probability.
    """
    close = coerce_series(close, "close")
    high = coerce_series(high, "high")
    low = coerce_series(low, "low")
    volume = coerce_series(volume, "volume")
    hl_range = high - low
    # Avoid division by zero for doji bars
    z = np.where(hl_range > 0, (close - low) / hl_range, 0.5)
    z = np.clip(z, 0.0, 1.0)

    buy_vol = volume * z
    sell_vol = volume * (1.0 - z)

    return pd.DataFrame(
        {
            "buy_volume": buy_vol.values,
            "sell_volume": sell_vol.values,
            "buy_fraction": z,
        },
        index=close.index,
    )


def adjusted_pin(
    buy_trades: pd.Series | NDArray[np.floating],
    sell_trades: pd.Series | NDArray[np.floating],
) -> dict[str, float]:
    """Adjusted Probability of Informed Trading (AdjPIN).

    Extends the standard PIN model (Easley et al., 1996) by separating
    information-driven order flow from symmetric order-flow shocks that
    arise from liquidity effects.  The Duarte & Young (2009) adjustment
    adds a regime where *both* buy and sell arrival rates increase
    simultaneously (a liquidity event), preventing the model from
    misattributing liquidity shocks as informed trading.

    The standard PIN is known to overstate the probability of informed
    trading during periods of high overall activity (e.g., index
    rebalancing, option expiration).  AdjPIN corrects for this bias.

    **When to use**: When you need a cleaner measure of information
    asymmetry that is not contaminated by correlated liquidity events.
    Preferred over standard PIN for cross-sectional comparisons where
    stocks have heterogeneous trading activity.

    **Interpretation**:

    - ``adj_pin`` < 0.10: Low information asymmetry, suitable for
      uninformed market-making.
    - ``adj_pin`` 0.10-0.30: Moderate; some informed activity detected.
    - ``adj_pin`` > 0.30: High information asymmetry; adverse selection
      risk is elevated.

    Parameters:
        buy_trades: Number of buy-initiated trades per period.
        sell_trades: Number of sell-initiated trades per period.

    Returns:
        Dictionary with keys ``'adj_pin'``, ``'pin_unadjusted'``,
        ``'alpha'``, ``'delta'``, ``'mu'``, ``'eps_b'``, ``'eps_s'``,
        ``'theta'`` (probability of symmetric activity shock).

    Example:
        >>> import numpy as np
        >>> from wraquant.microstructure.toxicity import adjusted_pin
        >>> rng = np.random.default_rng(42)
        >>> buys = rng.poisson(50, size=60)
        >>> sells = rng.poisson(50, size=60)
        >>> result = adjusted_pin(buys, sells)
        >>> 0 <= result['adj_pin'] <= 1
        True

    References:
        Duarte, J. & Young, L. (2009). "Why is PIN Priced?" *Journal of
        Financial Economics*, 91(2), 119-138.

    See Also:
        pin_model: Standard (unadjusted) PIN estimation.
        vpin: Real-time alternative using volume buckets.
    """
    b = coerce_array(buy_trades, "buy_trades")
    s = coerce_array(sell_trades, "sell_trades")
    n = len(b)

    # Initial guesses
    mu0 = float(np.mean(np.abs(b - s)))
    eps_b0 = float(np.mean(b)) * 0.5
    eps_s0 = float(np.mean(s)) * 0.5

    def neg_log_lik(params: NDArray[np.floating]) -> float:
        alpha, delta, mu, eps_b, eps_s, theta = params
        alpha = np.clip(alpha, 1e-6, 1 - 1e-6)
        delta = np.clip(delta, 1e-6, 1 - 1e-6)
        theta = np.clip(theta, 1e-6, 1 - 1e-6)
        mu = max(mu, 1e-6)
        eps_b = max(eps_b, 1e-6)
        eps_s = max(eps_s, 1e-6)

        ll = 0.0
        for i in range(n):
            bi, si = b[i], s[i]
            # Four regimes:
            # 1. No event, no shock
            # 2. Bad-news event (sell informed)
            # 3. Good-news event (buy informed)
            # 4. Symmetric liquidity shock (both sides increase)
            log_parts = np.array([
                np.log((1 - alpha) * (1 - theta))
                + _poisson_ll(bi, eps_b) + _poisson_ll(si, eps_s),
                np.log(alpha * delta * (1 - theta))
                + _poisson_ll(bi, eps_b) + _poisson_ll(si, eps_s + mu),
                np.log(alpha * (1 - delta) * (1 - theta))
                + _poisson_ll(bi, eps_b + mu) + _poisson_ll(si, eps_s),
                np.log(theta)
                + _poisson_ll(bi, eps_b + mu) + _poisson_ll(si, eps_s + mu),
            ])
            max_lp = np.max(log_parts)
            ll += max_lp + np.log(np.sum(np.exp(log_parts - max_lp)))
        return -ll

    x0 = np.array([0.5, 0.5, mu0, eps_b0, eps_s0, 0.2])
    bounds = [
        (1e-4, 1 - 1e-4),
        (1e-4, 1 - 1e-4),
        (1e-4, None),
        (1e-4, None),
        (1e-4, None),
        (1e-4, 1 - 1e-4),
    ]
    result = minimize(neg_log_lik, x0, bounds=bounds, method="L-BFGS-B")
    alpha, delta, mu, eps_b, eps_s, theta = result.x

    # Adjusted PIN excludes symmetric shock regime
    adj_pin = (alpha * mu) / (alpha * mu + eps_b + eps_s + 2 * theta * mu)
    # Unadjusted for comparison
    pin_unadj = (alpha * mu) / (alpha * mu + eps_b + eps_s)

    return {
        "adj_pin": float(np.clip(adj_pin, 0, 1)),
        "pin_unadjusted": float(np.clip(pin_unadj, 0, 1)),
        "alpha": float(alpha),
        "delta": float(delta),
        "mu": float(mu),
        "eps_b": float(eps_b),
        "eps_s": float(eps_s),
        "theta": float(theta),
    }


def toxicity_index(
    vpin_values: pd.Series | NDArray[np.floating],
    ofi_values: pd.Series | NDArray[np.floating],
    spread_values: pd.Series | NDArray[np.floating],
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> NDArray[np.floating]:
    """Composite toxicity index combining VPIN, OFI, and spread dynamics.

    Produces a single 0-100 score summarizing the overall level of adverse
    selection / order flow toxicity.  Each input component is independently
    normalized to [0, 1] via min-max scaling, then combined with the
    specified weights and rescaled to 0-100.

    **When to use**: For a single-number dashboard indicator of market
    toxicity that synthesizes multiple underlying signals.  Useful for
    setting risk limits (e.g., "pause quoting when toxicity > 70") or for
    cross-sectional comparison across instruments.

    **Interpretation**:

    - **0-20**: Low toxicity; market is safe for passive market-making.
    - **20-50**: Moderate toxicity; monitor order flow closely.
    - **50-80**: Elevated toxicity; consider widening quotes or reducing
      size.
    - **80-100**: Extreme toxicity; likely informed trading event in
      progress.

    Parameters:
        vpin_values: VPIN time series (typically in [0, 1]).
        ofi_values: Absolute order flow imbalance (|OFI|), higher = more
            toxic.
        spread_values: Spread time series (wider = more toxic).
        weights: Relative weights for (VPIN, OFI, spread).  Must sum to
            1.0.

    Returns:
        Composite toxicity index in [0, 100].

    Example:
        >>> import numpy as np
        >>> from wraquant.microstructure.toxicity import toxicity_index
        >>> vpin_vals = np.array([0.2, 0.3, 0.5, 0.4])
        >>> ofi_vals = np.array([0.1, -0.3, 0.5, -0.2])
        >>> spread_vals = np.array([0.01, 0.02, 0.03, 0.015])
        >>> idx = toxicity_index(vpin_vals, ofi_vals, spread_vals)
        >>> (idx >= 0).all() and (idx <= 100).all()
        True

    References:
        Easley, D., Lopez de Prado, M. M. & O'Hara, M. (2011). "The
        Microstructure of the 'Flash Crash'." *Journal of Portfolio
        Management*, 37(2), 118-128.

    See Also:
        vpin: One of the component inputs.
        order_flow_imbalance: Another component input.
    """
    def _min_max(arr: NDArray[np.floating]) -> NDArray[np.floating]:
        lo = np.nanmin(arr)
        hi = np.nanmax(arr)
        if hi - lo < 1e-15:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    v = _min_max(coerce_array(vpin_values, "vpin_values"))
    o = _min_max(np.abs(coerce_array(ofi_values, "ofi_values")))
    s = _min_max(coerce_array(spread_values, "spread_values"))

    w_v, w_o, w_s = weights
    composite = w_v * v + w_o * o + w_s * s
    return np.clip(composite * 100.0, 0.0, 100.0)


def informed_trading_intensity(
    buy_volume: pd.Series,
    sell_volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Time-varying probability of informed trading using a sequential model.

    Estimates the instantaneous probability that the marginal trade is
    information-driven, based on the sequential trade framework of Glosten
    & Milgrom (1985) and Easley & O'Hara (1987).

    The approach uses a rolling Bayesian update.  In each window, the
    fraction of trades on the "aggressive" side (whichever of buy or sell
    dominates) is used to infer the conditional probability that an
    information event is occurring, under the assumption that informed
    traders cluster on one side while uninformed traders arrive symmetrically.

    **When to use**: For real-time monitoring of informed trading intensity.
    Unlike the static PIN model, this provides a *time-varying* signal that
    can be used to dynamically adjust quotes or execution strategies.

    **Interpretation**: Values near 0.5 indicate balanced (uninformed)
    flow.  Values approaching 1.0 indicate that nearly all marginal volume
    is on one side, consistent with an active informed trader.

    Parameters:
        buy_volume: Buy-initiated volume per observation.
        sell_volume: Sell-initiated volume per observation.
        window: Rolling window for the Bayesian update.

    Returns:
        Rolling informed trading intensity in [0, 1].

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> buys = pd.Series(np.random.uniform(100, 500, 50))
        >>> sells = pd.Series(np.random.uniform(100, 500, 50))
        >>> intensity = informed_trading_intensity(buys, sells, window=10)
        >>> intensity.name
        'informed_trading_intensity'

    References:
        Easley, D. & O'Hara, M. (1987). "Price, Trade Size, and
        Information in Securities Markets." *Journal of Financial
        Economics*, 19(1), 69-90.

    See Also:
        pin_model: Static probability of informed trading.
        order_flow_imbalance: Simpler directional flow measure.
    """
    buy_volume = coerce_series(buy_volume, "buy_volume")
    sell_volume = coerce_series(sell_volume, "sell_volume")
    total = buy_volume + sell_volume
    imbalance = np.abs(buy_volume - sell_volume)

    # Fraction of aggressive volume in each window
    rolling_imbalance = imbalance.rolling(window).sum()
    rolling_total = total.rolling(window).sum()

    # Informed intensity = excess imbalance beyond chance
    # Under pure noise, imbalance is ~sqrt(n) * sigma, so normalize
    intensity = rolling_imbalance / rolling_total
    intensity = intensity.replace([np.inf, -np.inf], np.nan)
    intensity = intensity.clip(0.0, 1.0)
    intensity.name = "informed_trading_intensity"
    return intensity
